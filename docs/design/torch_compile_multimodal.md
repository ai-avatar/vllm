# torch.compile with Multimodal Encoders

`torch.compile` can now be applied to multimodal encoders and miscellaneous nn modules in vLLM, including vision-language models like LLaMA 4, Qwen-VL,
and similar encoder-based architectures.

This document covers the basics of how the `torch.compile` integration works for multimodal encoders in vLLM, as well as how to apply the decorator
to new models to improve performance.

!!! note
    For general information about `torch.compile` integration in vLLM, see the [torch.compile design document](./torch_compile.md).

## Overview

We have recently enabled the `@support_torch_compile` decorator to work for multiple nn module components within a model type; this enables
turning compile on for multimodal encoders, bringing performance improvements to additional components of the stack.

When applied to the vision block of [`Qwen2_5_vl`](https://github.com/vllm-project/vllm/pull/23207) we observe ~4.5% e2e perf improvements with
some increase in compilation time

This feature is off by default. Enable it with any of:

```bash
# Official compilation_config (preferred)
vllm serve <model> --compilation-config '{"compile_mm_encoder": true}'

# Env alias (same default-off flag)
VLLM_COMPILE_MM_ENCODER=1 vllm serve <model>

# NeMo SpeechLM / Nemotron Transcribe also accept the plugin hf-overrides
vllm serve <nemotron-transcribe> --hf-overrides '{"encoder_compile": true}'
```

In-tree vision encoders compile through `@support_torch_compile`. The NeMo
SpeechLM `perception.encoder` transformer blocks (including Sortformer when
the plugin mounts it there) are compiled after load. Flex q/k/v and
`create_block_mask` pad onto a closed bucket set (256-frame whole-forward
tiles and `window` for last-window remainders) and NeMo's flex wrapper is
compiled with `dynamic=False` so variable chunk lengths do not
re-specialize. `FORCE_USE_FLEX_ATTENTION` keeps short clips off
flex-decoding. The projector / `modality_adapter` stay eager.

## How Compilation Works for Multimodal Components

### APIs for Enablement

To compile a multimodal component such as an encoder, we follow the same mechanism as the LLM text backbone, with a few additional scaffoldings:

1. The `@support_torch_compile` decorator should include `enable_if=should_torch_compile_mm_encoder`. This will gate the compilation behind our
`compile_mm_encoder` configuration

2. The `@support_torch_compile` decorator should include `is_encoder=True` for encoder components. This is needed for compile range integration
(see Compile Range Integration). The decorator automatically uses the class name as the cache directory prefix, avoiding collisions between
independently compiled sub-modules (e.g. vision encoder components vs the text backbone).

### CompilationConfig

With the exception of `compile_mm_encoder: true`, the multimodal encoder will inherit from the same compilation config as the text LLM. We may extend
this for more configuration in the future.

## Applying torch.compile to a New Multimodal Model/Component

To apply `support_torch_compile` to a new general nn.Module, we advise following the same steps in [`debug_vllm_compile`](./debug_vllm_compile.md); this includes:

1. Applying `support_torch_compile` on initially small modules (such as basic MLP layers), then raising to more general modules until one reaches a good performance
tradeoff

2. Leveraging [`tlparse`](https://github.com/meta-pytorch/tlparse) to identify and eliminate the source of recompiles and graph breaks

3. Using `dynamic_arg_dims` and proper `dynamic_shapes_config` to handle dynamism.

### Common pitfalls

## VllmBackend Feature Support

### Compile ranges

The torch.compile integration will try to rely on max_batch_size to infer compilation ranges for dynamic shapes; however, for modules used in the encoder, this
shape can be difficult to infer due to the unspecified range of shapes the encoder may see as input. Therefore, we rely on `is_encoder=True` in the
`@support_torch_compile` decorator to alert torch.compile to the fact that this range cannot be inferred, and we default to the range (1, MAX_INT).

!!! note
    We may seek to tighten this range for better performance in the future

### Cudagraphs

Vision models that implement `SupportsEncoderCudaGraph` can capture
`cudagraph_mm_encoder` graphs through `EncoderCudaGraphManager`
(token-budget packing). That path is unchanged.

Nemotron Transcribe / NeMo SpeechLM does **not** implement that protocol.
These three flags are equivalent and all enable the plugin's
`EncoderGraphRunner` (length-bucket graphs, pad + mask):

```bash
--compilation-config '{"cudagraph_mm_encoder": true}'
VLLM_CUDAGRAPH_MM_ENCODER=1   # also accepts true/false
--hf-overrides '{"encoder_cuda_graphs": true}'
```

The effective graph flag is written back to both `compilation_config` and
the SpeechLM hf config. vLLM compiles `perception.encoder` blocks itself
(the plugin `encoder_compile` flag is an input only) after forcing
`FORCE_USE_FLEX_ATTENTION` on NeMo's `_flex_attention_compiled` /
`_get_flex_attention` so inductor cannot pick flex-decoding for inputs
shorter than 128 encoder frames (10.24 s). The leftover
`_use_flex_decoding` `Q < 128` guard is short-circuited so crossing
128 frames does not recompile. Layer compile and graph capture are
deferred until after KV allocation; a post-KV graph warmup then
captures every `(batch, frame-bucket)` (under BI: batch 1 × buckets
up to the audio cap) using `n_feat` from the perception preprocessor
/ encoder (`features` / `feat_in`, 128 mels here), then one
long-audio `3 * window` dummy. The plugin still windows by
`processed_len`, so last-window T is `valid % window`; flex q/k/v
and the block mask pad to `window` (or the next 256-bucket).
`mask_mod` reads the valid length from a tensor (not a Python int);
that tensor is written before each call and before CUDA-graph replay
(never baked during capture). Cached BlockMasks are a full bucket so
a short first request cannot empty tail blocks for a later longer
one. Any encoder call outside `inference_mode` (the timestamp
aligner) is promoted to inference tensors and uses uncompiled
FORCE_USE flex. Flex is compiled with `dynamic=False` so those
remainders cannot recompile. Compile-only skips dummy-audio warmup. Warmup time is
logged and is not counted as peak activation. Pending-compile state is stored on the unwrapped
module (`model.unwrap()` / `.runnable`) because
`CUDAGraphWrapper.__getattr__` forwards but `delattr` does not.
After warmup, serving-time capture is off. Under
`VLLM_BATCH_INVARIANT=1` the default batch grid is `(1,)`, the
replay path is a function of each item's valid length, inductor
timing autotune is disabled (`deterministic=True`) for
`layer.compile` and NeMo's import-time compiled flex wrapper, and
served tensors stay on the warmed shape set so later traffic cannot
change encoder numerics. `TORCH_LOGS=recompiles,guards` plus the
post-warmup recompile warning print the last path, processed
shape/len, and flex q shape/stride/block-mask.

```bash
VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve <nemotron-3.5-transcribe-bf16> \
  --dtype bfloat16 \
  --compilation-config '{"compile_mm_encoder": true, "cudagraph_mm_encoder": true}' \
  --mamba-backend triton \
  --no-enable-prefix-caching
```

## Troubleshooting

### Graph Breaks in Vision Encoders

Some vision encoder operations may cause graph breaks. To identify them:

```bash
TORCH_LOGS="+dynamo" vllm serve <MODEL>
```

Common causes of graph breaks in multimodal models:

- **Dynamic image sizes**: Use `dynamic_shapes_config` to handle variable resolutions
- **Untraceable operations**: Some operations (such as to_list) may not be supported by Dynamo
- **Conditional processing**: Data-dependent branching based on image properties

### Compilation Errors

If compilation fails for a multimodal model:

1. **Disable and test**: First verify the model works without compilation:
   ```bash
   vllm serve <model> --compilation-config='{"mode":0,"compile_mm_encoder":"false"}'
   ```

2. **Check logs**: Enable debug logging to see compilation details:
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> --compilation-config='{"compile_mm_encoder":"true"}'
   ```

3. **Report issues**: If you find a bug, [open an issue on GitHub](https://github.com/vllm-project/vllm/issues/new/choose)

## See Also

- [torch.compile Integration](./torch_compile.md) - Core design document
- [Debugging torch.compile](./debug_vllm_compile.md) - Detailed debugging guide
- [Multimodal Inputs](../features/multimodal_inputs.md) - How to pass multimodal data
- [Disaggregated Encoder](../features/disagg_encoder.md) - Scaling vision encoders
- [Supported Multimodal Models](../models/supported_models.md#list-of-multimodal-language-models) - Model compatibility

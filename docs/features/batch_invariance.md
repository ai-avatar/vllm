# Batch Invariance

!!! note
    Batch invariance is currently in beta. Some features are still under active development.
    Track progress and planned improvements at <https://github.com/vllm-project/vllm/issues/27433>

This document shows how to enable batch invariance in vLLM. Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch.

## Motivation

Batch invariance is crucial for several use cases:

- **Framework debugging**: Deterministic outputs make it easier to debug issues in the inference framework, as the same input will always produce the same output regardless of batching.
- **Model debugging**: Helps identify issues in model implementations by ensuring consistent behavior across different batch configurations.
- **Reinforcement Learning (RL)**: RL training often requires deterministic rollouts for reproducibility and stable training.
- **Large-scale inference systems**: Systems that use vLLM as a component benefit from deterministic behavior for testing, validation, and consistency guarantees.

## Hardware Requirements

Batch invariance is supported on the following platforms:

- NVIDIA GPUs with compute capability 8.0 or higher.
- AMD Instinct GPUs (gfx90a, gfx942, gfx950).
- Intel XPUs with Triton support.

### Attention Backend Selection for XPU

On XPU, Triton Attention backend is required for batch invariance.
Select this backend using Qwen/Qwen3-1.7B model as an example:

```python
llm = LLM(
    model="Qwen/Qwen3-1.7B",
    attention_config={"backend": "TRITON_ATTN"},
)
```

Or via the CLI:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-1.7B \
    --attention-config.backend TRITON_ATTN
```

## Enabling Batch Invariance

Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

```bash
export VLLM_BATCH_INVARIANT=1
```

### Online Inference (Server Mode)

To start a vLLM server with batch invariance enabled:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Then use the OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# These requests will produce deterministic outputs
# regardless of batch size or order
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

### Offline Inference

For offline batch inference with batch invariance:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## Tested Models

Batch invariance has been tested and verified on the following models:

- **DeepSeek series**: `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`
- **Qwen3 (Dense)**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen3-8B-AWQ`
- **Qwen3-VL (Vision-Language)**: `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct` (single image and video inputs)
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Qwen2.5**: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Llama 3**: Llama3.1 and 3.2 series, `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` for example
- **GPT-OSS**: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **Mistral**: `mistralai/Mistral-7B-v0.3`
- **GLM-4 series**: `zai-org/glm-4-9b-chat`
- **Phi series**: `microsoft/Phi-3.5-mini-instruct`, `microsoft/phi-4`
- **Granite 3.1 (MoE)**: `ibm-granite/granite-3.1-1b-a400m-instruct`, `ibm-granite/granite-3.1-3b-a800m-instruct`
- **Granite 3.1 (Dense)**: `ibm-granite/granite-3.1-2b-instruct`, `ibm-granite/granite-3.1-8b-instruct`
- **EXAONE 4.0 series**: `LGAI-EXAONE/EXAONE-4.0-1.2B`, `LGAI-EXAONE/EXAONE-4.0.1-32B`, `LGAI-EXAONE/EXAONE-4.0-32B`
- **OLMo 2**: `allenai/OLMo-2-0425-1B-Instruct`
- **ERNIE 4.5**: `baidu/ERNIE-4.5-0.3B-PT`
- **SmolLM2**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **PLaMo3**: `pfnet/plamo-3-nict-2b-base`
- **NemotronH / Mamba2 hybrids** (this fork): `MAMBA2_ATTN` is now a supported
  batch-invariant backend. Target for this work is NVIDIA Nemotron 3.5 Transcribe
  (NemotronH-style Mamba2 + attention + MoE, BF16) on A100/H100 80 GB. FP8 is
  out of scope.

Other models may also work, but these have been explicitly validated. If you encounter issues with a specific model, please report them on the [GitHub issue tracker](https://github.com/vllm-project/vllm/issues/new/choose).

## NemotronH / Mamba2 (this fork)

`VLLM_BATCH_INVARIANT=1` used to refuse startup with
`not supported for MAMBA2_ATTN`. This fork enables the mode for Mamba2 and pins
the SSD / SSU tiles so a request's greedy tokens do not depend on who else is
in the decode or prefill batch.

### What stays enabled

- **MTP speculative decoding.** The drafter and target share the pinned
  Mamba2 / MoE / attention / RMSNorm paths. Alone-vs-batched invariance is
  about each request's target-greedy tokens, not MTP-on vs MTP-off equality.
  Mixed prefill+decode steps stay allowed; they are not a source of
  short-clip flips.
- **Chunked prefill.** Partial prefill chunks are forced to end on a Mamba
  SSD `chunk_size` boundary (128 for NemotronH). An unaligned split computes
  intra-chunk terms differently from an unsplit run; aligned splits are
  bit-identical. The last chunk of a prompt is not clipped.
- **Prefix caching.** Not disabled. Cached Mamba/KV state is only valid if it
  was written under the same batch-invariant process. Production Nemotron
  Transcribe already uses `--no-enable-prefix-caching`; keep that until you
  have validated APC on the real checkpoint.
- **Audio / multimodal encoder.** Each encoder item is encoded alone in both
  the V1 `GPUModelRunner` and the V2 `EncoderRunner`, so the NeMo plugin
  cannot pad several clips into one GEMM. Production nemotron-api uses
  `VLLM_USE_V2_MODEL_RUNNER=0`. Optional
  `compile_mm_encoder` / `VLLM_COMPILE_MM_ENCODER=1` compiles
  `perception.encoder` blocks with dynamic shapes. Flex stays in that
  graph with `FORCE_USE_FLEX_ATTENTION` so inductor cannot pick
  flex-decoding for clips shorter than 128 encoder frames. Encoder CUDA
  graphs (`cudagraph_mm_encoder`, `VLLM_CUDAGRAPH_MM_ENCODER`, or
  `encoder_cuda_graphs`) are equivalent; they stay off unless any of
  those three is set. Under BI the default graph batch grid is `(1,)`,
  compile and capture are deferred until after KV allocation, a post-KV
  graph warmup captures batch-1 frame buckets and one long-audio
  `3 * window` dummy (n_feat from the preprocessor / encoder). Flex
  q/k/v and `create_block_mask` pad last-window remainders
  (`valid % window` from the plugin) onto that closed set.
  `mask_mod` reads valid length from a tensor (written before each
  call / CUDA-graph replay). `create_block_mask` keeps the caller's
  seq_lengths; the flex wrapper is `torch.compiler.disable`'d
  (recursive) so dynamo cannot trace pad / `set_stance` into a
  compiled layer, and q/k/v are made contiguous before the compiled
  flex call. Encoder forwards pin the compile-time `default_dtype`
  so a later process default cannot recompile on `GLOBAL_STATE`.
  Warmup also runs the plugin profile dummy
  (`3 * max_frames` when that cap is 4000).
  The timestamp aligner is `torch.compiler.disable`'d (no new
  unique_graphs) and keeps its own BlockMask / valid-length
  tensors so it cannot mutate the next compiled encode. A
  timestamp request replays the same compiled graphs as plain
  (including T=1512 / 121 s) and still returns CTC alignment
  states from those graphs. Compile-off does not wrap
  `perception.forward` or rewrite that arity. Flex
  compiles with `dynamic=False`. The path is a function of each item's
  valid length. Compiled encoder / flex kernels pin inductor
  `deterministic` (and disable timing `max_autotune` /
  `coordinate_descent_tuning`) so embeddings stay bitwise identical
  across process starts and after later traffic.

### What you must disable (explicit errors, not silent)

These abort startup so they cannot be left on by accident:

- `--mamba-backend flashinfer` — use the default `--mamba-backend triton`
- `--use-replayssm` — MTP still runs through standard Triton SSU
- `--enable-mamba-cache-stochastic-rounding`

### Enable

```bash
export VLLM_BATCH_INVARIANT=1
# equivalent in this tree: the env is read by vllm.envs before engine init

vllm serve <nemotron-3.5-transcribe-bf16> \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --mamba-backend triton \
  --no-enable-prefix-caching
```

Keep your usual MTP / chunked-prefill flags. Do not set
`--mamba-backend flashinfer` or `--use-replayssm`.

### Install this fork (torch 2.13 + cu130)

Build from this branch against PyTorch 2.13 and the CUDA 13.0 wheel index.
Never use system `python3` or bare `pip`.

```bash
# 1. Tooling
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed
source .venv/bin/activate

# 2. PyTorch 2.13 + cu130 first, then vLLM
uv pip install torch==2.13.0 --torch-backend=cu130

# Editable install (reuses precompiled CUDA extensions when the local
# torch/CUDA pair matches a published wheel):
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cu130

# Or a wheel you can copy onto the A100/H100 box:
# VLLM_USE_PRECOMPILED=1 uv build --wheel
# uv pip install dist/vllm-*.whl --torch-backend=cu130

uv pip install -r requirements/lint.txt
pre-commit install
```

If `VLLM_USE_PRECOMPILED=1` cannot find a matching wheel, drop it and compile
from source (needs `nvcc` from CUDA 13.0 and `ninja`):

```bash
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
uv pip install -e . --torch-backend=cu130
```

After install:

```bash
.venv/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
VLLM_BATCH_INVARIANT=1 .venv/bin/python -c \
  "from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionBackend as B; \
   assert B.supports_batch_invariance()"
```

Kernel tests (GPU when present; CPU covers the config / backend checks):

```bash
.venv/bin/python -m pytest \
  tests/kernels/mamba/test_mamba2_batch_invariant.py \
  tests/v1/determinism/test_mamba2_moe_batch_invariant.py \
  tests/v1/determinism/test_mamba2_bi_scheduler.py \
  tests/v1/determinism/test_mm_encoder_batch_invariant.py \
  tests/v1/determinism/test_gate_linear_batch_invariant.py -v
```

### Speed cost (A100 BF16 Nemotron 3.5 Transcribe, measured)

Prod flags, MTP 8, chunked prefill 4096, `--no-enable-prefix-caching`.
`tok/s` is completion tokens / wall.

| | c1 | c8 | c16 | c32 | 8×EN | KV cache |
| --- | --- | --- | --- | --- | --- | --- |
| baseline BF16 | 144 tok/s | 287 | 390 | 493 | 456 | 420k |
| BI + alignment + V1 per-item encode | 90 | 197 | 268 | 319 | 298 | 437k |
| cost | −37% | −31% | −31% | −35% | −35% | ≈ |

A100 BF16 Nemotron 3.5 Transcribe, prod flags (MTP on, mixed prefill+decode
steps allowed): **0/152** concurrent-vs-alone diffs, mix3 20/20 identical.
Cost is about **31–37%** vs non-BI baseline. A decode-only/prefill-only
step split is **not** enabled: it is not needed for invariance and cut
throughput roughly in half (129/140/151 tok/s at c8/16/32). H100 was not
re-measured; expect the same class of cost plus the explicit BI matmul on
the router gate. Memory is about unchanged.

FP8 weights, Marlin, and sm89 FP8 GEMM are intentionally not covered.

## Implementation Details

When batch invariance is enabled, vLLM:

1. Uses deterministic kernel implementations for attention and other operations
2. Ensures consistent numerical behavior across different batch sizes
3. Disables certain optimizations that may introduce non-determinism
4. Under tensor parallelism, keeps custom all-reduce on with a fixed reduction order (the 1-stage kernel is pinned, and large inputs are reduced in fixed-size chunks), and disables FlashInfer, AITER and QuickReduce all-reduce
5. On CUDA devices with tuned matmul table entries for the model's bf16 unquantized forward linear layers (Ada, Hopper, Blackwell),
   runs without `torch.compile` using breakable CUDA graphs so tile configs follow the runtime batch size; set `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to opt out.
6. Splits the LoRA shrink reduction into 8 fixed K chunks that are summed in a fixed order instead of with atomics
7. For Mamba2, pins SSD Triton tiles and the decode SSU launch config so they
   no longer depend on token count or `effective_batch`. FlashInfer SSU,
   ReplaySSM, and Mamba stochastic rounding are rejected rather than ignored.
   Prefill splits are aligned to the Mamba chunk size. Multimodal encoder
   items run one at a time on both the V1 and V2 runners. The MoE router
   gate (`GateLinear` `torch.mm(..., out_dtype=fp32)` on H100) is routed
   through the batch-invariant matmul. Compiled speech-encoder blocks
   and NeMo's flex wrapper pin inductor `deterministic` so tile
   selection is not timing-based across process starts. Served encoder
   frames are bucketed onto the warmed window / frame-bucket set so a
   later request cannot recompile a different kernel.

!!! warning
    Batch invariance under tensor parallelism is not yet supported for all-reduces whose size isn't a multiple of 16 bytes, for example a hidden size that isn't a multiple of 8 in fp16/bf16. Such tensors can switch between NCCL and custom all-reduce depending on batch size. All validated models meet this requirement.

!!! note
    Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.

## Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional GPU architectures
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the [tracking issue](https://github.com/vllm-project/vllm/issues/27433).

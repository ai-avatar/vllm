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
  flex-decoding for clips shorter than 128 encoder frames. NeMo's
  import-time compiled flex is also guarded with compile *off*: dynamo
  limits are raised process-wide (and on the encoder / diarizer thread),
  `kernel_options` / `inference_mode` are normalized so the cache is
  not request-order dependent. Serving never calls `set_default_dtype`
  (process default stays fp32). NeMo `_default_dtype` is a thread-local
  pin so Sortformer `torch.zeros(..., device=)` still gets bf16. A
  request-boundary check WARNs if the process default is not fp32.
  Startup fails if flex compiles `_flex_attention_hop_wrapper` with
  `backend=eager`.
  Encoder CUDA
  graphs (`cudagraph_mm_encoder`, `VLLM_CUDAGRAPH_MM_ENCODER`, or
  `encoder_cuda_graphs`) are equivalent; they stay off unless any of
  those three is set. Under BI the default graph batch grid is `(1,)`,
  compile and capture are deferred until after KV allocation, a post-KV
  graph warmup captures batch-1 frame buckets and long-audio window
  graphs (n_feat from the preprocessor / encoder), and the path is a
  function of each item's valid length.

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
  tests/v1/determinism/test_gate_linear_batch_invariant.py \
  tests/v1/determinism/test_ampere_matmul_static_config.py \
  tests/compile/test_inductor_autotune_pin.py \
  tests/compile/test_config.py -k dynamic_scale_rblock -v
```

### Inductor autotune pins (this fork)

Inductor fused add+RMSNorm reductions are single-config (R0_BLOCK 1024).
On every load, `_dynamic_scale_rblock()` used to append a halved 512
config and time the two by wall clock; a 512 win changes RMSNorm order
and is the decoder ALT state (~1 in 24). vLLM now defaults
`inductor_compile_config["dynamic_scale_rblock"] = False`, so those
kernels stay on 1024 with no runtime benchmark.

Multi-config kernels (for example rms_norm 4096 vs 2048) are still timed
once at cold compile. Those winners live as `*.best_config` files under
`TORCHINDUCTOR_CACHE_DIR`. Pin them per GPU SKU and load them before
precompile so every autotuner is single-config or a cache hit. After
warmup the worker logs `inductor runtime benchmarks: N (...)` and, under
`VLLM_BATCH_INVARIANT=1`, **fails startup** if N is not 0.

Do not set `inductor_compile_config.deterministic=true` /
`TORCHINDUCTOR_DETERMINISTIC`: that picks the second-largest RBLOCK and
is a different (broken) numeric path.

Pinned packs are GPU- and torch-version-specific. A100 (sm80) must not
be reused on L40S (sm89) or H100. Regenerate whenever torch, triton,
vLLM, the model graph, or inductor config changes.

#### Generate and load on the pod (A100 vs L40S)

```bash
# SKU key is family + compute capability, e.g. a100-sm80 or l40s-sm89
.venv/bin/python -m vllm.compilation.inductor_autotune_pin sku

# 1) Reference cold compile on this GPU, dump the AutotuneCache winners.
#    ALLOW_BENCH is required for this first run (the dump is the point).
rm -rf ~/.cache/vllm/torch_compile_cache
export VLLM_BATCH_INVARIANT=1
export VLLM_INDUCTOR_AUTOTUNE_ALLOW_BENCH=1
export VLLM_INDUCTOR_AUTOTUNE_PIN_DUMP=1
vllm serve <nemotron-3.5-transcribe-bf16> ...  # same flags as production
# pack lands in $VLLM_CACHE_ROOT/inductor_autotune_pins/$(python -m ... sku)/

# Or copy from an existing inductor cache after a REF process:
.venv/bin/python -m vllm.compilation.inductor_autotune_pin dump \
  --cache-dir "$TORCHINDUCTOR_CACHE_DIR" \
  --output ~/.cache/vllm/inductor_autotune_pins/a100-sm80

# 2) Subsequent starts: load the pack before precompile, fail if any
#    kernel still benchmarks.
unset VLLM_INDUCTOR_AUTOTUNE_ALLOW_BENCH VLLM_INDUCTOR_AUTOTUNE_PIN_DUMP
export VLLM_INDUCTOR_AUTOTUNE_PIN_DIR=~/.cache/vllm/inductor_autotune_pins/a100-sm80
VLLM_BATCH_INVARIANT=1 vllm serve <nemotron-3.5-transcribe-bf16> ...
# log must contain: inductor runtime benchmarks: 0
```

On L40S use `l40s-sm89` (the `sku` command prints it). Keep one pack per
SKU; do not mix. If the compile cache is wiped, keep the pin pack — load
still seeds `TORCHINDUCTOR_CACHE_DIR` before the new compile's
precompile.

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
6. On Ampere (sm80 / A100), Nemotron 3.5 Transcribe BF16 persistent-matmul
   tiles are keyed only on `(N, K)` and never on `M`, so a request that runs
   alone and the same row in a concurrent batch share one kernel config. The
   table in `_BATCH_INVARIANT_MATMUL_STATIC_CONFIGS["ampere"]` is the A100
   `--quick` sweep
   (`benchmarks/kernels/tune_ampere_batch_invariant_matmul.py`; 252/252
   candidates BI-ok). Unmatched shapes keep the generic 128×128×64 default.
   This does not enable breakable CUDA graphs (those exist for M-bucketed
   Ada/Hopper/Blackwell tables).
7. Pins inductor kernel selection so fused add+RMSNorm reductions do not
   re-time R0_BLOCK 1024 vs 512 on every process, and so multi-config
   kernels reuse a SKU-keyed AutotuneCache pack instead of re-benchmarking
   on every cold compile. See
   [Inductor autotune pins](#inductor-autotune-pins-this-fork).
8. Splits the LoRA shrink reduction into 8 fixed K chunks that are summed in a fixed order instead of with atomics
9. For Mamba2, pins SSD Triton tiles and the decode SSU launch config so they
   no longer depend on token count or `effective_batch`. FlashInfer SSU,
   ReplaySSM, and Mamba stochastic rounding are rejected rather than ignored.
   Prefill splits are aligned to the Mamba chunk size. Multimodal encoder
   items run one at a time on both the V1 and V2 runners. The MoE router
   gate (`GateLinear` `torch.mm(..., out_dtype=fp32)` on H100) is routed
   through the batch-invariant matmul.

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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant launch helpers for Mamba2 Triton kernels.

Triton's autotuner keys off tensor shapes, so a mixed-length prefill or a
different decode batch size can pick a different BLOCK tile and change the
reduction order. Under ``VLLM_BATCH_INVARIANT`` we launch the underlying
``@triton.jit`` kernel with a single conservative tile so every row sees the
same accumulation order.
"""

from typing import Any

import vllm.envs as envs
from vllm.config.mamba import MambaBackendEnum

_FLASHINFER_BI_MSG = (
    "VLLM_BATCH_INVARIANT requires --mamba-backend triton. "
    "FlashInfer SSU is not batch-invariant."
)
_REPLAYSSM_BI_MSG = (
    "VLLM_BATCH_INVARIANT does not support --use-replayssm. "
    "Disable ReplaySSM; MTP speculative decoding still works through "
    "the standard Triton selective-state-update path."
)
_STOCHASTIC_BI_MSG = (
    "VLLM_BATCH_INVARIANT is incompatible with "
    "--enable-mamba-cache-stochastic-rounding."
)


def check_mamba_batch_invariant_compat(
    *,
    backend: MambaBackendEnum,
    use_replayssm: bool = False,
    enable_stochastic_rounding: bool = False,
) -> None:
    """Raise if a Mamba setting cannot be batch-invariant.

    MTP, chunked prefill, and prefix caching stay enabled. ReplaySSM,
    FlashInfer SSU, and stochastic rounding must be turned off explicitly.
    """
    if not envs.VLLM_BATCH_INVARIANT:
        return
    if backend == MambaBackendEnum.FLASHINFER:
        raise RuntimeError(_FLASHINFER_BI_MSG)
    if use_replayssm:
        raise RuntimeError(_REPLAYSSM_BI_MSG)
    if enable_stochastic_rounding:
        raise RuntimeError(_STOCHASTIC_BI_MSG)


# Conservative GEMM tile used by chunk scan / chunk state / C·B bmm.
# num_stages=1 avoids the sm90 wgmma register-reuse race in ssd_chunk_scan.
SSD_GEMM_BI_CONFIG: dict[str, Any] = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "num_warps": 4,
    "num_stages": 1,
}

SSD_CUMSUM_BI_CONFIG: dict[str, Any] = {
    "BLOCK_SIZE_H": 8,
}

SSD_STATE_PASSING_BI_CONFIG: dict[str, Any] = {
    "BLOCK_SIZE": 128,
}


def _unwrap_triton_kernel(kernel: Any) -> Any:
    """Unwrap Autotuner / Heuristics wrappers, stopping at JITFunction.

    ``JITFunction.fn`` is the plain Python source. Unwrapping through it
    makes ``raw[grid]`` raise ``TypeError: 'function' object is not
    subscriptable`` on every SSD prefill launch.
    """
    try:
        from triton.runtime.jit import JITFunction
    except ImportError:  # pragma: no cover - triton always present on GPU
        JITFunction = ()  # type: ignore[misc, assignment]

    current = kernel
    seen: set[int] = set()
    while (
        not isinstance(current, JITFunction)
        and hasattr(current, "fn")
        and id(current.fn) not in seen
    ):
        seen.add(id(current))
        current = current.fn
    return current


def launch_triton(
    kernel: Any,
    grid: Any,
    *,
    bi_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Launch ``kernel[grid]``, pinning ``bi_config`` in batch-invariant mode.

    ``kernel`` may be a ``@triton.autotune`` wrapper. When
    ``VLLM_BATCH_INVARIANT`` is set, the underlying ``@triton.jit`` function
    is launched with ``bi_config`` so the tile does not depend on the current
    batch shape.
    """
    if bi_config is not None and envs.VLLM_BATCH_INVARIANT:
        raw = _unwrap_triton_kernel(kernel)
        return raw[grid](**kwargs, **bi_config)
    return kernel[grid](**kwargs)

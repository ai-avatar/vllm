# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance tests for Mamba2 ops used by NemotronH.

CPU tests cover backend enablement and config pinning. GPU tests compare
the first sequence/row when it runs alone (batch size 1) against the same
row packed with mixed-length neighbors (batch size N).
"""

from typing import Any

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.mamba.ops.batch_invariant import (
    SSD_CUMSUM_BI_CONFIG,
    SSD_GEMM_BI_CONFIG,
    SSD_STATE_PASSING_BI_CONFIG,
    check_mamba_batch_invariant_compat,
    launch_triton,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _get_default_ssm_launch_config,
    try_get_optimal_ssm_config,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.selector import get_mamba_attn_backend

DEVICE = current_platform.device_type

requires_triton_gpu = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Mamba2 Triton kernels require a CUDA-alike or XPU device.",
)


@pytest.fixture
def enable_batch_invariant(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def test_mamba2_backend_supports_batch_invariance(enable_batch_invariant):
    assert Mamba2AttentionBackend.supports_batch_invariance()
    assert Mamba2AttentionBackend.get_name() == "MAMBA2_ATTN"
    backend = get_mamba_attn_backend(MambaAttentionBackendEnum.MAMBA2)
    assert backend is Mamba2AttentionBackend


def test_ssu_config_ignores_batch_under_bi(enable_batch_invariant):
    pinned = try_get_optimal_ssm_config(
        headdim=64,
        dstate=128,
        batch=1,
        nheads=8,
        cache_dtype="bfloat16",
        is_blackwell=False,
    )
    other = try_get_optimal_ssm_config(
        headdim=64,
        dstate=128,
        batch=192,
        nheads=8,
        cache_dtype="bfloat16",
        is_blackwell=False,
    )
    assert pinned == other
    assert pinned == _get_default_ssm_launch_config(128, is_blackwell=False)


def test_incompatible_mamba_settings_are_rejected(enable_batch_invariant):
    from vllm.config.mamba import MambaBackendEnum

    check_mamba_batch_invariant_compat(
        backend=MambaBackendEnum.TRITON,
        use_replayssm=False,
        enable_stochastic_rounding=False,
    )
    with pytest.raises(RuntimeError, match="mamba-backend triton"):
        check_mamba_batch_invariant_compat(backend=MambaBackendEnum.FLASHINFER)
    with pytest.raises(RuntimeError, match="use-replayssm"):
        check_mamba_batch_invariant_compat(
            backend=MambaBackendEnum.TRITON, use_replayssm=True
        )
    with pytest.raises(RuntimeError, match="stochastic-rounding"):
        check_mamba_batch_invariant_compat(
            backend=MambaBackendEnum.TRITON, enable_stochastic_rounding=True
        )


def test_unwrap_stops_at_triton_jitfunction(enable_batch_invariant):
    """Autotune wrappers are stripped; JITFunction.fn must not be."""
    triton = pytest.importorskip("triton")
    from triton.runtime.jit import JITFunction

    from vllm.model_executor.layers.mamba.ops.batch_invariant import (
        _unwrap_triton_kernel,
    )

    @triton.jit
    def _dummy_kernel():
        pass

    assert isinstance(_dummy_kernel, JITFunction)
    assert _unwrap_triton_kernel(_dummy_kernel) is _dummy_kernel
    assert _dummy_kernel.fn is not _dummy_kernel

    class _Autotuned:
        fn = _dummy_kernel

    assert _unwrap_triton_kernel(_Autotuned()) is _dummy_kernel


def test_launch_triton_pins_config_under_bi(enable_batch_invariant):
    captured: dict[str, Any] = {}

    class _Raw:
        def __getitem__(self, grid):
            def _launch(**kwargs):
                captured["kwargs"] = kwargs
                captured["grid"] = grid

            return _launch

    class _Autotuned:
        fn = _Raw()

        def __getitem__(self, grid):
            def _launch(**kwargs):
                raise AssertionError("autotuned path must not run under BI")

            return _launch

    launch_triton(
        _Autotuned(),
        (1,),
        bi_config=SSD_GEMM_BI_CONFIG,
        x=1,
    )
    assert captured["grid"] == (1,)
    assert captured["kwargs"]["x"] == 1
    assert captured["kwargs"]["BLOCK_SIZE_M"] == SSD_GEMM_BI_CONFIG["BLOCK_SIZE_M"]
    assert captured["kwargs"]["num_warps"] == SSD_GEMM_BI_CONFIG["num_warps"]


def test_pinned_ssd_configs_are_conservative():
    assert SSD_GEMM_BI_CONFIG["num_stages"] == 1
    assert SSD_CUMSUM_BI_CONFIG["BLOCK_SIZE_H"] == 8
    assert SSD_STATE_PASSING_BI_CONFIG["BLOCK_SIZE"] == 128


def _ssd_inputs(seqlen, nheads, headdim, dstate, ngroups, dtype, device):
    x = torch.randn(seqlen, nheads, headdim, device=device, dtype=dtype)
    dt = torch.nn.functional.softplus(
        torch.randn(seqlen, nheads, device=device, dtype=dtype) - 4
    )
    a = -torch.exp(torch.rand(nheads, device=device, dtype=dtype))
    b = torch.randn(seqlen, ngroups, dstate, device=device, dtype=dtype)
    c = torch.randn(seqlen, ngroups, dstate, device=device, dtype=dtype)
    return x, dt, a, b, c


def _run_ssd(x, dt, a, b, c, chunk_size, cu_seqlens):
    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )
    from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

    cu_chunk, last_idx, seq_idx = compute_varlen_chunk_metadata(cu_seqlens, chunk_size)
    out = torch.empty_like(x)
    state = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        a,
        b,
        c,
        chunk_size,
        cu_seqlens=cu_seqlens.to(torch.int32),
        cu_chunk_seqlens=cu_chunk,
        last_chunk_indices=last_idx,
        seq_idx=seq_idx,
        out=out,
        D=None,
    )
    return out, state


@requires_triton_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ssd_varlen_batch1_matches_mixed_lengths(enable_batch_invariant, dtype):
    """First sequence's SSD output is bit-identical alone vs packed."""
    device = torch.device(DEVICE)
    torch.manual_seed(0)
    nheads, headdim, dstate, ngroups = 4, 32, 16, 2
    chunk_size = 16
    lengths = (13, 7, 24)
    total = sum(lengths)

    x, dt, a, b, c = _ssd_inputs(total, nheads, headdim, dstate, ngroups, dtype, device)
    cu_all = torch.tensor((0,) + lengths, device=device, dtype=torch.int32).cumsum(0)
    first = int(lengths[0])

    out_alone, state_alone = _run_ssd(
        x[:first].clone(),
        dt[:first].clone(),
        a,
        b[:first].clone(),
        c[:first].clone(),
        chunk_size,
        torch.tensor([0, first], device=device, dtype=torch.int32),
    )
    out_packed, state_packed = _run_ssd(x, dt, a, b, c, chunk_size, cu_all)

    assert torch.equal(out_packed[:first], out_alone)
    assert torch.equal(state_packed[0], state_alone[0])


@requires_triton_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ssu_decode_batch1_matches_batch_n(enable_batch_invariant, dtype):
    """Decode SSU row 0 is bit-identical for batch 1 vs N."""
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

    device = torch.device(DEVICE)
    torch.manual_seed(1)
    nheads, headdim, dstate, ngroups = 4, 32, 16, 2
    batch_n = 5

    def _make(batch: int):
        state = torch.randn(batch, nheads, headdim, dstate, device=device, dtype=dtype)
        x = torch.randn(batch, nheads, headdim, device=device, dtype=dtype)
        dt = torch.randn(batch, nheads, headdim, device=device, dtype=dtype)
        dt_bias = torch.rand(nheads, headdim, device=device) - 4.0
        a = -torch.rand(nheads, headdim, dstate, device=device) - 1.0
        b = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
        c = torch.randn(batch, ngroups, dstate, device=device, dtype=dtype)
        d = torch.randn(nheads, headdim, device=device)
        return state, x, dt, dt_bias, a, b, c, d

    state_n, x_n, dt_n, dt_bias, a, b_n, c_n, d = _make(batch_n)
    state_1 = state_n[:1].clone()
    x_1 = x_n[:1].clone()
    dt_1 = dt_n[:1].clone()
    b_1 = b_n[:1].clone()
    c_1 = c_n[:1].clone()

    out_1 = torch.empty_like(x_1)
    out_n = torch.empty_like(x_n)
    ssu_kwargs = dict(D=d, dt_bias=dt_bias, dt_softplus=True)
    selective_state_update(state_1, x_1, dt_1, a, b_1, c_1, out=out_1, **ssu_kwargs)
    selective_state_update(state_n, x_n, dt_n, a, b_n, c_n, out=out_n, **ssu_kwargs)

    assert torch.equal(out_n[:1], out_1)
    assert torch.equal(state_n[:1], state_1)


@requires_triton_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_conv_update_batch1_matches_batch_n(enable_batch_invariant, dtype):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    device = torch.device(DEVICE)
    torch.manual_seed(2)
    dim, width, batch_n = 64, 4, 6
    x_n = torch.randn(batch_n, dim, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_n = torch.randn(batch_n + 1, dim, width - 1, device=device, dtype=dtype)
    indices_n = torch.arange(1, batch_n + 1, dtype=torch.int32, device=device)

    x_1 = x_n[:1].clone()
    conv_1 = conv_n[:2].clone()
    indices_1 = torch.tensor([1], dtype=torch.int32, device=device)
    conv_n_in = conv_n.clone()

    out_1 = causal_conv1d_update(
        x_1, conv_1, weight, bias, activation="silu", conv_state_indices=indices_1
    )
    out_n = causal_conv1d_update(
        x_n, conv_n_in, weight, bias, activation="silu", conv_state_indices=indices_n
    )

    assert torch.equal(out_n[:1], out_1)
    assert torch.equal(conv_n_in[1:2], conv_1[1:2])


@requires_triton_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_rmsnorm_batch1_matches_batch_n(enable_batch_invariant, dtype):
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated

    device = torch.device(DEVICE)
    torch.manual_seed(3)
    hidden, batch_n = 128, 7
    x_n = torch.randn(batch_n, hidden, device=device, dtype=dtype)
    gate_n = torch.randn(batch_n, hidden, device=device, dtype=dtype)
    weight = torch.randn(hidden, device=device, dtype=dtype)

    out_1 = rms_norm_gated(
        x_n[:1].clone(),
        weight,
        bias=None,
        z=gate_n[:1].clone(),
        eps=1e-5,
        norm_before_gate=False,
    )
    out_n = rms_norm_gated(
        x_n, weight, bias=None, z=gate_n, eps=1e-5, norm_before_gate=False
    )
    assert torch.equal(out_n[:1], out_1)


def _ssd_split_meta(computed: int, n: int, chunk_size: int, device: torch.device):
    """Builder rule: chunks align to absolute position, not the split start."""
    from vllm.utils.math_utils import cdiv

    cu = [0]
    pos = 0
    remaining = n
    if computed % chunk_size:
        take = min(cdiv(computed, chunk_size) * chunk_size - computed, remaining)
        pos += take
        remaining -= take
        cu.append(pos)
    while remaining > 0:
        take = min(chunk_size, remaining)
        pos += take
        remaining -= take
        cu.append(pos)
    t = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
    return t(cu), t([len(cu) - 2]), t([0] * (len(cu) - 1))


def _run_ssd_with_start(x, dt, a, b, c, chunk_size, computed, initial_states):
    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )

    n = x.shape[0]
    device = x.device
    cu_chunk, last_idx, seq_idx = _ssd_split_meta(computed, n, chunk_size, device)
    out = torch.empty_like(x)
    state = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        a,
        b,
        c,
        chunk_size,
        cu_seqlens=torch.tensor([0, n], dtype=torch.int32, device=device),
        cu_chunk_seqlens=cu_chunk,
        last_chunk_indices=last_idx,
        seq_idx=seq_idx,
        out=out,
        D=None,
        initial_states=initial_states,
        dt_softplus=False,
        state_dtype=torch.float32,
    )
    return out, state


@requires_triton_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ssd_aligned_split_matches_unsplit(enable_batch_invariant, dtype):
    """Chunked-prefill resume at a chunk_size multiple is bit-identical."""
    device = torch.device(DEVICE)
    torch.manual_seed(5)
    nheads, headdim, dstate, ngroups = 4, 32, 16, 2
    chunk_size = 16
    seqlen = 80
    split = 48  # 48 % 16 == 0
    assert split % chunk_size == 0

    x, dt, a, b, c = _ssd_inputs(
        seqlen, nheads, headdim, dstate, ngroups, dtype, device
    )
    full, state_full = _run_ssd_with_start(x, dt, a, b, c, chunk_size, 0, None)
    part1, state1 = _run_ssd_with_start(
        x[:split], dt[:split], a, b[:split], c[:split], chunk_size, 0, None
    )
    part2, state2 = _run_ssd_with_start(
        x[split:],
        dt[split:],
        a,
        b[split:],
        c[split:],
        chunk_size,
        split,
        state1.clone(),
    )
    assert torch.equal(torch.cat([part1, part2], dim=0), full)
    assert torch.equal(state2, state_full)


@requires_triton_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ssd_unaligned_split_continuation_runs(enable_batch_invariant, dtype):
    """Resume from a non-zero, unaligned position (production scheduler avoids this).

    A100 kernel repros differ from the split token when
    ``s % chunk_size != 0``, including the final state. This only
    asserts the path runs and that the first half (one shot) matches.
    """
    device = torch.device(DEVICE)
    torch.manual_seed(6)
    nheads, headdim, dstate, ngroups = 4, 32, 16, 2
    chunk_size = 16
    seqlen = 80
    split = 37  # 37 % 16 == 5
    assert split % chunk_size != 0

    x, dt, a, b, c = _ssd_inputs(
        seqlen, nheads, headdim, dstate, ngroups, dtype, device
    )
    full, state_full = _run_ssd_with_start(x, dt, a, b, c, chunk_size, 0, None)
    part1, state1 = _run_ssd_with_start(
        x[:split], dt[:split], a, b[:split], c[:split], chunk_size, 0, None
    )
    part2, state2 = _run_ssd_with_start(
        x[split:],
        dt[split:],
        a,
        b[split:],
        c[split:],
        chunk_size,
        split,
        state1.clone(),
    )
    split_out = torch.cat([part1, part2], dim=0)
    assert part1.shape == x[:split].shape
    assert part2.shape == x[split:].shape
    # Unaligned intra-chunk terms and the final state need not match the
    # unsplit run on A100. The scheduler keeps production prefills off
    # this path. First-half (computed in one shot) must still match.
    assert torch.equal(part1, full[:split])
    assert split_out.shape == full.shape
    assert state2.shape == state_full.shape


@requires_triton_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_row_matches_when_packed_with_prefill(
    enable_batch_invariant, default_vllm_config, dtype
):
    """Per-layer decode row: alone vs packed with a large prefill-sized batch."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

    device = torch.device(DEVICE)
    torch.manual_seed(7)
    hidden = 128
    nheads, headdim, dstate, ngroups = 4, 32, 16, 2
    prefill_tokens = 48

    ln = RMSNorm(hidden, eps=1e-5).to(device=device, dtype=dtype)
    ln.weight.data.normal_()

    x_decode = torch.randn(1, hidden, device=device, dtype=dtype)
    x_prefill = torch.randn(prefill_tokens, hidden, device=device, dtype=dtype)
    residual_decode = torch.randn(1, hidden, device=device, dtype=dtype)
    residual_prefill = torch.randn(prefill_tokens, hidden, device=device, dtype=dtype)

    def _norm_row(x, residual):
        out, _ = ln(x.clone(), residual.clone())
        return out

    alone_norm = _norm_row(x_decode, residual_decode)
    packed_norm, _ = ln(
        torch.cat([x_decode, x_prefill], dim=0),
        torch.cat([residual_decode, residual_prefill], dim=0),
    )
    assert torch.equal(packed_norm[:1], alone_norm)

    state_1 = torch.randn(1, nheads, headdim, dstate, device=device, dtype=dtype)
    x_ssm_1 = torch.randn(1, nheads, headdim, device=device, dtype=dtype)
    dt_1 = torch.randn(1, nheads, headdim, device=device, dtype=dtype)
    dt_bias = torch.rand(nheads, headdim, device=device) - 4.0
    a = -torch.rand(nheads, headdim, dstate, device=device) - 1.0
    b_1 = torch.randn(1, ngroups, dstate, device=device, dtype=dtype)
    c_1 = torch.randn(1, ngroups, dstate, device=device, dtype=dtype)
    d = torch.randn(nheads, headdim, device=device)

    state_n = torch.cat(
        [
            state_1.clone(),
            torch.randn(
                prefill_tokens, nheads, headdim, dstate, device=device, dtype=dtype
            ),
        ],
        dim=0,
    )
    x_ssm_n = torch.cat(
        [
            x_ssm_1.clone(),
            torch.randn(prefill_tokens, nheads, headdim, device=device, dtype=dtype),
        ],
        dim=0,
    )
    dt_n = torch.cat(
        [
            dt_1.clone(),
            torch.randn(prefill_tokens, nheads, headdim, device=device, dtype=dtype),
        ],
        dim=0,
    )
    b_n = torch.cat(
        [
            b_1.clone(),
            torch.randn(prefill_tokens, ngroups, dstate, device=device, dtype=dtype),
        ],
        dim=0,
    )
    c_n = torch.cat(
        [
            c_1.clone(),
            torch.randn(prefill_tokens, ngroups, dstate, device=device, dtype=dtype),
        ],
        dim=0,
    )

    out_1 = torch.empty_like(x_ssm_1)
    out_n = torch.empty_like(x_ssm_n)
    ssu_kwargs = dict(D=d, dt_bias=dt_bias, dt_softplus=True)
    state_1_in = state_1.clone()
    selective_state_update(
        state_1_in, x_ssm_1, dt_1, a, b_1, c_1, out=out_1, **ssu_kwargs
    )
    selective_state_update(state_n, x_ssm_n, dt_n, a, b_n, c_n, out=out_n, **ssu_kwargs)
    assert torch.equal(out_n[:1], out_1)
    assert torch.equal(state_n[:1], state_1_in)

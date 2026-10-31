# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GateLinear must use the batch-invariant matmul under VLLM_BATCH_INVARIANT."""

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear_mod
from vllm.model_executor.determinism import batch_invariant as bi
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

pytestmark = pytest.mark.cpu_test


def _make_cuda_gate(monkeypatch: pytest.MonkeyPatch) -> GateLinear:
    for target in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(f"{target}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_world_size", lambda: 1
        )
    platform = gate_linear_mod.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: True)
    monkeypatch.setattr(platform, "is_rocm", lambda: False)
    monkeypatch.setattr(platform, "is_device_capability", lambda *a, **k: False)
    monkeypatch.setattr(platform, "is_device_capability_family", lambda *a, **k: False)
    return GateLinear(
        input_size=2048,
        output_size=64,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.bfloat16,
    )


def test_gate_linear_bi_uses_batch_invariant_matmul(monkeypatch: pytest.MonkeyPatch):
    """H100 tier-5 ``torch.mm(..., out_dtype=fp32)`` must not run under BI."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()

    called: dict[str, object] = {}

    def fake_mm(a, b, bias=None):
        called["shapes"] = (tuple(a.shape), tuple(b.shape))
        called["dtypes"] = (a.dtype, b.dtype)
        return torch.zeros(a.shape[0], b.shape[1], dtype=a.dtype)

    monkeypatch.setattr(bi, "mm_batch_invariant", fake_mm)

    gate = _make_cuda_gate(monkeypatch)
    assert gate.allow_cublas_router_gemm

    x = torch.randn(7, 2048, dtype=torch.bfloat16)
    output, _bias = gate(x)
    assert called["shapes"] == ((7, 2048), (2048, 64))
    assert called["dtypes"] == (torch.bfloat16, torch.bfloat16)
    assert output.shape == (7, 64)
    assert output.dtype == torch.float32
    envs.disable_envs_cache()

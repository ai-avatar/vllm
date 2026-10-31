# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance tests for unquantized fused MoE used by NemotronH."""

import pytest
import torch

from tests.kernels.moe.utils import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

requires_gpu = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Fused MoE Triton kernels require a CUDA-alike or XPU device.",
)


def test_moe_default_config_is_pinned_under_bi():
    cfg = get_default_config(
        M=1, E=8, N=128, K=64, topk=2, dtype=None, block_shape=None
    )
    assert cfg["BLOCK_SIZE_M"] == 64
    assert cfg["BLOCK_SIZE_N"] == 64
    assert cfg["BLOCK_SIZE_K"] == 32
    assert cfg["SPLIT_K"] == 1
    assert (
        get_default_config(
            M=192, E=8, N=128, K=64, topk=2, dtype=None, block_shape=None
        )
        == cfg
    )


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_moe_row0_matches_across_batch_sizes(dtype):
    device = torch.device(DEVICE)
    torch.manual_seed(4)
    num_experts, topk, hidden, intermediate = 4, 2, 64, 32
    batch_n = 6

    w1 = torch.randn(num_experts, 2 * intermediate, hidden, device=device, dtype=dtype)
    w2 = torch.randn(num_experts, hidden, intermediate, device=device, dtype=dtype)
    hidden_n = torch.randn(batch_n, hidden, device=device, dtype=dtype)
    score_n = torch.randn(batch_n, num_experts, device=device, dtype=torch.float32)

    out_1 = fused_moe(hidden_n[:1].clone(), w1, w2, score_n[:1].clone(), topk)
    out_n = fused_moe(hidden_n, w1, w2, score_n, topk)
    assert torch.equal(out_n[:1], out_1)

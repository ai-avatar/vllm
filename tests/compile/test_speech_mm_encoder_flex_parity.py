# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU parity: compiled speech-encoder flex blocks vs eager.

NeMo calls ``_get_flex_attention(q)`` which returns import-time
``torch.compile(flex_attention, dynamic=True)``. Inductor's
flex-decoding kernel (Q < 128 frames, ~10.24 s) is wrong for this
encoder's rel-pos ``score_mod`` and padding ``block_mask``. These cases
cover short clips (2–8 s at 80 ms/frame) through that NeMo call path.
Skip on CPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention as _raw_flex

from vllm.compilation.mm_encoder import (
    isolate_flex_attention_from_compile,
    speech_encoder_layer_compile_kwargs,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

# Same import-time compiled wrapper NeMo keeps on
# transformer_encoder_utils._flex_attention_compiled.
_flex_attention_compiled = torch.compile(_raw_flex, dynamic=True)


def _get_flex_attention(q):
    """Mirror ``nemo...transformer_encoder_utils._get_flex_attention``."""
    try:
        from nemo.collections.asr.modules.transformer_encoder_utils import (
            _get_flex_attention as nemo_get,
        )
    except ImportError:
        return _flex_attention_compiled
    return nemo_get(q)


def _try_create_block_mask():
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        pytest.skip("flex_attention is not available")
    return create_block_mask


class _NeMoStyleFlexBlock(nn.Module):
    """Encoder block that calls flex through NeMo's compiled wrapper."""

    def __init__(self, dim: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def forward(
        self, x: torch.Tensor, block_mask: object | None = None
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        batch, frames, dim = x.shape
        qkv = self.qkv(x).view(batch, frames, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        def score_mod(score, _b, _h, q_idx, kv_idx):
            return score + (q_idx - kv_idx).to(score.dtype) * 0.01

        flex_fn = _get_flex_attention(q)
        y = flex_fn(q, k, v, score_mod=score_mod, block_mask=block_mask)
        y = y.transpose(1, 2).reshape(batch, frames, dim)
        return self.out(y) + residual


def _padding_block_mask(create_block_mask, frames: int, valid: int, device):
    def mask_mod(_b, _h, q_idx, kv_idx):
        return (q_idx < valid) & (kv_idx < valid)

    return create_block_mask(mask_mod, 1, None, frames, frames, device=device)


def _rel_err(compiled: torch.Tensor, eager: torch.Tensor) -> float:
    scale = eager.detach().float().abs().amax().clamp_min(1e-3)
    return float((compiled.float() - eager.float()).abs().amax() / scale)


# 80 ms / encoder frame: 43≈3.4 s, 80≈6.4 s, 1500≈120 s, 2500≈200 s.
@pytest.mark.parametrize("frames", [43, 80, 1500, 2500])
def test_compiled_nemo_flex_path_matches_eager_short_and_long(frames: int):
    create_block_mask = _try_create_block_mask()
    isolate_flex_attention_from_compile()
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = _NeMoStyleFlexBlock().to(device=device, dtype=dtype)
    x = torch.randn(1, frames, 64, device=device, dtype=dtype)
    mask = _padding_block_mask(create_block_mask, frames, frames, device)

    with torch.no_grad():
        eager = block(x, mask)
        block.compile(**speech_encoder_layer_compile_kwargs())
        compiled = block(x, mask)
        compiled2 = block(x, mask)

    assert _rel_err(compiled, eager) < 0.05, (
        f"compiled vs eager rel={_rel_err(compiled, eager):.3f} at T={frames}"
    )
    assert torch.equal(compiled, compiled2)


def test_compiled_nemo_flex_path_matches_eager_at_t43():
    create_block_mask = _try_create_block_mask()
    isolate_flex_attention_from_compile()
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = _NeMoStyleFlexBlock().to(device=device, dtype=dtype)
    frames, valid = 80, 43
    x = torch.randn(1, frames, 64, device=device, dtype=dtype)
    x[:, valid:].zero_()
    mask = _padding_block_mask(create_block_mask, frames, valid, device)

    with torch.no_grad():
        eager = block(x, mask)
        block.compile(**speech_encoder_layer_compile_kwargs())
        compiled = block(x, mask)

    assert _rel_err(compiled[:, :valid], eager[:, :valid]) < 0.05

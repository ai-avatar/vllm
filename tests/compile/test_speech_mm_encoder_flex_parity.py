# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU parity: compiled speech-encoder flex blocks vs eager.

NeMo calls ``_get_flex_attention(q)`` which returns import-time
``torch.compile(flex_attention, dynamic=True)``. Inductor's
flex-decoding kernel (Q < 128 frames, ~10.24 s) is wrong for this
encoder's rel-pos ``score_mod`` and padding ``block_mask``. These cases
cover short clips (2–8 s at 80 ms/frame) through that NeMo call path.
A two-process case checks short and long-window embeddings stay
bitwise identical across cold compiles under BI. Skip on CPU.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import traceback

import pytest
import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention as _raw_flex

from vllm.compilation.mm_encoder import (
    isolate_flex_attention_from_compile,
    speech_encoder_layer_compile_kwargs,
    speech_encoder_served_frames,
)

_ENCODER_WINDOW = 550
_SHORT_FRAMES = 80
_LONG_FRAMES = 3 * _ENCODER_WINDOW + 256

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


def _pad_time(x: torch.Tensor, target: int) -> torch.Tensor:
    extra = target - x.shape[1]
    if extra <= 0:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, extra))


def _encode_windows(block, x: torch.Tensor, create_block_mask):
    """Whole-forward for <=2 windows; else chunk like ``_call_long``.

    Last windows pad to ``window`` so the compiled kernel only sees the
    closed shape set (same as ``speech_encoder_served_frames``).
    """
    valid = x.shape[1]
    if valid <= 2 * _ENCODER_WINDOW:
        target = speech_encoder_served_frames(
            valid,
            window=_ENCODER_WINDOW,
            max_windows=2,
            bucket_step=256,
            long=False,
        )
        padded = _pad_time(x, target)
        mask = _padding_block_mask(create_block_mask, target, valid, x.device)
        return block(padded, mask)[:, :valid]
    parts = []
    for start in range(0, valid, _ENCODER_WINDOW):
        chunk = x[:, start : start + _ENCODER_WINDOW]
        chunk_valid = chunk.shape[1]
        padded = _pad_time(chunk, _ENCODER_WINDOW)
        mask = _padding_block_mask(
            create_block_mask, _ENCODER_WINDOW, chunk_valid, x.device
        )
        parts.append(block(padded, mask)[:, :chunk_valid])
    return torch.cat(parts, dim=1)


def _tensor_to_payload(tensor: torch.Tensor) -> dict[str, object]:
    """CPU bytes only — torch tensor IPC uses SHM that dies with the child."""
    cpu = tensor.detach().contiguous().cpu().clone()
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "data": raw}


def _payload_to_tensor(payload: dict[str, object]) -> torch.Tensor:
    shape = payload["shape"]
    dtype_s = str(payload["dtype"]).removeprefix("torch.")
    raw = payload["data"]
    assert isinstance(shape, list)
    assert isinstance(raw, (bytes, bytearray))
    dtype = getattr(torch, dtype_s)
    storage = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return storage.view(dtype).reshape(shape).clone()


def _compiled_encoder_embeddings_worker(conn, seed: int, cache_dir: str) -> None:
    """Fresh process: compile the encoder and encode short + long clips."""
    try:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")
        import vllm.envs as envs
        from vllm.compilation import mm_encoder as mm_encoder_mod
        from vllm.compilation.mm_encoder import (
            configure_speech_encoder_dynamo,
            isolate_flex_attention_from_compile,
            speech_encoder_layer_compile_kwargs,
        )

        envs.disable_envs_cache()
        mm_encoder_mod._INDUCTOR_PINNED = False
        mm_encoder_mod._BI_FLEX_COMPILED = None
        isolate_flex_attention_from_compile()
        configure_speech_encoder_dynamo()
        create_block_mask = _try_create_block_mask()
        torch.manual_seed(seed)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        block = _NeMoStyleFlexBlock().to(device=device, dtype=dtype)
        block.compile(**speech_encoder_layer_compile_kwargs())
        embs = []
        with torch.no_grad():
            for frames in (_SHORT_FRAMES, _LONG_FRAMES):
                torch.manual_seed(seed + frames)
                x = torch.randn(1, frames, 64, device=device, dtype=dtype)
                embs.append(
                    _tensor_to_payload(_encode_windows(block, x, create_block_mask))
                )
        conn.send({"ok": True, "embs": embs})
    except Exception as exc:
        conn.send({"ok": False, "error": f"{exc}\n{traceback.format_exc()}"})
    finally:
        conn.close()


def test_compiled_encoder_embeddings_match_across_processes():
    """Two cold inductor compiles must match bitwise under BI.

    Long-window encoder path (>2 windows) plus a short clip. Separate
    cache dirs so this is two process starts, not a shared autotune hit.
    """
    ctx = multiprocessing.get_context("spawn")
    results = []
    for _ in range(2):
        cache_dir = tempfile.mkdtemp(prefix="speech-enc-bi-")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_compiled_encoder_embeddings_worker,
            args=(child_conn, 0, cache_dir),
        )
        proc.start()
        child_conn.close()
        try:
            if not parent_conn.poll(timeout=600):
                proc.terminate()
                proc.join(timeout=30)
                pytest.fail("compiled encoder worker timed out")
            results.append(parent_conn.recv())
        finally:
            parent_conn.close()
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
        assert proc.exitcode == 0, results[-1]

    for result in results:
        assert result["ok"], result.get("error")
    first = [_payload_to_tensor(p) for p in results[0]["embs"]]
    second = [_payload_to_tensor(p) for p in results[1]["embs"]]
    assert len(first) == 2
    assert first[0].shape[1] == _SHORT_FRAMES
    assert first[1].shape[1] == _LONG_FRAMES
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


def test_compiled_encoder_embeddings_stable_after_varied_burst():
    """Same long input must match bitwise after a burst of other lengths."""
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    import vllm.envs as envs
    from vllm.compilation import mm_encoder as mm_encoder_mod
    from vllm.compilation.mm_encoder import configure_speech_encoder_dynamo

    envs.disable_envs_cache()
    mm_encoder_mod._INDUCTOR_PINNED = False
    isolate_flex_attention_from_compile()
    configure_speech_encoder_dynamo()
    create_block_mask = _try_create_block_mask()
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = _NeMoStyleFlexBlock().to(device=device, dtype=dtype)
    block.compile(**speech_encoder_layer_compile_kwargs())
    long_x = torch.randn(1, _LONG_FRAMES, 64, device=device, dtype=dtype)
    burst = (1, 50, 80, 129, 256, 400, 549, 551, 800, 1100, 1101, 1651, 1906, 3000)

    with torch.no_grad():
        first = _encode_windows(block, long_x, create_block_mask)
        for frames in burst:
            other = torch.randn(1, frames, 64, device=device, dtype=dtype)
            _encode_windows(block, other, create_block_mask)
        second = _encode_windows(block, long_x, create_block_mask)

    assert first.shape[1] == _LONG_FRAMES
    assert torch.equal(first, second)
    envs.disable_envs_cache()

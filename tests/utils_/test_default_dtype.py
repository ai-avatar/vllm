# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for serving default_dtype isolation.

NeMo ``_default_dtype(stream_dtype)`` used to call ``set_default_dtype``.
Dynamo's ``convert_frame`` snapshots ``get_default_dtype()`` and restores
it; a lying get-wrap made that restore write bf16 into the process and
raise ``Global default_dtype state changed while dynamo tracing``.

The factory pin must produce the same tensors as the old global window
without changing the process default, including under ``torch.compile``.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import types
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.utils.default_dtype import (
    _coerce_streaming_state_dtype,
    check_serving_default_dtype,
    install_default_dtype_guards,
    process_default_dtype,
    serving_default_dtype_mismatch_count,
    virtual_default_dtype,
    wrap_nemo_default_dtype,
)
from vllm.utils.torch_utils import set_default_torch_dtype


@contextlib.contextmanager
def _restore_default_dtype():
    prev = process_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextlib.contextmanager
def _legacy_global_window(dtype: torch.dtype):
    """NeMo / old ``_pin_default_dtype`` get/set/restore."""
    prev = torch.get_default_dtype()
    if dtype != prev and dtype.is_floating_point:
        torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


class _SortformerLikeBlock(nn.Module):
    """LayerNorm + Linear + implicit ``torch.zeros`` (streaming pad)."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(x.size(0), x.size(1), x.size(2), device=x.device)
        return self.lin(self.norm(x)) + pad


def test_two_thread_unlocked_interleave_leaks_bf16():
    """Scripted A/B interleave of raw get/set/restore still leaks bf16."""
    with _restore_default_dtype():
        a_set = threading.Event()
        b_got = threading.Event()
        a_restored = threading.Event()

        def thread_a() -> None:
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            a_set.set()
            assert b_got.wait(timeout=5)
            torch.set_default_dtype(prev)
            a_restored.set()

        def thread_b() -> None:
            assert a_set.wait(timeout=5)
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            b_got.set()
            assert a_restored.wait(timeout=5)
            torch.set_default_dtype(prev)

        t_a = threading.Thread(target=thread_a)
        t_b = threading.Thread(target=thread_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)
        assert not t_a.is_alive() and not t_b.is_alive()
        assert process_default_dtype() == torch.bfloat16


def test_factory_pin_matches_legacy_window_values():
    """Implicit zeros / encoder block match the old set_default_dtype window."""
    install_default_dtype_guards()
    block = _SortformerLikeBlock().to(torch.bfloat16)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    with _restore_default_dtype():
        with _legacy_global_window(torch.bfloat16):
            legacy_zeros = torch.zeros(3, 5)
            legacy_long = torch.zeros(2, dtype=torch.long)
            legacy_out = block(x).detach().clone()
        with virtual_default_dtype(torch.bfloat16):
            pinned_zeros = torch.zeros(3, 5)
            pinned_long = torch.zeros(2, dtype=torch.long)
            pinned_out = block(x)
            assert torch.get_default_dtype() == torch.float32
        assert pinned_zeros.dtype == legacy_zeros.dtype == torch.bfloat16
        assert pinned_long.dtype == legacy_long.dtype == torch.long
        assert torch.equal(pinned_zeros, legacy_zeros)
        assert torch.equal(pinned_out, legacy_out)
        assert process_default_dtype() == torch.float32


def test_nemo_streaming_encoder_section_checker_thread_sees_fp32():
    """``_forward_online_impl``-style pin vs a step-boundary checker thread."""
    install_default_dtype_guards()

    @contextlib.contextmanager
    def _default_dtype(dtype: torch.dtype):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(prev)

    name = "nemo.collections.asr.modules._vllm_pee_dtype_test"
    holder = cast(Any, types.ModuleType(name))
    holder._default_dtype = _default_dtype
    sys.modules[name] = holder
    try:
        install_default_dtype_guards()
        pinned = holder._default_dtype
        assert getattr(pinned, "_vllm_default_dtype_virtual", False)

        inside = threading.Event()
        leave = threading.Event()
        seen: list[torch.dtype] = []
        errors: list[Exception] = []

        def init_streaming_state(batch_size: int, device: str) -> SimpleNamespace:
            return SimpleNamespace(
                spkcache=torch.zeros((batch_size, 4, 8), device=device),
                fifo=torch.zeros((batch_size, 2, 8), device=device),
                lengths=torch.zeros((batch_size,), dtype=torch.long, device=device),
            )

        def encoder_thread() -> None:
            stream_dtype = torch.bfloat16
            with pinned(stream_dtype):
                state = init_streaming_state(1, "cpu")
                preds = torch.zeros((1, 0, 4))
                assert state.spkcache.dtype == stream_dtype
                assert state.fifo.dtype == stream_dtype
                assert state.lengths.dtype == torch.long
                assert preds.dtype == stream_dtype
                assert torch.get_default_dtype() == torch.float32
                inside.set()
                assert leave.wait(timeout=5)

        def checker_thread() -> None:
            try:
                assert inside.wait(timeout=5)
                for _ in range(20):
                    seen.append(process_default_dtype())
                    check_serving_default_dtype("streaming-checker")
                    assert torch.get_default_dtype() == torch.float32
            except Exception as exc:
                errors.append(exc)
            finally:
                leave.set()

        with _restore_default_dtype():
            before = serving_default_dtype_mismatch_count()
            t_enc = threading.Thread(target=encoder_thread)
            t_chk = threading.Thread(target=checker_thread)
            t_enc.start()
            t_chk.start()
            t_enc.join(timeout=5)
            t_chk.join(timeout=5)
            assert not t_enc.is_alive() and not t_chk.is_alive()
            assert errors == []
            assert seen
            assert set(seen) == {torch.float32}
            assert process_default_dtype() == torch.float32
            assert serving_default_dtype_mismatch_count() == before
    finally:
        sys.modules.pop(name, None)


def test_dynamo_trace_under_factory_pin_keeps_process_fp32():
    """Sortformer-like block under the shim must not trip convert_frame."""
    install_default_dtype_guards()
    block = _SortformerLikeBlock().to(torch.bfloat16)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    with _restore_default_dtype():
        with _legacy_global_window(torch.bfloat16):
            legacy = block(x).detach().clone()
        with virtual_default_dtype(torch.bfloat16):
            compiled = torch.compile(block, backend="eager")
            out = compiled(x)
            assert torch.get_default_dtype() == torch.float32
        assert process_default_dtype() == torch.float32
        assert out.dtype == legacy.dtype
        assert torch.equal(out, legacy)


def test_coerce_streaming_state_dtype_matches_param_dtype():
    state = SimpleNamespace(
        spkcache=torch.zeros(1, 2, 4),
        fifo=torch.zeros(1, 1, 4),
        lengths=torch.zeros(1, dtype=torch.long),
    )
    out = _coerce_streaming_state_dtype(state, torch.bfloat16)
    assert out.spkcache.dtype == torch.bfloat16
    assert out.fifo.dtype == torch.bfloat16
    assert out.lengths.dtype == torch.long


def test_wrap_nemo_default_dtype_is_idempotent():
    @contextlib.contextmanager
    def _cm(dtype: torch.dtype):
        yield dtype

    once = wrap_nemo_default_dtype(_cm)
    twice = wrap_nemo_default_dtype(once)
    assert once is twice


def test_set_default_torch_dtype_still_sets_process_default():
    with _restore_default_dtype():
        with set_default_torch_dtype(torch.bfloat16):
            assert process_default_dtype() == torch.bfloat16
        assert process_default_dtype() == torch.float32


def test_check_serving_default_dtype_ok_when_fp32():
    with _restore_default_dtype():
        check_serving_default_dtype("test")


def test_check_serving_default_dtype_warns_never_raises(caplog_vllm, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    with _restore_default_dtype(), caplog_vllm.at_level("WARNING"):
        torch.set_default_dtype(torch.bfloat16)
        before = serving_default_dtype_mismatch_count()
        check_serving_default_dtype("unit-test")
        assert serving_default_dtype_mismatch_count() == before + 1
        assert "default_dtype is torch.bfloat16" in caplog_vllm.text
        assert "unit-test" in caplog_vllm.text

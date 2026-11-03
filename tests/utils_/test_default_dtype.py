# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the process-global default_dtype race.

Reproduces the EngineCore model-runner vs nemo-ctc-prepare interleave:
two threads do unlocked ``prev = get(); set(x); set(prev)`` and a
crossed restore leaves the process default at bf16. The locked helper
and the NeMo CM wrap keep the default at fp32.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import types
from typing import Any, cast

import pytest
import torch

import vllm.envs as envs
from vllm.utils.default_dtype import (
    check_serving_default_dtype,
    install_default_dtype_guards,
    locked_default_dtype,
    wrap_dtype_contextmanager,
)
from vllm.utils.torch_utils import set_default_torch_dtype


@contextlib.contextmanager
def _restore_default_dtype():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def test_two_thread_unlocked_interleave_leaks_bf16():
    """Scripted A/B interleave: B snapshots bf16, A restores fp32, B leaks."""
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
        assert torch.get_default_dtype() == torch.bfloat16


def test_locked_default_dtype_survives_two_thread_interleave():
    """Same two-thread pin; the section lock cannot leak bf16."""
    with _restore_default_dtype():
        started = threading.Barrier(2)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                started.wait(timeout=5)
                with locked_default_dtype(torch.bfloat16):
                    assert torch.get_default_dtype() == torch.bfloat16
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert errors == []
        assert torch.get_default_dtype() == torch.float32


def test_locked_section_blocks_other_thread_until_restore():
    """The RLock covers the whole get/set/body/restore, not just set()."""
    with _restore_default_dtype():
        a_holding = threading.Event()
        a_may_leave = threading.Event()
        b_entered = threading.Event()

        def thread_a() -> None:
            with locked_default_dtype(torch.bfloat16):
                a_holding.set()
                assert a_may_leave.wait(timeout=5)

        def thread_b() -> None:
            assert a_holding.wait(timeout=5)
            with locked_default_dtype(torch.bfloat16):
                b_entered.set()

        t_a = threading.Thread(target=thread_a)
        t_b = threading.Thread(target=thread_b)
        t_a.start()
        t_b.start()
        assert a_holding.wait(timeout=5)
        assert not b_entered.wait(timeout=0.2)
        a_may_leave.set()
        t_a.join(timeout=5)
        t_b.join(timeout=5)
        assert b_entered.is_set()
        assert torch.get_default_dtype() == torch.float32


def test_nemo_style_cm_wrap_prevents_crossed_restore():
    """Monkeypatched NeMo ``_default_dtype`` holds the process lock."""

    @contextlib.contextmanager
    def _default_dtype(dtype: torch.dtype):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(prev)

    name = "nemo.collections.asr.parts.utils._vllm_dtype_lock_test"
    holder = cast(Any, types.ModuleType(name))
    holder._default_dtype = _default_dtype
    sys.modules[name] = holder
    try:
        install_default_dtype_guards()
        wrapped = holder._default_dtype
        assert wrapped is not _default_dtype
        assert getattr(wrapped, "_vllm_default_dtype_locked", False)

        with _restore_default_dtype():
            started = threading.Barrier(2)
            errors: list[Exception] = []

            def worker() -> None:
                try:
                    started.wait(timeout=5)
                    with wrapped(torch.bfloat16):
                        assert torch.get_default_dtype() == torch.bfloat16
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
                assert not thread.is_alive()
            assert errors == []
            assert torch.get_default_dtype() == torch.float32
    finally:
        sys.modules.pop(name, None)


def test_wrap_dtype_contextmanager_is_idempotent():
    @contextlib.contextmanager
    def _cm(dtype: torch.dtype):
        yield dtype

    once = wrap_dtype_contextmanager(_cm)
    twice = wrap_dtype_contextmanager(once)
    assert once is twice


def test_set_default_torch_dtype_uses_process_lock():
    with _restore_default_dtype():
        with set_default_torch_dtype(torch.bfloat16):
            assert torch.get_default_dtype() == torch.bfloat16
        assert torch.get_default_dtype() == torch.float32


def test_check_serving_default_dtype_ok_when_fp32():
    with _restore_default_dtype():
        check_serving_default_dtype("test")


def test_check_serving_default_dtype_warns(caplog_vllm, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    with _restore_default_dtype(), caplog_vllm.at_level("WARNING"):
        torch.set_default_dtype(torch.bfloat16)
        check_serving_default_dtype("unit-test")
        assert "default_dtype is torch.bfloat16" in caplog_vllm.text
        assert "unit-test" in caplog_vllm.text


def test_check_serving_default_dtype_raises_under_bi(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    with _restore_default_dtype():
        torch.set_default_dtype(torch.bfloat16)
        with pytest.raises(RuntimeError, match="default_dtype"):
            check_serving_default_dtype("unit-test")

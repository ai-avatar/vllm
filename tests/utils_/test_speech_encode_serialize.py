# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the speech encode / flex process lock."""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest
import torch

from vllm.compilation import mm_encoder as mm_encoder_mod
from vllm.compilation.mm_encoder import isolate_flex_attention_from_compile
from vllm.utils.speech_encode_serialize import (
    _WRAPPED_ATTR,
    install_speech_encode_serialize,
    speech_encode_lock_held,
    speech_encode_section,
    wrap_with_speech_encode_lock,
)

pytestmark = pytest.mark.cpu_test


def test_concurrent_sections_do_not_overlap() -> None:
    in_section = 0
    max_in = 0
    observed = threading.Lock()

    def worker() -> None:
        nonlocal in_section, max_in
        with speech_encode_section():
            with observed:
                in_section += 1
                max_in = max(max_in, in_section)
            time.sleep(0.03)
            with observed:
                in_section -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max_in == 1
    assert in_section == 0
    assert not speech_encode_lock_held()


def test_section_is_reentrant() -> None:
    with speech_encode_section():
        assert speech_encode_lock_held()
        with speech_encode_section():
            assert speech_encode_lock_held()
        assert speech_encode_lock_held()
    assert not speech_encode_lock_held()


def test_wrap_is_idempotent() -> None:
    def fn() -> str:
        assert speech_encode_lock_held()
        return "ok"

    wrapped = wrap_with_speech_encode_lock(fn)
    assert wrap_with_speech_encode_lock(wrapped) is wrapped
    assert getattr(wrapped, _WRAPPED_ATTR)
    assert wrapped() == "ok"


def test_install_wraps_nemo_encode_and_ctc_prepare() -> None:
    mod = types.ModuleType("nemo.collections.speechlm2.vllm.salm.ctc_timestamps")

    def _encode_step() -> bool:
        return speech_encode_lock_held()

    def _worker_prepare_requests() -> bool:
        return speech_encode_lock_held()

    class ParallelExpert:
        def _asr_window_forward(self) -> bool:
            return speech_encode_lock_held()

        def _streaming_diar_step(self) -> bool:
            return speech_encode_lock_held()

    mod._encode_step = _encode_step  # type: ignore[attr-defined]
    mod._worker_prepare_requests = _worker_prepare_requests  # type: ignore[attr-defined]
    mod.ParallelExpert = ParallelExpert  # type: ignore[attr-defined]
    sys.modules[mod.__name__] = mod
    try:
        install_speech_encode_serialize()
        install_speech_encode_serialize()
        assert getattr(mod._encode_step, _WRAPPED_ATTR)
        assert getattr(mod._worker_prepare_requests, _WRAPPED_ATTR)
        assert getattr(ParallelExpert._asr_window_forward, _WRAPPED_ATTR)
        assert getattr(ParallelExpert._streaming_diar_step, _WRAPPED_ATTR)
        assert mod._encode_step() is True
        assert mod._worker_prepare_requests() is True
        assert ParallelExpert()._asr_window_forward() is True
        assert ParallelExpert()._streaming_diar_step() is True
        assert not speech_encode_lock_held()
    finally:
        del sys.modules[mod.__name__]


def test_create_block_mask_and_flex_hold_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.nn.attention.flex_attention as flex_mod

    held: list[bool] = []

    def fake_mask(*args, **kwargs):
        held.append(speech_encode_lock_held())
        return "mask"

    def fake_flex(*args, **kwargs):
        held.append(speech_encode_lock_held())
        return args[0] if args else None

    monkeypatch.setattr(flex_mod, "create_block_mask", fake_mask)
    isolate_flex_attention_from_compile()
    assert flex_mod.create_block_mask("q") == "mask"
    mm_encoder_mod._force_flex_attention_kernel(fake_flex)(object())
    assert held == [True, True]


def test_annotate_under_lock_does_not_keyerror() -> None:
    errors: list[BaseException] = []

    def worker() -> None:
        for _ in range(2000):
            try:
                with (
                    speech_encode_section(),
                    torch.fx.traceback.annotate({"fallback_to_eager": True}),
                ):
                    pass
            except KeyError as exc:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []

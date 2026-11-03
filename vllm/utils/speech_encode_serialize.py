# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serialize speech encode / flex / ``create_block_mask`` onto one process lock.

``torch.fx.traceback.current_meta`` is a process-global dict. Concurrent
``annotate`` from the model-runner encoder and NeMo's ``nemo-ctc-prepare``
thread (or Sortformer diar) interleave ``del current_meta["custom"]`` and
crash EngineCore with ``KeyError: 'custom'``. Dynamo caches and other
compile/fx state are likewise unsafe from two threads.

NeMo already runs CTC prepare off the engine step loop so decode is not
stalled, and it submits GPU work on the engine's CUDA stream (kernels
stay ordered). Moving prepare onto the model-runner thread would undo
that overlap. This lock covers the Python encode / flex / mask section
only: CTC-vs-decode overlap remains; encode-vs-encode is serialized.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

_LOCK = threading.RLock()
_TLS = threading.local()
_WRAPPED_ATTR = "_vllm_speech_encode_serialized"
_NEMO_ENCODE_NAMES = frozenset(
    (
        "_encode_step",
        "_worker_prepare_requests",
        "_asr_window_forward",
        "_streaming_diar_step",
        "_forward_online_impl",
    )
)


def speech_encode_lock_held() -> bool:
    """True when this thread is inside :func:`speech_encode_section`."""
    return getattr(_TLS, "depth", 0) > 0


@contextlib.contextmanager
def speech_encode_section() -> Iterator[None]:
    """Hold the process speech-encode lock (reentrant on this thread)."""
    depth = getattr(_TLS, "depth", 0)
    _TLS.depth = depth + 1
    _LOCK.acquire()
    try:
        yield
    finally:
        _LOCK.release()
        _TLS.depth = depth


def wrap_with_speech_encode_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run ``fn`` under :func:`speech_encode_section`. Idempotent."""
    if fn is None or getattr(fn, _WRAPPED_ATTR, False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with speech_encode_section():
            return fn(*args, **kwargs)

    setattr(wrapped, _WRAPPED_ATTR, True)
    return wrapped


def _is_nemo_module_name(name: str) -> bool:
    return name == "nemo" or name.startswith("nemo.")


def _wrap_named(owner: Any, name: str, value: Any) -> None:
    if name not in _NEMO_ENCODE_NAMES or not callable(value):
        return
    if getattr(value, _WRAPPED_ATTR, False):
        return
    try:
        setattr(owner, name, wrap_with_speech_encode_lock(value))
    except Exception:
        return


def _wrap_nemo_encode_sites() -> None:
    for name, mod in list(sys.modules.items()):
        if mod is None or not _is_nemo_module_name(name):
            continue
        mapping = getattr(mod, "__dict__", None)
        if not mapping:
            continue
        for attr, value in list(mapping.items()):
            _wrap_named(mod, attr, value)
            if not isinstance(value, type):
                continue
            cls_dict = getattr(value, "__dict__", None)
            if not cls_dict:
                continue
            for cls_attr, cls_value in list(cls_dict.items()):
                _wrap_named(value, cls_attr, cls_value)


def install_speech_encode_serialize() -> None:
    """Wrap loaded NeMo encode / CTC-prepare entries with the process lock.

    Always re-scans so a late NeMo import is still patched. Flex and
    ``create_block_mask`` take the same lock from ``mm_encoder``.
    """
    _wrap_nemo_encode_sites()

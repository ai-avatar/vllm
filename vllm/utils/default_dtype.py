# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide lock for torch's process-global ``default_dtype``.

``torch.get_default_dtype`` / ``torch.set_default_dtype`` are process-global
and not atomic as a get/set/restore pair. EngineCore runs a
``nemo-ctc-prepare`` thread next to the model-runner thread. NeMo's
``_default_dtype`` context manager and the old speech-encoder
``_pin_default_dtype`` helper both did ``prev = get(); set(x); ...; set(prev)``
without a lock. Two threads can interleave so that:

* a compiled frame sees ``GLOBAL_STATE`` ``default_dtype`` change mid-frame
* a crossed restore leaves the process default at bf16

The leftover bf16 default has crashed Sortformer ``layer_norm``
(``expected scalar type Float but found BFloat16``) and silently
perturbed CTC (dynamo guard fail on the creating frame).

Serving paths in this fork must pass ``dtype=`` explicitly and must not
call ``set_default_dtype``. NeMo call sites we cannot edit are wrapped
at model load so the entire set/restore section holds one ``RLock``.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_LOCK = threading.RLock()
_WRAPPED_ATTR = "_vllm_default_dtype_locked"

SERVING_DEFAULT_DTYPE = torch.float32


def default_dtype_lock() -> threading.RLock:
    """Process-wide lock covering a default_dtype get/set/restore section."""
    return _LOCK


@contextlib.contextmanager
def locked_default_dtype(dtype: torch.dtype | None) -> Iterator[None]:
    """Hold :func:`default_dtype_lock` around get/set/restore.

    ``dtype=None`` is a no-op. Used by load-time helpers such as
    ``set_default_torch_dtype``. Serving forwards should pass ``dtype=``
    on factory calls instead of changing the process default.
    """
    if dtype is None:
        yield
        return
    with _LOCK:
        prev = torch.get_default_dtype()
        if prev != dtype:
            torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            if torch.get_default_dtype() != prev:
                torch.set_default_dtype(prev)


def wrap_dtype_contextmanager(cm: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a get/set/restore context manager with :func:`default_dtype_lock`.

    The lock is held for the whole original section, not just each
    ``set_default_dtype`` call. Locking only ``set`` still allows a
    crossed restore to leak bf16.
    """
    if getattr(cm, _WRAPPED_ATTR, False):
        return cm

    @contextlib.contextmanager
    def _locked(*args: Any, **kwargs: Any) -> Iterator[None]:
        with _LOCK, cm(*args, **kwargs):
            yield

    with contextlib.suppress(Exception):
        _locked = functools.update_wrapper(_locked, cm)
    setattr(_locked, _WRAPPED_ATTR, True)
    return _locked


def _wrap_torch_set_default_dtype() -> None:
    """Serialize individual ``set_default_dtype`` calls on the same lock.

    Reentrant with :func:`locked_default_dtype` / wrapped NeMo CMs. This
    does not make an unlocked get/set/restore pair safe; those sections
    must hold the lock themselves.
    """
    current = torch.set_default_dtype
    if getattr(current, _WRAPPED_ATTR, False):
        return

    def _locked_set(dtype: torch.dtype) -> None:
        with _LOCK:
            current(dtype)

    with contextlib.suppress(Exception):
        _locked_set = functools.update_wrapper(_locked_set, current)
    setattr(_locked_set, _WRAPPED_ATTR, True)
    torch.set_default_dtype = _locked_set


def _is_nemo_module_name(name: str) -> bool:
    return name == "nemo" or name.startswith("nemo.")


def _wrap_attr_if_default_dtype(owner: Any, attr: str, value: Any) -> None:
    if attr != "_default_dtype" or not callable(value):
        return
    if getattr(value, _WRAPPED_ATTR, False):
        return
    try:
        setattr(owner, attr, wrap_dtype_contextmanager(value))
    except Exception:
        return


def _wrap_nemo_default_dtype_cms() -> None:
    """Wrap NeMo ``_default_dtype`` CMs already imported in this process."""
    for name, mod in list(sys.modules.items()):
        if mod is None or not _is_nemo_module_name(name):
            continue
        mapping = getattr(mod, "__dict__", None)
        if not mapping:
            continue
        _wrap_attr_if_default_dtype(
            mod, "_default_dtype", mapping.get("_default_dtype")
        )
        for _, obj in list(mapping.items()):
            if not isinstance(obj, type):
                continue
            cls_dict = getattr(obj, "__dict__", None)
            if not cls_dict:
                continue
            raw = cls_dict.get("_default_dtype")
            _wrap_attr_if_default_dtype(obj, "_default_dtype", raw)


def install_default_dtype_guards() -> None:
    """Wrap ``torch.set_default_dtype`` and imported NeMo ``_default_dtype`` CMs.

    Idempotent. Safe to call at model load and again after NeMo imports
    more modules (e.g. post-KV speech warmup).
    """
    _wrap_torch_set_default_dtype()
    _wrap_nemo_default_dtype_cms()


def check_serving_default_dtype(where: str = "request") -> None:
    """WARN (and raise under BI) if the process default is not fp32.

    Also (re)installs NeMo / torch wraps so a late import is still locked.
    """
    install_default_dtype_guards()
    current = torch.get_default_dtype()
    if current == SERVING_DEFAULT_DTYPE:
        return
    msg = (
        f"torch default_dtype is {current} at {where}; serving expects "
        f"{SERVING_DEFAULT_DTYPE}. A crossed set_default_dtype restore "
        "likely leaked (EngineCore model-runner vs nemo-ctc-prepare)."
    )
    logger.warning(msg)
    if envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError(msg)

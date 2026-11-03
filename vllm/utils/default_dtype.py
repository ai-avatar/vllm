# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the process ``default_dtype`` at fp32 during serving.

``torch.get_default_dtype`` / ``set_default_dtype`` are process-global.
NeMo ``parallel_expert_encoder._default_dtype(stream_dtype)`` (hot path:
``_forward_online_impl`` / ``_init_streaming_diar``) and the old fork
``_pin_default_dtype`` both called ``set_default_dtype``. Dynamo's
``convert_frame`` snapshots ``get_default_dtype()`` and restores it with
``set_default_dtype``; a Python wrap of ``get_default_dtype`` that
returns a thread-local bf16 makes that restore write bf16 into the
process and then raises ``Global default_dtype state changed while
dynamo tracing``.

Serving therefore never wraps ``get`` / ``set``. NeMo ``_default_dtype``
is replaced with a thread-local factory pin: ``torch.zeros(...,
device=)`` inside ``init_streaming_state`` still gets ``stream_dtype``
(same tensors as the old global window) while every other thread, and
dynamo, see process fp32.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_TLS = threading.local()
_WRAPPED_ATTR = "_vllm_default_dtype_virtual"
_FACTORY_NAMES = ("empty", "zeros", "ones", "full", "randn", "rand")
_STATE_FLOAT_ATTRS = (
    "spkcache",
    "spkcache_preds",
    "fifo",
    "mean_sil_emb",
)

SERVING_DEFAULT_DTYPE = torch.float32

_MISMATCH_COUNT = 0
_PROM: Any = None


def process_default_dtype() -> torch.dtype:
    """Process-global default (never a thread-local serving pin)."""
    return torch.get_default_dtype()


def serving_default_dtype_mismatch_count() -> int:
    """Times a request boundary saw a non-fp32 process default."""
    return _MISMATCH_COUNT


def _tls_dtype() -> torch.dtype | None:
    return getattr(_TLS, "dtype", None)


@contextlib.contextmanager
def virtual_default_dtype(dtype: torch.dtype | None) -> Iterator[None]:
    """Pin factory dtypes on this thread. Does not call get/set_default_dtype."""
    if dtype is None or not getattr(dtype, "is_floating_point", False):
        yield
        return
    prev = _tls_dtype()
    _TLS.dtype = dtype
    try:
        yield
    finally:
        if prev is None:
            _TLS.__dict__.pop("dtype", None)
        else:
            _TLS.dtype = prev


def _dtype_from_args(*args: Any, **kwargs: Any) -> torch.dtype | None:
    found = kwargs.get("dtype")
    if isinstance(found, torch.dtype):
        return found
    for arg in reversed(args):
        if isinstance(arg, torch.dtype):
            return arg
    return None


def wrap_nemo_default_dtype(cm: Callable[..., Any]) -> Callable[..., Any]:
    """Replace a NeMo get/set/restore CM with :func:`virtual_default_dtype`."""
    if getattr(cm, _WRAPPED_ATTR, False):
        return cm

    @contextlib.contextmanager
    def _virtual(*args: Any, **kwargs: Any) -> Iterator[None]:
        with virtual_default_dtype(_dtype_from_args(*args, **kwargs)):
            yield

    with contextlib.suppress(Exception):
        _virtual = functools.update_wrapper(_virtual, cm)
    setattr(_virtual, _WRAPPED_ATTR, True)
    return _virtual


def _wrap_factory(orig: Callable[..., Any]) -> Callable[..., Any]:
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        virt = _tls_dtype()
        if (
            virt is not None
            and kwargs.get("dtype") is None
            and kwargs.get("out") is None
        ):
            kwargs["dtype"] = virt
        return orig(*args, **kwargs)

    with contextlib.suppress(Exception):
        _wrapped = functools.update_wrapper(_wrapped, orig)
    setattr(_wrapped, _WRAPPED_ATTR, True)
    return _wrapped


def _wrap_torch_factories() -> None:
    for name in _FACTORY_NAMES:
        orig = getattr(torch, name, None)
        if orig is None or getattr(orig, _WRAPPED_ATTR, False):
            continue
        setattr(torch, name, _wrap_factory(orig))


def _is_nemo_module_name(name: str) -> bool:
    return name == "nemo" or name.startswith("nemo.")


def _wrap_attr_if_default_dtype(owner: Any, attr: str, value: Any) -> None:
    if attr != "_default_dtype" or not callable(value):
        return
    if getattr(value, _WRAPPED_ATTR, False):
        return
    try:
        setattr(owner, attr, wrap_nemo_default_dtype(value))
    except Exception:
        return


def _module_float_dtype(module: Any) -> torch.dtype | None:
    params = getattr(module, "parameters", None)
    if not callable(params):
        return None
    try:
        for param in params():
            if getattr(param, "is_floating_point", lambda: False)():
                return param.dtype
    except Exception:
        return None
    return None


def _coerce_streaming_state_dtype(state: Any, dtype: torch.dtype | None) -> Any:
    if state is None or dtype is None or not dtype.is_floating_point:
        return state
    for name in _STATE_FLOAT_ATTRS:
        tensor = getattr(state, name, None)
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.is_floating_point()
            and tensor.dtype != dtype
        ):
            setattr(state, name, tensor.to(dtype))
    return state


def _wrap_init_streaming_state(fn: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(fn, _WRAPPED_ATTR, False):
        return fn

    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        dtype = _module_float_dtype(self)
        state = fn(self, *args, **kwargs)
        return _coerce_streaming_state_dtype(state, dtype)

    with contextlib.suppress(Exception):
        _wrapped = functools.update_wrapper(_wrapped, fn)
    setattr(_wrapped, _WRAPPED_ATTR, True)
    return _wrapped


def _wrap_attr_if_init_streaming_state(owner: Any, value: Any) -> None:
    if not callable(value) or getattr(value, "__name__", "") != "init_streaming_state":
        return
    if getattr(value, _WRAPPED_ATTR, False):
        return
    try:
        owner.init_streaming_state = _wrap_init_streaming_state(value)
    except Exception:
        return


def _wrap_nemo_serving_sites() -> None:
    """No-op NeMo ``_default_dtype``; force streaming-state factory dtypes."""
    for name, mod in list(sys.modules.items()):
        if mod is None or not _is_nemo_module_name(name):
            continue
        mapping = getattr(mod, "__dict__", None)
        if not mapping:
            continue
        _wrap_attr_if_default_dtype(
            mod, "_default_dtype", mapping.get("_default_dtype")
        )
        _wrap_attr_if_init_streaming_state(mod, mapping.get("init_streaming_state"))
        for _, obj in list(mapping.items()):
            if not isinstance(obj, type):
                continue
            cls_dict = getattr(obj, "__dict__", None)
            if not cls_dict:
                continue
            _wrap_attr_if_default_dtype(
                obj, "_default_dtype", cls_dict.get("_default_dtype")
            )
            _wrap_attr_if_init_streaming_state(
                obj, cls_dict.get("init_streaming_state")
            )


def install_default_dtype_guards() -> None:
    """Install factory pins and NeMo no-op ``_default_dtype``.

    Does not wrap ``get_default_dtype`` / ``set_default_dtype``. Dynamo
    snapshots those and a lying ``get`` makes ``convert_frame`` restore
    bf16 into the process.
    """
    _wrap_torch_factories()
    _wrap_nemo_serving_sites()


def _inc_mismatch_metric() -> None:
    global _PROM
    if _PROM is False:
        return
    if _PROM is None:
        try:
            from prometheus_client import Counter

            _PROM = Counter(
                "vllm_serving_default_dtype_not_fp32_total",
                "Process torch default_dtype was not fp32 at a request boundary",
            )
        except ValueError:
            _PROM = False
            return
        except Exception:
            _PROM = False
            return
    _PROM.inc()


def check_serving_default_dtype(where: str = "request") -> None:
    """WARN + metric if the process default is not fp32. Never raises."""
    global _MISMATCH_COUNT
    install_default_dtype_guards()
    current = process_default_dtype()
    if current == SERVING_DEFAULT_DTYPE:
        return
    _MISMATCH_COUNT += 1
    logger.warning(
        "torch default_dtype is %s at %s; serving expects %s. "
        "A serving path called set_default_dtype (count=%d).",
        current,
        where,
        SERVING_DEFAULT_DTYPE,
        _MISMATCH_COUNT,
    )
    _inc_mismatch_metric()

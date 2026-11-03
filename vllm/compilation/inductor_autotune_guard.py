# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Count inductor CachingAutotuner runtime benchmarks after artifact load.

Wraps class methods on ``CachingAutotuner`` (never instance bound methods or
``AutotuneCache.save``, which get pickled into static autotuners). Under
``VLLM_BATCH_INVARIANT`` startup fails if any benchmark ran after warmup
unless ``VLLM_INDUCTOR_AUTOTUNE_ALLOW_BENCH=1``.

This torch build's ``benchmark_all_configs`` closes the static launcher
of every config slower than the running best *during* the bench loop.
The BI pick re-ranks afterwards, so the rule winner may already be
closed (``AssertionError: self.function is not None``). Under BI the
wrapper suppresses ``_close_static_launcher`` for that loop;
``autotune_to_one_config`` then releases losers via
``_release_static_launchers_except``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import vllm.envs as envs
from vllm.compilation.inductor_autotune_pick import (
    maybe_rewrite_timings,
    reset_pick_reports,
    should_pick_deterministic,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_installed = False
_lock = threading.Lock()
_events: list[tuple[str, float]] = []
_in_autotune = threading.local()


def is_installed() -> bool:
    return _installed


def reset() -> None:
    """Drop recorded events. Call at the start of worker warmup."""
    with _lock:
        _events.clear()
    reset_pick_reports()


def recorded_events() -> list[tuple[str, float]]:
    with _lock:
        return list(_events)


def format_summary(events: list[tuple[str, float]] | None = None) -> str:
    if events is None:
        events = recorded_events()
    if not events:
        return "inductor runtime benchmarks: 0"
    parts = [f"{name}={dt * 1000:.1f}ms" for name, dt in events]
    return f"inductor runtime benchmarks: {len(events)} (kernels: {', '.join(parts)})"


def _kernel_name(autotuner: Any) -> str:
    meta = getattr(autotuner, "inductor_meta", None) or {}
    name = meta.get("kernel_name") if isinstance(meta, dict) else None
    if isinstance(name, str) and name:
        return name
    fn = getattr(autotuner, "fn", None)
    return str(getattr(fn, "__name__", type(autotuner).__name__))


def record(autotuner: Any, duration_s: float) -> None:
    with _lock:
        _events.append((_kernel_name(autotuner), float(duration_s)))


def _wrap_method(cls: type, name: str, *, count: bool) -> None:
    original = getattr(cls, name, None)
    if original is None or getattr(original, "_vllm_inductor_bench_guard", False):
        return

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if name == "autotune_to_one_config":
            _in_autotune.active = True
        suppress_close = name == "benchmark_all_configs" and should_pick_deterministic()
        if suppress_close:
            _in_autotune.suppress_close = True
        t0 = time.perf_counter()
        try:
            result = original(self, *args, **kwargs)
            if name == "benchmark_all_configs":
                result = maybe_rewrite_timings(self, result)
            return result
        finally:
            if suppress_close:
                _in_autotune.suppress_close = False
            if count and (
                name != "benchmark_all_configs"
                or not getattr(_in_autotune, "active", False)
            ):
                record(self, time.perf_counter() - t0)
            if name == "autotune_to_one_config":
                _in_autotune.active = False

    wrapped._vllm_inductor_bench_guard = True  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(original, "__name__", name)
    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(cls, name, wrapped)


def _wrap_close_static_launcher(cls: type) -> None:
    """No-op ``_close_static_launcher`` while a BI bench loop is re-ranking."""
    original = getattr(cls, "_close_static_launcher", None)
    if original is None or getattr(original, "_vllm_inductor_bench_guard", False):
        return

    def wrapped(launcher: Any) -> Any:
        if getattr(_in_autotune, "suppress_close", False):
            return None
        return original(launcher)

    wrapped._vllm_inductor_bench_guard = True  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(original, "__name__", "_close_static_launcher")
    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    cls._close_static_launcher = staticmethod(wrapped)  # type: ignore[attr-defined]


def wrap_autotuner_class(cls: type) -> None:
    """Wrap autotune pick / bench / static-launcher close on ``cls``."""
    _wrap_method(cls, "autotune_to_one_config", count=True)
    _wrap_method(cls, "benchmark_all_configs", count=True)
    _wrap_close_static_launcher(cls)


def install() -> None:
    """Install class-level wrappers. Idempotent."""
    global _installed
    if _installed:
        return
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skipping inductor autotune guard (no CachingAutotuner): %s", exc)
        return
    wrap_autotuner_class(CachingAutotuner)
    _installed = True


def should_fail_on_benchmarks() -> bool:
    if envs.VLLM_INDUCTOR_AUTOTUNE_ALLOW_BENCH:
        return False
    return bool(envs.VLLM_BATCH_INVARIANT)


def check_inductor_runtime_benchmarks(
    *,
    fail: bool | None = None,
) -> int:
    """Log a one-line summary and optionally fail if any benchmark ran.

    Args:
        fail: Override the default (BI on and ALLOW_BENCH off).

    Returns:
        Number of recorded runtime benchmarks.

    Raises:
        RuntimeError: When ``fail`` is true and at least one benchmark ran.

    """
    events = recorded_events()
    summary = format_summary(events)
    logger.info("%s", summary)
    should_fail = should_fail_on_benchmarks() if fail is None else fail
    if should_fail and events:
        names = ", ".join(name for name, _ in events)
        raise RuntimeError(
            "Batch-invariant mode forbids inductor runtime autotune after "
            f"artifact load ({len(events)} benchmarks: {names}). Load a "
            "pinned .best_config pack (VLLM_INDUCTOR_AUTOTUNE_PIN_DIR) or "
            "set VLLM_INDUCTOR_AUTOTUNE_ALLOW_BENCH=1 while regenerating. "
            f"{summary}"
        )
    if events and not should_fail:
        logger.error("%s", summary)
    return len(events)

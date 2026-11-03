# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic inductor config pick for batch-invariant pin packs.

``CachingAutotuner.autotune_to_one_config`` normally does
``min(timings)``. Near-tie configs (rms_norm R0_BLOCK 4096 vs 2048)
flip with timing noise, so two cold dumps on the same SKU write
different packs and 18–28% of alone outputs change.

Under ``VLLM_BATCH_INVARIANT`` the pick ignores wall-clock and uses a
fixed order: largest ``R0_BLOCK``, then other ``*_BLOCK`` keys, then
warps / stages, then inductor's original list order. Invalid (inf)
configs stay excluded. The pack is then a cache of that rule, not a
source of numerics. The bench-loop wrapper keeps static launchers
open until ``autotune_to_one_config`` releases the losers (this torch
build otherwise closes every config slower than the running best).

Do not use inductor ``deterministic=true``: that picks the
second-largest RBLOCK and is a different (broken) numeric path.
"""

from __future__ import annotations

import threading
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

PICK_RULE = "largest_rblock"
_IGNORE_CONFIG_KEYS = frozenset({"time_taken_ms", "triton_cache_hash"})
_PLACEHOLDER_KERNEL_NAMES = frozenset(
    {
        "Placeholder.DESCRIPTIVE_NAME",
        "Placeholder.KERNEL_NAME",
        "DESCRIPTIVE_NAME",
        "KERNEL_NAME",
    }
)

_lock = threading.Lock()
_reports: list[dict[str, Any]] = []


def should_pick_deterministic() -> bool:
    """True when BI forbids timing-based inductor config choice."""
    return bool(envs.VLLM_BATCH_INVARIANT)


def reset_pick_reports() -> None:
    with _lock:
        _reports.clear()


def pick_reports() -> list[dict[str, Any]]:
    with _lock:
        return [dict(item) for item in _reports]


def record_pick_report(report: dict[str, Any]) -> None:
    with _lock:
        _reports.append(dict(report))


def _is_placeholder_kernel_name(name: str | None) -> bool:
    if not name or not isinstance(name, str):
        return True
    return name in _PLACEHOLDER_KERNEL_NAMES or name.startswith("Placeholder.")


def kernel_name(autotuner: Any) -> str:
    """Human kernel name; skip inductor ``Placeholder.DESCRIPTIVE_NAME``."""
    meta = getattr(autotuner, "inductor_meta", None) or {}
    meta_name = meta.get("kernel_name") if isinstance(meta, dict) else None
    fn = getattr(autotuner, "fn", None)
    fn_name = getattr(fn, "__name__", None)
    if isinstance(meta_name, str) and not _is_placeholder_kernel_name(meta_name):
        return meta_name
    if (
        isinstance(fn_name, str)
        and fn_name
        and not _is_placeholder_kernel_name(fn_name)
    ):
        return fn_name
    if isinstance(meta_name, str) and meta_name:
        return meta_name
    return str(fn_name or type(autotuner).__name__)


def _config_kwargs(config: Any) -> dict[str, Any]:
    kwargs = getattr(config, "kwargs", None)
    return dict(kwargs) if isinstance(kwargs, dict) else {}


def config_sort_key(config: Any, index: int = 0) -> tuple[Any, ...]:
    """Smaller is better: largest R0_BLOCK first, then other blocks."""
    kwargs = _config_kwargs(config)
    r0 = kwargs.get("R0_BLOCK")
    blocks = tuple(
        (name, -int(value))
        for name, value in sorted(kwargs.items())
        if name.endswith("_BLOCK") and isinstance(value, (int, float))
    )
    return (
        0 if r0 is not None else 1,
        -int(r0) if isinstance(r0, (int, float)) else 0,
        blocks,
        -int(getattr(config, "num_warps", 0) or 0),
        -int(getattr(config, "num_stages", 0) or 0),
        index,
    )


def config_fields(config: Any) -> dict[str, Any]:
    """JSON-stable view of a Triton / inductor config."""
    fields: dict[str, Any] = dict(_config_kwargs(config))
    for name in ("num_warps", "num_stages"):
        value = getattr(config, name, None)
        if value is not None:
            fields[name] = value
    return fields


def pick_deterministic_config(
    configs: list[Any],
) -> Any | None:
    """Return the rule winner among ``configs`` (largest R0_BLOCK)."""
    if not configs:
        return None
    ranked = sorted(
        enumerate(configs),
        key=lambda item: config_sort_key(item[1], item[0]),
    )
    return ranked[0][1]


def _finite_timings(timings: dict[Any, Any]) -> dict[Any, float]:
    finite: dict[Any, float] = {}
    for launcher, timing in timings.items():
        if isinstance(timing, (int, float)) and timing != float("inf"):
            finite[launcher] = float(timing)
    return finite


def _launcher_config(launcher: Any) -> Any:
    return getattr(launcher, "config", launcher)


def maybe_rewrite_timings(
    autotuner: Any,
    timings: Any,
) -> Any:
    """Replace bench times with rule ranks so ``min(timings)`` is stable.

    Real times are kept on the pick report so a dump can measure the
    rule against the timed winner. No-op when BI is off or ``timings``
    is not a launcher→ms map.
    """
    if not should_pick_deterministic() or not isinstance(timings, dict):
        return timings
    finite = _finite_timings(timings)
    if len(finite) < 2:
        return timings
    launchers = list(finite)
    ranked = sorted(
        enumerate(launchers),
        key=lambda item: config_sort_key(_launcher_config(item[1]), item[0]),
    )
    timed_launcher = min(finite, key=lambda launcher: finite[launcher])
    rule_launcher = ranked[0][1]
    rewritten = dict(timings)
    for rank, (_, launcher) in enumerate(ranked, start=1):
        rewritten[launcher] = float(rank)
    timed_ms = finite[timed_launcher]
    rule_ms = finite[rule_launcher]
    slower_pct = None
    if timed_ms > 0:
        slower_pct = (rule_ms - timed_ms) / timed_ms * 100.0
    kernel = kernel_name(autotuner)
    report = {
        "kernel": kernel,
        "pick_rule": PICK_RULE,
        "picked": config_fields(_launcher_config(rule_launcher)),
        "timed": config_fields(_launcher_config(timed_launcher)),
        "picked_ms": rule_ms,
        "timed_ms": timed_ms,
        "rule_slower_pct": slower_pct,
        "differed": rule_launcher is not timed_launcher,
    }
    record_pick_report(report)
    if report["differed"]:
        logger.info(
            "Inductor BI pick_rule=%s kernel=%s picked=%s (%.4f ms) "
            "timed=%s (%.4f ms, %+.2f%%)",
            PICK_RULE,
            kernel,
            report["picked"],
            rule_ms,
            report["timed"],
            timed_ms,
            slower_pct if slower_pct is not None else 0.0,
        )
    return rewritten


def normalize_best_config(payload: Any) -> Any:
    """Drop dump-noise fields so two packs of the same pick compare equal."""
    if not isinstance(payload, dict):
        return payload
    return {k: v for k, v in payload.items() if k not in _IGNORE_CONFIG_KEYS}

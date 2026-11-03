# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for inductor autotune pin packs and the warmup bench guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm.compilation.inductor_autotune_guard import (
    check_inductor_runtime_benchmarks,
    format_summary,
    recorded_events,
    reset,
    wrap_autotuner_class,
)
from vllm.compilation.inductor_autotune_pick import (
    PICK_RULE,
    config_sort_key,
    maybe_rewrite_timings,
    normalize_best_config,
    pick_deterministic_config,
    pick_reports,
    reset_pick_reports,
    should_pick_deterministic,
)
from vllm.compilation.inductor_autotune_pin import (
    compare_pin_packs,
    current_sku_key,
    dump_pinned_autotune_cache,
    format_pack_diff,
    install_pinned_autotune_cache,
    packs_match,
    pin_dump_dir,
    resolve_pin_pack_dir,
)
from vllm.compilation.inductor_autotune_pin import main as pin_main


@pytest.mark.parametrize(
    ("name", "cap", "expected"),
    [
        ("NVIDIA A100-SXM4-80GB", (8, 0), "a100-sm80"),
        ("NVIDIA L40S", (8, 9), "l40s-sm89"),
        ("NVIDIA L40", (8, 9), "l40-sm89"),
        ("NVIDIA H100 80GB HBM3", (9, 0), "h100-sm90"),
    ],
)
def test_sku_key_aliases(name: str, cap: tuple[int, int], expected: str):
    assert current_sku_key(gpu_name=name, compute_capability=cap) == expected


def test_dump_install_roundtrip(tmp_path: Path):
    cache = tmp_path / "inductor_cache"
    nested = cache / "ab" / "cd"
    nested.mkdir(parents=True)
    payload = '{"R0_BLOCK": 4096, "num_warps": 8, "configs_hash": "abc"}\n'
    (nested / "deadbeef.best_config").write_text(payload)
    (cache / "not_a_pin.json").write_text("{}")

    pack = tmp_path / "pack"
    manifest = dump_pinned_autotune_cache(
        cache, pack, sku_key="a100-sm80", gpu_name="NVIDIA A100-SXM4-80GB"
    )
    assert manifest["file_count"] == 1
    assert manifest["files"] == ["ab/cd/deadbeef.best_config"]
    assert (pack / "ab" / "cd" / "deadbeef.best_config").read_text() == payload
    assert (pack / "manifest.json").is_file()

    dest = tmp_path / "loaded"
    n = install_pinned_autotune_cache(dest, pack, sku_key="a100-sm80")
    assert n == 1
    assert (dest / "ab" / "cd" / "deadbeef.best_config").read_text() == payload


def test_resolve_pin_pack_dir_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pack = tmp_path / "custom"
    (pack / "nested").mkdir(parents=True)
    (pack / "nested" / "k.best_config").write_text("{}")
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_pin.envs.VLLM_INDUCTOR_AUTOTUNE_PIN_DIR",
        str(pack),
    )
    assert resolve_pin_pack_dir() == pack


def test_pin_dump_dir_parses_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_pin.envs.VLLM_INDUCTOR_AUTOTUNE_PIN_DUMP",
        None,
    )
    assert pin_dump_dir() is None

    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_pin.envs.VLLM_INDUCTOR_AUTOTUNE_PIN_DUMP",
        "1",
    )
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_pin.envs.VLLM_INDUCTOR_AUTOTUNE_PIN_DIR",
        None,
    )
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_pin.current_sku_key",
        lambda: "a100-sm80",
    )
    dump = pin_dump_dir()
    assert dump is not None
    assert dump.name == "a100-sm80"

    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_pin.envs.VLLM_INDUCTOR_AUTOTUNE_PIN_DUMP",
        str(tmp_path / "out"),
    )
    assert pin_dump_dir() == tmp_path / "out"


def test_cli_sku(capsys: pytest.CaptureFixture[str]):
    assert pin_main(["sku"]) == 0
    out = capsys.readouterr().out.strip()
    assert out


def test_cli_dump_install(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "k.best_config").write_text('{"num_warps": 4}\n')
    pack = tmp_path / "pack"
    dest = tmp_path / "dest"
    assert pin_main(["dump", "--cache-dir", str(cache), "--output", str(pack)]) == 0
    assert pin_main(["install", "--source", str(pack), "--cache-dir", str(dest)]) == 0
    assert (dest / "k.best_config").is_file()


class _Cfg:
    def __init__(
        self,
        r0: int | None,
        num_warps: int = 4,
        num_stages: int = 1,
        extra: dict | None = None,
    ):
        self.kwargs = dict(extra or {})
        if r0 is not None:
            self.kwargs["R0_BLOCK"] = r0
        self.num_warps = num_warps
        self.num_stages = num_stages


class _Launcher:
    def __init__(self, config: _Cfg):
        self.config = config


def test_largest_r0_block_wins_over_faster_near_tie():
    slow_big = _Cfg(4096)
    fast_small = _Cfg(2048)
    assert pick_deterministic_config([fast_small, slow_big]) is slow_big
    assert config_sort_key(slow_big, 1) < config_sort_key(fast_small, 0)


def test_rewrite_timings_picks_rule_not_clock(
    monkeypatch: pytest.MonkeyPatch,
):
    reset_pick_reports()
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_pick.envs.VLLM_BATCH_INVARIANT",
        True,
    )
    assert should_pick_deterministic()
    big = _Launcher(_Cfg(4096))
    small = _Launcher(_Cfg(2048))
    timings = {small: 0.10, big: 0.101}
    rewritten = maybe_rewrite_timings(
        type("T", (), {"inductor_meta": {"kernel_name": "rms_norm"}})(),
        timings,
    )
    assert min(rewritten, key=rewritten.get) is big
    reports = pick_reports()
    assert len(reports) == 1
    assert reports[0]["pick_rule"] == PICK_RULE
    assert reports[0]["differed"] is True
    assert reports[0]["picked"]["R0_BLOCK"] == 4096
    assert reports[0]["timed"]["R0_BLOCK"] == 2048
    assert reports[0]["rule_slower_pct"] == pytest.approx(1.0)


def test_rewrite_timings_skipped_without_bi(monkeypatch: pytest.MonkeyPatch):
    reset_pick_reports()
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_pick.envs.VLLM_BATCH_INVARIANT",
        False,
    )
    big = _Launcher(_Cfg(4096))
    small = _Launcher(_Cfg(2048))
    timings = {small: 0.10, big: 0.20}
    rewritten = maybe_rewrite_timings(object(), timings)
    assert rewritten is timings
    assert pick_reports() == []


def test_guard_rewrites_autotune_pick_under_bi(monkeypatch: pytest.MonkeyPatch):
    reset()
    reset_pick_reports()
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_pick.envs.VLLM_BATCH_INVARIANT",
        True,
    )

    class _Tune:
        inductor_meta = {"kernel_name": "fused_add_rms_norm"}

        def autotune_to_one_config(self):
            timings = self.benchmark_all_configs()
            return min(timings, key=timings.get)

        def benchmark_all_configs(self):
            return {
                _Launcher(_Cfg(2048)): 0.08,
                _Launcher(_Cfg(4096)): 0.081,
            }

    wrap_autotuner_class(_Tune)
    winner = _Tune().autotune_to_one_config()
    assert winner.config.kwargs["R0_BLOCK"] == 4096


def test_verify_ignores_time_taken_ms(tmp_path: Path):
    pack = tmp_path / "pack"
    ref = tmp_path / "ref"
    (pack / "ab").mkdir(parents=True)
    (ref / "ab").mkdir(parents=True)
    (pack / "ab" / "k.best_config").write_text(
        '{"R0_BLOCK": 4096, "num_warps": 8, "time_taken_ms": 12, '
        '"triton_cache_hash": "aaa"}\n'
    )
    (ref / "ab" / "k.best_config").write_text(
        '{"R0_BLOCK": 4096, "num_warps": 8, "time_taken_ms": 99, '
        '"triton_cache_hash": "bbb"}\n'
    )
    report = compare_pin_packs(pack, ref)
    assert packs_match(report)
    assert report["identical"] == ["ab/k.best_config"]


def test_verify_prints_differing_kernels(tmp_path: Path, capsys):
    pack = tmp_path / "pack"
    ref = tmp_path / "ref"
    pack.mkdir()
    ref.mkdir()
    (pack / "k.best_config").write_text('{"R0_BLOCK": 4096, "num_warps": 8}\n')
    (ref / "k.best_config").write_text('{"R0_BLOCK": 2048, "num_warps": 8}\n')
    assert pin_main(["verify", "--pack", str(pack), "--ref", str(ref)]) == 1
    out = capsys.readouterr().out
    assert "differing: 1" in out
    assert "differ k.best_config" in out
    assert "4096" in out
    assert "2048" in out
    report = compare_pin_packs(pack, ref)
    text = format_pack_diff(report)
    assert "differ k.best_config" in text


def test_normalize_drops_dump_noise():
    assert normalize_best_config(
        {"R0_BLOCK": 4096, "time_taken_ms": 1, "triton_cache_hash": "x"}
    ) == {"R0_BLOCK": 4096}


def test_dump_records_pick_rule(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "k.best_config").write_text('{"R0_BLOCK": 4096}\n')
    pack = tmp_path / "pack"
    manifest = dump_pinned_autotune_cache(cache, pack, sku_key="a100-sm80")
    assert manifest["pick_rule"] == PICK_RULE


class _FakeAutotuner:
    def __init__(self, name: str):
        self.inductor_meta = {"kernel_name": name}

    def autotune_to_one_config(self, *args, **kwargs):
        return self.benchmark_all_configs(*args, **kwargs)

    def benchmark_all_configs(self, *args, **kwargs):
        return {"ok": 1}


def test_guard_counts_autotune_once_and_fails_under_bi(
    monkeypatch: pytest.MonkeyPatch,
):
    wrap_autotuner_class(_FakeAutotuner)
    reset()
    tuner = _FakeAutotuner("triton_red_fused_add_rms_norm")
    assert tuner.autotune_to_one_config() == {"ok": 1}
    events = recorded_events()
    assert len(events) == 1
    assert events[0][0] == "triton_red_fused_add_rms_norm"
    assert "inductor runtime benchmarks: 1" in format_summary(events)

    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_guard.envs.VLLM_BATCH_INVARIANT",
        True,
    )
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_guard.envs.VLLM_INDUCTOR_AUTOTUNE_ALLOW_BENCH",
        False,
    )
    with pytest.raises(RuntimeError, match="forbids inductor runtime autotune"):
        check_inductor_runtime_benchmarks()


def test_guard_allow_bench_logs_without_fail(monkeypatch: pytest.MonkeyPatch):
    wrap_autotuner_class(_FakeAutotuner)
    reset()
    _FakeAutotuner("k2").benchmark_all_configs()
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_guard.envs.VLLM_BATCH_INVARIANT",
        True,
    )
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_guard.envs.VLLM_INDUCTOR_AUTOTUNE_ALLOW_BENCH",
        True,
    )
    assert check_inductor_runtime_benchmarks() == 1


def test_guard_zero_benchmarks_ok_under_bi(monkeypatch: pytest.MonkeyPatch):
    reset()
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_guard.envs.VLLM_BATCH_INVARIANT",
        True,
    )
    monkeypatch.setattr(
        "vllm.compilation.inductor_autotune_guard.envs.VLLM_INDUCTOR_AUTOTUNE_ALLOW_BENCH",
        False,
    )
    assert check_inductor_runtime_benchmarks() == 0
    assert format_summary() == "inductor runtime benchmarks: 0"


def test_wrap_does_not_replace_save_hook():
    """Regression: AutotuneCache.save must stay a real bound method."""
    pytest.importorskip("torch._inductor.runtime.autotune_cache")
    from torch._inductor.runtime.autotune_cache import AutotuneCache

    assert getattr(AutotuneCache.save, "_vllm_inductor_bench_guard", False) is False

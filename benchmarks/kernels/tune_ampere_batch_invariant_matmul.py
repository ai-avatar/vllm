#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep M-independent persistent-matmul tiles for A100 (sm80).

Batch invariance: a candidate may depend only on (N, K, dtype, device). The
same BLOCK_M/N/K, num_warps, and num_stages are used for every M. Each
survivor must produce bitwise-identical rows for M in {1..512} (configurable).

Usage (A100, from a vLLM checkout of this branch):

    source .venv/bin/activate
    export CUDA_VISIBLE_DEVICES=0
    .venv/bin/python benchmarks/kernels/tune_ampere_batch_invariant_matmul.py \
        --output ampere_bi_matmul_best.py \
        --json ampere_bi_matmul_best.json

    # Faster sanity pass (M in {1,2,4,...,512}):
    .venv/bin/python benchmarks/kernels/tune_ampere_batch_invariant_matmul.py \
        --quick --output /tmp/ampere_bi_quick.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from vllm.model_executor.determinism.batch_invariant import (
    matmul_kernel_persistent,
)
from vllm.model_executor.determinism.batch_invariant_configs import (
    _BATCH_INVARIANT_MATMUL_STATIC_CONFIGS,
    NEMOTRON_TRANSCRIBE_BF16_GEMM_SHAPES,
    _MatmulStaticConfig,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units

BLOCK_M_CANDIDATES = (16, 32, 64, 128)
BLOCK_N_CANDIDATES = (32, 64, 128, 256)
BLOCK_K_CANDIDATES = (32, 64, 128)
NUM_WARPS_CANDIDATES = (4, 8)
NUM_STAGES_CANDIDATES = (2, 3, 4)

# Conservative Ampere smem budget (A100 is 166912B). Extra Triton metadata
# sits on top of the A/B pipeline buffers.
A100_SMEM_BUDGET = 160000
GROUP_SIZE_M = 8
DEFAULT_TIMING_WEIGHTS = (0.40, 0.15, 0.15, 0.15, 0.15)
UNTUNED_DEFAULT = _MatmulStaticConfig(128, 128, 64, 8, 3)


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    n: int
    k: int


@dataclass
class CandidateResult:
    config: _MatmulStaticConfig
    bi_ok: bool
    bi_fail_m: int | None
    weighted_ms: float | None
    per_m_ms: dict[int, float]
    error: str | None = None


def _unique_nemotron_shapes() -> list[ShapeSpec]:
    seen: set[tuple[int, int]] = set()
    out: list[ShapeSpec] = []
    for name, n, k in NEMOTRON_TRANSCRIBE_BF16_GEMM_SHAPES:
        key = (n, k)
        if key in seen:
            continue
        seen.add(key)
        out.append(ShapeSpec(name, n, k))
    return out


def _parse_shapes(values: Sequence[str] | None) -> list[ShapeSpec]:
    if not values:
        return _unique_nemotron_shapes()
    out: list[ShapeSpec] = []
    for raw in values:
        if ":" in raw:
            name, nk = raw.split(":", 1)
        else:
            name, nk = "custom", raw
        n_s, k_s = nk.split(",")
        out.append(ShapeSpec(name.strip(), int(n_s), int(k_s)))
    return out


def _cfg_dict(cfg: _MatmulStaticConfig) -> dict[str, int]:
    return {
        "block_m": cfg.block_m,
        "block_n": cfg.block_n,
        "block_k": cfg.block_k,
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }


def estimate_smem_bytes(cfg: _MatmulStaticConfig, elem_size: int = 2) -> int:
    tiles = cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n
    return cfg.num_stages * elem_size * tiles


def iter_candidates(n: int, k: int) -> Iterable[_MatmulStaticConfig]:
    for block_m in BLOCK_M_CANDIDATES:
        for block_n in BLOCK_N_CANDIDATES:
            if block_n > max(n, 32) and block_n > n * 2:
                continue
            for block_k in BLOCK_K_CANDIDATES:
                if block_k > k:
                    continue
                for num_warps in NUM_WARPS_CANDIDATES:
                    for num_stages in NUM_STAGES_CANDIDATES:
                        cfg = _MatmulStaticConfig(
                            block_m, block_n, block_k, num_warps, num_stages
                        )
                        if estimate_smem_bytes(cfg) > A100_SMEM_BUDGET:
                            continue
                        yield cfg


def _launch(
    a: torch.Tensor,
    b: torch.Tensor,
    cfg: _MatmulStaticConfig,
    num_sms: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    c = out if out is not None else torch.empty((m, n), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (
            min(
                num_sms,
                triton.cdiv(m, meta["BLOCK_SIZE_M"])
                * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
            ),
        )

    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        None,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        1,
        NUM_SMS=num_sms,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=False,
        BLOCK_SIZE_M=cfg.block_m,
        BLOCK_SIZE_N=cfg.block_n,
        BLOCK_SIZE_K=cfg.block_k,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return c


def check_row_batch_invariance(
    a_full: torch.Tensor,
    b: torch.Tensor,
    cfg: _MatmulStaticConfig,
    m_values: Sequence[int],
    num_sms: int,
) -> tuple[bool, int | None, str | None]:
    """Row i of every M in m_values must match row i of max(m_values)."""
    max_m = m_values[-1]
    try:
        ref = _launch(a_full[:max_m], b, cfg, num_sms)
        torch.accelerator.synchronize()
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    for m in m_values:
        try:
            out = _launch(a_full[:m], b, cfg, num_sms)
            torch.accelerator.synchronize()
        except Exception as exc:
            return False, m, f"{type(exc).__name__}: {exc}"
        if not torch.equal(out, ref[:m]):
            return False, m, "row mismatch vs max-M reference"
    return True, None, None


def _bench_ms(fn, *, warmup: int, iters: int, use_graph: bool) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.accelerator.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.accelerator.synchronize()
        return start.elapsed_time(end) / iters
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters


def time_candidate(
    a_full: torch.Tensor,
    b: torch.Tensor,
    cfg: _MatmulStaticConfig,
    timing_ms: Sequence[int],
    num_sms: int,
    warmup: int,
    iters: int,
    use_graph: bool,
) -> dict[int, float]:
    per_m: dict[int, float] = {}
    n = b.shape[1]
    for m in timing_ms:
        a = a_full[:m]
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        # Force a compile of this (config, runtime) pair before timing.
        _launch(a, b, cfg, num_sms, out=out)
        torch.accelerator.synchronize()

        def run(a=a, out=out):
            _launch(a, b, cfg, num_sms, out=out)

        per_m[m] = _bench_ms(run, warmup=warmup, iters=iters, use_graph=use_graph)
    return per_m


def weighted_latency_ms(
    per_m_ms: dict[int, float],
    timing_ms: Sequence[int],
    weights: Sequence[float],
) -> float:
    total_w = 0.0
    acc = 0.0
    for m, w in zip(timing_ms, weights):
        if m in per_m_ms:
            acc += w * per_m_ms[m]
            total_w += w
    return acc / total_w if total_w else float("inf")


def format_python_table(
    winners: dict[tuple[int, int], _MatmulStaticConfig],
    names: dict[tuple[int, int], str],
) -> str:
    lines = [
        "# Generated by benchmarks/kernels/tune_ampere_batch_invariant_matmul.py",
        "# Paste into _BATCH_INVARIANT_MATMUL_STATIC_CONFIGS['ampere'].",
        '"ampere": {',
    ]
    for (n, k), cfg in sorted(winners.items()):
        name = names.get((n, k), "")
        comment = f"  # {name}" if name else ""
        lines.append(
            f"    ({n}, {k}): _MatmulStaticConfig("
            f"{cfg.block_m}, {cfg.block_n}, {cfg.block_k}, "
            f"{cfg.num_warps}, {cfg.num_stages}),{comment}"
        )
    lines.append("},")
    return "\n".join(lines) + "\n"


def _seed_for_shape(n: int, k: int) -> _MatmulStaticConfig:
    table = _BATCH_INVARIANT_MATMUL_STATIC_CONFIGS.get("ampere", {})
    return table.get((n, k), UNTUNED_DEFAULT)


def sweep_shape(
    spec: ShapeSpec,
    *,
    m_values: Sequence[int],
    timing_ms: Sequence[int],
    timing_weights: Sequence[float],
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    use_graph: bool,
    skip_timing: bool,
) -> tuple[_MatmulStaticConfig, list[CandidateResult]]:
    max_m = max(max(m_values), max(timing_ms))
    torch.manual_seed(0)
    a_full = torch.randn(max_m, spec.k, device=device, dtype=dtype)
    b = torch.randn(spec.k, spec.n, device=device, dtype=dtype)
    num_sms = num_compute_units(device.index)

    candidates = list(iter_candidates(spec.n, spec.k))
    seed = _seed_for_shape(spec.n, spec.k)
    if seed not in candidates:
        candidates.insert(0, seed)
    elif candidates[0] != seed:
        candidates.remove(seed)
        candidates.insert(0, seed)

    print(
        f"\n=== {spec.name}  (N={spec.n}, K={spec.k})  "
        f"{len(candidates)} candidates ===",
        flush=True,
    )
    results: list[CandidateResult] = []
    best: CandidateResult | None = None
    t0 = time.time()
    for i, cfg in enumerate(candidates):
        bi_ok, fail_m, err = check_row_batch_invariance(
            a_full, b, cfg, m_values, num_sms
        )
        per_m: dict[int, float] = {}
        weighted = None
        if bi_ok and not skip_timing:
            try:
                per_m = time_candidate(
                    a_full,
                    b,
                    cfg,
                    timing_ms,
                    num_sms,
                    warmup=warmup,
                    iters=iters,
                    use_graph=use_graph,
                )
                weighted = weighted_latency_ms(per_m, timing_ms, timing_weights)
            except Exception as exc:
                bi_ok = False
                err = f"timing {type(exc).__name__}: {exc}"
        rec = CandidateResult(
            config=cfg,
            bi_ok=bi_ok,
            bi_fail_m=fail_m,
            weighted_ms=weighted,
            per_m_ms=per_m,
            error=err,
        )
        results.append(rec)
        tag = "OK" if bi_ok else f"FAIL m={fail_m} {err}"
        wt = f"{weighted:.3f}ms" if weighted is not None else "-"
        print(
            f"  [{i + 1:3d}/{len(candidates)}] "
            f"M{cfg.block_m}/N{cfg.block_n}/K{cfg.block_k} "
            f"w{cfg.num_warps}/s{cfg.num_stages}  {tag}  {wt}",
            flush=True,
        )
        if bi_ok and weighted is not None:
            if best is None or (
                best.weighted_ms is not None and weighted < best.weighted_ms
            ):
                best = rec
        elif bi_ok and skip_timing and best is None:
            best = rec

    elapsed = time.time() - t0
    if best is None:
        raise RuntimeError(
            f"no BI-valid config for {spec.name} (N={spec.n}, K={spec.k})"
        )
    print(
        f"  winner {best.config}  weighted={best.weighted_ms}  in {elapsed:.1f}s",
        flush=True,
    )
    return best.config, results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--shape",
        action="append",
        default=None,
        help="Repeatable N,K or name:N,K. Default: Nemotron 3.5 Transcribe TP=1.",
    )
    p.add_argument("--output", default="ampere_bi_matmul_best.py")
    p.add_argument("--json", dest="json_path", default="ampere_bi_matmul_best.json")
    p.add_argument("--max-m", type=int, default=512)
    p.add_argument("--min-m", type=int, default=1)
    p.add_argument(
        "--quick",
        action="store_true",
        help="BI-check M in powers of two up to --max-m instead of every M.",
    )
    p.add_argument(
        "--timing-m",
        default="1,8,64,256,512",
        help="Comma-separated M values used for the latency score.",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Time with CUDA events only (skip CUDA graph replay).",
    )
    p.add_argument(
        "--bi-only",
        action="store_true",
        help="Skip timing; write the first BI-valid config per shape.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required to sweep Ampere tiles.", file=sys.stderr)
        return 2
    capability = current_platform.get_device_capability()
    print(
        f"device={torch.cuda.get_device_name(0)} capability={capability}",
        flush=True,
    )
    if capability is None or (capability.major, capability.minor) != (8, 0):
        print(
            "warning: this sweep targets A100 (sm80); results on other "
            "GPUs should not be pasted into the Ampere table",
            file=sys.stderr,
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    if args.quick:
        powers = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, args.max_m)
        m_values = tuple(dict.fromkeys(m for m in powers if m <= args.max_m))
        if args.min_m not in m_values:
            m_values = (args.min_m, *m_values)
    else:
        m_values = tuple(range(args.min_m, args.max_m + 1))
    timing_ms = tuple(int(x) for x in args.timing_m.split(",") if x)
    weights = DEFAULT_TIMING_WEIGHTS
    if len(timing_ms) != len(weights):
        weights = tuple([1.0 / len(timing_ms)] * len(timing_ms))

    shapes = _parse_shapes(args.shape)
    winners: dict[tuple[int, int], _MatmulStaticConfig] = {}
    names: dict[tuple[int, int], str] = {}
    json_rows: list[dict] = []

    for spec in shapes:
        names[(spec.n, spec.k)] = spec.name
        winner, results = sweep_shape(
            spec,
            m_values=m_values,
            timing_ms=timing_ms,
            timing_weights=weights,
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            use_graph=not args.no_cuda_graph,
            skip_timing=args.bi_only,
        )
        winners[(spec.n, spec.k)] = winner
        json_rows.append(
            {
                "name": spec.name,
                "N": spec.n,
                "K": spec.k,
                "winner": _cfg_dict(winner),
                "seed": _cfg_dict(_seed_for_shape(spec.n, spec.k)),
                "n_candidates": len(results),
                "n_bi_ok": sum(1 for r in results if r.bi_ok),
                "results": [
                    {
                        "config": _cfg_dict(r.config),
                        "bi_ok": r.bi_ok,
                        "bi_fail_m": r.bi_fail_m,
                        "weighted_ms": r.weighted_ms,
                        "per_m_ms": {str(k): v for k, v in r.per_m_ms.items()},
                        "error": r.error,
                    }
                    for r in results
                ],
            }
        )

    table = format_python_table(winners, names)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(table)
    with open(args.json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": None if capability is None else list(capability),
                "m_values": list(m_values),
                "timing_m": list(timing_ms),
                "shapes": json_rows,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.output} and {args.json_path}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

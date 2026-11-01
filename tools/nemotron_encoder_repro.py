# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compiled vs eager Nemotron Transcribe encoder outputs.

Run on a GPU pod with the real checkpoint to reproduce P3d failures
(aligner AutogradCUDA recompile / dense OOM, stale BlockMask valid
length, 7.9 s hallucination, 121 s truncation).

Examples:
  # Flex-wrapper only (no checkpoint):
  python tools/nemotron_encoder_repro.py --flex-only \\
      --seconds 7.9,11,121 --frames 99,256,550,1100,1513 --aligner --burst

  # Real model (vLLM plugin path):
  VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=0 \\
  python tools/nemotron_encoder_repro.py \\
      --model /path/to/nemotron-3.5-transcribe-bf16 \\
      --seconds 7.9,11,121 --aligner --burst

"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(float(part)) for part in text.split(",") if part.strip()]


def _seconds_to_frames(seconds: float, hop_s: float) -> int:
    return max(int(round(seconds / hop_s)), 1)


def _rel_err(left, right) -> float:
    scale = right.detach().float().abs().amax().clamp_min(1e-3)
    return float((left.float() - right.float()).abs().amax() / scale)


def _max_abs(left, right) -> float:
    return float((left.float() - right.float()).abs().amax())


def _unique_graphs() -> int:
    try:
        from torch._dynamo.utils import counters

        return int(counters["stats"]["unique_graphs"])
    except Exception:
        return -1


def _setup_flex(window: int = 550) -> Any:
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    import vllm.envs as envs
    from vllm.compilation import mm_encoder as mm_encoder_mod
    from vllm.compilation.mm_encoder import (
        configure_speech_encoder_dynamo,
        isolate_flex_attention_from_compile,
    )

    envs.disable_envs_cache()
    mm_encoder_mod._INDUCTOR_PINNED = False
    isolate_flex_attention_from_compile()
    configure_speech_encoder_dynamo()
    mm_encoder_mod.set_speech_encoder_shape_config(
        window=window, max_windows=2, bucket_step=256
    )
    mm_encoder_mod._MASK_STATE.clear()
    return mm_encoder_mod


def _run_flex_only(args: argparse.Namespace) -> int:
    import torch
    from torch.nn.attention.flex_attention import flex_attention as raw_flex

    from vllm.compilation.mm_encoder import _pad_flex_wrapper

    mm_encoder_mod = _setup_flex(args.window)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hop = args.hop_s
    frames = list(args.frames)
    for sec in args.seconds:
        frames.append(_seconds_to_frames(sec, hop))
    frames = sorted(set(int(f) for f in frames if f > 0))
    print(f"device={device} frames={frames} hop_s={hop}")

    force: list[bool] = []

    def eager_flex(q, k, v, **kwargs):
        opts = kwargs.get("kernel_options") or {}
        force.append(bool(opts.get("FORCE_USE_FLEX_ATTENTION")))
        return raw_flex(q, k, v, **kwargs)

    compiled = torch.compile(eager_flex, dynamic=False)
    mm_encoder_mod._EAGER_FLEX = eager_flex
    wrapped = _pad_flex_wrapper(compiled)

    def one(batch: int, t: int, *, inference: bool):
        q = torch.randn(batch, 16, t, 80, device=device, dtype=dtype)
        ctx = torch.inference_mode() if inference else torch.nullcontext()
        with ctx:
            mm_encoder_mod._begin_encoder_mask(t)
            return wrapped(q, q, q)

    print("warmup compiled buckets under inference_mode")
    torch._dynamo.reset()
    with torch.inference_mode():
        for t in (256, args.window, 2 * args.window):
            one(1, t, inference=True)
    after_warmup = _unique_graphs()
    print(f"  unique_graphs after warmup={after_warmup}")

    failed = 0
    if args.aligner:
        print("aligner path: B=2, no inference_mode, no requires_grad")
        force.clear()
        mm_encoder_mod._set_serving_watch(True)
        for t in (args.window, 600, 2 * args.window):
            try:
                out = one(2, t, inference=False)
                print(
                    f"  T={t} out={tuple(out.shape)} eager_force={force[-1:]} "
                    f"graphs={_unique_graphs()}"
                )
            except Exception as exc:
                failed += 1
                print(f"  T={t} FAIL: {exc}")
                traceback.print_exc()
        if _unique_graphs() != after_warmup and after_warmup >= 0:
            failed += 1
            print(
                f"  FAIL recompile: unique_graphs {after_warmup} -> {_unique_graphs()}"
            )
        mm_encoder_mod._set_serving_watch(False)

    print("compiled vs eager (B=1, inference_mode)")
    for t in frames:
        torch.manual_seed(t)
        q = torch.randn(1, 16, t, 80, device=device, dtype=dtype)
        with torch.inference_mode():
            mm_encoder_mod._begin_encoder_mask(t)
            compiled_out = wrapped(q, q, q)
            mm_encoder_mod._begin_encoder_mask(t)
            eager_out = eager_flex(
                q,
                q,
                q,
                kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
            )
        # Eager raw_flex is unpadded; compare the valid prefix.
        t_cmp = min(compiled_out.shape[-2], eager_out.shape[-2], t)
        rel = _rel_err(compiled_out[..., :t_cmp, :], eager_out[..., :t_cmp, :])
        mad = _max_abs(compiled_out[..., :t_cmp, :], eager_out[..., :t_cmp, :])
        ok = rel < 0.08
        print(f"  T={t} max|d|={mad:.4g} rel={rel:.4g} {'OK' if ok else 'FAIL'}")
        if not ok:
            failed += 1

    if args.burst:
        print("burst stability")
        probes = {}
        with torch.inference_mode():
            for t in frames:
                torch.manual_seed(1000 + t)
                q = torch.randn(1, 16, t, 80, device=device, dtype=dtype)
                mm_encoder_mod._begin_encoder_mask(t)
                probes[t] = (q, wrapped(q, q, q))
            for t in (1, 50, 99, 129, 400, 549, 551, 1513, 1906):
                other = torch.randn(1, 16, t, 80, device=device, dtype=dtype)
                mm_encoder_mod._begin_encoder_mask(t)
                wrapped(other, other, other)
            for t, (q, first) in probes.items():
                mm_encoder_mod._begin_encoder_mask(t)
                second = wrapped(q, q, q)
                match = bool(torch.equal(first, second))
                print(f"  T={t} bitwise={'OK' if match else 'FAIL'}")
                if not match:
                    failed += 1

    print(f"flex-only done failed={failed}")
    return 1 if failed else 0


def _load_vllm_encoder(model: str, compile_mm: bool):
    from vllm import LLM
    from vllm.compilation.mm_encoder import (
        isolate_flex_attention_from_compile,
        maybe_compile_speech_mm_encoder,
    )

    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    isolate_flex_attention_from_compile()
    llm = LLM(
        model=model,
        dtype="bfloat16",
        compilation_config={
            "compile_mm_encoder": compile_mm,
            "cudagraph_mm_encoder": compile_mm,
        },
        enable_prefix_caching=False,
    )
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    model_runner = None
    for name in ("model_executor", "engine_core"):
        obj = getattr(engine, name, None)
        if obj is None:
            continue
        driver = getattr(obj, "driver_worker", None) or getattr(obj, "worker", None)
        if driver is None:
            continue
        model_runner = getattr(driver, "model_runner", None)
        if model_runner is not None:
            break
    inner = getattr(model_runner, "model", None) if model_runner else None
    if inner is not None and vllm_config is not None:
        maybe_compile_speech_mm_encoder(inner, vllm_config)
    return llm, inner


def _encoder_forward(inner, processed, processed_len):
    perception = getattr(inner, "perception", None)
    encoder = getattr(perception, "encoder", None)
    runner = getattr(inner, "_encoder_graphs", None)
    if runner is not None and hasattr(runner, "_call_short"):
        frames = processed.shape[-1]
        if frames > 2 * int(getattr(runner, "window", 550)):
            return runner._call_long(processed, processed_len, False)
        return runner._call_short(processed, processed_len, False)
    if encoder is None:
        raise RuntimeError("could not find perception.encoder or _encoder_graphs")
    return encoder(processed, processed_len)


def _run_model(args: argparse.Namespace) -> int:
    import torch

    from vllm.compilation import mm_encoder as mm_encoder_mod

    hop = args.hop_s
    n_feat = args.n_feat
    frames = list(args.frames)
    for sec in args.seconds:
        frames.append(_seconds_to_frames(sec, hop))
    frames = sorted(set(int(f) for f in frames if f > 0))
    print(f"loading compiled encoder from {args.model}")
    compiled_llm, compiled_inner = _load_vllm_encoder(args.model, compile_mm=True)
    if compiled_inner is None:
        print("WARN: could not unwrap compiled model; flex-only fallback")
        return _run_flex_only(args)
    device = next(compiled_inner.parameters()).device
    dtype = next(p.dtype for p in compiled_inner.parameters() if p.is_floating_point())
    print(f"compiled model device={device} dtype={dtype}")

    def encode(inner, t: int, *, inference: bool, batch: int = 1):
        processed = torch.randn(batch, n_feat, t, device=device, dtype=dtype)
        processed_len = torch.tensor([t], device=device, dtype=torch.int64)
        mm_encoder_mod._begin_encoder_mask(t)
        ctx = torch.inference_mode() if inference else torch.nullcontext()
        with ctx:
            return _encoder_forward(inner, processed, processed_len)

    print("warmup + compiled encodings")
    torch._dynamo.reset()
    with torch.inference_mode():
        for t in (256, args.window, 2 * args.window):
            try:
                encode(compiled_inner, t, inference=True)
            except Exception as exc:
                print(f"  warmup T={t} skipped: {exc}")
    after_warmup = _unique_graphs()
    print(f"  unique_graphs after warmup={after_warmup}")

    failed = 0
    compiled_out = {}
    for t in frames:
        try:
            with torch.inference_mode():
                torch.manual_seed(t)
                compiled_out[t] = encode(compiled_inner, t, inference=True)
            print(f"  compiled T={t} ok")
        except Exception as exc:
            failed += 1
            print(f"  compiled T={t} FAIL: {exc}")
            traceback.print_exc()

    if args.aligner:
        print("aligner-like: outside inference_mode, return_ctc_timestamp_inputs")
        runner = getattr(compiled_inner, "_encoder_graphs", None)
        mm_encoder_mod._set_serving_watch(True)
        for t in (args.window, 600, 2 * args.window):
            processed = torch.randn(2, n_feat, t, device=device, dtype=dtype)
            processed_len = torch.full((2,), t, device=device, dtype=torch.int64)
            try:
                if runner is not None and hasattr(runner, "_call_short"):
                    out = runner._call_short(processed, processed_len, True)
                else:
                    out = encode(compiled_inner, t, inference=False, batch=2)
                print(
                    f"  aligner T={t} out_type={type(out).__name__} "
                    f"graphs={_unique_graphs()}"
                )
            except Exception as exc:
                failed += 1
                print(f"  aligner T={t} FAIL: {exc}")
                traceback.print_exc()
        if _unique_graphs() != after_warmup and after_warmup >= 0:
            failed += 1
            print(
                f"  FAIL recompile: unique_graphs {after_warmup} -> {_unique_graphs()}"
            )
        mm_encoder_mod._set_serving_watch(False)

    print("loading eager (compile off) encoder")
    del compiled_llm
    if device.type == "cuda":
        torch.accelerator.empty_cache()
    eager_llm, eager_inner = _load_vllm_encoder(args.model, compile_mm=False)
    if eager_inner is None:
        print("WARN: could not unwrap eager model")
        return 1 if failed else 0
    print("compiled vs eager")
    for t in frames:
        if t not in compiled_out:
            continue
        try:
            torch.manual_seed(t)
            eager_out = encode(eager_inner, t, inference=True)
            left = compiled_out[t]
            if isinstance(left, tuple):
                left = left[0]
            if isinstance(eager_out, tuple):
                eager_out = eager_out[0]
            if isinstance(left, list):
                left = left[0]
            if isinstance(eager_out, list):
                eager_out = eager_out[0]
            rel = _rel_err(left, eager_out)
            mad = _max_abs(left, eager_out)
            ok = rel < 0.08
            print(f"  T={t} max|d|={mad:.4g} rel={rel:.4g} {'OK' if ok else 'FAIL'}")
            if not ok:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"  T={t} compare FAIL: {exc}")
            traceback.print_exc()

    if args.burst and compiled_inner is not None:
        print("burst skipped after eager reload (run --flex-only --burst)")

    del eager_llm
    print(f"model repro done failed={failed}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="Nemotron Transcribe checkpoint")
    parser.add_argument(
        "--flex-only",
        action="store_true",
        help="Skip model load; exercise the patched flex wrapper only",
    )
    parser.add_argument(
        "--seconds",
        default="7.9,11,121",
        help="Comma-separated durations (converted at --hop-s)",
    )
    parser.add_argument(
        "--frames",
        default="99,256,550,1100,1513",
        help="Comma-separated encoder frame lengths",
    )
    parser.add_argument("--hop-s", type=float, default=0.08, help="Seconds per frame")
    parser.add_argument("--window", type=int, default=550)
    parser.add_argument("--n-feat", type=int, default=128)
    parser.add_argument(
        "--aligner",
        action="store_true",
        help="Exercise the timestamp-aligner call (outside inference_mode)",
    )
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Re-encode the same inputs after a mixed-length burst",
    )
    args = parser.parse_args(argv)
    args.seconds = _parse_floats(args.seconds)
    args.frames = _parse_ints(args.frames)
    if args.flex_only or not args.model:
        if not args.model and not args.flex_only:
            print("no --model; using --flex-only")
        return _run_flex_only(args)
    return _run_model(args)


if __name__ == "__main__":
    sys.exit(main())

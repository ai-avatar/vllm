# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compiled vs eager Nemotron Transcribe encoder outputs.

Run on a GPU pod with the real checkpoint to reproduce P3d/P3e failures
(aligner AutogradCUDA recompile / dense OOM, stale BlockMask, 7.9 s
hallucination, 121 s truncation, startup 500-frame dummy).

Examples:
  python tools/nemotron_encoder_repro.py --flex-only \\
      --seconds 7.9,11,121 --frames 99,256,500,550,1100,1513 --aligner --burst

  VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=0 \\
  python tools/nemotron_encoder_repro.py \\
      --model /path/to/nemotron-3.5-transcribe-bf16 \\
      --seconds 7.9,11,121 --aligner --burst

  # Offline LLM e2e (7.9 s, 121 s, ~25 min; plain + timestamp path):
  python tools/nemotron_encoder_repro.py --model /path/to/ckpt --e2e \\
      --audio /path/to/7.9s.wav,/path/to/ami_121s.wav,/path/to/25min.wav

"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import tempfile
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


def _nullcontext():
    return contextlib.nullcontext()


def _setup_flex(window: int = 550) -> Any:
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("TORCH_LOGS", "recompiles")
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
    from torch.nn.attention.flex_attention import flex_attention as imported_flex

    from vllm.compilation.mm_encoder import _pad_flex_wrapper

    mm_encoder_mod = _setup_flex(args.window)
    raw_flex = mm_encoder_mod._UNWRAPPED_FLEX or imported_flex
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hop = args.hop_s
    frames = list(args.frames)
    for sec in args.seconds:
        frames.append(_seconds_to_frames(sec, hop))
    frames = sorted(set(int(f) for f in frames if f > 0))
    print(f"device={device} frames={frames} hop_s={hop}")

    def eager_flex(q, k, v, **kwargs):
        return raw_flex(q, k, v, **kwargs)

    compiled = torch.compile(eager_flex, dynamic=False)
    mm_encoder_mod._EAGER_FLEX = mm_encoder_mod._disable_dynamo_fn(eager_flex)
    wrapped = _pad_flex_wrapper(compiled)

    def _qkv(batch: int, t: int):
        q = torch.randn(batch, 16, t, 80, device=device, dtype=dtype)
        return q, q.clone(), q.clone()

    def one(batch: int, t: int, *, inference: bool):
        q, k, v = _qkv(batch, t)
        ctx = torch.inference_mode() if inference else _nullcontext()
        with ctx:
            mm_encoder_mod._begin_encoder_mask(t)
            return wrapped(q, k, v)

    burst_ts = (1, 50, 99, 129, 400, 500, 549, 551, 1513, 1906)
    from vllm.compilation.mm_encoder import speech_encoder_flex_bucket

    warm_ts = {256, args.window, 2 * args.window, 600, *frames, *burst_ts}
    warm_buckets = sorted({speech_encoder_flex_bucket(t) for t in warm_ts})
    print("warmup compiled buckets under inference_mode", warm_buckets)
    torch._dynamo.reset()
    with torch.inference_mode():
        for t in warm_buckets:
            one(1, t, inference=True)
    after_warmup = _unique_graphs()
    print(f"  unique_graphs after warmup={after_warmup}")

    failed = 0
    if args.aligner:
        print("aligner path: B=2, no inference_mode, no requires_grad")
        mm_encoder_mod._set_serving_watch(True)
        for t in (args.window, 600, 2 * args.window):
            try:
                out = one(2, t, inference=False)
                print(f"  T={t} out={tuple(out.shape)} graphs={_unique_graphs()}")
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
        q, k, v = _qkv(1, t)
        with torch.inference_mode():
            mm_encoder_mod._begin_encoder_mask(t)
            compiled_out = wrapped(q, k, v)
            mm_encoder_mod._begin_encoder_mask(t)
            eager_out = eager_flex(
                q,
                k,
                v,
                kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
            )
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
                q, k, v = _qkv(1, t)
                mm_encoder_mod._begin_encoder_mask(t)
                probes[t] = ((q, k, v), wrapped(q, k, v))
            for t in burst_ts:
                oq, ok, ov = _qkv(1, t)
                mm_encoder_mod._begin_encoder_mask(t)
                wrapped(oq, ok, ov)
            for t, ((q, k, v), first) in probes.items():
                mm_encoder_mod._begin_encoder_mask(t)
                second = wrapped(q, k, v)
                match = bool(torch.equal(first, second))
                print(f"  T={t} bitwise={'OK' if match else 'FAIL'}")
                if not match:
                    failed += 1

    print(f"flex-only done failed={failed}")
    return 1 if failed else 0


def _load_vllm_encoder(model: str, compile_mm: bool, gpu_memory_utilization: float):
    from vllm import LLM
    from vllm.compilation.mm_encoder import (
        isolate_flex_attention_from_compile,
        maybe_compile_speech_mm_encoder,
    )

    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("TORCH_LOGS", "recompiles")
    isolate_flex_attention_from_compile()
    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
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


def _tensor_to_cpu(obj):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().contiguous().cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tensor_to_cpu(x) for x in obj)
    return obj


def _tensor_to_payload(tensor) -> dict[str, object]:
    import pybase64 as base64
    import torch

    cpu = tensor.detach().contiguous().cpu().clone()
    data = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "data": base64.b64encode(data).decode("ascii"),
    }


def _first_tensor(obj):
    import torch

    obj = _tensor_to_cpu(obj)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)) and obj:
        return _first_tensor(obj[0])
    raise TypeError(f"no tensor in {type(obj)}")


def _payload_to_tensor(payload: dict[str, object]):
    import pybase64 as base64
    import torch

    shape = payload["shape"]
    dtype_s = str(payload["dtype"]).removeprefix("torch.")
    raw = base64.b64decode(str(payload["data"]))
    dtype = getattr(torch, dtype_s)
    storage = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return storage.view(dtype).reshape(shape).clone()


def _shutdown_llm(llm) -> None:
    """Release GPU memory so a second engine can load in this or another process."""
    if llm is None:
        return
    for name in ("shutdown", "close"):
        fn = getattr(llm, name, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                fn()
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    if engine is not None:
        for name in ("shutdown", "destroy"):
            fn = getattr(engine, name, None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    fn()
        for child_name in ("engine_core", "model_executor"):
            child = getattr(engine, child_name, None)
            if child is None:
                continue
            fn = getattr(child, "shutdown", None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    fn()
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.accelerator.empty_cache()
            torch.accelerator.synchronize()
    except Exception:
        pass


def _eager_encode_worker(model: str, frames: list[int], n_feat: int, out_path: str):
    import torch

    from vllm.compilation import mm_encoder as mm_encoder_mod

    llm, inner = _load_vllm_encoder(
        model, compile_mm=False, gpu_memory_utilization=0.35
    )
    if inner is None:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"ok": False, "error": "no inner model"}, fh)
        return
    device = next(inner.parameters()).device
    dtype = next(p.dtype for p in inner.parameters() if p.is_floating_point())
    outs = {}
    for t in frames:
        torch.manual_seed(t)
        processed = torch.randn(1, n_feat, t, device=device, dtype=dtype)
        processed_len = torch.tensor([t], device=device, dtype=torch.int64)
        mm_encoder_mod._begin_encoder_mask(t)
        with torch.inference_mode():
            outs[str(t)] = _tensor_to_payload(
                _first_tensor(_encoder_forward(inner, processed, processed_len))
            )
    del llm
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"ok": True, "outs": outs}, fh)


def _run_eager_subprocess(model: str, frames: list[int], n_feat: int) -> dict:
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = tmp.name
    cmd = [
        sys.executable,
        "-c",
        (
            "from tools.nemotron_encoder_repro import _eager_encode_worker; "
            f"_eager_encode_worker({model!r}, {frames!r}, {n_feat}, {out_path!r})"
        ),
    ]
    env = os.environ.copy()
    env.setdefault("VLLM_BATCH_INVARIANT", "1")
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    env.setdefault("TORCH_LOGS", "recompiles")
    print(f"eager subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False, env=env)
    if proc.returncode != 0:
        return {"ok": False, "error": f"eager worker exit {proc.returncode}"}
    with open(out_path, encoding="utf-8") as fh:
        return json.load(fh)


def _run_model(args: argparse.Namespace) -> int:
    import torch

    from vllm.compilation import mm_encoder as mm_encoder_mod

    hop = args.hop_s
    n_feat = args.n_feat
    frames = list(args.frames)
    for sec in args.seconds:
        frames.append(_seconds_to_frames(sec, hop))
    frames = sorted(set(int(f) for f in frames if f > 0))
    print("eager encodings in a subprocess (before compiled engine)")
    eager = _run_eager_subprocess(args.model, frames, n_feat)
    if not eager.get("ok"):
        print(f"FAIL: eager subprocess failed: {eager.get('error')}")
        return 1
    print(f"loading compiled encoder from {args.model}")
    compiled_llm, compiled_inner = _load_vllm_encoder(
        args.model, compile_mm=True, gpu_memory_utilization=args.gpu_memory_utilization
    )
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
        ctx = torch.inference_mode() if inference else _nullcontext()
        with ctx:
            return _encoder_forward(inner, processed, processed_len)

    print("warmup + compiled encodings")
    torch._dynamo.reset()
    with torch.inference_mode():
        for t in (256, args.window, 2 * args.window, 500):
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
                compiled_out[t] = _tensor_to_payload(
                    _first_tensor(encode(compiled_inner, t, inference=True))
                )
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

    print("compiled vs eager")
    for t in frames:
        if t not in compiled_out:
            continue
        try:
            left = _payload_to_tensor(compiled_out[t])
            eager_out = _payload_to_tensor(eager["outs"][str(t)])
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

    print(f"model repro done failed={failed}")
    _shutdown_llm(compiled_llm)
    return 1 if failed else 0


_NEMOTRON_TRANSCRIBE_TEXT = "Transcribe with punctuation and capitalization."
_NEMOTRON_AUDIO_PLACEHOLDERS = ("<|audio|>", "<audio>", "<|AUDIO|>")
# nemotron-api VAD / fixed chunks are 2–5 min; encoder cache is 8192 tokens.
_E2E_CHUNK_S = 240.0
_E2E_MAX_SINGLE_S = 300.0


def _synth_wav(seconds: float, sr: int = 16000):
    import numpy as np

    n = max(int(seconds * sr), 1)
    t = np.arange(n, dtype=np.float32) / float(sr)
    wave = 0.08 * np.sin(2 * 3.14159265 * 180.0 * t)
    return wave, sr


def _load_audio_file(path: str, sr: int = 16000):
    from vllm.multimodal.media.audio import load_audio

    return load_audio(path, sr=sr)


def _e2e_wave(seconds: float, audio_paths: list[str], sr: int = 16000):
    """Prefer a real file (tiled/cropped to ``seconds``) over a tone."""
    import numpy as np

    need = max(int(seconds * sr), 1)
    wave = None
    if audio_paths:
        idx = 0
        if seconds >= 1000:
            idx = min(2, len(audio_paths) - 1)
        elif seconds >= 60:
            idx = min(1, len(audio_paths) - 1)
        wave, file_sr = _load_audio_file(audio_paths[idx], sr=sr)
        if file_sr and int(file_sr) != sr:
            sr = int(file_sr)
            need = max(int(seconds * sr), 1)
    if wave is None:
        try:
            from vllm.assets.audio import AudioAsset

            wave, file_sr = AudioAsset("mary_had_lamb").audio_and_sample_rate
            sr = int(file_sr) if file_sr else sr
            need = max(int(seconds * sr), 1)
        except Exception:
            return _synth_wav(seconds, sr)
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if wave.size < need:
        reps = (need + wave.size - 1) // max(wave.size, 1)
        wave = np.tile(wave, reps)
    return wave[:need], sr


def _print_output_text(outputs) -> str:
    if not outputs:
        return ""
    first = outputs[0]
    out = getattr(first, "outputs", None)
    if out:
        return str(out[0].text)
    return str(first)


def _audio_placeholder_from_llm(llm) -> str:
    tokenizer = None
    with contextlib.suppress(Exception):
        tokenizer = llm.get_tokenizer()
    for obj in (
        tokenizer,
        getattr(tokenizer, "processor", None) if tokenizer is not None else None,
        getattr(llm, "llm_engine", None),
    ):
        if obj is None:
            continue
        for name in ("audio_token", "audio_placeholder", "placeholder_audio"):
            raw = getattr(obj, name, None)
            if isinstance(raw, str) and raw:
                return raw
    return _NEMOTRON_AUDIO_PLACEHOLDERS[0]


def _wav_data_uri(wave, sr: int) -> str:
    import io
    import wave as wav_mod

    import numpy as np
    import pybase64 as base64

    pcm = np.clip(np.asarray(wave, dtype=np.float32) * 32767.0, -32768, 32767)
    buf = io.BytesIO()
    with wav_mod.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sr))
        handle.writeframes(pcm.astype("<i2").tobytes())
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _nemotron_chat_messages(wave, sr: int) -> list[dict[str, Any]]:
    """Same user turn as nemotron-api / the HF Transcribe card: text then audio."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _NEMOTRON_TRANSCRIBE_TEXT},
                {
                    "type": "audio_url",
                    "audio_url": {"url": _wav_data_uri(wave, sr)},
                },
            ],
        }
    ]


def _nemotron_e2e_generate_prompt(llm, wave, sr: int) -> dict[str, Any]:
    """Fallback generate() prompt: text then <|audio|>, no system message."""
    ph = _audio_placeholder_from_llm(llm)
    prompt = (
        f"<|im_start|>user\n{_NEMOTRON_TRANSCRIBE_TEXT}{ph}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
    tokenizer = None
    with contextlib.suppress(Exception):
        tokenizer = llm.get_tokenizer()
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        with contextlib.suppress(Exception):
            rendered = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _NEMOTRON_TRANSCRIBE_TEXT},
                            {"type": "audio"},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if rendered and "audio" in rendered.lower():
                prompt = rendered
    return {"prompt": prompt, "multi_modal_data": {"audio": (wave, sr)}}


def _chunk_e2e_audio(seconds: float, wave, sr: int) -> list[tuple[float, Any]]:
    """Split >5 min clips into ~4 min pieces (nemotron-api 2–5 min chunks)."""
    import numpy as np

    wave = np.asarray(wave).reshape(-1)
    if seconds <= _E2E_MAX_SINGLE_S:
        return [(seconds, wave)]
    chunk = max(int(_E2E_CHUNK_S * sr), 1)
    parts = []
    for start in range(0, wave.size, chunk):
        piece = wave[start : start + chunk]
        parts.append((len(piece) / float(sr), piece))
    print(f"  chunked {seconds}s -> {len(parts)} x ~{_E2E_CHUNK_S:.0f}s")
    return parts


def _e2e_one_request(llm, wave, sr: int, sampling, *, mm_kwargs=None):
    messages = _nemotron_chat_messages(wave, sr)
    try:
        return llm.chat(
            messages,
            sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
            mm_processor_kwargs=mm_kwargs,
        )
    except TypeError:
        return llm.chat(
            messages,
            sampling,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception:
        prompt = _nemotron_e2e_generate_prompt(llm, wave, sr)
        return llm.generate(prompt, sampling, mm_processor_kwargs=mm_kwargs)


def _run_e2e(args: argparse.Namespace) -> int:
    """Offline LLM.chat like nemotron-api: 7.9 s / 121 s / chunked long file."""
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("TORCH_LOGS", "recompiles")
    seconds = args.e2e_seconds or [7.9, 121.0, 1500.0]
    print(f"e2e LLM load {args.model} seconds={seconds} audio={args.audio}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        compilation_config={
            "compile_mm_encoder": True,
            "cudagraph_mm_encoder": True,
        },
        enable_prefix_caching=False,
    )
    after_load = _unique_graphs()
    print(f"unique_graphs after load={after_load}")
    failed = 0
    for sec in seconds:
        wave, sr = _e2e_wave(sec, args.audio)
        chunks = _chunk_e2e_audio(sec, wave, sr)
        print(
            f"--- plain {sec}s ({len(wave)} samples @ {sr}, {len(chunks)} chunk(s)) ---"
        )
        try:
            before = _unique_graphs()
            texts = []
            for chunk_s, piece in chunks:
                outs = _e2e_one_request(
                    llm,
                    piece,
                    sr,
                    SamplingParams(temperature=0.0, max_tokens=args.e2e_max_tokens),
                )
                texts.append(_print_output_text(outs))
            text = " ".join(part for part in texts if part)
            after = _unique_graphs()
            print(f"  graphs {before} -> {after}")
            print(f"  text[:240]={text[:240]!r}")
            if not text.strip():
                failed += 1
                print("  FAIL: empty transcript")
            if after_load >= 0 and after > after_load and before == after_load:
                print("  WARN: new graphs on this request")
        except Exception as exc:
            failed += 1
            print(f"  plain FAIL: {exc}")
            traceback.print_exc()
        print(f"--- timestamps/verbose {sec}s ---")
        try:
            before = _unique_graphs()
            texts = []
            for chunk_s, piece in chunks:
                outs = _e2e_one_request(
                    llm,
                    piece,
                    sr,
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=args.e2e_max_tokens,
                        extra_args={"return_ctc_timestamp_inputs": True},
                    ),
                    mm_kwargs={
                        "return_timestamps": True,
                        "timestamp_granularities": ["word"],
                    },
                )
                texts.append(_print_output_text(outs))
            text = " ".join(part for part in texts if part)
            after = _unique_graphs()
            print(f"  graphs {before} -> {after}")
            print(f"  text[:240]={text[:240]!r}")
            if not text.strip():
                failed += 1
                print("  FAIL: empty transcript")
            if after != before and before >= 0:
                failed += 1
                print("  FAIL: timestamp path created new dynamo graphs")
        except Exception as exc:
            failed += 1
            print(f"  timestamps FAIL: {exc}")
            traceback.print_exc()
    print(f"e2e done failed={failed} unique_graphs={_unique_graphs()}")
    _shutdown_llm(llm)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("TORCH_LOGS", "recompiles")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="Nemotron Transcribe checkpoint")
    parser.add_argument(
        "--flex-only",
        action="store_true",
        help="Skip model load; exercise the patched flex wrapper only",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Offline LLM.generate for 7.9s / 121s / ~25min, plain + timestamps",
    )
    parser.add_argument(
        "--e2e-seconds",
        default="7.9,121,1500",
        help="Durations for --e2e (1500 s ≈ 25 min)",
    )
    parser.add_argument("--e2e-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--audio",
        default="",
        help="Comma-separated real audio paths for --e2e (LibriSpeech/AMI)",
    )
    parser.add_argument(
        "--seconds",
        default="7.9,11,121",
        help="Comma-separated durations (converted at --hop-s)",
    )
    parser.add_argument(
        "--frames",
        default="99,256,500,550,1100,1513",
        help="Comma-separated encoder frame lengths",
    )
    parser.add_argument("--hop-s", type=float, default=0.08, help="Seconds per frame")
    parser.add_argument("--window", type=int, default=550)
    parser.add_argument("--n-feat", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
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
    args.e2e_seconds = _parse_floats(args.e2e_seconds)
    args.audio = [part.strip() for part in args.audio.split(",") if part.strip()]
    if args.e2e:
        if not args.model:
            print("--e2e requires --model")
            return 2
        return _run_e2e(args)
    if args.flex_only or not args.model:
        if not args.model and not args.flex_only:
            print("no --model; using --flex-only")
        return _run_flex_only(args)
    return _run_model(args)


if __name__ == "__main__":
    sys.exit(main())

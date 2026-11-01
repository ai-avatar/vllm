# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile plugin / speech multimodal encoders under ``compile_mm_encoder``.

vLLM already compiles in-tree vision encoders via ``@support_torch_compile``
and ``should_torch_compile_mm_encoder``. The NeMo SpeechLM plugin's
perception encoder is built outside that decorator. This module compiles
``perception.encoder`` transformer blocks (the same set as the plugin's
``EncoderGraphRunner``) and maps the official compilation flags onto the
plugin's ``encoder_compile`` / ``encoder_cuda_graphs`` hf-overrides.

Inductor's flex-decoding kernel is wrong for this encoder's rel-pos
``score_mod`` and padding ``block_mask`` when ``Q`` is shorter than 128
frames. Flex stays inside the compiled graph with
``FORCE_USE_FLEX_ATTENTION`` so the main flex kernel is used. Graph-
breaking the NeMo ``_flex_attention_compiled`` wrapper is correct but
~6× slower. vLLM owns layer compile so the plugin cannot
``compile_layers`` first under graphs.

The plugin windows long audio by ``processed_len``, so the last
window's T is ``valid % window`` even when the tensor is padded.
Dynamo then recompiles flex / ``create_block_mask`` on that remainder
(``seq_lengths``, Q time, often strides). Flex q/k/v and the block
mask are padded onto a closed bucket set and NeMo's flex wrapper is
compiled with ``dynamic=False`` so every length in ``0..cap`` hits a
warmed kernel. Layer compile stays dynamic as a fallback for non-flex
ops. Long-graph warmup is one 3-window dummy plus the per-window
graph; remainders pad to ``window``. The timestamp aligner (and any
encoder call outside ``inference_mode``) uses uncompiled FORCE_USE
flex after promoting inputs to inference tensors. The pad wrapper is
``torch.compiler.disable``'d so dynamo cannot ``set_stance`` from
inside a compiled layer; q/k/v are contiguous before the compiled
flex call. ``mask_mod`` reads valid length from a tensor so new
lengths cannot recompile. Cached BlockMasks are structural (full
bucket); the valid-length tensor is written before each call / graph
replay, never baked during capture. Warmup includes the plugin
profile dummy (``3 * max_frames`` when that is the 4000-frame cap).

Speech encoder CUDA graphs are captured lazily after KV allocation, with
batch sizes ``(1,)`` under ``VLLM_BATCH_INVARIANT``. The replay path is a
function of each item's valid length, not the padded tensor width.
Layer compile is also deferred until after KV so inductor workspaces are
not counted as peak activation. A post-KV graph warmup then captures
every (batch, frame-bucket) using ``n_feat`` from the perception
preprocessor / encoder, then one long-audio ``3 * window`` shape
(BI batch 1). Compile-only skips dummy-audio warmup.
Pending-compile state is stored on the unwrapped module:
``CUDAGraphWrapper.__getattr__`` forwards, but ``delattr`` does not.

Under ``VLLM_BATCH_INVARIANT``, inductor ``deterministic`` is pinned
on ``layer.compile`` and NeMo's flex wrapper so timing autotune cannot
select different tiles across process starts. Served tensors and flex
inputs stay on that closed set so later traffic cannot recompile.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_COMPILED_ATTR = "_vllm_mm_encoder_compiled"
_FLEX_ISOLATED_ATTR = "_vllm_speech_encoder_flex_isolated"
_FLEX_PADDED_ATTR = "_vllm_speech_encoder_flex_padded"
_BI_FLEX_COMPILED_ATTR = "_vllm_speech_encoder_flex_bi_compiled"
_RUNNER_PATCH_ATTR = "_vllm_speech_graph_patched"
_PENDING_LAYERS_ATTR = "_vllm_speech_encoder_pending_compile"
_DYNAMO_CONFIGURED = False
_INDUCTOR_PINNED = False
_BI_FLEX_COMPILED: Any = None
_UNWRAPPED_CREATE_BLOCK_MASK: Any = None
_UNWRAPPED_FLEX: Any = None
_EAGER_FLEX: Any = None
_DYNAMO_DISABLED_ATTR = "_vllm_speech_encoder_dynamo_disabled"
_ALIGNER_ACTIVE = False
_ENCODER_WRAP_ATTR = "_vllm_speech_encoder_inference_wrapped"
# Model-param dtype used at compile/warmup. Request-time encoder
# forwards must pin the same default_dtype or dynamo recompiles on
# GLOBAL_STATE and the new graph changes embeddings.
_ENCODER_DEFAULT_DTYPE: torch.dtype | None = None
_LAST_ENCODER_PATH = ""
_LAST_PROCESSED_SHAPE: tuple[int, ...] | None = None
_LAST_PROCESSED_LEN: int | None = None
_LAST_FLEX_INPUT: dict[str, Any] = {}
# Per-request valid length from create_block_mask / processed_len.
# Must not leak across requests: _begin_encoder_mask resets it.
_MASK_VALID_LEN: int | None = None
# Per-device tensor valid-length + cached bucket BlockMasks. mask_mod must
# close over these tensors (not Python ints) or dynamo guards the length.
# Cache key is (bucket, B, H) only: structure is always a full bucket so
# a short valid cannot mark tail blocks empty for a later longer valid.
# Aligner traffic uses a ``:aligner`` device-key suffix so fill_ / cache
# writes cannot mutate the serving tensors compiled encodes replay.
_MASK_STATE: dict[str, dict[str, Any]] = {}
# getattr on CUDAGraphWrapper forwards to .runnable; delattr does not.
# Keep a side table so pending layers survive wrap/unwrap either way.
_PENDING_LAYERS: weakref.WeakKeyDictionary[Any, list[nn.Module]] = (
    weakref.WeakKeyDictionary()
)

_NEMOTRON_SPEECH_ARCH_MARKERS = ("NeMoSpeechLM",)
_NEMOTRON_SPEECH_MODEL_TYPES = ("nemo_speechlm",)

# Windowed Sortformer / long-audio forwards hit more shape classes than
# dynamo's default ``recompile_limit`` (8). Past the limit, compiled
# blocks silently run eager code.
_SPEECH_ENCODER_RECOMPILE_LIMIT = 64
# A100 80GB + ~63GB weights: a 16-row grid left ~1.1 GiB for KV.
_MAX_ENCODER_GRAPH_BATCH = 8
# 80 ms / encoder frame; 256-frame graph buckets.
_ENCODER_FRAME_HOP_S = 0.08
_DEFAULT_FRAME_BUCKET = 256
_DEFAULT_MAX_WINDOWS = 2
# First long-audio path is 3 windows (2-window clips stay whole-forward).
_LONG_WARMUP_MIN_WINDOWS = 3
# Flex / last-window remainders pad onto these (overridden from the runner).
_FLEX_WINDOW = 550
_FLEX_MAX_WINDOWS = _DEFAULT_MAX_WINDOWS
_FLEX_BUCKET_STEP = _DEFAULT_FRAME_BUCKET
_SERVING_WATCH = False
_RECOMPILE_HOOK_INSTALLED = False
_FEAT_ATTRS = (
    "features",
    "n_mels",
    "nfilt",
    "n_filt",
    "num_features",
    "n_features",
    "feat_in",
    "_feat_in",
    "n_feat",
)


def is_nemo_speechlm_config(hf_config: object | None) -> bool:
    """True when ``hf_config`` is a NeMo SpeechLM / Nemotron Transcribe config."""
    if hf_config is None:
        return False
    model_type = str(getattr(hf_config, "model_type", "") or "")
    if model_type in _NEMOTRON_SPEECH_MODEL_TYPES:
        return True
    architectures = getattr(hf_config, "architectures", None) or []
    return any(
        any(marker in str(arch) for marker in _NEMOTRON_SPEECH_ARCH_MARKERS)
        for arch in architectures
    )


def default_speech_encoder_graph_batch_sizes(
    vllm_config: object | None = None,
) -> tuple[int, ...]:
    """Graph batch grid: ``(1,)`` under BI, otherwise powers of two.

    Sized from ``max_num_seqs`` and capped so capture is not counted as
    peak activation next to the 30B weights.
    """
    if envs.VLLM_BATCH_INVARIANT:
        return (1,)
    max_seqs = _MAX_ENCODER_GRAPH_BATCH
    if vllm_config is not None:
        sched = getattr(vllm_config, "scheduler_config", None)
        if sched is not None:
            raw = getattr(sched, "max_num_seqs", None)
            if raw:
                max_seqs = min(int(raw), _MAX_ENCODER_GRAPH_BATCH)
    sizes: list[int] = []
    n = 1
    while n <= max(1, max_seqs):
        sizes.append(n)
        n *= 2
    return tuple(sizes)


def unwrap_speech_encoder_model(model: Any) -> Any:
    """Walk ``unwrap()`` / ``.runnable`` to the inner module.

    ``CUDAGraphWrapper.__getattr__`` forwards to the runnable, but
    ``delattr`` does not. Pending-compile state must live on the
    unwrapped module (or in ``_PENDING_LAYERS``).
    """
    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        unwrap = getattr(current, "unwrap", None)
        if callable(unwrap):
            try:
                inner = unwrap()
            except Exception:
                inner = None
            if inner is not None and inner is not current:
                current = inner
                continue
        runnable = getattr(current, "runnable", None)
        if runnable is not None and runnable is not current:
            current = runnable
            continue
        break
    return current


def _store_pending_layers(model: Any, layers: list[nn.Module]) -> None:
    owner = unwrap_speech_encoder_model(model)
    with contextlib.suppress(Exception):
        setattr(owner, _PENDING_LAYERS_ATTR, layers)
    with contextlib.suppress(TypeError):
        _PENDING_LAYERS[owner] = layers


def _pop_pending_layers(model: Any) -> list[nn.Module] | None:
    owner = unwrap_speech_encoder_model(model)
    layers = getattr(owner, _PENDING_LAYERS_ATTR, None)
    if layers is None:
        layers = getattr(model, _PENDING_LAYERS_ATTR, None)
    if layers is None:
        with contextlib.suppress(TypeError, KeyError):
            layers = _PENDING_LAYERS.pop(owner, None)
    if layers is None:
        with contextlib.suppress(TypeError, KeyError):
            layers = _PENDING_LAYERS.pop(model, None)
    for obj in (owner, model):
        with contextlib.suppress(AttributeError):
            delattr(obj, _PENDING_LAYERS_ATTR)
        with contextlib.suppress(TypeError, KeyError):
            _PENDING_LAYERS.pop(obj, None)
    return layers


def sync_mm_encoder_compile_flags(vllm_config: VllmConfig) -> None:
    """Align compilation_config, env vars, and NeMo Speech hf-overrides.

    Sources (any one enables the corresponding flag):

    * ``compilation_config.compile_mm_encoder`` /
      ``compilation_config.cudagraph_mm_encoder``
    * ``VLLM_COMPILE_MM_ENCODER`` / ``VLLM_CUDAGRAPH_MM_ENCODER``
    * hf-overrides ``encoder_compile`` / ``encoder_cuda_graphs``

    All three graph sources are equivalent, including under
    ``VLLM_BATCH_INVARIANT``. The effective graph value is written back
    to both ``compilation_config.cudagraph_mm_encoder`` and (for
    SpeechLM) ``hf_config.encoder_cuda_graphs``. Layer compile is owned
    by vLLM, so ``hf_config.encoder_compile`` is cleared and the plugin
    must not ``compile_layers`` itself.
    """
    compilation = vllm_config.compilation_config
    if envs.VLLM_COMPILE_MM_ENCODER:
        compilation.compile_mm_encoder = True
    if envs.VLLM_CUDAGRAPH_MM_ENCODER:
        compilation.cudagraph_mm_encoder = True

    hf_config = None
    if vllm_config.model_config is not None:
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
    if hf_config is None:
        return

    if getattr(hf_config, "encoder_compile", False):
        compilation.compile_mm_encoder = True
    if getattr(hf_config, "encoder_cuda_graphs", False):
        compilation.cudagraph_mm_encoder = True

    if not is_nemo_speechlm_config(hf_config):
        return

    # Plugin EncoderGraphRunner(compile_layers=True) inductor-compiles
    # before flex_attention is isolated and ignores this module's compile
    # path. vLLM compiles the same blocks after load instead.
    hf_config.encoder_compile = False

    want_graphs = bool(compilation.cudagraph_mm_encoder)
    compilation.cudagraph_mm_encoder = want_graphs
    hf_config.encoder_cuda_graphs = want_graphs
    if want_graphs:
        if getattr(hf_config, "encoder_cuda_graph_batch_sizes", None) is None:
            hf_config.encoder_cuda_graph_batch_sizes = list(
                default_speech_encoder_graph_batch_sizes(vllm_config)
            )
        logger.warning_once(
            "Speech encoder CUDA graphs pad to length buckets and mask "
            "by length. Capture is deferred until after KV allocation "
            "and is lazy per (batch, bucket). Under "
            "VLLM_BATCH_INVARIANT the default batch grid is (1,) and "
            "the path is a function of each item's valid length."
        )


class SpeechEncoderGraphFallback(RuntimeError):
    """Graph helper failed; ``__call__`` must return None (eager encode)."""


def set_speech_encoder_shape_config(
    runner: Any = None,
    *,
    window: int | None = None,
    max_windows: int | None = None,
    bucket_step: int | None = None,
    model: Any = None,
) -> None:
    """Pin the flex / last-window bucket sizes from the graph runner."""
    global _FLEX_WINDOW, _FLEX_MAX_WINDOWS, _FLEX_BUCKET_STEP
    if window is None:
        window = _encoder_window(runner, model)
    if window:
        _FLEX_WINDOW = max(int(window), 1)
    if max_windows is None and runner is not None:
        raw = getattr(runner, "max_windows", None)
        if raw:
            max_windows = int(raw)
    if max_windows:
        _FLEX_MAX_WINDOWS = max(int(max_windows), 1)
    if bucket_step is None and runner is not None:
        bucket_step = _frame_bucket_step(runner)
    if bucket_step:
        _FLEX_BUCKET_STEP = max(int(bucket_step), 1)


def speech_encoder_flex_bucket(
    frames: int,
    *,
    window: int | None = None,
    max_windows: int | None = None,
    bucket_step: int | None = None,
) -> int:
    """Map a flex / window length onto the closed compile set.

    Already-served 256-frame and ``n * window`` sizes stay as-is so short
    CUDA graphs keep their warmed T. Off-bucket lengths (last-window
    remainders from the plugin slicing on ``processed_len``) pad to
    ``window``, the next 256-bucket, or the next ``n * window``.
    """
    if frames <= 0:
        return frames
    win = max(int(window) if window is not None else _FLEX_WINDOW, 1)
    max_w = max(int(max_windows) if max_windows is not None else _FLEX_MAX_WINDOWS, 1)
    step = max(int(bucket_step) if bucket_step is not None else _FLEX_BUCKET_STEP, 1)
    short_cap = win * max_w
    if frames <= short_cap and frames % step == 0:
        return frames
    if frames % win == 0:
        return frames
    if frames <= win:
        # A profile dummy may set runner.window to max_frames (e.g. 4000)
        # while the plugin still slices at ~500. Do not pad a short q to
        # that max. Sliding chunks here are ~500-550, not 4k+.
        if win >= max(8 * step, 2048):
            return ((frames + step - 1) // step) * step
        return win
    if frames <= short_cap:
        return ((frames + step - 1) // step) * step
    if frames % step == 0:
        return frames
    return ((frames + win - 1) // win) * win


def _as_pos_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return int(value.item())
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_encoder_path(
    path: str, processed: torch.Tensor, processed_len: torch.Tensor
) -> None:
    global _LAST_ENCODER_PATH, _LAST_PROCESSED_SHAPE, _LAST_PROCESSED_LEN
    _LAST_ENCODER_PATH = path
    if isinstance(processed, torch.Tensor):
        _LAST_PROCESSED_SHAPE = tuple(int(s) for s in processed.shape)
    else:
        _LAST_PROCESSED_SHAPE = None
    if isinstance(processed_len, torch.Tensor) and processed_len.numel() > 0:
        _LAST_PROCESSED_LEN = int(processed_len.max().item())
    else:
        _LAST_PROCESSED_LEN = None


def _record_flex_input(
    q: torch.Tensor,
    block_mask: Any,
    *,
    orig_t: int,
    bucket: int,
) -> None:
    _LAST_FLEX_INPUT.clear()
    _LAST_FLEX_INPUT.update(
        q_shape=tuple(int(s) for s in q.shape),
        q_stride=tuple(int(s) for s in q.stride()),
        q_dtype=str(q.dtype),
        q_contig=bool(q.is_contiguous()),
        orig_t=orig_t,
        bucket=bucket,
        path=_LAST_ENCODER_PATH,
        processed=_LAST_PROCESSED_SHAPE,
        processed_len=_LAST_PROCESSED_LEN,
        block_mask_seq=getattr(block_mask, "seq_lengths", None),
    )


_ENCODER_RECOMPILE_FILE_MARKERS = (
    "transformer_encoder",
    "transformer_encoder_utils",
    "nemo/collections/asr",
    "nemo_speech",
    "speechlm",
)
_ENCODER_RECOMPILE_SKIP_FUNCS = frozenset(
    {
        "_recompile_watch_context",
        "_recompile_code_is_encoder",
        "_recompile_filename_is_encoder",
        "_recompile_target_code",
        "_install_speech_encoder_recompile_watch",
    }
)


def _recompile_filename_is_encoder(filename: str) -> bool:
    path = filename.replace("\\", "/")
    return any(marker in path for marker in _ENCODER_RECOMPILE_FILE_MARKERS)


def _recompile_target_code() -> Any:
    """The ``CodeType`` dynamo is compiling (``convert_frame._compile``).

    The user-context hook is invoked from every process compile, including
    LM CUDA-graph capture. Only that frame's ``code`` local is the function
    being compiled; walking the rest of the stack (or leftover encoder
    path/bucket) false-positives ``activation.forward_native`` /
    ``grouped_topk``.
    """
    frame: Any = sys._getframe()
    while frame is not None:
        code = frame.f_code
        filename = code.co_filename.replace("\\", "/")
        if code.co_name == "_compile" and "convert_frame" in filename:
            obj = frame.f_locals.get("code")
            if type(obj) is type(code):
                return obj
            return None
        frame = frame.f_back
    return None


def _recompile_code_is_encoder() -> bool:
    """True only when the compiled code object's file is the speech encoder."""
    code = _recompile_target_code()
    if code is None:
        return False
    return _recompile_filename_is_encoder(code.co_filename)


def _recompile_watch_context() -> str:
    if not _recompile_code_is_encoder():
        return ""
    extra = (
        f"path={_LAST_ENCODER_PATH} processed={_LAST_PROCESSED_SHAPE} "
        f"len={_LAST_PROCESSED_LEN} flex_q={_LAST_FLEX_INPUT.get('q_shape')} "
        f"stride={_LAST_FLEX_INPUT.get('q_stride')} "
        f"contig={_LAST_FLEX_INPUT.get('q_contig')} "
        f"bucket={_LAST_FLEX_INPUT.get('bucket')} "
        f"mask={_LAST_FLEX_INPUT.get('block_mask_seq')}"
    )
    if _SERVING_WATCH:
        logger.warning("Speech encoder dynamo recompile after warmup; %s", extra)
    return extra[:256]


def _device_key(device: torch.device | str | None) -> str:
    if device is None:
        dev = torch.device("cpu")
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device
    index = 0 if dev.index is None else int(dev.index)
    return f"{dev.type}:{index}"


def _is_inference_mode() -> bool:
    for name in ("is_inference_mode_enabled", "is_inference_mode"):
        fn = getattr(torch, name, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                return bool(fn())
    return False


def _tensor_is_inference(tensor: torch.Tensor) -> bool:
    fn = getattr(tensor, "is_inference", None)
    if callable(fn):
        with contextlib.suppress(Exception):
            return bool(fn())
    return False


def _is_torch_compiling() -> bool:
    """True while dynamo/inductor is tracing or compiling a region."""
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        for name in ("is_compiling", "is_dynamo_compiling"):
            fn = getattr(compiler, name, None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    if bool(fn()):
                        return True
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        fn = getattr(dynamo, "is_compiling", None)
        if callable(fn):
            with contextlib.suppress(Exception):
                return bool(fn())
    return False


def _disable_dynamo_fn(fn: Any) -> Any:
    """Return ``fn`` wrapped so dynamo cannot compile it (recursive)."""
    if fn is None or getattr(fn, _DYNAMO_DISABLED_ATTR, False):
        return fn
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None) if compiler is not None else None
    if not callable(disable):
        disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    if not callable(disable):
        return fn
    try:
        out = disable(fn, recursive=True)
    except TypeError:
        out = disable(fn)
    with contextlib.suppress(Exception):
        setattr(out, _DYNAMO_DISABLED_ATTR, True)
    return out


def _graph_break_if_compiling() -> None:
    """Force a graph break if dynamo still traced into a disabled helper."""
    if not _is_torch_compiling():
        return
    dynamo = getattr(torch, "_dynamo", None)
    graph_break = getattr(dynamo, "graph_break", None) if dynamo is not None else None
    if callable(graph_break):
        graph_break()


@contextlib.contextmanager
def _force_eager_compute():
    """Keep the aligner encoder call out of dynamo (no new unique_graphs).

    ``torch.compiler.set_stance`` raises if it is called while dynamo is
    compiling. Skip it in that case; the caller must already be outside
    the compiled region (``compiler.disable`` / graph break).
    """
    if _is_torch_compiling():
        yield
        return
    compiler = getattr(torch, "compiler", None)
    set_stance = getattr(compiler, "set_stance", None) if compiler is not None else None
    if callable(set_stance):
        with set_stance("force_eager"):
            yield
        return
    yield


def _is_cuda_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    fn = getattr(torch.cuda, "is_current_stream_capturing", None)
    if not callable(fn):
        return False
    with contextlib.suppress(Exception):
        return bool(fn())
    return False


def _begin_encoder_mask(valid: int | None = None) -> None:
    """Reset per-request valid length so a prior clip cannot leak."""
    global _MASK_VALID_LEN
    _MASK_VALID_LEN = int(valid) if valid is not None and valid > 0 else None


def speech_encoder_set_mask_valid(
    device: torch.device | str | None, q_len: int, kv_len: int | None = None
) -> None:
    """Write the live valid-length tensors (must run before graph replay)."""
    global _MASK_VALID_LEN
    kv = int(q_len) if kv_len is None else int(kv_len)
    _MASK_VALID_LEN = int(q_len)
    _set_mask_valid_len(_mask_state(device), int(q_len), kv)


def _mask_state(device: torch.device | str | None) -> dict[str, Any]:
    """Reuse one tensor-backed mask_mod per device (no Python-int guards).

    The timestamp aligner fill_s ``valid_q`` / rebuilds BlockMasks. Keep
    that cache on a separate key so an aligner call cannot mutate the
    serving tensors the next compiled encode replays.
    """
    key = _device_key(device)
    if _ALIGNER_ACTIVE:
        key = f"{key}:aligner"
    state = _MASK_STATE.get(key)
    if state is not None:
        return state
    if device is None:
        dev = torch.device("cpu")
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device
    # Must not be inference tensors: aligner fill_s them outside inference_mode.
    with torch.inference_mode(False):
        valid_q = torch.zeros((), device=dev, dtype=torch.int32)
        valid_kv = torch.zeros((), device=dev, dtype=torch.int32)

    def mask_mod(_b: Any, _h: Any, q_idx: Any, kv_idx: Any):
        return (q_idx < valid_q) & (kv_idx < valid_kv)

    state = {
        "mask_mod": mask_mod,
        "valid_q": valid_q,
        "valid_kv": valid_kv,
        "masks": {},
    }
    _MASK_STATE[key] = state
    return state


def speech_encoder_mask_mod_closes_over_tensors(mask_mod: Any) -> bool:
    """True when ``mask_mod`` valid-length cells are tensors, not ints."""
    closure = getattr(mask_mod, "__closure__", None)
    if not closure:
        return False
    saw_tensor = False
    for cell in closure:
        value = cell.cell_contents
        if isinstance(value, int):
            return False
        if isinstance(value, torch.Tensor):
            saw_tensor = True
    return saw_tensor


def _set_mask_valid_len(state: dict[str, Any], q_len: int, kv_len: int) -> None:
    # A fill_ recorded inside CUDA-graph capture bakes the capture-time
    # length; replay would ignore later writes. Skip and let the caller
    # write the tensor before replay.
    if _is_cuda_capturing():
        return
    with torch.inference_mode(False):
        state["valid_q"].fill_(int(q_len))
        state["valid_kv"].fill_(int(kv_len))


def _resolve_flex_valid(orig_t: int, bucket: int) -> int:
    """True valid length: do not treat on-bucket padded q as fully valid."""
    if orig_t < bucket:
        if _MASK_VALID_LEN is not None and 0 < _MASK_VALID_LEN < orig_t:
            return int(_MASK_VALID_LEN)
        return orig_t
    if _MASK_VALID_LEN is not None and 0 < _MASK_VALID_LEN <= orig_t:
        return int(_MASK_VALID_LEN)
    if _LAST_PROCESSED_LEN is not None and 0 < _LAST_PROCESSED_LEN <= orig_t:
        return int(_LAST_PROCESSED_LEN)
    return orig_t


def _cached_block_mask(
    bucket: int,
    valid: int,
    q: torch.Tensor,
    *,
    batch: int | None = None,
    heads: int | None = None,
) -> Any:
    """Full-bucket BlockMask; pad tail is masked via the live valid tensor.

    Structure is always created with valid=bucket so a short first request
    cannot mark tail blocks empty for a later longer request. ``valid`` is
    written into the captured tensors after create / cache hit.
    """
    create = _create_block_mask_impl()
    if create is None or not isinstance(q, torch.Tensor):
        return None
    device = q.device
    state = _mask_state(device)
    if batch is None:
        batch = int(q.shape[0])
    if heads is None and q.ndim >= 4:
        heads = int(q.shape[1])
    cache_key = (int(bucket), int(batch), heads)
    mask = state["masks"].get(cache_key)
    if mask is None:
        # Materialize every block as present; mask_mod applies at runtime.
        _set_mask_valid_len(state, bucket, bucket)
        kwargs: dict[str, Any] = {"device": device}
        with contextlib.suppress(TypeError):
            kwargs["separate_full_blocks"] = False
        try:
            mask = create(state["mask_mod"], batch, heads, bucket, bucket, **kwargs)
        except Exception:
            mask = None
        if mask is not None:
            state["masks"][cache_key] = mask
    _set_mask_valid_len(state, valid, valid)
    return mask


def _pad_flex_qkv(tensor: torch.Tensor, target: int) -> torch.Tensor:
    cur = tensor.shape[-2]
    if target <= cur:
        return tensor if tensor.is_contiguous() else tensor.contiguous()
    return torch.nn.functional.pad(tensor, (0, 0, 0, target - cur)).contiguous()


def _crop_flex_output(out: Any, orig_t: int) -> Any:
    if isinstance(out, tuple):
        if not out:
            return out
        head, *rest = out
        if isinstance(head, torch.Tensor) and head.shape[-2] > orig_t:
            return (head[..., :orig_t, :], *rest)
        return out
    if isinstance(out, torch.Tensor) and out.shape[-2] > orig_t:
        return out[..., :orig_t, :]
    return out


def _flex_qkv_from_call(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, Any, Any]:
    q = kwargs["query"] if "query" in kwargs else (args[0] if args else None)
    k = kwargs["key"] if "key" in kwargs else (args[1] if len(args) > 1 else None)
    v = kwargs["value"] if "value" in kwargs else (args[2] if len(args) > 2 else None)
    return q, k, v


def _set_flex_qkv(
    args: list[Any],
    kwargs: dict[str, Any],
    q: Any,
    k: Any,
    v: Any,
) -> None:
    if "query" in kwargs:
        kwargs["query"] = q
    elif args:
        args[0] = q
    if "key" in kwargs:
        kwargs["key"] = k
    elif len(args) > 1:
        args[1] = k
    if "value" in kwargs:
        kwargs["value"] = v
    elif len(args) > 2:
        args[2] = v


def _create_block_mask_impl() -> Any:
    if _UNWRAPPED_CREATE_BLOCK_MASK is not None:
        return _UNWRAPPED_CREATE_BLOCK_MASK
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        return None
    return create_block_mask


def _detach_flex_tensor(tensor: Any) -> Any:
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor


def _as_inference_tensor(tensor: Any) -> Any:
    """New inference tensor so dynamo sees CUDA, not AutogradCUDA."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.requires_grad:
        tensor = tensor.detach()
    if _tensor_is_inference(tensor):
        return tensor
    with torch.inference_mode():
        return tensor.clone()


def _promote_to_inference(
    args: tuple[Any, ...] | list[Any], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args_out = tuple(_as_inference_tensor(arg) for arg in args)
    kwargs_out = {key: _as_inference_tensor(value) for key, value in kwargs.items()}
    return args_out, kwargs_out


def _should_use_eager_flex(*tensors: Any) -> bool:
    """Eager FORCE_USE when the call does not match compiled serving.

    The NeMo timestamp aligner re-runs the encoder outside inference_mode
    with ordinary (not requires_grad) AutogradCUDA tensors. Routing on
    ``requires_grad`` alone never fires. While dynamo is tracing a
    serving layer, stay on the compiled path (never ``set_stance``).
    """
    if _is_torch_compiling():
        return False
    if _ALIGNER_ACTIVE:
        return True
    tensor_list = [t for t in tensors if isinstance(t, torch.Tensor)]
    if not tensor_list:
        return False
    if not _is_inference_mode():
        return True
    for tensor in tensor_list:
        if tensor.requires_grad:
            return True
        if not _tensor_is_inference(tensor):
            return True
    return False


@contextlib.contextmanager
def _aligner_runtime():
    """Match serving dispatch keys: inference, no grad, eager-flex flag."""
    global _ALIGNER_ACTIVE, _MASK_VALID_LEN, _LAST_PROCESSED_LEN
    prev = _ALIGNER_ACTIVE
    prev_valid = _MASK_VALID_LEN
    prev_processed = _LAST_PROCESSED_LEN
    _ALIGNER_ACTIVE = True
    # Do not let a prior serving valid clip the aligner's own T.
    _MASK_VALID_LEN = None
    _LAST_PROCESSED_LEN = None
    already = _is_inference_mode()
    try:
        if already:
            yield
            return
        with torch.inference_mode(), torch.no_grad():
            yield
    finally:
        _ALIGNER_ACTIVE = prev
        _MASK_VALID_LEN = prev_valid
        _LAST_PROCESSED_LEN = prev_processed


@contextlib.contextmanager
def _speech_encoder_runtime():
    """Aligner / any non-inference caller: same inference_mode as serving."""
    if _is_inference_mode():
        yield
        return
    with _aligner_runtime():
        yield


def _wrap_create_block_mask(create_mask: Any) -> Any:
    """Force CUDA + tensor-stable valid length on ``create_block_mask``."""
    global _UNWRAPPED_CREATE_BLOCK_MASK
    if create_mask is None or getattr(create_mask, _FLEX_ISOLATED_ATTR, False):
        return create_mask
    _UNWRAPPED_CREATE_BLOCK_MASK = create_mask

    def wrapped(
        mask_mod: Any,
        B: Any,
        H: Any,
        Q_LEN: Any,
        KV_LEN: Any,
        *args: Any,
        **kwargs: Any,
    ):
        if kwargs.get("device") is None:
            with contextlib.suppress(Exception):
                kwargs["device"] = torch.device("cuda")
        q_len = _as_pos_int(Q_LEN)
        kv_len = _as_pos_int(KV_LEN)
        if q_len is None or kv_len is None or q_len <= 0 or kv_len <= 0:
            return create_mask(mask_mod, B, H, Q_LEN, KV_LEN, *args, **kwargs)
        device = kwargs.get("device")
        true_valid = min(q_len, kv_len)
        global _MASK_VALID_LEN
        _MASK_VALID_LEN = true_valid
        # Keep the caller's seq_lengths. Bucket masks are created only
        # after the flex wrapper pads q/k/v to the same size.
        state = _mask_state(device)
        _set_mask_valid_len(state, true_valid, true_valid)
        kwargs = dict(kwargs)
        with contextlib.suppress(TypeError):
            kwargs["separate_full_blocks"] = False
        return create_mask(state["mask_mod"], B, H, Q_LEN, KV_LEN, *args, **kwargs)

    dynamo = getattr(torch, "_dynamo", None)
    disable = getattr(dynamo, "disable", None) if dynamo is not None else None
    if callable(disable):
        with contextlib.suppress(Exception):
            wrapped = disable(wrapped, recursive=False)
    setattr(wrapped, _FLEX_ISOLATED_ATTR, True)
    return wrapped


def _call_flex_with_static_bucket(
    fn: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Pad flex q/k/v + block_mask to a warmed bucket, then crop.

    Must not run under dynamo: pad / contiguous / mask happen in eager so
    the compiled flex kernel only sees a closed bucket. Aligner / any
    non-inference call uses uncompiled FORCE_USE flex after promoting
    q/k/v to inference tensors. Valid length is never taken from the
    padded q shape when processed_len / create_block_mask Q_LEN is
    smaller (on-bucket pad leak).
    """
    _graph_break_if_compiling()
    kwargs = dict(kwargs)
    args_list = list(args)
    q, k, v = _flex_qkv_from_call(tuple(args_list), kwargs)
    use_eager = _should_use_eager_flex(q, k, v)
    if use_eager:
        q = _as_inference_tensor(q)
        k = _as_inference_tensor(k)
        v = _as_inference_tensor(v)
        _set_flex_qkv(args_list, kwargs, q, k, v)
    eager = _EAGER_FLEX or fn
    call_fn = eager if use_eager else fn
    opts = dict(kwargs.get("kernel_options") or {})
    opts["FORCE_USE_FLEX_ATTENTION"] = True
    kwargs["kernel_options"] = opts
    if not isinstance(q, torch.Tensor) or q.ndim < 3:
        return fn(*args_list, **kwargs)

    def _pad_mask_and_call(
        q: Any, k: Any, v: Any, args_list: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        q = _detach_flex_tensor(q)
        k = _detach_flex_tensor(k)
        v = _detach_flex_tensor(v)
        # Transpose from (B, T, H, D) leaves contig=False; dynamo guards
        # that stride unless we materialize before the compiled flex call.
        if isinstance(q, torch.Tensor) and not q.is_contiguous():
            q = q.contiguous()
        if isinstance(k, torch.Tensor) and not k.is_contiguous():
            k = k.contiguous()
        if isinstance(v, torch.Tensor) and not v.is_contiguous():
            v = v.contiguous()
        orig_t = int(q.shape[-2])
        # Never grow to an incoming mask's seq_lengths (aligner 4000-frame leak).
        bucket = speech_encoder_flex_bucket(orig_t)
        valid = _resolve_flex_valid(orig_t, bucket)
        if bucket > orig_t:
            q = _pad_flex_qkv(q, bucket)
            if isinstance(k, torch.Tensor):
                k = _pad_flex_qkv(k, bucket)
            if isinstance(v, torch.Tensor):
                v = _pad_flex_qkv(v, bucket)
        q = q.contiguous()
        if isinstance(k, torch.Tensor):
            k = k.contiguous()
        if isinstance(v, torch.Tensor):
            v = v.contiguous()
        new_mask = _cached_block_mask(bucket, valid, q)
        if new_mask is not None:
            kwargs["block_mask"] = new_mask
        _record_flex_input(q, kwargs.get("block_mask"), orig_t=orig_t, bucket=bucket)
        _set_flex_qkv(args_list, kwargs, q, k, v)
        dtype = (
            q.dtype if isinstance(q, torch.Tensor) and q.is_floating_point() else None
        )
        with _pin_default_dtype(dtype):
            if use_eager:
                out = _disable_dynamo_fn(call_fn)(*args_list, **kwargs)
            else:
                out = call_fn(*args_list, **kwargs)
        if bucket > orig_t:
            return _crop_flex_output(out, orig_t)
        return out

    if use_eager:
        with _aligner_runtime(), _force_eager_compute():
            return _pad_mask_and_call(q, k, v, args_list, kwargs)
    return _pad_mask_and_call(q, k, v, args_list, kwargs)


_call_flex_with_static_bucket = _disable_dynamo_fn(_call_flex_with_static_bucket)


def _pad_flex_wrapper(fn: Any) -> Any:
    """Pad flex inputs outside the compiled region (no T guard)."""
    if fn is None or getattr(fn, _FLEX_PADDED_ATTR, False):
        return fn

    def wrapped(*args: Any, **kwargs: Any):
        return _call_flex_with_static_bucket(fn, args, kwargs)

    with contextlib.suppress(Exception):
        wrapped = functools.update_wrapper(wrapped, fn)
    wrapped = _disable_dynamo_fn(wrapped)
    setattr(wrapped, _FLEX_PADDED_ATTR, True)
    setattr(wrapped, _FLEX_ISOLATED_ATTR, True)
    return wrapped


def _force_flex_attention_kernel(fn: Any) -> Any:
    """Wrap ``fn`` so inductor cannot pick flex-decoding for short Q."""
    if fn is None or getattr(fn, _FLEX_ISOLATED_ATTR, False):
        return fn

    def wrapped(*args: Any, **kwargs: Any):
        opts = dict(kwargs.get("kernel_options") or {})
        opts["FORCE_USE_FLEX_ATTENTION"] = True
        kwargs["kernel_options"] = opts
        return fn(*args, **kwargs)

    with contextlib.suppress(Exception):
        wrapped = functools.update_wrapper(wrapped, fn)
    setattr(wrapped, _FLEX_ISOLATED_ATTR, True)
    return wrapped


def _set_module_attr(module: Any, name: str, value: Any) -> None:
    setattr(module, name, value)


def _patch_get_flex_attention(getter: Any) -> Any:
    if getter is None or getattr(getter, _FLEX_ISOLATED_ATTR, False):
        return getter

    def wrapped(q: Any, *args: Any, **kwargs: Any):
        fn = getter(q, *args, **kwargs)
        if getattr(fn, _BI_FLEX_COMPILED_ATTR, False) or getattr(
            fn, _FLEX_PADDED_ATTR, False
        ):
            return fn
        return _pad_flex_wrapper(_force_flex_attention_kernel(fn))

    with contextlib.suppress(Exception):
        wrapped = functools.update_wrapper(wrapped, getter)
    setattr(wrapped, _FLEX_ISOLATED_ATTR, True)
    return wrapped


def _patch_flex_decoding_guard() -> None:
    """Skip inductor's ``Q < 128`` guard when the main kernel is forced.

    ``_use_flex_decoding`` evaluates ``guard_or_false(Q < 128)`` *before*
    checking ``FORCE_USE_FLEX_ATTENTION``, so the 128-frame guard is
    still installed and dynamo recompiles around that boundary.
    """
    try:
        from torch._inductor.kernel.flex import flex_decoding as flex_dec
    except ImportError:
        return
    orig = getattr(flex_dec, "_use_flex_decoding", None)
    if orig is None or getattr(orig, _FLEX_ISOLATED_ATTR, False):
        return

    def wrapped(
        query: Any,
        kv_indices: Any,
        value: Any,
        kernel_options: Any,
        enable_gqa: bool = False,
    ) -> bool:
        opts = kernel_options or {}
        if opts.get("FORCE_USE_FLEX_ATTENTION") or opts.get("BACKEND") == "TRITON":
            return False
        return orig(query, kv_indices, value, opts, enable_gqa)

    with contextlib.suppress(Exception):
        wrapped = functools.update_wrapper(wrapped, orig)
    setattr(wrapped, _FLEX_ISOLATED_ATTR, True)
    flex_dec._use_flex_decoding = wrapped
    with contextlib.suppress(Exception):
        from torch._inductor.kernel.flex import flex_attention as flex_attn

        if getattr(flex_attn, "_use_flex_decoding", None) is orig:
            flex_attn._use_flex_decoding = wrapped


def isolate_flex_attention_from_compile() -> bool:
    """Force the non-decoding flex kernel on NeMo's compiled wrapper.

    NeMo calls ``transformer_encoder_utils._get_flex_attention(q)``, which
    returns import-time ``_flex_attention_compiled = torch.compile(
    flex_attention, dynamic=True)``. Graph-breaking that wrapper is
    correct (short-clip WER ~3.1%) but ~6× slower. Injecting
    ``FORCE_USE_FLEX_ATTENTION`` keeps flex inside the inductor graph on
    the main kernel. Inductor's ``_use_flex_decoding`` 128-frame guard
    is short-circuited so crossing 128 frames does not recompile.
    q/k/v and ``create_block_mask`` are padded onto static buckets
    outside that compiled region so last-window remainders cannot
    re-specialize. ``mask_mod`` reads valid length from a captured
    tensor, not a Python int. Any encoder call outside
    ``inference_mode`` (the timestamp aligner) is promoted to
    inference tensors and uses uncompiled FORCE_USE flex so
    AutogradCUDA / dtype guards cannot recompile into dense attention.

    Always re-scans loaded modules so a late NeMo import is still
    patched.

    Returns:
        True when the global op was wrapped, or already wrapped.

    """
    _patch_flex_decoding_guard()
    try:
        import torch.nn.attention.flex_attention as flex_mod
    except ImportError:
        flex_mod = None
        fn = None
    else:
        fn = getattr(flex_mod, "flex_attention", None)
    wrapped_global = False
    if fn is not None:
        global _UNWRAPPED_FLEX, _EAGER_FLEX
        if _UNWRAPPED_FLEX is None:
            _UNWRAPPED_FLEX = fn
        source = _UNWRAPPED_FLEX
        isolated = _force_flex_attention_kernel(source)
        _EAGER_FLEX = _disable_dynamo_fn(isolated)
        public = _pad_flex_wrapper(isolated)
        if public is not fn:
            flex_mod.flex_attention = public
            for module in list(sys.modules.values()):
                if module is None:
                    continue
                try:
                    if getattr(module, "flex_attention", None) is fn:
                        module.flex_attention = public  # type: ignore[attr-defined]
                except Exception:
                    continue
        wrapped_global = True
        create_mask = getattr(flex_mod, "create_block_mask", None)
        isolated_mask = _wrap_create_block_mask(create_mask)
        if isolated_mask is not None and isolated_mask is not create_mask:
            flex_mod.create_block_mask = isolated_mask
            for module in list(sys.modules.values()):
                if module is None:
                    continue
                try:
                    if getattr(module, "create_block_mask", None) is create_mask:
                        module.create_block_mask = (  # type: ignore[attr-defined]
                            isolated_mask
                        )
                except Exception:
                    continue

    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            compiled = getattr(module, "_flex_attention_compiled", None)
            if compiled is not None:
                _set_module_attr(
                    module,
                    "_flex_attention_compiled",
                    _pad_flex_wrapper(_force_flex_attention_kernel(compiled)),
                )
            compiled_alt = getattr(module, "flex_attention_compiled", None)
            if compiled_alt is not None:
                _set_module_attr(
                    module,
                    "flex_attention_compiled",
                    _pad_flex_wrapper(_force_flex_attention_kernel(compiled_alt)),
                )
            getter = getattr(module, "_get_flex_attention", None)
            if callable(getter):
                _set_module_attr(
                    module, "_get_flex_attention", _patch_get_flex_attention(getter)
                )
        except Exception:
            continue
    _replace_nemo_flex_compile_under_bi()
    return wrapped_global


def speech_encoder_inductor_config() -> dict[str, Any]:
    """Inductor flags that make compiled encoder kernels restart-stable.

    Flex lowering always calls ``autotune_select_algorithm`` and times
    BLOCK_M/N candidates unless ``config.deterministic``. GEMM
    ``max_autotune`` / ``coordinate_descent_tuning`` do the same.
    Different process starts can pick different tiles, so bf16
    reduction trees diverge and long-window transcripts drift.
    ``TORCHINDUCTOR_DETERMINISTIC=1`` alone is reset by the first
    compiled frame (pytorch#198563); set the config keys instead.
    """
    if not envs.VLLM_BATCH_INVARIANT:
        return {}
    config: dict[str, Any] = {
        "deterministic": True,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    try:
        from torch._inductor import config as inductor_config
    except ImportError:
        return config
    if hasattr(inductor_config, "batch_invariant"):
        config["batch_invariant"] = True
    return config


def _apply_speech_encoder_inductor_pins() -> dict[str, Any]:
    """Pin process-level inductor settings under BI. Idempotent."""
    global _INDUCTOR_PINNED
    config = speech_encoder_inductor_config()
    if not config:
        return config
    try:
        from torch._inductor import config as inductor_config
    except ImportError:
        return config
    changed = False
    for name, value in config.items():
        if not hasattr(inductor_config, name):
            continue
        if getattr(inductor_config, name) != value:
            setattr(inductor_config, name, value)
            changed = True
    if changed and not _INDUCTOR_PINNED:
        logger.info(
            "Pinned speech-encoder inductor config under VLLM_BATCH_INVARIANT (%s).",
            ", ".join(f"{k}={v}" for k, v in config.items()),
        )
    _INDUCTOR_PINNED = True
    return config


def _pin_speech_encoder_inductor():
    """Keep inductor pins active around compile and first forwards."""
    config = _apply_speech_encoder_inductor_pins()
    if not config:
        return contextlib.nullcontext()
    try:
        from torch._inductor import config as inductor_config
    except ImportError:
        return contextlib.nullcontext()
    patch = {
        key: value for key, value in config.items() if hasattr(inductor_config, key)
    }
    return inductor_config.patch(**patch)


def _compile_flex_attention_under_bi(fn: Any) -> Any:
    """Recompile NeMo's flex wrapper with deterministic inductor options."""
    global _BI_FLEX_COMPILED
    if fn is None or getattr(fn, _BI_FLEX_COMPILED_ATTR, False):
        return fn
    if _BI_FLEX_COMPILED is not None:
        return _BI_FLEX_COMPILED
    options = speech_encoder_inductor_config()
    if not options:
        return fn
    source = _UNWRAPPED_FLEX or fn
    isolated = _force_flex_attention_kernel(source)
    global _EAGER_FLEX
    _EAGER_FLEX = _disable_dynamo_fn(isolated)
    compiled = torch.compile(isolated, dynamic=False, options=options)
    with contextlib.suppress(Exception):
        compiled = functools.update_wrapper(compiled, isolated)
    padded = _pad_flex_wrapper(compiled)
    setattr(padded, _BI_FLEX_COMPILED_ATTR, True)
    setattr(padded, _FLEX_ISOLATED_ATTR, True)
    _BI_FLEX_COMPILED = padded
    return padded


def _replace_nemo_flex_compile_under_bi() -> None:
    """Replace import-time ``torch.compile(flex)`` under BI.

    That wrapper has no inductor options, so first invocation would
    still time flex tiles even after ``layer.compile`` is pinned.
    """
    if not envs.VLLM_BATCH_INVARIANT:
        return
    try:
        import torch.nn.attention.flex_attention as flex_mod
    except ImportError:
        return
    source = _UNWRAPPED_FLEX or getattr(flex_mod, "flex_attention", None)
    if source is None:
        return
    compiled = _compile_flex_attention_under_bi(source)
    if compiled is source:
        return
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            current = getattr(module, "_flex_attention_compiled", None)
            if current is not None and not getattr(
                current, _BI_FLEX_COMPILED_ATTR, False
            ):
                _set_module_attr(module, "_flex_attention_compiled", compiled)
            current_alt = getattr(module, "flex_attention_compiled", None)
            if current_alt is not None and not getattr(
                current_alt, _BI_FLEX_COMPILED_ATTR, False
            ):
                _set_module_attr(module, "flex_attention_compiled", compiled)
        except Exception:
            continue


def _install_speech_encoder_recompile_watch() -> None:
    """Log dynamo recompiles after warmup (they change encoder numerics)."""
    global _RECOMPILE_HOOK_INSTALLED
    if _RECOMPILE_HOOK_INSTALLED:
        return
    try:
        from torch._dynamo.utils import register_hook_for_recompile_user_context
    except ImportError:
        return

    def hook() -> str:
        return _recompile_watch_context()

    register_hook_for_recompile_user_context(hook)
    _RECOMPILE_HOOK_INSTALLED = True


def _set_serving_watch(active: bool) -> None:
    global _SERVING_WATCH
    _SERVING_WATCH = bool(active)


def configure_speech_encoder_dynamo() -> None:
    """Raise dynamo recompile limits and pin inductor under BI."""
    _apply_speech_encoder_inductor_pins()
    _replace_nemo_flex_compile_under_bi()
    _install_speech_encoder_recompile_watch()
    global _DYNAMO_CONFIGURED
    if _DYNAMO_CONFIGURED:
        return
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None or not hasattr(dynamo, "config"):
        return
    config = dynamo.config
    for name in (
        "recompile_limit",
        "cache_size_limit",
        "accumulated_recompile_limit",
    ):
        if not hasattr(config, name):
            continue
        current = getattr(config, name)
        if isinstance(current, int) and current < _SPEECH_ENCODER_RECOMPILE_LIMIT:
            setattr(config, name, _SPEECH_ENCODER_RECOMPILE_LIMIT)
    _DYNAMO_CONFIGURED = True


def speech_encoder_layer_compile_kwargs() -> dict[str, Any]:
    """Kwargs for ``nn.Module.compile`` on speech-encoder blocks.

    Always inductor + dynamic shapes on the layer. Flex is compiled
    separately with ``dynamic=False`` after q/k/v and the block mask
    are padded onto a closed bucket set. ``FORCE_USE_FLEX_ATTENTION``
    keeps short audio off flex-decoding. Do not use
    ``backend="eager"``: dynamo then captures the flex HOP as dense
    math (~5× slower). Under BI, pass inductor ``options`` so timing
    autotune cannot pick different tiles after a restart.
    """
    kwargs: dict[str, Any] = {"dynamic": True}
    inductor = speech_encoder_inductor_config()
    if inductor:
        kwargs["options"] = inductor
    return kwargs


def collect_speech_encoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return top-level transformer blocks of ``perception.encoder``.

    Matches the NeMo Speech plugin's ``EncoderGraphRunner`` compile loop
    (``perception.encoder.named_modules()`` lists named ``layers``), but
    skips nested ``*.layers.*.layers`` so a block is not compiled again
    as a child of an already-selected parent. ``perception.proj`` /
    ``modality_adapter`` stay eager. Sortformer is mounted as
    ``perception.encoder``, not a top-level ``pe_encoder`` attribute.
    """
    getter = getattr(model, "get_mm_encoder_compile_modules", None)
    if callable(getter):
        return [module for module in getter() if isinstance(module, nn.Module)]

    perception = getattr(model, "perception", None)
    if not isinstance(perception, nn.Module):
        return []
    encoder = getattr(perception, "encoder", None)
    if not isinstance(encoder, nn.Module):
        return []

    layers: list[nn.Module] = []
    seen: set[int] = set()
    for name, module in encoder.named_modules():
        if not (isinstance(module, nn.ModuleList) and name.endswith("layers")):
            continue
        # "layers" or "asr_encoder.layers", not "layers.0.ffn.layers".
        if "layers" in name.split(".")[:-1]:
            continue
        for layer in module:
            ident = id(layer)
            if ident not in seen:
                seen.add(ident)
                layers.append(layer)
    return layers


def _module_floating_dtype(module: nn.Module) -> torch.dtype | None:
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    return None


def _pin_default_dtype(dtype: torch.dtype | None):
    """Pin ``default_dtype`` so dynamo does not recompile on GLOBAL_STATE."""
    if dtype is None or not dtype.is_floating_point:
        return contextlib.nullcontext()

    class _Pin:
        def __enter__(self):
            self.prev = torch.get_default_dtype()
            if self.prev != dtype:
                torch.set_default_dtype(dtype)
            return self

        def __exit__(self, *exc):
            if torch.get_default_dtype() != self.prev:
                torch.set_default_dtype(self.prev)
            return False

    return _Pin()


def remember_speech_encoder_default_dtype(
    model: Any = None, dtype: torch.dtype | None = None
) -> torch.dtype | None:
    """Record the compile-time default dtype for later encoder forwards."""
    global _ENCODER_DEFAULT_DTYPE
    if dtype is not None and dtype.is_floating_point:
        _ENCODER_DEFAULT_DTYPE = dtype
        return _ENCODER_DEFAULT_DTYPE
    if model is not None:
        inner = unwrap_speech_encoder_model(model)
        if isinstance(inner, nn.Module):
            found = _module_floating_dtype(inner)
            if found is not None:
                _ENCODER_DEFAULT_DTYPE = found
    return _ENCODER_DEFAULT_DTYPE


@contextlib.contextmanager
def speech_encoder_global_state(dtype: torch.dtype | None = None):
    """Match request-time GLOBAL_STATE to compile/warmup (default_dtype)."""
    pin = dtype if dtype is not None else _ENCODER_DEFAULT_DTYPE
    with _pin_default_dtype(pin):
        yield


def _encoder_dtype_runtime():
    return speech_encoder_global_state(_ENCODER_DEFAULT_DTYPE)


def _crop_to_valid_audio(
    audio_signal: torch.Tensor, audio_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop waveforms to the longest valid sample so path = f(item)."""
    if not isinstance(audio_signal, torch.Tensor):
        return audio_signal, audio_lengths
    if not isinstance(audio_lengths, torch.Tensor) or audio_lengths.numel() == 0:
        return audio_signal, audio_lengths
    valid = int(audio_lengths.max().item())
    if valid >= 0 and audio_signal.shape[-1] > valid:
        audio_signal = audio_signal[..., :valid]
    return audio_signal, audio_lengths


def _crop_processed(
    processed: torch.Tensor, processed_len: torch.Tensor
) -> torch.Tensor:
    """Crop encoder features to the longest valid frame."""
    if not isinstance(processed, torch.Tensor) or processed.ndim < 3:
        return processed
    if not isinstance(processed_len, torch.Tensor) or processed_len.numel() == 0:
        return processed
    valid = int(processed_len.max().item())
    if 0 < valid < processed.shape[2]:
        return processed[:, :, :valid]
    return processed


def speech_encoder_served_frames(
    valid: int,
    runner: Any = None,
    *,
    window: int | None = None,
    max_windows: int | None = None,
    bucket_step: int | None = None,
    long: bool | None = None,
) -> int:
    """Pad ``valid`` frames onto the closed compile / graph set.

    Whole-forward uses the 256-frame buckets. The long-window path
    pads the last window to ``window`` so remainder lengths cannot
    trigger a serving-time recompile or lazy last-width capture.
    """
    if valid <= 0:
        return valid
    if window is None:
        window = _encoder_window(runner)
    if max_windows is None:
        if runner is not None:
            max_windows = int(
                getattr(runner, "max_windows", None) or _DEFAULT_MAX_WINDOWS
            )
        else:
            max_windows = _DEFAULT_MAX_WINDOWS
    if bucket_step is None:
        bucket_step = _frame_bucket_step(runner)
    bucket_step = max(int(bucket_step), 1)
    if window is not None:
        win = int(window)
        if long is None:
            long = valid > int(max_windows) * win
        if long:
            return ((valid + win - 1) // win) * win
    return ((valid + bucket_step - 1) // bucket_step) * bucket_step


def _pad_processed(processed: torch.Tensor, target_frames: int) -> torch.Tensor:
    if not isinstance(processed, torch.Tensor) or processed.ndim < 3:
        return processed
    frames = processed.shape[2]
    if target_frames <= frames:
        return processed
    return torch.nn.functional.pad(processed, (0, target_frames - frames))


def _mark_processed_frames_dynamic(processed: torch.Tensor) -> torch.Tensor:
    if not isinstance(processed, torch.Tensor) or processed.ndim < 3:
        return processed
    mark = getattr(getattr(torch, "_dynamo", None), "mark_dynamic", None)
    if mark is None:
        return processed
    with contextlib.suppress(Exception):
        mark(processed, 2)
    return processed


def _prepare_processed(
    processed: torch.Tensor,
    processed_len: torch.Tensor,
    runner: Any,
    *,
    long: bool,
) -> torch.Tensor:
    """Crop to the item, then pad to the closed served shape."""
    processed = _crop_processed(processed, processed_len)
    if not isinstance(processed, torch.Tensor) or processed.ndim < 3:
        return processed
    valid = processed.shape[2]
    if isinstance(processed_len, torch.Tensor) and processed_len.numel() > 0:
        valid = int(processed_len.max().item())
    target = speech_encoder_served_frames(valid, runner, long=long)
    processed = _pad_processed(processed, target)
    return _mark_processed_frames_dynamic(processed)


def _processed_valid_len(
    processed: torch.Tensor, processed_len: torch.Tensor
) -> int | None:
    if isinstance(processed_len, torch.Tensor) and processed_len.numel() > 0:
        return int(processed_len.max().item())
    if isinstance(processed, torch.Tensor) and processed.ndim >= 3:
        return int(processed.shape[2])
    return None


def _prime_mask_from_processed(
    processed: torch.Tensor, processed_len: torch.Tensor
) -> None:
    """Write valid length before a graphed/compiled encoder call."""
    valid = _processed_valid_len(processed, processed_len)
    _begin_encoder_mask(valid)
    if valid is None:
        return
    device = processed.device if isinstance(processed, torch.Tensor) else None
    speech_encoder_set_mask_valid(device, valid, valid)


def _prime_mask_from_call_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Best-effort valid-length write from ``encoder.forward`` arguments."""
    processed = kwargs.get("audio_signal", kwargs.get("processed"))
    processed_len = kwargs.get("audio_lengths", kwargs.get("processed_len"))
    if processed is None:
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim >= 2:
                processed = arg
                break
    if processed_len is None:
        for arg in args:
            if arg is processed:
                continue
            if isinstance(arg, torch.Tensor) and arg.ndim <= 1:
                processed_len = arg
                break
    if isinstance(processed, torch.Tensor):
        _prime_mask_from_processed(processed, processed_len)


def _wrap_speech_encoder_inference(model: Any) -> bool:
    """Run perception.encoder under inference_mode (aligner call site)."""
    inner = unwrap_speech_encoder_model(model) if model is not None else None
    if inner is None:
        inner = model
    perception = getattr(inner, "perception", None)
    encoder = getattr(perception, "encoder", None)
    if encoder is None or getattr(encoder, _ENCODER_WRAP_ATTR, False):
        return False
    orig = getattr(encoder, "forward", None)
    if not callable(orig):
        return False

    def wrapped(*args: Any, **kwargs: Any):
        with _encoder_dtype_runtime(), _speech_encoder_runtime():
            args, kwargs = _promote_to_inference(args, kwargs)
            _prime_mask_from_call_args(args, kwargs)
            if _ALIGNER_ACTIVE:
                with _force_eager_compute():
                    return _disable_dynamo_fn(orig)(*args, **kwargs)
            return orig(*args, **kwargs)

    with contextlib.suppress(Exception):
        wrapped = functools.update_wrapper(wrapped, orig)
    encoder.forward = wrapped
    setattr(encoder, _ENCODER_WRAP_ATTR, True)
    return True


def _call_graph_runner_per_item(
    call: Any,
    audio_signal: torch.Tensor,
    audio_lengths: torch.Tensor,
    return_ctc_timestamp_inputs: bool,
) -> Any:
    """Encode each row alone so the graph/eager path cannot see mates."""
    outs: list[Any] = []
    for i in range(audio_signal.shape[0]):
        row, length = _crop_to_valid_audio(
            audio_signal[i : i + 1], audio_lengths[i : i + 1]
        )
        result = call(
            row,
            length,
            return_ctc_timestamp_inputs,
        )
        if result is None:
            return None
        if return_ctc_timestamp_inputs:
            return None
        embs = result[0] if isinstance(result, tuple) else result
        outs.extend(embs if isinstance(embs, list) else [embs])
    return outs


def _capture_one_graph(
    runner: Any, rows: int, frames: int, n_feat: int, dtype: torch.dtype
) -> None:
    configure_speech_encoder_dynamo()
    if _SERVING_WATCH:
        logger.warning(
            "Speech encoder serving-time CUDA graph capture: "
            "%d rows x %d frames (closed set should prevent this)",
            rows,
            frames,
        )
    # Dummy is a full bucket; write valid before capture so the graph
    # loads that tensor instead of recording a fill_ of this length.
    _begin_encoder_mask(frames)
    with contextlib.suppress(Exception):
        speech_encoder_set_mask_valid(torch.device("cuda"), frames, frames)
    with (
        _pin_default_dtype(dtype if dtype.is_floating_point else None),
        _pin_speech_encoder_inductor(),
    ):
        runner._capture(rows, frames, n_feat, dtype)


def configure_speech_encoder_graphs(
    model: nn.Module, vllm_config: VllmConfig | None = None
) -> bool:
    """Patch the plugin graph runner: defer, lazy capture, item-pure path.

    Safe on dummy objects that are not an ``EncoderGraphRunner``.
    Returns True when the runner was patched.
    """
    runner = getattr(model, "_encoder_graphs", None)
    if runner is None or getattr(runner, _RUNNER_PATCH_ATTR, False):
        return False
    required = (
        "_capture_or_disable",
        "_call_short",
        "_capture_all",
        "_capture",
        "_frame_bucket",
        "_split",
        "graphs",
    )
    if any(not hasattr(runner, name) for name in required):
        return False

    set_speech_encoder_shape_config(runner, model=model)
    sizes = default_speech_encoder_graph_batch_sizes(vllm_config)
    current = tuple(int(b) for b in getattr(runner, "batch_sizes", ()) or ())
    hf_config = None
    if vllm_config is not None and getattr(vllm_config, "model_config", None):
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
    user_sizes = getattr(hf_config, "encoder_cuda_graph_batch_sizes", None)
    if user_sizes is None and current and current != sizes:
        runner.batch_sizes = list(sizes)
        if hasattr(runner, "max_rows_x_frames") and hasattr(runner, "window"):
            runner.max_rows_x_frames = sizes[-1] * int(runner.window)

    runner._vllm_allow_capture = False
    orig_call_short = runner._call_short
    orig_call_long = getattr(runner, "_call_long", None)
    orig_cls_call = type(runner).__call__

    def capture_or_disable(n_feat: int, dtype: torch.dtype) -> bool:
        if not getattr(runner, "_vllm_allow_capture", False):
            return False
        if getattr(runner, "pool", None) is None:
            with contextlib.suppress(Exception):
                runner.pool = torch.cuda.graph_pool_handle()
        # Do not call _capture_all: capture the used (batch, bucket) later.
        return True

    def call_short(
        processed: torch.Tensor,
        processed_len: torch.Tensor,
        return_ctc_timestamp_inputs: bool,
    ):
        with _encoder_dtype_runtime():
            return _call_short_impl(
                processed, processed_len, return_ctc_timestamp_inputs
            )

    def _call_short_impl(
        processed: torch.Tensor,
        processed_len: torch.Tensor,
        return_ctc_timestamp_inputs: bool,
    ):
        processed = _prepare_processed(processed, processed_len, runner, long=False)
        _record_encoder_path("short", processed, processed_len)
        # Timestamps add a CTC aligner *after* transcription. The
        # serving encode (inference_mode) must stay on the same
        # compiled / graphed path as plain. Eager-route only the
        # post-transcription aligner (not inference_mode).
        if return_ctc_timestamp_inputs and not _is_inference_mode():
            with _aligner_runtime(), _force_eager_compute():
                _prime_mask_from_processed(processed, processed_len)
                return _disable_dynamo_fn(orig_call_short)(
                    processed, processed_len, return_ctc_timestamp_inputs
                )
        _prime_mask_from_processed(processed, processed_len)
        frames = processed.shape[2]
        bucket = runner._frame_bucket(frames)
        n_feat = processed.shape[1]
        closed = getattr(runner, "_vllm_closed_shapes", False)
        for size in runner._split(processed.shape[0], bucket):
            if (size, bucket) not in runner.graphs:
                if closed or not getattr(runner, "_vllm_allow_capture", False):
                    if closed:
                        logger.error(
                            "Speech encoder missing closed-set graph "
                            "%d rows x %d frames; not capturing at serve time",
                            size,
                            bucket,
                        )
                    raise SpeechEncoderGraphFallback(
                        "speech encoder graph capture is not enabled yet"
                    )
                try:
                    _capture_one_graph(runner, size, bucket, n_feat, processed.dtype)
                except SpeechEncoderGraphFallback:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Speech encoder CUDA graph capture failed for "
                        "%d rows x %d frames; encoding eagerly: %s",
                        size,
                        bucket,
                        exc,
                    )
                    if not closed:
                        runner.graphs.clear()
                        runner.enabled = False
                    raise SpeechEncoderGraphFallback(str(exc)) from exc
        return orig_call_short(processed, processed_len, return_ctc_timestamp_inputs)

    def call_long(
        processed: torch.Tensor,
        processed_len: torch.Tensor,
        return_ctc_timestamp_inputs: bool,
    ):
        with _encoder_dtype_runtime():
            processed = _prepare_processed(processed, processed_len, runner, long=True)
            _record_encoder_path("long", processed, processed_len)
            assert orig_call_long is not None
            if return_ctc_timestamp_inputs and not _is_inference_mode():
                with _aligner_runtime(), _force_eager_compute():
                    _prime_mask_from_processed(processed, processed_len)
                    return _disable_dynamo_fn(orig_call_long)(
                        processed, processed_len, return_ctc_timestamp_inputs
                    )
            _prime_mask_from_processed(processed, processed_len)
            return orig_call_long(processed, processed_len, return_ctc_timestamp_inputs)

    cls = type(runner)
    if not getattr(cls, "_vllm_speech_graph_call_wrapped", False):

        def cls_call(
            self,
            audio_signal: torch.Tensor,
            audio_lengths: torch.Tensor,
            return_ctc_timestamp_inputs: bool = False,
        ):
            if not getattr(self, _RUNNER_PATCH_ATTR, False):
                return orig_cls_call(
                    self, audio_signal, audio_lengths, return_ctc_timestamp_inputs
                )
            if not getattr(self, "enabled", True):
                return None
            if not self.graphs and not getattr(self, "_vllm_allow_capture", False):
                return None
            audio_signal, audio_lengths = _crop_to_valid_audio(
                audio_signal, audio_lengths
            )
            try:
                if (
                    envs.VLLM_BATCH_INVARIANT
                    and isinstance(audio_signal, torch.Tensor)
                    and audio_signal.shape[0] > 1
                    and not return_ctc_timestamp_inputs
                ):
                    return _call_graph_runner_per_item(
                        lambda audio, length, ctc: orig_cls_call(
                            self, audio, length, ctc
                        ),
                        audio_signal,
                        audio_lengths,
                        return_ctc_timestamp_inputs,
                    )
                return orig_cls_call(
                    self, audio_signal, audio_lengths, return_ctc_timestamp_inputs
                )
            except SpeechEncoderGraphFallback:
                return None

        cls.__call__ = cls_call
        cls._vllm_speech_graph_call_wrapped = True

    runner._capture_or_disable = capture_or_disable
    runner._call_short = call_short
    if orig_call_long is not None:
        runner._call_long = call_long
    setattr(runner, _RUNNER_PATCH_ATTR, True)
    logger.info(
        "Speech encoder CUDA graphs will capture lazily after KV "
        "allocation (batch_sizes=%s).",
        list(getattr(runner, "batch_sizes", sizes)),
    )
    return True


def enable_speech_encoder_graph_capture(model: nn.Module | None) -> None:
    """Allow lazy graph capture now that KV memory is reserved."""
    if model is None:
        return
    runner = getattr(model, "_encoder_graphs", None)
    if runner is None or not getattr(runner, _RUNNER_PATCH_ATTR, False):
        return
    runner._vllm_allow_capture = True
    logger.info_once("Speech encoder CUDA graphs enabled for lazy per-bucket capture.")


def _compile_speech_encoder_layers(model: nn.Module, layers: list[nn.Module]) -> int:
    configure_speech_encoder_dynamo()
    compile_kwargs = speech_encoder_layer_compile_kwargs()
    compiled = 0
    dtype = _module_floating_dtype(model)
    with _pin_default_dtype(dtype), _pin_speech_encoder_inductor():
        for layer in layers:
            if getattr(layer, _COMPILED_ATTR, False):
                continue
            try:
                layer.compile(**compile_kwargs)
            except Exception as exc:
                logger.warning(
                    "Failed to compile %s; encoder stays eager for that block: %s",
                    type(layer).__name__,
                    exc,
                )
                continue
            setattr(layer, _COMPILED_ATTR, True)
            compiled += 1
    if compiled:
        extra = ""
        if compile_kwargs.get("options"):
            extra = ", inductor deterministic (no timing autotune)"
        logger.info(
            "Compiled %d speech-encoder layer(s) with dynamic layer "
            "shapes and static flex buckets (compile_mm_encoder=True, "
            "backend=inductor, FORCE_USE_FLEX_ATTENTION%s).",
            compiled,
            extra,
        )
    return compiled


def compile_pending_speech_encoder_layers(model: nn.Module | None) -> int:
    """Compile layers deferred past memory profiling."""
    if model is None:
        return 0
    isolate_flex_attention_from_compile()
    configure_speech_encoder_dynamo()
    layers = _pop_pending_layers(model)
    if not layers:
        return 0
    compiled = _compile_speech_encoder_layers(
        unwrap_speech_encoder_model(model), layers
    )
    if compiled:
        with contextlib.suppress(Exception):
            torch.accelerator.empty_cache()
    return compiled


def _audio_cap_frames() -> int:
    duration_s = int(getattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 600) or 600)
    return max(1, int(duration_s / _ENCODER_FRAME_HOP_S))


def _frame_bucket_step(runner: Any) -> int:
    bucket_fn = getattr(runner, "_frame_bucket", None)
    if callable(bucket_fn):
        with contextlib.suppress(Exception):
            step = int(bucket_fn(1))
            if step > 0:
                return step
    raw = getattr(runner, "frame_bucket", None)
    if raw:
        return max(int(raw), 1)
    return _DEFAULT_FRAME_BUCKET


def _whole_forward_max_frames(runner: Any) -> int | None:
    if runner is None:
        return None
    for name in ("max_frames", "max_seq_len"):
        raw = getattr(runner, name, None)
        if raw:
            return int(raw)
    window = getattr(runner, "window", None)
    if not window:
        return None
    windows = getattr(runner, "max_windows", None) or _DEFAULT_MAX_WINDOWS
    return int(window) * int(windows)


def speech_encoder_warmup_frame_buckets(runner: Any = None) -> list[int]:
    """256-frame buckets covering whole-forward graphs up to the audio cap."""
    step = _frame_bucket_step(runner)
    cap = _audio_cap_frames()
    limit = cap
    max_fwd = _whole_forward_max_frames(runner)
    if max_fwd is not None:
        limit = min(limit, max_fwd)
    buckets: list[int] = []
    n = step
    while n < limit:
        buckets.append(n)
        n += step
    last = ((max(limit, 1) + step - 1) // step) * step
    if last not in buckets:
        buckets.append(last)
    return buckets


_CHUNK_ATTRS = (
    "chunk_size",
    "chunk_length",
    "window_size",
    "att_window",
    "streaming_chunk_size",
    "asr_chunk_size",
    "att_context_size",
)


def _chunk_from_obj(obj: Any) -> int | None:
    if obj is None:
        return None
    for name in _CHUNK_ATTRS:
        alt = getattr(obj, name, None)
        if isinstance(alt, (list, tuple)) and alt:
            alt = alt[0]
        n = _as_pos_int(alt)
        if n is not None and 8 < n < 2048:
            return n
    return None


def _encoder_window(runner: Any, model: Any = None) -> int | None:
    """Sliding encoder chunk. ``runner.window`` is often ``max_frames``."""
    raw = _as_pos_int(getattr(runner, "window", None) if runner else None)
    max_f = _as_pos_int(getattr(runner, "max_frames", None) if runner else None)
    objs: list[Any] = [runner]
    if runner is not None:
        for name in ("encoder", "asr_encoder", "model", "perception"):
            objs.append(getattr(runner, name, None))
    if model is not None:
        inner = unwrap_speech_encoder_model(model)
        perception = getattr(inner, "perception", None)
        encoder = getattr(perception, "encoder", None)
        objs.extend(
            (
                inner,
                perception,
                encoder,
                getattr(encoder, "asr_encoder", None) if encoder else None,
            )
        )
    for obj in objs:
        chunk = _chunk_from_obj(obj)
        if chunk is None:
            continue
        if raw is None or raw > 2 * chunk or (max_f is not None and raw == max_f):
            return chunk
    if raw is not None and raw >= 2048:
        return None
    if raw is not None and max_f is not None and raw == max_f and raw > 1100:
        return None
    if raw:
        return raw
    return None


def speech_encoder_warmup_long_frames(
    runner: Any = None, model: Any = None
) -> list[int]:
    """One 3-window dummy for long-audio window-graph warmup (BI batch 1).

    Flex q/k/v and ``create_block_mask`` pad last-window remainders to
    ``window``, so extra ``n_windows * window`` captures only inflate
    reused-cache ready time.
    """
    set_speech_encoder_shape_config(runner, model=model)
    window = _encoder_window(runner, model)
    if window is None:
        return []
    short_windows = int(getattr(runner, "max_windows", None) or _DEFAULT_MAX_WINDOWS)
    n_windows = max(_LONG_WARMUP_MIN_WINDOWS, short_windows + 1)
    length = n_windows * window
    if length > _audio_cap_frames():
        return []
    return [length]


def speech_encoder_profile_dummy_frames(
    runner: Any = None, model: Any = None
) -> list[int]:
    """Processed lengths ``profile_run`` still forwards after graph warmup.

    The plugin dummy is ``(1, n_feat, 3 * max_frames)`` when
    ``max_frames`` is the 4000-frame cap (12000). That exceeds the
    audio-duration cap, so the 3-window warmup never saw it.
    """
    set_speech_encoder_shape_config(runner, model=model)
    max_f = _as_pos_int(getattr(runner, "max_frames", None) if runner else None)
    if max_f is None or max_f < 2048:
        return []
    dummy = 3 * max_f
    warmed = set(speech_encoder_warmup_long_frames(runner, model))
    if dummy in warmed:
        return []
    return [dummy]


def _int_feat(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        n = int(value)
        return n if n > 0 else None
    return None


def _feat_from_obj(obj: Any) -> int | None:
    if obj is None:
        return None
    for name in _FEAT_ATTRS:
        n = _int_feat(getattr(obj, name, None))
        if n is not None:
            return n
    for cfg_name in ("cfg", "config", "_cfg"):
        cfg = getattr(obj, cfg_name, None)
        if cfg is None or cfg is obj:
            continue
        n = _feat_from_obj(cfg)
        if n is not None:
            return n
    return None


def speech_encoder_n_feat(model: Any = None, runner: Any = None) -> int | None:
    """Mel / feat_in width from the perception preprocessor or encoder.

    The plugin's ``_capture`` builds ``(batch, n_feat, n_frame)``. This
    checkpoint's preprocessor emits 128 mels; a hardcoded 80-mel default
    makes every warmup capture fail and then disables window graphs.
    """
    inner = unwrap_speech_encoder_model(model) if model is not None else None
    perception = getattr(inner, "perception", None)
    encoder = getattr(perception, "encoder", None)
    preprocessor = None
    if perception is not None:
        preprocessor = getattr(perception, "preprocessor", None) or getattr(
            perception, "preproc", None
        )
    for obj in (
        preprocessor,
        getattr(preprocessor, "featurizer", None) if preprocessor is not None else None,
        encoder,
        runner,
        getattr(runner, "encoder", None) if runner is not None else None,
        getattr(runner, "preprocessor", None) if runner is not None else None,
        perception,
        inner,
        getattr(inner, "hf_config", None) if inner is not None else None,
    ):
        n = _feat_from_obj(obj)
        if n is not None:
            return n
        if obj is None:
            continue
        for child_name in ("preprocessor", "encoder"):
            n = _feat_from_obj(getattr(obj, child_name, None))
            if n is not None:
                return n
    return None


@dataclass
class SpeechEncoderWarmupStats:
    """Result of ``warmup_speech_encoder`` (graphs only; compile-only skips)."""

    elapsed_s: float = 0.0
    captured: int = 0
    long_captured: int = 0
    failed: int = 0


def _warmup_graphs(runner: Any, model: Any) -> SpeechEncoderWarmupStats:
    stats = SpeechEncoderWarmupStats()
    if runner is None or not getattr(runner, _RUNNER_PATCH_ATTR, False):
        return stats
    if not getattr(runner, "_vllm_allow_capture", False):
        return stats
    if getattr(runner, "pool", None) is None:
        with contextlib.suppress(Exception):
            runner.pool = torch.cuda.graph_pool_handle()
    inner = unwrap_speech_encoder_model(model)
    n_feat = speech_encoder_n_feat(inner, runner)
    if n_feat is None:
        logger.error(
            "Speech encoder warmup captured 0 graph(s): could not resolve "
            "n_feat from perception.preprocessor features / encoder feat_in. "
            "Refusing the 80-mel default (this checkpoint uses 128)."
        )
        stats.failed = 1
        return stats
    dtype = _module_floating_dtype(inner if isinstance(inner, nn.Module) else model)
    if dtype is None:
        dtype = torch.float32
    batches = [
        int(b)
        for b in (
            getattr(runner, "batch_sizes", None)
            or default_speech_encoder_graph_batch_sizes()
        )
    ]
    logger.info("Speech encoder warmup capturing with n_feat=%d", n_feat)
    for rows in batches:
        for frames in speech_encoder_warmup_frame_buckets(runner):
            if (rows, frames) in getattr(runner, "graphs", {}):
                continue
            try:
                _capture_one_graph(runner, rows, frames, n_feat, dtype)
                stats.captured += 1
            except Exception as exc:
                stats.failed += 1
                logger.error(
                    "Speech encoder warmup capture failed for %d rows x %d "
                    "frames (n_feat=%d): %s",
                    rows,
                    frames,
                    n_feat,
                    exc,
                )
    _warmup_long_graphs(runner, inner, n_feat, dtype, stats)
    if stats.captured == 0 and stats.long_captured == 0:
        logger.error(
            "Speech encoder warmup captured 0 graph(s) (%d failure(s)). "
            "Window / long-audio graphs must not be disabled; check n_feat.",
            stats.failed,
        )
    return stats


def _model_device(model: Any) -> torch.device:
    inner = unwrap_speech_encoder_model(model)
    if isinstance(inner, nn.Module):
        for param in inner.parameters():
            return param.device
    return torch.device("cpu")


def _warmup_long_graphs(
    runner: Any,
    model: Any,
    n_feat: int,
    dtype: torch.dtype,
    stats: SpeechEncoderWarmupStats,
) -> None:
    """Capture long-audio window graphs via ``_call_long`` (BI batch 1)."""
    call_long = getattr(runner, "_call_long", None)
    if not callable(call_long):
        return
    lengths = list(speech_encoder_warmup_long_frames(runner, model))
    for extra in speech_encoder_profile_dummy_frames(runner, model):
        if extra not in lengths:
            lengths.append(extra)
    if not lengths:
        return
    window = _encoder_window(runner, model)
    device = _model_device(model)
    logger.info(
        "Speech encoder warmup capturing long-audio window graphs "
        "(%d shape(s), window=%s, BI batch 1).",
        len(lengths),
        window,
    )
    for frames in lengths:
        processed = torch.zeros(1, n_feat, frames, device=device, dtype=dtype)
        processed_len = torch.tensor([frames], device=device, dtype=torch.int64)
        try:
            call_long(processed, processed_len, False)
            stats.long_captured += 1
            stats.captured += 1
        except Exception as exc:
            stats.failed += 1
            logger.error(
                "Speech encoder long-window warmup failed for 1 x %d frames "
                "(n_feat=%d): %s",
                frames,
                n_feat,
                exc,
            )
    if lengths and window and (1, window) not in getattr(runner, "graphs", {}):
        try:
            _capture_one_graph(runner, 1, window, n_feat, dtype)
            stats.long_captured += 1
            stats.captured += 1
        except Exception as exc:
            stats.failed += 1
            logger.error(
                "Speech encoder window-graph warmup failed for 1 x %d frames "
                "(n_feat=%d): %s",
                window,
                n_feat,
                exc,
            )


def warmup_speech_encoder(model: nn.Module | None) -> SpeechEncoderWarmupStats:
    """Capture graph buckets after KV. Compile-only does not dummy-encode.

    Under ``VLLM_BATCH_INVARIANT`` the whole-forward grid is batch 1 ×
    frame buckets, then ``_call_long`` warms one ``3 * window`` dummy.
    Flex pads last-window remainders to ``window``. ``layer.compile``
    uses dynamic shapes, so compile-only skips this warmup.
    Must not run during memory profiling.

    Returns:
        Capture counts and elapsed seconds.

    """
    stats = SpeechEncoderWarmupStats()
    if model is None:
        return stats
    inner = unwrap_speech_encoder_model(model)
    runner = getattr(inner, "_encoder_graphs", None)
    want_graphs = (
        runner is not None
        and getattr(runner, _RUNNER_PATCH_ATTR, False)
        and getattr(runner, "_vllm_allow_capture", False)
    )
    if not want_graphs:
        return stats

    started = time.perf_counter()
    _set_serving_watch(False)
    set_speech_encoder_shape_config(runner, model=inner)
    with torch.inference_mode(), _pin_speech_encoder_inductor():
        stats = _warmup_graphs(runner, inner)
    stats.elapsed_s = time.perf_counter() - started
    if (
        envs.VLLM_BATCH_INVARIANT
        and runner is not None
        and getattr(runner, "graphs", None)
    ):
        runner._vllm_closed_shapes = True
        runner._vllm_allow_capture = False
        _set_serving_watch(True)
        logger.info(
            "Speech encoder closed compile/graph shapes after warmup "
            "(%d graph(s), %d long-window). Serving-time capture is off.",
            stats.captured,
            stats.long_captured,
        )
    logger.info(
        "Speech encoder warmup finished in %.2f s (captured %d graph(s), "
        "%d long-window, %d failure(s)).",
        stats.elapsed_s,
        stats.captured,
        stats.long_captured,
        stats.failed,
    )
    return stats


def enable_speech_encoder_after_kv(model: nn.Module | None) -> SpeechEncoderWarmupStats:
    """Compile deferred layers, enable capture, and warm graph buckets.

    Pins ``default_dtype`` across compile + capture so dynamo does not
    recompile on ``GLOBAL_STATE``. Runs after KV allocation so this is
    not counted as peak activation. Compile-only skips dummy encodes.
    """
    if model is None:
        return SpeechEncoderWarmupStats()
    inner = unwrap_speech_encoder_model(model)
    dtype = None
    if isinstance(inner, nn.Module):
        dtype = _module_floating_dtype(inner)
    remember_speech_encoder_default_dtype(inner, dtype)
    with _pin_default_dtype(dtype):
        compile_pending_speech_encoder_layers(model)
        enable_speech_encoder_graph_capture(model)
        _wrap_speech_encoder_inference(inner)
        return warmup_speech_encoder(model)


def maybe_compile_speech_mm_encoder(
    model: nn.Module, vllm_config: VllmConfig, *, defer: bool = False
) -> int:
    """Compile speech-encoder layers when ``compile_mm_encoder`` is on.

    Always prepares graph-runner patches (defer / lazy / item-pure path)
    when the plugin built ``EncoderGraphRunner``. Layer compile is not
    deferred to the plugin: vLLM compiles the same blocks after forcing
    the non-decoding flex kernel, including when graphs are on.

    When ``defer`` is True (model runners), ``layer.compile`` runs after
    KV allocation so inductor workspaces are not counted as peak
    activation.

    Returns the number of layers compiled now (0 when deferred).
    """
    configure_speech_encoder_graphs(model, vllm_config)
    remember_speech_encoder_default_dtype(model)
    _wrap_speech_encoder_inference(model)
    want_compile = bool(vllm_config.compilation_config.compile_mm_encoder)
    if want_compile or getattr(model, "_encoder_graphs", None) is not None:
        isolate_flex_attention_from_compile()
        configure_speech_encoder_dynamo()
        _wrap_speech_encoder_inference(model)

    if not want_compile:
        return 0

    inner = unwrap_speech_encoder_model(model)
    layers = collect_speech_encoder_layers(
        inner if isinstance(inner, nn.Module) else model
    )
    if not layers:
        return 0
    if defer:
        _store_pending_layers(model, layers)
        logger.info(
            "Deferring compile of %d speech-encoder layer(s) until after "
            "KV allocation.",
            len(layers),
        )
        return 0
    return _compile_speech_encoder_layers(
        inner if isinstance(inner, nn.Module) else model, layers
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile plugin / speech multimodal encoders under ``compile_mm_encoder``.

vLLM already compiles in-tree vision encoders via ``@support_torch_compile``
and ``should_torch_compile_mm_encoder``. The NeMo SpeechLM plugin's
perception encoder is built outside that decorator. This module compiles
``perception.encoder`` transformer blocks (the same set as the plugin's
``EncoderGraphRunner``) with dynamic shapes so variable audio lengths do
not re-specialize, and it maps the official compilation flags onto the
plugin's ``encoder_compile`` / ``encoder_cuda_graphs`` hf-overrides.

Inductor's flex-decoding kernel is wrong for this encoder's rel-pos
``score_mod`` and padding ``block_mask`` when ``Q`` is shorter than 128
frames. Flex stays inside the compiled graph with
``FORCE_USE_FLEX_ATTENTION`` so the main flex kernel is used. Graph-
breaking the NeMo ``_flex_attention_compiled`` wrapper is correct but
~6× slower. vLLM owns layer compile so the plugin cannot
``compile_layers`` first under graphs.

Speech encoder CUDA graphs are captured lazily after KV allocation, with
batch sizes ``(1,)`` under ``VLLM_BATCH_INVARIANT``. The replay path is a
function of each item's valid length, not the padded tensor width.
Layer compile is also deferred until after KV so inductor workspaces are
not counted as peak activation.
"""

from __future__ import annotations

import contextlib
import functools
import sys
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
_RUNNER_PATCH_ATTR = "_vllm_speech_graph_patched"
_PENDING_LAYERS_ATTR = "_vllm_speech_encoder_pending_compile"
_DYNAMO_CONFIGURED = False

_NEMOTRON_SPEECH_ARCH_MARKERS = ("NeMoSpeechLM",)
_NEMOTRON_SPEECH_MODEL_TYPES = ("nemo_speechlm",)

# Windowed Sortformer / long-audio forwards hit more shape classes than
# dynamo's default ``recompile_limit`` (8). Past the limit, compiled
# blocks silently run eager code.
_SPEECH_ENCODER_RECOMPILE_LIMIT = 64
# A100 80GB + ~63GB weights: a 16-row grid left ~1.1 GiB for KV.
_MAX_ENCODER_GRAPH_BATCH = 8


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
        return _force_flex_attention_kernel(getter(q, *args, **kwargs))

    with contextlib.suppress(Exception):
        wrapped = functools.update_wrapper(wrapped, getter)
    setattr(wrapped, _FLEX_ISOLATED_ATTR, True)
    return wrapped


def isolate_flex_attention_from_compile() -> bool:
    """Force the non-decoding flex kernel on NeMo's compiled wrapper.

    NeMo calls ``transformer_encoder_utils._get_flex_attention(q)``, which
    returns import-time ``_flex_attention_compiled = torch.compile(
    flex_attention, dynamic=True)``. Graph-breaking that wrapper is
    correct (short-clip WER ~3.1%) but ~6× slower. Injecting
    ``FORCE_USE_FLEX_ATTENTION`` keeps flex inside the inductor graph on
    the main kernel.

    Always re-scans loaded modules so a late NeMo import is still
    patched.

    Returns:
        True when the global op was wrapped, or already wrapped.

    """
    try:
        import torch.nn.attention.flex_attention as flex_mod
    except ImportError:
        flex_mod = None
        fn = None
    else:
        fn = getattr(flex_mod, "flex_attention", None)
    wrapped_global = False
    if fn is not None:
        isolated = _force_flex_attention_kernel(fn)
        if isolated is not fn:
            flex_mod.flex_attention = isolated
            for module in list(sys.modules.values()):
                if module is None:
                    continue
                try:
                    if getattr(module, "flex_attention", None) is fn:
                        module.flex_attention = isolated  # type: ignore[attr-defined]
                except Exception:
                    continue
        wrapped_global = True
        create_mask = getattr(flex_mod, "create_block_mask", None)
        if create_mask is not None and not getattr(
            create_mask, _FLEX_ISOLATED_ATTR, False
        ):

            def _create_mask_on_gpu(*args: Any, **kwargs: Any):
                if kwargs.get("device") is None:
                    with contextlib.suppress(Exception):
                        kwargs["device"] = torch.device("cuda")
                return create_mask(*args, **kwargs)

            setattr(_create_mask_on_gpu, _FLEX_ISOLATED_ATTR, True)
            flex_mod.create_block_mask = _create_mask_on_gpu
            for module in list(sys.modules.values()):
                if module is None:
                    continue
                try:
                    if getattr(module, "create_block_mask", None) is create_mask:
                        module.create_block_mask = (  # type: ignore[attr-defined]
                            _create_mask_on_gpu
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
                    _force_flex_attention_kernel(compiled),
                )
            compiled_alt = getattr(module, "flex_attention_compiled", None)
            if compiled_alt is not None:
                _set_module_attr(
                    module,
                    "flex_attention_compiled",
                    _force_flex_attention_kernel(compiled_alt),
                )
            getter = getattr(module, "_get_flex_attention", None)
            if callable(getter):
                _set_module_attr(
                    module, "_get_flex_attention", _patch_get_flex_attention(getter)
                )
        except Exception:
            continue
    return wrapped_global


def configure_speech_encoder_dynamo() -> None:
    """Raise dynamo recompile limits for variable encoder window shapes."""
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

    Always inductor + dynamic shapes. Flex stays in the graph with
    ``FORCE_USE_FLEX_ATTENTION`` so short audio does not take
    flex-decoding. Do not use ``backend="eager"``: dynamo then
    captures the flex HOP as dense math (~5× slower).
    """
    return {"dynamic": True}


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
    with _pin_default_dtype(dtype if dtype.is_floating_point else None):
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
        processed = _crop_processed(processed, processed_len)
        frames = processed.shape[2]
        bucket = runner._frame_bucket(frames)
        n_feat = processed.shape[1]
        for size in runner._split(processed.shape[0], bucket):
            if (size, bucket) not in runner.graphs:
                if not getattr(runner, "_vllm_allow_capture", False):
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
                    runner.graphs.clear()
                    runner.enabled = False
                    raise SpeechEncoderGraphFallback(str(exc)) from exc
        return orig_call_short(processed, processed_len, return_ctc_timestamp_inputs)

    def call_long(
        processed: torch.Tensor,
        processed_len: torch.Tensor,
        return_ctc_timestamp_inputs: bool,
    ):
        processed = _crop_processed(processed, processed_len)
        assert orig_call_long is not None
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
    compile_kwargs = speech_encoder_layer_compile_kwargs()
    compiled = 0
    dtype = _module_floating_dtype(model)
    with _pin_default_dtype(dtype):
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
        logger.info(
            "Compiled %d speech-encoder layer(s) with dynamic shapes "
            "(compile_mm_encoder=True, backend=inductor, "
            "FORCE_USE_FLEX_ATTENTION).",
            compiled,
        )
    return compiled


def compile_pending_speech_encoder_layers(model: nn.Module | None) -> int:
    """Compile layers deferred past memory profiling."""
    if model is None:
        return 0
    isolate_flex_attention_from_compile()
    layers = getattr(model, _PENDING_LAYERS_ATTR, None)
    if not layers:
        return 0
    delattr(model, _PENDING_LAYERS_ATTR)
    compiled = _compile_speech_encoder_layers(model, layers)
    if compiled:
        with contextlib.suppress(Exception):
            torch.accelerator.empty_cache()
    return compiled


def enable_speech_encoder_after_kv(model: nn.Module | None) -> None:
    """Compile deferred layers and allow lazy graph capture after KV."""
    compile_pending_speech_encoder_layers(model)
    enable_speech_encoder_graph_capture(model)


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
    want_compile = bool(vllm_config.compilation_config.compile_mm_encoder)
    if want_compile or getattr(model, "_encoder_graphs", None) is not None:
        isolate_flex_attention_from_compile()
        configure_speech_encoder_dynamo()

    if not want_compile:
        return 0

    layers = collect_speech_encoder_layers(model)
    if not layers:
        return 0
    if defer:
        setattr(model, _PENDING_LAYERS_ATTR, layers)
        logger.info(
            "Deferring compile of %d speech-encoder layer(s) until after "
            "KV allocation.",
            len(layers),
        )
        return 0
    return _compile_speech_encoder_layers(model, layers)

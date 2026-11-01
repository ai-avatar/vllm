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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as nn

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_COMPILED_ATTR = "_vllm_mm_encoder_compiled"

_NEMOTRON_SPEECH_ARCH_MARKERS = ("NeMoSpeechLM",)
_NEMOTRON_SPEECH_MODEL_TYPES = ("nemo_speechlm",)


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


def sync_mm_encoder_compile_flags(vllm_config: VllmConfig) -> None:
    """Align compilation_config, env vars, and NeMo Speech hf-overrides.

    Sources (any one enables the corresponding flag):

    * ``compilation_config.compile_mm_encoder`` /
      ``compilation_config.cudagraph_mm_encoder``
    * ``VLLM_COMPILE_MM_ENCODER`` / ``VLLM_CUDAGRAPH_MM_ENCODER``
    * hf-overrides ``encoder_compile`` / ``encoder_cuda_graphs``

    All three graph sources are equivalent, including under
    ``VLLM_BATCH_INVARIANT``. The effective value is written back to both
    ``compilation_config.cudagraph_mm_encoder`` and (for SpeechLM)
    ``hf_config.encoder_cuda_graphs`` so they match what will run.
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

    if compilation.compile_mm_encoder:
        hf_config.encoder_compile = True

    want_graphs = bool(compilation.cudagraph_mm_encoder)
    compilation.cudagraph_mm_encoder = want_graphs
    hf_config.encoder_cuda_graphs = want_graphs
    if want_graphs:
        logger.warning_once(
            "Speech encoder CUDA graphs pad to length buckets and mask "
            "by length. VLLM_BATCH_INVARIANT still encodes one item at a "
            "time. Confirm embeddings vs eager if you enable graphs."
        )


def speech_encoder_layer_compile_kwargs() -> dict[str, Any]:
    """Kwargs for ``nn.Module.compile`` on speech-encoder blocks.

    Under ``VLLM_BATCH_INVARIANT``, use the eager backend. Inductor GEMM
    templates emit their own Triton mm and skip the ``aten::mm`` /
    ``aten::addmm`` library impls that pin SM80 batch-invariant matmul.
    """
    kwargs: dict[str, Any] = {"dynamic": True}
    if envs.VLLM_BATCH_INVARIANT:
        kwargs["backend"] = "eager"
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


def maybe_compile_speech_mm_encoder(model: nn.Module, vllm_config: VllmConfig) -> int:
    """Compile speech-encoder layers when ``compile_mm_encoder`` is on.

    Returns the number of layers compiled. No-op when the flag is off, the
    plugin already built ``EncoderGraphRunner`` (it compiles those layers
    itself), or ``perception.encoder`` has no layer lists.

    CUDA graphs are not captured here. Vision models keep using
    ``EncoderCudaGraphManager`` + ``SupportsEncoderCudaGraph``. SpeechLM
    graphs stay in the plugin's ``EncoderGraphRunner``.
    """
    if not vllm_config.compilation_config.compile_mm_encoder:
        return 0
    if getattr(model, "_encoder_graphs", None) is not None:
        logger.info(
            "Speech encoder CUDA graphs are active; layer compile is owned "
            "by EncoderGraphRunner."
        )
        return 0

    layers = collect_speech_encoder_layers(model)
    if not layers:
        return 0

    compile_kwargs = speech_encoder_layer_compile_kwargs()
    compiled = 0
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
        backend = compile_kwargs.get("backend", "inductor")
        logger.info(
            "Compiled %d speech-encoder layer(s) with dynamic shapes "
            "(compile_mm_encoder=True, backend=%s).",
            compiled,
            backend,
        )
    return compiled

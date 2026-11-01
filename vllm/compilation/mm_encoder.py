# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile plugin / speech multimodal encoders under ``compile_mm_encoder``.

vLLM already compiles in-tree vision encoders via ``@support_torch_compile``
and ``should_torch_compile_mm_encoder``. The NeMo SpeechLM plugin's
perception / Sortformer towers are built outside that decorator. This
module compiles their transformer blocks with dynamic shapes so variable
audio lengths do not re-specialize, and it maps the official compilation
flags onto the plugin's ``encoder_compile`` / ``encoder_cuda_graphs``
hf-overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    Official compilation flags are written back onto the SpeechLM hf config
    so the plugin's ``EncoderGraphRunner`` sees them. CUDA graphs stay off
    under batch-invariant mode unless the user set ``encoder_cuda_graphs``
    explicitly (graphs pad to length buckets).
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
    explicit_graphs = bool(getattr(hf_config, "encoder_cuda_graphs", False))
    if explicit_graphs:
        compilation.cudagraph_mm_encoder = True

    if not is_nemo_speechlm_config(hf_config):
        return

    if compilation.compile_mm_encoder:
        hf_config.encoder_compile = True

    if compilation.cudagraph_mm_encoder:
        if envs.VLLM_BATCH_INVARIANT and not explicit_graphs:
            logger.warning(
                "cudagraph_mm_encoder is ignored under VLLM_BATCH_INVARIANT "
                "unless hf-overrides set encoder_cuda_graphs=true. Compile "
                "still runs with dynamic shapes; graphs pad to length buckets."
            )
        else:
            hf_config.encoder_cuda_graphs = True


def collect_speech_encoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return transformer blocks under perception / pe_encoder.

    Matches the NeMo Speech plugin's ``EncoderGraphRunner`` compile loop:
    every ``nn.ModuleList`` whose name ends with ``layers``.
    """
    getter = getattr(model, "get_mm_encoder_compile_modules", None)
    if callable(getter):
        return [module for module in getter() if isinstance(module, nn.Module)]

    roots: list[nn.Module] = []
    for attr in ("perception", "pe_encoder"):
        root = getattr(model, attr, None)
        if isinstance(root, nn.Module):
            roots.append(root)

    layers: list[nn.Module] = []
    seen: set[int] = set()
    for root in roots:
        for name, module in root.named_modules():
            if isinstance(module, nn.ModuleList) and name.endswith("layers"):
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
    itself), or no perception / pe_encoder layers are found.

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

    compiled = 0
    for layer in layers:
        if getattr(layer, _COMPILED_ATTR, False):
            continue
        try:
            layer.compile(dynamic=True)
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
            "(compile_mm_encoder=True).",
            compiled,
        )
    return compiled

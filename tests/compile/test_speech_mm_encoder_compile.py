# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for SpeechLM / Nemotron Transcribe encoder compile wiring."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.mm_encoder import (
    collect_speech_encoder_layers,
    is_nemo_speechlm_config,
    maybe_compile_speech_mm_encoder,
    sync_mm_encoder_compile_flags,
)
from vllm.config.compilation import CompilationConfig

pytestmark = pytest.mark.cpu_test


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.compile_calls = 0

    def compile(self, *args, **kwargs):  # type: ignore[override]
        self.compile_calls += 1
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _Perception(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_Layer(), _Layer()])


class _SpeechModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.perception = _Perception()


def _vllm_config(*, compile_mm: bool = False, graphs: bool = False):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            compile_mm_encoder=compile_mm,
            cudagraph_mm_encoder=graphs,
        ),
        model_config=None,
    )


def test_compile_mm_encoder_defaults_off():
    assert not CompilationConfig().compile_mm_encoder
    assert not CompilationConfig().cudagraph_mm_encoder


def test_env_enables_compile_mm_encoder(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_COMPILE_MM_ENCODER", "1")
    monkeypatch.setattr(envs, "VLLM_COMPILE_MM_ENCODER", True)
    envs.disable_envs_cache()
    assert CompilationConfig().compile_mm_encoder
    envs.disable_envs_cache()


def test_is_nemo_speechlm_config():
    assert is_nemo_speechlm_config(
        SimpleNamespace(
            model_type="nemo_speechlm",
            architectures=["NeMoSpeechLMForConditionalGeneration"],
        )
    )
    assert not is_nemo_speechlm_config(
        SimpleNamespace(model_type="qwen2", architectures=["Qwen2ForCausalLM"])
    )


def test_sync_maps_hf_overrides_and_writes_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_COMPILE_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_CUDAGRAPH_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    hf = SimpleNamespace(
        model_type="nemo_speechlm",
        architectures=["NeMoSpeechLMForConditionalGeneration"],
        encoder_compile=True,
        encoder_cuda_graphs=False,
    )
    compilation = SimpleNamespace(compile_mm_encoder=False, cudagraph_mm_encoder=False)
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_config=hf),
    )
    sync_mm_encoder_compile_flags(vllm_config)
    assert compilation.compile_mm_encoder
    assert hf.encoder_compile is True
    envs.disable_envs_cache()


def test_explicit_graphs_allowed_under_bi(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_COMPILE_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_CUDAGRAPH_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    hf = SimpleNamespace(
        model_type="nemo_speechlm",
        architectures=["NeMoSpeechLMForConditionalGeneration"],
        encoder_cuda_graphs=True,
    )
    compilation = SimpleNamespace(compile_mm_encoder=False, cudagraph_mm_encoder=False)
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_config=hf),
    )
    sync_mm_encoder_compile_flags(vllm_config)
    assert compilation.cudagraph_mm_encoder
    assert hf.encoder_cuda_graphs is True
    envs.disable_envs_cache()


def test_sync_does_not_auto_enable_graphs_under_bi(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_COMPILE_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_CUDAGRAPH_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    hf = SimpleNamespace(
        model_type="nemo_speechlm",
        architectures=["NeMoSpeechLMForConditionalGeneration"],
    )
    compilation = SimpleNamespace(compile_mm_encoder=True, cudagraph_mm_encoder=True)
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_config=hf),
    )
    sync_mm_encoder_compile_flags(vllm_config)
    assert not getattr(hf, "encoder_cuda_graphs", False)
    assert hf.encoder_compile is True
    envs.disable_envs_cache()


def test_collects_perception_layers():
    model = _SpeechModel()
    layers = collect_speech_encoder_layers(model)
    assert len(layers) == 2
    assert all(isinstance(layer, _Layer) for layer in layers)


def test_maybe_compile_is_noop_when_disabled():
    model = _SpeechModel()
    compiled = maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=False))
    assert compiled == 0
    assert all(layer.compile_calls == 0 for layer in model.perception.encoder.layers)


def test_maybe_compile_compiles_each_layer_once():
    model = _SpeechModel()
    compiled = maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True))
    assert compiled == 2
    assert all(layer.compile_calls == 1 for layer in model.perception.encoder.layers)
    compiled_again = maybe_compile_speech_mm_encoder(
        model, _vllm_config(compile_mm=True)
    )
    assert compiled_again == 0
    assert all(layer.compile_calls == 1 for layer in model.perception.encoder.layers)


def test_maybe_compile_skips_when_plugin_graphs_own_compile():
    model = _SpeechModel()
    model._encoder_graphs = object()
    compiled = maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True))
    assert compiled == 0
    assert all(layer.compile_calls == 0 for layer in model.perception.encoder.layers)

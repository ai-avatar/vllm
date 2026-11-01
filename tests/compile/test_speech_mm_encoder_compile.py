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
    speech_encoder_layer_compile_kwargs,
    sync_mm_encoder_compile_flags,
)
from vllm.config.compilation import CompilationConfig

pytestmark = pytest.mark.cpu_test


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.compile_calls = 0
        self.compile_kwargs: dict = {}

    def compile(self, *args, **kwargs):  # type: ignore[override]
        self.compile_calls += 1
        self.compile_kwargs = dict(kwargs)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _NestedLayer(_Layer):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Module()
        self.inner.layers = nn.ModuleList([_Layer()])


class _Perception(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_NestedLayer(), _Layer()])
        self.encoder.asr_encoder = nn.Module()
        self.encoder.asr_encoder.layers = nn.ModuleList([_Layer()])
        self.modality_adapter = nn.Module()
        self.modality_adapter.layers = nn.ModuleList([_Layer()])
        self.proj = nn.Linear(4, 4)


class _SpeechModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.perception = _Perception()


def _speech_hf(**kwargs):
    values = {
        "model_type": "nemo_speechlm",
        "architectures": ["NeMoSpeechLMForConditionalGeneration"],
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _vllm_config(*, compile_mm: bool = False, graphs: bool = False, hf=None):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            compile_mm_encoder=compile_mm,
            cudagraph_mm_encoder=graphs,
        ),
        model_config=None if hf is None else SimpleNamespace(hf_config=hf),
    )


def test_compile_mm_encoder_defaults_off():
    assert not CompilationConfig().compile_mm_encoder
    assert not CompilationConfig().cudagraph_mm_encoder


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_env_accepts_truthy_compile_flag(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("VLLM_COMPILE_MM_ENCODER", value)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_COMPILE_MM_ENCODER
        assert CompilationConfig().compile_mm_encoder
    finally:
        envs.disable_envs_cache()


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
def test_env_accepts_falsy_compile_flag(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("VLLM_COMPILE_MM_ENCODER", value)
    envs.disable_envs_cache()
    try:
        assert not envs.VLLM_COMPILE_MM_ENCODER
    finally:
        envs.disable_envs_cache()


def test_env_rejects_invalid_compile_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_COMPILE_MM_ENCODER", "maybe")
    envs.disable_envs_cache()
    try:
        with pytest.raises(ValueError, match="VLLM_COMPILE_MM_ENCODER"):
            _ = envs.VLLM_COMPILE_MM_ENCODER
    finally:
        envs.disable_envs_cache()


def test_is_nemo_speechlm_config():
    assert is_nemo_speechlm_config(_speech_hf())
    assert not is_nemo_speechlm_config(
        SimpleNamespace(model_type="qwen2", architectures=["Qwen2ForCausalLM"])
    )


def test_sync_maps_hf_overrides_and_writes_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_COMPILE_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_CUDAGRAPH_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    hf = _speech_hf(encoder_compile=True, encoder_cuda_graphs=False)
    compilation = SimpleNamespace(compile_mm_encoder=False, cudagraph_mm_encoder=False)
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_config=hf),
    )
    sync_mm_encoder_compile_flags(vllm_config)
    assert compilation.compile_mm_encoder
    assert hf.encoder_compile is True
    assert compilation.cudagraph_mm_encoder is False
    assert hf.encoder_cuda_graphs is False
    envs.disable_envs_cache()


@pytest.mark.parametrize(
    "source",
    ["compilation_config", "env", "hf_overrides"],
)
def test_graph_sources_are_equivalent_under_bi(
    monkeypatch: pytest.MonkeyPatch, source: str
):
    monkeypatch.setattr(envs, "VLLM_COMPILE_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_CUDAGRAPH_MM_ENCODER", source == "env")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    hf = _speech_hf(encoder_cuda_graphs=(source == "hf_overrides"))
    compilation = SimpleNamespace(
        compile_mm_encoder=False,
        cudagraph_mm_encoder=(source == "compilation_config"),
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_config=hf),
    )
    sync_mm_encoder_compile_flags(vllm_config)
    assert compilation.cudagraph_mm_encoder is True
    assert hf.encoder_cuda_graphs is True
    envs.disable_envs_cache()


def test_sync_clears_inactive_graph_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_COMPILE_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_CUDAGRAPH_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    hf = _speech_hf()
    compilation = SimpleNamespace(compile_mm_encoder=True, cudagraph_mm_encoder=False)
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_config=hf),
    )
    sync_mm_encoder_compile_flags(vllm_config)
    assert compilation.cudagraph_mm_encoder is False
    assert hf.encoder_cuda_graphs is False
    assert hf.encoder_compile is True
    envs.disable_envs_cache()


def test_collects_only_top_level_encoder_layers():
    model = _SpeechModel()
    layers = collect_speech_encoder_layers(model)
    encoder = model.perception.encoder
    expected = [
        encoder.layers[0],
        encoder.layers[1],
        encoder.asr_encoder.layers[0],
    ]
    assert layers == expected
    assert encoder.layers[0].inner.layers[0] not in layers
    assert model.perception.modality_adapter.layers[0] not in layers
    assert not hasattr(model, "pe_encoder")


def test_maybe_compile_is_noop_when_disabled():
    model = _SpeechModel()
    compiled = maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=False))
    assert compiled == 0
    encoder = model.perception.encoder
    assert encoder.layers[0].compile_calls == 0
    assert encoder.layers[0].inner.layers[0].compile_calls == 0
    assert model.perception.modality_adapter.layers[0].compile_calls == 0


def test_maybe_compile_skips_adapter_and_nested_layers():
    model = _SpeechModel()
    compiled = maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True))
    assert compiled == 3
    encoder = model.perception.encoder
    assert encoder.layers[0].compile_calls == 1
    assert encoder.layers[1].compile_calls == 1
    assert encoder.asr_encoder.layers[0].compile_calls == 1
    assert encoder.layers[0].inner.layers[0].compile_calls == 0
    assert model.perception.modality_adapter.layers[0].compile_calls == 0
    compiled_again = maybe_compile_speech_mm_encoder(
        model, _vllm_config(compile_mm=True)
    )
    assert compiled_again == 0
    assert encoder.layers[0].compile_calls == 1


def test_maybe_compile_skips_when_plugin_graphs_own_compile():
    model = _SpeechModel()
    model._encoder_graphs = object()
    compiled = maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True))
    assert compiled == 0
    assert model.perception.encoder.layers[0].compile_calls == 0


def test_compile_kwargs_use_eager_backend_under_bi(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    assert speech_encoder_layer_compile_kwargs() == {"dynamic": True}

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    assert speech_encoder_layer_compile_kwargs() == {
        "dynamic": True,
        "backend": "eager",
    }
    model = _SpeechModel()
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True))
    assert model.perception.encoder.layers[0].compile_kwargs == {
        "dynamic": True,
        "backend": "eager",
    }
    envs.disable_envs_cache()

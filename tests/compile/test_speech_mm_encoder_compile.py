# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for SpeechLM / Nemotron Transcribe encoder compile wiring."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation import mm_encoder as mm_encoder_mod
from vllm.compilation.mm_encoder import (
    collect_speech_encoder_layers,
    configure_speech_encoder_dynamo,
    configure_speech_encoder_graphs,
    default_speech_encoder_graph_batch_sizes,
    enable_speech_encoder_graph_capture,
    is_nemo_speechlm_config,
    isolate_flex_attention_from_compile,
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
    # vLLM owns layer compile; the plugin must not compile_layers.
    assert hf.encoder_compile is False
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
    assert hf.encoder_compile is False
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


def test_maybe_compile_still_compiles_when_plugin_graphs_exist():
    model = _SpeechModel()
    model._encoder_graphs = object()
    compiled = maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True))
    assert compiled == 3
    assert model.perception.encoder.layers[0].compile_calls == 1
    assert model.perception.encoder.layers[0].compile_kwargs == {"dynamic": True}


def test_compile_kwargs_stay_inductor_under_bi(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    assert speech_encoder_layer_compile_kwargs() == {"dynamic": True}

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    assert speech_encoder_layer_compile_kwargs() == {"dynamic": True}
    model = _SpeechModel()
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True))
    assert model.perception.encoder.layers[0].compile_kwargs == {"dynamic": True}
    envs.disable_envs_cache()


def test_graph_batch_defaults_are_singleton_under_bi(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    assert default_speech_encoder_graph_batch_sizes() == (1,)
    hf = _speech_hf()
    compilation = SimpleNamespace(compile_mm_encoder=False, cudagraph_mm_encoder=True)
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_config=hf),
        scheduler_config=SimpleNamespace(max_num_seqs=192),
    )
    sync_mm_encoder_compile_flags(vllm_config)
    assert hf.encoder_cuda_graph_batch_sizes == [1]
    envs.disable_envs_cache()


def test_graph_batch_defaults_follow_max_num_seqs(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4))
    assert default_speech_encoder_graph_batch_sizes(vllm_config) == (1, 2, 4)
    vllm_config.scheduler_config.max_num_seqs = 192
    assert default_speech_encoder_graph_batch_sizes(vllm_config) == (
        1,
        2,
        4,
        8,
    )
    envs.disable_envs_cache()


def test_graph_batch_honors_explicit_hf_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_COMPILE_MM_ENCODER", False)
    monkeypatch.setattr(envs, "VLLM_CUDAGRAPH_MM_ENCODER", False)
    envs.disable_envs_cache()
    hf = _speech_hf(encoder_cuda_graphs=True, encoder_cuda_graph_batch_sizes=[1, 2])
    compilation = SimpleNamespace(compile_mm_encoder=False, cudagraph_mm_encoder=False)
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_config=hf),
    )
    sync_mm_encoder_compile_flags(vllm_config)
    assert hf.encoder_cuda_graph_batch_sizes == [1, 2]
    envs.disable_envs_cache()


def test_isolate_flex_attention_marks_the_op():
    assert isolate_flex_attention_from_compile()
    from torch.nn.attention.flex_attention import flex_attention

    assert getattr(flex_attention, mm_encoder_mod._FLEX_ISOLATED_ATTR)


def test_dynamo_recompile_limit_is_raised():
    configure_speech_encoder_dynamo()
    config = torch._dynamo.config
    limit = getattr(config, "recompile_limit", getattr(config, "cache_size_limit", 0))
    assert limit >= 64


class _FakeGraphRunner:
    def __init__(self) -> None:
        self.graphs: dict[tuple[int, int], object] = {}
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        self.window = 550
        self.max_rows_x_frames = 128 * 550
        self.enabled = True
        self.pool = None
        self.capture_all_calls = 0
        self.capture_calls: list[tuple[int, int]] = []
        self.short_frames: list[int] = []

    def _capture_all(self, n_feat: int, dtype: torch.dtype) -> None:
        self.capture_all_calls += 1

    def _capture_or_disable(self, n_feat: int, dtype: torch.dtype) -> bool:
        self._capture_all(n_feat, dtype)
        return True

    def _frame_bucket(self, frames: int) -> int:
        return ((frames + 255) // 256) * 256

    def _split(self, rows: int, frames: int) -> list[int]:
        return [rows]

    def _capture(self, rows: int, frames: int, n_feat: int, dtype: torch.dtype) -> None:
        self.capture_calls.append((rows, frames))
        self.graphs[rows, frames] = object()

    def _call_short(
        self,
        processed: torch.Tensor,
        processed_len: torch.Tensor,
        return_ctc_timestamp_inputs: bool,
    ):
        self.short_frames.append(processed.shape[2])
        bucket = self._frame_bucket(processed.shape[2])
        assert (processed.shape[0], bucket) in self.graphs
        return ["short"], None

    def _call_long(
        self,
        processed: torch.Tensor,
        processed_len: torch.Tensor,
        return_ctc_timestamp_inputs: bool,
    ):
        return ["long"], None

    def __call__(
        self,
        audio_signal: torch.Tensor,
        audio_lengths: torch.Tensor,
        return_ctc_timestamp_inputs: bool = False,
    ):
        if not self.graphs and not self._capture_or_disable(80, torch.float32):
            return None
        frames = audio_signal.shape[-1]
        processed = torch.zeros(audio_signal.shape[0], 80, frames)
        processed_len = torch.full((audio_signal.shape[0],), frames)
        return self._call_short(processed, processed_len, return_ctc_timestamp_inputs)


def test_dummy_graph_object_is_not_patched():
    model = _SpeechModel()
    model._encoder_graphs = object()
    assert configure_speech_encoder_graphs(model) is False


def test_graphs_do_not_capture_during_profiling(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True, graphs=True))
    assert getattr(runner, mm_encoder_mod._RUNNER_PATCH_ATTR)
    audio = torch.zeros(1, 1600)
    lengths = torch.tensor([1600])
    assert runner(audio, lengths) is None
    assert runner.capture_all_calls == 0
    assert runner.capture_calls == []
    envs.disable_envs_cache()


def test_graphs_capture_lazily_after_kv_and_crop_to_length(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True, graphs=True))
    enable_speech_encoder_graph_capture(model)
    audio = torch.zeros(1, 4096)
    lengths = torch.tensor([1600])
    assert runner(audio, lengths) == (["short"], None)
    assert runner.capture_all_calls == 0
    assert runner.capture_calls == [(1, 1792)]
    assert runner.short_frames == [1600]
    # Same item, padded wider: same bucket, no new capture.
    audio2 = torch.zeros(1, 8000)
    assert runner(audio2, lengths) == (["short"], None)
    assert runner.capture_calls == [(1, 1792)]
    assert runner.short_frames == [1600, 1600]
    envs.disable_envs_cache()


def test_graphs_path_is_per_item_under_bi(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=False, graphs=True))
    enable_speech_encoder_graph_capture(model)
    audio = torch.zeros(2, 4096)
    lengths = torch.tensor([1600, 4096])
    out = runner(audio, lengths)
    assert out == ["short", "short"]
    assert runner.short_frames == [1600, 4096]
    assert runner.capture_all_calls == 0
    envs.disable_envs_cache()

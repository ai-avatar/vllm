# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for SpeechLM / Nemotron Transcribe encoder compile wiring."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation import mm_encoder as mm_encoder_mod
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.mm_encoder import (
    collect_speech_encoder_layers,
    compile_pending_speech_encoder_layers,
    configure_speech_encoder_dynamo,
    configure_speech_encoder_graphs,
    default_speech_encoder_graph_batch_sizes,
    enable_speech_encoder_after_kv,
    enable_speech_encoder_graph_capture,
    is_nemo_speechlm_config,
    isolate_flex_attention_from_compile,
    maybe_compile_speech_mm_encoder,
    speech_encoder_inductor_config,
    speech_encoder_layer_compile_kwargs,
    speech_encoder_n_feat,
    speech_encoder_served_frames,
    speech_encoder_warmup_frame_buckets,
    speech_encoder_warmup_long_frames,
    sync_mm_encoder_compile_flags,
    unwrap_speech_encoder_model,
    warmup_speech_encoder,
)
from vllm.config import CUDAGraphMode
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
        self.preprocessor = nn.Module()
        self.preprocessor.features = 128
        self.encoder = nn.Module()
        self.encoder.feat_in = 128
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


def test_maybe_compile_still_compiles_when_plugin_graphs_exist(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    model = _SpeechModel()
    model._encoder_graphs = object()
    compiled = maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True))
    assert compiled == 3
    assert model.perception.encoder.layers[0].compile_calls == 1
    assert model.perception.encoder.layers[0].compile_kwargs == {"dynamic": True}
    envs.disable_envs_cache()


def test_compile_kwargs_stay_inductor_under_bi(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    assert speech_encoder_layer_compile_kwargs() == {"dynamic": True}
    assert speech_encoder_inductor_config() == {}

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    kwargs = speech_encoder_layer_compile_kwargs()
    assert kwargs["dynamic"] is True
    assert "backend" not in kwargs
    options = kwargs["options"]
    assert options["deterministic"] is True
    assert options["max_autotune"] is False
    assert options["coordinate_descent_tuning"] is False
    assert speech_encoder_inductor_config() == options
    model = _SpeechModel()
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True))
    assert model.perception.encoder.layers[0].compile_kwargs == kwargs
    envs.disable_envs_cache()


def test_configure_pins_inductor_deterministic_under_bi(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    from torch._inductor import config as inductor_config

    prev = {
        name: getattr(inductor_config, name)
        for name in (
            "deterministic",
            "max_autotune",
            "coordinate_descent_tuning",
            "batch_invariant",
        )
        if hasattr(inductor_config, name)
    }
    try:
        mm_encoder_mod._INDUCTOR_PINNED = False
        configure_speech_encoder_dynamo()
        assert inductor_config.deterministic is True
        assert inductor_config.max_autotune is False
        assert inductor_config.coordinate_descent_tuning is False
        if hasattr(inductor_config, "batch_invariant"):
            assert inductor_config.batch_invariant is True
    finally:
        for name, value in prev.items():
            setattr(inductor_config, name, value)
        mm_encoder_mod._INDUCTOR_PINNED = False
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


def test_bi_recompiles_nemo_flex_wrapper(monkeypatch: pytest.MonkeyPatch):
    import types

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    mm_encoder_mod._BI_FLEX_COMPILED = None

    def fake_compiled(*args, **kwargs):
        return args[0] if args else None

    fake_mod = types.ModuleType(
        "nemo.collections.asr.modules.transformer_encoder_utils"
    )
    fake_mod._flex_attention_compiled = fake_compiled  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake_mod.__name__, fake_mod)
    isolate_flex_attention_from_compile()
    compiled = fake_mod._flex_attention_compiled  # type: ignore[attr-defined]
    assert getattr(compiled, mm_encoder_mod._BI_FLEX_COMPILED_ATTR)
    assert compiled is not fake_compiled
    envs.disable_envs_cache()
    mm_encoder_mod._BI_FLEX_COMPILED = None


def test_patches_nemo_compiled_flex_wrapper(monkeypatch: pytest.MonkeyPatch):
    import types

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    recorded: dict[str, dict] = {}

    def fake_compiled(*args, **kwargs):
        recorded["opts"] = dict(kwargs.get("kernel_options") or {})
        return args[0] if args else None

    def fake_get(q):
        return fake_mod._flex_attention_compiled  # type: ignore[attr-defined]

    fake_mod = types.ModuleType(
        "nemo.collections.asr.modules.transformer_encoder_utils"
    )
    fake_mod._flex_attention_compiled = fake_compiled  # type: ignore[attr-defined]
    fake_mod._get_flex_attention = fake_get  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake_mod.__name__, fake_mod)

    isolate_flex_attention_from_compile()
    compiled = fake_mod._flex_attention_compiled  # type: ignore[attr-defined]
    assert getattr(compiled, mm_encoder_mod._FLEX_ISOLATED_ATTR)
    compiled(object())
    assert recorded["opts"]["FORCE_USE_FLEX_ATTENTION"] is True
    wrapped_get = fake_mod._get_flex_attention  # type: ignore[attr-defined]
    assert getattr(wrapped_get, mm_encoder_mod._FLEX_ISOLATED_ATTR)
    wrapped_get(None)(object())
    assert recorded["opts"]["FORCE_USE_FLEX_ATTENTION"] is True
    envs.disable_envs_cache()


def test_maybe_compile_can_defer_until_after_kv():
    model = _SpeechModel()
    compiled = maybe_compile_speech_mm_encoder(
        model, _vllm_config(compile_mm=True), defer=True
    )
    assert compiled == 0
    assert model.perception.encoder.layers[0].compile_calls == 0
    assert compile_pending_speech_encoder_layers(model) == 3
    assert model.perception.encoder.layers[0].compile_calls == 1
    assert compile_pending_speech_encoder_layers(model) == 0


def test_pending_compile_survives_cudagraph_wrapper(monkeypatch: pytest.MonkeyPatch):
    """Delattr on CUDAGraphWrapper does not forward; unwrap / side table must."""
    monkeypatch.setattr(
        "vllm.compilation.cuda_graph.current_platform.get_global_graph_pool",
        lambda: object(),
    )
    model = _SpeechModel()
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=True), defer=True)
    wrapper = CUDAGraphWrapper(
        model,
        SimpleNamespace(compilation_config=SimpleNamespace()),
        runtime_mode=CUDAGraphMode.FULL,
    )
    assert unwrap_speech_encoder_model(wrapper) is model
    assert getattr(wrapper, mm_encoder_mod._PENDING_LAYERS_ATTR)
    with pytest.raises(AttributeError):
        delattr(wrapper, mm_encoder_mod._PENDING_LAYERS_ATTR)
    assert compile_pending_speech_encoder_layers(wrapper) == 3
    assert model.perception.encoder.layers[0].compile_calls == 1
    assert compile_pending_speech_encoder_layers(wrapper) == 0


def test_pending_compile_setattr_on_already_wrapped_model(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "vllm.compilation.cuda_graph.current_platform.get_global_graph_pool",
        lambda: object(),
    )
    model = _SpeechModel()
    wrapper = CUDAGraphWrapper(
        model,
        SimpleNamespace(compilation_config=SimpleNamespace()),
        runtime_mode=CUDAGraphMode.FULL,
    )
    maybe_compile_speech_mm_encoder(wrapper, _vllm_config(compile_mm=True), defer=True)
    assert mm_encoder_mod._PENDING_LAYERS_ATTR not in vars(wrapper)
    assert hasattr(model, mm_encoder_mod._PENDING_LAYERS_ATTR)
    assert compile_pending_speech_encoder_layers(wrapper) == 3
    assert not hasattr(model, mm_encoder_mod._PENDING_LAYERS_ATTR)


def test_flex_decoding_guard_short_circuits_when_forced():
    isolate_flex_attention_from_compile()
    from torch._inductor.kernel.flex.flex_decoding import _use_flex_decoding

    assert (
        _use_flex_decoding(None, None, None, {"FORCE_USE_FLEX_ATTENTION": True}, False)
        is False
    )
    assert _use_flex_decoding(None, None, None, {"BACKEND": "TRITON"}, False) is False


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
        self.max_windows = 2
        self.max_frames = 2 * self.window
        self.max_rows_x_frames = 128 * 550
        self.enabled = True
        self.pool = None
        self.feat_in = 128
        self.capture_all_calls = 0
        self.capture_calls: list[tuple[int, int]] = []
        self.capture_n_feats: list[int] = []
        self.short_frames: list[int] = []
        self.long_frames: list[int] = []
        self._vllm_closed_shapes = False
        self._vllm_allow_capture = False

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
        if n_feat != self.feat_in:
            raise RuntimeError(
                f"audio_signal should have shape (batch, {self.feat_in}, "
                f"n_frame) but got last dimension {n_feat}"
            )
        self.capture_calls.append((rows, frames))
        self.capture_n_feats.append(n_feat)
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
        if processed.shape[1] != self.feat_in:
            raise RuntimeError(
                f"audio_signal should have shape (batch, {self.feat_in}, "
                f"n_frame) but got last dimension {processed.shape[1]}"
            )
        self.long_frames.append(processed.shape[2])
        return ["long"], None

    def __call__(
        self,
        audio_signal: torch.Tensor,
        audio_lengths: torch.Tensor,
        return_ctc_timestamp_inputs: bool = False,
    ):
        if not self.graphs and not self._capture_or_disable(
            self.feat_in, torch.float32
        ):
            return None
        frames = audio_signal.shape[-1]
        processed = torch.zeros(audio_signal.shape[0], self.feat_in, frames)
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
    assert runner.short_frames == [1792]
    # Same item, padded wider: same bucket, no new capture.
    audio2 = torch.zeros(1, 8000)
    assert runner(audio2, lengths) == (["short"], None)
    assert runner.capture_calls == [(1, 1792)]
    assert runner.short_frames == [1792, 1792]
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
    assert runner.short_frames == [1792, 4096]
    assert runner.capture_all_calls == 0
    envs.disable_envs_cache()


def test_failed_graph_capture_falls_back_at_call(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()

    def boom(*args, **kwargs):
        raise RuntimeError("Cannot copy between CPU and CUDA tensors")

    runner._capture = boom  # type: ignore[method-assign]
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=False, graphs=True))
    enable_speech_encoder_after_kv(model)
    audio = torch.zeros(1, 1600)
    lengths = torch.tensor([1600])
    assert runner(audio, lengths) is None
    assert runner.enabled is False
    assert runner(audio, lengths) is None
    envs.disable_envs_cache()


def test_warmup_captures_bi_buckets_after_kv(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 40)
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()
    runner.window = 550
    runner.max_frames = 550
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(
        model, _vllm_config(compile_mm=True, graphs=True), defer=True
    )
    assert warmup_speech_encoder(model).captured == 0
    assert runner.capture_calls == []
    stats = enable_speech_encoder_after_kv(model)
    assert stats.captured > 0
    assert stats.long_captured == 0
    assert stats.failed == 0
    assert runner.long_frames == []
    assert model.perception.encoder.layers[0].compile_calls == 1
    assert runner.capture_all_calls == 0
    assert {rows for rows, _ in runner.capture_calls} == {1}
    buckets = speech_encoder_warmup_frame_buckets(runner)
    assert buckets == [256, 512]
    assert {frames for _, frames in runner.capture_calls} == set(buckets)
    assert runner.capture_n_feats == [128] * len(runner.capture_calls)
    n_captured = len(runner.capture_calls)
    enable_speech_encoder_after_kv(model)
    assert len(runner.capture_calls) == n_captured
    envs.disable_envs_cache()


def test_warmup_survives_cudagraph_wrapper(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 20)
    monkeypatch.setattr(
        "vllm.compilation.cuda_graph.current_platform.get_global_graph_pool",
        lambda: object(),
    )
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()
    runner.window = 256
    runner.max_frames = 256
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(
        model, _vllm_config(compile_mm=True, graphs=True), defer=True
    )
    wrapper = CUDAGraphWrapper(
        model,
        SimpleNamespace(compilation_config=SimpleNamespace()),
        runtime_mode=CUDAGraphMode.FULL,
    )
    stats = enable_speech_encoder_after_kv(wrapper)
    assert stats.captured > 0
    assert model.perception.encoder.layers[0].compile_calls == 1
    assert (1, 256) in runner.graphs
    assert runner.capture_n_feats == [128]
    envs.disable_envs_cache()


def test_n_feat_comes_from_preprocessor_not_80():
    model = _SpeechModel()
    assert speech_encoder_n_feat(model) == 128
    del model.perception.preprocessor.features
    assert speech_encoder_n_feat(model) == 128  # encoder.feat_in
    del model.perception.encoder.feat_in
    runner = _FakeGraphRunner()
    assert speech_encoder_n_feat(model, runner) == 128
    assert speech_encoder_n_feat(_SpeechModel()) != 80


def test_warmup_wrong_n_feat_fails_loudly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 20)
    monkeypatch.setattr(mm_encoder_mod, "speech_encoder_n_feat", lambda *a, **k: 80)
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()
    runner.window = 256
    runner.max_frames = 256
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=False, graphs=True))
    stats = enable_speech_encoder_after_kv(model)
    assert stats.captured == 0
    assert stats.failed > 0
    assert runner.capture_calls == []
    assert runner.enabled is True
    envs.disable_envs_cache()


def test_compile_only_skips_dummy_encode_warmup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 3600)
    envs.disable_envs_cache()
    model = _SpeechModel()
    maybe_compile_speech_mm_encoder(
        model, _vllm_config(compile_mm=True, graphs=False), defer=True
    )
    stats = enable_speech_encoder_after_kv(model)
    assert stats.captured == 0
    assert stats.failed == 0
    assert stats.elapsed_s == 0.0
    assert model.perception.encoder.layers[0].compile_calls == 1
    envs.disable_envs_cache()


def test_served_frames_pad_long_remainder_to_window(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 600)
    envs.disable_envs_cache()
    runner = _FakeGraphRunner()
    runner.window = 550
    runner.max_windows = 2
    assert speech_encoder_served_frames(80, runner, long=False) == 256
    assert speech_encoder_served_frames(1100, runner, long=False) == 1280
    assert speech_encoder_served_frames(1101, runner, long=True) == 1650
    assert speech_encoder_served_frames(3 * 550 + 50, runner, long=True) == 4 * 550
    assert speech_encoder_served_frames(4 * 550, runner, long=True) == 4 * 550
    envs.disable_envs_cache()


def test_long_warmup_frames_are_every_window_count(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 160)
    envs.disable_envs_cache()
    runner = _FakeGraphRunner()
    runner.window = 550
    runner.max_windows = 2
    frames = speech_encoder_warmup_long_frames(runner)
    assert frames == [3 * 550]
    assert 2 * 550 + 256 not in frames
    assert all(length % 550 == 0 for length in frames)
    assert all(length <= int(160 / 0.08) for length in frames)
    envs.disable_envs_cache()


def test_long_call_pads_remainder_to_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=False, graphs=True))
    enable_speech_encoder_graph_capture(model)
    processed = torch.zeros(1, 128, 3 * 550 + 50)
    processed_len = torch.tensor([3 * 550 + 50])
    assert runner._call_long(processed, processed_len, False) == (["long"], None)
    assert runner.long_frames == [4 * 550]
    envs.disable_envs_cache()


def test_warmup_closes_serving_time_capture(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 160)
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()
    runner.window = 550
    runner.max_windows = 2
    runner.max_frames = 1100
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=False, graphs=True))
    stats = enable_speech_encoder_after_kv(model)
    assert stats.captured > 0
    assert runner._vllm_closed_shapes
    assert runner._vllm_allow_capture is False
    n_captured = len(runner.capture_calls)
    processed = torch.zeros(1, 128, 1101)
    runner._call_long(processed, torch.tensor([1101]), False)
    assert runner.long_frames[-1] == 3 * 550
    assert len(runner.capture_calls) == n_captured
    envs.disable_envs_cache()


def test_warmup_captures_long_window_graphs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 160)
    envs.disable_envs_cache()
    model = _SpeechModel()
    runner = _FakeGraphRunner()
    runner.window = 550
    runner.max_windows = 2
    runner.max_frames = 1100
    model._encoder_graphs = runner
    maybe_compile_speech_mm_encoder(model, _vllm_config(compile_mm=False, graphs=True))
    stats = enable_speech_encoder_after_kv(model)
    assert stats.long_captured > 0
    assert stats.failed == 0
    assert stats.captured > stats.long_captured
    expected = speech_encoder_warmup_long_frames(runner)
    assert runner.long_frames == expected
    assert all(rows == 1 for rows, _ in runner.capture_calls)
    assert min(runner.long_frames) > 1100
    envs.disable_envs_cache()

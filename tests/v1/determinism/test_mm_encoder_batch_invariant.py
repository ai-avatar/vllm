# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder isolation under VLLM_BATCH_INVARIANT."""

from unittest.mock import MagicMock

import pytest
import torch

import vllm.envs as envs
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def enable_batch_invariant(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def test_execute_mm_encoder_runs_items_one_at_a_time(
    enable_batch_invariant, monkeypatch: pytest.MonkeyPatch
):
    batch_sizes: list[int] = []

    def fake_group(items, device=None, pin_memory=False):
        batch_sizes.append(len(items))
        yield "audio", len(items), {"audio": [None] * len(items)}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.mm.encoder_runner.group_and_batch_mm_kwargs",
        fake_group,
    )

    model = MagicMock()
    model.embed_multimodal.side_effect = lambda **kwargs: [
        torch.zeros(1, 4) for _ in kwargs["audio"]
    ]

    runner = EncoderRunner(
        model=model,
        max_num_tokens=16,
        hidden_size=4,
        encoder_cache=MagicMock(),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    items = [("audio", None), ("audio", None), ("audio", None)]
    outputs = runner.execute_mm_encoder(items)
    assert len(outputs) == 3
    assert batch_sizes == [1, 1, 1]


def test_execute_mm_encoder_batches_when_bi_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    batch_sizes: list[int] = []

    def fake_group(items, device=None, pin_memory=False):
        batch_sizes.append(len(items))
        yield "audio", len(items), {"audio": [None] * len(items)}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.mm.encoder_runner.group_and_batch_mm_kwargs",
        fake_group,
    )
    model = MagicMock()
    model.embed_multimodal.side_effect = lambda **kwargs: [
        torch.zeros(1, 4) for _ in kwargs["audio"]
    ]
    runner = EncoderRunner(
        model=model,
        max_num_tokens=16,
        hidden_size=4,
        encoder_cache=MagicMock(),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    outputs = runner.execute_mm_encoder([("audio", None), ("audio", None)])
    assert len(outputs) == 2
    assert batch_sizes == [2]
    envs.disable_envs_cache()


def test_v1_gpu_model_runner_encodes_audio_one_at_a_time(
    enable_batch_invariant, monkeypatch: pytest.MonkeyPatch
):
    """V1 GPUModelRunner._execute_mm_encoder must not batch audios under BI.

    Production nemotron-api sets VLLM_USE_V2_MODEL_RUNNER=0; the V2
    EncoderRunner change does not run on that path.
    """
    from contextlib import contextmanager
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    embed_batch_sizes: list[int] = []

    def fake_group(items, device=None, pin_memory=False):
        yield "audio", len(items), {"audio": [None] * len(items)}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        fake_group,
    )

    class Model:
        def embed_multimodal(self, **kwargs):
            n = len(kwargs["audio"])
            embed_batch_sizes.append(n)
            return [torch.zeros(2, 4) for _ in range(n)]

    @contextmanager
    def _no_timing(*_args, **_kwargs):
        yield

    runner = SimpleNamespace(
        observability_config=None,
        model=Model(),
        lora_config=None,
        is_multimodal_pruning_enabled=False,
        requires_sequential_video_encoding=False,
        encoder_cudagraph_manager=None,
        device=torch.device("cpu"),
        timed_encoder_operation=_no_timing,
    )
    runner._batch_mm_inputs_from_scheduler = lambda _out: (
        ["h0", "h1", "h2"],
        [("audio", None), ("audio", None), ("audio", None)],
        [("r0", None), ("r1", None), ("r2", None)],
    )
    runner._cache_encoder_output = lambda *args, **kwargs: None

    outputs = GPUModelRunner._execute_mm_encoder(
        runner,
        SimpleNamespace(ec_manager_metadata=None, free_encoder_mm_hashes=[]),
    )
    assert len(outputs) == 3
    assert embed_batch_sizes == [1, 1, 1]


def test_v1_gpu_model_runner_batches_audio_when_bi_off(
    monkeypatch: pytest.MonkeyPatch,
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    embed_batch_sizes: list[int] = []

    def fake_group(items, device=None, pin_memory=False):
        yield "audio", len(items), {"audio": [None] * len(items)}

    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        fake_group,
    )

    class Model:
        def embed_multimodal(self, **kwargs):
            n = len(kwargs["audio"])
            embed_batch_sizes.append(n)
            return [torch.zeros(2, 4) for _ in range(n)]

    @contextmanager
    def _no_timing(*_args, **_kwargs):
        yield

    runner = SimpleNamespace(
        observability_config=None,
        model=Model(),
        lora_config=None,
        is_multimodal_pruning_enabled=False,
        requires_sequential_video_encoding=False,
        encoder_cudagraph_manager=None,
        device=torch.device("cpu"),
        timed_encoder_operation=_no_timing,
    )
    runner._batch_mm_inputs_from_scheduler = lambda _out: (
        ["h0", "h1"],
        [("audio", None), ("audio", None)],
        [("r0", None), ("r1", None)],
    )
    runner._cache_encoder_output = lambda *args, **kwargs: None

    outputs = GPUModelRunner._execute_mm_encoder(
        runner,
        SimpleNamespace(ec_manager_metadata=None, free_encoder_mm_hashes=[]),
    )
    assert len(outputs) == 2
    assert embed_batch_sizes == [2]
    envs.disable_envs_cache()

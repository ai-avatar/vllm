# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for Mamba2 batch-invariant scheduler chunk alignment."""

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.scheduler import Scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def enable_batch_invariant(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def _bi_stub(chunk_size: int = 128) -> SimpleNamespace:
    return SimpleNamespace(
        has_mamba_layers=True,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(get_mamba_chunk_size=lambda: chunk_size)
        ),
    )


def test_bi_mamba_chunk_aligned_clips_unaligned_partial(enable_batch_invariant):
    stub = _bi_stub(128)
    request = SimpleNamespace(
        num_computed_tokens=0, num_prompt_tokens=2615, num_tokens=2615
    )
    # 2296 % 128 == 120; clip to 17 * 128 = 2176.
    assert Scheduler._bi_mamba_chunk_aligned(stub, request, 2296) == 2176
    # Aligned splits are left alone.
    assert Scheduler._bi_mamba_chunk_aligned(stub, request, 2304) == 2304
    assert Scheduler._bi_mamba_chunk_aligned(stub, request, 2048) == 2048
    # Finishing the prompt is not a continuation; do not clip.
    assert Scheduler._bi_mamba_chunk_aligned(stub, request, 2615) == 2615


def test_bi_mamba_chunk_aligned_mid_prompt_and_disabled(enable_batch_invariant):
    stub = _bi_stub(128)
    request = SimpleNamespace(
        num_computed_tokens=2048, num_prompt_tokens=2615, num_tokens=2615
    )
    # 2048 + 248 = 2296, still a partial prefill.
    # end=2296, 2296 // 128 * 128 - 2048 = 128.
    assert Scheduler._bi_mamba_chunk_aligned(stub, request, 248) == 128
    # Last remainder that finishes the prompt is unclipped.
    request.num_computed_tokens = 2560
    assert Scheduler._bi_mamba_chunk_aligned(stub, request, 55) == 55

    stub.has_mamba_layers = False
    request.num_computed_tokens = 0
    assert Scheduler._bi_mamba_chunk_aligned(stub, request, 2296) == 2296


def test_bi_mamba_chunk_aligned_noop_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    envs.disable_envs_cache()
    stub = _bi_stub(128)
    request = SimpleNamespace(
        num_computed_tokens=0, num_prompt_tokens=2615, num_tokens=2615
    )
    assert Scheduler._bi_mamba_chunk_aligned(stub, request, 2296) == 2296
    envs.disable_envs_cache()


def test_scheduler_aligns_partial_prefill_to_chunk_size(enable_batch_invariant):
    # opt-125m derived max_model_len is 2048.
    scheduler = create_scheduler(
        max_num_batched_tokens=1519,
        max_model_len=2048,
        enable_chunked_prefill=True,
    )
    scheduler.has_mamba_layers = True
    scheduler.vllm_config.model_config.get_mamba_chunk_size = lambda: 128  # type: ignore[method-assign]
    request = create_requests(1, num_tokens=1800, max_tokens=1, req_ids=["long"])[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    scheduled = output.num_scheduled_tokens[request.request_id]
    assert scheduled > 0
    assert scheduled % 128 == 0
    assert scheduled == 1408  # 1519 // 128 * 128

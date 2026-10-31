# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for Mamba2 batch-invariant scheduler alignment and isolation."""

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.v1.core.sched.scheduler import Scheduler

from tests.v1.core.utils import create_requests, create_scheduler

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
        _bi_prefill_scheduled=False,
        _bi_decode_scheduled=False,
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


def test_bi_skip_mixed_phase_isolates_prefill_and_decode(enable_batch_invariant):
    stub = _bi_stub()
    prefill = SimpleNamespace(
        num_computed_tokens=0, num_prompt_tokens=4000, num_tokens=4000
    )
    decode = SimpleNamespace(
        num_computed_tokens=80, num_prompt_tokens=80, num_tokens=81
    )
    assert not Scheduler._bi_skip_mixed_phase(stub, prefill)
    Scheduler._bi_note_scheduled(stub, prefill)
    assert stub._bi_prefill_scheduled
    assert Scheduler._bi_skip_mixed_phase(stub, decode)
    assert not Scheduler._bi_skip_mixed_phase(stub, prefill)

    stub._bi_prefill_scheduled = False
    stub._bi_decode_scheduled = False
    Scheduler._bi_note_scheduled(stub, decode)
    assert stub._bi_decode_scheduled
    assert Scheduler._bi_skip_mixed_phase(stub, prefill)
    assert not Scheduler._bi_skip_mixed_phase(stub, decode)


def test_scheduler_aligns_partial_prefill_to_chunk_size(enable_batch_invariant):
    # opt-125m derived max_model_len is 2048.
    scheduler = create_scheduler(
        max_num_batched_tokens=1519,
        max_model_len=2048,
        enable_chunked_prefill=True,
    )
    scheduler.has_mamba_layers = True
    scheduler.vllm_config.model_config.get_mamba_chunk_size = lambda: 128
    request = create_requests(1, num_tokens=1800, max_tokens=1, req_ids=["long"])[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    scheduled = output.num_scheduled_tokens[request.request_id]
    assert scheduled > 0
    assert scheduled % 128 == 0
    assert scheduled == 1408  # 1519 // 128 * 128


def test_scheduler_does_not_mix_prefill_and_decode(enable_batch_invariant):
    from vllm.v1.outputs import ModelRunnerOutput

    scheduler = create_scheduler(
        max_num_batched_tokens=2048,
        max_model_len=2048,
        enable_chunked_prefill=True,
    )
    scheduler.has_mamba_layers = True
    scheduler.vllm_config.model_config.get_mamba_chunk_size = lambda: 128

    short_req = create_requests(1, num_tokens=64, max_tokens=8, req_ids=["short"])[0]
    scheduler.add_request(short_req)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens[short_req.request_id] == 64
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=[short_req.request_id],
            req_id_to_index={short_req.request_id: 0},
            sampled_token_ids=[[1]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    long_req = create_requests(1, num_tokens=1800, max_tokens=8, req_ids=["long"])[0]
    scheduler.add_request(long_req)
    second = scheduler.schedule()
    scheduled = second.num_scheduled_tokens
    # Running decode is scheduled first and pins the step as decode-only.
    assert short_req.request_id in scheduled
    assert long_req.request_id not in scheduled

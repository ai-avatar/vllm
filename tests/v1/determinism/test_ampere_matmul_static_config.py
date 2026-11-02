# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere (sm80) persistent-matmul tiles must ignore M / batch size.

Ada/Hopper tables still use M-buckets with a shared BLOCK_K. Ampere cannot:
a request that runs alone and the same row inside a concurrent batch must
launch identical BLOCK_M/N/K, num_warps, and num_stages.
"""

import pytest
import torch
from utils import skip_if_not_cuda

import vllm.envs as envs
import vllm.model_executor.determinism.batch_invariant as bi
import vllm.model_executor.determinism.batch_invariant_configs as bi_configs
from vllm.model_executor.determinism.batch_invariant import (
    matmul_batch_invariant,
)
from vllm.model_executor.determinism.batch_invariant_configs import (
    _BATCH_INVARIANT_MATMUL_STATIC_CONFIGS,
    _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS,
    NEMOTRON_TRANSCRIBE_BF16_GEMM_SHAPES,
    _get_matmul_config,
    _get_tuned_matmul_arch_family,
    _static_config_kernel_kwargs,
    reset_matmul_config_cache,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

DEVICE_TYPE = current_platform.device_type

_BF16_DEFAULT = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_stages": 3,
    "num_warps": 8,
}


def _unique_nemotron_nk() -> list[tuple[str, int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[str, int, int]] = []
    for name, n, k in NEMOTRON_TRANSCRIBE_BF16_GEMM_SHAPES:
        if (n, k) in seen:
            continue
        seen.add((n, k))
        out.append((name, n, k))
    return out


@pytest.fixture
def ampere_static_lookup():
    reset_matmul_config_cache()
    bi_configs._STATIC_MATMUL_CONFIGS_FOR_DEVICE = (
        _BATCH_INVARIANT_MATMUL_STATIC_CONFIGS["ampere"]
    )
    bi_configs._TUNED_MATMUL_CONFIGS_FOR_DEVICE = None
    bi_configs._TUNED_MATMUL_CONFIGS_RESOLVED = True
    yield
    reset_matmul_config_cache()


@pytest.mark.cpu_test
def test_sm80_maps_to_ampere_not_ada():
    assert _get_tuned_matmul_arch_family(DeviceCapability(8, 0)) == "ampere"
    assert _get_tuned_matmul_arch_family(DeviceCapability(8, 9)) == "ada"
    assert _get_tuned_matmul_arch_family(DeviceCapability(8, 6)) is None
    assert "ampere" not in _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS
    assert "ampere" in _BATCH_INVARIANT_MATMUL_STATIC_CONFIGS


@pytest.mark.cpu_test
def test_nemotron_shapes_are_in_ampere_table():
    table = _BATCH_INVARIANT_MATMUL_STATIC_CONFIGS["ampere"]
    for name, n, k in _unique_nemotron_nk():
        assert (n, k) in table, f"{name} ({n}, {k}) missing from Ampere table"
        cfg = table[(n, k)]
        smem = (
            cfg.num_stages * 2 * (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n)
        )
        assert smem <= 160000, f"{name} Ampere tile exceeds A100 smem ({smem})"


# A100 --quick sweep winners from ampere_bi_matmul_best.py (M {1,2,4..512}).
_A100_QUICK_SWEEP_AMPERE = {
    (128, 2688): (32, 32, 128, 4, 3),
    (1856, 2688): (32, 64, 128, 4, 3),
    (2688, 1856): (32, 128, 128, 4, 3),
    (2688, 3712): (32, 128, 128, 4, 3),
    (2688, 4096): (32, 128, 128, 4, 3),
    (2688, 5376): (64, 64, 128, 8, 3),
    (3712, 2688): (32, 128, 128, 8, 3),
    (4608, 2688): (32, 128, 128, 8, 3),
    (10304, 2688): (64, 128, 128, 8, 3),
    (131072, 2688): (64, 256, 64, 8, 3),
}


@pytest.mark.cpu_test
def test_ampere_table_matches_a100_quick_sweep():
    table = _BATCH_INVARIANT_MATMUL_STATIC_CONFIGS["ampere"]
    assert set(table) == set(_A100_QUICK_SWEEP_AMPERE)
    for nk, expected in _A100_QUICK_SWEEP_AMPERE.items():
        cfg = table[nk]
        assert (
            cfg.block_m,
            cfg.block_n,
            cfg.block_k,
            cfg.num_warps,
            cfg.num_stages,
        ) == expected, nk
    # Heuristic seed tiles (block_k 64) changed numerics; none may remain.
    seed_ampere = {
        (131072, 2688): (32, 128, 64, 8, 3),
        (4608, 2688): (64, 128, 64, 8, 3),
        (2688, 4096): (64, 128, 64, 8, 3),
        (1856, 2688): (64, 128, 64, 8, 3),
        (2688, 1856): (64, 64, 64, 8, 3),
        (3712, 2688): (64, 128, 64, 8, 3),
        (2688, 3712): (64, 128, 64, 8, 3),
        (128, 2688): (64, 32, 64, 4, 3),
        (10304, 2688): (64, 128, 64, 8, 3),
        (2688, 5376): (64, 128, 64, 8, 3),
    }
    for nk, seed in seed_ampere.items():
        cfg = table[nk]
        actual = (
            cfg.block_m,
            cfg.block_n,
            cfg.block_k,
            cfg.num_warps,
            cfg.num_stages,
        )
        assert actual != seed, f"{nk} still using seed tile {seed}"


@pytest.mark.cpu_test
def test_ampere_lookup_ignores_m(ampere_static_lookup):
    table = _BATCH_INVARIANT_MATMUL_STATIC_CONFIGS["ampere"]
    for name, n, k in _unique_nemotron_nk():
        cfg_m1 = _get_matmul_config(1, n, k, torch.bfloat16, _BF16_DEFAULT)
        cfg_m512 = _get_matmul_config(512, n, k, torch.bfloat16, _BF16_DEFAULT)
        assert cfg_m1 == cfg_m512, name
        assert cfg_m1 == _static_config_kernel_kwargs(table[(n, k)]), name
        assert cfg_m1 is not _BF16_DEFAULT


@pytest.mark.cpu_test
def test_ampere_unknown_shape_and_fp16_keep_default(ampere_static_lookup):
    missing = _get_matmul_config(8, 7, 9, torch.bfloat16, _BF16_DEFAULT)
    assert missing is _BF16_DEFAULT
    fp16 = _get_matmul_config(8, 1856, 2688, torch.float16, _BF16_DEFAULT)
    assert fp16 is _BF16_DEFAULT


@pytest.mark.cpu_test
def test_init_batch_invariance_is_noop_when_unset(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    envs.disable_envs_cache()
    lib = bi._batch_invariant_LIB
    mode = bi._batch_invariant_MODE
    try:
        if lib is not None:
            lib._destroy()
        bi._batch_invariant_LIB = None
        bi._batch_invariant_MODE = False
        bi.init_batch_invariance()
        assert bi._batch_invariant_MODE is False
        assert bi._batch_invariant_LIB is None
    finally:
        bi._batch_invariant_LIB = lib
        bi._batch_invariant_MODE = mode
        envs.disable_envs_cache()


@skip_if_not_cuda
@pytest.mark.parametrize(
    "n,k",
    [(1856, 2688), (128, 2688)],
    ids=["mlp_up", "router_gate"],
)
def test_nemotron_static_tiles_row0_matches_across_m(n, k, ampere_static_lookup):
    """Row 0 is bitwise identical for M=1 and a larger batch."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)
    a = torch.randn(32, k, dtype=torch.bfloat16, device=device)
    b = torch.randn(k, n, dtype=torch.bfloat16, device=device)
    alone = matmul_batch_invariant(a[:1], b)
    batched = matmul_batch_invariant(a, b)
    assert torch.equal(alone[0], batched[0])


@skip_if_not_cuda
def test_stock_mm_does_not_call_persistent_kernel(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    envs.disable_envs_cache()
    lib = bi._batch_invariant_LIB
    mode = bi._batch_invariant_MODE
    called = {"n": 0}

    def spy(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("stock torch.mm must not use the BI kernel")

    try:
        if lib is not None:
            lib._destroy()
        bi._batch_invariant_LIB = None
        bi._batch_invariant_MODE = False
        monkeypatch.setattr(bi, "matmul_persistent", spy)
        a = torch.randn(8, 32, device=DEVICE_TYPE, dtype=torch.bfloat16)
        b = torch.randn(32, 16, device=DEVICE_TYPE, dtype=torch.bfloat16)
        out = torch.mm(a, b)
        assert out.shape == (8, 16)
        assert called["n"] == 0
    finally:
        bi._batch_invariant_LIB = lib
        bi._batch_invariant_MODE = mode
        envs.disable_envs_cache()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-QSA-T01: the XPU launch-override hook validates and applies per kernel."""

import pytest

from vllm.models.qwen4_exp.nvidia.ops import qsa_xpu_tuning as xpu_tune


@pytest.fixture(autouse=True)
def _reset():
    xpu_tune.set_overrides({})
    yield
    xpu_tune.set_overrides(None)


def test_overrides_apply_per_kernel_and_default_otherwise():
    xpu_tune.set_overrides(
        {"sparse_gqa_prefill": {"block_n": 64, "num_warps": 8, "grf_mode": "256"}}
    )
    assert xpu_tune.get("sparse_gqa_prefill", "block_n", 32) == 64
    assert xpu_tune.get("sparse_gqa_prefill", "splits", 1) == 1
    assert xpu_tune.get("sparse_gqa_decode", "block_n", 32) == 32
    assert xpu_tune.launch_extra("sparse_gqa_prefill") == {"grf_mode": "256"}
    assert xpu_tune.launch_extra("sparse_gqa_decode") == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"no_such_kernel": {}},
        {"merge": {"block_n": 64}},
        {"decode_score": {"grf_mode": "1024"}},
    ],
)
def test_overrides_reject_unknown_names(bad):
    with pytest.raises(ValueError):
        xpu_tune.set_overrides(bad)


def test_default_grf_is_not_passed():
    xpu_tune.set_overrides({"merge": {"grf_mode": "default", "num_warps": 4}})
    assert xpu_tune.launch_extra("merge") == {}

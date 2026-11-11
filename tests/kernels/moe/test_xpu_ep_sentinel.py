# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU expert-parallel sentinel kernel: ids as ``torch.where`` maps them,
and a zeroed ``rows_per_expert``; and the expert-parallel call into the fused
MoE with and without the kernels' ``ids_valid`` / ``rows_per_expert``.

The kernel runs on a card, and on CPU under the Triton interpreter (in a
subprocess, so the interpreter setting stays out of this session)."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

_INTERPRETED_CHECK = """
import torch
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
    sentinel_ids_and_counts,
)

g = torch.Generator().manual_seed(0)
for num_tokens, top_k, num_local_experts, sentinel in (
    (1, 1, 4, 0),
    (7, 8, 64, 255),
    (300, 8, 300, 0),
):
    ids = torch.randint(-1, 256, (num_tokens, top_k), dtype=torch.int32, generator=g)
    out, counts = sentinel_ids_and_counts(ids, sentinel, num_local_experts)
    assert torch.equal(out, torch.where(ids < 0, sentinel, ids)), (num_tokens, top_k)
    assert out.shape == ids.shape and out.dtype == ids.dtype
    assert counts.dtype == torch.int32 and counts.shape == (num_local_experts,)
    assert not counts.any()
print("SENTINEL-OK")
"""


def test_sentinel_ids_and_counts_interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    run = subprocess.run(
        [sys.executable, "-c", _INTERPRETED_CHECK],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0 and "SENTINEL-OK" in run.stdout, run.stderr[-3000:]


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")
@pytest.mark.parametrize("num_tokens", [1, 7, 300, 4096])
@pytest.mark.parametrize("top_k", [1, 8])
@pytest.mark.parametrize("num_local_experts", [4, 64, 300])
@pytest.mark.parametrize("sentinel", [0, 255])
def test_sentinel_ids_and_counts(num_tokens, top_k, num_local_experts, sentinel):
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        sentinel_ids_and_counts,
    )

    ids = torch.randint(-1, 256, (num_tokens, top_k), dtype=torch.int32, device="xpu")
    out, counts = sentinel_ids_and_counts(ids, sentinel, num_local_experts)
    torch.testing.assert_close(out, torch.where(ids < 0, sentinel, ids), rtol=0, atol=0)
    assert out.shape == ids.shape and out.dtype == ids.dtype
    assert counts.dtype == torch.int32 and counts.shape == (num_local_experts,)
    assert int(counts.abs().sum()) == 0


class _FusedMoeIdsValid:
    def apply(
        self,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        a1q_scale=None,
        ids_valid=False,
        rows_per_expert=None,
    ):
        pass


class _FusedMoeOpHandlesIds(_FusedMoeIdsValid):
    HANDLES_UNOWNED_IDS = True
    CLEARS_ROWS_PER_EXPERT = True


class _FusedMoeOneMarker(_FusedMoeIdsValid):
    HANDLES_UNOWNED_IDS = True


class _FusedMoePlain:
    def apply(self, output, hidden_states, topk_weights, topk_ids, a1q_scale=None):
        pass


@pytest.mark.parametrize(
    "stub, expected",
    [
        (_FusedMoeOpHandlesIds, "OP_HANDLES_IDS"),
        (_FusedMoeOneMarker, "IDS_VALID"),
        (_FusedMoeIdsValid, "IDS_VALID"),
        (_FusedMoePlain, "PADDING_MASK"),
    ],
)
@pytest.mark.parametrize("eplb", [False, True])
def test_ep_ids_contract(monkeypatch, stub, expected, eplb):
    """Both kernels markers: the op handles the ids; one marker or only the
    ids_valid parameters: ids_valid; neither: the padding-row mask. EPLB keeps
    the layer rewriting the ids itself."""
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    monkeypatch.setattr(xpu_moe, "XpuFusedMoe", stub, raising=False)
    xpu_moe.fused_moe_ep_contract.cache_clear()
    try:
        moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=eplb)
        )
        contract = xpu_moe.ep_ids_contract_for(moe_config)
    finally:
        xpu_moe.fused_moe_ep_contract.cache_clear()
    if eplb and expected == "OP_HANDLES_IDS":
        expected = "IDS_VALID"
    assert contract is xpu_moe.EPIdsContract[expected]


@pytest.mark.parametrize("contract", ["OP_HANDLES_IDS", "IDS_VALID", "PADDING_MASK"])
@pytest.mark.parametrize("ep_rank", [0, 2])
def test_expert_parallel_apply(monkeypatch, contract, ep_rank):
    """OP_HANDLES_IDS: the routed ids as they are, no sentinel kernel, neither
    argument. IDS_VALID: sentinel-mapped ids from the one kernel,
    ids_valid=True and a zeroed int32 rows_per_expert. PADDING_MASK: the same
    ids through torch.where and neither argument, so the fused MoE keeps its
    mask."""
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    num_global, num_local = 256, 64
    sentinel = num_global - 1 if ep_rank == 0 else 0
    kernel_calls = []

    def reference_sentinel(topk_ids, sentinel_, num_local_experts):
        kernel_calls.append(1)
        assert sentinel_ == sentinel and num_local_experts == num_local
        counts = torch.zeros(num_local_experts, dtype=torch.int32)
        return torch.where(topk_ids < 0, sentinel_, topk_ids), counts

    seen = {}

    class Impl:
        num_experts = num_local

        def apply(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(xpu_moe, "sentinel_ids_and_counts", reference_sentinel)
    monkeypatch.setattr(xpu_moe, "staged_impl", lambda impl, n, slot: impl)
    experts = object.__new__(xpu_moe.XPUExperts)
    experts.tiers = None
    experts.fused_moe_impl = Impl()
    experts.impl_apply_kwargs = {}
    experts.ep_contract = xpu_moe.EPIdsContract[contract]
    experts.moe_config = SimpleNamespace(ep_size=4, ep_rank=ep_rank)
    ids = torch.tensor([[3, -1], [-1, 7]], dtype=torch.int32)
    hidden = torch.zeros(2, 8)
    experts.apply(
        torch.zeros(2, 8),
        hidden,
        None,
        None,
        torch.ones(2, 2),
        ids,
        None,
        num_global,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    if contract == "OP_HANDLES_IDS":
        assert seen["topk_ids"] is ids and not kernel_calls
        assert "ids_valid" not in seen and "rows_per_expert" not in seen
        return
    assert torch.equal(seen["topk_ids"], torch.where(ids < 0, sentinel, ids))
    if contract == "IDS_VALID":
        assert kernel_calls == [1]
        assert seen["ids_valid"] is True
        assert seen["rows_per_expert"].dtype == torch.int32
        assert not seen["rows_per_expert"].any()
    else:
        assert not kernel_calls
        assert "ids_valid" not in seen and "rows_per_expert" not in seen

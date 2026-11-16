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
            ep_size=4, moe_parallel_config=SimpleNamespace(enable_eplb=eplb)
        )
        contract = xpu_moe.ep_ids_contract_for(moe_config)
    finally:
        xpu_moe.fused_moe_ep_contract.cache_clear()
    if eplb and expected == "OP_HANDLES_IDS":
        expected = "IDS_VALID"
    assert contract is xpu_moe.EPIdsContract[expected]


@pytest.mark.parametrize(
    "stub, expected",
    [
        (_FusedMoeOpHandlesIds, "OP_HANDLES_IDS"),
        (_FusedMoeOneMarker, "IDS_VALID"),
        (_FusedMoePlain, "PADDING_MASK"),
    ],
)
@pytest.mark.parametrize("ep_size", [1, 4])
def test_ep_ids_contract_eplb_op_ids(monkeypatch, stub, expected, ep_size):
    """VLLM_XPU_MOE_EPLB_OP_IDS=1: under EPLB a fused MoE that filters the ids
    itself gets them as they are (no sentinel kernel); other kernels keep
    their contract."""
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    monkeypatch.setenv("VLLM_XPU_MOE_EPLB_OP_IDS", "1")
    monkeypatch.setattr(xpu_moe, "XpuFusedMoe", stub, raising=False)
    xpu_moe.fused_moe_ep_contract.cache_clear()
    try:
        moe_config = SimpleNamespace(
            ep_size=ep_size, moe_parallel_config=SimpleNamespace(enable_eplb=True)
        )
        contract = xpu_moe.ep_ids_contract_for(moe_config)
    finally:
        xpu_moe.fused_moe_ep_contract.cache_clear()
    assert contract is xpu_moe.EPIdsContract[expected]


def test_w4a8_prep_report(monkeypatch):
    """The boot line names what the installed W4A8 apply runs around its
    GEMMs: a wheel without the lean prep, the lean prep, or the stock prep
    restored by VLLM_XPU_W4A8_STOCK_PREP=1."""
    import types

    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    pkg = types.ModuleType("vllm_xpu_kernels")
    mod = types.ModuleType("vllm_xpu_kernels.moe_w4a8")
    pkg.moe_w4a8 = mod
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", pkg)
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.moe_w4a8", mod)
    assert "no lean prep" in xpu_moe.w4a8_prep()
    mod.stock_prep = lambda: False
    assert "lean prep" in xpu_moe.w4a8_prep()
    assert "no padding mask" in xpu_moe.w4a8_prep()
    mod.stock_prep = lambda: True
    assert "VLLM_XPU_W4A8_STOCK_PREP=1" in xpu_moe.w4a8_prep()


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
    # A strided view, as a router may hand over: the op reads a dense array.
    ids = torch.tensor([[3, 99, -1, 99], [-1, 99, 7, 99]], dtype=torch.int32)[:, ::2]
    assert not ids.is_contiguous()
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
        assert seen["topk_ids"].is_contiguous() and not kernel_calls
        assert torch.equal(seen["topk_ids"], ids)
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


_INTERPRETED_ROUTE_FUSED = """
import torch
from vllm.platforms import current_platform

# The EPLB mapping kernel is defined for CUDA-alike and XPU platforms only;
# claim CUDA-alike while it is defined (XPU would import the kernels wheel).
current_platform.is_cuda_alike = lambda: True
from vllm.model_executor.layers.fused_moe.router import base_router

# The interpreter's atomic_rmw takes only a contiguous bool mask; the record
# mask reaches it as a broadcast view.
import numpy as np
from triton.runtime import interpreter as it

_rmw = it.InterpreterBuilder.create_atomic_rmw


def _rmw_bool_mask(self, op, ptr, val, mask, sem, scope):
    mask = it.TensorHandle(np.ascontiguousarray(mask.data, dtype=bool), mask.dtype)
    return _rmw(self, op, ptr, val, mask, sem, scope)


it.InterpreterBuilder.create_atomic_rmw = _rmw_bool_mask


def sentinel_ids_and_counts(ids, sentinel, n):
    # The sentinel kernel's ids (proved equal to this above and on the card).
    return torch.where(ids < 0, sentinel, ids), torch.zeros(n, dtype=torch.int32)

g = torch.Generator().manual_seed(1)
num_logical, slots = 64, 3
cases = 0
for num_tokens, top_k, num_physical, sentinel in (
    (1, 8, 80, 79),
    (8, 10, 80, 0),
    (13, 10, 96, 95),
    (300, 4, 64, 0),
):
    replicas = torch.randint(1, slots + 1, (num_logical,), generator=g)
    l2p = torch.full((num_logical, slots), -1, dtype=torch.int64)
    for e in range(num_logical):
        r = int(replicas[e])
        l2p[e, :r] = torch.randperm(num_physical, generator=g)[:r]
    ids = torch.randint(-1, num_logical + 2, (num_tokens, top_k), generator=g)
    ids = ids.to(torch.int32)
    ids[-1] = -1  # a padding row
    unpadded = torch.tensor([max(num_tokens - 1, 0)], dtype=torch.int32)
    on = torch.tensor([1], dtype=torch.int32)
    load_a = torch.zeros(num_physical, dtype=torch.int64)
    load_b = torch.zeros(num_physical, dtype=torch.int64)
    mapped = base_router._eplb_map_and_record_triton(
        ids, l2p, replicas, load_a, on, unpadded
    )
    ref, counts = sentinel_ids_and_counts(mapped, sentinel, 16)
    fused = base_router._eplb_map_and_record_triton(
        ids, l2p, replicas, load_b, on, unpadded, sentinel=sentinel
    )
    assert fused.dtype == ref.dtype and torch.equal(fused, ref), num_tokens
    assert torch.equal(load_a, load_b)
    assert (ref == sentinel).any()
    cases += 1
print("ROUTE-FUSED-OK", cases)
"""


def test_route_fused_mapping_writes_the_sentinel_interpreted():
    """VLLM_XPU_MOE_ROUTE_FUSED: the EPLB mapping kernel with the sentinel gives
    the ids the mapping kernel and then the sentinel kernel give, and records
    the same expert load (replicas, unrouted, out-of-range and padding ids)."""
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    run = subprocess.run(
        [sys.executable, "-c", _INTERPRETED_ROUTE_FUSED],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0 and "ROUTE-FUSED-OK 4" in run.stdout, (
        run.stderr[-3000:] + run.stdout
    )


@pytest.mark.parametrize("ep_rank", [0, 2])
@pytest.mark.parametrize(
    "stub, marked, skips",
    [
        (_FusedMoeOpHandlesIds, True, True),
        (_FusedMoeOpHandlesIds, False, False),
        (_FusedMoeIdsValid, True, False),
    ],
)
def test_route_fused_apply(monkeypatch, ep_rank, stub, marked, skips):
    """VLLM_XPU_MOE_ROUTE_FUSED=1 under IDS_VALID: ids the router marked with
    this rank's sentinel go to the fused MoE as they are, with an int32
    rows_per_expert it clears itself, and no sentinel kernel; without the mark,
    or with a kernels build that does not clear the counts, the kernel runs."""
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    monkeypatch.setenv("VLLM_XPU_MOE_ROUTE_FUSED", "1")
    monkeypatch.setattr(xpu_moe, "XpuFusedMoe", stub, raising=False)
    xpu_moe.fused_moe_ep_contract.cache_clear()
    num_global, num_local = 256, 64
    sentinel = num_global - 1 if ep_rank == 0 else 0
    kernel_calls = []

    def reference_sentinel(topk_ids, sentinel_, num_local_experts):
        kernel_calls.append(1)
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
    experts.ep_contract = xpu_moe.EPIdsContract.IDS_VALID
    experts.moe_config = SimpleNamespace(ep_size=4, ep_rank=ep_rank)
    ids = torch.tensor([[3, sentinel], [sentinel, 7]], dtype=torch.int32)
    if marked:
        ids.xpu_ids_sentinel = sentinel
    try:
        experts.apply(
            torch.zeros(2, 8),
            torch.zeros(2, 8),
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
    finally:
        xpu_moe.fused_moe_ep_contract.cache_clear()
    assert torch.equal(seen["topk_ids"], ids)
    assert seen["ids_valid"] is True
    assert seen["rows_per_expert"].dtype == torch.int32
    assert seen["rows_per_expert"].shape == (num_local,)
    assert kernel_calls == ([] if skips else [1])


@pytest.mark.parametrize(
    "ep_size, dp_size, op_ids, stub, expected",
    [
        (4, 1, "0", _FusedMoeOpHandlesIds, "rank"),
        (4, 1, "1", _FusedMoeOpHandlesIds, None),
        (4, 1, "1", _FusedMoeIdsValid, "rank"),
        (1, 1, "0", _FusedMoeOpHandlesIds, None),
        (4, 2, "0", _FusedMoeOpHandlesIds, None),
    ],
)
@pytest.mark.parametrize("rank", [0, 3])
def test_eplb_route_sentinel(
    monkeypatch, ep_size, dp_size, op_ids, stub, expected, rank
):
    """The router writes the experts' sentinel only where the experts would
    rewrite unrouted ids themselves: expert parallelism, no data parallel
    dispatch, and not the op-filtered EPLB contract."""
    import vllm.distributed as dist
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    monkeypatch.setenv("VLLM_XPU_MOE_EPLB_OP_IDS", op_ids)
    monkeypatch.setattr(xpu_moe, "XpuFusedMoe", stub, raising=False)
    monkeypatch.setattr(
        dist,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=ep_size, rank_in_group=rank),
    )
    monkeypatch.setattr(
        dist, "get_dp_group", lambda: SimpleNamespace(world_size=dp_size)
    )
    xpu_moe.fused_moe_ep_contract.cache_clear()
    try:
        got = xpu_moe.eplb_route_sentinel(256)
    finally:
        xpu_moe.fused_moe_ep_contract.cache_clear()
    if expected is None:
        assert got is None
    else:
        assert got == (255 if rank == 0 else 0)

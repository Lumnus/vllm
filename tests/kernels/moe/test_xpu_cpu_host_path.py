# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU tier's host path around the op, per layer call: the calls listed
as arrays (numpy, or the kernels' select op in one pass), each expert taken
from the host cache once, the per-pair buffers and the pair map kept between
layer calls. The op receives what the per-call lists gave it, bit for bit."""

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

from .test_xpu_cpu_experts import REAL_OP, _host_env, _reference_op
from .xpu_store_harness import HIDDEN

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def op(monkeypatch):
    """The host op, real when loaded, else the torch reference."""
    if not REAL_OP:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    cx.reset_runtime()
    yield REAL_OP
    cx.reset_runtime()


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _per_call_lists(tiers, ids):
    """The lists as the per-call code made them: a store key per call, the
    flat pair index, the calls per tier."""
    tier = tiers._tier_of_global[ids]
    tok, k = np.nonzero(tier == HOST)
    slots = tiers._slot_of_global[ids[tok, k]]
    layer = tiers.store_binding.store_layer
    keys = [(layer, int(e)) for e in tiers.slot_logical[slots]]
    counts = {
        "card": int((tier == FAST).sum()),
        "pinned": int((tier == POOL).sum()),
        "cpu": int((tier == HOST).sum()),
    }
    return keys, tok.astype(np.int64) * ids.shape[1] + k, counts


def _routings(num_global):
    rng = np.random.default_rng(3)
    for tokens in (1, 5, 64):
        for dtype in (np.int32, np.int64):
            yield rng.integers(0, num_global, size=(tokens, 2)).astype(dtype)


def test_host_selection_lists_what_the_per_call_lists_did(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    for tiers in env.tiers:
        for ids in _routings(tiers.num_global):
            keys, pair, counts = _per_call_lists(tiers, ids)
            sel = tiers.host_selection(ids)
            assert sel.keys() == keys
            assert sel.pair.tolist() == pair.tolist()
            assert sel.counts == counts
            assert tiers.host_pairs(ids)[0] == keys
            assert tiers.host_pairs(ids)[1].tolist() == pair.tolist()
    tstore.reset_sessions()


def test_each_expert_is_taken_from_the_cache_once(monkeypatch, tmp_path, op):
    """A layer call whose calls repeat experts acquires each once, in order
    of first use (the order the per-call list showed the cache), and the op
    gets every call's row as before."""
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    rt = tiers.cpu
    seen, calls = [], []
    acquire = rt.cache.acquire

    def spy(keys):
        seen.append(list(keys))
        return acquire(keys)

    run_op = rt.run_op

    def spy_op(out, x, units, call_unit, *rest):
        calls.append(call_unit.tolist())
        return run_op(out, x, units, call_unit, *rest)

    monkeypatch.setattr(rt.cache, "acquire", spy)
    monkeypatch.setattr(rt, "run_op", spy_op)
    ids = torch.tensor([[6, 5], [5, 6], [7, 5]], dtype=torch.int32)
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(4)).bfloat16()
    rows = torch.empty(6, HIDDEN, dtype=torch.bfloat16)
    pair = rt.compute_pairs(tiers, ids, x, rows)
    keys, _, _ = _per_call_lists(tiers, ids.numpy())
    assert seen == [list(dict.fromkeys(keys))]
    row_of = {key: rt.cache.row_of(key) for key in keys}
    assert calls == [[row_of[key] for key in keys]]
    assert pair.tolist() == [0, 1, 2, 3, 4, 5]
    tstore.reset_sessions()


def test_the_pair_map_goes_through_a_kept_pinned_buffer(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    rt = tiers.cpu
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(3, HIDDEN, generator=gen).bfloat16()
    w = torch.full((3, 2), 0.5)
    want = None
    for ids in (
        torch.tensor([[0, 5], [6, 3], [7, 5]], dtype=torch.int32),
        torch.tensor([[5, 1], [2, 3], [0, 7]], dtype=torch.int32),
    ):
        pending = rt.post(tiers, x, ids, w)
        host = rt.finish_pairs(pending, x.device)
        b = pending.buffers
        _, pair, _ = _per_call_lists(tiers, ids.numpy())
        expect = np.full(6, -1, dtype=np.int32)
        expect[pair] = np.arange(pair.size)
        assert host.row_map.reshape(-1).tolist() == expect.tolist()
        if want is None:
            want = b.row_map.data_ptr()
        assert b.row_map.data_ptr() == want  # the same buffer, refilled
    tstore.reset_sessions()


def _select_or_skip():
    if cx._select_op() is None:
        pytest.skip("the loaded kernels have no int4_moe_cpu_select")


def test_the_select_op_lists_what_numpy_lists(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    _select_or_skip()
    for tiers in env.tiers:
        rt = tiers.cpu
        for ids in _routings(tiers.num_global):
            want = tiers.host_selection(ids)
            row_map = torch.full(ids.shape, 7, dtype=torch.int32)
            got = rt._select(tiers, torch.from_numpy(ids), ids, row_map)
            assert got.pair.tolist() == want.pair.tolist()
            assert got.logical.tolist() == want.logical.tolist()
            assert got.counts == want.counts and got.layer == want.layer
            assert got.distinct == list(dict.fromkeys(want.logical.tolist()))
            expect = np.full(ids.size, -1, dtype=np.int32)
            expect[want.pair] = np.arange(want.pair.size)
            assert row_map.reshape(-1).tolist() == expect.tolist()
        x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
        with pytest.raises(cx.CpuExpertError, match="outside the 8 physical"):
            rt.compute(
                tiers,
                torch.tensor([[1, 8]], dtype=torch.int32),
                torch.ones(1, 2),
                x,
                torch.empty_like(x),
            )
    tstore.reset_sessions()


def test_the_outputs_are_the_same_with_and_without_the_select_op(
    monkeypatch, tmp_path, op
):
    env = _host_env(monkeypatch, tmp_path)
    _select_or_skip()
    tiers = env.tiers[1]
    rt = tiers.cpu
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(64, HIDDEN, generator=gen).bfloat16()
    w = torch.rand(64, 2, generator=gen)
    for ids in _routings(tiers.num_global):
        if ids.shape[0] != 64:
            continue
        t = torch.from_numpy(ids)
        got = {}
        for name, select in (("op", rt._select_op), ("numpy", None)):
            monkeypatch.setattr(rt, "_select_op", select)
            rows = torch.empty(128, HIDDEN, dtype=torch.bfloat16)
            pair = rt.compute_pairs(tiers, t, x, rows)
            out = torch.empty_like(x)
            assert rt.compute(tiers, t, w, x, out)
            got[name] = (pair, rows[: len(pair)].clone(), out)
        assert got["op"][0].tolist() == got["numpy"][0].tolist()
        assert _same(got["op"][1], got["numpy"][1])
        assert _same(got["op"][2], got["numpy"][2])
    tstore.reset_sessions()


def test_the_select_switch(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_EXPERTS_SELECT", "numpy")
    assert cx._select_op() is None
    monkeypatch.setenv("VLLM_CPU_EXPERTS_SELECT", "fast")
    with pytest.raises(cx.CpuExpertError, match="'op' or 'numpy'"):
        cx._select_op()
    monkeypatch.delenv("VLLM_CPU_EXPERTS_SELECT")
    ns = cx._ops()
    loaded = ns is not None and hasattr(ns, "int4_moe_cpu_select")
    assert (cx._select_op() is not None) == loaded


def test_the_pairs_read_their_inputs_in_place(monkeypatch, tmp_path, op):
    """With an op that takes call_x_row, a per-pair call hands it x as it is
    (no gathered copy) and the rows are the same bits as through the
    gather."""
    env = _host_env(monkeypatch, tmp_path)
    if not cx.op_takes_x_row():
        pytest.skip("the loaded kernels' op has no call_x_row")
    tiers = env.tiers[1]
    rt = tiers.cpu
    assert rt._x_row
    gen = torch.Generator().manual_seed(8)
    x = torch.randn(64, HIDDEN, generator=gen).bfloat16()
    seen = []
    run_op = rt.run_op

    def spy(out, xs, *rest):
        seen.append(xs.shape[0])
        return run_op(out, xs, *rest)

    monkeypatch.setattr(rt, "run_op", spy)
    for ids in _routings(tiers.num_global):
        if ids.shape[0] != 64:
            continue
        t = torch.from_numpy(ids)
        got = {}
        for in_place in (True, False):
            rt._x_row = in_place
            seen.clear()
            rows = torch.empty(128, HIDDEN, dtype=torch.bfloat16)
            pair = rt.compute_pairs(tiers, t, x, rows)
            assert seen == [64 if in_place else len(pair)]
            got[in_place] = (pair, rows[: len(pair)].clone())
        assert got[True][0].tolist() == got[False][0].tolist()
        assert _same(got[True][1], got[False][1])
    tstore.reset_sessions()

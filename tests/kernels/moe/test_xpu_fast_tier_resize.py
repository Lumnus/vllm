# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert tiers' card tier resized while serving (--expert-store-config
fast_rows_max), on the CPU: the card rows in a reserved device range (the
VMM ops faked over host memory), the slots moved in and out by load with the
pinned rows or the store as the source, the rows' memory mapped and
released, the worker's runtime control op, and the pinned rows laid out
again when the pinned bound moves (pinned_release)."""

import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.runtime_control as wrc
from tests.v1.worker.test_xpu_extensible_kv_cache import (  # noqa: F401
    GRANULE,
    fake_xpu,
)
from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.layers.fused_moe import xpu_fast_tier as ft
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

from .test_xpu_cpu_experts import CPU_CFG, REAL_OP, _reference_op, _settle
from .xpu_store_harness import FAST as LOAD_FAST
from .xpu_store_harness import E, expert_in_store, run

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def op(monkeypatch):
    if not REAL_OP:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    cx.reset_runtime()
    yield REAL_OP
    cx.reset_runtime()


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, fake_xpu):  # noqa: F811
    monkeypatch.setattr(xt, "_RUNTIME_SETTINGS_REGISTERED", False)
    monkeypatch.setattr(wrc, "_TIER_SETTINGS", {})
    monkeypatch.setattr(xt, "_LIVE_TIERS", weakref.WeakSet())
    # The fake VMM ops serve device 0 from host memory.
    monkeypatch.setattr(ft.FastRows, "device_override", torch.device("xpu", 0))
    yield
    tstore.reset_sessions()


def _env(monkeypatch, tmp_path, fast_max=6, pinned=None, **extra):
    """A store built by one start, then the engine under test filling from
    it with a movable card tier (and CPU experts with ``pinned``)."""
    run(monkeypatch, tmp_path, store=True)
    xt._LIVE_TIERS.clear()
    cfg = dict(extra, fast_rows_max=fast_max)
    if pinned is not None:
        cfg = dict(CPU_CFG, **cfg)
    return run(monkeypatch, tmp_path, store=True, pinned=pinned, store_extra=cfg)


def _load(tiers, hot):
    """EPLB's view of the slots' load, hottest first in ``hot``."""
    load = np.zeros(tiers.num_local)
    for i, s in enumerate(hot):
        load[s] = 100 - i
    return load


def _assert_serves_the_store(env, tiers):
    summary = tiers.scrub(np.arange(E), full=True)
    assert summary["mismatches"] == 0, summary["first"]
    assert summary["maps_equal"] and summary["placement_equal"]


def _fast_ptrs(tiers):
    fast = tiers._impls[0]
    return [getattr(fast, f).data_ptr() for f, _ in xt._TIER_FIELDS]


def test_the_card_tier_sits_in_a_reserved_range(monkeypatch, tmp_path, fake_xpu):  # noqa: F811
    env = _env(monkeypatch, tmp_path)
    for tiers, layer in zip(env.tiers, env.layers):
        fast = tiers._impls[0]
        # the kernel sees the capacity: graphs captured now stay valid
        assert fast.num_experts == 6 and fast.local_experts_num == 6
        for impl_field, _ in xt._TIER_FIELDS:
            assert getattr(fast, impl_field).shape[0] == 6
        # the layer's parameters are the kernel's tensors (rebound, not copied)
        assert (
            layer.w13_weight is fast.w13 and layer.w13_scales is fast.gemm1_wei_scales
        )
        assert tiers.fast_rows.rows == LOAD_FAST
        _assert_serves_the_store(env, tiers)
    assert ft.rows_now() == LOAD_FAST and ft.rows_range() == (3, 6)
    # four fields, each within one granule at these sizes, per layer
    assert fake_xpu.ops.physical_bytes == ft.mapped_bytes() == 2 * 4 * GRANULE


def test_grow_moves_the_hottest_slots_onto_the_card(monkeypatch, tmp_path):
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    p = tiers.placement
    tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    ptrs = _fast_ptrs(tiers)
    stats = tiers.resize_fast(5)
    # 4 and 3 were pinned: copied from their pinned rows, nothing read
    assert sorted(np.flatnonzero(p.tier == FAST).tolist()) == [3, 4, 5, 6, 7]
    assert stats["units_read"] == 0 and stats["to_card"] == 2
    assert sorted(p.row[p.tier == FAST].tolist()) == [0, 1, 2, 3, 4]
    assert p.num_fast == 5 and tiers.num_fast == 5 and not p.fast_free
    assert _fast_ptrs(tiers) == ptrs  # the base pointers never move
    _assert_serves_the_store(env, tiers)
    for s in (3, 4):
        assert tiers.fast_map[s] == p.row[s] and tiers.slow_map[s] == -1


def test_shrink_moves_the_coldest_out_and_frees_the_top_rows(monkeypatch, tmp_path):
    env = _env(monkeypatch, tmp_path, fast_max=6, fast_rows_min=1)
    tiers = env.tiers[0]
    p = tiers.placement
    tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    tiers.resize_fast(5)
    # the load turns: 3 and 4 (rows 3, 4 or below) stay hottest
    tiers._slot_load = _load(tiers, [3, 4, 7, 6, 5, 2, 1, 0])
    stats = tiers.resize_fast(2)
    assert sorted(np.flatnonzero(p.tier == FAST).tolist()) == [3, 4]
    assert set(p.row[p.tier == FAST].tolist()) == {0, 1}
    # 5, 6, 7 went to pinned rows, read from the store (the card is not read)
    assert stats["units_read"] == 3 and stats["off_card"] == 3
    assert p.num_fast == 2 and tiers.fast_rows.rows == 2
    _assert_serves_the_store(env, tiers)
    # the free pinned rows are EPLB's spares again
    assert p.pool.free_count >= p.spare


def test_grow_and_shrink_back_keep_the_sets_for_the_same_load(monkeypatch, tmp_path):
    """The same load gives the same tier sets back after a grow and a
    shrink: the same placement, so the same sums in every combine mode."""
    env = _env(monkeypatch, tmp_path)
    for tiers in env.tiers:
        tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    before = [t.placement.tier.copy() for t in env.tiers]
    ft.apply(6)
    ft.apply(LOAD_FAST)
    for tiers, tier in zip(env.tiers, before):
        assert np.array_equal(tiers.placement.tier, tier)
        _assert_serves_the_store(env, tiers)


def test_a_rearrangement_after_a_resize_keeps_the_new_size(monkeypatch, tmp_path):
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    tiers.resize_fast(5)
    load = np.array([100, 90, 1, 1, 1, 1, 1, 80], dtype=float)
    tiers.commit(tiers.prepare(np.arange(E), np.arange(E), load, 1.1))
    p = tiers.placement
    assert int((p.tier == FAST).sum()) == 5
    assert p.tier[0] == FAST and p.tier[1] == FAST
    _assert_serves_the_store(env, tiers)


def test_without_cpu_experts_the_floor_is_the_pinned_rows(monkeypatch, tmp_path):
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    assert tiers.fast_range() == (LOAD_FAST, 6)
    with pytest.raises(ValueError, match="3..6"):
        tiers.resize_fast(2)
    xt._LIVE_TIERS.clear()
    tstore.reset_sessions()
    env = _env(monkeypatch, tmp_path / "b", fast_rows_min=1)
    tiers = env.tiers[0]
    # two more pinned rows per layer pay for a floor of one
    assert tiers.num_pool_rows == E - LOAD_FAST + env.spare + 2
    assert tiers.fast_range() == (1, 6)
    tiers.resize_fast(1)
    assert int((tiers.placement.tier == FAST).sum()) == 1
    assert tiers.placement.pool.free_count >= tiers.placement.spare
    _assert_serves_the_store(env, tiers)


def test_with_cpu_experts_slots_leave_the_card_to_the_cpu(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path, pinned=2)
    tiers = env.tiers[0]
    p = tiers.placement
    # card 5 6 7, pinned 4 3, CPU 2 1 0
    tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    assert tiers.fast_range() == (1, 6)
    _settle(tiers.cpu.cache)
    stats = tiers.resize_fast(5)
    # 4 and 3 come from their pinned rows; the pinned rows then take the
    # hottest CPU slots (2, 1), read from the host cache or the store
    assert sorted(np.flatnonzero(p.tier == FAST).tolist()) == [3, 4, 5, 6, 7]
    assert sorted(np.flatnonzero(p.tier == POOL).tolist()) == [1, 2]
    assert p.tier[0] == HOST and stats["units_read"] == 2
    _assert_serves_the_store(env, tiers)
    stats = tiers.resize_fast(1)
    # 5 stays (hottest); 3 4 7 6 leave, coldest first: each is hotter than
    # the coldest pinned slot, so it takes a pinned row and that slot goes to
    # the CPU; in the end 6 and 7 hold the pinned rows (read from the store),
    # 3 4 and 1 2 0 are computed on the CPU
    assert np.flatnonzero(p.tier == FAST).tolist() == [5]
    assert sorted(np.flatnonzero(p.tier == POOL).tolist()) == [6, 7]
    assert stats["units_read"] == 2
    assert int((p.tier == HOST).sum()) == 5
    assert p.num_fast == 1 and tiers.num_host == 5
    _assert_serves_the_store(env, tiers)
    # a slot computed on the CPU gives the reference result
    host_slot = int(np.flatnonzero(p.tier == HOST)[0])
    x = torch.randn(1, 256, generator=torch.Generator().manual_seed(3)).bfloat16()
    out = torch.empty_like(x)
    assert tiers.cpu.compute(
        tiers,
        torch.tensor([[host_slot, 5]], dtype=torch.int32),
        torch.ones(1, 2),
        x,
        out,
    )


def test_the_resize_waits_for_a_prepared_eplb_move(monkeypatch, tmp_path):
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    before = tiers.placement.tier.copy()
    tiers._eplb_pending = True
    assert "EPLB move is prepared" in ft.check(5)
    with pytest.raises(RuntimeError, match="EPLB move is prepared"):
        tiers.resize_fast(5)
    assert np.array_equal(tiers.placement.tier, before)
    tiers._eplb_pending = False


def test_a_failure_part_way_puts_every_layer_back(monkeypatch, tmp_path, fake_xpu):  # noqa: F811
    env = _env(monkeypatch, tmp_path)
    for tiers in env.tiers:
        tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    before = [t.placement.tier.copy() for t in env.tiers]
    mapped = fake_xpu.ops.physical_bytes
    real = xt.ExpertTiers.resize_fast

    def flaky(self, rows, **kw):
        if self is env.tiers[1] and rows == 6:
            raise RuntimeError("injected: the drive went away")
        return real(self, rows, **kw)

    monkeypatch.setattr(xt.ExpertTiers, "resize_fast", flaky)
    with pytest.raises(RuntimeError, match="injected"):
        ft.apply(6)
    assert ft.rows_now() == LOAD_FAST
    for tiers, tier in zip(env.tiers, before):
        assert tiers.fast_rows.rows == LOAD_FAST
        assert np.array_equal(tiers.placement.tier, tier)
        _assert_serves_the_store(env, tiers)
    assert fake_xpu.ops.physical_bytes == mapped


def test_a_grow_the_device_cannot_back_moves_nothing(monkeypatch, tmp_path, fake_xpu):  # noqa: F811
    """The second layer's rows cannot be mapped: the first layer's mapping
    is given back and no slot moved on either layer."""
    env = _env(monkeypatch, tmp_path)
    before = [t.placement.tier.copy() for t in env.tiers]
    mapped = fake_xpu.ops.physical_bytes
    real = ft.FastRows.commit

    def commit(self, rows):
        if self is env.tiers[1].fast_rows:
            raise RuntimeError("UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY")
        return real(self, rows)

    monkeypatch.setattr(ft.FastRows, "commit", commit)
    with pytest.raises(RuntimeError, match="OUT_OF_DEVICE_MEMORY"):
        ft.apply(6)
    assert fake_xpu.ops.physical_bytes == mapped
    for tiers, tier in zip(env.tiers, before):
        assert tiers.fast_rows.rows == LOAD_FAST
        assert np.array_equal(tiers.placement.tier, tier)


def test_fast_rows_release_whole_objects_above_the_rows_kept(monkeypatch, fake_xpu):  # noqa: F811
    """Rows of 1 MiB: two per 2 MiB object; a release gives back the objects
    wholly above the rows that stay, and the rows below keep their bytes."""
    like = [
        torch.zeros(1, 1 << 20, dtype=torch.uint8),
        torch.zeros(1, 4096, dtype=torch.bfloat16),
    ]
    rows = ft.FastRows(like, 16, GRANULE)
    views = rows.views(like)
    assert views[0].shape == (16, 1 << 20) and views[1].dtype == torch.bfloat16
    assert rows.bytes_to(9) == 5 * GRANULE + GRANULE  # w 9 MiB -> 5 objects, s 1
    rows.commit(9)
    assert fake_xpu.ops.physical_bytes == 6 * GRANULE
    views[0][:9] = torch.arange(9, dtype=torch.uint8)[:, None]
    base = views[0].data_ptr()
    assert rows.releasable_bytes(4) == 3 * GRANULE  # objects [4,6) [6,8) [8,9)
    released = rows.release_to(4)
    assert released == 3 * GRANULE and rows.rows == 4
    assert fake_xpu.ops.physical_bytes == 3 * GRANULE
    assert torch.equal(views[0][:4, 0], torch.arange(4, dtype=torch.uint8))
    rows.commit(10)
    assert rows.views(like)[0].data_ptr() == base
    assert torch.equal(views[0][:4, 0], torch.arange(4, dtype=torch.uint8))
    # a commit the device cannot back maps nothing and keeps the count
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    fake_xpu.ops.fail_create = True
    physical = fake_xpu.ops.physical_bytes
    with pytest.raises(RuntimeError, match="OUT_OF_DEVICE_MEMORY"):
        rows.commit(16)
    assert fake_xpu.ops.physical_bytes == physical and rows.rows == 10
    fake_xpu.ops.fail_create = False
    rows.buffer.free()


def test_the_control_surface_moves_the_card_tier(monkeypatch, tmp_path):
    env = _env(monkeypatch, tmp_path)
    for tiers in env.tiers:
        tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    kv = SimpleNamespace(
        external_mapped_bytes=[],
        refresh_torch_memory_limit=lambda: None,
        buffer=SimpleNamespace(device=torch.device("cpu")),
    )
    worker = SimpleNamespace(
        rank=0,
        model_runner=SimpleNamespace(eplb_state=None, extensible_kv_cache=kv),
        cache_config=SimpleNamespace(),
    )
    monkeypatch.setattr(wrc, "_ep_rank", lambda w: 0)
    monkeypatch.setattr(ft, "_free_device_bytes", lambda device: 1 << 40)
    state = wrc.handle(worker, "state", args={"only": ["tiers"], "fast_rows": [5]})
    fast = state["tiers"]["fast"]
    assert fast["rows"] == 3 and fast["min"] == 3 and fast["max"] == 6
    assert fast["capacity"] == 6 and fast["layers"] == 2
    assert fast["bytes_to"] == {"5": 0}  # within the granules mapped at load
    assert state["tiers"]["resizable"] and state["tiers"]["resize_blocked"] is None
    assert (
        "outside 3..6"
        in wrc.handle(worker, "check", "tiers.fast", {"rows": 7})["refusal"]
    )
    assert (
        "integer" in wrc.handle(worker, "check", "tiers.fast", {"rows": "5"})["refusal"]
    )
    assert (
        "unknown"
        in wrc.handle(worker, "check", "tiers.fast", {"rows": 5, "x": 1})["refusal"]
    )
    reply = wrc.handle(worker, "apply", "tiers.fast", {"rows": 5})
    assert reply["ok"], reply
    assert reply["changes"] == {"fast_rows": [3, 5]}
    assert reply["result"]["fast"]["rows"] == [3, 5]
    assert all(int((t.placement.tier == FAST).sum()) == 5 for t in env.tiers)
    # the KV cache's torch cap leaves out what the card tier maps from now
    assert len(kv.external_mapped_bytes) == 1 and kv.external_mapped_bytes[0]() == 0
    undone = wrc.handle(worker, "restore", "tiers.fast")
    assert undone["ok"]
    assert all(int((t.placement.tier == FAST).sum()) == 3 for t in env.tiers)
    # too little device memory for a grow: refused with the numbers
    monkeypatch.setattr(ft, "bytes_to", lambda rows: 1 << 30)
    monkeypatch.setattr(ft, "_free_device_bytes", lambda device: 1 << 20)
    refusal = wrc.handle(worker, "check", "tiers.fast", {"rows": 5})["refusal"]
    assert "KV cache must give them back first" in refusal
    # refused while EPLB rearranges
    worker.model_runner.eplb_state = SimpleNamespace(
        model_states={"m": SimpleNamespace(rebalanced=True)}
    )
    refusal = wrc.handle(worker, "check", "tiers.fast", {"rows": 4})["refusal"]
    assert "in flight" in refusal


def test_a_fixed_card_tier_refuses_with_the_reason(monkeypatch, tmp_path):
    run(monkeypatch, tmp_path, store=True)
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(eplb_state=None))
    refusal = wrc.handle(worker, "check", "tiers.fast", {"rows": 4})["refusal"]
    assert "fast_rows_max" in refusal
    state = wrc.handle(worker, "state", args={"only": ["tiers"]})
    assert state["tiers"]["fast"] is None
    assert "fast_rows_max" in state["tiers"]["resize_blocked"]


def test_the_kv_cap_counts_the_card_tier_from_registration(monkeypatch):
    """refresh_torch_memory_limit subtracts the change of the card tier's
    mapped bytes since it registered, not what it held at load."""
    from vllm.v1.worker.extensible_kv_cache import ExtensibleKVCache

    held = {"bytes": 100}
    kv = SimpleNamespace(
        torch_memory_budget=1000,
        physical_bytes=300,
        external_mapped_bytes=[],
        torch_memory_limit=None,
    )
    kv._apply_torch_memory_limit = lambda n: setattr(kv, "torch_memory_limit", n)
    monkeypatch.setattr(ft, "mapped_bytes", lambda: held["bytes"])
    ft.register_with_kv_cache(kv)
    ft.register_with_kv_cache(kv)  # once only
    assert len(kv.external_mapped_bytes) == 1
    assert ExtensibleKVCache.refresh_torch_memory_limit(kv) == 700
    held["bytes"] = 160
    assert ExtensibleKVCache.refresh_torch_memory_limit(kv) == 640


# -- the pinned rows follow the bound (pinned_release) ---------------------------


def test_lowering_the_bound_lays_the_pinned_rows_out_smaller(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path, pinned=2, pinned_release=True)
    tiers = env.tiers[0]
    p = tiers.placement
    tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    rows_at_load = tiers.num_pool_rows
    assert tiers.pinned_bound_range() == (1, 5)
    weights = tiers.eplb_weights()
    xt.apply_host_tier_bound(1)
    for t in env.tiers:
        assert t.placement.num_pinned == 1
        assert t.num_pool_rows == t.pool_rows_for(1) == rows_at_load - 1
        for tensor in t.slow.values():
            assert tensor.shape[0] == t.num_pool_rows
        slow = t._impls[1]
        assert slow.num_experts == t.num_pool_rows
        assert slow.w13.shape[0] == t.num_pool_rows
        _assert_serves_the_store(env, t)
    assert np.flatnonzero(p.tier == POOL).tolist() == [4]
    assert p.row[4] == 0  # laid out from row 0
    # EPLB's weights keep their shapes and no longer hold the old rows
    assert weights[0].parts[1].device.type == "meta"
    assert weights[0].shape[0] == E
    # raising past the rows sized at load lays out more rows
    _settle(tiers.cpu.cache)
    xt.apply_host_tier_bound(5)
    for t in env.tiers:
        assert (
            t.placement.num_pinned == 5 and int((t.placement.tier == HOST).sum()) == 0
        )
        assert t.num_pool_rows == t.pool_rows_for(5)
        _assert_serves_the_store(env, t)
    xt.apply_host_tier_bound(2)
    assert all(t.placement.num_pinned == 2 for t in env.tiers)


def test_without_pinned_release_the_rows_stay(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path, pinned=2)
    rows = env.tiers[0].num_pool_rows
    shapes = {k: v.shape for k, v in env.tiers[0].slow.items()}
    xt.apply_host_tier_bound(1)
    assert env.tiers[0].num_pool_rows == rows
    assert {k: v.shape for k, v in env.tiers[0].slow.items()} == shapes


def test_expert_in_store_is_what_the_card_rows_hold(monkeypatch, tmp_path):
    """Bytes on the card come from the store or a pinned row, bit for bit."""
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    tiers.fill_to(np.arange(E), _load(tiers, [0, 1, 2, 3, 4, 5, 6, 7]), 1.1)
    tiers.resize_fast(6)
    fast = tiers._impls[0]
    p = tiers.placement
    for s in np.flatnonzero(p.tier == FAST):
        unit = expert_in_store(env, 1, int(tiers.slot_logical[s]))
        for impl_field, key in xt._TIER_FIELDS:
            got = xt._u8(getattr(fast, impl_field))[p.row[s]]
            assert torch.equal(got, unit[key].reshape(-1).view(torch.uint8))

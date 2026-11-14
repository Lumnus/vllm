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
    grow_ranks,
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
    ft.reset_adoption_summary()
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
    grown = ft.apply(6)
    back = ft.apply(LOAD_FAST)
    # the digest the card run compares: equal before the grow and after
    assert grown["placement"][0] != grown["placement"][1]
    assert back["placement"][1] == grown["placement"][0]
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
        model_states={
            "m": SimpleNamespace(
                rebalanced=True,
                pending_result=None,
                model_name="m",
                model=SimpleNamespace(num_moe_layers=2),
            )
        }
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


# -- what the card tier costs: the rows in use, once -----------------------------


def _field_bytes(tiers):
    fast = tiers._impls[0]
    return [
        getattr(fast, f)[: tiers.num_fast].numel() * getattr(fast, f).element_size()
        for f, _ in xt._TIER_FIELDS
    ]


def test_the_move_at_load_gives_up_every_old_storage(monkeypatch, tmp_path, fake_xpu):  # noqa: F811
    """Boot sizing: what the card tier holds after the move is the rows in
    use (whole granules), not the reserved range, and every storage the rows
    left is gone (nothing else keeps its memory)."""
    env = _env(monkeypatch, tmp_path)
    summary = ft.adoption_summary()
    assert summary["layers"] == len(env.tiers) == 2 and summary["logged"]
    assert summary["stale"] == 0 and summary["released"] == 0
    assert summary["mapped"] == ft.mapped_bytes() == fake_xpu.ops.physical_bytes
    # the old storages were the rows in use, field by field
    assert summary["old"] == sum(sum(_field_bytes(t)) for t in env.tiers)
    for tiers in env.tiers:
        rows = tiers.fast_rows
        assert rows.rows == LOAD_FAST < rows.capacity
        per_field = [ft._round_up(LOAD_FAST * rb, GRANULE) for rb in rows.row_bytes]
        assert rows.mapped_bytes == sum(per_field)


def _process_in_place(self) -> None:
    """The harness's post-load step done the way the real one is on this
    model (the weights keep their storage: `replace_parameter` copies a
    same-size result into the loaded storage, the int4 repack is in place),
    so the card rows stay where construction put them."""
    from types import SimpleNamespace as NS

    for i, tiers in enumerate(self.tiers):
        layer = self.layers[i]
        for key in ("w13", "w2"):
            q = getattr(layer, f"{key}_qweight").data
            kernel = q.transpose(1, 2).contiguous().view(torch.uint8)
            raw = q.view(-1).view(torch.uint8)
            raw.copy_(kernel.reshape(-1))
            setattr(
                layer,
                f"{key}_weight",
                torch.nn.Parameter(raw.view(kernel.shape), requires_grad=False),
            )
            s = getattr(layer, f"{key}_scales")
            t = s.data.transpose(1, 2).contiguous()
            s.data.view(-1).copy_(t.reshape(-1))
            s.data = s.data.view(-1).view(t.shape)
        tiers.finalize(layer, NS(fused_experts=NS(tiers=None)))
    for i, tiers in enumerate(self.tiers):
        layer = self.layers[i]
        tiers.impls(
            layer.w13_weight, layer.w13_scales, layer.w2_weight, layer.w2_scales,
            2, "silu", None,
        )  # fmt: skip


def _fake_kernel_in_place(self, w13, w13_scales, w13_bias, w2, w2_scales, w2_bias,
                          **kw):  # fmt: skip
    """The int4 relayout in place, as the kernels do it on xe2."""
    if not hasattr(w13, "xpu_fused_moe"):
        for w in (w13, w2):
            w.data.copy_(w.data ^ 0x88)
            w.data = w.data.view(torch.int8)
        w13.xpu_fused_moe = True
    self.w13, self.w2 = w13, w2
    self.gemm1_wei_scales, self.gemm2_wei_scales = w13_scales, w2_scales
    self.expert_map = kw["expert_map"]
    self.num_experts = kw["num_experts"]


def test_the_rows_go_into_the_range_before_the_weights_load(
    monkeypatch,
    tmp_path,
    fake_xpu,  # noqa: F811
):
    """Construction puts the card rows' parameters into their range; loading
    and the post-load step write there; weight processing finds them there
    (nothing copied, no torch storage freed after loading), and the tier
    serves the store, grows and shrinks."""
    from . import xpu_store_harness as hx

    monkeypatch.setattr(hx.Env, "process", _process_in_place)
    monkeypatch.setattr(hx.FakeXpuFusedMoe, "__init__", _fake_kernel_in_place)
    env = _env(monkeypatch, tmp_path)
    summary = ft.adoption_summary()
    assert summary["when"] == {"at construction"} and summary["logged"]
    assert summary["layers"] == 2 and summary["stale"] == 0
    assert fake_xpu.ops.physical_bytes == ft.mapped_bytes() == summary["mapped"]
    for tiers, layer in zip(env.tiers, env.layers):
        fast = tiers._impls[0]
        kernel = [getattr(fast, f) for f, _ in xt._TIER_FIELDS]
        assert tiers.fast_rows.holds(kernel)
        assert all(t.shape[0] == tiers.fast_rows.capacity for t in kernel)
        assert layer.w13_qweight.data_ptr() == fast.w13.data_ptr()
        _assert_serves_the_store(env, tiers)
    tiers = env.tiers[0]
    tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    tiers.fast_rows.commit(5)
    tiers.resize_fast(5)
    _assert_serves_the_store(env, tiers)
    tiers.resize_fast(LOAD_FAST)
    tiers.fast_rows.release_to(LOAD_FAST)
    _assert_serves_the_store(env, tiers)


def _moves_line(monkeypatch, caplog, torch_now, free_delta):
    """The moves' line and its WARNING for one layer, with torch's
    allocated/reserved at the line and the move's own free-memory change."""
    g = 1 << 30
    caplog.clear()
    ft.reset_adoption_summary()
    old = ft.OldStorages([torch.zeros(1024, dtype=torch.uint8)])  # freed now
    monkeypatch.setattr(ft, "_free_or_none", lambda device: 10 * g + free_delta)
    monkeypatch.setattr(ft, "_torch_or_none", lambda device: torch_now)
    ft.finish_move("l0", old, 0, torch.device("cpu"), 10 * g, 1)
    msgs = [r.getMessage() for r in caplog.records]
    line = [m for m in msgs if "moved into their reserved ranges" in m]
    warned = [m for m in msgs if "will be smaller" in m]
    return line, warned


def test_the_moves_line_reads_what_the_kv_sizing_will_pay(monkeypatch, caplog_vllm):
    """The line reports torch's cache that stays reserved after the moves and
    a cache release (what the KV sizing pays for), and warns only when that
    is large; a move's own free-memory change (at construction, reused by
    the rest of the load) raises no alarm."""
    g = 1 << 30
    # the card boot of construction moves: the moves' own deltas short by
    # 3.4 GiB, torch holding 0.29 GiB at the line: no WARNING
    line, warned = _moves_line(
        monkeypatch, caplog_vllm, (int(6.19 * g), int(6.48 * g)), -int(3.43 * g)
    )
    assert line and "keeps 0.29 GiB cached" in line[0] and not warned
    # the card boot of the move after loading: torch holds 4.1 GiB
    line, warned = _moves_line(monkeypatch, caplog_vllm, (7 * g, int(11.1 * g)), 0)
    assert warned and "keeps 4.10 GiB cached" in warned[0]
    assert "reserved 11.10 GiB" in warned[0]


def test_a_capacity_range_maps_only_the_rows_in_use(fake_xpu):  # noqa: F811
    """1 MiB rows, capacity 130, 98 in use: 49 objects of 2 MiB for the
    weights and one for the scales; the 32 rows above cost nothing."""
    like = [
        torch.zeros(1, 1 << 20, dtype=torch.uint8),
        torch.zeros(1, 4096, dtype=torch.bfloat16),
    ]
    rows = ft.FastRows(like, 130, GRANULE)
    rows.commit(98)
    assert fake_xpu.ops.physical_bytes == rows.mapped_bytes == (49 + 1) * GRANULE
    # a grow maps the rows it adds, once; a shrink gives the same back
    assert rows.bytes_to(114) == 8 * GRANULE
    rows.commit(114)
    assert fake_xpu.ops.physical_bytes == (57 + 1) * GRANULE
    assert rows.release_to(98) == 8 * GRANULE
    assert fake_xpu.ops.physical_bytes == (49 + 1) * GRANULE
    rows.buffer.free()


def test_an_old_storage_another_tensor_holds_is_released_in_place(
    monkeypatch,
    tmp_path,
    fake_xpu,  # noqa: F811
):
    """A tensor taken on the card rows before the move (an alias of w2) would
    keep their old storage, and its device memory, for good: the move gives
    it back in place and says who held it; the card tier still serves."""
    held = []
    adopt = xt.ExpertTiers._adopt_fast_rows

    def alias_then_adopt(self, fast):
        held.append(fast.w2.data)  # another holder of the old w2 storage
        return adopt(self, fast)

    monkeypatch.setattr(xt.ExpertTiers, "_adopt_fast_rows", alias_then_adopt)
    env = _env(monkeypatch, tmp_path)
    summary = ft.adoption_summary()
    w2_bytes = sum(_field_bytes(t)[1] for t in env.tiers)
    assert summary["stale"] == summary["released"] == w2_bytes > 0
    assert all(h.untyped_storage().nbytes() == 0 for h in held)
    assert "Tensor" in summary["holders"]
    for tiers in env.tiers:
        _assert_serves_the_store(env, tiers)


def test_an_old_storage_in_a_larger_buffer_is_reported_not_released(
    monkeypatch,
    tmp_path,
    fake_xpu,  # noqa: F811
):
    """Rows whose storage is a part of a larger buffer (shared with something
    else) are never released in place: reported only."""
    held = []
    adopt = xt.ExpertTiers._adopt_fast_rows

    def in_a_larger_buffer_then_adopt(self, fast):
        w2 = fast.w2
        n = w2.numel() * w2.element_size()
        big = torch.zeros(2 * n, dtype=torch.uint8)
        big[:n].copy_(w2.data.reshape(-1).view(torch.uint8))
        w2.data = big[:n].view(w2.dtype).view(w2.shape)
        held.append(big)
        return adopt(self, fast)

    monkeypatch.setattr(
        xt.ExpertTiers, "_adopt_fast_rows", in_a_larger_buffer_then_adopt
    )
    env = _env(monkeypatch, tmp_path)
    summary = ft.adoption_summary()
    assert summary["stale"] == sum(b.numel() for b in held) and summary["released"] == 0
    assert all(b.untyped_storage().nbytes() == b.numel() for b in held)
    for tiers in env.tiers:
        _assert_serves_the_store(env, tiers)


def test_a_grow_maps_only_bytes_that_are_there():
    """The ranks' memory check of a grow (before anything is mapped)."""
    g = 1 << 30
    room = dict(free=3 * g, reclaimable=0, floor=g, cap_room=None)
    assert ft.grow_refusal(0, 0, **room) is None
    # the KV cache released the bytes in this resize: no second charge
    assert ft.grow_refusal(2 * g, 2 * g, **room) is None
    # nothing released (a profile's step): torch keeps its floor
    assert ft.grow_refusal(2 * g, 0, **room) is None
    assert "floor" in ft.grow_refusal(3 * g, 0, **room)
    # torch's unused cache is not room for the rows themselves
    why = ft.grow_refusal(3 * g, 3 * g, free=2 * g, reclaimable=4 * g, floor=0,
                          cap_room=None)  # fmt: skip
    assert "2.00 GiB free" in why and "give them back first" in why
    # under torch's cap
    assert "cap" in ft.grow_refusal(g, 0, free=3 * g, reclaimable=0, floor=0,
                                     cap_room=g // 2)  # fmt: skip
    assert ft.grow_refusal(g, g, free=3 * g, reclaimable=0, floor=0,
                           cap_room=0) is None  # fmt: skip


def test_the_ranks_refuse_a_grow_the_device_cannot_back(monkeypatch, tmp_path):
    env = _env(monkeypatch, tmp_path)
    for tiers in env.tiers:
        tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    g = 1 << 30
    kv = SimpleNamespace(
        external_mapped_bytes=[],
        refresh_torch_memory_limit=lambda: None,
        buffer=SimpleNamespace(device=torch.device("cpu")),
        reserved_headroom_bytes=g // 2,
        torch_memory_budget=None,
        physical_bytes=4 * g,
    )
    worker = SimpleNamespace(
        rank=0,
        model_runner=SimpleNamespace(eplb_state=None, extensible_kv_cache=kv),
        cache_config=SimpleNamespace(),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_grow_torch_floor_mib=None)
        ),
    )
    monkeypatch.setattr(wrc, "_ep_rank", lambda w: 0)
    monkeypatch.setattr(ft, "_free_device_bytes", lambda device: 1 << 40)
    monkeypatch.setattr(ft, "bytes_to", lambda rows: g)
    # 1 GiB of torch cache unused: not room for the rows
    memory = {"free": g // 4, "reserved": g}
    monkeypatch.setattr(
        ft, "device_memory", lambda device: (memory["free"], 0, 0, memory["reserved"])
    )
    emptied = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: emptied.append(1))
    refusal = wrc.handle(worker, "check", "tiers.fast", {"rows": 5})["refusal"]
    assert "0.25 GiB free" in refusal and emptied  # the cache was given back first
    # a precheck leaves the memory out (asked before the KV cache gives)
    pre = wrc.handle(worker, "check", "tiers.fast", {"rows": 5, "precheck": True})
    assert pre["refusal"] is None
    # what the KV cache released for this resize is there: no floor on it
    memory["free"] = g + g // 8
    ok = wrc.handle(worker, "check", "tiers.fast", {"rows": 5, "kv_released": g})
    assert ok["refusal"] is None
    # the same bytes without the KV cache must also leave torch its floor
    memory["reserved"] = 0
    refusal = wrc.handle(worker, "check", "tiers.fast", {"rows": 5})["refusal"]
    assert refusal is not None and "floor" in refusal
    assert all(int((t.placement.tier == FAST).sum()) == 3 for t in env.tiers)


def test_a_kv_fit_after_a_grow_leaves_the_card_rows_their_memory(monkeypatch):
    """The KV grow floor after a card tier grow of N bytes taken from the KV
    cache allows the same spare as before (not the rows' N), and after the
    shrink back the KV cache may take the N again: charged once each way."""
    from vllm.platforms import current_platform
    from vllm.v1.worker import extensible_kv_cache as ekv

    g, blk = 1 << 30, 1 << 20
    held = {"bytes": 0}
    kv = SimpleNamespace(
        physical_bytes=6 * g,
        external_mapped_bytes=[lambda: held["bytes"]],
        torch_memory_budget=10 * g,
        num_committed_blocks=6 * g // blk,
    )
    kv.physical_bytes_for = lambda n: n * blk
    kv.blocks_within = lambda b: b // blk
    monkeypatch.setattr(ekv, "_torch_memory_now", lambda device: (3 * g, 3 * g, 3 * g))
    free = {"bytes": 4 * g}
    monkeypatch.setattr(
        current_platform, "free_device_memory", lambda device=None: free["bytes"]
    )

    def fit():
        return ekv._grow_floor_at(kv, 1 << 30, g // 2, "test", torch.device("cpu"))

    before = fit().max_blocks * blk - kv.physical_bytes
    # the tier grows by 1 GiB that the KV cache gave: KV smaller, rows mapped
    kv.physical_bytes -= g
    kv.num_committed_blocks = kv.physical_bytes // blk
    held["bytes"] = g
    after_grow = fit().max_blocks * blk - kv.physical_bytes
    assert after_grow == before == g // 2  # the same spare, not the rows'
    # the tier shrinks back: the KV cache may take the 1 GiB again
    held["bytes"] = 0
    free["bytes"] += g
    assert fit().max_blocks * blk - kv.physical_bytes == g + g // 2


def test_a_kv_restore_passes_the_measurement_not_the_floor(grow_ranks):  # noqa: F811
    """A restore (a resize's KV memory given back) goes past what the
    request's measurement allows, up to the device and torch floor, never
    past them."""
    from vllm.v1.worker import extensible_kv_cache as ekv

    ranks = grow_ranks([40])
    cache = ranks.caches[0]
    # the request's budget fits 30 blocks; the device fits 40
    short = cache.physical_bytes_for(40) - cache.physical_bytes_for(30)

    def grow(target, restore):
        return ekv.grow_kv_cache(
            ranks.runners[0],
            target,
            init_free_memory=1024 << 20,
            requested_memory=(1024 << 20) - short,
            reduce_min=lambda v: v,
            restore=restore,
        )

    assert grow(20, True) == 20  # below what fits: as an ordinary grow
    assert grow(60, False) == 30  # the measurement binds an ordinary grow
    assert grow(60, True) == 40  # a restore goes to the floor, not past it
    assert cache.last_grow["vote_blocks"] == 40


def test_a_grow_shrink_cycle_leaves_torch_its_floor(grow_ranks, monkeypatch):  # noqa: F811
    """The card's cycle (16:26Z): KV at its boot size, +8 rows take 251
    blocks, −8 give them back while serving has left torch more reserved
    than at boot. The KV cache must not go back past the torch floor: after
    the return, torch's cap still covers its reservation plus the serving
    peak's headroom (else the next 2K x 4 step runs the card out)."""
    from vllm.v1.worker import extensible_kv_cache as ekv

    # the card's numbers in blocks of 3.37 MiB, scaled by 1/100: after +8 the
    # KV cache holds 20 (2,010), the restore asks for 23 (2,260); torch
    # allocated 22 (7.27 GiB), reserved 24 (7.77), keep 3 (1.13); the floor
    # allows 21 (2,113), the measurement 23 (2,259)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda device=None: (32 << 30, 32 << 30)
    )
    monkeypatch.setattr(
        torch.xpu, "set_per_process_memory_fraction", lambda fraction, device=None: None
    )
    ranks = grow_ranks([60], committed=20)
    cache = ranks.caches[0]
    assert cache.physical_bytes_for(23) > cache.physical_bytes_for(21)
    blk = 8 << 20  # torch's numbers in units of 8 MiB
    allocated, keep, reserved = 22 * blk, 3 * blk, 24 * blk
    ranks.torch_stats = {
        "allocated_bytes.all.current": allocated,
        "allocated_bytes.all.peak": allocated + 3 * blk,
    }
    ranks.torch_reserved = reserved
    cache.reserved_headroom_bytes = keep
    cache.torch_memory_budget = cache.physical_bytes_for(21) + reserved + keep
    n = ekv.grow_kv_cache(
        ranks.runners[0],
        23,
        init_free_memory=1024 << 20,
        requested_memory=1024 << 20,
        reduce_min=lambda v: v,
        restore=True,
    )
    assert 20 < n < 23  # fewer blocks, the floor kept (295a2cdff6 gave 23)
    cap_after = cache.torch_memory_budget - cache.physical_bytes
    assert cap_after - reserved >= keep  # the next peak has its headroom


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
    # EPLB is the one writer of the placement: the bound is a capacity, the
    # slots move in its tier pass, and the rows go once the slots have
    for t in env.tiers:
        assert t.tier_pass_wanted
    _tier_pass(env)
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
    assert p.row[4] < 2  # laid out from row 0 (slot order)
    # EPLB's weights keep their shapes and no longer hold the old rows
    assert weights[0].parts[1].device.type == "meta"
    assert weights[0].shape[0] == E
    # raising past the rows sized at load lays out more rows
    _settle(tiers.cpu.cache)
    xt.apply_host_tier_bound(5)
    for _ in range(4):  # each pass reads within the spare rows
        _tier_pass(env)
    for t in env.tiers:
        assert (
            t.placement.num_pinned == 5 and int((t.placement.tier == HOST).sum()) == 0
        )
        assert t.num_pool_rows == t.pool_rows_for(5)
        _assert_serves_the_store(env, t)
    xt.apply_host_tier_bound(2)
    _tier_pass(env)
    assert all(t.placement.num_pinned == 2 for t in env.tiers)
    assert all(t.num_pool_rows == t.pool_rows_for(2) for t in env.tiers)


def _tier_pass(env):
    """One EPLB tier pass over every layer (old map = new map; the tiers cut
    from the load), each layer prepared and committed as EPLB's cycle does."""
    for t in env.tiers:
        phys = np.arange(t.num_global)
        t.commit(t.prepare(phys, phys, _load(t, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1))


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


# -- who chooses and who writes ---------------------------------------------------


def test_the_placement_chooses_the_slots_by_eplbs_load():
    """The card change's tier choice is the placement's (EPLB's tier-choice
    module), by the heat order it is given, bounded by the reads allowed."""
    from vllm.model_executor.offloader.expert_tier_map import TierPlacement

    # 8 slots: card 0 1 2, pinned 3 4, CPU 5 6 7; slot 7 hottest
    p = TierPlacement(8, 3, 2, lambda e: True, num_pool_slots=2)
    rank = np.arange(8)
    p.grow_fast_rows(5)
    new_fast, new_pool = p.plan_card_round(rank, 5, budget=1)
    # the two hottest are CPU slots (a read each): one fits the budget
    assert np.flatnonzero(new_fast).tolist() == [0, 1, 2, 7]
    assert np.array_equal(new_pool, p.is_pool())
    assert p.plan_card_round(rank, 5, budget=0) is None
    # shrink toward 2 card slots from 0 1 2: the coldest (0) leaves; colder
    # than every pinned slot, it goes to the CPU with nothing read
    p2 = TierPlacement(8, 3, 2, lambda e: True, num_pool_slots=2)
    new_fast, new_pool = p2.plan_card_round(rank, 2, budget=0)
    assert np.flatnonzero(new_fast).tolist() == [1, 2]
    assert np.array_equal(new_pool, p2.is_pool())
    # the pinned rows refill with the hottest CPU slots, bounded by the reads
    p3 = TierPlacement(8, 3, 2, lambda e: True, num_pool_slots=2)
    p3.tier[3] = HOST  # one pinned row left empty
    p3.row[3] = -1
    want = p3.plan_pinned_refill(rank, budget=1)
    assert np.flatnonzero(want).tolist() == [4, 7]
    assert p3.plan_pinned_refill(rank, budget=0) is None


def test_every_move_of_a_resize_is_a_commit_on_eplbs_path(monkeypatch, tmp_path):
    """A resize writes maps and rows through ExpertTiers.commit_move (the
    commit EPLB's rearrangements use) and, for the slots moved down before
    rows are dropped, TierPlacement.compact_fast; nothing else writes."""
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    tiers.fill_to(np.arange(E), _load(tiers, [5, 6, 7, 4, 3, 2, 1, 0]), 1.1)
    commits, compacts = [], []
    real_commit = xt.ExpertTiers.commit_move
    real_compact = type(tiers.placement).compact_fast

    def commit(self, move):
        commits.append(move.plan.to_fast)
        return real_commit(self, move)

    def compact(self, *a, **kw):
        compacts.append(a[0])
        return real_compact(self, *a, **kw)

    monkeypatch.setattr(xt.ExpertTiers, "commit_move", commit)
    monkeypatch.setattr(type(tiers.placement), "compact_fast", compact)
    writes = []
    real_write = tiers.ops.compute_write

    def write(dst, values):
        writes.append((len(commits), len(compacts)))
        return real_write(dst, values)

    monkeypatch.setattr(tiers.ops, "compute_write", write)
    tiers.resize_fast(5)
    tiers.resize_fast(LOAD_FAST)
    assert commits and compacts == [LOAD_FAST]
    # each map write happened inside a commit or a compaction
    assert all(c > 0 for c, _ in writes)
    _assert_serves_the_store(env, tiers)


def test_the_entry_is_eplbs_and_waits_for_its_rearrangement(monkeypatch, tmp_path):
    from vllm.distributed.eplb import runtime_control as erc

    env = _env(monkeypatch, tmp_path)
    busy = SimpleNamespace(
        model_states={
            "m": SimpleNamespace(
                rebalanced=True,
                pending_result=None,
                model_name="m",
                model=SimpleNamespace(num_moe_layers=2),
            )
        }
    )
    with pytest.raises(RuntimeError, match="in flight"):
        erc.resize_card_tier(busy, 5)
    assert ft.rows_now() == LOAD_FAST
    idle = SimpleNamespace(model_states={})
    monkeypatch.setattr(ft, "_free_device_bytes", lambda device: 1 << 40)
    assert erc.resize_card_tier(idle, 5)["rows"] == [LOAD_FAST, 5]
    assert all(t.placement.num_fast == 5 for t in env.tiers)

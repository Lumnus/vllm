# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert tiers' round-4 parts together, on the CPU: EPLB as the one
writer of the placement while the pinned bound, its pinned rows and the card
tier change size; the promotion between rearrangements staying removed; the
debug lines of the tiers."""

import numpy as np
import pytest

from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.offloader.expert_tier_map import (
    FAST,
    HOST,
    POOL,
    TierPlacement,
)
from vllm.v1 import runtime_debug as rd

from .test_xpu_cpu_experts import _settle
from .test_xpu_fast_tier_resize import (  # noqa: F401
    _assert_serves_the_store,
    _env,
    _fresh,
    _load,
    fake_xpu,
    op,
)

pytestmark = pytest.mark.skip_global_cleanup

ORDER = [5, 6, 7, 4, 3, 2, 1, 0]  # hottest first


@pytest.fixture(autouse=True)
def _debug_off():
    yield
    for part in list(rd._ACTIVE):
        rd.disarm(part, "test end")
    rd._LAST.clear()


def _tier_pass(env, order=ORDER):
    """One EPLB tier pass over every layer: the placement of the experts
    stays (old map = new map), the tiers are cut from the load, each layer's
    move prepared and committed as EPLB's async cycle does."""
    for t in env.tiers:
        phys = np.arange(t.num_global)
        t.commit(t.prepare(phys, phys, _load(t, order), 1.1))


def _pinned(t):
    return int((t.placement.tier == POOL).sum())


def test_lowering_the_bound_moves_nothing_until_eplbs_pass(monkeypatch, tmp_path, op):  # noqa: F811
    env = _env(
        monkeypatch, tmp_path, pinned=2, pinned_release=True, host_promote_rows=0
    )
    for t in env.tiers:
        t.fill_to(np.arange(t.num_global), _load(t, ORDER), 1.1)
    rows_at_load = env.tiers[0].num_pool_rows
    before = [t.placement.tier.copy() for t in env.tiers]
    xt.apply_host_tier_bound(1)
    for t, tier in zip(env.tiers, before):
        # the capacity only: no slot moved, no row laid out yet
        assert np.array_equal(t.placement.tier, tier)
        assert t.placement.num_pinned == 1 and _pinned(t) == 2
        assert t.num_pool_rows == rows_at_load
        assert t.tier_pass_wanted and t._release_rows_to == t.pool_rows_for(1)
    _tier_pass(env)
    for t in env.tiers:
        # EPLB moved the coldest pinned slot to the CPU; then the rows went
        assert _pinned(t) == 1 and not t.tier_pass_wanted
        assert t._release_rows_to is None
        assert t.num_pool_rows == t.pool_rows_for(1) == rows_at_load - 1
        assert all(x.shape[0] == t.num_pool_rows for x in t.slow.values())
        assert t._impls[1].num_experts == t.num_pool_rows
        _assert_serves_the_store(env, t)
    assert np.flatnonzero(env.tiers[0].placement.tier == POOL).tolist() == [4]


def test_rows_beyond_the_slots_in_use_go_back_at_once(monkeypatch, tmp_path, op):  # noqa: F811
    """The promotion rows above the bound serve no slot: lowering the bound
    lays the rows out for it at once (a row relayout, no slot changes its
    tier); the slot over the bound still waits for EPLB's pass."""
    env = _env(monkeypatch, tmp_path, pinned=2, pinned_release=True)
    for t in env.tiers:
        t.fill_to(np.arange(t.num_global), _load(t, ORDER), 1.1)
    before = [t.placement.tier.copy() for t in env.tiers]
    xt.apply_host_tier_bound(1)
    for t, tier in zip(env.tiers, before):
        assert np.array_equal(t.placement.tier, tier) and _pinned(t) == 2
        assert t.num_pool_rows == t.pool_rows_for(1) and t._release_rows_to is None
        _assert_serves_the_store(env, t)
    _tier_pass(env)
    assert all(_pinned(t) == 1 for t in env.tiers)


def test_raising_lays_rows_out_first_and_eplbs_passes_fill_them(
    monkeypatch,
    tmp_path,
    op,  # noqa: F811
):
    env = _env(monkeypatch, tmp_path, pinned=2, pinned_release=True)
    for t in env.tiers:
        t.fill_to(np.arange(t.num_global), _load(t, ORDER), 1.1)
    _settle(env.tiers[0].cpu.cache)
    xt.apply_host_tier_bound(5)
    for t in env.tiers:
        assert t.num_pool_rows == t.pool_rows_for(5)  # more rows at once
        assert _pinned(t) == 2  # filled by EPLB, not here
    passes = 0
    while any(int((t.placement.tier == HOST).sum()) for t in env.tiers):
        _tier_pass(env)
        passes += 1
        assert passes <= 4
    # each pass reads within the spare rows: more than one pass
    assert passes >= 2
    for t in env.tiers:
        assert _pinned(t) == 5
        _assert_serves_the_store(env, t)


def test_a_lower_then_a_higher_bound_before_the_pass(monkeypatch, tmp_path, op):  # noqa: F811
    env = _env(
        monkeypatch, tmp_path, pinned=2, pinned_release=True, host_promote_rows=0
    )
    for t in env.tiers:
        t.fill_to(np.arange(t.num_global), _load(t, ORDER), 1.1)
    rows_at_load = env.tiers[0].num_pool_rows
    xt.apply_host_tier_bound(1)
    xt.apply_host_tier_bound(2)  # nothing moved yet: nothing to lay out
    for t in env.tiers:
        assert t.num_pool_rows == rows_at_load and t._release_rows_to is None
        assert t.placement.num_pinned == 2
    _tier_pass(env)
    for t in env.tiers:
        assert _pinned(t) == 2 and t.num_pool_rows == rows_at_load
        _assert_serves_the_store(env, t)


def test_the_card_resize_waits_for_eplbs_pass_on_the_layer(monkeypatch, tmp_path, op):  # noqa: F811
    """A card resize moves no expert while EPLB has a move of the layer
    prepared, and ranks slots by EPLB's load alone."""
    env = _env(monkeypatch, tmp_path, pinned=2)
    t = env.tiers[0]
    t.fill_to(np.arange(t.num_global), _load(t, ORDER), 1.1)
    phys = np.arange(t.num_global)
    move = t.prepare(phys, phys, _load(t, ORDER), 1.1)
    with pytest.raises(RuntimeError, match="EPLB move is prepared"):
        t.resize_fast(5)
    t.commit(move)
    load = t._slot_load.copy()
    rank = t._heat_rank()
    assert np.array_equal(np.argsort(rank), np.argsort(load, kind="stable"))
    t.resize_fast(5)
    assert sorted(np.flatnonzero(t.placement.tier == FAST).tolist()) == [3, 4, 5, 6, 7]
    _assert_serves_the_store(env, t)


def test_the_removed_promotion_stays_removed(monkeypatch, tmp_path, op):  # noqa: F811
    """No path but EPLB's moves a slot between tiers: the promotion between
    rearrangements and its counter are gone from every merged part."""
    for name in ("between_steps", "_finish_promotion", "_drop_promotion"):
        assert not hasattr(xt.ExpertTiers, name), name
    assert not hasattr(TierPlacement, "promote_host")
    env = _env(monkeypatch, tmp_path, pinned=2)
    for name in ("_promotion", "_host_count"):
        assert not hasattr(env.tiers[0], name), name


def test_the_tiers_debug_lines(monkeypatch, tmp_path, op):  # noqa: F811
    env = _env(
        monkeypatch, tmp_path, pinned=2, pinned_release=True, host_promote_rows=0
    )
    for t in env.tiers:
        t.fill_to(np.arange(t.num_global), _load(t, ORDER), 1.1)
    rd.arm("tiers", 60.0, 1000)
    xt.apply_host_tier_bound(1)
    after_bound = rd.state()["on"]["tiers"]["lines"]
    assert after_bound == len(env.tiers)  # one capacity line per layer
    _tier_pass(env)
    # per layer: its commit and its rows laid out
    assert rd.state()["on"]["tiers"]["lines"] == after_bound + 2 * len(env.tiers)
    env.tiers[0].resize_fast(4)
    assert rd.state()["on"]["tiers"]["lines"] > after_bound + 2 * len(env.tiers) + 1
    rd.disarm("tiers", "asked")
    lines = rd.state()["on"]
    assert "tiers" not in lines

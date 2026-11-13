# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB's runtime controls over layers with experts computed on the CPU
(--cpu-offload-pinned-gb): pause, a rearrangement on demand, injected
allocation failures, the pinned bound, EPLB's tier passes and its seed. EPLB
is the one writer of the placement: every tier change below happens in its
rearrangements and tier passes.

A one-rank ``EplbState`` over real ``ExpertTiers`` with host slots and a real
store (see ``xpu_store_harness``); the host op is the torch reference."""

import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.distributed.eplb.async_worker as async_worker_mod
import vllm.distributed.eplb.eplb_state as eplb_state_mod
import vllm.distributed.eplb.runtime_control as control
import vllm.v1.worker.runtime_control as wrc
from vllm.distributed.eplb import fault_injection
from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import HOST, POOL

from ..kernels.moe.test_xpu_cpu_experts import CPU_CFG, _reference_op
from ..kernels.moe.xpu_store_harness import HIDDEN, LAYERS, E, run
from .eplb_single_rank import FakeMoEModel, add_model, single_rank_eplb
from .test_eplb_store_tiers import _no_stream, _Recorder, _Stream

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    fault_injection.clear()
    monkeypatch.setattr(xt, "_RUNTIME_SETTINGS_REGISTERED", False)
    monkeypatch.setattr(wrc, "_TIER_SETTINGS", {})
    monkeypatch.setattr(xt, "_LIVE_TIERS", weakref.WeakSet())
    if not cx.host_op_available():
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    cx.reset_runtime()
    yield
    fault_injection.clear()
    cx.reset_runtime()
    tstore.reset_sessions()


def _build(monkeypatch, tmp_path, eplb_extra=None, **cpu_extra):
    """Store built at a first start; the second start has 2 pinned slots
    and 3 host slots per layer (of 5 off the card), 2 spare rows."""
    run(monkeypatch, tmp_path, store=True, spare=2)
    xt._LIVE_TIERS.clear()  # the start that built the store is not this one
    env = run(
        monkeypatch,
        tmp_path,
        store=True,
        spare=2,
        pinned=2,
        store_extra=dict(CPU_CFG, **cpu_extra),
    )
    model = FakeMoEModel(
        LAYERS, E, weights=[tiers.eplb_weights() for tiers in env.tiers]
    )
    eplb = {"step_interval": 10, "window_size": 4, **(eplb_extra or {})}
    state = single_rank_eplb(
        monkeypatch,
        use_async=True,
        policy="incremental",
        max_moves_per_rank=2,
        **eplb,
    )
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    monkeypatch.setattr(
        eplb_state_mod,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=group, cpu_all_reduce_=lambda t: t),
    )
    monkeypatch.setattr(
        control, "get_ep_group", lambda: SimpleNamespace(device_group=group)
    )
    monkeypatch.setattr(
        async_worker_mod,
        "get_eplb_group",
        lambda: SimpleNamespace(device_group=group, cpu_group=group),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(async_worker_mod, "device_stream", _no_stream)
    monkeypatch.setattr(eplb_state_mod, "logger", _Recorder())

    def forbidden(*args, **kwargs):
        raise AssertionError("a store-backed layer must not use EPLB's transfers")

    monkeypatch.setattr(async_worker_mod, "transfer_layer", forbidden)
    monkeypatch.setattr(eplb_state_mod, "move_from_buffer", forbidden)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    model_state.communicator.set_stream = lambda s: None
    return env, state, model_state


def _cycle(state, model_state, start) -> None:
    """``start()`` (a step or a rearrangement) and, if it began an async
    rearrangement, its worker cycle on a thread with the commits here."""
    errors: list[BaseException] = []
    start()
    if not model_state.rebalanced:
        return

    def worker():
        try:
            state.rearrange_event.wait(stream=None)
            async_worker_mod._transfer_cycle(state, _Stream(), False)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            state.async_worker_error = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    for _ in range(100000):
        if not model_state.rebalanced:
            break
        if model_state.pending_result is not None:
            state._consume_async_result(model_state, 0)
        else:
            thread.join(0.001)
            if errors:
                raise errors[0]
    thread.join(10)
    assert not errors and not model_state.rebalanced


def _serve_host(env, tiers_index=0, times=3):
    """Steps that route to the host slots of a layer (the CPU computes them)."""
    tiers = env.tiers[tiers_index]
    host = np.flatnonzero(tiers.placement.tier == HOST).tolist()
    ids = torch.tensor([[host[0], host[1]], [host[0], host[2]]], dtype=torch.int32)
    x = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    for _ in range(times):
        assert tiers.cpu.compute(tiers, ids, torch.ones(2, 2), x, torch.empty_like(x))
    return host[0]


def _hot(model_state):
    """Logical 6 and 7 hot in layer 0, the rest cold, in every record of the
    window (a step overwrites one record with its own load)."""
    loads = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 60, 70], [9, 8, 7, 1, 1, 1, 1, 1]], dtype=torch.int32
    )
    window = model_state.expert_load_window
    per_slot = torch.gather(loads, 1, model_state.physical_to_logical_map.long())
    window.copy_(per_slot.to(window.dtype).expand_as(window))


def test_paused_eplb_keeps_the_cpu_tier_serving(monkeypatch, tmp_path):
    """Paused, the intervals end without a rearrangement or a tier pass and
    the placement stays; the CPU tier still serves its slots."""
    env, state, model_state = _build(
        monkeypatch, tmp_path, eplb_extra={"tier_interval": 3}
    )
    tiers = env.tiers[0]
    p = tiers.placement
    tier = p.tier.copy()
    p2l = model_state.physical_to_logical_map.clone()
    state.rearrangements_paused = True
    _serve_host(env)
    for _ in range(25):
        _hot(model_state)
        state.step()
    assert state.paused_intervals >= 2 and state.rearrangements == 0
    assert state.tier_passes == 0
    assert torch.equal(model_state.physical_to_logical_map, p2l)
    assert np.array_equal(p.tier, tier)
    assert tiers.cpu.stats.per_tier["cpu"] > 0
    tstore.reset_sessions()


def test_a_rearrangement_on_demand_cuts_the_tiers_from_the_window(
    monkeypatch, tmp_path
):
    """eplb.rearrange at a step: the cycle that follows cuts the tiers from
    the window (the hot experts leave the CPU), within the pinned bound."""
    env, state, model_state = _build(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    p = tiers.placement
    state.expert_rearrangement_step = 0
    assert control.check_trigger(state) is None
    control.trigger(state)
    _hot(model_state)
    _cycle(state, model_state, state.step)
    assert state.rearrangements == 1
    assert int((p.tier == POOL).sum()) == 2
    assert set(tiers.slot_logical[p.tier != HOST].tolist()) >= {6, 7}
    for t in env.tiers:
        t.placement.check()
        fm, sm = t.placement.maps(t.slot_global, t.num_global)
        assert t.fast_map.tolist() == fm.tolist()
        assert t.slow_map.tolist() == sm.tolist()
        assert not t._eplb_pending
    summary = tiers.scrub(model_state.physical_to_logical_map[0].numpy())
    assert summary["mismatches"] == 0 and summary["placement_equal"]
    tstore.reset_sessions()


@pytest.mark.parametrize("stage", ["load", "plan"])
def test_an_injected_failure_leaves_the_cpu_tier_as_it_was(
    monkeypatch, tmp_path, stage
):
    """An allocation failure injected at the rearrangement (fault.arm): the
    rearrangement is skipped before any layer is prepared, so no layer is
    left waiting for a move (bound changes go on), the placement and the
    maps are unchanged, the CPU still computes its slots, and the next
    rearrangement runs."""
    env, state, model_state = _build(monkeypatch, tmp_path)
    before = [(t.placement.tier.copy(), t.slow_map.clone()) for t in env.tiers]
    fault_injection.arm(stage, 1)
    _hot(model_state)
    _cycle(state, model_state, state.rearrange)
    assert state.skipped_rearrangements == 1 and not model_state.rebalanced
    for t, (tier, slow_map) in zip(env.tiers, before):
        assert np.array_equal(t.placement.tier, tier)
        assert torch.equal(t.slow_map, slow_map)
        assert not t._eplb_pending
    assert xt.check_host_tier_bound(1) is None
    _serve_host(env)
    _hot(model_state)
    _cycle(state, model_state, state.rearrange)  # disarmed: runs
    assert state.skipped_rearrangements == 1 and state.rearrangements == 1
    tiers = env.tiers[0]
    assert int((tiers.placement.tier == POOL).sum()) == 2
    assert set(tiers.slot_logical[tiers.placement.tier != HOST].tolist()) >= {6, 7}
    tstore.reset_sessions()


def test_the_bound_moves_the_slots_in_eplbs_next_tier_pass(monkeypatch, tmp_path):
    """tiers.set host_tier_bound changes the capacity only; EPLB's next step
    runs a tier pass that moves the slots to it (EPLB is the one writer).
    Refused while a rearrangement is in flight."""
    env, state, model_state = _build(monkeypatch, tmp_path)
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(eplb_state=state))
    monkeypatch.setattr(wrc, "_ep_rank", lambda w: 0)
    _hot(model_state)
    state.rearrange()
    assert model_state.rebalanced
    refusal = wrc.handle(worker, "check", "tiers.set", {"host_tier_bound": 1})
    assert "in flight" in refusal["refusal"]
    _cycle(state, model_state, lambda: None)
    pinned = [int((t.placement.tier == POOL).sum()) for t in env.tiers]
    reply = wrc.handle(worker, "apply", "tiers.set", {"host_tier_bound": 1})
    assert reply["ok"], reply
    # nothing moved yet: the layers ask EPLB for a pass
    assert [int((t.placement.tier == POOL).sum()) for t in env.tiers] == pinned
    assert all(t.tier_pass_wanted for t in env.tiers)
    p2l = model_state.physical_to_logical_map.clone()
    _hot(model_state)
    _cycle(state, model_state, state.step)
    assert state.tier_passes == 1 and state.rearrangements == 1
    assert all(int((t.placement.tier == POOL).sum()) == 1 for t in env.tiers)
    assert not any(t.tier_pass_wanted for t in env.tiers)
    assert torch.equal(model_state.physical_to_logical_map, p2l)
    for t in env.tiers:
        t.placement.check()
    # the next full rearrangement keeps the bound
    _hot(model_state)
    _cycle(state, model_state, state.rearrange)
    assert all(int((t.placement.tier == POOL).sum()) == 1 for t in env.tiers)
    tstore.reset_sessions()


def test_tier_passes_every_n_steps_sort_the_tiers_within_their_budget(
    monkeypatch, tmp_path
):
    """tier_interval 2, tier_max_moves 1, no full rearrangement: passes every
    second step move one expert per layer each, from the window, until the
    hot experts are off the CPU; no expert changes rank or slot."""
    env, state, model_state = _build(
        monkeypatch,
        tmp_path,
        eplb_extra={"tier_interval": 2, "tier_max_moves": 1, "step_interval": 10**6},
    )
    tiers = env.tiers[0]
    p = tiers.placement
    p2l = model_state.physical_to_logical_map.clone()
    assert set(tiers.slot_logical[p.tier == HOST].tolist()) >= {6, 7}
    for _ in range(40):
        _hot(model_state)
        _cycle(state, model_state, state.step)
        if set(tiers.slot_logical[p.tier != HOST].tolist()) >= {6, 7}:
            break
    assert set(tiers.slot_logical[p.tier != HOST].tolist()) >= {6, 7}
    assert state.tier_passes >= 2 and state.rearrangements == 0
    assert all(m.get("units", 0) <= 1 for t in env.tiers for m in t.move_log)
    assert torch.equal(model_state.physical_to_logical_map, p2l)
    summary = tiers.scrub(p2l[0].numpy(), full=True)
    assert summary["mismatches"] == 0 and summary["placement_equal"]
    # the classes by the same window: logical 6 and 7 hot in layer 0
    r = tiers.class_readout(1.5, 0.5, 1.1)
    assert r["hot"] == 2 and r["warm_or_hot_on_cpu"] == 0
    s = control.settings(state)
    assert s["tier_interval"] == 2 and s["tier_max_moves"] == 1
    assert control.check_settings(state, {"tier_warm": 3.0}) is not None
    assert control.check_settings(state, {"tier_interval": -1}) is not None
    control.apply_settings(state, {"tier_interval": 0})
    passes = state.tier_passes
    for _ in range(6):
        _hot(model_state)
        _cycle(state, model_state, state.step)
    assert state.tier_passes == passes
    tstore.reset_sessions()


def test_a_seed_places_experts_and_tiers_when_eplb_has_no_loads(monkeypatch, tmp_path):
    """No state_path: EPLB's seed_path (here a rankings file of the 0.32
    line: layer number -> experts hottest first) gives the loads the restore
    fills the slots by: the hottest on the card, the next pinned, the rest
    on the CPU."""
    import json

    seed = tmp_path / "rankings.json"
    seed.write_text(json.dumps({"0": [7, 6, 5, 4, 3, 2, 1, 0], "1": [0, 1, 2, 3, 4]}))
    env, state, model_state = _build(
        monkeypatch, tmp_path, eplb_extra={"seed_path": str(seed)}
    )
    assert state.has_tier_seed() and model_state.has_saved_load
    monkeypatch.setattr(eplb_state_mod, "all_reduce", lambda *a, **k: None)
    state.restore_saved_placement()
    for tiers, card, pinned in ((env.tiers[0], {5, 6, 7}, {3, 4}),
                                (env.tiers[1], {0, 1, 2}, {3, 4})):  # fmt: skip
        p = tiers.placement
        p.check()
        assert set(tiers.slot_logical[p.tier == 0].tolist()) == card
        assert set(tiers.slot_logical[p.tier == POOL].tolist()) == pinned
        assert p.tier[int(np.argmax(tiers.temperature()))] == 0
        summary = tiers.scrub(None, full=True)
        assert summary["mismatches"] == 0 and summary["maps_equal"]
    tstore.reset_sessions()


def test_without_a_seed_nothing_is_read(monkeypatch, tmp_path):
    env, state, model_state = _build(monkeypatch, tmp_path)
    assert not state.has_tier_seed() and not model_state.has_saved_load
    tstore.reset_sessions()

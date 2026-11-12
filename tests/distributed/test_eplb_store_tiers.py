# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB over layers whose expert slots are placed through the expert store.

The layers are real ``ExpertTiers`` over a real store built from a synthetic
int4 checkpoint (the kernels and the accelerator are faked, see
``xpu_store_harness``). A one-rank ``EplbState`` runs its async cycle: the
worker prepares each layer by reading the store, the main thread commits the
tier maps next to EPLB's own. No expert buffer exists and the communicator is
never asked to move a row.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.distributed.eplb.async_worker as async_worker_mod
import vllm.distributed.eplb.eplb_state as eplb_state_mod
from vllm.distributed.eplb.expert_weights import StoreBackedWeight, expert_mover
from vllm.distributed.eplb.policy import EPLB_POLICIES
from vllm.model_executor.offloader import expert_tier_store as tstore

from ..kernels.moe.xpu_store_harness import (
    FAST,
    LAYERS,
    E,
    Env,
    checkpoint,
    expert_in_store,
    run,
)
from .eplb_single_rank import FakeMoEModel, add_model, single_rank_eplb

pytestmark = pytest.mark.skip_global_cleanup


class _Recorder:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg, *args, **kwargs) -> None:
        self.lines.append(msg % args if args else msg)

    debug = warning = error = exception = info


def _build(monkeypatch, tmp_path, spare=3, **eplb_kwargs):
    env = run(monkeypatch, tmp_path, store=True, spare=spare)
    model = FakeMoEModel(
        LAYERS, E, weights=[tiers.eplb_weights() for tiers in env.tiers]
    )
    eplb_kwargs.setdefault("use_async", True)
    eplb_kwargs.setdefault("policy", "incremental")
    eplb_kwargs.setdefault("max_moves_per_rank", min(2, spare))
    state = single_rank_eplb(monkeypatch, **eplb_kwargs)
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    monkeypatch.setattr(
        eplb_state_mod,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=group, cpu_all_reduce_=lambda t: t),
    )
    monkeypatch.setattr(
        async_worker_mod,
        "get_eplb_group",
        lambda: SimpleNamespace(device_group=group, cpu_group=group),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(async_worker_mod, "device_stream", _no_stream)
    log = _Recorder()
    monkeypatch.setattr(eplb_state_mod, "logger", log)

    def forbidden(*args, **kwargs):
        raise AssertionError("a store-backed layer must not use EPLB's transfers")

    monkeypatch.setattr(async_worker_mod, "transfer_layer", forbidden)
    monkeypatch.setattr(eplb_state_mod, "move_from_buffer", forbidden)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    model_state.communicator.set_stream = lambda s: None
    return env, model, state, model_state, log


class _no_stream:  # noqa: N801 - a context manager standing in for device_stream
    def __init__(self, stream=None):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class _Stream:
    def synchronize(self) -> None:
        pass


def _window(model_state, logical_loads):
    phy2log = model_state.physical_to_logical_map
    loads = torch.tensor(logical_loads, dtype=torch.int32)
    model_state.expert_load_window.zero_()
    model_state.expert_load_window[0] = torch.gather(loads, 1, phy2log.long())


def _run_async_cycle(state, model_state) -> None:
    """The async worker's cycle on a thread, the main thread's per-step part
    on this one: commit each layer the worker prepared."""
    errors: list[BaseException] = []

    def worker():
        try:
            state.rearrange_event.wait(stream=None)  # as the worker's loop does
            async_worker_mod._transfer_cycle(state, _Stream(), False)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            state.async_worker_error = exc

    state.rearrange()
    assert model_state.rebalanced
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    for _ in range(100000):
        if not model_state.rebalanced:
            break
        if model_state.pending_result is not None:
            # what the step hook does: commit the layer, or leave it
            state._consume_async_result(model_state, 0)
        else:
            thread.join(0.001)
            if errors:
                raise errors[0]
    thread.join(10)
    assert not errors and not model_state.rebalanced


def _slot_bytes(tiers, slot):
    """What the kernels would read for a slot: its row in its tier."""
    fast, slow = tiers._impls
    r = int(tiers.placement.row[slot])
    if tiers.placement.tier[slot] == 0:
        return {
            "w13": fast.w13[r].view(torch.uint8),
            "w2": fast.w2[r].view(torch.uint8),
            "w13_scales": fast.gemm1_wei_scales[r],
            "w2_scales": fast.gemm2_wei_scales[r],
        }
    return {k: v[r] for k, v in tiers.slow.items()}


def _holds(env, tiers_index, slot, logical) -> bool:
    want = expert_in_store(env, tiers_index, logical)
    got = _slot_bytes(env.tiers[tiers_index], slot)
    return all(
        torch.equal(
            got[k].reshape(-1).view(torch.uint8), want[k].reshape(-1).view(torch.uint8)
        )
        for k in want
    )


def test_store_backed_layers_have_no_expert_buffer_and_no_tier_order(
    monkeypatch, tmp_path
):
    env, model, state, model_state, log = _build(monkeypatch, tmp_path)
    assert all(isinstance(ws[0], StoreBackedWeight) for ws in model.expert_weights)
    assert all(
        expert_mover(ws) is env.tiers[i] for i, ws in enumerate(model.expert_weights)
    )
    assert model_state.expert_buffer == []
    assert model_state.movers is not None and model_state.fast_slots is None
    assert any("expert store mover" in line for line in log.lines)
    # EPLB sees the host tier by its slots, not by the spare rows
    weight = model.expert_weights[0][0]
    assert weight.tier_sizes == (FAST, E - FAST) and len(weight) == E
    # ... and cannot reach a row by slot: which row serves a slot is the
    # mover's state, so a by-slot access would touch another expert
    with pytest.raises(RuntimeError, match="not addressable by slot"):
        weight[0]
    with pytest.raises(RuntimeError, match="not addressable by slot"):
        weight.rows_written([E - 1])
    with pytest.raises(RuntimeError, match="not addressable by slot"):
        weight.view(torch.uint8).tier_of(0)
    # a synchronous rearrangement is refused, a profile rearrangement is free
    with pytest.raises(NotImplementedError, match="asynchronously"):
        state.rearrange(blocking=True)
    assert state.rearrange(is_profile=True) is None
    tstore.reset_sessions()


def test_async_cycle_reads_the_store_and_puts_hot_experts_on_the_card(
    monkeypatch, tmp_path
):
    env, model, state, model_state, log = _build(monkeypatch, tmp_path, spare=4)
    loads = [[1, 1, 1, 0, 2, 40, 50, 60], [9, 8, 7, 1, 1, 1, 1, 1]]
    _window(model_state, loads)
    _run_async_cycle(state, model_state)
    p0, p1 = env.tiers[0].placement, env.tiers[1].placement
    assert set(np.flatnonzero(p0.tier == 0).tolist()) == {5, 6, 7}
    assert set(np.flatnonzero(p1.tier == 0).tolist()) == {0, 1, 2}  # already hottest
    for i in range(LAYERS):
        env.tiers[i].placement.check()
        for slot in range(E):
            assert _holds(env, i, slot, slot), (i, slot)  # same placement
    # the tier maps the kernels read agree with the host state
    for tiers in env.tiers:
        fm, sm = tiers.placement.maps(tiers.slot_global, tiers.num_global)
        assert tiers.fast_map.tolist() == fm.tolist()
        assert tiers.slow_map.tolist() == sm.tolist()
    # bytes came from the store: counted in the logged line
    line = next(x for x in log.lines if x.startswith("Expert store rearrangement"))
    assert "units read" in line and "prepare" in line and "commit" in line
    assert int(line.split("): ")[1].split(" units")[0]) >= 3
    # the same load again changes nothing and reads nothing
    log.lines.clear()
    before = [t.placement.tier.copy() for t in env.tiers]
    _window(model_state, loads)
    _run_async_cycle(state, model_state)
    for t, b in zip(env.tiers, before):
        assert (t.placement.tier == b).all()
    line = next(x for x in log.lines if x.startswith("Expert store rearrangement"))
    assert ": 0 units read" in line
    tstore.reset_sessions()


def test_two_ranks_balance_by_reading_the_store(monkeypatch, tmp_path):
    """Rank-balance moves, as the incremental policy plans them for two ranks,
    land on the receiving rank from the store; nothing is sent."""
    # One pass over every expert builds the store; each rank then holds its
    # half of the experts and reads the others' from the store.
    run(monkeypatch, tmp_path, store=True, spare=3)
    tstore.reset_sessions()
    ranks = []
    for r in range(2):
        env = Env(
            monkeypatch, tmp_path, True, fast=2, spare=3, rank=r, ep_size=2,
            reset=(r == 0),
        )  # fmt: skip
        env.construct()
        env.load()
        env.process()
        ranks.append(env)
    assert all(t.fills_from_store for env in ranks for t in env.tiers)
    per = E // 2
    policy = EPLB_POLICIES["incremental"].configured(2)
    old = torch.arange(E).unsqueeze(0).expand(LAYERS, -1).contiguous()
    # rank 0's experts are hot: the policy moves some of them to rank 1
    load = torch.tensor([[50, 40, 30, 20, 1, 1, 1, 1]] * LAYERS, dtype=torch.float32)
    new = policy.rebalance_experts(load, E, 1, 1, 2, old)
    assert (new != old).any()
    for layer in range(LAYERS):
        prepared = {}
        for r, env in enumerate(ranks):
            tiers = env.tiers[layer]
            # the placement the tiers hold is EPLB's old one on this rank
            assert (
                tiers.slot_logical.tolist()
                == old[layer, r * per : (r + 1) * per].tolist()
            )
            prepared[r] = tiers.prepare(
                old[layer].numpy(), new[layer].numpy(), load[layer].numpy(), 1.1
            )
        for r, env in enumerate(ranks):
            env.tiers[layer].commit(prepared[r])
    for r, env in enumerate(ranks):
        for layer in range(LAYERS):
            tiers = env.tiers[layer]
            assert (
                tiers.slot_logical.tolist()
                == new[layer, r * per : (r + 1) * per].tolist()
            )
            tiers.placement.check()
            for slot in range(per):
                logical = int(new[layer, r * per + slot])
                want = expert_in_store(env, layer, logical)
                got = _slot_bytes(tiers, slot)
                for k in want:
                    assert torch.equal(
                        got[k].reshape(-1).view(torch.uint8),
                        want[k].reshape(-1).view(torch.uint8),
                    ), (r, layer, slot, k)
    moved = int((new != old).sum())
    assert moved > 0
    tstore.reset_sessions()


def test_restore_fills_the_saved_placement_straight_from_the_store(
    monkeypatch, tmp_path
):
    from vllm.distributed.eplb.eplb_state_file import write_expert_loads

    path = str(tmp_path / "eplb.json")
    saved = {
        f"model.layers.{i}.mlp.experts": loads
        for i, loads in enumerate([[0, 0, 0, 0, 0, 7, 9, 8], [5, 0, 0, 9, 0, 0, 0, 8]])
    }
    write_expert_loads(path, saved)
    env, model, state, model_state, log = _build(
        monkeypatch, tmp_path, spare=3, state_path=path
    )
    assert model_state.has_saved_load
    state.restore_saved_placement()
    assert any("restored the saved placement by filling" in x for x in log.lines)
    assert set(np.flatnonzero(env.tiers[0].placement.tier == 0).tolist()) == {5, 6, 7}
    assert set(np.flatnonzero(env.tiers[1].placement.tier == 0).tolist()) == {0, 3, 7}
    for i in range(LAYERS):
        env.tiers[i].placement.check()
        for slot in range(E):
            assert _holds(
                env, i, slot, int(model_state.physical_to_logical_map[i, slot])
            )
        fm, sm = env.tiers[i].placement.maps(env.tiers[i].slot_global, E)
        assert env.tiers[i].fast_map.tolist() == fm.tolist()
        assert env.tiers[i].slow_map.tolist() == sm.tolist()
        assert (
            env.tiers[i].slot_logical.tolist()
            == model_state.physical_to_logical_map[i].tolist()
        )
    assert checkpoint(0, 0)  # (the harness's bytes are deterministic)
    tstore.reset_sessions()


def test_swaps_per_rearrangement_are_bounded_by_the_spare_rows(monkeypatch, tmp_path):
    """Two spare rows: at most two experts are read per layer and rank per
    rearrangement, so three hot experts reach the card over two cycles."""
    env, model, state, model_state, log = _build(
        monkeypatch, tmp_path, spare=2, max_moves_per_rank=2
    )
    loads = [[1, 1, 1, 0, 2, 40, 50, 60], [1, 1, 1, 0, 2, 40, 50, 60]]
    _window(model_state, loads)
    _run_async_cycle(state, model_state)
    first = set(np.flatnonzero(env.tiers[0].placement.tier == 0).tolist())
    assert len(first & {5, 6, 7}) == 2 and len(first) == FAST
    _window(model_state, loads)
    _run_async_cycle(state, model_state)
    assert set(np.flatnonzero(env.tiers[0].placement.tier == 0).tolist()) == {5, 6, 7}
    for i in range(LAYERS):
        for slot in range(E):
            assert _holds(env, i, slot, slot), (i, slot)
    tstore.reset_sessions()


def _fast_set(tiers) -> set[int]:
    return set(np.flatnonzero(tiers.placement.tier == 0).tolist())


def _maps_agree(tiers) -> bool:
    fm, sm = tiers.placement.maps(tiers.slot_global, tiers.num_global)
    return tiers.fast_map.tolist() == fm.tolist() and (
        tiers.slow_map.tolist() == sm.tolist()
    )


@pytest.mark.parametrize("bad_layer", [0, 1])
def test_a_failed_read_abandons_the_rearrangement_and_serving_goes_on(
    monkeypatch, tmp_path, bad_layer
):
    """A layer whose reads fail: nothing of it is mapped, its rows are back,
    the main thread sees no error, the cycle ends, and the next cycle works."""
    from vllm.model_executor.offloader.expert_store import ExpertStoreError

    env, model, state, model_state, log = _build(monkeypatch, tmp_path, spare=4)
    loads = [[1, 1, 1, 0, 2, 40, 50, 60]] * LAYERS
    _window(model_state, loads)
    bad = env.tiers[bad_layer]
    real_fill = bad._fill_rows

    def drive_gone(reads, rows):
        raise ExpertStoreError("drive gone")

    bad._fill_rows = drive_gone
    free = [t.placement.pool.free_count for t in env.tiers]
    _run_async_cycle(state, model_state)  # raises if the worker or the hook did
    assert state.async_worker_error is None
    assert model_state.abandoned_rearrangements == 1
    assert any(f"abandoned at layer {bad_layer} of" in x for x in log.lines)
    # layers before the bad one are committed, the bad one and later are not
    for i, tiers in enumerate(env.tiers):
        want = {5, 6, 7} if i < bad_layer else {0, 1, 2}
        assert _fast_set(tiers) == want, i
        assert _maps_agree(tiers)
        tiers.placement.check()
        for slot in range(E):
            assert _holds(env, i, slot, slot), (i, slot)
    assert env.tiers[bad_layer].placement.pool.free_count == free[bad_layer]
    # the next cycle, with the reads back, finishes the job
    bad._fill_rows = real_fill
    _window(model_state, loads)
    _run_async_cycle(state, model_state)
    assert all(_fast_set(t) == {5, 6, 7} for t in env.tiers)
    assert model_state.abandoned_rearrangements == 1
    tstore.reset_sessions()


def test_a_rank_gives_its_rows_back_when_a_peer_could_not_prepare(
    monkeypatch, tmp_path
):
    """Every rank takes the same branch: this rank prepared the layer, a peer
    did not, so this rank commits nothing and returns what it read into."""
    env, model, state, model_state, log = _build(monkeypatch, tmp_path, spare=4)
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)

    def with_a_peers_vote(flags):
        flags += torch.tensor([0, 0, 1], dtype=flags.dtype)  # a peer abandons
        return flags

    monkeypatch.setattr(
        eplb_state_mod,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=group, cpu_all_reduce_=with_a_peers_vote),
    )
    _window(model_state, [[1, 1, 1, 0, 2, 40, 50, 60]] * LAYERS)
    free = [t.placement.pool.free_count for t in env.tiers]
    _run_async_cycle(state, model_state)
    assert model_state.abandoned_rearrangements == 1
    assert any("see their logs" in x for x in log.lines)
    for i, tiers in enumerate(env.tiers):
        assert _fast_set(tiers) == {0, 1, 2} and _maps_agree(tiers)
        assert tiers.placement.pool.free_count == free[i]
    tstore.reset_sessions()


def test_scrub_compares_what_the_kernels_read_with_the_store(monkeypatch, tmp_path):
    env, model, state, model_state, log = _build(monkeypatch, tmp_path, spare=4)
    (summary,) = state.scrub_expert_store()  # on request, whatever the setting
    assert summary["slots"] == LAYERS * E and summary["layers"] == LAYERS
    assert summary["mismatches"] == 0 and summary["first"] is None
    assert summary["maps_differ"] == 0 and summary["placement_differs"] == 0
    assert any(
        x.startswith("Expert store scrub (EP rank 0, on request, every slot)")
        for x in log.lines
    )
    # after a rearrangement every slot still holds its unit
    _window(model_state, [[1, 1, 1, 0, 2, 40, 50, 60]] * LAYERS)
    _run_async_cycle(state, model_state)
    assert state.scrub_expert_store()[0]["mismatches"] == 0
    tiers = env.tiers[0]
    fast, _ = tiers._impls
    placement = tiers.placement
    assert placement.tier[5] == 0 and placement.tier[0] == 1  # 5 up, 0 down

    # a wrong byte in a fast row
    cell = fast.w2[int(placement.row[5])].view(torch.uint8).view(-1)
    cell[3] ^= 1
    bad = state.scrub_expert_store()[0]
    assert bad["mismatches"] == 1
    assert "slot 5 (expert 5, fast row" in bad["first"]
    assert any("first mismatch" in x and "slot 5" in x for x in log.lines)
    cell[3] ^= 1
    # a wrong byte where the kernels read a pool row: through the view
    cell = tiers.slow_views["w13_scales"][int(placement.row[0])]
    cell = cell.view(torch.uint8).view(-1)
    cell[0] ^= 1
    bad = state.scrub_expert_store()[0]
    assert bad["mismatches"] == 1 and "slot 0 (expert 0, pool row" in bad["first"]
    cell[0] ^= 1
    # a tier map on the device that is not the placement's
    g = tiers.slot_global[5]
    row = int(tiers.fast_map[g])
    tiers.fast_map[g] = -1
    bad = state.scrub_expert_store()[0]
    assert bad["maps_differ"] == 1 and "tier maps on the device" in bad["first"]
    tiers.fast_map[g] = row
    # EPLB routing another expert to a slot than the one its row holds
    model_state.physical_to_logical_map[0, 2] = 3
    bad = state.scrub_expert_store()[0]
    assert bad["placement_differs"] == 1 and "EPLB routes expert 3" in bad["first"]
    model_state.physical_to_logical_map[0, 2] = 2
    clean = state.scrub_expert_store()[0]
    assert clean["mismatches"] == 0 and clean["first"] is None
    tstore.reset_sessions()


def test_scrub_every_checks_after_load_and_after_each_rearrangement(
    monkeypatch, tmp_path
):
    env, model, state, model_state, log = _build(monkeypatch, tmp_path, spare=4)

    def scrubs(kind):
        head = f"Expert store scrub (EP rank 0, {kind}"
        return [x for x in log.lines if x.startswith(head)]

    # off by default: a step and a rearrangement scrub nothing
    state.step(is_dummy=True, is_profile=True)
    _window(model_state, [[1, 1, 1, 0, 2, 40, 50, 60]] * LAYERS)
    _run_async_cycle(state, model_state)
    assert not [x for x in log.lines if x.startswith("Expert store scrub")]
    # on: the first step after load checks every slot, once
    env.store_cfg.scrub_every = 1
    model_state.load_scrubbed = False
    state.step(is_dummy=True, is_profile=True)
    state.step(is_dummy=True, is_profile=True)
    assert len(scrubs("load, every slot")) == 1
    assert f"{LAYERS * E} slots of {LAYERS} layers" in scrubs("load")[0]
    assert ", 0 mismatches" in scrubs("load")[0]
    # ... and each rearrangement checks the slots it moved
    _window(model_state, [[60, 50, 40, 0, 2, 1, 1, 1]] * LAYERS)
    _run_async_cycle(state, model_state)
    (line,) = scrubs("rearrangement, slots moved since the last scrub")
    moved = int(line.split("): ")[1].split(" slots")[0])
    assert 0 < moved < LAYERS * E and ", 0 mismatches" in line
    tstore.reset_sessions()


def test_a_prepared_move_that_is_dropped_gives_its_rows_back(monkeypatch, tmp_path):
    env, model, state, model_state, log = _build(monkeypatch, tmp_path, spare=3)
    tiers = env.tiers[0]
    free = tiers.placement.pool.free_count
    new = tiers.slot_logical.copy()
    load = np.array([1.0, 1, 1, 0, 2, 40, 50, 60])
    prepared = tiers.prepare_move(new, load, 1.1)
    assert tiers.placement.pool.free_count < free  # three rows hold read experts
    model_state.movers[0].discard(prepared)
    assert tiers.placement.pool.free_count == free
    # the placement did not move
    assert tiers.slot_logical.tolist() == list(range(E))
    assert set(np.flatnonzero(tiers.placement.tier == 0).tolist()) == {0, 1, 2}
    tstore.reset_sessions()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert tiers backed by the expert store, on CPU: the store is built from a
synthetic int4 checkpoint at the first load and fills the host tier at the
next, the host rows equal the ones the checkpoint path produces, and tier moves
read the store."""

import numpy as np
import pytest
import torch

from vllm.model_executor.offloader import expert_tier_store as tstore

from .xpu_store_harness import FAST, LAYERS, E, Env, expert_in_store, pool_rows, run

pytestmark = pytest.mark.skip_global_cleanup


def test_build_then_fill_give_the_host_rows_the_checkpoint_path_gives(
    monkeypatch, tmp_path
):
    # reference: no store, today's path
    ref = run(monkeypatch, tmp_path / "ref", store=False)
    ref_rows = [pool_rows(t) for t in ref.tiers]
    assert all(t.store_binding is None and t.placement is None for t in ref.tiers)
    assert all(t.slow["w13"].shape[0] == t.num_slow for t in ref.tiers)

    # first load: the store is built, the host rows come from the checkpoint
    first = run(monkeypatch, tmp_path, store=True)
    session = first.tiers[0].store_binding.session
    assert session.mode == "build" and session.is_open
    assert all(t.slow["w13"].shape[0] == t.num_slow + 2 for t in first.tiers)
    for built, want in zip(first.tiers, ref_rows):
        for key, rows in pool_rows(built).items():
            assert torch.equal(rows, want[key]), key
    # the spare rows are zero
    assert not first.tiers[0].slow["w13"][first.tiers[0].num_slow :].any()
    # every expert is in the store in kernel layout: fast ones as the kernel
    # reads them after its relayout, host ones as the repack gives them
    for i, tiers in enumerate(first.tiers):
        for slot in range(E):
            got = expert_in_store(first, i, slot)
            if slot < FAST:
                fast = tiers._impls[0]
                want = {
                    "w13": fast.w13[slot].view(torch.uint8),
                    "w2": fast.w2[slot].view(torch.uint8),
                    "w13_scales": fast.gemm1_wei_scales[slot],
                    "w2_scales": fast.gemm2_wei_scales[slot],
                }
            else:
                row = slot - FAST
                want = {k: v[row] for k, v in tiers.slow.items()}
            for key, tensor in want.items():
                assert torch.equal(
                    got[key].reshape(-1).view(torch.uint8),
                    tensor.reshape(-1).view(torch.uint8),
                ), (i, slot, key)
    first_rows = [pool_rows(t) for t in first.tiers]
    tstore.reset_sessions()

    # second load: the loader skips the host slots, the store fills them
    second = run(monkeypatch, tmp_path, store=True)
    session = second.tiers[0].store_binding.session
    assert session.mode == "fill"
    assert all(t.fills_from_store for t in second.tiers)
    assert all(t.twins == {} for t in second.tiers)  # no pageable twins
    for filled, want in zip(second.tiers, ref_rows):
        for key, rows in pool_rows(filled).items():
            assert torch.equal(rows, want[key]), key
    for filled, built in zip([pool_rows(t) for t in second.tiers], first_rows):
        for key in filled:
            assert torch.equal(filled[key], built[key])
    stats = session.store.stats()
    assert stats["reads"] >= LAYERS * (E - FAST)
    tstore.reset_sessions()


def test_a_store_of_another_checkpoint_is_rebuilt_or_refused(monkeypatch, tmp_path):
    run(monkeypatch, tmp_path, store=True)
    tstore.reset_sessions()
    # same dirs, other checkpoint identity: auto rebuilds
    monkeypatch.setattr(
        tstore, "checkpoint_identity", lambda model_config: "another-checkpoint"
    )
    env = run(monkeypatch, tmp_path, store=True)
    assert env.tiers[0].store_binding.session.mode == "build"
    assert env.tiers[0].store_binding.session.status == "mismatch"
    tstore.reset_sessions()
    # build=never refuses a store that does not match
    monkeypatch.setattr(
        tstore, "checkpoint_identity", lambda model_config: "third-checkpoint"
    )
    env = Env(monkeypatch, tmp_path, True, build="never")
    with pytest.raises(ValueError, match="build=never"):
        env.construct()
    tstore.reset_sessions()
    # build=rebuild builds even when the store matches
    monkeypatch.setattr(
        tstore, "checkpoint_identity", lambda model_config: "another-checkpoint"
    )
    env = run(monkeypatch, tmp_path, store=True, build="rebuild")
    assert env.tiers[0].store_binding.session.mode == "build"
    tstore.reset_sessions()


def test_the_staging_pool_is_pinned_only_when_asked(monkeypatch, tmp_path):
    """Direct reads must land in memory every kernel takes: the staging slots
    are ordinary memory unless ``pool_pinned`` is set."""
    env = Env(monkeypatch, tmp_path, True)
    monkeypatch.setattr(
        "vllm.utils.platform_utils.is_pin_memory_available", lambda: True
    )
    env.construct()
    assert env.tiers[0].store_binding.session.pin is False
    tstore.reset_sessions()
    env = Env(monkeypatch, tmp_path, True)
    monkeypatch.setattr(
        "vllm.utils.platform_utils.is_pin_memory_available", lambda: True
    )
    env.store_cfg.pool_pinned = True
    env.construct()
    assert env.tiers[0].store_binding.session.pin is True
    tstore.reset_sessions()


def _flip_unit(env, layer: int, logical: int) -> None:
    store = env.tiers[layer].store_binding.session.store
    stripe, offset, _ = store.layout.locate(layer, logical)
    with open(store._stripes[stripe].path, "r+b") as handle:
        handle.seek(offset + 10)
        byte = handle.read(1)
        handle.seek(offset + 10)
        handle.write(bytes([byte[0] ^ 0x20]))


def test_a_unit_that_fails_its_digest_is_never_mapped(monkeypatch, tmp_path):
    from vllm.model_executor.offloader.expert_store import StoreCorruptError

    env = Env(monkeypatch, tmp_path, True)
    env.store_cfg.check_every = 1
    env.construct()
    env.load()
    env.process()
    tiers = env.tiers[1]
    placement = tiers.placement
    _flip_unit(env, 1, 1)  # the unit of slot 1 (fast at load)
    free = placement.pool.free_count
    tier, row = placement.tier.copy(), placement.row.copy()
    maps = (tiers.fast_map.clone(), tiers.slow_map.clone())
    with pytest.raises(StoreCorruptError, match=r"unit \(1, 1\)"):
        tiers.demote(1)  # would read the unit into a pool row and map it
    assert placement.pool.free_count == free
    assert (placement.tier == tier).all() and (placement.row == row).all()
    assert torch.equal(tiers.fast_map, maps[0])
    assert torch.equal(tiers.slow_map, maps[1])
    stats = tiers.store_binding.session.store.stats()
    assert stats["check_failures"] == 1
    tiers.demote(2)  # a clean unit still moves
    assert placement.tier[2] == 1
    tstore.reset_sessions()


def test_tier_moves_read_the_store_and_keep_every_slot_mapped_once(
    monkeypatch, tmp_path
):
    env = run(monkeypatch, tmp_path, store=True)
    tiers = env.tiers[1]
    placement = tiers.placement
    assert placement is not None and placement.num_pool_rows == E - FAST + 2
    fast, slow = tiers._impls
    assert slow.num_experts == placement.num_pool_rows  # the capacity, not the slots

    def bytes_of_slot(slot):
        if placement.tier[slot] == 0:
            r = int(placement.row[slot])
            return {
                "w13": fast.w13[r].view(torch.uint8).clone(),
                "w2": fast.w2[r].view(torch.uint8).clone(),
            }
        r = int(placement.row[slot])
        return {"w13": tiers.slow["w13"][r].clone(), "w2": tiers.slow["w2"][r].clone()}

    def mapped_once():
        fm, sm = tiers.fast_map, tiers.slow_map
        for slot, g in enumerate(tiers.slot_global):
            f, p = int(fm[g]), int(sm[g])
            assert (f >= 0) != (p >= 0), slot
            tier, row = (0, f) if f >= 0 else (1, p)
            assert (int(placement.tier[slot]), int(placement.row[slot])) == (tier, row)

    mapped_once()
    expect = {
        s: {"w13": e["w13"].reshape(-1), "w2": e["w2"].reshape(-1)}
        for s in range(E)
        for e in [expert_in_store(env, 1, s)]
    }
    # swap a hot pool slot with a cold fast slot
    tiers.swap(promote=6, demote=1)
    mapped_once()
    assert placement.tier[6] == 0 and placement.tier[1] == 1
    for slot in (6, 1):
        got = bytes_of_slot(slot)
        for key in got:
            assert torch.equal(
                got[key].reshape(-1).view(torch.uint8),
                expect[slot][key].view(torch.uint8),
            ), (slot, key)
    # demote alone frees a fast row, promote uses it
    tiers.demote(0)
    assert placement.fast_free == [0]
    tiers.promote(4)
    assert placement.fast_free == [] and placement.tier[4] == 0
    with pytest.raises(RuntimeError, match="no free fast row"):
        tiers.promote(5)
    mapped_once()
    for slot in range(E):
        got = bytes_of_slot(slot)
        for key in got:
            assert torch.equal(
                got[key].reshape(-1).view(torch.uint8),
                expect[slot][key].view(torch.uint8),
            ), (slot, key)

    # a placement change: new experts in two slots, read from the store
    new = tiers.slot_logical.copy()
    new[2], new[5] = 3, 7  # experts other slots hold: replicas, fine for a test
    load = np.ones(E)
    load[5] = 100.0
    move = tiers.prepare_move(new, load, 1.1)
    assert move.stats["units"] >= 2
    tiers.commit_move(move)
    mapped_once()
    assert tiers.slot_logical.tolist() == new.tolist()
    for slot in (2, 5):
        got = bytes_of_slot(slot)
        want = expert_in_store(env, 1, int(new[slot]))
        assert torch.equal(
            got["w13"].reshape(-1).view(torch.uint8),
            want["w13"].reshape(-1).view(torch.uint8),
        )
    assert placement.tier[5] == 0  # the hot new expert is on the card
    tstore.reset_sessions()

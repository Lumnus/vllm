# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The store session of the expert tiers across ranks: two ranks, as threads
with a real barrier between them, build one store; a build that died is
resumed; what the session refuses."""

import os
import threading

import numpy as np
import pytest

from vllm.config.expert_store import ExpertStoreConfig
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_store import ExpertStore, ExpertStoreWriter

SIZES = {"w13": 4096, "w2": 2048, "w13_scales": 512, "w2_scales": 1024}


def _tensors(layer: int, e: int, salt: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(layer * 100 + e + salt)
    return {k: rng.integers(0, 256, n, dtype=np.uint8) for k, n in SIZES.items()}


def _names(layers: int) -> list[str]:
    return [f"model.layers.{i}.mlp.experts" for i in range(layers)]


def _build_on_two_ranks(cfg, sch, ident, layers, experts):
    """Each rank registers every layer, writes the half of the experts it
    owns layer by layer, and reports each layer done, as the tiers do."""
    gate = threading.Barrier(2, timeout=20)
    errors: list[BaseException] = []
    sessions: dict[int, tstore.StoreSession] = {}

    def rank(r: int) -> None:
        try:
            session = tstore.StoreSession(
                cfg, sch, ident, rank=r, num_ranks=2, barrier=gate.wait, pin=False
            )
            sessions[r] = session
            assert session.mode == "build"
            for name in _names(layers):
                session.register(name)
            own = range(r * experts // 2, (r + 1) * experts // 2)
            for layer, name in enumerate(_names(layers)):
                session.write_units(layer, [(e, _tensors(layer, e)) for e in own])
                session.layer_done(name)
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            errors.append(exc)
            gate.abort()

    threads = [threading.Thread(target=rank, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    return sessions, errors


def _check_all_units(store: ExpertStore, layers: int, experts: int) -> None:
    for layer in range(layers):
        for e in range(experts):
            with store.get((layer, e)) as entry:
                got = entry.tensors()
                for k, want in _tensors(layer, e).items():
                    assert np.array_equal(got[k].numpy().reshape(-1), want)


def _config(tmp_path, stripes: int) -> ExpertStoreConfig:
    dirs = [str(tmp_path / f"d{i}") for i in range(stripes)]
    for d in dirs:
        os.makedirs(d)
    return ExpertStoreConfig(
        dirs=dirs, pool_gib=0.001, io_threads=2, direct=False, check_every=1
    )


def test_two_ranks_build_one_store_through_the_barriers(tmp_path):
    layers, experts = 3, 8
    sch = tstore.build_schema(layers, experts, SIZES)
    ident = tstore.store_identity(sch, "ckpt", "transform")
    cfg = _config(tmp_path, stripes=2)
    sessions, errors = _build_on_two_ranks(cfg, sch, ident, layers, experts)
    assert not errors, errors
    assert ExpertStore.status(cfg.dirs, ident) == "complete"
    for r in range(2):
        assert sessions[r].is_open
        _check_all_units(sessions[r].store, layers, experts)
        assert sessions[r].store.stats()["check_failures"] == 0
        sessions[r].close()


def test_two_ranks_resume_a_build_that_died_after_one_rank_closed(tmp_path):
    """First start: rank 1 wrote (other bytes) and closed, rank 0 died before
    it wrote. The next start builds again and ends with the right bytes."""
    layers, experts = 2, 8
    sch = tstore.build_schema(layers, experts, SIZES)
    ident = tstore.store_identity(sch, "ckpt", "transform")
    cfg = _config(tmp_path, stripes=1)
    w0 = ExpertStoreWriter(cfg.dirs, sch, ident, writer_id="rank0")
    w1 = ExpertStoreWriter(cfg.dirs, sch, ident, writer_id="rank1")
    for layer in range(layers):
        for e in range(4, 8):
            w1.write(layer, e, _tensors(layer, e, salt=7))
    w1.close()
    del w0
    assert ExpertStore.status(cfg.dirs, ident) == "incomplete"
    sessions, errors = _build_on_two_ranks(cfg, sch, ident, layers, experts)
    assert not errors, errors
    _check_all_units(sessions[0].store, layers, experts)
    for session in sessions.values():
        session.close()


def test_a_store_of_another_identity_is_refused_and_left_alone(tmp_path):
    layers, experts = 2, 8
    sch = tstore.build_schema(layers, experts, SIZES)
    ident = tstore.store_identity(sch, "ckpt", "transform")
    cfg = _config(tmp_path, stripes=1)
    sessions, errors = _build_on_two_ranks(cfg, sch, ident, layers, experts)
    assert not errors, errors
    for session in sessions.values():
        session.close()
    other = tstore.store_identity(sch, "ckpt", "another transform")
    assert other != ident
    with pytest.raises(ValueError, match="does not replace it"):
        tstore.StoreSession(
            cfg, sch, other, rank=0, num_ranks=1, barrier=lambda: None, pin=False
        )
    assert ExpertStore.status(cfg.dirs, ident) == "complete"
    # an unfinished store of another identity is refused the same way,
    # before any rank could wait in a barrier for a writer that cannot open
    half = _config(tmp_path / "half", stripes=1)
    ExpertStoreWriter(half.dirs, sch, ident, writer_id="rank0")
    with pytest.raises(ValueError, match="does not replace it"):
        tstore.StoreSession(
            half, sch, other, rank=1, num_ranks=2, barrier=lambda: None, pin=False
        )


def test_a_build_with_a_layer_missing_is_refused_before_the_first_barrier(tmp_path):
    layers, experts = 3, 8
    sch = tstore.build_schema(layers, experts, SIZES)
    ident = tstore.store_identity(sch, "ckpt", "transform")
    cfg = _config(tmp_path, stripes=1)
    waited = []
    session = tstore.StoreSession(
        cfg,
        sch,
        ident,
        rank=1,
        num_ranks=2,
        barrier=lambda: waited.append(1),
        pin=False,
    )
    for name in _names(layers)[:2]:  # one decoder layer has no routed experts
        session.register(name)
    with pytest.raises(ValueError, match="every decoder layer"):
        session.write_units(0, [(4, _tensors(0, 4))])
    assert not waited and os.listdir(cfg.dirs[0]) == []

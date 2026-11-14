# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The card tier's resize beside EPLB's moves, at the engine core: EPLB owns
every expert move, so a resize whose commit finds one of its rearrangements
or tier passes in flight waits for it between steps (up to its deadline)
instead of failing; any other refusal still fails at once. Fake engine and
ranks of the card tier's own tests."""

import time
from types import SimpleNamespace

import pytest

from vllm.v1.engine.fast_tier import FastTierController, FastTierState

from .test_fast_tier_resize import Engine, Ranks

IN_FLIGHT = (
    "rank 0: an async rearrangement of m is in flight at layer 3 of 48; "
    "ask after it completes"
)


class BusyRanks(Ranks):
    """Ranks whose EPLB has a pass in flight for the next `busy` commits."""

    def __init__(self, busy: int, **kw):
        super().__init__(**kw)
        self.busy = busy

    def worker_op(self, op, args):
        if self.busy:
            self.busy -= 1
            self.ops.append((op, args["rows"], "refused"))
            return {"op": op, "applied": "none", "ranks": [], "error": IN_FLIGHT}
        return super().worker_op(op, args)


def _setup(busy, drain_steps=1):
    engine, ranks = Engine(drain_steps=drain_steps), BusyRanks(busy)
    control = SimpleNamespace(_rpc=ranks.rpc, _worker_op=ranks.worker_op)
    return engine, ranks, FastTierController(engine, control)


def test_a_grow_waits_for_eplbs_pass_then_commits():
    engine, ranks, ctl = _setup(busy=2)
    reply = ctl.start({"rows": 16})
    assert reply["fast"]["state"] == "draining"
    engine.drain()
    ctl.advance()  # the KV memory is released; EPLB moves experts: wait
    assert ctl.resize.state == FastTierState.COMMITTING
    assert "waiting for EPLB" in ctl.resize.message
    ctl.advance()  # still in flight
    assert ctl.resize.state == FastTierState.COMMITTING
    ctl.advance()  # the pass ended: the rows are committed
    assert ctl.resize.state == FastTierState.FINISHED
    assert ranks.rows == [16, 16] and engine.resizes == []  # nothing given back


def test_a_grow_gives_up_at_its_deadline_and_the_kv_cache_gets_its_memory_back():
    engine, ranks, ctl = _setup(busy=10**6)
    ctl.start({"rows": 16, "deadline_s": 0.05})
    blocks_before = ctl.resize.kv_blocks_before
    engine.drain()
    ctl.advance()
    assert ctl.resize.state == FastTierState.COMMITTING
    time.sleep(0.1)
    ctl.advance()
    assert ctl.resize.state == FastTierState.FAILED
    assert "in flight" in ctl.resize.message
    assert ranks.rows == [8, 8] and engine.resizes == [blocks_before]


def test_a_shrink_waits_for_eplbs_pass_too():
    engine, ranks, ctl = _setup(busy=1)
    reply = ctl.start({"rows": 4})
    assert reply["applied"] == "all" and reply["fast"]["state"] == "committing"
    ctl.advance()
    assert ctl.resize.state == FastTierState.FINISHED
    assert ranks.rows == [4, 4] and len(engine.resizes) == 1  # KV grown by it


@pytest.mark.parametrize("rows", [16, 4])
def test_any_other_refusal_fails_at_once(rows):
    engine, ranks, ctl = _setup(busy=0)
    ranks.fail_rank = 1
    ctl.start({"rows": rows, "from_kv": False})
    assert ctl.resize.state == FastTierState.FAILED
    assert ctl.resize.message == "rank 1: injected"

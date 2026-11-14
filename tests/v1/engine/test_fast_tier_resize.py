# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine core's side of a card tier resize (runtime control
``tiers.fast``): the device memory taken from the KV cache by a shrink that
drains, the rows committed on every rank once it has released, refusals,
failures and the memory given back; with a fake engine and fake ranks."""

from types import SimpleNamespace

import pytest

from vllm.v1.engine.fast_tier import FastTierController, FastTierState
from vllm.v1.engine.kv_cache_resize import KVCacheResize, KVCacheResizeState
from vllm.v1.engine.runtime_control import ControlRefused, RuntimeControl

GIB = 1 << 30
PER_BLOCK = 1 << 20


class Ranks:
    """Two ranks' card tiers as the workers report and change them."""

    def __init__(self, rows=8, lo=1, hi=24, per_row=GIB // 8):
        self.rows = [rows, rows]
        self.lo, self.hi, self.per_row = lo, hi, per_row
        self.fail_rank: int | None = None
        self.ops: list[tuple] = []
        # what the ranks' check answers (precheck and the commit's check)
        self.refusal: str | None = None
        self.checks: list[dict] = []

    def fast(self, rank, targets):
        return {
            "rows": self.rows[rank],
            "min": self.lo,
            "max": self.hi,
            "torch": getattr(self, "torch", None),
            "bytes_to": {
                str(t): max(t - self.rows[rank], 0) * self.per_row for t in targets
            },
        }

    def rpc(self, phase, op="", args=None):
        args = args or {}
        if phase == "check":
            self.checks.append(dict(args))
            return [{"rank": r, "refusal": self.refusal} for r in range(2)]
        assert phase == "state"
        return [
            {
                "rank": r,
                "tiers": {"fast": self.fast(r, args.get("fast_rows") or [])},
                "kv": {"bytes_per_block": PER_BLOCK, "committed_blocks": 4000},
            }
            for r in range(2)
        ]

    def worker_op(self, op, args):
        assert op == "tiers.fast"
        self.checks.append(dict(args))
        if self.refusal:
            # as RuntimeControl._worker_op: a check refusal is raised
            raise ControlRefused(f"rank 0: {self.refusal}")
        self.ops.append((op, args["rows"]))
        old = list(self.rows)
        replies = []
        for r in range(2):
            if r == self.fail_rank:
                replies.append({"rank": r, "ok": False, "error": "injected"})
                continue
            released = max(self.rows[r] - args["rows"], 0) * self.per_row
            self.rows[r] = args["rows"]
            replies.append(
                {
                    "rank": r,
                    "ok": True,
                    "result": {"fast": {"released_bytes": released}},
                }
            )
        if self.fail_rank is not None:
            self.rows = old  # the others undone
            return {
                "op": op,
                "applied": "none",
                "ranks": replies,
                "rolled_back": True,
                "error": "rank 1: injected",
            }
        return {"op": op, "applied": "all", "ranks": replies}


class Engine:
    """The engine core's KV resize surface, enough for the controller."""

    def __init__(self, blocks=4000, drain_steps=1):
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks=blocks)
        )
        self._kv_resize = None
        self.drain_steps = drain_steps
        self.refuse: str | None = None
        self.resizes: list[int] = []
        self.restores: list[bool] = []
        self.published = 0

    def request_kv_cache_release(self, bytes_per_rank, urgent=False, deadline_s=None):
        if self.refuse:
            raise ValueError(self.refuse)
        blocks = -(-bytes_per_rank // PER_BLOCK)
        cur = self.vllm_config.cache_config.num_gpu_blocks
        self._kv_resize = KVCacheResize(
            "shrink", cur, cur - blocks, KVCacheResizeState.DRAINING,
            deadline_s=deadline_s or 0, preempt=urgent,
        )  # fmt: skip
        return self._kv_resize.record(cur)

    def drain(self, state=KVCacheResizeState.FINISHED):
        """One step boundary of the busy loop: the shrink moves on."""
        kv = self._kv_resize
        if kv is None or not kv.state.active:
            return
        self.drain_steps -= 1
        if self.drain_steps > 0 and state == KVCacheResizeState.FINISHED:
            return
        kv.state = state
        if state == KVCacheResizeState.FINISHED:
            self.vllm_config.cache_config.num_gpu_blocks = kv.to_blocks

    def resize_kv_cache(self, target, deadline_s=None, preempt=False, restore=False):
        self.resizes.append(target)
        self.restores.append(restore)
        cur = self.vllm_config.cache_config.num_gpu_blocks
        self.vllm_config.cache_config.num_gpu_blocks = target
        self._kv_resize = KVCacheResize(
            "grow", cur, target, KVCacheResizeState.FINISHED
        )
        return self._kv_resize.record(target)

    def _publish_scheduler_stats(self):
        self.published += 1


@pytest.fixture
def setup():
    engine, ranks = Engine(drain_steps=2), Ranks()
    control = SimpleNamespace(_rpc=ranks.rpc, _worker_op=ranks.worker_op)
    return engine, ranks, FastTierController(engine, control)


def test_a_grow_drains_the_kv_cache_then_commits(setup):
    engine, ranks, ctl = setup
    reply = ctl.start({"rows": 16})
    fast = reply["fast"]
    assert reply["applied"] == "all" and fast["state"] == "draining"
    assert fast["direction"] == "grow" and fast["bytes_per_rank"] == GIB
    assert fast["kv"]["to_blocks"] == 4000 - GIB // PER_BLOCK
    assert ranks.ops == []  # nothing on the ranks until the memory is free
    engine.drain()
    ctl.advance()
    assert ranks.ops == [] and ctl.resize.state == FastTierState.DRAINING
    engine.drain()
    ctl.advance()  # the busy loop's next iteration, right after the release
    assert ranks.ops == [("tiers.fast", 16)] and ranks.rows == [16, 16]
    record = ctl.resize.record(16)
    assert record["state"] == "finished"
    assert (
        record["kv_blocks_before"] == 4000 and record["kv_blocks_after"] == 4000 - 1024
    )
    assert engine.published == 1


def test_a_grow_on_an_idle_engine_commits_within_the_call(setup):
    engine, ranks, ctl = setup
    engine.drain_steps = 0

    real = engine.request_kv_cache_release

    def instant(*a, **kw):
        record = real(*a, **kw)
        engine.drain()  # an idle pool drains in the call (_begin_kv_cache_shrink)
        return record

    engine.request_kv_cache_release = instant
    reply = ctl.start({"delta": 4})
    assert reply["fast"]["state"] == "finished" and ranks.rows == [12, 12]


def test_a_shrink_gives_the_memory_back_to_the_kv_cache(setup):
    engine, ranks, ctl = setup
    reply = ctl.start({"rows": 4})
    assert reply["applied"] == "all"
    fast = reply["fast"]
    assert fast["state"] == "finished" and fast["direction"] == "shrink"
    assert ranks.rows == [4, 4] and fast["bytes_per_rank"] == GIB // 2
    # the KV cache grows by what the rows released, in whole blocks
    assert engine.resizes == [4000 + (GIB // 2) // PER_BLOCK]
    assert fast["kv_blocks_after"] == 4000 + 512


def test_refusals_change_nothing(setup):
    engine, ranks, ctl = setup
    with pytest.raises(ControlRefused, match="outside 1..24"):
        ctl.start({"rows": 30})
    with pytest.raises(ControlRefused, match="exactly one"):
        ctl.start({"rows": 3, "delta": 1})
    with pytest.raises(ControlRefused, match="integers"):
        ctl.start({"rows": "3"})
    with pytest.raises(ControlRefused, match="unknown"):
        ctl.start({"rows": 3, "now": True})
    engine.refuse = "the floor: one request of max_model_len must fit"
    with pytest.raises(ControlRefused, match="cannot give back .* max_model_len"):
        ctl.start({"rows": 20})
    engine.refuse = None
    engine._kv_resize = KVCacheResize("shrink", 4000, 3900, KVCacheResizeState.DRAINING)
    with pytest.raises(ControlRefused, match="KV cache resize is in progress"):
        ctl.start({"rows": 20})
    assert ranks.ops == [] and ranks.rows == [8, 8]
    engine._kv_resize = None
    ctl.start({"rows": 12})
    with pytest.raises(ControlRefused, match="one at a time"):
        ctl.start({"rows": 4})


def test_ranks_that_disagree_are_refused(setup):
    _, ranks, ctl = setup
    ranks.rows = [8, 6]
    with pytest.raises(ControlRefused, match="different card rows"):
        ctl.start({"rows": 8})


def test_an_aborted_kv_shrink_leaves_the_card_tier(setup):
    engine, ranks, ctl = setup
    ctl.start({"rows": 16})
    engine.drain(KVCacheResizeState.ABORTED)
    engine._kv_resize.message = "deadline of 60 s passed with 3 fenced blocks held"
    ctl.advance()
    record = ctl.resize.record(8)
    assert record["state"] == "failed" and "aborted" in record["message"]
    assert "deadline" in record["message"]
    assert ranks.ops == [] and ranks.rows == [8, 8]
    assert engine.resizes == []  # the aborted shrink gave its blocks back itself


def test_a_rank_that_cannot_take_the_rows_gives_the_kv_back(setup):
    engine, ranks, ctl = setup
    engine.drain_steps = 1
    ctl.start({"rows": 16})
    ranks.fail_rank = 1
    engine.drain()
    ctl.advance()
    record = ctl.resize.record(8)
    assert record["state"] == "failed" and "injected" in record["message"]
    assert record["rolled_back"] is True and ranks.rows == [8, 8]
    # the KV cache grows back to its size before the grow
    assert engine.resizes == [4000] and record["kv_blocks_after"] == 4000


def test_unchanged_and_no_kv(setup):
    engine, ranks, ctl = setup
    assert ctl.start({"delta": 0})["fast"]["state"] == "unchanged"
    reply = ctl.start({"rows": 10, "from_kv": False})
    assert reply["fast"]["state"] == "finished" and ranks.rows == [10, 10]
    assert engine._kv_resize is None  # no KV resize
    reply = ctl.start({"rows": 8, "from_kv": False})
    assert engine.resizes == [] and ranks.rows == [8, 8]


def test_the_surface_routes_the_verb_and_reports_the_tier():
    engine, ranks = Engine(drain_steps=1), Ranks()
    engine._tower_residency = None
    engine.get_kv_cache_resize_state = lambda: {"state": "idle"}
    engine.vllm_config.cache_config.enable_extensible_kv_cache = True
    engine.vllm_config.cache_config.kv_cache_release_chunk_mib = 2
    engine.vllm_config.cache_config.kv_cache_grow_torch_floor_mib = None
    control = RuntimeControl(engine)
    control._rpc = ranks.rpc
    control._worker_op = ranks.worker_op
    assert "tiers.fast" in control.state(["tiers"])["operations"]
    reply = control.apply("tiers.fast", {"rows": 4})
    assert reply["fast"]["state"] == "finished"
    assert engine._fast_tier is not None
    state = control.state(["tiers"])
    assert state["fast_tier"]["rows"] == 4 and state["fast_tier"]["range"] == [1, 24]
    assert state["fast_tier"]["record"]["to_rows"] == 4
    assert state["fast_tier"]["ranks_agree"]
    # GET /runtime_control?only=fast reads the ranks' tiers part for it
    assert control.state(["fast"])["fast_tier"]["rows"] == 4


def test_the_engine_core_has_no_card_tier_resize_by_default():
    from vllm.v1.engine.core import EngineCore

    assert EngineCore._fast_tier is None


# -- the exchange charged once, refusals before the KV cache gives -------------

EPLB_BUSY = (
    "an async rearrangement of m is in flight at layer 0 of 48; ask after it completes"
)


def test_a_grow_takes_its_bytes_from_the_kv_cache_once(setup):
    """The KV shrink is the grow's bytes, and the ranks' commit is told so
    (they then do not ask the device for the same bytes again)."""
    engine, ranks, ctl = setup
    engine.drain_steps = 1
    ctl.start({"rows": 16})
    engine.drain()
    ctl.advance()
    assert ranks.rows == [16, 16]
    assert engine.vllm_config.cache_config.num_gpu_blocks == 4000 - GIB // PER_BLOCK
    commit = ranks.checks[-1]
    assert commit == {"rows": 16, "kv_released": GIB}
    # precheck first, memory aside
    assert ranks.checks[0] == {"rows": 16, "precheck": True}
    assert engine.resizes == []  # no second charge, no grow back


def test_a_shrink_gives_its_bytes_back_once(setup):
    engine, ranks, ctl = setup
    ctl.start({"rows": 4})
    assert engine.resizes == [4000 + (GIB // 2) // PER_BLOCK]
    ctl.advance()
    ctl.advance()
    # once, and as a restore: what the rows released, not a measured vote
    assert len(engine.resizes) == 1 and engine.restores == [True]


def test_a_grow_and_its_shrink_move_the_same_blocks(setup):
    """N rows up take N rows' blocks; N rows down give the same blocks back."""
    engine, ranks, ctl = setup
    engine.drain_steps = 1
    ctl.start({"delta": 8})
    engine.drain()
    ctl.advance()
    taken = 4000 - engine.vllm_config.cache_config.num_gpu_blocks
    ctl.start({"delta": -8})
    assert engine.vllm_config.cache_config.num_gpu_blocks == 4000
    assert taken == GIB // PER_BLOCK and engine.restores == [True]


def test_a_grow_waits_for_eplb_before_the_kv_cache_gives(setup):
    engine, ranks, ctl = setup
    engine.drain_steps = 1
    ranks.refusal = EPLB_BUSY
    reply = ctl.start({"rows": 16})
    assert reply["fast"]["state"] == "waiting"
    assert engine._kv_resize is None  # nothing taken while EPLB moves
    ctl.advance()
    assert ctl.resize.state == FastTierState.WAITING and engine._kv_resize is None
    ranks.refusal = None
    ctl.advance()  # EPLB done: now the KV shrink starts
    assert ctl.resize.state == FastTierState.DRAINING
    engine.drain()
    ctl.advance()
    assert ctl.resize.state == FastTierState.FINISHED and ranks.rows == [16, 16]


def test_a_grow_waits_for_eplb_that_starts_after_the_kv_shrink(setup):
    """The ranks refuse at the commit's check (raised, as _worker_op does):
    the resize waits for EPLB with the KV memory held, then commits."""
    engine, ranks, ctl = setup
    engine.drain_steps = 1
    ctl.start({"rows": 16})
    ranks.refusal = EPLB_BUSY
    engine.drain()
    ctl.advance()
    assert ctl.resize.state == FastTierState.COMMITTING
    assert "waiting for EPLB" in ctl.resize.message and ranks.ops == []
    ranks.refusal = None
    ctl.advance()
    assert ctl.resize.state == FastTierState.FINISHED and engine.resizes == []


def test_a_wait_past_the_deadline_restores_the_kv_cache_as_it_was(setup, monkeypatch):
    import vllm.v1.engine.fast_tier as ftmod

    engine, ranks, ctl = setup
    engine.drain_steps = 1
    clock = {"t": 1000.0}
    monkeypatch.setattr(ftmod.time, "monotonic", lambda: clock["t"])
    ctl.start({"rows": 16, "deadline_s": 30})
    ranks.refusal = EPLB_BUSY
    engine.drain()
    ctl.advance()
    clock["t"] += 31
    ctl.advance()
    record = ctl.resize.record(8)
    assert record["state"] == "failed" and "in flight" in record["message"]
    # the KV cache back to its size before, as a restore (no measured vote)
    assert engine.resizes == [4000] and engine.restores == [True]
    assert record["kv_blocks_after"] == 4000 and ranks.rows == [8, 8]


def test_a_refused_grow_with_nothing_taken_is_a_refusal(setup):
    """A grow without the KV cache (a profile's step) that the ranks refuse
    at their memory check: the verb is refused, nothing is mapped."""
    engine, ranks, ctl = setup
    ranks.refusal = None

    def refuse_commit(op, args):
        raise ControlRefused(
            "rank 0: the device has 0.40 GiB free and the rows need 1.00 GiB: "
            "the KV cache must give them back first"
        )

    ctl.control._worker_op = refuse_commit
    with pytest.raises(ControlRefused, match="must give them back first"):
        ctl.start({"rows": 16, "from_kv": False})
    assert ctl.resize.state == FastTierState.FAILED and not ctl.resize.state.active
    assert ranks.rows == [8, 8] and engine.resizes == [] and engine._kv_resize is None
    with pytest.raises(ControlRefused, match="must give them back first"):
        ctl.start({"rows": 4})  # a shrink refused the same way


def test_any_other_precheck_refusal_changes_nothing(setup):
    engine, ranks, ctl = setup
    ranks.refusal = "an EPLB move is prepared on layer 3; ask after it commits"
    reply = ctl.start({"rows": 16})  # prepared move: EPLB, waited for
    assert reply["fast"]["state"] == "waiting"
    ctl.resize.state = FastTierState.FAILED
    ranks.refusal = "the card tier is fixed in this engine"
    with pytest.raises(ControlRefused, match="fixed"):
        ctl.start({"rows": 16})
    assert engine._kv_resize is None and ranks.ops == []


def test_a_refusal_is_logged_in_one_line(caplog_vllm):
    """The engine core answers a refused runtime control call without an
    ERROR and a traceback; other failures keep them."""
    import logging

    from vllm.v1.engine.core import EngineCoreProc

    caplog = caplog_vllm
    out = []

    def call(exc):
        output = SimpleNamespace(result=None, failure_message=None)

        def raise_it():
            raise exc

        EngineCoreProc._invoke_utility_method(
            "runtime_control", raise_it, output, out.append
        )
        return output

    refused = call(ControlRefused("82 card rows per layer is outside 90..130"))
    assert refused.failure_message.endswith("outside 90..130")
    records = [r for r in caplog.records if "runtime_control" in r.getMessage()]
    assert records and all(r.levelno < logging.WARNING for r in records)
    assert all(r.exc_info is None for r in records)
    caplog.clear()
    call(RuntimeError("boom"))
    assert any(r.levelno == logging.ERROR and r.exc_info for r in caplog.records)


def test_the_reply_carries_torch_room_after_the_resize(setup):
    """A caller (a profile, an operator) sees each rank's torch room after
    the operation in the resize's record."""
    from vllm.model_executor.layers.fused_moe.xpu_fast_tier import torch_room

    engine, ranks, ctl = setup
    g = GIB
    ranks.torch = torch_room(7 * g, 8 * g, 1 * g, 9 * g)
    reply = ctl.start({"rows": 4})
    after = reply["fast"]["torch_after"]
    assert len(after) == 2 and after[0]["device_room"] == 2 * g
    assert after[0]["cap_room"] == g
    # without the KV cache's cap: no cap room
    assert torch_room(1, 2, None, None)["cap_room"] is None

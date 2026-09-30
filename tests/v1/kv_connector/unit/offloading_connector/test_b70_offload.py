# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70 patch 0014: env-gated offload fixes for hybrid (full + Mamba) models."""

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.test_scheduler import (
    _make_partial_tail_request,
    _make_partial_tail_scheduler,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_offload.base import LookupResult, get_offload_group_idx

FULL, MAMBA = 0, 1  # group ids of _make_mamba_hybrid_kv_cache_config


def _hybrid(monkeypatch, **env):
    for name in (
        "B70_OFFLOAD_TRACE",
        "B70_OFFLOAD_JUNCTION",
        "B70_OFFLOAD_EMPTY_ADVANCE_GUARD",
        "B70_OFFLOAD_GROUP_EVICT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    scheduler = _make_partial_tail_scheduler()
    request = _make_partial_tail_request(scheduler)  # 30 tokens, 16-token chunks
    request.shared_prefix_boundary = 0
    request.skip_reading_prefix_cache = False
    return scheduler, request


def _lookup_with(stored: set, full_hits: bool = True):
    def lookup(key, req_context):
        if get_offload_group_idx(key) == FULL:
            return LookupResult.HIT if full_hits else LookupResult.MISS
        return LookupResult.HIT if key in stored else LookupResult.MISS

    return lookup


@pytest.mark.parametrize("trace", ["0", "1", "2"])
def test_junction_set_when_mamba_group_zeroes_full_attention_hit(monkeypatch, trace):
    scheduler, request = _hybrid(
        monkeypatch, B70_OFFLOAD_JUNCTION="1", B70_OFFLOAD_TRACE=trace
    )
    stored: set = set()
    scheduler.manager.lookup.side_effect = _lookup_with(stored)

    # Full attention holds chunk 0 (tokens 0..16), the Mamba group holds no
    # state at 16: the whole request misses, as in the Flash-Next repro.
    tokens, _ = scheduler.get_num_new_matched_tokens(request, 0)
    assert tokens == 0
    # ... and the recompute is told to retain the Mamba state at 16.
    assert request.shared_prefix_boundary == 16

    # The recompute's aligned hand-off at the junction is stored ...
    scheduler.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    jobs = scheduler._build_aligned_boundary_store_jobs({"req": [(MAMBA, 77, 16)]})
    assert len(jobs) == 1
    [job] = scheduler._jobs.values()
    [key] = job.keys
    stored.add(key)

    # ... so the next revisit hits the shared prefix.
    scheduler._req_status["req"].transfer_jobs.clear()
    tokens, _ = scheduler.get_num_new_matched_tokens(request, 0)
    assert tokens == 16


def test_junction_off_leaves_request_untouched(monkeypatch):
    scheduler, request = _hybrid(monkeypatch)
    scheduler.manager.lookup.side_effect = _lookup_with(set())
    tokens, _ = scheduler.get_num_new_matched_tokens(request, 0)
    assert tokens == 0
    assert request.shared_prefix_boundary == 0


def test_junction_not_set_when_full_attention_misses(monkeypatch):
    scheduler, request = _hybrid(monkeypatch, B70_OFFLOAD_JUNCTION="1")
    scheduler.manager.lookup.side_effect = _lookup_with(set(), full_hits=False)
    tokens, _ = scheduler.get_num_new_matched_tokens(request, 0)
    assert tokens == 0
    assert request.shared_prefix_boundary == 0


def test_junction_never_lowers_an_existing_junction(monkeypatch):
    scheduler, request = _hybrid(monkeypatch, B70_OFFLOAD_JUNCTION="1")
    request.shared_prefix_boundary = 24  # e.g. from a GPU prefix-cache hit
    scheduler.manager.lookup.side_effect = _lookup_with(set())
    scheduler.get_num_new_matched_tokens(request, 0)
    assert request.shared_prefix_boundary == 24


def test_gpu_mamba_manager_retains_state_at_junction():
    """The GPU side: with sparse retention (interval 0) the Mamba manager keeps
    only the replay boundary -- plus the junction when one is set."""
    spec = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    common = dict(
        start_block=0,
        end_block=4,
        alignment_tokens=16,
        kv_cache_spec=spec,
        use_eagle=False,
        retention_interval=0,
    )
    without = MambaManager.reachable_block_mask(reachable_boundaries=[63], **common)
    with_junction = MambaManager.reachable_block_mask(
        reachable_boundaries=[63, 32], **common
    )
    assert without == [False, False, True, False]
    assert with_junction == [False, True, True, False]


def test_backstep_boundaries_sit_below_the_replay_boundary():
    from vllm.v1.core.kv_cache_coordinator import b70_backstep_boundaries

    # Flash-Next geometry: 832-token blocks, doc of 59,919 prompt tokens.
    assert b70_backstep_boundaries(59_919, 832, 2) == (59_072, 58_240)
    assert b70_backstep_boundaries(59_919, 832, 0) == ()
    assert b70_backstep_boundaries(900, 832, 3) == ()  # never 0 or below


@pytest.mark.parametrize("steps", [0, 2])
def test_coordinator_replay_boundaries_include_backstep(monkeypatch, steps):
    from types import SimpleNamespace

    from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator

    coord = object.__new__(UnitaryKVCacheCoordinator)
    coord.eagle_group_ids = ()
    coord.scheduler_block_size = 16
    coord.b70_gdn_backstep = steps
    request = SimpleNamespace(num_prompt_tokens=64)
    boundaries = coord.get_replay_boundaries(request)
    if steps == 0:
        assert boundaries == (63,)  # upstream behaviour
    else:
        assert boundaries == (16, 32, 63)
        spec = MambaSpec(
            block_size=16,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        )
        mask = MambaManager.reachable_block_mask(
            start_block=0,
            end_block=4,
            alignment_tokens=16,
            kv_cache_spec=spec,
            use_eagle=False,
            retention_interval=0,
            reachable_boundaries=boundaries,
        )
        assert mask == [True, True, True, False]


def _store_setup(monkeypatch, guard: bool, trace: str = "0"):
    from types import SimpleNamespace

    from vllm.v1.request import RequestStatus

    env = {"B70_OFFLOAD_TRACE": trace}
    if guard:
        env["B70_OFFLOAD_EMPTY_ADVANCE_GUARD"] = "1"
    scheduler, request = _hybrid(monkeypatch, **env)
    request.num_computed_tokens = 0
    request.status = RequestStatus.RUNNING
    req_status = scheduler._req_status["req"]
    req_status.group_states[FULL].block_ids[:] = [11]
    req_status.group_states[MAMBA].block_ids[:] = [99]
    req_status.update_offload_keys()
    output = SimpleNamespace(num_scheduled_tokens={"req": 16}, finished_req_ids=set())
    return scheduler, req_status, output


def _present_manager(scheduler, present: set, pending: set):
    from vllm.v1.kv_offload.base import PrepareStoreOutput

    from tests.v1.kv_connector.unit.offloading_connector.utils import (
        MockLoadStoreSpec,
    )

    def prepare_store(keys, req_context):
        keys = [k for k in keys if k not in present]
        return PrepareStoreOutput(
            keys_to_store=keys, store_spec=MockLoadStoreSpec(keys), evicted_keys=[]
        )

    def lookup(key, req_context):
        if key in pending:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT if key in present else LookupResult.MISS

    scheduler.manager.prepare_store.side_effect = prepare_store
    scheduler.manager.lookup.side_effect = lookup


@pytest.mark.parametrize("trace", ["0", "1"])
def test_guard_holds_index_while_skipped_key_is_write_pending(monkeypatch, trace):
    scheduler, req_status, output = _store_setup(monkeypatch, True, trace)
    key = req_status.group_states[FULL].offload_keys[0]
    present, pending = {key}, {key}  # stored by another request, not landed yet
    _present_manager(scheduler, present, pending)

    assert scheduler._build_store_jobs(output) == {}
    assert req_status.group_states[FULL].next_stored_chunk_idx == 0  # held

    pending.clear()  # the other request's store completed
    assert scheduler._build_store_jobs(output) == {}
    assert req_status.group_states[FULL].next_stored_chunk_idx == 1


def test_guard_off_keeps_upstream_advance(monkeypatch):
    scheduler, req_status, output = _store_setup(monkeypatch, False)
    key = req_status.group_states[FULL].offload_keys[0]
    _present_manager(scheduler, {key}, {key})
    assert scheduler._build_store_jobs(output) == {}
    # upstream (#56795): advances past the not-yet-ready key
    assert req_status.group_states[FULL].next_stored_chunk_idx == 1


def test_guard_reoffers_a_key_whose_pending_store_failed(monkeypatch):
    scheduler, req_status, output = _store_setup(monkeypatch, True)
    key = req_status.group_states[FULL].offload_keys[0]
    present, pending = {key}, {key}
    _present_manager(scheduler, present, pending)
    assert scheduler._build_store_jobs(output) == {}
    # the other store failed and its key was removed: this request stores it
    present.clear()
    pending.clear()
    jobs = scheduler._build_store_jobs(output)
    assert len(jobs) == 1
    [job] = scheduler._jobs.values()
    assert job.keys == {key}


# --- 0014f: cheap TRACE=1, GROUP_EVICT opt-in ---------------------------------


def _capture_b70(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
        b70_offload,
    )

    lines: dict[str, list[str]] = {"info": [], "debug": []}
    monkeypatch.setattr(
        b70_offload, "log", lambda fmt, *a: lines["info"].append(fmt % a)
    )
    monkeypatch.setattr(
        b70_offload, "log_debug", lambda fmt, *a: lines["debug"].append(fmt % a)
    )
    return lines


@pytest.mark.parametrize(
    "raw,level", [("", 0), ("0", 0), ("1", 1), ("true", 1), ("2", 2), ("x", 0)]
)
def test_trace_level_parsing(monkeypatch, raw, level):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
        b70_offload,
    )

    monkeypatch.setenv("B70_OFFLOAD_TRACE", raw)
    assert b70_offload.trace_level() == level
    assert b70_offload.trace_enabled() == (level >= 1)
    if level != 1:  # level 1 also depends on the logger's DEBUG state
        assert b70_offload.trace_per_request() == (level >= 2)


def test_trace1_logs_only_events_once_per_request(monkeypatch):
    lines = _capture_b70(monkeypatch)
    scheduler, request = _hybrid(
        monkeypatch, B70_OFFLOAD_JUNCTION="1", B70_OFFLOAD_TRACE="1"
    )
    if scheduler._b70_trace:
        pytest.skip("vLLM logger at DEBUG: TRACE=1 builds per-request lines")
    scheduler.manager.lookup.side_effect = _lookup_with(set())
    for _ in range(3):  # a waiting request is looked up again every step
        tokens, _ = scheduler.get_num_new_matched_tokens(request, 0)
        assert tokens == 0
    events = [ln for ln in lines["info"] if not ln.startswith("enabled")]
    assert any(ln.startswith("junction-set req=req") for ln in events), events
    assert sum(ln.startswith("zeroed_by req=req") for ln in events) == 1, events
    assert not any(ln.startswith("lookup req=") for ln in lines["info"])
    assert not lines["debug"]  # the per-request lines are not even built
    # no per-key result counting at TRACE=1
    assert scheduler._b70_counts is None


def test_trace2_logs_every_lookup(monkeypatch):
    lines = _capture_b70(monkeypatch)
    scheduler, request = _hybrid(monkeypatch, B70_OFFLOAD_TRACE="2")
    scheduler.manager.lookup.side_effect = _lookup_with(set())
    for _ in range(2):
        scheduler.get_num_new_matched_tokens(request, 0)
    assert sum(ln.startswith("lookup req=req") for ln in lines["info"]) == 2


def test_trace_off_builds_nothing(monkeypatch):
    lines = _capture_b70(monkeypatch)
    scheduler, request = _hybrid(monkeypatch)
    scheduler.manager.lookup.side_effect = _lookup_with(set())
    scheduler.get_num_new_matched_tokens(request, 0)
    assert lines == {"info": [], "debug": []}
    assert scheduler._b70_state_logger is None


@pytest.mark.parametrize("evict", ["0", "1"])
def test_group_evict_gates_scheduler_touch(monkeypatch, evict):
    scheduler, request = _hybrid(monkeypatch, B70_OFFLOAD_GROUP_EVICT=evict)
    scheduler.manager.lookup.side_effect = _lookup_with(set())
    scheduler.manager.touch.reset_mock()
    scheduler.get_num_new_matched_tokens(request, 0)
    # off = upstream v0.30.0 (per-lookup touch); on = #51787 (no touch)
    assert scheduler.manager.touch.called == (evict == "0")


@pytest.mark.parametrize("policy", ["lru", "arc"])
@pytest.mark.parametrize("evict", ["0", "1"])
def test_group_evict_selects_policy_class(monkeypatch, policy, evict):
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
    from vllm.v1.kv_offload.cpu.policies.arc_group_evict import (
        GroupEvictARCCachePolicy,
    )
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
    from vllm.v1.kv_offload.cpu.policies.lru_group_evict import (
        GroupEvictLRUCachePolicy,
    )

    monkeypatch.setenv("B70_OFFLOAD_GROUP_EVICT", evict)
    manager = CPUOffloadingManager(num_chunks=4, cache_policy=policy)
    expected = {
        ("lru", "0"): LRUCachePolicy,
        ("lru", "1"): GroupEvictLRUCachePolicy,
        ("arc", "0"): ARCCachePolicy,
        ("arc", "1"): GroupEvictARCCachePolicy,
    }[(policy, evict)]
    assert type(manager._policy) is expected


def test_group_evict_off_lru_evicts_in_upstream_order(monkeypatch):
    """Off: touch reorders an evictable chunk immediately (upstream); a
    finished request does not reorder anything."""
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    monkeypatch.delenv("B70_OFFLOAD_GROUP_EVICT", raising=False)
    manager = CPUOffloadingManager(num_chunks=2, cache_policy="lru")
    ctx = ReqContext(req_id="r")
    k1, k2, k3 = (make_offload_key(bytes([i]) * 8, 0) for i in (1, 2, 3))
    manager.prepare_store([k1, k2], ctx)
    manager.complete_store([k1, k2], ctx)
    manager.touch([k1], ctx)  # k1 becomes most recent
    manager.on_request_finished(ctx)  # no-op when off
    out = manager.prepare_store([k3], ctx)
    assert out is not None and out.evicted_keys == [k2]

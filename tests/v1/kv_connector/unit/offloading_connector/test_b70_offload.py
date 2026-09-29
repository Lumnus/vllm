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


@pytest.mark.parametrize("trace", ["0", "1"])
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

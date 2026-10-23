# SPDX-License-Identifier: Apache-2.0
"""B70-0057 (#57053 backport) + B70-0058: dynamic MTP depth by batch size on Model Runner V2 (CPU only).

1. The cudagraph descriptor builder gives every (batch size, K) of the
   schedule an exact, unpadded FULL uniform-decode graph when the capture list
   holds every multiple of 1+K, and it does not divide by zero for the
   drafter's own decode manager (decode_query_len 1).
2. The scheduler trims the drafter's num_spec_tokens drafts to the step's K,
   drops them at K = 0, and the B70-0026 uniform-drafts guard compares with K.
"""

from types import SimpleNamespace

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu import cudagraph_utils as cgu

SCHEDULE = [[1, 4, 3], [5, 8, 2], [9, 16, 1]]
MAX_SEQS = 16
K_MAX = 3


def c3_capture_sizes(schedule=SCHEDULE, max_seqs=MAX_SEQS, k_max=K_MAX):
    """The C3 engine-def list: every multiple of 1+K for each K in use, for
    every batch size 1..max_seqs, plus 1 and the prefill sizes."""
    dense = build_dynamic_sd_schedule_lookup(schedule, max_seqs, k_max)
    qs = sorted({k + 1 for k in dense[1:]})
    sizes = {1, 256, 512, 1024}
    for q in qs:
        sizes |= {b * q for b in range(1, max_seqs + 1)}
    return sorted(sizes), qs


def _manager(decode_query_len, sizes, schedule=SCHEDULE):
    m = object.__new__(cgu.CudaGraphManager)
    m.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: schedule is not None,
            num_speculative_tokens_per_batch_size=schedule,
        ),
        num_speculative_tokens=K_MAX,
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
    )
    m.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=max(sizes)
    )
    m.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    m.max_num_reqs = MAX_SEQS
    m.decode_query_len = decode_query_len
    m.varlen_decode = False
    m.lora_capture_cases = [0]
    m.ubatch_runner = None
    m._lora_dispatch_map, m._max_lora_case = {}, 0
    m._candidates, m._capture_descs = {}, {}
    m._init_candidates()
    m._graphs_captured = True
    return m


def test_lookup():
    dense = build_dynamic_sd_schedule_lookup(SCHEDULE, MAX_SEQS, K_MAX)
    assert dense[1:] == [3] * 4 + [2] * 4 + [1] * 8


def test_target_graphs_unpadded_for_every_k_and_batch():
    sizes, qs = c3_capture_sizes()
    assert qs == [2, 3, 4]
    m = _manager(K_MAX + 1, sizes)
    full = {(d.uniform_token_count, d.num_reqs)
            for d in m._capture_descs[CUDAGraphMode.FULL]}
    for q in qs:
        for b in range(1, MAX_SEQS + 1):
            assert (q, b) in full, (q, b)
            desc = m.dispatch(b, b * q, q, 0)
            assert desc.cg_mode == CUDAGraphMode.FULL
            assert desc.num_tokens == b * q and desc.num_reqs == b, (q, b, desc)


def test_serving_list_pads_k1_without_the_c3_sizes():
    # The A+B list (multiples of 4) with a K=1 tier: 3 requests (6 tokens)
    # would replay the 8-token graph, i.e. a padded FULL replay (skill 7a.14).
    sizes = [1, 2, 4, 8, 12, 16] + list(range(20, 65, 4)) + [256, 512, 1024]
    m = _manager(K_MAX + 1, sizes)
    desc = m.dispatch(3, 6, 2, 0)
    assert desc.cg_mode != CUDAGraphMode.FULL or desc.num_tokens != 6


def test_draft_decode_manager_no_zero_division():
    sizes, _ = c3_capture_sizes()
    m = _manager(1, sizes)  # drafter decode: one token per draft position
    full = {d.uniform_token_count for d in m._capture_descs[CUDAGraphMode.FULL]}
    assert full == {1}


def test_static_unchanged():
    sizes = [1, 2, 4, 8, 16]
    m = _manager(K_MAX + 1, sizes, schedule=None)
    full = {(d.uniform_token_count, d.num_tokens)
            for d in m._capture_descs[CUDAGraphMode.FULL]}
    assert {q for q, _ in full} == {4}


# ---- scheduler: update_draft_token_ids ----

class _Req:
    def __init__(self):
        self.spec_token_ids = []
        self.is_prefill_chunk = False

    def is_finished(self):
        return False


def _sched(dyn_k, lookup=True):
    from vllm.v1.core.sched.scheduler import Scheduler

    s = object.__new__(Scheduler)
    s.requests = {"a": _Req(), "b": _Req()}
    s.structured_output_manager = SimpleNamespace(validate_tokens=lambda r, t: list(t))
    s.num_spec_tokens = K_MAX
    s.dynamic_sd_lookup = (
        build_dynamic_sd_schedule_lookup(SCHEDULE, MAX_SEQS, K_MAX) if lookup else None
    )
    s._b70_dyn_k = dyn_k
    return s


@pytest.mark.parametrize("k", [3, 2, 1])
def test_trim_to_k(k):
    s = _sched(k)
    s.update_draft_token_ids(
        SimpleNamespace(req_ids=["a", "b"], draft_token_ids=[[5, 6, 7], [8, 9, 10]])
    )
    assert s.requests["a"].spec_token_ids == [5, 6, 7][:k]
    assert s.requests["b"].spec_token_ids == [8, 9, 10][:k]


def test_k0_drops_drafts():
    s = _sched(0)
    s.update_draft_token_ids(SimpleNamespace(req_ids=["a"], draft_token_ids=[[5, 6, 7]]))
    assert s.requests["a"].spec_token_ids == []


def test_uniform_guard_against_k(monkeypatch):
    # a host can have /run/b70-probes/uniform-drafts-off (guard off since 10-02, #600
    # in the kernels); force the guard on here to test its comparison.
    import vllm.v1.core.sched.scheduler as sch

    monkeypatch.setattr(sch, "_b70_uniform_drafts_on", lambda: True)
    s = _sched(2)
    s.structured_output_manager = SimpleNamespace(validate_tokens=lambda r, t: list(t)[:1])
    s.update_draft_token_ids(SimpleNamespace(req_ids=["a"], draft_token_ids=[[5, 6, 7]]))
    assert s.requests["a"].spec_token_ids == []  # 1 valid of K=2: plain decode


def test_static_path_unchanged():
    s = _sched(1, lookup=False)
    s.update_draft_token_ids(SimpleNamespace(req_ids=["a"], draft_token_ids=[[5, 6, 7]]))
    assert s.requests["a"].spec_token_ids == [5, 6, 7]


# ---- B70-0057: drafter decode graphs specialised per K ----

def test_specialize_spec_tokens():
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
        SpeculatorBatchDescriptor,
        SpeculatorCudaGraphManager,
    )

    m = object.__new__(SpeculatorCudaGraphManager)
    base = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=4, num_reqs=4, uniform_token_count=1
    )
    k2 = SpeculatorBatchDescriptor(**vars(base), num_speculative_tokens=2)
    m.graphs = {k2: object()}
    assert m.specialize_spec_tokens(base, 2) == k2  # captured for K=2: replay it
    k3 = m.specialize_spec_tokens(base, 3)  # not captured for K=3: eager, never a wrong graph
    assert k3.cg_mode == CUDAGraphMode.NONE

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU check of VLLM_XPU_MOE_TOPK_ROUTE (run by test_xpu_moe_entry.py in a
subprocess under the Triton interpreter).

Today: FusedTopKRouter's softmax top-k (a stand-in of topk_softmax's
TopKGating: fp32 softmax, greatest first, lower index on ties, renormalized,
padding rows -1) followed by vLLM's own EPLB mapping kernel
(_eplb_map_and_record_i32_kernel, interpreted) with and without the
sentinel. Switch on: BaseRouter._select_experts calls topk_route, whose
op is a stand-in of topk_softmax_route written from its csrc epilogue
(RouteArgs / route_map_and_record in csrc/moe/topk.cpp) over the same
top-k. Weights, ids, the sentinel mark and the expert-load record must be
equal, so the csrc epilogue's rule is the mapping kernel's."""

import os
from types import SimpleNamespace

import torch


def topk_ref(logits, k, renorm, is_padding=None):
    p = logits.float().softmax(-1)
    m = logits.shape[0]
    w = torch.zeros((m, k), dtype=torch.float32)
    ids = torch.zeros((m, k), dtype=torch.int32)
    for t in range(m):
        row = p[t].clone()
        for j in range(k):
            best = int(torch.argmax(row))  # first index of the max: lower wins
            w[t, j] = p[t, best]
            ids[t, j] = best
            row[best] = -10000.0
        if renorm:
            w[t] = w[t] / w[t].sum()
        if is_padding is not None and bool(is_padding[t]):
            w[t] = 0
            ids[t] = -1
    return w, ids


def route_ref(ids, l2p, replicas, load, record, num_unpadded, sentinel, write_sentinel):
    """csrc/moe/topk.cpp route_map_and_record, per written id."""
    out = ids.clone()
    m, k = ids.shape
    for row in range(m):
        for j in range(k):
            logical = int(ids[row, j])
            phys = -1
            if 0 <= logical < replicas.shape[0]:
                rep = int(replicas[logical])
                hashed = (row * 2654435769) & 0xFFFFFFFF
                if rep > 0:
                    phys = int(l2p[logical, hashed % rep])
            rec = bool(record) and 0 <= phys < load.shape[0]
            if num_unpadded is not None:
                rec = rec and row < int(num_unpadded)
            if rec:
                load[phys] += 1
            if write_sentinel and phys < 0:
                phys = sentinel
            out[row, j] = phys
    return out


def main():
    os.environ.pop("VLLM_XPU_MOE_TOPK_ROUTE", None)
    # base_router defines the Triton EPLB mapping on CUDA-alike / XPU only;
    # import it as there, so today's path maps (interpreted).
    from vllm.platforms import current_platform

    was = current_platform.is_cuda_alike
    current_platform.is_cuda_alike = lambda: True
    try:
        import vllm.model_executor.layers.fused_moe.router.base_router as br
    finally:
        current_platform.is_cuda_alike = was
    assert hasattr(br, "_eplb_map_and_record_triton"), "Triton mapping absent"
    # The interpreter's atomic_rmw takes only a contiguous bool mask; the
    # record mask reaches it as a broadcast view (as in
    # test_xpu_ep_sentinel.py).
    import numpy as np
    from triton.runtime import interpreter as it

    rmw = it.InterpreterBuilder.create_atomic_rmw

    def rmw_bool_mask(self, op, ptr, val, mask, sem, scope):
        mask = it.TensorHandle(np.ascontiguousarray(mask.data, dtype=bool), mask.dtype)
        return rmw(self, op, ptr, val, mask, sem, scope)

    it.InterpreterBuilder.create_atomic_rmw = rmw_bool_mask
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.fused_moe import xpu_moe_entry
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )

    xpu_moe_entry._TEST_NATIVE = True
    gen = torch.Generator().manual_seed(3)

    def topk_softmax(w, ids, tei, logits, renorm, bias=None, is_padding=None):
        rw, rids = topk_ref(logits, w.shape[1], renorm, is_padding)
        w.copy_(rw)
        ids.copy_(rids.to(ids.dtype))
        tei.zero_()

    ops.topk_softmax = topk_softmax

    def route_op(
        w,
        ids,
        tei,
        logits,
        renorm,
        is_padding,
        l2p,
        replicas,
        load,
        record,
        num_unpadded,
        sentinel,
        write_sentinel,
    ):
        rw, rids = topk_ref(logits, w.shape[1], renorm, is_padding)
        w.copy_(rw)
        ids.copy_(
            route_ref(
                rids,
                l2p,
                replicas,
                load,
                record,
                num_unpadded,
                sentinel,
                write_sentinel,
            )
        )
        tei.zero_()

    xpu_moe_entry._route_op = lambda: route_op

    n_cases = 0
    for n_logical, redundant, topk in ((16, 0, 4), (16, 8, 4), (64, 16, 10)):
        n_phys = n_logical + redundant
        # A placement with up to 3 replicas per logical expert, shuffled.
        phys_to_log = torch.cat(
            [
                torch.arange(n_logical),
                torch.randint(0, n_logical, (redundant,), generator=gen),
            ]
        )
        phys_to_log = phys_to_log[torch.randperm(n_phys, generator=gen)]
        slots = 1 + redundant
        l2p = torch.full((n_logical, slots), -1, dtype=torch.long)
        rc = torch.zeros((n_logical,), dtype=torch.long)
        for p, lg in enumerate(phys_to_log.tolist()):
            l2p[lg, rc[lg]] = p
            rc[lg] += 1
        for m in (1, 2, 4, 7, 8, 13, 16, 33):
            for record in (True, False):
                for unpadded in (None, max(1, m - 2)):
                    for sentinel in (None, n_phys - 1, 0):
                        logits = torch.randn((m, n_logical), generator=gen)
                        logits[0, 1] = logits[0, 0]  # a tie
                        logits = logits.to(torch.bfloat16)
                        res = []
                        for on in (False, True):
                            if on:
                                os.environ["VLLM_XPU_MOE_TOPK_ROUTE"] = "1"
                            else:
                                os.environ.pop("VLLM_XPU_MOE_TOPK_ROUTE", None)
                            load = torch.zeros((n_phys,), dtype=torch.int32)
                            st = SimpleNamespace(
                                expert_load_view=load,
                                logical_to_physical_map=l2p,
                                logical_replica_count=rc,
                                should_record_tensor=torch.tensor(record),
                                num_unpadded_tokens_tensors=[
                                    None
                                    if unpadded is None
                                    else torch.tensor(unpadded, dtype=torch.int32)
                                ],
                                residency_mask=None,
                                residency_stats=None,
                            )
                            r = FusedTopKRouter(
                                topk, n_logical, "softmax", True, eplb_state=st
                            )
                            r._xpu_sentinel = sentinel
                            r._xpu_topk_route_why = None if on else "off"
                            w, ids = r._select_experts(
                                torch.empty((m, 8)), logits.clone(), None
                            )
                            res.append(
                                (
                                    w,
                                    ids,
                                    load.clone(),
                                    getattr(ids, "xpu_ids_sentinel", None),
                                )
                            )
                        (w0, i0, l0, s0), (w1, i1, l1, s1) = res
                        key = (n_logical, redundant, m, record, unpadded, sentinel)
                        assert torch.equal(
                            w0.view(torch.int32), w1.view(torch.int32)
                        ), key
                        assert torch.equal(i0.to(torch.int32), i1), key
                        assert torch.equal(l0, l1), key
                        assert s0 == s1, key
                        n_cases += 1
    print("XPU-TOPK-ROUTE-OK", n_cases)


if __name__ == "__main__":
    main()

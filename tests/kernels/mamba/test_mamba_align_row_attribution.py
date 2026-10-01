# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row attribution of the mamba "align" copy kernels under a permuted idx_mapping.

B70-0022 regression test (port of the tests in vllm#55506, plus a V2 mixed-batch
case). The V2 runner hands the align context the SOURCE per-request-slot block
tables (row = request-state slot). The grid runs in batch order and
``idx_mapping`` resolves batch row -> slot, so every table read must use the
slot, never the batch row. With an identity mapping the two are the same, which
is why the existing equivalence test cannot see this axis.

Runs on CUDA, or on CPU through the Triton interpreter (TRITON_INTERPRET=1).
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.kernels.mamba.test_precopy_mamba_align import (
    MAX_COLS,
    NUM_LAYERS,
    _build_meta,
    _build_state,
    _reference,
)
from vllm.platforms import current_platform
from vllm.v1.worker.mamba_utils import (
    get_aligned_state_indices_multi_group_kernel,
    postprocess_mamba_fused_kernel,
    precopy_mamba_align_fused_kernel,
)

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() or _INTERPRET),
    reason="needs CUDA or the Triton CPU interpreter (TRITON_INTERPRET=1)",
)


def _device() -> torch.device:
    return torch.device("cuda" if current_platform.is_cuda() else "cpu")


def _sync() -> None:
    if current_platform.is_cuda():
        torch.cuda.synchronize()


def _source_tables(num_slots: int, device):
    """[num_slots, MAX_COLS]; every slot owns block ids no other slot references."""
    bt = torch.empty(num_slots, MAX_COLS, dtype=torch.int32, device=device)
    for r in range(num_slots):
        bt[r] = torch.arange(
            1 + r * MAX_COLS, 1 + (r + 1) * MAX_COLS, dtype=torch.int32, device=device
        )
    return bt, num_slots * MAX_COLS + 1


def _assert_pools(convs, ssms, conv_ref, ssm_ref, bt, slots):
    """Whole-pool equality, then a per-slot message naming the first wrong slot."""
    for slot in slots:
        blocks = bt[slot].long().cpu()
        for layer in range(NUM_LAYERS):
            assert torch.equal(convs[layer].cpu()[blocks], conv_ref[layer][blocks]), (
                f"slot {slot}: conv state of layer {layer} wrong"
            )
            assert torch.equal(ssms[layer].cpu()[blocks], ssm_ref[layer][blocks]), (
                f"slot {slot}: ssm state of layer {layer} wrong"
            )
    for layer in range(NUM_LAYERS):
        assert torch.equal(convs[layer].cpu(), conv_ref[layer])
        assert torch.equal(ssms[layer].cpu(), ssm_ref[layer])


def _cpu_states(convs, ssms):
    return [c.cpu() for c in convs], [s.cpu() for s in ssms]


# --------------------------------------------------------------------------
# 1. vllm#55506 tests, ported (4 slots, batch order (2, 0, 3, 1)).
# --------------------------------------------------------------------------
_PERM = (2, 0, 3, 1)


@pytest.mark.parametrize("conv_state_dim_first", [False, True])
def test_precopy_indexes_block_tables_by_req_slot(conv_state_dim_first):
    device = _device()
    torch.manual_seed(0)
    n = len(_PERM)
    bt, num_blocks = _source_tables(n, device)
    # Per SLOT: 0 copies col1->col0; 1 same block; 2 fresh; 3 col2->col0 bias 1.
    src_col = torch.tensor([1, 1, -1, 2], dtype=torch.int32, device=device)
    dst_col = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device=device)
    bias = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=device)

    convs, ssms = _build_state(num_blocks, device, conv_state_dim_first)
    c0, s0 = _cpu_states(convs, ssms)
    conv_ref, ssm_ref = _reference(
        c0, s0, bt.cpu(), src_col.cpu(), dst_col.cpu(), bias.cpu(), n,
        conv_state_dim_first,
    )
    meta = _build_meta(convs, ssms, device, conv_state_dim_first)
    bt_ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64, device=device)
    idx_mapping = torch.tensor(_PERM, dtype=torch.int32, device=device)
    precopy_mamba_align_fused_kernel[(n, NUM_LAYERS * 2, 1)](
        dst_col, src_col, bias, bt_ptrs, bt.stride(0), *meta, idx_mapping, n,
        COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=conv_state_dim_first,
        HAS_IDX_MAPPING=True, TEMPORAL_TILES=1,
    )
    _sync()
    _assert_pools(convs, ssms, conv_ref, ssm_ref, bt, range(n))


@pytest.mark.parametrize("conv_state_dim_first", [False, True])
def test_postprocess_indexes_block_tables_by_req_slot(conv_state_dim_first):
    device = _device()
    torch.manual_seed(0)
    n = len(_PERM)
    block_size = 8
    bt, num_blocks = _source_tables(n, device)
    num_accepted = torch.tensor([1, 1, 1, 2], dtype=torch.int32, device=device)
    state_idx = torch.tensor([1, 0, 1, 2], dtype=torch.int32, device=device)
    num_scheduled = torch.ones(n, dtype=torch.int32, device=device)
    num_computed = torch.tensor([8, 8, 5, 7], dtype=torch.int32, device=device)
    num_draft = torch.ones(n, dtype=torch.int32, device=device)
    num_accepted_out = num_accepted.clone()

    convs, ssms = _build_state(num_blocks, device, conv_state_dim_first)
    c0, s0 = _cpu_states(convs, ssms)
    conv_ref, ssm_ref = _reference(
        c0, s0, bt.cpu(),
        torch.tensor([1, 0, -1, 2], dtype=torch.int32),
        torch.zeros(n, dtype=torch.int32),
        torch.tensor([0, 0, 0, 1], dtype=torch.int32),
        n, conv_state_dim_first,
    )
    meta = _build_meta(convs, ssms, device, conv_state_dim_first)
    bt_ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64, device=device)
    idx_mapping = torch.tensor(_PERM, dtype=torch.int32, device=device)
    postprocess_mamba_fused_kernel[(n, NUM_LAYERS * 2, 1)](
        num_accepted, state_idx, num_scheduled, num_computed, num_draft,
        bt_ptrs, bt.stride(0), *meta, num_accepted_out, idx_mapping, n,
        block_size=block_size, COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=conv_state_dim_first, HAS_IDX_MAPPING=True,
        PRECOMPUTED_NEW_COMPUTED=False, TEMPORAL_TILES=1,
    )
    _sync()
    _assert_pools(convs, ssms, conv_ref, ssm_ref, bt, range(n))


# --------------------------------------------------------------------------
# 2. B70: V2 postprocess-align (PRECOMPUTED_NEW_COMPUTED), mixed batch.
#    8 request-state slots, 5 real requests scheduled in batch order
#    (6, 1, 4, 0, 3): verify rows that cross a block boundary, a verify row that
#    stays in its block, a chunked-prefill row that ends block-aligned, and a
#    chunked-prefill row whose state already sits at the destination (resets
#    num_accepted).
#    Slots 2, 5, 7 are not scheduled; their decision values would copy if a
#    kernel ever read them, so touching them is detected too.
# --------------------------------------------------------------------------
_BLOCK = 8
_BATCH = (6, 1, 4, 0, 3)


def _postprocess_reference_cols(num_accepted, state_idx, new_computed, slots):
    """Mirror of postprocess_mamba_fused_kernel's decision (PRECOMPUTED path)."""
    src = torch.full((8,), -1, dtype=torch.int32)
    dst = torch.zeros(8, dtype=torch.int32)
    bias = torch.zeros(8, dtype=torch.int32)
    acc_out = num_accepted.clone()
    for s in slots:
        acc, sidx, nc = int(num_accepted[s]), int(state_idx[s]), int(new_computed[s])
        running = nc - acc + 1
        aligned = (nc // _BLOCK) * _BLOCK
        if aligned < running:
            continue
        b = aligned - running
        d = aligned // _BLOCK - 1
        if sidx == d:
            acc_out[s] = 1
            if b == 0:
                continue
        src[s], dst[s], bias[s] = sidx, d, b
    return src, dst, bias, acc_out


@pytest.mark.parametrize("conv_state_dim_first", [False, True])
def test_v2_postprocess_align_mixed_batch_by_req_slot(conv_state_dim_first):
    device = _device()
    torch.manual_seed(1)
    bt, num_blocks = _source_tables(8, device)
    # Per SLOT (index = request-state slot).
    #   slot 6: verify, accepted 3 -> computed 17 (running 15): aligned 16, bias 1,
    #           dst col 1, src col 0                                   (copy)
    #   slot 1: verify, accepted 2 -> computed 24 (running 23): aligned 24, bias 1,
    #           dst col 2, src col 1                                   (copy)
    #   slot 4: verify, accepted 2 -> computed 13 (running 12): aligned 8 < 12 (skip)
    #   slot 0: chunked prefill, 0 sampled (acc 1) ends at 16: dst col 1, src 0 (copy)
    #   slot 3: chunked prefill ends at 24, state already at col 2         (skip)
    #   slots 2, 5, 7: not scheduled, values that WOULD copy if read.
    num_accepted = torch.tensor([1, 2, 1, 1, 2, 1, 3, 1], dtype=torch.int32)
    state_idx = torch.tensor([0, 1, 0, 2, 1, 0, 0, 0], dtype=torch.int32)
    new_computed = torch.tensor([16, 24, 16, 24, 13, 16, 17, 32], dtype=torch.int32)
    slots = list(_BATCH)
    src, dst, bias, acc_ref = _postprocess_reference_cols(
        num_accepted, state_idx, new_computed, slots
    )
    assert int(src[6]) == 0 and int(src[0]) == 0 and int(src[1]) == 1  # shape check

    convs, ssms = _build_state(num_blocks, device, conv_state_dim_first)
    c0, s0 = _cpu_states(convs, ssms)
    conv_ref, ssm_ref = _reference(
        c0, s0, bt.cpu(), src, dst, bias, 8, conv_state_dim_first
    )
    meta = _build_meta(convs, ssms, device, conv_state_dim_first)
    bt_ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64, device=device)
    idx_mapping = torch.tensor(_BATCH, dtype=torch.int32, device=device)
    acc = num_accepted.to(device)
    snapshot = acc.clone()
    n = len(_BATCH)
    postprocess_mamba_fused_kernel[(n, NUM_LAYERS * 2, 1)](
        snapshot, state_idx.to(device), None, new_computed.to(device), None,
        bt_ptrs, bt.stride(0), *meta, acc, idx_mapping, n,
        block_size=_BLOCK, COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=conv_state_dim_first, HAS_IDX_MAPPING=True,
        PRECOMPUTED_NEW_COMPUTED=True, TEMPORAL_TILES=1,
    )
    _sync()
    _assert_pools(convs, ssms, conv_ref, ssm_ref, bt, range(8))
    assert torch.equal(acc.cpu(), acc_ref)


@pytest.mark.parametrize("conv_state_dim_first", [False, True])
def test_v2_precopy_mixed_batch_by_req_slot(conv_state_dim_first):
    """Same batch shape for the pre-forward boundary copy (preprocess_state)."""
    device = _device()
    torch.manual_seed(2)
    bt, num_blocks = _source_tables(8, device)
    # Per SLOT: src_col (-1 fresh), dst_col, bias. Unscheduled slots 2, 5, 7
    # carry copies that must not happen.
    src_col = torch.tensor([0, 2, 1, 1, -1, 0, 0, 3], dtype=torch.int32)
    dst_col = torch.tensor([1, 3, 2, 1, 0, 1, 1, 4], dtype=torch.int32)
    bias = torch.tensor([0, 2, 0, 0, 0, 1, 1, 0], dtype=torch.int32)
    scheduled = torch.zeros(8, dtype=torch.bool)
    scheduled[list(_BATCH)] = True
    ref_src = torch.where(scheduled, src_col, torch.full_like(src_col, -1))

    convs, ssms = _build_state(num_blocks, device, conv_state_dim_first)
    c0, s0 = _cpu_states(convs, ssms)
    conv_ref, ssm_ref = _reference(
        c0, s0, bt.cpu(), ref_src, dst_col, bias, 8, conv_state_dim_first
    )
    meta = _build_meta(convs, ssms, device, conv_state_dim_first)
    bt_ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64, device=device)
    idx_mapping = torch.tensor(_BATCH, dtype=torch.int32, device=device)
    n = len(_BATCH)
    precopy_mamba_align_fused_kernel[(n, NUM_LAYERS * 2, 1)](
        dst_col.to(device), src_col.to(device), bias.to(device), bt_ptrs,
        bt.stride(0), *meta, idx_mapping, n,
        COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=conv_state_dim_first,
        HAS_IDX_MAPPING=True, TEMPORAL_TILES=1,
    )
    _sync()
    _assert_pools(convs, ssms, conv_ref, ssm_ref, bt, range(8))


# --------------------------------------------------------------------------
# 3. get_aligned_state_indices (Kimi-K3 KDA path only): permuted rows + padding.
# --------------------------------------------------------------------------
def _aligned_indices(bt, seq_lens, idx_mapping, num_rows, num_state_slots=2):
    device = bt.device
    out = torch.full(
        (1, num_rows, num_state_slots), -7, dtype=torch.int32, device=device
    )
    get_aligned_state_indices_multi_group_kernel[(1,)](
        torch.tensor([bt.data_ptr()], dtype=torch.int64, device=device),
        seq_lens, out, idx_mapping, idx_mapping.numel(), bt.stride(0),
        seq_lens.stride(0), out.stride(0), out.stride(1), out.stride(2),
        num_rows, CACHE_BLOCK_SIZE=_BLOCK, NUM_GROUPS=1, BLOCK_GROUPS=1,
        NUM_STATE_SLOTS=num_state_slots, BLOCK_STATE_SLOTS=num_state_slots,
        BLOCK_ROWS=32, HAS_IDX_MAPPING=True,
    )
    _sync()
    return out.cpu()


def test_aligned_state_indices_resolve_rows_through_idx_mapping():
    device = _device()
    bt, _ = _source_tables(8, device)
    seq_lens = torch.tensor([17, 3, 30, 9, 8, 0, 0, 0], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor(_BATCH, dtype=torch.int32, device=device)
    out = _aligned_indices(bt, seq_lens, idx_mapping, num_rows=8)
    btc = bt.cpu()
    for b, slot in enumerate(_BATCH):
        first = max((int(seq_lens[b]) - 1) // _BLOCK, 0)
        assert out[0, b].tolist() == btc[slot, first : first + 2].tolist(), b
    # Padded rows 5..7 must not read past the 5-entry mapping (they used to).
    assert all(v >= 0 for v in out[0, 5:].flatten().tolist())


@pytest.mark.xfail(
    strict=True,
    reason="vllm#55506 @5f71d6f resolves padded rows to request slot 0's live "
    "blocks; the pre-PR gathered path read the zeroed padding row (null block 0)",
)
def test_aligned_state_indices_padded_rows_hit_null_block():
    device = _device()
    bt, _ = _source_tables(8, device)
    seq_lens = torch.tensor([17, 3, 30, 9, 8, 0, 0, 0], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor(_BATCH, dtype=torch.int32, device=device)
    out = _aligned_indices(bt, seq_lens, idx_mapping, num_rows=8)
    assert out[0, 5:].flatten().tolist() == [0] * 6

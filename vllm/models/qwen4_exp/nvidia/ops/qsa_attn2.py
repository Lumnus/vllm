# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-QSA-K: Xe2-shaped sparse paged GQA attention for Qwen4Exp QSA (Intel XPU).

Selected with ``B70_QSA_ATTN_IMPL=attn2`` (default ``splitk`` = the original
``_qsa_sparse_paged_gqa_splitk_kernel``). Same numerical contract as the
original: BF16 (or fp8-e4m3) K/V, BF16 Q, FP32 scores / softmax /
accumulation, FP32 partials merged by log-sum-exp, BF16 rounding of the
attention output before the FP32 sigmoid gate.

Why a second kernel (measured in runs/2026-W43/2026-10-25-qsa-triton-xpu, C1
and K): the original holds a whole ``[BLOCK_M, HEAD_DIM]`` FP32 accumulator
plus gathered K and V tiles in one program. At head_dim 256 that alone is the
whole 256-GRF file of a Xe2 hardware thread, so every config spills 11-46 KB
per thread. Adding warps does not help: Triton's Intel backend lays a chained
(Q.K -> P.V) dot's warps along M, and M is only the 6 (padded 8) heads of one
KV group, so extra warps replicate the same work. The SYCL kernels at the bar
(analytics-zoo vllm-xpu-kernels ``token_split_attention_v3``) instead keep
each subgroup's live state small and get their parallelism from many short
token partials merged afterwards.

attn2 does the same within Triton:

1. **One program = one row x one KV group x TILES_PER_PROG tiles of BLOCK_N
   selected tokens, one subgroup (num_warps 1).** All 6 heads of the group are
   the dot's M (padded to 8), so each gathered K/V row serves every head.
2. **Each tile is finished in place**: Q.K over head_dim chunks (a runtime loop,
   so only one Q chunk and one K chunk are ever live), an exact softmax over the
   tile's BLOCK_N scores, then P.V chunk by chunk with each ``[BM, DC]`` output
   chunk stored as soon as it is computed. No full-head_dim accumulator exists.
3. **Each tile writes an FP32 partial** (normalised output + log2-sum-exp, the
   layout the original split-K uses), and ``_qsa_attn2_merge_kernel`` combines a
   row's tiles with an online log-sum-exp merge, rounds to BF16 and applies the
   gate. The merge reads the row's valid count, so tiles past it are never read.
4. **Rows go in chunks** (``row_chunk``) so the partial workspace stays bounded
   (prefill default: 256 rows x 6 heads x 33 tiles x 1 KB = 51 MB at BLOCK_N 64;
   decode 64 rows x 6 x 65 x 1 KB = 25 MB at BLOCK_N 32). It is a transient
   torch.empty outside the engine's profiled peak, like the original's
   partials.
5. Every dot is M 8, N = BLOCK_N or DC >= 16, K = DC or BLOCK_N >= 16: DPAS
   eligible and legal under Triton 3.8.0+xpu's N/K >= 16 minimum.

Launch choices go through ``B70_QSA_TUNE`` keys ``sparse_attn2_decode`` and
``sparse_attn2_prefill`` (fields in qsa_xpu_tuning); the merge reads ``merge``.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from . import qsa_xpu_tuning as xpu_tune

logger = init_logger(__name__)

ATTN_IMPL_ENV = "B70_QSA_ATTN_IMPL"
_IMPLS = ("splitk", "attn2")


def attn_impl() -> str:
    """Selected sparse attention implementation (read per call; host-side only)."""
    impl = os.environ.get(ATTN_IMPL_ENV, "splitk").strip() or "splitk"
    if impl not in _IMPLS:
        raise ValueError(f"{ATTN_IMPL_ENV}={impl!r}; expected one of {_IMPLS}")
    logger.info_once("B70-QSA: sparse attention implementation = %s", impl)
    return impl


@triton.jit
def _attn2_dequant(tile):
    # Same decode as qsa._maybe_dequant: e4m3 bytes -> fp16 of x * 2^-8 (the
    # host folds 2^8 into the scales); e4m3 view -> fp16; anything else as is.
    if tile.dtype == tl.uint8:
        code = tile.to(tl.uint16)
        bits = ((code & 0x80) << 8) | ((code & 0x7F) << 7)
        return bits.to(tl.float16, bitcast=True)
    elif tile.dtype == tl.float8e4nv:
        return tile.to(tl.float16)
    else:
        return tile


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_attn2_tile_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    softmax_scale,
    output_scale,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_DCHUNK: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
) -> None:
    DC: tl.constexpr = HEAD_DIM // NUM_DCHUNK
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    prog = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    request_ok = (request >= 0) & (request < num_requests)
    # Trailing count column of the packed selection buffer (see qsa.py).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    num_tiles = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)

    head_offsets = tl.arange(0, BLOCK_M)
    head_mask = head_offsets < GROUP_SIZE
    dc_offsets = tl.arange(0, DC)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    q_rows = (
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
    )
    score_scale = softmax_scale * 1.4426950408889634

    for t in range(TILES_PER_PROG):
        tile = prog * TILES_PER_PROG + t
        if tile < num_tiles:
            columns = tile * BLOCK_N + column_offsets
            logical_token = tl.load(
                indices_ptr + row * stride_indices_row + columns,
                mask=columns < TOPK,
                other=-1,
            )
            safe_token = tl.maximum(logical_token, 0)
            logical_page = safe_token // PAGE_SIZE
            page_offset = safe_token % PAGE_SIZE
            valid = (
                request_ok & (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
            )
            physical_page = tl.load(
                block_table_ptr
                + safe_request * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
            safe_page = tl.maximum(physical_page, 0).to(tl.int64)
            # One int64 base per token; elements add a contiguous dim offset.
            k_base = (
                k_cache_ptr
                + safe_page * stride_k_block
                + page_offset * stride_k_token
                + kv_head * stride_k_head
            )
            v_base = (
                v_cache_ptr
                + safe_page * stride_v_block
                + page_offset * stride_v_token
                + kv_head * stride_v_head
            )

            # Q.K, head_dim chunk by chunk (runtime loop: one chunk live).
            scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for c in range(NUM_DCHUNK):
                q_chunk = tl.load(
                    q_rows + c * DC + dc_offsets[None, :],
                    mask=head_mask[:, None],
                    other=0.0,
                )
                k_chunk = tl.load(
                    k_base[None, :] + c * DC + dc_offsets[:, None],
                    mask=valid[None, :],
                    other=0.0,
                )
                scores = tl.dot(
                    q_chunk, _attn2_dequant(k_chunk).to(q_chunk.dtype), acc=scores
                )

            # Exact softmax over this tile.
            scores = tl.where(valid[None, :], scores * score_scale, -1.0e20)
            tile_max = tl.max(scores, axis=1)
            probabilities = tl.where(
                valid[None, :], tl.math.exp2(scores - tile_max[:, None]), 0.0
            )
            tile_sum = tl.sum(probabilities, axis=1)
            has_values = tile_sum > 0
            inv_sum = tl.where(
                has_values, output_scale / tl.maximum(tile_sum, 1.0e-20), 0.0
            )
            partial_row = (tile * num_rows + row).to(tl.int64)
            tl.store(
                partial_lse_ptr
                + partial_row * NUM_QUERY_HEADS
                + first_head
                + head_offsets,
                tl.where(
                    has_values,
                    tile_max + tl.math.log2(tl.maximum(tile_sum, 1.0e-20)),
                    -float("inf"),
                ),
                mask=head_mask,
            )
            part_rows = (
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
                * HEAD_DIM
            )

            # P.V, chunk by chunk; each [BM, DC] output chunk is stored at once.
            for c in range(NUM_DCHUNK):
                v_chunk = _attn2_dequant(
                    tl.load(
                        v_base[:, None] + c * DC + dc_offsets[None, :],
                        mask=valid[:, None],
                        other=0.0,
                    )
                )
                out_chunk = tl.dot(probabilities.to(v_chunk.dtype), v_chunk)
                tl.store(
                    part_rows + c * DC + dc_offsets[None, :],
                    out_chunk * inv_sum[:, None],
                    mask=head_mask[:, None],
                )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_attn2_merge_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    indices_ptr,
    output_ptr,
    output_gate_ptr,
    stride_indices_row,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
) -> None:
    """Online log-sum-exp merge of one (row, head)'s tile partials + gate."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    num_tiles = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)
    dim_offsets = tl.arange(0, HEAD_DIM)
    running_max = -float("inf")
    denominator = 0.0
    merged = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for tile in range(num_tiles):
        partial_row = (tile * num_rows + row).to(tl.int64)
        lse = tl.load(partial_lse_ptr + partial_row * NUM_QUERY_HEADS + head)
        if lse > -float("inf"):
            next_max = tl.maximum(running_max, lse)
            alpha = tl.math.exp2(running_max - next_max)
            weight = tl.math.exp2(lse - next_max)
            part = tl.load(
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + head) * HEAD_DIM
                + dim_offsets
            )
            merged = merged * alpha + part * weight
            denominator = denominator * alpha + weight
            running_max = next_max
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    # Original contract: BF16 attention output, then the FP32 sigmoid gate.
    merged = merged.to(output_ptr.dtype.element_ty).to(tl.float32)
    gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged * tl.sigmoid(gate),
    )


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
# Defaults: the fastest configs measured on a B70 (Triton 3.8.0+xpu, IGC 2.41.5,
# compute-runtime 26.35; run 2026-W43/2026-10-25-qsa-triton-xpu, B2a): BLOCK_N 32 x
# 8 head_dim chunks at 256 GRF, zero spill, for both regions. Decode 8 rows
# 54-56 us (rows 32: 84-202, rows 64: 191-372, 4K-131K); prefill 4,096 rows
# 9.3 ms at 32K, 12.6 ms at 131K. At 128 GRF the same tile spills 192 B and
# runs ~17 % slower, so grf_mode "256" is part of the default.
_DEFAULTS = {
    "sparse_attn2_decode": {
        "block_n": 32,
        "num_dchunk": 8,
        "tiles_per_prog": 1,
        "row_chunk": 4096,
        "num_warps": 1,
        "num_stages": 1,
    },
    "sparse_attn2_prefill": {
        "block_n": 32,
        "num_dchunk": 8,
        "tiles_per_prog": 1,
        "row_chunk": 512,
        "num_warps": 1,
        "num_stages": 1,
    },
}
_DEFAULT_GRF = "256"
# Rows at/above which a use_prefill_config call takes the prefill region.
_PREFILL_ROWS = 2048


def _region(num_rows: int, use_prefill_config: bool) -> str:
    if use_prefill_config and num_rows >= _PREFILL_ROWS:
        return "sparse_attn2_prefill"
    return "sparse_attn2_decode"


def select_attn2_config(
    num_rows: int, num_kv_heads: int, use_prefill_config: bool, selection_width: int
) -> dict:
    """Launch config of one call: a pure function of shapes (graph-capture stable)."""
    del num_kv_heads
    region = _region(num_rows, use_prefill_config)

    def get(field):
        return xpu_tune.get(region, field, _DEFAULTS[region][field])

    block_n = get("block_n")
    num_dchunk = get("num_dchunk")
    tiles_per_prog = get("tiles_per_prog")
    num_tiles = triton.cdiv(selection_width, block_n)
    num_warps = get("num_warps")
    if num_warps > 1 and block_n >= 32:
        # IGC 2.41.5 dies with SIGFPE ("Internal Compiler Error: Floating point
        # exception") compiling these, which kills the process (measured: every
        # BLOCK_N 32/64 config with num_warps 2, Triton 3.8.0 and Intel main).
        raise ValueError(
            f"B70_QSA attn2: num_warps {num_warps} with block_n {block_n} crashes "
            "IGC 2.41.5; use num_warps 1 (or block_n 16)"
        )
    extra = xpu_tune.launch_extra(region)
    if xpu_tune.get(region, "grf_mode", None) is None:
        extra = {"grf_mode": _DEFAULT_GRF}
    return {
        "region": region,
        "block_n": block_n,
        "num_dchunk": num_dchunk,
        "tiles_per_prog": min(tiles_per_prog, num_tiles),
        "num_tiles": num_tiles,
        "row_chunk": max(1, min(get("row_chunk"), num_rows)),
        "num_warps": num_warps,
        "num_stages": get("num_stages"),
        "extra": extra,
    }


def merge_kwargs() -> dict:
    return {
        "num_warps": xpu_tune.get("merge", "num_warps", 2),
        "num_stages": 1,
        **xpu_tune.launch_extra("merge"),
    }


def qsa_sparse_paged_attention_attn2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor,
    output_gate_view: torch.Tensor,
    softmax_scale: float,
    output_scale: float,
    block_m: int,
) -> torch.Tensor:
    """Launch attn2; the caller (qsa.qsa_sparse_paged_attention) validated inputs."""
    head_dim = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    group_size = q.shape[1] // num_kv_heads
    selection_width = logical_indices.shape[1] - 1
    cfg = select_attn2_config(
        q.shape[0], num_kv_heads, use_prefill_config, selection_width
    )
    dc = head_dim // cfg["num_dchunk"]
    assert dc >= 16 and dc * cfg["num_dchunk"] == head_dim, "attn2 head_dim chunking"
    rows_total = q.shape[0]
    row_chunk = cfg["row_chunk"]
    num_tiles = cfg["num_tiles"]
    partial_output = torch.empty(
        (num_tiles, row_chunk, q.shape[1], head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    partial_lse = torch.empty(
        (num_tiles, row_chunk, q.shape[1]), dtype=torch.float32, device=q.device
    )
    progs = triton.cdiv(num_tiles, cfg["tiles_per_prog"])
    for r0 in range(0, rows_total, row_chunk):
        rows = min(row_chunk, rows_total - r0)
        q_c = q[r0 : r0 + rows]
        ind_c = logical_indices[r0 : r0 + rows]
        req_c = token_to_req[r0 : r0 + rows]
        _qsa_attn2_tile_kernel[(rows, num_kv_heads, progs)](
            q_c,
            k_cache,
            v_cache,
            ind_c,
            block_table,
            req_c,
            partial_output,
            partial_lse,
            softmax_scale,
            output_scale,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            logical_indices.stride(0),
            block_table.stride(0),
            rows,
            k_cache.shape[0],
            block_table.shape[0],
            TOPK=selection_width,
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=q.shape[1],
            BLOCK_M=block_m,
            BLOCK_N=cfg["block_n"],
            NUM_DCHUNK=cfg["num_dchunk"],
            TILES_PER_PROG=cfg["tiles_per_prog"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            **cfg["extra"],
        )
        _qsa_attn2_merge_kernel[(rows, q.shape[1])](
            partial_output,
            partial_lse,
            ind_c,
            out[r0 : r0 + rows],
            output_gate_view[r0 : r0 + rows],
            logical_indices.stride(0),
            out.stride(0),
            out.stride(1),
            output_gate_view.stride(0),
            output_gate_view.stride(1),
            rows,
            TOPK=selection_width,
            BLOCK_N=cfg["block_n"],
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=q.shape[1],
            **merge_kwargs(),
        )
    return out


def attn2_profiles(
    num_kv_heads: int, selection_width: int, max_rows: int = 8192
) -> set[tuple]:
    """Every (block_n, num_dchunk, tiles_per_prog, warps, stages, extra) in use."""
    out = set()
    for num_rows in range(1, max_rows + 1):
        for use_prefill_config in (False, True):
            c = select_attn2_config(
                num_rows, num_kv_heads, use_prefill_config, selection_width
            )
            out.add(
                (
                    c["block_n"],
                    c["num_dchunk"],
                    c["tiles_per_prog"],
                    c["num_warps"],
                    c["num_stages"],
                    tuple(sorted(c["extra"].items())),
                )
            )
    return out

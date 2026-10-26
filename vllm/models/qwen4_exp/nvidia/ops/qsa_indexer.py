# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA index selection."""

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from ...common import qsa_debug as _qsa_debug
from . import qsa_xpu_tuning as xpu_tune

logger = init_logger(__name__)

_TOPK_WORKSPACE_BYTES = 1024 * 1024
_DECODE_BLOCK_N = 64


def _min_dot_n() -> int:
    """B70-QSA: floor for the decode scorer's tl.dot N (query rows x heads).

    1 compiles the original kernel (CUDA). Intel Triton 3.8.0+xpu rejects
    tl.dot with N < 16 at compile time ("Input shapes should have M >= 1,
    N >= 16 and K >= 16"; measured on B70, G1 2026-10-25), and a plain decode
    has N = 4. So XPU pads N to 16 by default; padded rows are masked.
    B70_QSA_MIN_DOT_N (XPU only) overrides, e.g. 32 for the tuning sweep.
    B70-QSA-T: values below 16 are honoured only with B70_QSA_ALLOW_SMALL_DOT=1,
    the arm for a Triton that accepts N < 16 (Intel backend main, whose
    min_dot_size is (1, 1, 1)); on 3.8.0 that arm fails to compile by design.
    """
    if current_platform.is_xpu():
        value = int(os.environ.get("B70_QSA_MIN_DOT_N", "16"))
        if os.environ.get("B70_QSA_ALLOW_SMALL_DOT", "0") == "1":
            return max(1, value)
        return max(16, value)
    return 1


def _decode_block_n() -> int:
    """Key tile of the decode scorer (B70-QSA-T: overridable on XPU)."""
    if current_platform.is_xpu():
        return xpu_tune.get("decode_score", "block_n", _DECODE_BLOCK_N)
    return _DECODE_BLOCK_N


def _decode_launch_config(dtype: torch.dtype) -> tuple[int, int]:
    """(STAGES, num_warps) for the decode scoring kernel.

    B70-QSA: every decode tile choice goes through here. The XPU entry is an
    UNTUNED copy of the GB300 one (a BMG-measured table replaces it).
    """
    if current_platform.is_xpu():
        stages = xpu_tune.get("decode_score", "stages", 2)
        warps = 1 if dtype == torch.float8_e4m3fn else 2
        return stages, xpu_tune.get("decode_score", "num_warps", warps)
    # tuned on GB300
    return 2, 1 if dtype == torch.float8_e4m3fn else 2


def _prefill_launch_config(dtype: torch.dtype) -> tuple[int, int, int, int, int]:
    """(TILE_R, BLOCK_N, K_TILES, STAGES, num_warps) for prefill scoring.

    B70-QSA: every prefill tile choice goes through here; the XPU entry is an
    UNTUNED copy of the GB300 one. Its dot is [BLOCK_N, D] x [D, TILE_R*heads],
    so N = TILE_R * padded heads (256 here) is DPAS-eligible on XPU.
    """
    if current_platform.is_xpu():
        base = (
            (32, 64, 16, 2, 8) if dtype == torch.float8_e4m3fn else (64, 64, 16, 2, 4)
        )
        names = ("tile_r", "block_n", "k_tiles", "stages", "num_warps")
        return tuple(  # type: ignore[return-value]
            xpu_tune.get("prefill_score", n, v) for n, v in zip(names, base)
        )
    # tuned on GB300
    if dtype == torch.float8_e4m3fn:
        return 32, 64, 16, 2, 8
    return 64, 64, 16, 2, 4


@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MIN_DOT_N: tl.constexpr = 1,
) -> None:
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # B70-QSA: the dot's N is DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED (4 at a
    # plain decode). MIN_DOT_N (default 1 = unchanged) pads the query rows so
    # N >= MIN_DOT_N, e.g. 16 for DPAS on XPU; padded rows are masked.
    DECODE_QUERY_LEN_PADDED: tl.constexpr = max(
        triton.next_power_of_2(DECODE_QUERY_LEN),
        MIN_DOT_N // NUM_HEADS_PADDED,
    )
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(
        visible_blocks_ptr + rows,
        mask=valid_query_offsets,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(NUM_COLUMNS, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED)
    query_offset = n // NUM_HEADS_PADDED
    head = n % NUM_HEADS_PADDED
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(
            scores,
            (BLOCK_N, DECODE_QUERY_LEN_PADDED, NUM_HEADS_PADDED),
        )
        score = tl.sum(scores, axis=2)
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < NUM_COLUMNS)
            & (columns[:, None] < visible[None, :]),
        )


@triton.jit(do_not_specialize=["num_rows", "query_offset"])
def _qsa_mqa_paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    query_offset,
    page_table_width,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    num_columns = page_table_width * PAGE_SIZE
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(
        tl.load(query_start_loc_ptr + request) - query_base, query_offset
    )
    request_end = tl.minimum(
        tl.load(query_start_loc_ptr + request + 1) - query_base, query_end
    )
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(
        visible_blocks_ptr + absolute_rows,
        mask=valid_rows,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_row_offsets = m // NUM_HEADS_PADDED
    q_rows = absolute_row_start + q_row_offsets
    heads = m % NUM_HEADS_PADDED
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS)
        & (absolute_row_start + q_row_offsets[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, page_table_width - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        # With an fp8 cache, SM90 lowers this dot to reduced-precision wgmma
        # accumulation; SM100 tcgen05 accumulates in fp32. The SM90 error
        # (~1e-4 relative) is far below the e4m3 quantization noise floor.
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2)
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=store_mask,
        )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    visible_blocks_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    OUTPUT_WIDTH: tl.constexpr = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    visible_blocks = tl.load(visible_blocks_ptr + row)
    complete_blocks = tl.minimum(visible_blocks, BLOCK_TOPK)
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (columns < OUTPUT_WIDTH) & (is_expanded | is_tail) & (token >= 0)
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=columns < OUTPUT_WIDTH,
    )
    # TRAILING COUNT COLUMN: the packed buffer is OUTPUT_WIDTH+1 wide; the
    # last column holds this row's valid-entry count (expanded blocks plus
    # causal tail). It is NOT a token index — the sparse attention kernel
    # reads it as the row's tile-loop bound.
    if tl.program_id(1) == 0:
        tl.store(
            output_ptr + row * stride_output_row + OUTPUT_WIDTH * stride_output_column,
            expanded_count + tail_count,
        )


def _decode_tiles_per_program(num_requests: int, columns: int) -> int:
    if current_platform.is_xpu():
        forced = xpu_tune.get("decode_score", "tiles_per_prog", 0)
        if forced:
            return forced
    programs = num_requests * triton.cdiv(columns, _decode_block_n())
    if programs < 16384:
        return 1
    if programs < 32768:
        return 2
    if programs < 131072:
        return 4
    return 8


def _qsa_decode_warmup_profiles(
    max_dql: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
    columns: int,
) -> tuple[tuple[int, int], ...]:
    profiles: list[tuple[int, int]] = []
    for dql in range(1, max_dql + 1):
        max_reqs = min(max_num_reqs, max_num_batched_tokens // dql)
        requests_by_grouping: dict[int, int] = {}
        for num_requests in range(1, max_reqs + 1):
            tiles_per_program = _decode_tiles_per_program(num_requests, columns)
            requests_by_grouping.setdefault(tiles_per_program, num_requests)
        profiles.extend(
            (dql, num_requests) for num_requests in requests_by_grouping.values()
        )
    return tuple(profiles)


def warmup_qsa_mqa_paged_decode(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    max_decode_query_len: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Compile every reachable decode specialization without launching it."""
    page_size = k_cache.shape[1]
    page_table_width = page_table.shape[1]
    columns = page_table_width * page_size
    profiles = _qsa_decode_warmup_profiles(
        max_decode_query_len,
        max_num_reqs,
        max_num_batched_tokens,
        columns,
    )
    if not profiles:
        return ()

    k_cache_ptr = TritonWarmupTensor(k_cache.dtype, shape=tuple(k_cache.shape))
    page_table_ptr = TritonWarmupTensor(
        page_table.dtype,
        shape=(max_num_reqs, page_table_width),
    )
    visible_blocks_ptr = TritonWarmupTensor(torch.int32)

    for decode_query_len, num_requests in profiles:
        num_rows = decode_query_len * num_requests
        q_ptr = TritonWarmupTensor(
            k_cache.dtype,
            shape=(num_rows, num_heads, head_dim),
        )
        logits_ptr = TritonWarmupTensor(
            torch.float32,
            shape=(num_rows, columns),
        )
        tiles_per_program = _decode_tiles_per_program(num_requests, columns)
        stages, num_warps = _decode_launch_config(k_cache.dtype)
        _qsa_mqa_paged_uniform_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            page_table_ptr,
            visible_blocks_ptr,
            logits_ptr,
            num_heads * head_dim,
            head_dim,
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            columns,
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=page_table_width,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            DECODE_QUERY_LEN=decode_query_len,
            BLOCK_N=_decode_block_n(),
            TILES_PER_PROG=tiles_per_program,
            STAGES=stages,
            MIN_DOT_N=_min_dot_n(),
            num_warps=num_warps,
            grid=(
                num_requests,
                triton.cdiv(columns, _decode_block_n() * tiles_per_program),
            ),
            **xpu_tune.launch_extra("decode_score"),
        )
    return profiles


def _prefill_logits(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    logits_workspace: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
) -> torch.Tensor:
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]
    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]
    assert logits_workspace.is_contiguous()
    assert logits_workspace.numel() >= num_queries * logits_width

    # slice from pre-allocated workspace
    logits = logits_workspace[: num_queries * logits_width].view(
        num_queries, logits_width
    )

    TILE_R, BLOCK_N, K_TILES, STAGES, num_warps = _prefill_launch_config(k_cache.dtype)
    grid = (
        page_table.shape[0],
        triton.cdiv(min(num_queries, max_query_len), TILE_R),
        triton.cdiv(logits_width, BLOCK_N * K_TILES),
    )
    _qsa_mqa_paged_prefill_kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        num_queries,
        query_offset,
        page_table.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        TILE_R=TILE_R,
        BLOCK_N=BLOCK_N,
        K_TILES=K_TILES,
        STAGES=STAGES,
        num_warps=num_warps,
        **xpu_tune.launch_extra("prefill_score"),
    )
    return logits


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor,
) -> None:
    """Expand compressed blocks and compact the causal tail of the open group."""
    assert token_topk % compress_ratio == 0
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    assert block_indices.shape == (query_positions.numel(), block_topk)
    assert visible_blocks.shape == query_positions.shape
    # +1: the packed buffer's trailing column holds each row's valid-entry
    # count (never a token index); the index region below only writes
    # columns [0, output_width).
    assert out.shape == (block_indices.shape[0], output_width + 1)
    column_block = xpu_tune.get("expand", "column_block", 256)
    grid = (block_indices.shape[0], triton.cdiv(output_width, column_block))
    _expand_qsa_indices_kernel[grid](
        block_indices,
        query_positions,
        visible_blocks,
        out,
        *block_indices.stride(),
        *out.stride(),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        COLUMN_BLOCK=column_block,
        num_warps=xpu_tune.get("expand", "num_warps", 4),
        **xpu_tune.launch_extra("expand"),
    )


def _qsa_topk_mode() -> str:
    """B70-QSA: top-k backend off CUDA, from B70_QSA_TOPK.

    ``native``: vllm-xpu-kernels ``torch.ops._C.top_k_per_row_decode`` (SYCL,
    registered for kXPU). ``torch``: exact, deterministic torch path below.
    ``auto`` (default): native when the op is registered, else torch.
    ``native_det`` (B70-QSA-Q): native, then made identical to ``torch``.
    B70-QSA-D01: by default the repair is ``_qsa_topk_select`` anchored on the
    native k-th score (device ops only, no host sync, no full-width sort);
    under graph capture it is ``_qsa_topk_select`` with a radix-select
    threshold. ``B70_QSA_TOPK_DET_IMPL=legacy`` restores the Q repair (tie rows
    redone on the torch path after a host sync; torch path under capture).
    ``radix`` (B70-QSA-D01): ``_qsa_topk_select`` with the radix-select
    threshold, no native op; identical output to ``torch``.
    """
    mode = os.environ.get("B70_QSA_TOPK", "auto")
    if mode not in ("auto", "native", "torch", "native_det", "radix"):
        raise ValueError(
            f"B70_QSA_TOPK must be auto|native|torch|native_det|radix, got {mode!r}"
        )
    if mode == "auto":
        has_native = hasattr(torch.ops._C, "top_k_per_row_decode")
        mode = "native" if has_native else "torch"
    return mode


def _qsa_topk_torch(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
    block_indices: torch.Tensor,
) -> None:
    """Deterministic row top-k with the cooperative/persistent_topk contract.

    B70-QSA: columns >= visible_blocks[row] are ignored; a row with fewer than
    block_topk visible blocks gets 0..visible-1 then -1. Ties break toward the
    lower column, exactly, by sorting an int64 key (order-preserving float32
    bits << 32 | inverted column) -- ReLU-summed scores tie at 0.0 often, and
    float top-k picks among ties arbitrarily. Output order: descending score.
    """
    rows, width = logits.shape
    if rows == 0:
        return
    # B70-QSA-09: the int64 key costs ~5 transient int64 copies of the logits
    # (bits, flip, key, mask select, topk input) -- ~10x the fp32 logits. A
    # 4096-row prefill chunk at long context OOMed rank 0 (192 MiB alloc, 179 MiB
    # free; W1 2026-10-25). Rows are independent, so process them in chunks
    # sized to B70_QSA_TOPK_TORCH_CHUNK_MB (default 64) of key memory.
    budget = int(os.environ.get("B70_QSA_TOPK_TORCH_CHUNK_MB", "64")) << 20
    step = max(1, budget // max(1, width * 8 * 5))
    if step < rows:
        logger.info_once(
            "B70-QSA: torch top-k processes rows in chunks of %d (width %d)",
            step,
            width,
        )
        for start in range(0, rows, step):
            end = min(rows, start + step)
            _qsa_topk_torch_rows(
                logits[start:end],
                visible_blocks[start:end],
                block_topk,
                block_indices[start:end],
            )
        return
    _qsa_topk_torch_rows(logits, visible_blocks, block_topk, block_indices)


def _qsa_topk_torch_rows(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
    block_indices: torch.Tensor,
) -> None:
    """One chunk of _qsa_topk_torch (same contract)."""
    rows, width = logits.shape
    k = min(block_topk, width)
    bits = logits.contiguous().view(torch.int32).to(torch.int64)
    # IEEE-754 total order on the int view: flip the magnitude bits of negatives.
    bits = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    columns = torch.arange(width, device=logits.device, dtype=torch.int64)
    key = (bits << 32) | (0xFFFFFFFF - columns)
    visible = visible_blocks[:rows].to(torch.int64).clamp(0, width)
    live = columns.unsqueeze(0) < visible.unsqueeze(1)
    key = torch.where(live, key, torch.iinfo(torch.int64).min)
    top = torch.topk(key, k, dim=1, largest=True, sorted=True).indices
    ranks = torch.arange(k, device=logits.device, dtype=torch.int64)
    top = torch.where(ranks.unsqueeze(0) < visible.unsqueeze(1), top, -1)
    block_indices[:, :k].copy_(top.to(torch.int32))
    if k < block_topk:
        block_indices[:, k:].fill_(-1)


def _qsa_det_impl() -> str:
    """B70-QSA-D01: native_det repair, ``select`` (default) or ``legacy``."""
    impl = os.environ.get("B70_QSA_TOPK_DET_IMPL", "select")
    if impl not in ("select", "legacy"):
        raise ValueError(f"B70_QSA_TOPK_DET_IMPL must be select|legacy, got {impl!r}")
    return impl


def _ordered_bits(logits: torch.Tensor) -> torch.Tensor:
    """int32 view of fp32 scores whose signed order is the torch key's order.

    Same transform as _qsa_topk_torch_rows (flip the magnitude bits of the
    negatives), so -0.0 sorts below +0.0 there and here alike.
    """
    bits = logits.contiguous().view(torch.int32)
    return torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


def _radix_passes(rows: int) -> tuple[tuple[int, int], ...]:
    """(shift, bits) per pass, MSB first; two 16-bit passes while the per-row
    histograms stay under 32 MiB, else four 8-bit passes."""
    if rows * (1 << 16) * 4 <= 32 << 20:
        return ((16, 16), (0, 16))
    return ((24, 8), (16, 8), (8, 8), (0, 8))


def _radix_threshold(
    ordered: torch.Tensor, live: torch.Tensor, want: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact k-th largest ``ordered`` value per row by MSB-first radix select.

    Returns (threshold [rows] int64 holding an int32 value, remaining [rows]
    int64): the ``want``-th largest live value, and how many live entries equal
    to it belong to the top ``want``. Histograms are integer scatter_add, so
    the result is exact and run-to-run identical. Rows with want == 0 return
    garbage that the caller never uses (they have no live column).
    """
    rows = ordered.shape[0]
    dev = ordered.device
    match = live
    remaining = want.clone()
    threshold = torch.zeros(rows, dtype=torch.int64, device=dev)
    row_base = torch.arange(rows, dtype=torch.int64, device=dev).unsqueeze(1)
    for i, (shift, bits) in enumerate(_radix_passes(rows)):
        bins = 1 << bits
        if i == 0:
            # arithmetic shift keeps the sign: [-bins/2, bins/2) -> [0, bins)
            digit = (ordered >> shift).to(torch.int64) + bins // 2
        else:
            digit = ((ordered >> shift) & (bins - 1)).to(torch.int64)
        hist = torch.zeros(rows * bins, dtype=torch.int32, device=dev)
        hist.scatter_add_(
            0, (digit + row_base * bins).reshape(-1), match.to(torch.int32).reshape(-1)
        )
        # at_least[:, d] = live matching entries whose digit is >= d
        at_least = hist.view(rows, bins).flip(1).cumsum(1).flip(1)
        chosen = (at_least >= remaining.unsqueeze(1)).sum(1) - 1
        chosen = chosen.clamp(0, bins - 1)
        above = at_least.gather(1, (chosen + 1).clamp(max=bins - 1).unsqueeze(1))
        above = torch.where(chosen + 1 < bins, above.squeeze(1), 0)
        remaining = remaining - above
        match = match & (digit == chosen.unsqueeze(1))
        value = chosen - bins // 2 if i == 0 else chosen
        threshold = threshold + (value << shift)
    return threshold, remaining


def _qsa_topk_select(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
    block_indices: torch.Tensor,
    native_indices: torch.Tensor | None = None,
) -> None:
    """B70-QSA-D01: _qsa_topk_torch's exact output without a full-width sort.

    Same contract and the same output, element for element: per row the
    k_eff = min(block_topk, width, visible) largest (score, -column) keys over
    the live columns, in descending key order, then -1.

    1. Threshold T and r (how many live entries equal to T to keep): from the
       native selection when given (T = its smallest selected score; any
       correct top-k set has the true k-th score, whatever its tie choice), else
       by exact radix select.
    2. Keep every live column above T, plus the r lowest live columns equal
       to T (cumsum rank). That is exactly the set of the k_eff largest keys,
       since the key breaks equal scores toward the lower column.
    3. Compact the kept columns (cumsum position + scatter) and sort only the
       k_eff kept keys.

    Device ops only: no host sync, graph-capturable without native_indices.
    Rows are processed in chunks of B70_QSA_TOPK_SELECT_CHUNK_MB (default 256)
    of transient memory.
    """
    rows, width = logits.shape
    if rows == 0:
        return
    budget = int(os.environ.get("B70_QSA_TOPK_SELECT_CHUNK_MB", "256")) << 20
    step = max(1, budget // max(1, width * 40))
    for start in range(0, rows, step):
        end = min(rows, start + step)
        _qsa_topk_select_rows(
            logits[start:end],
            visible_blocks[start:end],
            block_topk,
            block_indices[start:end],
            None if native_indices is None else native_indices[start:end],
        )


def _qsa_topk_select_rows(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
    block_indices: torch.Tensor,
    native_indices: torch.Tensor | None,
) -> None:
    """One chunk of _qsa_topk_select (same contract)."""
    rows, width = logits.shape
    dev = logits.device
    k = min(block_topk, width)
    ordered = _ordered_bits(logits)
    columns = torch.arange(width, device=dev, dtype=torch.int64)
    visible = visible_blocks[:rows].to(torch.int64).clamp(0, width)
    live = columns.unsqueeze(0) < visible.unsqueeze(1)
    want = visible.clamp(max=k)
    if native_indices is None:
        threshold, remaining = _radix_threshold(ordered, live, want)
    else:
        # The native op compares floats: its k-th score is the true k-th
        # float value, but -0.0 == +0.0 there while the key orders -0.0
        # below +0.0. Resolve a zero threshold to the right signed zero.
        sel = native_indices[:rows, :k].long()
        valid = (sel >= 0) & (sel < visible.unsqueeze(1))
        picked = logits.gather(1, sel.clamp(0, width - 1))
        kth = torch.where(valid, picked, float("inf")).min(dim=1).values
        threshold = _ordered_bits(kth).to(torch.int64)
        wide = ordered.to(torch.int64)
        at_least_pos_zero = (live & (wide >= 0)).sum(1)
        threshold = torch.where(
            kth == 0,
            torch.where(at_least_pos_zero >= want, 0, -1),  # +0.0 / -0.0 keys
            threshold,
        )
        above = (live & (wide > threshold.unsqueeze(1))).sum(1)
        remaining = want - above
        if os.environ.get("B70_QSA_TOPK_SELECT_CHECK", "0") == "1":
            equal = (live & (wide == threshold.unsqueeze(1))).sum(1)
            bad = (want > 0) & ((remaining < 1) | (remaining > equal))
            if bool(bad.any()):  # host sync: debug only
                raise RuntimeError(
                    "B70-QSA: native top-k set is not a top-k set in "
                    f"{int(bad.sum())} rows (NaN scores?)"
                )
    wide = ordered.to(torch.int64)
    above_mask = live & (wide > threshold.unsqueeze(1))
    equal = live & (wide == threshold.unsqueeze(1))
    equal_rank = equal.to(torch.int32).cumsum(1)
    keep = above_mask | (equal & (equal_rank <= remaining.unsqueeze(1)))
    # Compact: the j-th kept column (ascending) lands in slot j; the rest go
    # to a scratch slot k that is dropped.
    slot = torch.where(keep, keep.to(torch.int32).cumsum(1) - 1, k)
    packed = torch.full((rows, k + 1), -1, dtype=torch.int64, device=dev)
    packed.scatter_(1, slot.to(torch.int64), columns.unsqueeze(0).expand(rows, width))
    chosen = packed[:, :k]
    valid = chosen >= 0
    col = chosen.clamp_min(0)
    key = (ordered.gather(1, col).to(torch.int64) << 32) | (0xFFFFFFFF - col)
    floor = torch.iinfo(torch.int64).min
    key = torch.where(valid, key, floor)
    key = torch.sort(key, dim=1, descending=True).values
    top = torch.where(key == floor, -1, 0xFFFFFFFF - (key & 0xFFFFFFFF))
    block_indices[:, :k].copy_(top.to(torch.int32))
    if k < block_topk:
        block_indices[:, k:].fill_(-1)


def _qsa_topk_native(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
    block_indices: torch.Tensor,
) -> None:
    assert block_indices.is_contiguous()
    seq_lens = visible_blocks[: logits.shape[0]].to(torch.int32).contiguous()
    # next_n=1: row i scores columns [0, seq_lens[i]); rows with fewer than
    # topK columns get 0..len-1 then -1, as cooperative/persistent_topk do.
    torch.ops._C.top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        block_indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        block_topk,
    )


def _qsa_topk_make_deterministic(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
    block_indices: torch.Tensor,
) -> None:
    """B70-QSA-Q: turn a native selection into exactly _qsa_topk_torch's.

    Rows without a boundary tie have one correct set; it only needs the torch
    order (descending score, lower column first). Rows with a tie at the k-th
    score are recomputed on the torch path. Eager only: the tie rows are
    gathered with a host sync.
    """
    rows, width = logits.shape
    if rows == 0:
        return
    k = min(block_topk, width)
    tied = _qsa_debug.boundary_tie_rows(logits, visible_blocks, block_indices, k)
    selected = block_indices[:rows, :k].long()
    valid = selected >= 0
    columns = selected.clamp_min(0)
    bits = logits.gather(1, columns).contiguous().view(torch.int32).to(torch.int64)
    bits = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    key = (bits << 32) | (0xFFFFFFFF - columns)
    key = torch.where(valid, key, torch.iinfo(torch.int64).min)
    order = torch.argsort(key, dim=1, descending=True)
    block_indices[:rows, :k].copy_(selected.gather(1, order).to(torch.int32))
    tied_rows = tied.nonzero().flatten()
    if tied_rows.numel():
        redo = torch.empty(
            (tied_rows.numel(), block_indices.shape[1]),
            dtype=block_indices.dtype,
            device=block_indices.device,
        )
        _qsa_topk_torch(
            logits.index_select(0, tied_rows),
            visible_blocks[:rows].index_select(0, tied_rows),
            block_topk,
            redo,
        )
        block_indices.index_copy_(0, tied_rows, redo)


def _topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    block_indices: torch.Tensor,
    topk_workspace: torch.Tensor,
) -> None:
    # similar dispatch logic as DeepSeek indexer
    block_topk = token_topk // compress_ratio
    if not current_platform.is_cuda():
        # B70-QSA: cooperative/persistent_topk are CUDA-only csrc ops.
        mode = _qsa_topk_mode()
        det_impl = _qsa_det_impl() if mode == "native_det" else ""
        logger.info_once(
            "B70-QSA: QSA block top-k backend = %s%s",
            mode,
            f" (repair = {det_impl})" if det_impl else "",
        )
        if mode == "radix" or (det_impl == "select" and _qsa_debug.is_capturing()):
            _qsa_topk_select(logits, visible_blocks, block_topk, block_indices)
            return
        if mode == "torch" or (mode == "native_det" and _qsa_debug.is_capturing()):
            _qsa_topk_torch(logits, visible_blocks, block_topk, block_indices)
            return
        _qsa_topk_native(logits, visible_blocks, block_topk, block_indices)
        if _qsa_debug.TIES:
            _qsa_debug.count_boundary_ties(
                logits, visible_blocks, block_indices, block_topk
            )
        if det_impl == "select":
            _qsa_topk_select(
                logits,
                visible_blocks,
                block_topk,
                block_indices,
                native_indices=block_indices,
            )
        elif det_impl == "legacy":
            _qsa_topk_make_deterministic(
                logits, visible_blocks, block_topk, block_indices
            )
        return
    use_cooperative_topk = (
        logits.shape[0] <= 64
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    topk_op = (
        torch.ops._C.cooperative_topk
        if use_cooperative_topk
        else torch.ops._C.persistent_topk
    )
    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )


def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    block_indices: torch.Tensor,
) -> None:
    """Score and select compressed blocks for a request-major decode batch.

    Args:
        q: Query tensor shaped ``[num_requests * decode_query_len, heads,
            head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Request block table shaped ``[num_requests, max_pages]``.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        decode_query_len: Number of query tokens per request.
        block_indices: Compressed-index output buffer.

    """
    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert decode_query_len > 0 and q.shape[0] % decode_query_len == 0
    assert q.dtype == k_cache.dtype, "Q and the compressed K cache must match"
    num_requests = q.shape[0] // decode_query_len
    assert page_table.shape[0] == num_requests
    assert visible_blocks.shape == (q.shape[0],)

    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    tiles_per_program = _decode_tiles_per_program(num_requests, columns)
    stages, num_warps = _decode_launch_config(k_cache.dtype)
    grid = (
        num_requests,
        triton.cdiv(columns, _decode_block_n() * tiles_per_program),
    )
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        k_cache,
        page_table,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        DECODE_QUERY_LEN=decode_query_len,
        BLOCK_N=_decode_block_n(),
        TILES_PER_PROG=tiles_per_program,
        STAGES=stages,
        MIN_DOT_N=_min_dot_n(),
        num_warps=num_warps,
        **xpu_tune.launch_extra("decode_score"),
    )
    _topk(
        logits,
        visible_blocks,
        token_topk,
        compress_ratio,
        block_indices,
        torch.empty((_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device),
    )


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Score and select compressed prefill blocks in bounded chunks.

    Args:
        q: Packed prefill query tensor shaped ``[num_tokens, heads, head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Block table shaped ``[num_requests, max_pages]``.
        query_start_loc: Packed prefill query offsets with a terminal offset.
            Offsets may share the base of a larger backing tensor.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        max_query_len: Maximum number of query tokens in one request.
        block_indices: Compressed-index output buffer.
        max_seq_len: Longest context length in the batch this step.

    """
    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert q.dtype == k_cache.dtype, "Q and the compressed K cache must match"
    rows = q.shape[0]
    # No row scores beyond cdiv(max_seq_len, compress_ratio) compressed
    # columns. Round up to 64 to keep the logits row stride
    # cooperative_topk-compatible.
    logits_width = triton.cdiv(triton.cdiv(max_seq_len, compress_ratio), 64) * 64
    logits_width = min(max(64, logits_width), page_table.shape[1] * k_cache.shape[1])

    # chunk the inputs to keep temp logits below VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    # always allocate the worst case to avoid memory fragmentation.
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))

    budget_bytes = max(max_logits_bytes, page_table.shape[1] * k_cache.shape[1] * 4)
    logits_workspace = q.new_empty(budget_bytes // 4, dtype=torch.float32)

    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
    )

    for query_start in range(0, rows, rows_per_chunk):
        query_end = min(query_start + rows_per_chunk, rows)
        query_slice = slice(query_start, query_end)
        logits = _prefill_logits(
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            logits_workspace,
            max_query_len,
            logits_width,
            query_offset=query_start,
            num_queries=query_end - query_start,
        )
        _topk(
            logits,
            visible_blocks[query_slice],
            token_topk,
            compress_ratio,
            block_indices[query_slice],
            topk_workspace,
        )


__all__ = [
    "expand_qsa_block_indices",
    "qsa_select_paged_decode",
    "qsa_select_paged_prefill",
    "warmup_qsa_mqa_paged_decode",
]

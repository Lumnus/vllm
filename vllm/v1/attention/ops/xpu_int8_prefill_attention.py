# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native INT8 prefill attention over an int8_per_token_head KV cache (XPU).

The paged decode kernel of vllm-xpu-kernels reads the int8 cache directly, but
only for decode rows; prefill and mixed batches dequantize every referenced
block into a bf16/fp16 paged scratch and run the BF16 FlashAttention kernel
over it (`xpu_int8_per_token_head.forward_prefill`). This module removes that
upcast for causal prefill: it reads the int8 codes and their fp32 scales from
the cache in place and runs the dots on the matrix engine in int8.

Numerical contract (`pv_int8=False`, AttentionConfig `int8_prefill_dot="qk"`):

1. Q is quantized per (token, query head) with the KV writer's rounding:
   scale = absmax / 127, x * (1 / scale), half away from zero, clamped to
   [-127, 127] (`quantize_query_int8`).
2. Q.K is an int8 x int8 -> int32 dot over the whole head (exact: at most
   head_dim * 127 * 127 < 2**31), then S = int32 * q_scale * k_scale in FP32.
   The K codes and scales are the cached ones, the same the decode kernel
   reads; the fresh chunk's K/V are already in the cache (the layer writes them
   before it attends), so cached and fresh tokens are represented alike.
3. Causal mask on absolute positions (query i of a sequence of length L with
   q_len new tokens sits at L - q_len + i); FP32 online softmax in base 2.
4. P.V: P * v_scale (per key column, after the row sum, so the normalizer is
   unaffected; the decode kernel's order) rounded to BF16, times the V codes
   widened to BF16 (exact), FP32 accumulation.

`pv_int8=True` (`"qk_pv"`) also runs P.V in int8: P * v_scale is quantized per
query row of each KV tile (scale = row max / 127, round half up, P >= 0), the
dot is int8 x int8 -> int32 and the tile's scale multiplies the int32 result.

Against today's dequantize-then-BF16 path the K/V operands are closer to the
cache (no BF16 rounding of code * scale); the query gains an int8 rounding
(relative step 1/127 of the row's absmax) and, with "qk_pv", the attention
weights one (step 1/127 of the tile row's maximum, so weights below half a
step of their tile's maximum drop to zero). Decode on the same cached tokens
keeps a BF16 query, so prefill and decode agree on K/V exactly and differ by
the query rounding.

Tiles: one program = BLOCK_M query tokens of one sequence x one query head x
the output columns of its `dv_split` part. A KV tile is BLOCK_N consecutive
slots of one cache block (BLOCK_N divides the block size), so every K, V, Q
and output operand is a 2D tensor-descriptor block and the only gathered
addresses are one block-table entry per tile and the tile's scales: no
per-element address tensors. Q.K loops over head-dim chunks of QK_DC; the
output lives in up to four column accumulators. Heads are the fastest grid
axis, so the programs of one KV group walk the same K/V tiles together.
Programs find their sequence by a binary search over `query_start_loc` on the
device: no host reads, no scratch, graph-safe.
"""

from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import find_seq_idx

logger = init_logger(__name__)

# AttentionConfig.int8_prefill_dot -> kernel mode.
INT8_PREFILL_DOT_MODES = {"off": 0, "qk": 1, "qk_pv": 2}


class Int8PrefillLaunch(NamedTuple):
    block_m: int = 64
    """Query tokens per program (the dots' M)."""
    block_n: int = 64
    """KV slots per tile (Q.K's N, P.V's K: >= 32 for an int8 P.V); lowered to
    the largest power of two that divides the cache block size."""
    qk_chunk: int = 64
    """Head-dim chunk of Q.K (the dot's K: >= 32 for int8 DPAS)."""
    dv_chunks: int = 4
    """Output column accumulators per program: 1, 2 or 4."""
    dv_split: int = 1
    """Programs per (query block, head) along the output columns; each
    recomputes Q.K, so > 1 trades Q.K work for registers."""
    num_warps: int = 16
    grf_mode: str = "256"


# Chosen from an ahead-of-time compile for BMG-G31 (Triton 3.9.0 Intel backend,
# IGC from oneAPI 2026.1, 256 GRF, head_dim 256, 6 query heads per KV head,
# tiles of 64 slots of a 1600-slot block): both dots lower to s8 x s8 DPAS and
# the descriptor tiles to 2D block messages. Spill-free: 64 rows on 16
# subgroups, 128 on 32, 32 on 8 (4 rows per subgroup). 64 rows on 8 subgroups
# spills 4.6 KB, with dv_chunks 2 x dv_split 2 0.8 KB. Not timed on a card
# yet: the one-card bench (benchmarks/kernels/benchmark_xpu_int8_prefill_attention.py
# --sweep) picks among these.
_LAUNCH = Int8PrefillLaunch()
# Launch options of the XPU backend only (emptied by the interpreter tests).
_GRF_MODE = {"grf_mode": "256"}

_logged_first_launch: set[tuple] = set()


def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def tile_block_n(launch: Int8PrefillLaunch, page: int) -> int:
    """KV tile size for a cache block of `page` slots: the largest power of
    two <= launch.block_n that divides `page` (a tile never crosses a block)."""
    block_n = launch.block_n
    while block_n > 1 and page % block_n:
        block_n //= 2
    return block_n


def check_launch(
    launch: Int8PrefillLaunch, head_dim: int, pv_int8: bool, page: int | None = None
) -> None:
    """Raise ValueError unless `launch` tiles `head_dim` (and a cache block of
    `page` slots) legally."""
    cols = launch.dv_chunks * launch.dv_split
    block_n = launch.block_n if page is None else tile_block_n(launch, page)
    problems = []
    if launch.dv_chunks not in (1, 2, 4):
        problems.append("dv_chunks must be 1, 2 or 4")
    if not _pow2(launch.dv_split) or head_dim % cols:
        problems.append(f"dv_chunks * dv_split must divide head_dim {head_dim}")
    elif head_dim // cols < 16 or not _pow2(head_dim // cols):
        problems.append("output chunks must be powers of two >= 16")
    if not _pow2(launch.qk_chunk) or launch.qk_chunk < 32 or head_dim % launch.qk_chunk:
        problems.append("qk_chunk must be a power of two >= 32 dividing head_dim")
    if not _pow2(launch.block_m) or launch.block_m < 8:
        problems.append("block_m must be a power of two >= 8")
    if not _pow2(block_n) or block_n < (32 if pv_int8 else 16):
        problems.append(
            f"block_n {block_n} (cache block {page}) must be a power of two >= "
            "32 (int8 P.V) or 16"
        )
    if problems:
        raise ValueError(
            f"int8 prefill attention launch {launch}: {'; '.join(problems)}"
        )


@triton.jit(
    do_not_specialize=["stride_q_row", "stride_q_head"],
    do_not_specialize_on_alignment=["q_ptr"],
)
def _quant_query_kernel(
    q_ptr,
    q_i8_ptr,
    q_scale_ptr,
    stride_q_row,
    stride_q_head,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
) -> None:
    """Symmetric int8 Q, one fp32 scale per (token, query head), with the
    per-token-head KV writer's rounding. Writes contiguous [rows, heads,
    head_dim] int8 and [rows, heads] fp32."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    dims = tl.arange(0, HEAD_DIM)
    x = tl.load(q_ptr + row * stride_q_row + head * stride_q_head + dims).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1.0e-10)
    y = x * (1.0 / scale)
    y = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    out_row = row * NUM_HEADS + head
    tl.store(q_i8_ptr + out_row * HEAD_DIM + dims, y.to(tl.int8))
    tl.store(q_scale_ptr + out_row, scale)


def quantize_query_int8(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(int8 Q [rows, heads, head_dim], fp32 scale [rows, heads]) of a BF16/FP16
    query whose last dimension is contiguous."""
    rows, heads, head_dim = query.shape
    assert query.stride(2) == 1
    q_i8 = torch.empty((rows, heads, head_dim), dtype=torch.int8, device=query.device)
    q_scale = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
    if rows:
        _quant_query_kernel[(rows, heads)](
            query,
            q_i8,
            q_scale,
            query.stride(0),
            query.stride(1),
            NUM_HEADS=heads,
            HEAD_DIM=head_dim,
            num_warps=1,
        )
    return q_i8, q_scale


@triton.jit
def _pv_chunk(pw, v_desc, col0, p_scale, PV_INT8: tl.constexpr):
    # One [BLOCK_M, DVC] output chunk of P.V over this tile's KV slots.
    v_chunk = v_desc.load([0, col0])
    if PV_INT8:
        return tl.dot(pw, v_chunk).to(tl.float32) * p_scale[:, None]
    else:
        return tl.dot(pw, v_chunk.to(tl.bfloat16))


@triton.jit
def _store_chunk(o_desc, row0, col0, acc, inv_l):
    # Rows past this sequence's last query (the next sequence's) are outside
    # the descriptor and not written.
    o_desc.store([row0, col0], (acc * inv_l[:, None]).to(o_desc.dtype))


@triton.jit(do_not_specialize=["num_seqs"])
def _int8_prefill_attn_kernel(
    q_ptr,
    q_scale_ptr,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    bt_ptr,
    cu_q_ptr,
    seq_lens_ptr,
    out_ptr,
    qk_scale,
    stride_q_row,
    stride_q_head,
    stride_qs_row,
    stride_k_blk,
    stride_k_slot,
    stride_k_head,
    stride_v_blk,
    stride_v_slot,
    stride_v_head,
    stride_ks_blk,
    stride_ks_slot,
    stride_ks_head,
    stride_vs_blk,
    stride_vs_slot,
    stride_vs_head,
    stride_bt,
    stride_o_row,
    stride_o_head,
    num_seqs,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QK_DC: tl.constexpr,
    DVC: tl.constexpr,
    NUM_DVC: tl.constexpr,
    DV_SPLIT: tl.constexpr,
    PV_INT8: tl.constexpr,
) -> None:
    pid_h = tl.program_id(0)
    q_block = tl.program_id(1)
    head = pid_h // DV_SPLIT
    dv0 = (pid_h % DV_SPLIT) * (DVC * NUM_DVC)
    kv_head = head // GROUP

    seq = find_seq_idx(cu_q_ptr, q_block, num_seqs, BLOCK_M, True)
    q_start = tl.load(cu_q_ptr + seq)
    q_len = tl.load(cu_q_ptr + seq + 1) - q_start
    local = q_block - (q_start // BLOCK_M + seq)
    if local * BLOCK_M >= q_len:
        return
    seq_len = tl.load(seq_lens_ptr + seq)
    # The new tokens are the last q_len of the sequence, already in the cache.
    ctx = seq_len - q_len

    offs_m = local * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < q_len
    row0 = q_start + local * BLOCK_M
    rows_end = q_start + q_len
    q_pos = ctx + offs_m
    # q_scale * softmax_scale * log2(e), per query row.
    row_scale = (
        tl.load(
            q_scale_ptr + (q_start + offs_m).to(tl.int64) * stride_qs_row + head,
            mask=m_mask,
            other=0.0,
        )
        * qk_scale
    )
    # Causal: this block's last query sees keys [0, kv_end).
    kv_end = ctx + tl.minimum(q_len, (local + 1) * BLOCK_M)

    offs_n0 = tl.arange(0, BLOCK_N)
    bt_row = bt_ptr + seq.to(tl.int64) * stride_bt
    # Query rows past this sequence read as zeros.
    q_desc = tl.make_tensor_descriptor(
        q_ptr + head * stride_q_head,
        shape=[rows_end, HEAD_DIM],
        strides=[stride_q_row, 1],
        block_shape=[BLOCK_M, QK_DC],
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_M, DVC], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, DVC], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_M, DVC], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_M, DVC], dtype=tl.float32)

    for start in range(0, kv_end, BLOCK_N):
        # The tile is slots [slot0, slot0 + BLOCK_N) of one cache block: 2D
        # blocks of BLOCK_N rows at the slot stride. Slots past kv_end hold
        # stale but finite codes and are masked below.
        page = tl.load(bt_row + start // PAGE).to(tl.int64)
        slot0 = start % PAGE
        k_desc = tl.make_tensor_descriptor(
            k_ptr
            + page * stride_k_blk
            + slot0 * stride_k_slot
            + kv_head * stride_k_head,
            shape=[BLOCK_N, HEAD_DIM],
            strides=[stride_k_slot, 1],
            block_shape=[BLOCK_N, QK_DC],
        )
        v_desc = tl.make_tensor_descriptor(
            v_ptr
            + page * stride_v_blk
            + slot0 * stride_v_slot
            + kv_head * stride_v_head,
            shape=[BLOCK_N, HEAD_DIM],
            strides=[stride_v_slot, 1],
            block_shape=[BLOCK_N, DVC],
        )
        offs_n = start + offs_n0
        n_ok = offs_n < kv_end

        # Q.K: int8 x int8 -> int32, head-dim chunk by chunk (exact).
        acc_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        for c in range(HEAD_DIM // QK_DC):
            q_chunk = q_desc.load([row0, c * QK_DC])
            k_chunk = k_desc.load([0, c * QK_DC])
            acc_qk = tl.dot(q_chunk, tl.trans(k_chunk), acc=acc_qk)
        # Masked loads keep stale scales past the sequence end out (NaN * 0).
        k_scale = tl.load(
            ks_ptr
            + page * stride_ks_blk
            + (slot0 + offs_n0) * stride_ks_slot
            + kv_head * stride_ks_head,
            mask=n_ok,
            other=0.0,
        )
        scores = acc_qk.to(tl.float32) * row_scale[:, None] * k_scale[None, :]
        allowed = n_ok[None, :] & (offs_n[None, :] <= q_pos[:, None])
        scores = tl.where(allowed, scores, float("-inf"))

        # Online softmax, base 2. Key 0 is visible to every row, so the first
        # tile makes every row maximum finite.
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        v_scale = tl.load(
            vs_ptr
            + page * stride_vs_blk
            + (slot0 + offs_n0) * stride_vs_slot
            + kv_head * stride_vs_head,
            mask=n_ok,
            other=0.0,
        )
        # V dequantization folded into P (after the row sum). Masked slots
        # have P = 0, so their (stale but finite) V codes add nothing.
        pv = p * v_scale[None, :]
        if PV_INT8:
            p_scale = tl.maximum(tl.max(pv, axis=1) / 127.0, 1.0e-30)
            pw = tl.minimum(tl.floor(pv * (1.0 / p_scale)[:, None] + 0.5), 127.0).to(
                tl.int8
            )
        else:
            p_scale = l_i  # unused
            pw = pv.to(tl.bfloat16)

        acc0 = acc0 * alpha[:, None] + _pv_chunk(pw, v_desc, dv0, p_scale, PV_INT8)
        if NUM_DVC > 1:
            acc1 = acc1 * alpha[:, None] + _pv_chunk(
                pw, v_desc, dv0 + DVC, p_scale, PV_INT8
            )
        if NUM_DVC > 2:
            acc2 = acc2 * alpha[:, None] + _pv_chunk(
                pw, v_desc, dv0 + 2 * DVC, p_scale, PV_INT8
            )
            acc3 = acc3 * alpha[:, None] + _pv_chunk(
                pw, v_desc, dv0 + 3 * DVC, p_scale, PV_INT8
            )

    inv_l = 1.0 / tl.where(l_i > 0.0, l_i, 1.0)
    o_desc = tl.make_tensor_descriptor(
        out_ptr + head * stride_o_head,
        shape=[rows_end, HEAD_DIM],
        strides=[stride_o_row, 1],
        block_shape=[BLOCK_M, DVC],
    )
    _store_chunk(o_desc, row0, dv0, acc0, inv_l)
    if NUM_DVC > 1:
        _store_chunk(o_desc, row0, dv0 + DVC, acc1, inv_l)
    if NUM_DVC > 2:
        _store_chunk(o_desc, row0, dv0 + 2 * DVC, acc2, inv_l)
        _store_chunk(o_desc, row0, dv0 + 3 * DVC, acc3, inv_l)


def _kernel_args(
    q_i8: torch.Tensor,
    q_scale: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    num_seqs: int,
) -> tuple:
    return (
        q_i8,
        q_scale,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        block_table,
        query_start_loc,
        seq_lens,
        output,
        softmax_scale * 1.4426950408889634,
        q_i8.stride(0),
        q_i8.stride(1),
        q_scale.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        num_seqs,
    )


def _kernel_constexprs(
    launch: Int8PrefillLaunch,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page: int,
    pv_int8: bool,
) -> dict:
    return dict(
        GROUP=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        PAGE=page,
        BLOCK_M=launch.block_m,
        BLOCK_N=tile_block_n(launch, page),
        QK_DC=launch.qk_chunk,
        DVC=head_dim // (launch.dv_chunks * launch.dv_split),
        NUM_DVC=launch.dv_chunks,
        DV_SPLIT=launch.dv_split,
        PV_INT8=pv_int8,
        num_warps=launch.num_warps,
        num_stages=1,
        **({"grf_mode": launch.grf_mode} if _GRF_MODE else {}),
    )


def int8_prefill_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    pv_int8: bool,
    launch: Int8PrefillLaunch | None = None,
) -> torch.Tensor:
    """Causal attention of `query` [tokens, heads, head_dim] (BF16/FP16) over
    the int8_per_token_head split-layout views (`split_layout_views`), each
    sequence's new tokens being the last of its `seq_lens` and already in the
    cache. Writes `output` [tokens, heads, head_dim] (last dimension
    contiguous); device-only (no host reads)."""
    launch = launch or _LAUNCH
    num_tokens, num_heads, head_dim = query.shape
    _, page, num_kv_heads, _ = key_cache.shape
    num_seqs = query_start_loc.numel() - 1
    check_launch(launch, head_dim, pv_int8, page)
    if num_heads % num_kv_heads:
        raise ValueError(f"{num_heads} query heads over {num_kv_heads} KV heads")
    if output.stride(-1) != 1:
        raise ValueError("int8 prefill attention needs a contiguous output head")
    if num_tokens == 0 or num_seqs <= 0:
        return output
    key = (launch, pv_int8, head_dim, num_heads, num_kv_heads, page)
    if key not in _logged_first_launch:
        _logged_first_launch.add(key)
        logger.info(
            "INT8 prefill attention on XPU, first launch: Q.K int8 x int8 -> "
            "int32, P.V %s; %d query / %d KV heads, head_dim %d, cache block %d "
            "(KV tiles of %d), launch %s.",
            "int8 x int8 -> int32" if pv_int8 else "BF16 x BF16 (V codes widened)",
            num_heads,
            num_kv_heads,
            head_dim,
            page,
            tile_block_n(launch, page),
            tuple(launch),
        )
    q_i8, q_scale = quantize_query_int8(query)
    grid = (
        num_heads * launch.dv_split,
        num_tokens // launch.block_m + num_seqs,
    )
    _int8_prefill_attn_kernel[grid](
        *_kernel_args(
            q_i8,
            q_scale,
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            block_table,
            query_start_loc,
            seq_lens,
            output,
            softmax_scale,
            num_seqs,
        ),
        **_kernel_constexprs(launch, num_heads, num_kv_heads, head_dim, page, pv_int8),
    )
    return output


def warmup_int8_prefill_attention(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    num_heads: int,
    dtype: torch.dtype,
    pv_int8: bool,
    launch: Int8PrefillLaunch | None = None,
) -> None:
    """Compile the query quantization and the attention kernel for this
    layer's cache and block-table strides. The attention launch has one
    sequence with no query tokens, so its only program returns at once and
    nothing in the cache is read."""
    launch = launch or _LAUNCH
    head_dim = key_cache.shape[-1]
    _, page, num_kv_heads, _ = key_cache.shape
    check_launch(launch, head_dim, pv_int8, page)
    dev = key_cache.device
    query = torch.zeros((1, num_heads, head_dim), dtype=dtype, device=dev)
    q_i8, q_scale = quantize_query_int8(query)
    output = torch.empty_like(query)
    cu_q = torch.zeros(2, dtype=torch.int32, device=dev)
    seq_lens = torch.zeros(1, dtype=torch.int32, device=dev)
    _int8_prefill_attn_kernel[(num_heads * launch.dv_split, 1)](
        *_kernel_args(
            q_i8,
            q_scale,
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            block_table,
            cu_q,
            seq_lens,
            output,
            1.0,
            1,
        ),
        **_kernel_constexprs(launch, num_heads, num_kv_heads, head_dim, page, pv_int8),
    )


__all__ = [
    "INT8_PREFILL_DOT_MODES",
    "Int8PrefillLaunch",
    "check_launch",
    "int8_prefill_attention",
    "quantize_query_int8",
    "tile_block_n",
    "warmup_int8_prefill_attention",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# The split-K HC down projection follows wu1ff's B70-LLM-Controller, used here
# under the MIT License (see the notice in hyperconnection.py).
"""Batch-invariant HC down projection for the XPU Qwen4-Exp model.

The HC down GEMM is skinny (K = hc * n_embd = 10240, N = 336 or 320). For up
to ``PARALLEL_MAX_ROWS`` rows it runs as BF16 tile products over fixed K tiles,
summed in FP32 in a fixed order: the result of a row does not depend on how many
other rows share the batch, and it repeats run to run. ``hc_silu`` of the
low-rank columns is folded into the finish kernel, rounded from the stored BF16
value exactly as ``hc_silu`` rounds it.

The HC up projection plus the gated mix can run as one native kernel
(``libastra_hc.so``, shipped beside the vllm-xpu-kernels extension) for 1-8 rows.
"""

from pathlib import Path

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Rows up to which the down projection takes the split-K route. Above it the
# stock GEMM runs (the ordered serial-K GEMM is several times slower there).
PARALLEL_MAX_ROWS = 32
_HC_INNER = 10240


@triton.jit
def _hc_partials(
    x,
    weight,
    partials,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    VALID_N: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
):
    rows = tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    kk = tl.program_id(0) * 128 + tl.arange(0, 128)
    a = tl.load(
        x + rows[:, None] * XS + kk[None, :],
        (rows[:, None] < M) & (kk[None, :] < K),
        other=0,
    )
    b = tl.load(
        weight + cols[None, :] * WS + kk[:, None],
        (cols[None, :] < VALID_N) & (kk[:, None] < K),
        other=0,
    )
    value = tl.dot(a, b)
    tl.store(
        partials + tl.program_id(0) * M * N + rows[:, None] * N + cols[None, :],
        value,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _hc_partials_batched(
    x,
    weight,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    VALID: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(2) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.program_id(0) * BK + tl.arange(0, BK)
    a = tl.load(
        x + rows[:, None] * XS + kk[None, :],
        (rows[:, None] < M) & (kk[None, :] < K),
        other=0,
    )
    b = tl.load(
        weight + cols[None, :] * WS + kk[:, None],
        (cols[None, :] < VALID) & (kk[:, None] < K),
        other=0,
    )
    y = tl.dot(a, b)
    tl.store(
        out + tl.program_id(0) * M * N + rows[:, None] * N + cols[None, :],
        y,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _hc_finish_silu(
    partials,
    output,
    lora_out,
    M: tl.constexpr,
    N: tl.constexpr,
    TILES: tl.constexpr,
    LORA: tl.constexpr,
    HC: tl.constexpr,
):
    """Sum the K tiles in order, store BF16, and write silu(v / HC) of the first
    LORA columns computed from the stored value."""
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < M * N
    value = tl.full((256,), 0, tl.float32)
    for step in range(TILES):
        value += tl.load(partials + step * M * N + offsets, mask, other=0)
    stored = value.to(output.dtype.element_ty)
    tl.store(output + offsets, stored, mask)
    row = offsets // N
    col = offsets % N
    v = stored.to(tl.float32) / HC
    y = v * tl.sigmoid(v)
    tl.store(lora_out + row * LORA + col, y, mask & (col < LORA))


def takes_parallel_route(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``parallel_hc_gemm_silu`` serves this down projection."""
    rows, inner = x.shape
    return (
        (1 <= rows <= 8 or (9 <= rows <= PARALLEL_MAX_ROWS and inner == _HC_INNER))
        and x.dtype == weight.dtype == torch.bfloat16
        and x.stride(1) == 1
        and weight.stride(1) == 1
        and inner == weight.shape[1]
    )


def parallel_hc_gemm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    valid_columns: int,
    lora_rank: int,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x @ weight[:valid_columns].T`` (columns past ``valid_columns`` are 0)
    and ``hc_silu`` of its first ``lora_rank`` columns.

    Returns ``(out [rows, columns], silu_lora [rows, lora_rank])``."""
    rows, columns = x.shape[0], weight.shape[0]
    assert 0 < lora_rank <= valid_columns <= columns
    partials, tiles = _parallel_partials(x, weight, valid_columns)
    output = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    lora = torch.empty((rows, lora_rank), dtype=x.dtype, device=x.device)
    _hc_finish_silu[(triton.cdiv(rows * columns, 256),)](
        partials, output, lora, rows, columns, tiles, lora_rank, hc_count, num_warps=4
    )
    return output, lora


# Folding the HC up projection's activation quant into the finish kernel. The
# int8 rows and scales are those of ``ops.scaled_int8_quant`` on the stored BF16
# lora, the per-token rule of vllm-xpu-kernels' int8_quant_math.h written out
# step for step (it is the rule of ``xpu_ops.dynamic_per_token_int8_quant_ref``):
#   scale = bf16_rne(max(absmax / 127, 1e-5))   IEEE division, RNE in integers
#   q     = clamp(round_half_even(x / scale), -128, 127)
# The rounding does not trust the quotient at a tie. The native kernel's note:
# on Xe2 quotients that are exactly q + 0.5 came out just under it, which is
# how an earlier Triton fold that rounded the quotient missed by one on a few
# elements per tensor. As in quantize_one, the quotient only picks f = floor;
# x against (f + 0.5) * scale (exact in fp32 for BF16 x and scale) decides.


@triton.jit
def _i8_scale_bf16(amax):
    s = tl.div_rn(amax, 127.0)
    # torch.clamp(min=1e-5) keeps NaN; (NaN < 1e-5) is False, so does this.
    s = tl.where(s < 1e-5, 1e-5, s)
    u = s.to(tl.uint32, bitcast=True)
    u = u + 0x7FFF + ((u >> 16) & 1)
    r = ((u >> 16) << 16).to(tl.float32, bitcast=True)
    return tl.where(s != s, s, r)


@triton.jit
def _i8_quantize(x, scale):
    f = tl.floor(tl.div_rn(x, scale))
    boundary = (f + 0.5) * scale
    half_f = f * 0.5
    tie = tl.where(tl.floor(half_f) == half_f, f, f + 1.0)
    q = tl.where(x > boundary, f + 1.0, tl.where(x < boundary, f, tie))
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    return q.to(tl.int8)


@triton.jit
def _hc_finish_silu_quant(
    partials,
    output,
    q_out,
    s_out,
    M: tl.constexpr,
    N: tl.constexpr,
    TILES: tl.constexpr,
    LORA: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``_hc_finish_silu`` for one row per program, with the per-token int8
    quant of the BF16 lora in place of the lora store: the same tile-sum order,
    BF16 store and silu per element, then the quant rule above over the row."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    offsets = row * N + cols
    value = tl.full((BLOCK,), 0, tl.float32)
    for step in range(TILES):
        value += tl.load(partials + step * M * N + offsets, mask, other=0)
    stored = value.to(output.dtype.element_ty)
    tl.store(output + offsets, stored, mask)
    v = stored.to(tl.float32) / HC
    y = v * tl.sigmoid(v)
    lmask = cols < LORA
    x = y.to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.where(lmask, tl.abs(x), 0.0), axis=0)
    scale = _i8_scale_bf16(amax)
    tl.store(q_out + row * LORA + cols, _i8_quantize(x, scale), lmask)
    tl.store(s_out + row, scale)


def parallel_hc_gemm_silu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    valid_columns: int,
    lora_rank: int,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``parallel_hc_gemm_silu`` with the lora returned as the int8 rows and fp32
    ``[rows, 1]`` scales ``ops.scaled_int8_quant`` makes of it.

    Returns ``(out [rows, columns], q [rows, lora_rank], scale [rows, 1])``."""
    rows, columns = x.shape[0], weight.shape[0]
    assert 0 < lora_rank <= valid_columns <= columns
    partials, tiles = _parallel_partials(x, weight, valid_columns)
    output = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    q = torch.empty((rows, lora_rank), dtype=torch.int8, device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _hc_finish_silu_quant[(rows,)](
        partials,
        output,
        q,
        scale,
        rows,
        columns,
        tiles,
        lora_rank,
        hc_count,
        triton.next_power_of_2(columns),
        num_warps=4,
    )
    return output, q, scale


@triton.jit
def _hc_up_mix_i8_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    x_ptr,
    y_ptr,
    M,
    K,
    stride_am,
    stride_bk,
    stride_bn,
    stride_x,
    stride_y,
    SB_PER_COL: tl.constexpr,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """The INT8 up GEMM (``triton_scaled_mm``'s int32 accumulation and its
    epilogue ``(scale_a * acc) * scale_b`` rounded to BF16) with the gated mix
    of ``_hc_gate_mix_kernel`` in the epilogue: per output column the HC
    streams' GEMM tiles in stream order, ``sigmoid(gate) * x`` summed in FP32,
    divided by HC. The up output never goes to memory."""
    HC_DIM: tl.constexpr = DIM // HC
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    rmask = rows < M
    cmask = cols < HC_DIM
    mask = rmask[:, None] & cmask[None, :]
    scale_a = tl.load(sa_ptr + rows, rmask)[:, None]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for stream in tl.static_range(HC):
        n = stream * HC_DIM + cols
        dot = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            kk = k * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
            kmask = kk < K
            a = tl.load(
                a_ptr + rows[:, None] * stride_am + kk[None, :],
                rmask[:, None] & kmask[None, :],
                other=0,
            )
            b = tl.load(
                b_ptr + kk[:, None] * stride_bk + n[None, :] * stride_bn,
                kmask[:, None] & cmask[None, :],
                other=0,
            )
            dot = tl.dot(a, b, dot, out_dtype=tl.int32)
        if SB_PER_COL:  # noqa: SIM108 (constexpr branch)
            scale_b = tl.load(sb_ptr + n, cmask)[None, :]
        else:
            scale_b = tl.load(sb_ptr)
        gate = scale_a * dot.to(tl.float32)
        gate = scale_b * gate.to(tl.float32)
        g = gate.to(tl.bfloat16)
        x = tl.load(x_ptr + rows[:, None] * stride_x + n[None, :], mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc /= HC
    tl.store(y_ptr + rows[:, None] * stride_y + cols[None, :], acc, mask)


# Rows up to which the fused up + gated mix runs; it stays on the rows where
# the stock INT8 linear runs its Triton GEMM, whose bits it reproduces.
UP_MIX_I8_MAX_ROWS = 64


def int8_up_mix(
    q: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    xn: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """``hc_gate_mix(xn, triton_scaled_mm(q, weight, scale, weight_scale), hc)``
    in one kernel. ``weight`` is the INT8 linear's ``[K, N]`` view, ``scale``
    the per-token ``[rows, 1]`` fp32 scales, ``weight_scale`` per column or
    one scalar."""
    rows, inner = q.shape
    dim = weight.shape[1]
    assert xn.shape == (rows, dim) and dim % hc_count == 0
    assert q.stride(1) == 1 and xn.stride(1) == 1 and rows <= UP_MIX_I8_MAX_ROWS
    out = torch.empty((rows, dim // hc_count), dtype=xn.dtype, device=xn.device)
    if rows == 0:
        return out
    block_n = 64
    block_m = max(16, triton.next_power_of_2(rows))
    per_col = weight_scale.numel() != 1
    _hc_up_mix_i8_kernel[(triton.cdiv(dim // hc_count, block_n), 1)](
        q,
        weight,
        scale.reshape(-1),
        weight_scale.reshape(-1),
        xn,
        out,
        rows,
        inner,
        q.stride(0),
        weight.stride(0),
        weight.stride(1),
        xn.stride(0),
        out.stride(0),
        per_col,
        dim,
        hc_count,
        block_m,
        block_n,
        128,
        num_warps=4,
    )
    return out


# The same GEMM and mix as _hc_up_mix_i8_kernel, laid out for the decode rows.
# That kernel loads its operands through pointer tensors: the [BLOCK_K, BLOCK_N]
# int64 address and mask tensors of every weight tile stay live across the HC
# streams' unrolled K loops, the tiles reach the DPAS layout through shared
# local memory, and the grid is 2560 / 64 = 40 work-groups for 32 Xe-cores.
# Here the int8 tiles come in through tensor descriptors (2D block loads
# straight into the DPAS operand layout, as triton_scaled_mm loads them on
# XPU); the four streams' weight tiles of one K step are loaded together into
# four int32 accumulators, so a work-group keeps four loads in flight; and the
# column tile is narrow, so the grid is 2560 / BLOCK_N work-groups.
#
# The bits are those of the unfused path for the same reason as before: the
# int32 dot is exact, so the K tiling does not reach the result; the epilogue
# is triton_scaled_mm's ((scale_a * acc) * scale_b, rounded to BF16) and the
# mix kernel's (sigmoid of the BF16 gate times x, summed from zero in stream
# order in FP32, divided by HC, stored BF16).


@triton.jit
def _hc_up_mix_term(
    dot,
    scale_a,
    sb_ptr,
    x_ptr,
    rows,
    cols,
    n0,
    stride_x,
    mask,
    cmask,
    SB_PER_COL: tl.constexpr,
):
    """One stream's ``sigmoid(bf16(scale_b * (scale_a * dot))) * x`` in FP32."""
    n = n0 + cols
    if SB_PER_COL:  # noqa: SIM108 (constexpr branch)
        scale_b = tl.load(sb_ptr + n, cmask)[None, :]
    else:
        scale_b = tl.load(sb_ptr)
    gate = scale_a * dot.to(tl.float32)
    gate = scale_b * gate
    g = gate.to(tl.bfloat16)
    x = tl.load(x_ptr + rows[:, None] * stride_x + n[None, :], mask, other=0.0)
    return tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)


@triton.jit
def _hc_up_mix_i8_td_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    x_ptr,
    y_ptr,
    M,
    K,
    stride_am,
    stride_b,
    stride_x,
    stride_y,
    SB_PER_COL: tl.constexpr,
    B_T: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``hc_gate_mix(x, triton_scaled_mm(a, b, sa, sb), 4)`` for HC = 4 streams.
    ``b`` is the ``[K, DIM]`` int8 weight: with ``B_T`` a view of ``[DIM, K]``
    rows (``stride_b`` is the row stride of those rows), else ``K`` rows of
    ``DIM`` (``stride_b`` their stride). Program ``(n, m)`` writes output
    columns ``[n * BLOCK_N, (n + 1) * BLOCK_N)`` of rows ``[m * BLOCK_M, ...)``."""
    HC: tl.constexpr = 4
    HC_DIM: tl.constexpr = DIM // HC
    m0 = tl.program_id(1) * BLOCK_M
    c0 = tl.program_id(0) * BLOCK_N
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    if B_T:
        b_desc = tl.make_tensor_descriptor(
            b_ptr, shape=[DIM, K], strides=[stride_b, 1], block_shape=[BLOCK_N, BLOCK_K]
        )
    else:
        b_desc = tl.make_tensor_descriptor(
            b_ptr, shape=[K, DIM], strides=[stride_b, 1], block_shape=[BLOCK_K, BLOCK_N]
        )
    d0 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    d1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    d2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    d3 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K
        a = a_desc.load([m0, kk])
        if B_T:
            b0 = tl.trans(b_desc.load([c0, kk]))
            b1 = tl.trans(b_desc.load([HC_DIM + c0, kk]))
            b2 = tl.trans(b_desc.load([2 * HC_DIM + c0, kk]))
            b3 = tl.trans(b_desc.load([3 * HC_DIM + c0, kk]))
        else:
            b0 = b_desc.load([kk, c0])
            b1 = b_desc.load([kk, HC_DIM + c0])
            b2 = b_desc.load([kk, 2 * HC_DIM + c0])
            b3 = b_desc.load([kk, 3 * HC_DIM + c0])
        d0 = tl.dot(a, b0, d0, out_dtype=tl.int32)
        d1 = tl.dot(a, b1, d1, out_dtype=tl.int32)
        d2 = tl.dot(a, b2, d2, out_dtype=tl.int32)
        d3 = tl.dot(a, b3, d3, out_dtype=tl.int32)
    rows = m0 + tl.arange(0, BLOCK_M).to(tl.int64)
    cols = c0 + tl.arange(0, BLOCK_N).to(tl.int64)
    rmask = rows < M
    cmask = cols < HC_DIM
    mask = rmask[:, None] & cmask[None, :]
    scale_a = tl.load(sa_ptr + rows, rmask)[:, None]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc += _hc_up_mix_term(
        d0, scale_a, sb_ptr, x_ptr, rows, cols, 0, stride_x, mask, cmask, SB_PER_COL
    )
    acc += _hc_up_mix_term(
        d1,
        scale_a,
        sb_ptr,
        x_ptr,
        rows,
        cols,
        HC_DIM,
        stride_x,
        mask,
        cmask,
        SB_PER_COL,
    )
    acc += _hc_up_mix_term(
        d2,
        scale_a,
        sb_ptr,
        x_ptr,
        rows,
        cols,
        2 * HC_DIM,
        stride_x,
        mask,
        cmask,
        SB_PER_COL,
    )
    acc += _hc_up_mix_term(
        d3,
        scale_a,
        sb_ptr,
        x_ptr,
        rows,
        cols,
        3 * HC_DIM,
        stride_x,
        mask,
        cmask,
        SB_PER_COL,
    )
    acc /= HC
    tl.store(y_ptr + rows[:, None] * stride_y + cols[None, :], acc, mask)


# (BLOCK_N, BLOCK_K, num_warps) of the fast kernel; VLLM_XPU_HC_UP_TILE
# ("bn,bk,warps") overrides it for a measurement. 2560 / 16 = 160 work-groups,
# five per Xe-core on the B70. At 64 rows it takes 8 warps: with 4 the
# compiled kernel spills (2,240 B per thread for Xe2), with 8 it keeps the
# 192 B of the smaller tiles.
FAST_UP_TILE = (16, 128, 4)
FAST_UP_WARPS_64 = 8
_TD_DEVICES: set = set()


def _fast_up_tile() -> tuple[int, int, int]:
    spec = envs.VLLM_XPU_HC_UP_TILE.strip()
    if not spec:
        return FAST_UP_TILE
    bn, bk, warps = (int(v) for v in spec.split(","))
    assert bn >= 16 and bk >= 32 and (bn & (bn - 1)) == 0 and (bk & (bk - 1)) == 0
    return bn, bk, warps


def takes_fast_up(q: torch.Tensor, weight: torch.Tensor, hc_count: int) -> bool:
    """Whether ``int8_up_mix_fast`` serves these operands: four streams and the
    tensor-descriptor rules ``triton_scaled_mm`` applies (unit inner strides,
    16-byte aligned rows and bases), plus 16-byte aligned stream starts along a
    weight's inner dimension when its rows run along N."""
    inner = q.shape[1]
    b_t = weight.stride(1) != 1 and weight.stride(0) == 1
    b_row = weight.stride(1) if b_t else weight.stride(0)
    return (
        hc_count == 4
        and (b_t or (weight.shape[1] // hc_count) % 16 == 0)
        and q.dtype == weight.dtype == torch.int8
        and q.stride(1) == 1
        and (weight.stride(1) == 1 or b_t)
        and (q.stride(0) * q.element_size()) % 16 == 0
        and (b_row * weight.element_size()) % 16 == 0
        and inner % 16 == 0
        and q.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def int8_up_mix_fast(
    q: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    xn: torch.Tensor,
    hc_count: int,
    tile: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """``int8_up_mix`` with tensor-descriptor loads, the four streams' tiles of a
    K step in flight together and a narrow column tile; the same bits.
    Operands it does not take (``takes_fast_up``) go to ``int8_up_mix``."""
    rows, inner = q.shape
    dim = weight.shape[1]
    assert xn.shape == (rows, dim) and dim % hc_count == 0
    assert xn.stride(1) == 1 and rows <= UP_MIX_I8_MAX_ROWS
    if not takes_fast_up(q, weight, hc_count):
        logger.warning_once(
            "HC up + gated mix: operands outside the fast kernel's layout "
            "(q %s stride %s, weight stride %s, hc %d); the first fused kernel runs.",
            tuple(q.shape),
            q.stride(),
            weight.stride(),
            hc_count,
        )
        return int8_up_mix(q, scale, weight, weight_scale, xn, hc_count)
    out = torch.empty((rows, dim // hc_count), dtype=xn.dtype, device=xn.device)
    if rows == 0:
        return out
    bm = triton.next_power_of_2(rows)
    if tile is None and not envs.VLLM_XPU_HC_UP_TILE.strip():
        bn, bk, warps = FAST_UP_TILE
        if bm >= 64:
            warps = FAST_UP_WARPS_64
    else:
        bn, bk, warps = tile or _fast_up_tile()
    b_t = weight.stride(1) != 1 and weight.stride(0) == 1
    if q.device.type != "cpu" and q.device not in _TD_DEVICES:
        from vllm.triton_utils.allocation import set_triton_allocator

        set_triton_allocator(q.device)
        _TD_DEVICES.add(q.device)
    logger.info_once(
        "HC up + gated mix: INT8 fast fused kernel (tensor-descriptor loads, four "
        "streams per K step), first launch: %d rows, tile (%d, %d, %d), %d warps.",
        rows,
        bm,
        bn,
        bk,
        warps,
    )
    _hc_up_mix_i8_td_kernel[(triton.cdiv(dim // hc_count, bn), triton.cdiv(rows, bm))](
        q,
        weight,
        scale.reshape(-1),
        weight_scale.reshape(-1),
        xn,
        out,
        rows,
        inner,
        q.stride(0),
        weight.stride(1) if b_t else weight.stride(0),
        xn.stride(0),
        out.stride(0),
        weight_scale.numel() != 1,
        b_t,
        dim,
        bm,
        bn,
        bk,
        num_warps=warps,
    )
    return out


def _parallel_partials(
    x: torch.Tensor, weight: torch.Tensor, valid_columns: int
) -> tuple[torch.Tensor, int]:
    """FP32 partials ``[tiles, rows, columns]`` of the split-K down GEMM."""
    rows, inner = x.shape
    columns = weight.shape[0]
    block_k = 64 if rows > 8 else 128
    tiles = triton.cdiv(inner, block_k)
    partials = torch.empty((tiles, rows, columns), dtype=torch.float32, device=x.device)
    if rows > 8:
        _hc_partials_batched[(tiles, triton.cdiv(columns, 32), triton.cdiv(rows, 16))](
            x,
            weight,
            partials,
            rows,
            columns,
            inner,
            valid_columns,
            x.stride(0),
            weight.stride(0),
            block_k,
            32,
            num_warps=4,
        )
    else:
        _hc_partials[(tiles, triton.cdiv(columns, 32))](
            x,
            weight,
            partials,
            rows,
            columns,
            inner,
            valid_columns,
            x.stride(0),
            weight.stride(0),
            num_warps=4,
        )
    return partials, tiles


_UP_ROWS = (1, 8)
_UP_SHAPE = (_HC_INNER, 320)
_up_op = None
_up_state = "unloaded"


def _load_up_op():
    global _up_op, _up_state
    if _up_state != "unloaded":
        return _up_op
    _up_state = "missing"
    try:
        import vllm_xpu_kernels

        lib = Path(vllm_xpu_kernels.__file__).with_name("libastra_hc.so")
        torch.ops.load_library(str(lib))
        _up_op = torch.ops.astra_hc_v3.up
        _up_state = "loaded"
        logger.info_once("HC up + gated mix: native kernel for 1-8 rows (%s).", lib)
    except Exception as e:  # noqa: BLE001 - optional native library
        logger.warning_once(
            "HC up + gated mix: native kernel unavailable (%s); the Triton path "
            "runs at every width.",
            e,
        )
    return _up_op


def takes_native_up(lora: torch.Tensor, up_weight: torch.Tensor) -> bool:
    return (
        _UP_ROWS[0] <= lora.shape[0] <= _UP_ROWS[1]
        and lora.device.type == "xpu"
        and up_weight.dtype == lora.dtype == torch.bfloat16
        and tuple(up_weight.shape) == _UP_SHAPE
        and _load_up_op() is not None
    )


def native_up_mix(
    lora: torch.Tensor, up_weight: torch.Tensor, normalized: torch.Tensor, n_embd: int
) -> torch.Tensor:
    """``hc_gate_mix(normalized, lora @ up_weight.T, hc)`` in one native kernel."""
    output = torch.empty((lora.shape[0], n_embd), device=lora.device, dtype=lora.dtype)
    _up_op(lora, up_weight.detach(), normalized, output)
    return output

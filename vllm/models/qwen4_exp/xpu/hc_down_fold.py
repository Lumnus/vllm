# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The INT8 HC down projection's activation quant folded into the kernel that
writes its input (``VLLM_XPU_HC_DOWN_FOLD``).

On a checkpoint with an INT8 HC down projection, ``xn`` (the grouped Gemma
RMSNorm of the hyper-connection state, after the pending combine) goes to
vllm-xpu-kernels' ``int8_dense_w8a8_fused_gemm``, which launches its own
per-token quant kernel and then the s8 x s8 GEMM. Here the kernel that writes
``xn`` also writes its int8 rows and fp32 scales, and the GEMM runs on the
pre-quantized entry ``int8_dense_w8a8_gemm``: two launches per module instead
of three.

The norm runs one program per (row, stream), as ``hc_combine_norm`` and
``grouped_gemma_rmsnorm`` do; a row's absmax spans its four streams. Each
program computes its stream's ``out`` and ``xn`` exactly as those kernels do
(the same code, tiles and warps), stores them, and adds one value to the row's
int64 arrival word with a single atomic: its stream's absmax (a bf16 value, so
15 bits) at bit ``15 * stream`` and 1 at bit 60. The atomic returns the word
as it was, so the program that brings the count to ``hc`` holds all ``hc``
absmax values; it alone writes the row's int8 codes and scale, then clears the
word for the next launch and for graph replay. No ordering between different
addresses is needed: the absmax values and the count travel in the same word.

The last program needs the whole row of ``xn``:
  1  (default) it recomputes the other streams' ``xn`` from the kernel's own
     inputs (read-only in the launch), with the same code, so it needs no
     memory ordering at all;
  2  it reads the other programs' stored ``xn``; the atomic is acq_rel at GPU
     scope. This relies on the backend making those stores visible through
     the release/acquire pair, which the CPU interpreter cannot show; the
     card test decides whether mode 2 is usable.

The rule (``int8_dense.cpp`` ``quant_rows_launch``, with IEEE division; it is
the calibration rule of the INT8 HC down pass):
  mx = max |x| over the bf16 row (NaN ignored), inv = mx > 0 ? 127 / mx : 0
  (IEEE), xs = mx * fp32(1 / 127), q = clamp(rint(x * inv), -127, 127),
  NaN -> -127.
The native op's device division is not compiled IEEE (its file does not get
the precise-division flag), so against today's device codes the fold moves a
rounding point on the elements where that division differs: a stated
contract, its own switch, judged against the BF16 compare base."""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# The arrival word: hc absmax fields of 15 bits, the count at bit 60.
FIELD_BITS = 15
COUNT_SHIFT = 60
MAX_HC = 4
# Identity: stream s of a row is the s-th program of that row. Tests launch
# the streams in other orders (2 bits per stream).
IDENTITY_ORDER = 0b11100100
# fp32(1 / 127) as the device folds the constant 1.0f / 127.0f (IEEE).
INV127_BITS = int(
    (torch.ones((), dtype=torch.float32) / torch.full((), 127.0)).view(torch.int32)
)

_logged_first = False


@triton.jit
def _q127(x, inv):
    """``clamp(rint(x * inv), -127, 127)`` with NaN -> -127, as int8. The
    product is rounded once and only compared (never subtracted), so FMA
    contraction cannot reach it; ``f + 0.5`` is exact for |f| < 2**22."""
    p = x * inv
    f = tl.floor(p)
    b = f + 0.5
    half = f * 0.5
    up = tl.where(tl.floor(half) == half, f, f + 1.0)
    r = tl.where(p > b, f + 1.0, tl.where(p < b, f, up))
    r = tl.where(r != r, -127.0, r)
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)
    return r.to(tl.int8)


@triton.jit
def _stored_abs(y, mask):
    """|x| of the bf16 values a tile stores, 0 where masked or NaN (the native
    kernel's fmax ignores NaN)."""
    yb = y.to(tl.bfloat16).to(tl.float32)
    return tl.where(mask & (yb == yb), tl.abs(yb), 0.0)


@triton.jit
def _bf16_bits(amax):
    """The bf16 bit pattern (int64, 15 bits) of a non-negative fp32 value that
    is a bf16 value."""
    return (amax.to(tl.int32, bitcast=True) >> 16).to(tl.int64)


@triton.jit
def _combine_norm_stream(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    row,
    stream,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STORE: tl.constexpr,
):
    """``_hc_combine_norm_kernel``'s body for one (row, stream), line for
    line (nvidia/ops/hc.py); returns the fp32 ``y`` it stores, its mask and
    offsets. ``STORE=False`` recomputes without storing."""
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    w_offs = offs_inner if W_SHARED else offs

    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    if inj_ptr is not None:
        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(
        block_ptr + row * stride_block + offs_inner,
        mask_inner,
        other=0.0,
    )
    if inj_ptr is not None:
        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
        block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)
    if inj_ptr is None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    if STORE:
        tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

    out = out.to(tl.float32)
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    if inj_ptr is not None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    if STORE:
        tl.store(y_ptr + row * stride_y + offs, y, mask_inner)
    return y, mask_inner, offs_inner


@triton.jit
def _norm_stream(
    x_ptr,
    w_ptr,
    y_ptr,
    row,
    group_id,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    STORE: tl.constexpr,
):
    """``_grouped_gemma_rmsnorm_kernel``'s body for one (row, group), line for
    line (nvidia/ops/hc.py)."""
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(GROUP_DIM)

    offs_g = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * GROUP_DIM + offs_g
    mask = offs_g < GROUP_DIM
    w_offs = offs_g if W_SHARED else offsets

    x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + w_offs, mask, other=0.0)

    rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    y = x * rrms
    y += y * w.to(tl.float32)

    if STORE:
        tl.store(y_ptr + row * stride_y + offsets, y, mask)
    return y, mask, offs_g


@triton.jit
def _arrive(word_ptr, row, stream, amax_bits, HC: tl.constexpr, SEM: tl.constexpr):
    """Add this stream's absmax and one arrival to the row's word; returns
    (whether this program is the row's last, the complete word)."""
    field = (amax_bits << (stream.to(tl.int64) * 15)) + (tl.full([], 1, tl.int64) << 60)
    old = tl.atomic_add(word_ptr + row, field, sem=SEM, scope="gpu")
    full = old + field
    last = ((full >> 60) & 7) == HC
    return last, full


@triton.jit
def _row_scale(full, HC: tl.constexpr, INV127: tl.constexpr):
    """The row's absmax from the complete word, and ``(inv, xs)``."""
    m = full & 0x7FFF
    for s in tl.static_range(1, HC):
        m = tl.maximum(m, (full >> (15 * s)) & 0x7FFF)
    amax = (m.to(tl.int32) << 16).to(tl.float32, bitcast=True)
    pos = amax > 0.0
    num = tl.full([], 127.0, tl.float32)
    inv = tl.where(pos, tl.div_rn(num, tl.where(pos, amax, 1.0)), 0.0)
    xs = amax * tl.full([], INV127, tl.int32).to(tl.float32, bitcast=True)
    return inv, xs


@triton.jit
def _hc_combine_norm_q_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    word_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    stride_q,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REREAD: tl.constexpr,
    SEM: tl.constexpr,
    ORDER: tl.constexpr,
    INV127: tl.constexpr,
):
    pid = tl.program_id(0)
    row = (pid // HC).to(tl.int64)
    stream = (ORDER >> (2 * (pid % HC))) & 3
    y, mask, offs_inner = _combine_norm_stream(
        block_ptr,
        res_ptr,
        inj_ptr,
        w_ptr,
        out_ptr,
        y_ptr,
        row,
        stream,
        stride_block,
        stride_res,
        stride_inj,
        stride_out,
        stride_y,
        HC_DIM,
        HC,
        W_SHARED,
        EPS,
        BLOCK_SIZE,
        True,
    )
    amax = tl.max(tl.max(_stored_abs(y, mask), axis=1), axis=0)
    last, full = _arrive(word_ptr, row, stream, _bf16_bits(amax), HC, SEM)
    if last:
        inv, xs = _row_scale(full, HC, INV127)
        for s in tl.static_range(HC):
            if REREAD:
                ys = tl.load(
                    y_ptr + row * stride_y + s * HC_DIM + offs_inner, mask, other=0.0
                ).to(tl.float32)
            else:
                yr, _, _ = _combine_norm_stream(
                    block_ptr,
                    res_ptr,
                    inj_ptr,
                    w_ptr,
                    out_ptr,
                    y_ptr,
                    row,
                    s,
                    stride_block,
                    stride_res,
                    stride_inj,
                    stride_out,
                    stride_y,
                    HC_DIM,
                    HC,
                    W_SHARED,
                    EPS,
                    BLOCK_SIZE,
                    False,
                )
                ys = yr.to(tl.bfloat16).to(tl.float32)
            tl.store(
                q_ptr + row * stride_q + s * HC_DIM + offs_inner, _q127(ys, inv), mask
            )
        tl.store(s_ptr + row, xs)
        tl.atomic_xchg(word_ptr + row, tl.zeros([], tl.int64), sem="relaxed")


@triton.jit
def _hc_norm_q_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    word_ptr,
    stride_x,
    stride_y,
    stride_q,
    DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    REREAD: tl.constexpr,
    SEM: tl.constexpr,
    ORDER: tl.constexpr,
    INV127: tl.constexpr,
):
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    pid = tl.program_id(0)
    row = (pid // NUM_GROUPS).to(tl.int64)
    group_id = (ORDER >> (2 * (pid % NUM_GROUPS))) & 3
    y, mask, offs_g = _norm_stream(
        x_ptr,
        w_ptr,
        y_ptr,
        row,
        group_id,
        stride_x,
        stride_y,
        DIM,
        NUM_GROUPS,
        W_SHARED,
        EPS,
        True,
    )
    amax = tl.max(_stored_abs(y, mask), axis=0)
    last, full = _arrive(word_ptr, row, group_id, _bf16_bits(amax), NUM_GROUPS, SEM)
    if last:
        inv, xs = _row_scale(full, NUM_GROUPS, INV127)
        for s in tl.static_range(NUM_GROUPS):
            if REREAD:
                ys = tl.load(
                    y_ptr + row * stride_y + s * GROUP_DIM + offs_g, mask, other=0.0
                ).to(tl.float32)
            else:
                yr, _, _ = _norm_stream(
                    x_ptr,
                    w_ptr,
                    y_ptr,
                    row,
                    s,
                    stride_x,
                    stride_y,
                    DIM,
                    NUM_GROUPS,
                    W_SHARED,
                    EPS,
                    False,
                )
                ys = yr.to(tl.bfloat16).to(tl.float32)
            tl.store(
                q_ptr + row * stride_q + s * GROUP_DIM + offs_g, _q127(ys, inv), mask
            )
        tl.store(s_ptr + row, xs)
        tl.atomic_xchg(word_ptr + row, tl.zeros([], tl.int64), sem="relaxed")


def new_arrival_words(rows: int, device: torch.device | str) -> torch.Tensor:
    """A zeroed arrival word per row; every launch leaves it zeroed."""
    return torch.zeros(max(int(rows), 1), dtype=torch.int64, device=device)


def _words_for(words: torch.Tensor, rows: int) -> torch.Tensor:
    if rows <= words.numel():
        return words
    # More rows than the buffer was sized for at load (it is sized to the
    # scheduler's token budget): a larger one, outside any graph capture.
    logger.warning_once(
        "XPU HC down fold: %d rows exceed the arrival-word buffer (%d); "
        "allocating a larger one.",
        rows,
        words.numel(),
    )
    return new_arrival_words(rows, words.device)


def _check(hc: int, mode: int) -> None:
    assert 1 <= hc <= MAX_HC, f"the arrival word holds up to {MAX_HC} streams"
    assert mode in (1, 2), mode


def _log_first(kind: str, rows: int, mode: int) -> None:
    global _logged_first
    if _logged_first:
        return
    _logged_first = True
    logger.info(
        "XPU HC down fold (VLLM_XPU_HC_DOWN_FOLD=%d), first launch: %d rows; "
        "%s kernel writes xn and its per-token int8 rows (IEEE 127/amax, the "
        "calibration rule), last stream of each row by one atomic arrival word "
        "(%s); INT8 down GEMM on int8_dense_w8a8_gemm.",
        mode,
        rows,
        kind,
        "recomputes the row" if mode == 1 else "re-reads xn under acq_rel",
    )


def combine_norm_q(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
    words: torch.Tensor,
    mode: int,
    order: int = IDENTITY_ORDER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``hc_combine_norm`` plus the int8 rows ``[N, DIM]`` and fp32 scales
    ``[N, 1]`` of its ``xn``: returns ``(out, xn, xq, xs)``."""
    _check(hc_count, mode)
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    if injection_logits is not None:
        assert injection_logits.shape == (N, hc_count)
        assert injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous()
    assert norm_weight.numel() in (hc_dim, DIM)
    stride_injection = injection_logits.stride(0) if injection_logits is not None else 0

    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    xq = torch.empty((N, DIM), dtype=torch.int8, device=residual.device)
    xs = torch.empty((N, 1), dtype=torch.float32, device=residual.device)
    if N == 0:
        return out, y, xq, xs
    words = _words_for(words, N)
    _log_first("the combine + norm", N, mode)
    _hc_combine_norm_q_kernel[(N * hc_count,)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        xq,
        xs,
        words,
        block_output.stride(0),
        residual.stride(0),
        stride_injection,
        out.stride(0),
        y.stride(0),
        xq.stride(0),
        hc_dim,
        hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        BLOCK_SIZE=512,
        REREAD=mode == 2,
        SEM="acq_rel" if mode == 2 else "relaxed",
        ORDER=order,
        INV127=INV127_BITS,
    )
    return out, y, xq, xs


def norm_q(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
    words: torch.Tensor,
    mode: int,
    order: int = IDENTITY_ORDER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``grouped_gemma_rmsnorm`` plus the int8 rows and fp32 ``[N, 1]``
    scales of its output: returns ``(xn, xq, xs)``."""
    _check(num_groups, mode)
    N, DIM = x.shape
    assert x.stride(1) == 1, "grouped Gemma RMSNorm requires unit inner stride"
    assert weight.is_contiguous(), "grouped Gemma RMSNorm weight must be contiguous"
    assert DIM % num_groups == 0
    group_dim = DIM // num_groups
    assert weight.numel() in (group_dim, DIM)

    y = x.new_empty(x.shape)
    xq = torch.empty((N, DIM), dtype=torch.int8, device=x.device)
    xs = torch.empty((N, 1), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, xq, xs
    words = _words_for(words, N)
    _log_first("the grouped norm", N, mode)
    _hc_norm_q_kernel[(N * num_groups,)](
        x,
        weight,
        y,
        xq,
        xs,
        words,
        x.stride(0),
        y.stride(0),
        xq.stride(0),
        DIM,
        num_groups,
        W_SHARED=weight.numel() == group_dim,
        EPS=eps,
        REREAD=mode == 2,
        SEM="acq_rel" if mode == 2 else "relaxed",
        ORDER=order,
        INV127=INV127_BITS,
    )
    return y, xq, xs


def gemm_available() -> bool:
    """Whether vllm-xpu-kernels has the pre-quantized INT8 dense GEMM."""
    return hasattr(torch.ops._xpu_C, "int8_dense_w8a8_gemm")


def int8_down_gemm_q(
    xq: torch.Tensor, xs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """The s8 x s8 GEMM of ``int8_dense_w8a8_fused_gemm`` on rows already
    quantized: ``xq`` int8 ``[T, K]``, ``xs`` fp32 ``[T, 1]``, ``weight`` int8
    ``[N, K]`` with fp32 row scales."""
    return torch.ops._xpu_C.int8_dense_w8a8_gemm(xq, xs, weight, weight_scale, None)

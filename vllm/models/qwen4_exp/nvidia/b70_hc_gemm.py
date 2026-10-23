# SPDX-License-Identifier: Apache-2.0
"""Fixed-order BF16 HC down projection with FP32 accumulation."""

import os
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton

_large_hc = None


@triton.jit(do_not_specialize=["M"])
def _ordered_hc_gemm(
    x,
    weight,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    VALID_N: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 128,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for step in range(triton.cdiv(K, BK)):
        kk = step * BK + inner
        a = tl.load(
            x + rows[:, None] * XS0 + kk[None, :] * XS1,
            (rows[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            weight + cols[None, :] * WS0 + kk[:, None] * WS1,
            (cols[None, :] < VALID_N) & (kk[:, None] < K),
            other=0,
        )
        accumulator += tl.dot(a, b)
    tl.store(
        output + rows[:, None] * N + cols[None, :],
        accumulator,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def ordered_hc_gemm(x, weight, valid_columns):
    global _large_hc
    assert x.dtype == weight.dtype == torch.bfloat16
    rows, inner = x.shape
    columns = weight.shape[0]
    assert inner == weight.shape[1] and 0 < valid_columns <= columns
    if (
        256 < rows <= 1024
        and inner == 10240
        and columns in (320, 336)
        and valid_columns == (324 if columns == 336 else 320)
        and x.device.type == "xpu"
        and x.is_contiguous()
        and weight.is_contiguous()
    ):
        # Block loads and DPAS consume the existing BF16 checkpoint layout.
        # The native FP32 accumulation matches the ordered projection exactly;
        # neither a transposed weight copy nor padded token rows are needed.
        if _large_hc is None:
            import vllm_xpu_kernels

            torch.ops.load_library(str(
                Path(vllm_xpu_kernels.__file__).with_name("liblong_hc.so")
            ))
            _large_hc = torch.ops.long_hc.forward
        return _large_hc(x, weight, valid_columns, 1)
    if (
        (1 <= rows <= 8 or (9 <= rows <= 32 and inner == 10240))
        and x.stride(1) == weight.stride(1) == 1
        and os.getenv("VLLM_XPU_C1_PARALLEL_HC", "1") == "1"
    ):
        return parallel_hc_gemm(x, weight, valid_columns)
    output = torch.empty((rows, columns), device=x.device, dtype=x.dtype)
    # Preserve fixed K ordering; wide prefill tiles share more weight loads.
    bm, bn = (32, 64) if rows >= 256 and inner == 10240 else (16, 32)
    _ordered_hc_gemm[(triton.cdiv(rows, bm), triton.cdiv(columns, bn))](
        x,
        weight,
        output,
        rows,
        columns,
        inner,
        *x.stride(),
        *weight.stride(),
        valid_columns,
        BM=bm, BN=bn,
        num_warps=4,
        num_stages=2,
    )
    return output


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
def _hc_finish(partials, output, M: tl.constexpr, N: tl.constexpr, TILES: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.full((256,), 0, tl.float32)
    for step in range(TILES):
        value += tl.load(partials + step * M * N + offsets, offsets < M * N, other=0)
    tl.store(output + offsets, value, offsets < M * N)


@triton.jit
def _hc_partials_batched(x, weight, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             VALID: tl.constexpr, XS: tl.constexpr, WS: tl.constexpr,
             BK: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(2) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.program_id(0) * BK + tl.arange(0, BK)
    a = tl.load(x + rows[:, None] * XS + kk[None, :],
                (rows[:, None] < M) & (kk[None, :] < K), other=0)
    b = tl.load(weight + cols[None, :] * WS + kk[:, None],
                (cols[None, :] < VALID) & (kk[:, None] < K), other=0)
    y = tl.dot(a, b)
    tl.store(out + tl.program_id(0) * M * N + rows[:, None] * N + cols[None, :],
             y, (rows[:, None] < M) & (cols[None, :] < N))


def parallel_hc_gemm(x, weight, valid_columns):
    """BF16 products with deterministic, batch-invariant FP32 tile summation.

    This changes the FP32 addition order from the serial K-loop. It preserves
    BF16 inputs/output and avoids both atomics and width-dependent reductions.
    """
    rows, inner = x.shape
    columns = weight.shape[0]
    block_k = 64 if rows > 8 else 128
    tiles = triton.cdiv(inner, block_k)
    partials = torch.empty((tiles, rows, columns), dtype=torch.float32, device=x.device)
    output = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    if rows > 8:
        _hc_partials_batched[(tiles, triton.cdiv(columns, 32), triton.cdiv(rows, 16))](
            x, weight, partials, rows, columns, inner, valid_columns,
            x.stride(0), weight.stride(0), block_k, 32, num_warps=4,
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
    _hc_finish[(triton.cdiv(rows * columns, 256),)](
        partials,
        output,
        rows,
        columns,
        tiles,
        num_warps=4,
    )
    return output

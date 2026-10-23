# SPDX-License-Identifier: Apache-2.0
"""Fixed-order BF16 HC down projection with FP32 accumulation."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _ordered_hc_gemm(
    x,
    weight,
    output,
    M: tl.constexpr,
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
    assert x.dtype == weight.dtype == torch.bfloat16
    rows, inner = x.shape
    columns = weight.shape[0]
    assert inner == weight.shape[1] and 0 < valid_columns <= columns
    output = torch.empty((rows, columns), device=x.device, dtype=x.dtype)
    _ordered_hc_gemm[(triton.cdiv(rows, 16), triton.cdiv(columns, 32))](
        x,
        weight,
        output,
        rows,
        columns,
        inner,
        *x.stride(),
        *weight.stride(),
        valid_columns,
        num_warps=4,
        num_stages=2,
    )
    return output

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-K2: HyperConnection producers that also emit the per-token int8 rows.

Each kernel computes exactly what its ``ops/hc.py`` / ``b70_hc_gemm.py``
counterpart computes (same fp32 expressions, same bf16 store), then quantizes
the stored bf16 row with the XPU ``scaled_int8_quant`` rule
(``b70_i8_producer.i8_scale`` / ``i8_quant``). One program owns one token row,
because the per-token scale needs the whole row.

  hc_gate_mix_quant   = hc_gate_mix               -> in_proj_qkvz / qkv_proj
  hc_silu_quant       = hc_silu                    -> HC up (input_mix_weight_up)
  parallel_hc_gemm_silu_quant
                      = parallel_hc_gemm_silu (V2) -> HC up
"""

import torch
from vllm.triton_utils import tl, triton

from vllm.model_executor.layers import b70_i8_producer as _pq
from vllm.model_executor.layers.b70_i8_producer import i8_quant, i8_scale


@triton.jit
def _hc_gate_mix_q_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    stride_x,
    stride_g,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NT_PAD: tl.constexpr,
):
    HC_DIM: tl.constexpr = DIM // HC
    row = tl.program_id(0).to(tl.int64)
    offs = (
        tl.arange(0, NT_PAD)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    mask = offs < HC_DIM
    # _hc_gate_mix_kernel, all tiles of the row in one program.
    acc = tl.zeros([NT_PAD, BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * HC_DIM + offs
        g = tl.load(g_ptr + row * stride_g + offsets, mask, other=0.0)
        x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc /= HC
    yb = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * stride_y + offs, yb, mask)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=1), axis=0)
    s = i8_scale(amax)
    tl.store(q_ptr + row * HC_DIM + offs, i8_quant(yf, s), mask)
    tl.store(s_ptr + row, s)


def hc_gate_mix_quant(
    x: torch.Tensor, gate: torch.Tensor, hc_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hc_gate_mix(x, gate, hc_count) + its int8 rows [N, HC_DIM] and fp32 [N, 1]."""
    n, dim = gate.shape
    assert x.shape == gate.shape and dim % hc_count == 0
    assert x.stride(1) == 1 and gate.stride(1) == 1
    assert x.dtype == torch.bfloat16
    hc_dim = dim // hc_count
    out = x.new_empty(n, hc_dim)
    xq = torch.empty((n, hc_dim), dtype=torch.int8, device=x.device)
    xs = torch.empty((n, 1), dtype=torch.float32, device=x.device)
    block = 512
    if n:
        _hc_gate_mix_q_kernel[(n,)](
            x,
            gate,
            out,
            xq,
            xs,
            x.stride(0),
            gate.stride(0),
            out.stride(0),
            dim,
            hc_count,
            block,
            triton.next_power_of_2(triton.cdiv(hc_dim, block)),
            num_warps=_pq.NUM_WARPS,
        )
    return out, xq, xs


@triton.jit
def _hc_silu_q_kernel(
    x_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(DIM)
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM
    # _hc_silu_kernel
    x = tl.load(x_ptr + row * stride_x + offs, mask).to(tl.float32) / HC
    y = x * tl.sigmoid(x)
    yb = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * stride_y + offs, yb, mask)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=0)
    s = i8_scale(amax)
    tl.store(q_ptr + row * DIM + offs, i8_quant(yf, s), mask)
    tl.store(s_ptr + row, s)


def hc_silu_quant(
    x: torch.Tensor, hc_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hc_silu(x, hc_count) + its int8 rows and fp32 [N, 1] scales."""
    num_tokens, dim = x.shape
    assert x.stride(1) == 1 and x.dtype == torch.bfloat16
    out = x.new_empty(x.shape)
    xq = torch.empty((num_tokens, dim), dtype=torch.int8, device=x.device)
    xs = torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device)
    if num_tokens:
        _hc_silu_q_kernel[(num_tokens,)](
            x,
            out,
            xq,
            xs,
            x.stride(0),
            out.stride(0),
            DIM=dim,
            HC=hc_count,
            num_warps=_pq.NUM_WARPS,
        )
    return out, xq, xs


@triton.jit
def _hc_finish_silu_q_kernel(
    partials,
    output,
    lora_out,
    q_ptr,
    s_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TILES: tl.constexpr,
    LORA: tl.constexpr,
    HC: tl.constexpr,
    N_PAD: tl.constexpr,
):
    """_hc_finish_silu (V2) for one row: the same fp32 tile sum in the same step
    order, BF16 store, hc_silu of the first LORA columns from the stored value;
    then the int8 quant of the silu row."""
    row = tl.program_id(0)
    col = tl.arange(0, N_PAD)
    offsets = row * N + col
    mask = col < N
    value = tl.full((N_PAD,), 0, tl.float32)
    for step in range(TILES):
        value += tl.load(partials + step * M * N + offsets, mask, other=0)
    stored = value.to(output.dtype.element_ty)
    tl.store(output + offsets, stored, mask)
    v = stored.to(tl.float32) / HC
    y = v * tl.sigmoid(v)
    lmask = col < LORA
    yb = y.to(lora_out.dtype.element_ty)
    tl.store(lora_out + row * LORA + col, yb, lmask)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.where(lmask, tl.abs(yf), 0.0), axis=0)
    s = i8_scale(amax)
    tl.store(q_ptr + row * LORA + col, i8_quant(yf, s), lmask)
    tl.store(s_ptr + row, s)


def parallel_hc_gemm_silu_quant(x, weight, valid_columns, lora_rank, hc_count):
    """parallel_hc_gemm_silu (B70-V2-HC) + the int8 rows of the silu output.

    Returns (out [rows, columns], silu_lora [rows, lora_rank], xq, xs)."""
    from ..b70_hc_gemm import _parallel_partials

    rows, _ = x.shape
    columns = weight.shape[0]
    assert 0 < lora_rank <= columns and x.dtype == torch.bfloat16
    partials, tiles = _parallel_partials(x, weight, valid_columns)
    output = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    lora = torch.empty((rows, lora_rank), dtype=x.dtype, device=x.device)
    xq = torch.empty((rows, lora_rank), dtype=torch.int8, device=x.device)
    xs = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _hc_finish_silu_q_kernel[(rows,)](
        partials,
        output,
        lora,
        xq,
        xs,
        rows,
        columns,
        tiles,
        lora_rank,
        hc_count,
        triton.next_power_of_2(columns),
        num_warps=_pq.NUM_WARPS,
    )
    return output, lora, xq, xs

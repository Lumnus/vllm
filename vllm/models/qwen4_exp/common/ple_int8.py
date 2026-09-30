# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 row-scale storage for the Qwen4Exp PLE n-gram table.

The PLE n-gram table of Qwen3.8-Flash-Next has ~320M rows of 160 values. In
BF16 that is ~95 GiB of pinned host memory. Stored as symmetric INT8 with one
float32 scale per row it is ~49 GiB, and the reconstruction error (relative L2
~0.66% on the real table) sits below anything measured downstream.

Row layout ("packed row"): ``dim`` int8 values followed by the row's float32
scale in little-endian byte order, ``dim + 4`` bytes in total, stored as
uint8. Keeping the scale inside the row means one gather, one all-reduce and
one copy move a row and its scale together; nothing else in the lookup path
needs to know about the format.

Quantisation (deterministic, done once at load time on the source device):

    s = max(|x|) / 127                      (float32, from the exact values)
    q = round_half_even(x / s), clamped to [-127, 127]   (0 when s == 0)
    x ~= q * s                              (float32, then one rounding)

This module only depends on torch so it can be unit tested (and reused by
offline tools) without initialising a vLLM platform.
"""

import torch

SCALE_BYTES = 4
"""Bytes of the float32 row scale stored after the int8 values."""

_QUANT_CHUNK_ROWS = 1 << 18


def packed_row_bytes(dim: int) -> int:
    """Bytes per stored row for a logical embedding width ``dim``."""
    return dim + SCALE_BYTES


def quantize_rows(rows: torch.Tensor) -> torch.Tensor:
    """Quantise ``rows`` [n, dim] (any float dtype) to packed uint8 [n, dim + 4]."""
    if rows.ndim != 2:
        raise ValueError(f"expected a 2D tensor of rows, got {tuple(rows.shape)}")
    if not rows.is_floating_point():
        raise ValueError(f"expected floating-point rows, got {rows.dtype}")
    num_rows, dim = rows.shape
    values = rows.to(torch.float32)
    scale = values.abs().amax(dim=1, keepdim=True) / 127.0
    quantized = (values / scale.clamp_min(1e-30)).round().clamp_(-127, 127)
    packed = torch.empty(
        num_rows, packed_row_bytes(dim), dtype=torch.uint8, device=rows.device
    )
    packed[:, :dim] = quantized.to(torch.int8).view(torch.uint8)
    # float32 bytes in host order; every platform vLLM runs on is little-endian.
    packed[:, dim:] = (
        scale.contiguous().view(torch.uint8).reshape(num_rows, SCALE_BYTES)
    )
    return packed


def dequantize_rows(
    packed: torch.Tensor,
    dim: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantise packed rows ``[..., k * (dim + 4)]`` to ``[..., k * dim]``.

    ``k`` packed rows may sit side by side in the last dimension (the n-gram
    heads of one token). ``q * s`` is computed in float32 and rounded once.
    """
    row_bytes = packed_row_bytes(dim)
    if packed.dtype != torch.uint8 or packed.shape[-1] % row_bytes:
        raise ValueError(
            f"INT8 PLE rows must be uint8 [..., k * {row_bytes}], got "
            f"{packed.dtype} {tuple(packed.shape)}"
        )
    lead = packed.shape[:-1]
    rows = packed.reshape(*lead, packed.shape[-1] // row_bytes, row_bytes)
    values = rows[..., :dim].view(torch.int8).to(torch.float32)
    scales = rows[..., dim:].contiguous().view(torch.float32)
    return (values * scales).to(output_dtype).reshape(*lead, -1)


def row_scales(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """The float32 scales of packed rows [n, dim + 4] as [n]."""
    return packed[:, dim:].contiguous().view(torch.float32).reshape(-1)


def copy_quantized_shard_(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
    chunk_rows: int = _QUANT_CHUNK_ROWS,
) -> int:
    """Quantise the rows of a checkpoint shard owned by this rank and store them.

    ``destination`` is this rank's packed table [>= tp_end - tp_start, dim + 4]
    (uint8, any device). ``loaded_weight`` holds checkpoint rows
    ``[checkpoint_start, checkpoint_start + n)`` either as floating-point
    [n, dim] (quantised here, in chunks, on its own device) or already packed
    as uint8 [n, dim + 4] (copied verbatim). Returns the number of rows stored.
    """
    if destination.dtype != torch.uint8 or destination.ndim != 2:
        raise ValueError("destination must be a 2D uint8 packed-row table")
    dim = destination.shape[1] - SCALE_BYTES
    if loaded_weight.ndim != 2:
        raise ValueError("loaded weight must be 2D")
    packed_input = loaded_weight.dtype == torch.uint8
    expected_width = destination.shape[1] if packed_input else dim
    if loaded_weight.shape[1] != expected_width:
        raise ValueError(
            f"PLE shard width {loaded_weight.shape[1]} does not match the INT8 "
            f"table ({dim} values + {SCALE_BYTES} scale bytes)"
        )
    if checkpoint_start < 0 or tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid checkpoint or TP row range")
    if destination.shape[0] < tp_end - tp_start:
        raise ValueError("destination does not cover the requested TP range")
    begin = max(checkpoint_start, tp_start)
    end = min(checkpoint_start + loaded_weight.shape[0], tp_end)
    if begin >= end:
        return 0
    for start in range(begin, end, chunk_rows):
        stop = min(end, start + chunk_rows)
        source = loaded_weight[start - checkpoint_start : stop - checkpoint_start]
        packed = source if packed_input else quantize_rows(source)
        with torch.no_grad():
            destination[start - tp_start : stop - tp_start].copy_(
                packed.to(device=destination.device)
            )
    return end - begin

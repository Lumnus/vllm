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
    rows, inner = x.shape
    columns = weight.shape[0]
    assert 0 < lora_rank <= valid_columns <= columns
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
    output = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    lora = torch.empty((rows, lora_rank), dtype=x.dtype, device=x.device)
    _hc_finish_silu[(triton.cdiv(rows * columns, 256),)](
        partials, output, lora, rows, columns, tiles, lora_rank, hc_count, num_warps=4
    )
    return output, lora


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


def _parallel_route_miss(x: torch.Tensor, weight: torch.Tensor) -> str | None:
    """The first term of ``takes_parallel_route`` that fails, or None."""
    rows, inner = x.shape
    if not (1 <= rows <= 8 or (9 <= rows <= PARALLEL_MAX_ROWS and inner == _HC_INNER)):
        return "rows"
    if not x.dtype == weight.dtype == torch.bfloat16:
        return f"dtype x={x.dtype} weight={weight.dtype}"
    if x.stride(1) != 1 or weight.stride(1) != 1:
        return f"stride x={tuple(x.stride())} weight={tuple(weight.stride())}"
    if inner != weight.shape[1]:
        return f"inner x={inner} weight={tuple(weight.shape)}"
    return None


def _native_up_miss(lora: torch.Tensor, up_weight: torch.Tensor) -> str | None:
    """The first term of ``takes_native_up`` that fails, or None."""
    if not _UP_ROWS[0] <= lora.shape[0] <= _UP_ROWS[1]:
        return "rows"
    if lora.device.type != "xpu":
        return f"device {lora.device}"
    if not up_weight.dtype == lora.dtype == torch.bfloat16:
        return f"dtype lora={lora.dtype} up={up_weight.dtype}"
    if tuple(up_weight.shape) != _UP_SHAPE:
        return f"up shape {tuple(up_weight.shape)} (native: {_UP_SHAPE})"
    if _load_up_op() is None:
        return "native library not loaded"
    return None


_reported: set[tuple[str, str]] = set()


def report_route(
    stage: str, x: torch.Tensor, weight: torch.Tensor, taken: bool
) -> None:
    """Log once per process and outcome which way a batch-invariant HC call
    went: the split-K down / native up route, or the first term that sent it to
    the stock GEMM. Row-count misses (prefill, wide verify batches) are expected."""
    if taken:
        key = (stage, "taken")
    else:
        check = _parallel_route_miss if stage == "down" else _native_up_miss
        key = (stage, check(x, weight) or "unknown")
    if key in _reported:
        return
    _reported.add(key)
    if taken:
        logger.info(
            "HC %s: batch-invariant route taken (first at %d rows).",
            stage,
            x.shape[0],
        )
    elif key[1] == "rows":
        logger.info(
            "HC %s: %d rows is outside the batch-invariant range; stock GEMM.",
            stage,
            x.shape[0],
        )
    else:
        logger.warning(
            "HC %s: stock GEMM at %d rows instead of the batch-invariant route: %s.",
            stage,
            x.shape[0],
            key[1],
        )


def native_up_mix(
    lora: torch.Tensor, up_weight: torch.Tensor, normalized: torch.Tensor, n_embd: int
) -> torch.Tensor:
    """``hc_gate_mix(normalized, lora @ up_weight.T, hc)`` in one native kernel."""
    output = torch.empty((lora.shape[0], n_embd), device=lora.device, dtype=lora.dtype)
    _up_op(lora, up_weight.detach(), normalized, output)
    return output

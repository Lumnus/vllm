# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Running the delayed, fused hyper-connection on XPU follows wu1ff's
# B70-LLM-Controller, used here under the MIT License:
#
# Copyright (c) 2026 wu1ff
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Fused hyper-connections for the XPU Qwen4-Exp model.

The state between blocks is ``hc`` parallel residual streams ``[T, hc, n_embd]``.
Each combine is delayed to the next mix, where it fuses with that mix's grouped
RMSNorm, and the down and injection projections run as one GEMM. The Triton kernels
are the ones the NVIDIA model runs. They hold the intermediate values in fp32 and
round once at each materialized tensor.

The kernels sit behind custom ops, like the PLE gate, so they stay opaque to
``torch.compile`` and run at call time with the real row count. HC holds no
per-request state, so these ops do not split a piecewise graph.

Under a DFlash2 drafter the target's HC down projection must not depend on the
batch: the drafter reads the target's features, and a verify step of 8 rows has
to reproduce what one row computes. Up to ``hc_gemm.PARALLEL_MAX_ROWS`` rows the
down GEMM then runs split-K in a fixed order with ``hc_silu`` folded in, and for
1-8 rows the up GEMM and the gated mix run as one native kernel when it loads.

The merged down and injection GEMM takes the model's quantization config. When an
online INT8 config names it (``quantization_config.targets``), its weight is int8
per output row and the GEMM runs as W8A8 on vllm-xpu-kernels'
``int8_dense_w8a8_fused_gemm``: per-token int8 activations and exact int32
accumulation, so a row's result does not depend on the batch it runs in.

The up projection, and the final mixer's down projection, are BF16 weight holders
unless the quantization config quantizes them: then they are linear layers, and
an INT8 checkpoint (compressed-tensors W8A8, per-row int8 ``weight`` and
``weight_scale``) or an online ``targets`` entry that names them runs through
the layer's own method, the INT8 dense GEMM of every other W8A8 linear
(per-token int8 activations; Triton at small row counts, oneDNN above). A merged
down projection that is INT8 runs on ``int8_dense_w8a8_fused_gemm`` when the
kernels have it, and through its layer's method otherwise. Which path a module
takes is fixed when its weights load, so a traced graph holds no per-call choice.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen4ExpTextConfig

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op, eager_bypass_custom_op

from ..nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from . import hc_down_fold, hc_gemm

logger = init_logger(__name__)


class _WeightContainer(nn.Module):
    """Holds a single ``weight`` parameter (no bias, no forward).

    Used for the hyper-connection weights the quantization config leaves BF16,
    loaded directly from the checkpoint. The container exists only so the weight
    sits at the exact ``<name>.weight`` path the checkpoint expects.
    """

    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape))


def int8_down_gemm(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """``x`` ``[T, K]`` bf16 @ int8 ``weight`` ``[N, K]`` with fp32 row scales
    ``[N, 1]``: per-token int8 quant and an s8 x s8 GEMM with exact int32
    accumulation (vllm-xpu-kernels). Batch-invariant per row."""
    return torch.ops._xpu_C.int8_dense_w8a8_fused_gemm(
        x.contiguous(), weight, weight_scale, None
    )


def _int8_down(
    xn: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    xq: torch.Tensor | None,
    xs: torch.Tensor | None,
) -> torch.Tensor:
    """The INT8 down GEMM: on ``xn``'s int8 rows when the norm kernel wrote
    them (``VLLM_XPU_HC_DOWN_FOLD``), else with its own quant."""
    if xq is not None:
        assert xs is not None
        return hc_down_fold.int8_down_gemm_q(xq, xs, weight, weight_scale)
    return int8_down_gemm(xn, weight, weight_scale)


def _gate_mix_and_inject(
    xn: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hc: int,
    lowrank: int,
    batch_invariant: bool = False,
    down_scale: torch.Tensor | None = None,
    down_xq: torch.Tensor | None = None,
    down_xs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``xn``: normalized ``[T, hc * n_embd]``. Returns the block input and the
    injection logits, which are empty when ``down_weight`` has no injection rows.
    With ``down_scale``, ``down_weight`` is int8 ``[N, K]`` and the down GEMM runs
    INT8 W8A8 (on ``down_xq`` / ``down_xs`` when given)."""
    if down_scale is not None:
        down = _int8_down(xn, down_weight, down_scale, down_xq, down_xs)
        lora = hc_silu(down[:, :lowrank], hc)
        if batch_invariant and hc_gemm.takes_native_up(lora, up_weight):
            block_input = hc_gemm.native_up_mix(lora, up_weight, xn, xn.shape[1] // hc)
        else:
            block_input = hc_gate_mix(xn, F.linear(lora, up_weight), hc)
        return block_input, down[:, lowrank : lowrank + hc]
    if batch_invariant and hc_gemm.takes_parallel_route(xn, down_weight):
        # Columns past the down and injection rows are the merged GEMM's padding.
        valid = min(down_weight.shape[0], lowrank + hc)
        down, lora = hc_gemm.parallel_hc_gemm_silu(xn, down_weight, valid, lowrank, hc)
        if hc_gemm.takes_native_up(lora, up_weight):
            block_input = hc_gemm.native_up_mix(lora, up_weight, xn, xn.shape[1] // hc)
        else:
            block_input = hc_gate_mix(xn, F.linear(lora, up_weight), hc)
        return block_input, down[:, lowrank : lowrank + hc]
    down = F.linear(xn, down_weight)
    gate = F.linear(hc_silu(down[:, :lowrank], hc), up_weight)
    return hc_gate_mix(xn, gate, hc), down[:, lowrank : lowrank + hc]


def hc_mix(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc: int,
    lowrank: int,
    batch_invariant: bool = False,
    down_scale: torch.Tensor | None = None,
    down_fold: int = 0,
    fold_words: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq = xs = None
    if down_fold and down_scale is not None and fold_words is not None:
        xn, xq, xs = hc_down_fold.norm_q(
            hidden, norm_weight, eps, hc, fold_words, down_fold
        )
    else:
        xn = grouped_gemma_rmsnorm(hidden, norm_weight, eps, hc)
    return _gate_mix_and_inject(
        xn, down_weight, up_weight, hc, lowrank, batch_invariant, down_scale, xq, xs
    )


def hc_combine_and_mix(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc: int,
    lowrank: int,
    batch_invariant: bool = False,
    down_scale: torch.Tensor | None = None,
    down_fold: int = 0,
    fold_words: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xq = xs = None
    if down_fold and down_scale is not None and fold_words is not None:
        combined, xn, xq, xs = hc_down_fold.combine_norm_q(
            hidden, block_output, injection, norm_weight, eps, hc, fold_words, down_fold
        )
    else:
        combined, xn = hc_combine_norm(
            hidden, block_output, injection, norm_weight, eps, hc
        )
    block_input, new_injection = _gate_mix_and_inject(
        xn, down_weight, up_weight, hc, lowrank, batch_invariant, down_scale, xq, xs
    )
    return combined, block_input, new_injection


def _hc_mix_fake(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc: int,
    lowrank: int,
    batch_invariant: bool = False,
    down_scale: torch.Tensor | None = None,
    down_fold: int = 0,
    fold_words: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = hidden.shape[0]
    down = hidden.new_empty((tokens, down_weight.shape[0]))
    block_input = hidden.new_empty((tokens, hidden.shape[1] // hc))
    # The same strides as the real column slice of the down projection.
    return block_input, down[:, lowrank : lowrank + hc]


def _hc_combine_and_mix_fake(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc: int,
    lowrank: int,
    batch_invariant: bool = False,
    down_scale: torch.Tensor | None = None,
    down_fold: int = 0,
    fold_words: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_input, new_injection = _hc_mix_fake(
        hidden, norm_weight, down_weight, up_weight, eps, hc, lowrank
    )
    return hidden.new_empty(hidden.shape), block_input, new_injection


def _hc_combine_fake(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    hc: int,
) -> torch.Tensor:
    return hidden.new_empty(hidden.shape)


direct_register_custom_op(
    op_name="qwen4_exp_xpu_hc_mix",
    op_func=hc_mix,
    mutates_args=[],
    fake_impl=_hc_mix_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_xpu_hc_combine_and_mix",
    op_func=hc_combine_and_mix,
    mutates_args=[],
    fake_impl=_hc_combine_and_mix_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_xpu_hc_combine",
    op_func=hc_combine,
    mutates_args=[],
    fake_impl=_hc_combine_fake,
)
_mix_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_mix", hc_mix)
_combine_and_mix_op = eager_bypass_custom_op(
    "qwen4_exp_xpu_hc_combine_and_mix", hc_combine_and_mix
)
_combine_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_combine", hc_combine)


# The split mix, for a module whose up projection (or final-mixer down projection)
# is a quantized linear layer. The layer's own method runs between these ops, as
# every other quantized linear does; the ops hold the HC kernels, so they stay
# opaque to torch.compile like the fused ones above.


def hc_norm(
    hidden: torch.Tensor, norm_weight: torch.Tensor, eps: float, hc: int
) -> torch.Tensor:
    return grouped_gemma_rmsnorm(hidden, norm_weight, eps, hc)


def hc_combine_then_norm(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return hc_combine_norm(hidden, block_output, injection, norm_weight, eps, hc)


def hc_norm_q(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc: int,
    fold_words: torch.Tensor,
    down_fold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``hc_norm`` with the INT8 down GEMM's int8 rows and scales of its output
    (``VLLM_XPU_HC_DOWN_FOLD``): ``(xn, xq, xs)``."""
    return hc_down_fold.norm_q(hidden, norm_weight, eps, hc, fold_words, down_fold)


def hc_combine_then_norm_q(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc: int,
    fold_words: torch.Tensor,
    down_fold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``hc_combine_then_norm`` with the int8 rows and scales of ``xn``:
    ``(combined, xn, xq, xs)``."""
    return hc_down_fold.combine_norm_q(
        hidden, block_output, injection, norm_weight, eps, hc, fold_words, down_fold
    )


def hc_down_lora(
    xn: torch.Tensor,
    down_weight: torch.Tensor,
    hc: int,
    lowrank: int,
    batch_invariant: bool = False,
    down_scale: torch.Tensor | None = None,
    down_xq: torch.Tensor | None = None,
    down_xs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(hc_silu(lora), injection logits)`` of a BF16 down weight, or of an INT8
    one with ``down_scale`` on the fused W8A8 GEMM (on ``down_xq`` /
    ``down_xs`` when given): the down half of ``_gate_mix_and_inject``."""
    if down_scale is not None:
        down = _int8_down(xn, down_weight, down_scale, down_xq, down_xs)
        lora = hc_silu(down[:, :lowrank], hc)
    elif batch_invariant and hc_gemm.takes_parallel_route(xn, down_weight):
        valid = min(down_weight.shape[0], lowrank + hc)
        down, lora = hc_gemm.parallel_hc_gemm_silu(xn, down_weight, valid, lowrank, hc)
    else:
        down = F.linear(xn, down_weight)
        lora = hc_silu(down[:, :lowrank], hc)
    return lora, down[:, lowrank : lowrank + hc]


def hc_lowrank_silu(down: torch.Tensor, hc: int, lowrank: int) -> torch.Tensor:
    """``hc_silu`` of the low-rank columns of a down projection's output."""
    return hc_silu(down[:, :lowrank], hc)


def hc_gate_mix_op(xn: torch.Tensor, gate: torch.Tensor, hc: int) -> torch.Tensor:
    return hc_gate_mix(xn, gate, hc)


def hc_up_mix(
    lora: torch.Tensor,
    up_weight: torch.Tensor,
    xn: torch.Tensor,
    hc: int,
    batch_invariant: bool = False,
) -> torch.Tensor:
    """The BF16 up GEMM and the gated mix: the up half of ``_gate_mix_and_inject``."""
    if batch_invariant and hc_gemm.takes_native_up(lora, up_weight):
        return hc_gemm.native_up_mix(lora, up_weight, xn, xn.shape[1] // hc)
    return hc_gate_mix(xn, F.linear(lora, up_weight), hc)


def hc_down_lora_q(
    xn: torch.Tensor,
    down_weight: torch.Tensor,
    hc: int,
    lowrank: int,
    batch_invariant: bool = False,
    down_scale: torch.Tensor | None = None,
    down_xq: torch.Tensor | None = None,
    down_xs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``hc_down_lora`` with the lora handed on as the int8 rows and fp32
    ``[T, 1]`` scales the up projection's INT8 linear makes of it
    (``ops.scaled_int8_quant``). On the split-K route the quant runs inside the
    finish kernel (``hc_gemm.parallel_hc_gemm_silu_quant``); elsewhere the
    linear's own quant op runs on the lora. Returns ``(q, scale, inject)``."""
    if (
        down_scale is None
        and batch_invariant
        and hc_gemm.takes_parallel_route(xn, down_weight)
    ):
        valid = min(down_weight.shape[0], lowrank + hc)
        down, q, scale = hc_gemm.parallel_hc_gemm_silu_quant(
            xn, down_weight, valid, lowrank, hc
        )
        return q, scale, down[:, lowrank : lowrank + hc]
    lora, inject = hc_down_lora(
        xn, down_weight, hc, lowrank, batch_invariant, down_scale, down_xq, down_xs
    )
    q, scale, _ = ops.scaled_int8_quant(lora.contiguous(), None, None, True)
    return q, scale, inject


def hc_up_mix_q(
    q: torch.Tensor,
    scale: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    xn: torch.Tensor,
    hc: int,
    gemm: int,
    fused: int,
) -> torch.Tensor:
    """The INT8 up projection on pre-quantized rows and the gated mix. ``gemm``
    names the linear's own GEMM (0: ``int8_scaled_mm_func``, Triton at every
    width; 1: ``xpu_int8_scaled_mm_func``, Triton up to
    ``VLLM_XPU_INT8_ONEDNN_MIN_ROWS`` and oneDNN above). Where that GEMM is
    Triton and ``fused`` is set, one kernel runs both: 1 the first fused kernel
    (``hc_gemm.int8_up_mix``), 3 the fast one (``hc_gemm.int8_up_mix_fast``);
    0 keeps the GEMM and the gated mix apart."""
    from vllm.model_executor.kernels.linear.scaled_mm import xpu as xpu_mm

    rows = q.shape[0]
    triton_rows = (
        hc_gemm.UP_MIX_I8_MAX_ROWS
        if gemm == 0
        else min(hc_gemm.UP_MIX_I8_MAX_ROWS, xpu_mm.XPU_INT8_ONEDNN_MIN_ROWS)
    )
    if fused == 3 and rows <= triton_rows:
        return hc_gemm.int8_up_mix_fast(q, scale, up_weight, up_scale, xn, hc)
    if fused and rows <= triton_rows:
        return hc_gemm.int8_up_mix(q, scale, up_weight, up_scale, xn, hc)
    func = (
        torch.ops.vllm.int8_scaled_mm_func
        if gemm == 0
        else torch.ops.vllm.xpu_int8_scaled_mm_func
    )
    gate = func(q, up_weight, scale, up_scale, xn.dtype, None)
    return hc_gate_mix(xn, gate, hc)


def _hc_down_lora_q_fake(
    xn: torch.Tensor,
    down_weight: torch.Tensor,
    hc: int,
    lowrank: int,
    batch_invariant: bool = False,
    down_scale: torch.Tensor | None = None,
    down_xq: torch.Tensor | None = None,
    down_xs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = xn.shape[0]
    down = xn.new_empty((tokens, down_weight.shape[0]))
    return (
        xn.new_empty((tokens, lowrank), dtype=torch.int8),
        xn.new_empty((tokens, 1), dtype=torch.float32),
        down[:, lowrank : lowrank + hc],
    )


def _hc_up_mix_q_fake(
    q: torch.Tensor,
    scale: torch.Tensor,
    up_weight: torch.Tensor,
    up_scale: torch.Tensor,
    xn: torch.Tensor,
    hc: int,
    gemm: int,
    fused: int,
) -> torch.Tensor:
    return xn.new_empty((xn.shape[0], xn.shape[1] // hc))


def _hc_norm_fake(
    hidden: torch.Tensor, norm_weight: torch.Tensor, eps: float, hc: int
) -> torch.Tensor:
    return hidden.new_empty(hidden.shape)


def _hc_combine_then_norm_fake(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return hidden.new_empty(hidden.shape), hidden.new_empty(hidden.shape)


def _fold_rows_fake(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = hidden.shape[0]
    return (
        hidden.new_empty(hidden.shape, dtype=torch.int8),
        hidden.new_empty((tokens, 1), dtype=torch.float32),
    )


def _hc_norm_q_fake(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc: int,
    fold_words: torch.Tensor,
    down_fold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (hidden.new_empty(hidden.shape), *_fold_rows_fake(hidden))


def _hc_combine_then_norm_q_fake(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc: int,
    fold_words: torch.Tensor,
    down_fold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        hidden.new_empty(hidden.shape),
        hidden.new_empty(hidden.shape),
        *_fold_rows_fake(hidden),
    )


def _hc_down_lora_fake(
    xn: torch.Tensor,
    down_weight: torch.Tensor,
    hc: int,
    lowrank: int,
    batch_invariant: bool = False,
    down_scale: torch.Tensor | None = None,
    down_xq: torch.Tensor | None = None,
    down_xs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = xn.shape[0]
    down = xn.new_empty((tokens, down_weight.shape[0]))
    return xn.new_empty((tokens, lowrank)), down[:, lowrank : lowrank + hc]


def _hc_lowrank_silu_fake(down: torch.Tensor, hc: int, lowrank: int) -> torch.Tensor:
    return down.new_empty((down.shape[0], lowrank))


def _hc_gate_mix_fake(xn: torch.Tensor, gate: torch.Tensor, hc: int) -> torch.Tensor:
    return xn.new_empty((xn.shape[0], xn.shape[1] // hc))


def _hc_up_mix_fake(
    lora: torch.Tensor,
    up_weight: torch.Tensor,
    xn: torch.Tensor,
    hc: int,
    batch_invariant: bool = False,
) -> torch.Tensor:
    return xn.new_empty((xn.shape[0], xn.shape[1] // hc))


for _name, _func, _fake in (
    ("qwen4_exp_xpu_hc_norm", hc_norm, _hc_norm_fake),
    (
        "qwen4_exp_xpu_hc_combine_then_norm",
        hc_combine_then_norm,
        _hc_combine_then_norm_fake,
    ),
    ("qwen4_exp_xpu_hc_down_lora", hc_down_lora, _hc_down_lora_fake),
    ("qwen4_exp_xpu_hc_lowrank_silu", hc_lowrank_silu, _hc_lowrank_silu_fake),
    ("qwen4_exp_xpu_hc_gate_mix", hc_gate_mix_op, _hc_gate_mix_fake),
    ("qwen4_exp_xpu_hc_up_mix", hc_up_mix, _hc_up_mix_fake),
    ("qwen4_exp_xpu_hc_down_lora_q", hc_down_lora_q, _hc_down_lora_q_fake),
    ("qwen4_exp_xpu_hc_up_mix_q", hc_up_mix_q, _hc_up_mix_q_fake),
    ("qwen4_exp_xpu_hc_norm_q", hc_norm_q, _hc_norm_q_fake),
    (
        "qwen4_exp_xpu_hc_combine_then_norm_q",
        hc_combine_then_norm_q,
        _hc_combine_then_norm_q_fake,
    ),
):
    direct_register_custom_op(
        op_name=_name, op_func=_func, mutates_args=[], fake_impl=_fake
    )
_norm_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_norm", hc_norm)
_combine_then_norm_op = eager_bypass_custom_op(
    "qwen4_exp_xpu_hc_combine_then_norm", hc_combine_then_norm
)
_down_lora_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_down_lora", hc_down_lora)
_lowrank_silu_op = eager_bypass_custom_op(
    "qwen4_exp_xpu_hc_lowrank_silu", hc_lowrank_silu
)
_gate_mix_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_gate_mix", hc_gate_mix_op)
_up_mix_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_up_mix", hc_up_mix)
_down_lora_q_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_down_lora_q", hc_down_lora_q)
_up_mix_q_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_up_mix_q", hc_up_mix_q)
_norm_q_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_norm_q", hc_norm_q)
_combine_then_norm_q_op = eager_bypass_custom_op(
    "qwen4_exp_xpu_hc_combine_then_norm_q", hc_combine_then_norm_q
)


def _up_int8_kernel(layer: nn.Module) -> tuple[int, object] | str:
    """``(gemm, kernel)`` of an INT8 up linear whose method quantizes its input
    per token, symmetric and dynamic, then runs a Triton INT8 GEMM (``gemm`` as
    in ``hc_up_mix_q``); else why the fold does not apply. The kernel's
    ``_get_layer_params`` gives the ``[K, N]`` weight and its scales."""
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
        XPUInt8ScaledMMLinearKernel,
    )

    method = layer.quant_method
    kernel = getattr(getattr(layer, "scheme", None), "kernel", None) or getattr(
        method, "int8_linear", None
    )
    if not isinstance(kernel, TritonInt8ScaledMMLinearKernel):
        return f"up linear kernel {type(kernel).__name__}"
    w_q, w_s, i_s, i_zp, azp_adj = kernel._get_layer_params(layer)
    if i_s is not None or i_zp is not None or azp_adj is not None:
        return "static or asymmetric input quantization"
    if getattr(layer, "bias", None) is not None:
        return "bias"
    if w_q.dtype != torch.int8 or w_s.numel() not in (1, w_q.shape[1]):
        return f"weight {w_q.dtype}, scale {tuple(w_s.shape)}"
    gemm = 1 if isinstance(kernel, XPUInt8ScaledMMLinearKernel) else 0
    return gemm, kernel


def _quantized_linear(
    input_size: int,
    output_size: int,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> ReplicatedLinear | None:
    """A replicated linear for an HC projection that the quantization config
    quantizes (from the checkpoint, or online through ``targets``); ``None`` when it
    stays BF16, so that weight keeps its raw holder and today's kernels."""
    if quant_config is None:
        return None
    layer = ReplicatedLinear(
        input_size,
        output_size,
        bias=False,
        quant_config=quant_config,
        prefix=prefix,
        return_bias=False,
        disable_tp=True,
    )
    if isinstance(layer.quant_method, UnquantizedLinearMethod):
        return None
    return layer


def _dflash2_target() -> bool:
    """A BF16 target on XPU serving a DFlash2 drafter."""
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False
    spec = vllm_config.speculative_config
    model_config = vllm_config.model_config
    on = (
        current_platform.is_xpu()
        and spec is not None
        and spec.method == "dflash"
        and spec.draft_model_config is not None
        and spec.draft_model_config.architectures == ["DFlash2DraftModel"]
        and model_config is not None
        and model_config.dtype == torch.bfloat16
    )
    if on:
        logger.info_once(
            "DFlash2 target HC: batch-invariant split-K down projection up to %d "
            "rows with hc_silu fused; native up + gated mix for 1-8 rows.",
            hc_gemm.PARALLEL_MAX_ROWS,
        )
    return on


class HyperConnection(nn.Module):
    """Hyper-connection mix/combine block.

    Replaces the standard per-layer RMSNorm + residual add.

    ``mix`` reads one mixed ``[T, n_embd]`` view:
        grouped-RMSNorm (per stream) -> gamma (``hc_norm``) ->
        low-rank down -> ``silu(x / hc)`` -> up -> sigmoid gate ->
        ``x * gate`` -> mean-collapse the streams.
    It also returns the ``[T, hc]`` injection logits of this block's combine.

    ``combine`` writes a block output back into the wide state:
        ``residual + block_out * 2 * sigmoid(inject / hc)``.
    The ``2 * sigmoid`` centres the scatter weights on 1, so an
    untrained injection matrix reproduces the plain residual add. It depends only
    on the logits, so any block can apply a pending combine.

    ``combine_and_mix`` is ``combine`` followed by ``mix`` in one pass.

    ``has_inject=False`` is used for the top-level ``hyper_connection_mixer``
    (the final output norm), which has no ``block_inject_weight``.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        has_inject: bool = True,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.hc = config.hc_count
        self.n_embd = config.hidden_size
        self.hc_dim = self.hc * self.n_embd
        self.lowrank = config.hc_lowrank
        self.eps = config.rms_norm_eps
        self.has_inject = has_inject
        self.batch_invariant = _dflash2_target()

        self.hc_norm = _WeightContainer([self.hc_dim])
        self.input_mix_weight_up: nn.Module = _quantized_linear(
            self.lowrank, self.hc_dim, quant_config, f"{prefix}.input_mix_weight_up"
        ) or _WeightContainer([self.hc_dim, self.lowrank])
        if has_inject:
            # The checkpoint's down and injection weights share one GEMM, with its
            # row count padded to 16 like the NVIDIA model.
            pad = -(self.lowrank + self.hc) % 16
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hc_dim,
                [self.lowrank, self.hc] + ([pad] if pad else []),
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.input_mix_weight_down_block_inject",
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down: nn.Module = _quantized_linear(
                self.hc_dim,
                self.lowrank,
                quant_config,
                f"{prefix}.input_mix_weight_down",
            ) or _WeightContainer([self.lowrank, self.hc_dim])
        self._up_layer = isinstance(self.input_mix_weight_up, LinearBase)
        # "bf16", "fused_int8" (int8_dense_w8a8_fused_gemm) or "layer" (the down
        # layer's own method); set when the weights load.
        self._down_path: str | None = None
        # VLLM_XPU_HC_UP_FUSED on an INT8 up linear: (gemm, kernel, fused) for
        # hc_up_mix_q, set when the weights load; None keeps the layer path.
        self._up_fold: tuple[int, object, int] | None = None
        self._up_fold_reason = ""
        # VLLM_XPU_HC_DOWN_FOLD on the fused INT8 down GEMM: 1 or 2 when the
        # norm kernel writes xn's int8 rows (hc_down_fold), with the module's
        # per-row arrival words; set when the weights load.
        self._down_fold = 0
        self._down_fold_reason = ""
        self._fold_words: torch.Tensor | None = None

    @property
    def _down_linear(self) -> LinearBase | None:
        if self.has_inject:
            return self.input_mix_weight_down_block_inject
        down = self.input_mix_weight_down
        return down if isinstance(down, LinearBase) else None

    def _resolve_down_path(self) -> str:
        layer = self._down_linear
        if layer is None or isinstance(layer.quant_method, UnquantizedLinearMethod):
            return "bf16"
        if not self.has_inject:
            # The final mixer's down projection has no injection rows: it runs
            # through its layer like the up projection.
            return "layer"
        weight = layer.weight
        scale = getattr(layer, "weight_scale", None)
        if weight.dtype != torch.int8 or scale is None:
            raise NotImplementedError(
                f"{layer.prefix}: the HC down projection runs BF16 or online INT8 "
                f"(int8_per_channel) or checkpoint INT8 W8A8; "
                f"{type(layer.quant_method).__name__} is not supported. Leave it out "
                "of quantization_config.targets."
            )
        if hasattr(torch.ops._xpu_C, "int8_dense_w8a8_fused_gemm"):
            return "fused_int8"
        return "layer"

    def process_weights_after_loading(self) -> None:
        """Fix the down projection's path once its quant method has processed it
        (the loader runs this after the module's layers). The merged INT8 GEMM's
        padding rows have no checkpoint source: zero them, with unit scales, so
        their columns (sliced off) hold no stray values."""
        self._down_path = self._resolve_down_path()
        self._resolve_up_fold()
        self._resolve_down_fold()
        if self.has_inject and self._down_path != "bf16":
            layer = self.input_mix_weight_down_block_inject
            valid = self.lowrank + self.hc
            weight = layer.weight.data
            rows = weight.t() if weight.shape[-1] != self.hc_dim else weight
            if rows.shape[0] > valid:
                rows[valid:].zero_()
                scale = layer.weight_scale.data
                if scale.numel() == rows.shape[0]:
                    scale.view(-1)[valid:] = 1.0

    def _resolve_up_fold(self) -> None:
        """VLLM_XPU_HC_UP_FUSED: 1 folds the up linear's activation quant into
        the down projection's finish kernel and runs the INT8 up GEMM with the
        gated mix in its epilogue; 2 folds the quant only (the linear's GEMM and
        the stock gated mix); 3 is 1 on the fast fused kernel. Applies to an
        INT8 up linear with dynamic symmetric per-token activations on the
        Triton INT8 GEMM, next to a down projection that does not run through
        its own layer."""
        self._up_fold = None
        mode = envs.VLLM_XPU_HC_UP_FUSED
        if mode not in (1, 2, 3):
            self._up_fold_reason = "off"
            return
        if not self._up_layer:
            self._up_fold_reason = "up projection is BF16"
            return
        if self._down_path == "layer":
            self._up_fold_reason = "down projection runs through its layer"
            return
        found = _up_int8_kernel(self.input_mix_weight_up)
        if isinstance(found, str):
            self._up_fold_reason = found
            return
        gemm, kernel = found
        self._up_fold = (gemm, kernel, 0 if mode == 2 else mode)
        self._up_fold_reason = "on"

    def _resolve_down_fold(self) -> None:
        """VLLM_XPU_HC_DOWN_FOLD: the INT8 down GEMM's per-token quant runs in
        the tail of the kernel that writes ``xn`` (the combine + norm, or the
        norm of the stack's first mix), and the GEMM takes the pre-quantized
        rows (hc_down_fold). 1 recomputes the row in its last program, 2
        re-reads it under acq_rel. Applies to a down projection on the fused
        INT8 GEMM when the kernels have the pre-quantized entry."""
        self._down_fold = 0
        self._fold_words = None
        mode = envs.VLLM_XPU_HC_DOWN_FOLD
        if mode not in (1, 2):
            self._down_fold_reason = "off"
            return
        if self.down_path != "fused_int8":
            self._down_fold_reason = f"down projection {self.down_path}"
            return
        if not 1 <= self.hc <= hc_down_fold.MAX_HC:
            self._down_fold_reason = f"{self.hc} streams"
            return
        if not hc_down_fold.gemm_available():
            self._down_fold_reason = "no int8_dense_w8a8_gemm in vllm-xpu-kernels"
            return
        vllm_config = get_current_vllm_config_or_none()
        sched = getattr(vllm_config, "scheduler_config", None)
        rows = getattr(sched, "max_num_batched_tokens", None) or 8192
        device = self.input_mix_weight_down_block_inject.weight.device
        self._fold_words = hc_down_fold.new_arrival_words(rows, device)
        self._down_fold = mode
        self._down_fold_reason = "on"

    @property
    def down_path(self) -> str:
        # Fixed by the loader; a module used without it resolves at each call.
        if self._down_path is None:
            return self._resolve_down_path()
        return self._down_path

    @property
    def up_path(self) -> str:
        return "layer" if self._up_layer else "bf16"

    def _down_operands(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The down weight, and its row scales when it is INT8 on the fused GEMM."""
        path = self.down_path
        if not self.has_inject:
            return self.input_mix_weight_down.weight, None
        layer = self.input_mix_weight_down_block_inject
        if path == "bf16":
            return layer.weight, None
        if path == "layer":
            raise NotImplementedError(
                f"{layer.prefix}: the INT8 HC down projection runs through its "
                "layer: int8_dense_w8a8_fused_gemm is not in this vllm-xpu-kernels "
                "build."
            )
        logger.info_once(
            "XPU HC down projection: INT8 W8A8 (per-token int8 activations, "
            "exact int32 accumulation)."
        )
        # The INT8 linear kernels keep the weight as a [K, N] view of [N, K] rows.
        weight = layer.weight
        if weight.shape[-1] != self.hc_dim:
            weight = weight.t()
        return weight, layer.weight_scale

    @property
    def _down_weight(self) -> torch.Tensor:
        return self._down_operands()[0]

    @property
    def _split(self) -> bool:
        """Whether the mix runs as ops around a quantized layer's GEMM."""
        return self._up_layer or self.down_path == "layer"

    def _split_mix(
        self, xn: torch.Tensor, fold_rows: tuple[torch.Tensor, ...] = ()
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The mix of the normalized ``xn``, with each projection that is a
        quantized layer run through its own method. ``fold_rows``: ``xn``'s
        int8 rows and scales for the INT8 down GEMM, when the norm kernel wrote
        them (``VLLM_XPU_HC_DOWN_FOLD``)."""
        if self._up_fold is not None:
            gemm, kernel, fused = self._up_fold
            up_weight, up_scale = kernel._get_layer_params(self.input_mix_weight_up)[:2]
            down_weight, down_scale = self._down_operands()
            q, scale, inject = _down_lora_q_op(
                xn,
                down_weight,
                self.hc,
                self.lowrank,
                self.batch_invariant,
                down_scale,
                *fold_rows,
            )
            block_input = _up_mix_q_op(
                q, scale, up_weight, up_scale, xn, self.hc, gemm, fused
            )
            return block_input, inject
        if self.down_path == "layer":
            down_linear = self._down_linear
            assert down_linear is not None
            down = down_linear(xn)
            lora = _lowrank_silu_op(down, self.hc, self.lowrank)
            inject = down[:, self.lowrank : self.lowrank + self.hc]
        else:
            down_weight, down_scale = self._down_operands()
            lora, inject = _down_lora_op(
                xn,
                down_weight,
                self.hc,
                self.lowrank,
                self.batch_invariant,
                down_scale,
                *fold_rows,
            )
        if self._up_layer:
            block_input = _gate_mix_op(xn, self.input_mix_weight_up(lora), self.hc)
        else:
            block_input = _up_mix_op(
                lora,
                self.input_mix_weight_up.weight,
                xn,
                self.hc,
                self.batch_invariant,
            )
        return block_input, inject

    def _flat(self, wide: torch.Tensor) -> torch.Tensor:
        return wide.reshape(wide.shape[0], self.hc_dim)

    def _wide(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.view(flat.shape[0], self.hc, self.n_embd)

    def mix(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [T, hc, n_embd]
        if self._split:
            if self._down_fold:
                xn, xq, xs = _norm_q_op(
                    self._flat(x),
                    self.hc_norm.weight,
                    self.eps,
                    self.hc,
                    self._fold_words,
                    self._down_fold,
                )
                return self._split_mix(xn, (xq, xs))
            xn = _norm_op(self._flat(x), self.hc_norm.weight, self.eps, self.hc)
            return self._split_mix(xn)
        down_weight, down_scale = self._down_operands()
        fold_args = (self._down_fold, self._fold_words) if self._down_fold else ()
        return _mix_op(
            self._flat(x),
            self.hc_norm.weight,
            down_weight,
            self.input_mix_weight_up.weight,
            self.eps,
            self.hc,
            self.lowrank,
            self.batch_invariant,
            down_scale,
            *fold_args,
        )

    def combine_and_mix(
        self,
        residual: torch.Tensor,
        block_out: torch.Tensor,
        inject: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # residual: [T, hc, n_embd]; block_out: [T, n_embd]; inject: [T, hc]
        if self._split:
            fold_rows: tuple[torch.Tensor, ...] = ()
            if self._down_fold:
                combined, xn, xq, xs = _combine_then_norm_q_op(
                    self._flat(residual),
                    block_out,
                    inject,
                    self.hc_norm.weight,
                    self.eps,
                    self.hc,
                    self._fold_words,
                    self._down_fold,
                )
                fold_rows = (xq, xs)
            else:
                combined, xn = _combine_then_norm_op(
                    self._flat(residual),
                    block_out,
                    inject,
                    self.hc_norm.weight,
                    self.eps,
                    self.hc,
                )
            mixed, new_inject = self._split_mix(xn, fold_rows)
            return self._wide(combined), mixed, new_inject
        down_weight, down_scale = self._down_operands()
        fold_args = (self._down_fold, self._fold_words) if self._down_fold else ()
        combined, mixed, new_inject = _combine_and_mix_op(
            self._flat(residual),
            block_out,
            inject,
            self.hc_norm.weight,
            down_weight,
            self.input_mix_weight_up.weight,
            self.eps,
            self.hc,
            self.lowrank,
            self.batch_invariant,
            down_scale,
            *fold_args,
        )
        return self._wide(combined), mixed, new_inject

    def combine(
        self,
        residual: torch.Tensor,
        block_out: torch.Tensor,
        inject: torch.Tensor,
    ) -> torch.Tensor:
        # residual: [T, hc, n_embd]; block_out: [T, n_embd]; inject: [T, hc]
        return self._wide(_combine_op(self._flat(residual), block_out, inject, self.hc))


def log_hc_paths(root: nn.Module, scope: str) -> None:
    """One line per loaded model: how many hyper-connections run a projection
    through a quantized layer or the fused INT8 down GEMM. Silent when all are
    BF16."""
    modules = [m for m in root.modules() if isinstance(m, HyperConnection)]
    up = [m for m in modules if m.up_path == "layer"]
    fused = [m for m in modules if m.down_path == "fused_int8"]
    layer = [m for m in modules if m.down_path == "layer"]
    if not (up or fused or layer):
        return
    methods = sorted(
        {
            type(linear.quant_method).__name__
            for m in modules
            for linear in (m.input_mix_weight_up, m._down_linear)
            if isinstance(linear, LinearBase)
            and not isinstance(linear.quant_method, UnquantizedLinearMethod)
        }
    )
    logger.info(
        "XPU hyper-connections (%s): %d modules; up projection through its "
        "quantized layer (INT8 dense GEMM): %d; down projection on "
        "int8_dense_w8a8_fused_gemm: %d; down projection through its quantized "
        "layer: %d (final mixer among them: %d); methods: %s.",
        scope,
        len(modules),
        len(up),
        len(fused),
        len(layer),
        sum(1 for m in layer if not m.has_inject),
        ", ".join(methods),
    )
    mode = envs.VLLM_XPU_HC_UP_FUSED
    if mode in (1, 2, 3):
        folded = [m for m in modules if m._up_fold is not None]
        reasons = sorted({m._up_fold_reason for m in modules if m._up_fold is None})
        logger.info(
            "XPU HC up fused (VLLM_XPU_HC_UP_FUSED=%d, %s): %d of %d modules; "
            "the up projection's per-token int8 quant runs inside the down "
            "projection's finish kernel (split-K route) or on the lora (other "
            "routes)%s. Not folded: %s.",
            mode,
            scope,
            len(folded),
            len(modules),
            "; INT8 up GEMM + gated mix in one kernel up to "
            f"{hc_gemm.UP_MIX_I8_MAX_ROWS} rows where the linear runs Triton"
            + (
                " (fast kernel: tensor-descriptor loads, four streams per K "
                "step, column tile {}, K tile {}, {} warps)".format(
                    *hc_gemm._fast_up_tile()
                )
                if mode == 3
                else ""
            )
            if mode in (1, 3)
            else "; the linear's GEMM and the stock gated mix",
            ", ".join(reasons) or "none",
        )
    mode = envs.VLLM_XPU_HC_DOWN_FOLD
    if mode in (1, 2):
        folded = [m for m in modules if m._down_fold]
        reasons = sorted({m._down_fold_reason for m in modules if not m._down_fold})
        logger.info(
            "XPU HC down fold (VLLM_XPU_HC_DOWN_FOLD=%d, %s): %d of %d modules; "
            "the INT8 down GEMM's per-token quant (IEEE 127/amax, the calibration "
            "rule) runs in the tail of the kernel that writes xn, the last stream "
            "of each row found by one atomic arrival word (%s), and the GEMM takes "
            "the rows on int8_dense_w8a8_gemm. Not folded: %s.",
            mode,
            scope,
            len(folded),
            len(modules),
            "it recomputes the row" if mode == 1 else "it re-reads xn under acq_rel",
            ", ".join(reasons) or "none",
        )

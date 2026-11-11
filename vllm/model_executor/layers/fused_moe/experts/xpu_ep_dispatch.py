# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fewer small kernels around the XPU expert-parallel fused MoE.

Per MoE layer, the expert-parallel XPU path launches around the routing:

1. ``torch.where(topk_ids < 0, sentinel, topk_ids)`` in ``XPUExperts.apply``:
   compare, scalar fill and select, three kernels;
2. ``(topk_ids < 0).all(1)`` and an out-of-place ``masked_fill`` of the hidden
   states in ``XpuFusedMoe._apply_kernel``: compare, reduce, a copy of the
   hidden states and the fill, four kernels;
3. ``torch.zeros(rows_per_expert)``: one fill.

After the sentinel rewrite every id is >= 0, so the padding mask of step 2 is
False on every row and its ``masked_fill`` is the identity. Here one Triton
kernel writes the sentinel-mapped ids and zeroes ``rows_per_expert``, and the
fused MoE runs the kernel body without step 2. Eight launches become one per
layer, and the output is the same: the ids are the same integers and the hidden
states the same values.

The kernel body is ``XpuFusedMoe._apply_kernel`` from vllm-xpu-kernels, reduced
to the case this path takes (unquantized 16-bit activations, no hooks). It runs
only when the AST fingerprint of the installed ``_apply_kernel`` is one this
body was taken from; any other wheel logs once and keeps its own path. The
natural home of the change is the kernels package (a caller-validated id path
in ``XpuFusedMoe.apply``); this module retires when that lands.
"""

import ast
import hashlib
import inspect
import textwrap

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# AST fingerprints of XpuFusedMoe._apply_kernel the body below was taken from
# (vllm-xpu-kernels 0.1.18.dev0, l34 k3 eef6ab2c7).
_KNOWN_APPLY_KERNEL = frozenset({"c5b29b6cf70cd285"})

_armed: bool | None = None
_logged = False


def fingerprint(func) -> str:
    """AST fingerprint of a function, line numbers and comments excluded."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return hashlib.sha256(ast.dump(tree.body[0]).encode()).hexdigest()[:16]


def _wheel_matches(impl) -> bool:
    global _armed
    if _armed is None:
        func = type(impl)._apply_kernel
        try:
            got = fingerprint(func)
        except (OSError, TypeError, SyntaxError) as e:
            got = f"unreadable ({e})"
        _armed = got in _KNOWN_APPLY_KERNEL
        if not _armed:
            logger.info(
                "XPU MoE EP dispatch: XpuFusedMoe._apply_kernel fingerprint %s is "
                "not one of %s; the kernels package's own path runs",
                got,
                sorted(_KNOWN_APPLY_KERNEL),
            )
    return _armed


def eligible(impl, hidden_states: torch.Tensor, output: torch.Tensor, kwargs) -> bool:
    """The reduced body covers this call: kernel path, 16-bit activations and
    output, no activation quantization, no hooks, no W4A8."""
    if getattr(impl, "_use_ref", True) or kwargs:
        return False
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        return False
    if output.dtype != hidden_states.dtype:
        return False
    return _wheel_matches(impl)


# n_ids is not specialised: one compiled variant serves every batch size.
@triton.jit(do_not_specialize=["n_ids"])
def _sentinel_zero_kernel(
    ids_ptr, out_ptr, n_ids, sentinel, rpe_ptr, n_exp, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_ids
    x = tl.load(ids_ptr + offs, mask=m, other=0)
    tl.store(out_ptr + offs, tl.where(x < 0, sentinel, x).to(x.dtype), mask=m)
    tl.store(rpe_ptr + offs, tl.zeros([BLOCK], dtype=tl.int32), mask=offs < n_exp)


def sentinel_and_zero(
    topk_ids: torch.Tensor, sentinel: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``where(ids < 0, sentinel, ids)`` and a zeroed int32 ``rows_per_expert``,
    in one launch."""
    ids = topk_ids.contiguous()
    out = torch.empty_like(ids)
    rows_per_expert = torch.empty(
        (num_local_experts,), dtype=torch.int32, device=ids.device
    )
    block = 256
    grid = (
        max(triton.cdiv(ids.numel(), block), triton.cdiv(num_local_experts, block), 1),
    )
    _sentinel_zero_kernel[grid](
        ids,
        out,
        ids.numel(),
        sentinel,
        rows_per_expert,
        num_local_experts,
        BLOCK=block,
        num_warps=1,
    )
    return out.view(topk_ids.shape), rows_per_expert


def apply(
    impl,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sentinel: int,
    expert_map: torch.Tensor | None = None,
) -> None:
    """``XpuFusedMoe.apply`` for an expert-parallel layer whose unrouted ids
    (-1) go to ``sentinel``, an expert this rank does not own."""
    global _logged
    if not _logged:
        _logged = True
        logger.info(
            "XPU MoE EP dispatch: sentinel ids and rows_per_expert in one kernel, "
            "dead padding mask skipped"
        )
    topk_ids, rows_per_expert = sentinel_and_zero(topk_ids, sentinel, impl.num_experts)
    if expert_map is None:
        expert_map = impl.expert_map
    hidden_states = hidden_states.contiguous()
    num_rows, hidden_size = hidden_states.shape
    num_moe_inputs = impl.n_experts_per_token * num_rows
    device = hidden_states.device

    remapped_hidden_states = torch.empty(
        (num_moe_inputs, hidden_size), dtype=hidden_states.dtype, device=device
    )
    unpermuted_row_to_permuted_row = torch.empty(
        (num_rows, impl.n_experts_per_token), dtype=torch.int32, device=device
    )
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=hidden_states,
        hidden_states_scales=None,
        remapped_hidden_states=remapped_hidden_states,
        remapped_hidden_states_scales=None,
        expert_map=expert_map,
        rows_per_expert=rows_per_expert,
        unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
        topk_ids=topk_ids,
        total_experts_num=impl.total_experts_num,
        local_experts_num=impl.local_experts_num,
        expert_scale_desc=None,
    )

    gemm1_output = torch.empty(
        (num_moe_inputs, 2 * impl.inter_size), dtype=output.dtype, device=device
    )
    impl._grouped_gemm(
        remapped_hidden_states,
        None,
        impl.w13,
        impl.gemm1_wei_scales,
        impl.w13_bias,
        gemm1_output,
        rows_per_expert,
        2 * impl.inter_size,
        hidden_size,
    )
    if impl.gemm1_clamp_limit is not None and impl.gemm1_clamp_limit > 0:
        gemm1_output[:, : impl.inter_size].clamp_(max=impl.gemm1_clamp_limit)
        gemm1_output[:, impl.inter_size :].clamp_(
            min=-impl.gemm1_clamp_limit, max=impl.gemm1_clamp_limit
        )
    act_output = torch.empty(
        (num_moe_inputs, impl.inter_size * impl.inter_size_scale),
        dtype=gemm1_output.dtype,
        device=device,
    )
    impl.act_func(act_output, gemm1_output)

    gemm2_output = torch.empty(
        (num_moe_inputs, hidden_size), dtype=output.dtype, device=device
    )
    impl._grouped_gemm(
        act_output,
        None,
        impl.w2,
        impl.gemm2_wei_scales,
        impl.w2_bias,
        gemm2_output,
        rows_per_expert,
        hidden_size,
        impl.inter_size * impl.inter_size_scale,
    )
    torch.ops._moe_C.moe_gather(
        output,
        gemm2_output,
        topk_weights,
        unpermuted_row_to_permuted_row,
        impl.num_experts,
    )

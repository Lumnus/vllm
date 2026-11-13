# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import enum
import functools
import inspect
from typing import Any

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.eplb.expert_weights import ExpertWeight
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import (
    ExpertTiers,
    remap_unrouted,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt4Static32,
    kInt8DynamicTokenSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

if current_platform.is_xpu():
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe


_STAGED_FIELDS = ("w13", "w2", "gemm1_wei_scales", "gemm2_wei_scales")


def staged_impl(impl: "XpuFusedMoe", num_tokens: int, slot: str) -> "XpuFusedMoe":
    """``impl``, or a shallow copy that reads its UVA-offloaded weights and
    scales from the offloader's staging buffers for this step
    (``--cpu-offload-stage-min-tokens``). XpuFusedMoe holds its own weight
    tensors, so the experts stage them here instead of through the w1/w2
    arguments of the modular kernel."""
    from vllm.model_executor.offloader.base import get_offloader

    offloader = get_offloader()
    staged: dict[str, torch.Tensor] = {}
    for field in _STAGED_FIELDS:
        tensor = getattr(impl, field, None)
        if tensor is None:
            continue
        copy_ = offloader.maybe_stage(tensor, num_tokens, f"{slot}.{field}")
        if copy_ is not tensor:
            staged[field] = copy_
    if not staged:
        return impl
    impl = copy.copy(impl)
    for field, tensor in staged.items():
        setattr(impl, field, tensor)
    return impl


# n_ids is not specialised: one compiled variant serves every batch size.
@triton.jit(do_not_specialize=["n_ids"])
def _sentinel_ids_kernel(
    ids_ptr, out_ptr, n_ids, sentinel, counts_ptr, n_experts, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_ids
    x = tl.load(ids_ptr + offs, mask=m, other=0)
    tl.store(out_ptr + offs, tl.where(x < 0, sentinel, x).to(x.dtype), mask=m)
    tl.store(
        counts_ptr + offs, tl.zeros([BLOCK], dtype=tl.int32), mask=offs < n_experts
    )


def sentinel_ids_and_counts(
    topk_ids: torch.Tensor, sentinel: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``where(topk_ids < 0, sentinel, topk_ids)`` and a zeroed int32
    ``rows_per_expert`` for the fused MoE to fill, in one launch."""
    ids = topk_ids.contiguous()
    out = torch.empty_like(ids)
    counts = torch.empty((num_local_experts,), dtype=torch.int32, device=ids.device)
    block = 256
    grid = (max(triton.cdiv(max(ids.numel(), num_local_experts), block), 1),)
    _sentinel_ids_kernel[grid](
        ids,
        out,
        ids.numel(),
        sentinel,
        counts,
        num_local_experts,
        BLOCK=block,
        num_warps=1,
    )
    return out.view(topk_ids.shape), counts


@triton.jit(do_not_specialize=["hidden"])
def _pick_rows_kernel(
    y_ptr,
    a_ptr,
    a_rows_ptr,
    b_ptr,
    b_rows_ptr,
    c_ptr,
    c_rows_ptr,
    hidden,
    HAS_B: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_C: tl.constexpr,
):
    """y[i] = a[a_rows[i]] if a_rows[i] >= 0, else b[b_rows[i]] if
    b_rows[i] >= 0, else c[c_rows[i]] if c_rows[i] >= 0 (HAS_C: the rows
    the CPU computed), else zeros; i runs over the (token, top-k) pairs."""
    i = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < hidden
    # The row numbers as vectors, so every mask is a full vector.
    ra = tl.zeros([BLOCK], dtype=tl.int32) + tl.load(a_rows_ptr + i)
    in_a = ra >= 0
    free = ra < 0
    v = tl.load(
        a_ptr + tl.maximum(ra, 0).to(tl.int64) * hidden + offs,
        mask=m & in_a,
        other=0.0,
    )
    if HAS_B:
        rb = tl.zeros([BLOCK], dtype=tl.int32) + tl.load(b_rows_ptr + i)
        vb = tl.load(
            b_ptr + tl.maximum(rb, 0).to(tl.int64) * hidden + offs,
            mask=m & (rb >= 0) & (ra < 0),
            other=0.0,
        )
        v = tl.where(in_a, v, vb)
        free = free & (rb < 0)
    if HAS_C:
        rc = tl.zeros([BLOCK], dtype=tl.int32) + tl.load(c_rows_ptr + i)
        vc = tl.load(
            c_ptr + tl.maximum(rc, 0).to(tl.int64) * hidden + offs,
            mask=m & (rc >= 0) & free,
            other=0.0,
        )
        v = tl.where(free, vc, v)
    tl.store(y_ptr + i.to(tl.int64) * hidden + offs, v, mask=m)


@triton.jit(do_not_specialize=["hidden"])
def _sum_topk_kernel(
    out_ptr,
    w_ptr,
    a_ptr,
    a_rows_ptr,
    b_ptr,
    b_rows_ptr,
    c_ptr,
    c_rows_ptr,
    hidden,
    TOPK: tl.constexpr,
    DIRECT: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_C: tl.constexpr,
):
    """out[t] = sum over k = 0 .. TOPK-1, in that order, of w[t, k] times the
    row of (t, k): row t*TOPK+k of a (DIRECT), or a[a_rows] / b[b_rows] /
    c[c_rows] as in _pick_rows_kernel. fp32 accumulation, one store in out's
    dtype. A pair no source holds adds an exact zero, so the order of the
    others, and the result, do not depend on which source held which pair."""
    t = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < hidden
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        i = t * TOPK + k
        wk = tl.load(w_ptr + i).to(tl.float32)
        if DIRECT:
            v = tl.load(a_ptr + i.to(tl.int64) * hidden + offs, mask=m, other=0.0)
        else:
            ra = tl.zeros([BLOCK], dtype=tl.int32) + tl.load(a_rows_ptr + i)
            in_a = ra >= 0
            free = ra < 0
            v = tl.load(
                a_ptr + tl.maximum(ra, 0).to(tl.int64) * hidden + offs,
                mask=m & in_a,
                other=0.0,
            )
            if HAS_B:
                rb = tl.zeros([BLOCK], dtype=tl.int32) + tl.load(b_rows_ptr + i)
                vb = tl.load(
                    b_ptr + tl.maximum(rb, 0).to(tl.int64) * hidden + offs,
                    mask=m & (rb >= 0) & (ra < 0),
                    other=0.0,
                )
                v = tl.where(in_a, v, vb)
                free = free & (rb < 0)
            if HAS_C:
                rc = tl.zeros([BLOCK], dtype=tl.int32) + tl.load(c_rows_ptr + i)
                vc = tl.load(
                    c_ptr + tl.maximum(rc, 0).to(tl.int64) * hidden + offs,
                    mask=m & (rc >= 0) & free,
                    other=0.0,
                )
                v = tl.where(free, vc, v)
        acc += v.to(tl.float32) * wk
    tl.store(
        out_ptr + t.to(tl.int64) * hidden + offs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m,
    )


_COMBINE_BLOCK = 1024
# Tokens per exchange in "exact" mode: bounds the per-(token, top-k) buffer
# to EXACT_CHUNK x top_k x hidden values.
EXACT_CHUNK = 512


def sum_topk(
    out: torch.Tensor,
    topk_weights: torch.Tensor,
    sources: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """``out[t] = sum_k topk_weights[t, k] * row(t, k)`` in k order, fp32
    accumulation, one rounding to ``out``'s dtype. ``sources`` are one to
    three ``(rows, row_map)``: the expert-grouped outputs of a fused-MoE call
    and its ``[tokens, top_k]`` map (-1: not computed there), or the rows the
    CPU computed and theirs; a pair is taken from the first source that
    holds it. The sum is the same bits whichever source held which pair."""
    num_tokens, top_k = topk_weights.shape
    if num_tokens == 0:
        return
    hidden = out.shape[-1]
    (a, a_rows), (b, b_rows), (c, c_rows) = _three(sources)
    grid = (num_tokens, triton.cdiv(hidden, _COMBINE_BLOCK))
    _sum_topk_kernel[grid](
        out,
        topk_weights.contiguous(),
        a,
        a_rows.contiguous(),
        b,
        b_rows.contiguous(),
        c,
        c_rows.contiguous(),
        hidden,
        TOPK=top_k,
        DIRECT=False,
        HAS_B=len(sources) > 1,
        BLOCK=_COMBINE_BLOCK,
        HAS_C=len(sources) > 2,
    )


def _three(
    sources: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``sources`` padded to three with the first (unread: HAS_B / HAS_C)."""
    assert 1 <= len(sources) <= 3, "one to three sources"
    return list(sources) + [sources[0]] * (3 - len(sources))


def sum_topk_direct(
    out: torch.Tensor, topk_weights: torch.Tensor, per_k: torch.Tensor
) -> None:
    """``sum_topk`` over a dense ``[tokens * top_k, hidden]`` buffer."""
    num_tokens, top_k = topk_weights.shape
    if num_tokens == 0:
        return
    hidden = out.shape[-1]
    grid = (num_tokens, triton.cdiv(hidden, _COMBINE_BLOCK))
    _sum_topk_kernel[grid](
        out,
        topk_weights.contiguous(),
        per_k,
        per_k,
        per_k,
        per_k,
        per_k,
        per_k,
        hidden,
        TOPK=top_k,
        DIRECT=True,
        HAS_B=False,
        BLOCK=_COMBINE_BLOCK,
        HAS_C=False,
    )


def pick_rows(
    per_k: torch.Tensor, sources: list[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """``per_k[i]`` = the row of pair ``i`` from the first source holding it,
    zeros where none does (see ``sum_topk``)."""
    if per_k.shape[0] == 0:
        return
    hidden = per_k.shape[-1]
    (a, a_rows), (b, b_rows), (c, c_rows) = _three(sources)
    grid = (per_k.shape[0], triton.cdiv(hidden, _COMBINE_BLOCK))
    _pick_rows_kernel[grid](
        per_k,
        a,
        a_rows.contiguous().view(-1),
        b,
        b_rows.contiguous().view(-1),
        c,
        c_rows.contiguous().view(-1),
        hidden,
        HAS_B=len(sources) > 1,
        BLOCK=_COMBINE_BLOCK,
        HAS_C=len(sources) > 2,
    )


def combine_exact(
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    sources: list[tuple[torch.Tensor, torch.Tensor]],
    ep_rank: int,
    all_reduce: Any,
    chunk: int = EXACT_CHUNK,
) -> None:
    """The routed sum independent of where each expert was computed.

    Every (token, top-k) pair is computed by exactly one expert-parallel rank
    (one tier of it): each rank writes the pairs it holds into a dense
    ``[tokens * top_k, hidden]`` buffer, zeros elsewhere, and the all-reduce
    adds them; with one nonzero term per element that sum is exact. Each
    rank then has every pair, and rank 0 sums them in k order (``sum_topk``);
    the other ranks output zeros, so the layer's usual all-reduce of the
    ranks' outputs adds this sum to nothing. Done in chunks of ``chunk``
    tokens to bound the buffer."""
    num_tokens, top_k = topk_weights.shape
    hidden = output.shape[-1]
    if ep_rank != 0:
        output.zero_()
    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        per_k = torch.empty(
            ((end - start) * top_k, hidden),
            dtype=sources[0][0].dtype,
            device=output.device,
        )
        pick_rows(per_k, [(rows, rmap[start:end]) for rows, rmap in sources])
        per_k = all_reduce(per_k)
        if ep_rank == 0:
            sum_topk_direct(output[start:end], topk_weights[start:end], per_k)


class _Captured:
    """The expert-grouped outputs of one fused-MoE call, from its
    ``output_hook``: (gemm2 output, row map)."""

    def __init__(self) -> None:
        self.rows: torch.Tensor | None = None
        self.row_map: torch.Tensor | None = None

    def __call__(
        self, gemm2_output: torch.Tensor, act: torch.Tensor, row_map: torch.Tensor
    ) -> None:
        self.rows, self.row_map = gemm2_output, row_map

    def source(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.rows is not None and self.row_map is not None, (
            "the fused MoE did not call its output_hook"
        )
        return self.rows, self.row_map


@functools.cache
def fused_moe_has_output_hook() -> bool:
    return "output_hook" in inspect.signature(XpuFusedMoe.apply).parameters


class PlacementTrace:
    """Debug: where each routed expert of every token was computed.

    Set ``VLLM_XPU_MOE_PLACEMENT_TRACE=<prefix>`` and create ``<prefix>.on``
    while the request of interest runs alone; each rank appends one JSON line
    per MoE layer call to ``<prefix>.rank<R>.jsonl``: the layer, the call
    number, and per token and top-k the physical expert id, the router
    weight, and on this rank ``F<row>`` (device tier), ``S<row>`` (host
    tier), ``C`` (computed on this rank's CPU, --cpu-offload-pinned-gb) or
    ``-`` (another rank), plus the logical expert where this rank
    knows it (expert store on). Graph replays run no Python, so only eager
    steps are traced (boot with --enforce-eager for a full trace)."""

    MAX_LINES = 200_000

    def __init__(self, prefix: str, ep_rank: int):
        self.prefix = prefix
        self.path = f"{prefix}.rank{ep_rank}.jsonl"
        self.ep_rank = ep_rank
        self.calls: dict[str, int] = {}
        self.lines = 0

    def active(self) -> bool:
        import os

        accelerator = torch.accelerator
        capturing = (
            accelerator.is_available() and accelerator.current_stream().is_capturing()
        )
        return (
            self.lines < self.MAX_LINES
            and not capturing
            and os.path.exists(self.prefix + ".on")
        )

    def record(
        self,
        tiers: "ExpertTiers",
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        import json

        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        name = tiers.layer_name
        call = self.calls.get(name, 0)
        self.calls[name] = call + 1
        assert tiers.fast_map is not None and tiers.slow_map is not None
        with gpu_sync_allowed():
            ids = topk_ids.long().cpu()
            weights = topk_weights.float().cpu()
            fast_map = tiers.fast_map.cpu()
            slow_map = tiers.slow_map.cpu()
        valid = (ids >= 0) & (ids < tiers.num_global)
        safe = torch.where(valid, ids, torch.zeros_like(ids))
        fast_row = torch.where(valid, fast_map[safe].long(), -1)
        slow_row = torch.where(valid, slow_map[safe].long(), -1)
        logical = tiers.slot_logical
        # This rank's tier per physical id when slots are computed on the CPU
        # (neither map holds those: they would read as another rank's).
        cpu_tier = (
            getattr(tiers, "_tier_of_global", None)
            if getattr(tiers, "cpu", None) is not None
            else None
        )
        where: list[list[str]] = []
        experts: list[list[int]] = []
        for t in range(ids.shape[0]):
            row_where, row_logical = [], []
            for k in range(ids.shape[1]):
                g = int(ids[t, k])
                if fast_row[t, k] >= 0:
                    row_where.append(f"F{int(fast_row[t, k])}")
                elif slow_row[t, k] >= 0:
                    row_where.append(f"S{int(slow_row[t, k])}")
                elif (
                    cpu_tier is not None
                    and 0 <= g < len(cpu_tier)
                    and (
                        int(cpu_tier[g]) == 2  # expert_tier_map.HOST
                    )
                ):
                    row_where.append("C")
                else:
                    row_where.append("-")
                slot = tiers.local_of.get(g, -1) if logical is not None else -1
                row_logical.append(int(logical[slot]) if slot >= 0 else -1)
            where.append(row_where)
            experts.append(row_logical)
        entry = {
            "layer": name,
            "call": call,
            "rank": self.ep_rank,
            "tokens": int(ids.shape[0]),
            "physical": ids.tolist(),
            "weight": [[round(float(x), 6) for x in r] for r in weights],
            "where": where,
            "logical": experts if logical is not None else None,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.lines += 1


class EPIdsContract(enum.Enum):
    """Who handles the routed ids a rank does not own, per vllm-xpu-kernels."""

    # XpuFusedMoe.apply skips negative, out-of-range and unowned ids and clears
    # its own rows_per_expert: the layer passes the routed ids as they are.
    OP_HANDLES_IDS = "op"
    # apply takes ids_valid / rows_per_expert: the layer maps unrouted ids to
    # an unowned sentinel and zeroes the counts in one kernel.
    IDS_VALID = "ids_valid"
    # Neither: the layer maps unrouted ids with torch.where and the fused MoE
    # keeps its padding-row mask.
    PADDING_MASK = "padding_mask"


# Class attributes XpuFusedMoe sets when apply() itself handles unowned ids
# and clears rows_per_expert. Both are needed for EPIdsContract.OP_HANDLES_IDS.
FUSED_MOE_OP_MARKERS = ("HANDLES_UNOWNED_IDS", "CLEARS_ROWS_PER_EXPERT")


@functools.cache
def fused_moe_ep_contract() -> EPIdsContract:
    """What the installed XpuFusedMoe does with the routed ids, read once."""
    if all(getattr(XpuFusedMoe, name, False) for name in FUSED_MOE_OP_MARKERS):
        return EPIdsContract.OP_HANDLES_IDS
    params = inspect.signature(XpuFusedMoe.apply).parameters
    if "ids_valid" in params and "rows_per_expert" in params:
        return EPIdsContract.IDS_VALID
    return EPIdsContract.PADDING_MASK


def ep_ids_contract_for(moe_config: FusedMoEConfig) -> EPIdsContract:
    """The contract an expert-parallel layer uses. With EPLB the layer keeps
    rewriting the ids itself: rebalancing changes which experts a rank owns,
    and that path was not validated with the op filtering them."""
    contract = fused_moe_ep_contract()
    if (
        contract is EPIdsContract.OP_HANDLES_IDS
        and moe_config.moe_parallel_config.enable_eplb
    ):
        contract = EPIdsContract.IDS_VALID
    if moe_config.ep_size <= 1:
        return contract
    logger.info_once(
        "XPU MoE expert parallelism: %s.",
        {
            EPIdsContract.OP_HANDLES_IDS: "the fused MoE filters unowned ids and "
            "clears its expert counts; the routed ids are passed as they are",
            EPIdsContract.IDS_VALID: "unrouted ids are mapped to an unowned "
            "sentinel and the expert counts zeroed in one kernel (ids_valid)",
            EPIdsContract.PADDING_MASK: "vllm-xpu-kernels' XpuFusedMoe.apply takes "
            "no ids_valid / rows_per_expert; the fused MoE keeps its padding-row "
            "mask",
        }[contract],
    )
    return contract


class XPUExperts(mk.FusedMoEExpertsModular):
    # Extra keyword arguments for every XpuFusedMoe.apply call.
    impl_apply_kwargs: dict = {}
    # Expert slots in device and host tiers, set by ExpertTiers.finalize.
    tiers: ExpertTiers | None = None
    # How a token's routed experts are summed (VLLM_XPU_MOE_COMBINE) and the
    # debug placement trace; set per instance in __init__.
    combine_mode: str = "split"
    placement_trace: "PlacementTrace | None" = None
    # apply() reads the weights of the XpuFusedMoe it builds on first use, and
    # stages those itself (staged_impl).
    reads_weight_args = False

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self.fused_moe_impl: XpuFusedMoe | None = None
        is_xe2_or_xe3 = torch.ops._xpu_C.is_xe2_arch() or torch.ops._xpu_C.is_xe3_arch()
        self._expects_unquantized_inputs = is_xe2_or_xe3
        self.ep_contract = ep_ids_contract_for(moe_config)
        # VLLM_XPU_MOE_COMBINE: "split" is the stock sum (see envs.py).
        self.combine_mode: str = envs.VLLM_XPU_MOE_COMBINE
        trace = envs.VLLM_XPU_MOE_PLACEMENT_TRACE
        self.placement_trace = (
            PlacementTrace(trace, moe_config.ep_rank) if trace else None
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return self._expects_unquantized_inputs

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _build_impl(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk: int,
        activation: MoEActivation,
        expert_map: torch.Tensor | None,
    ) -> None:
        if (
            self.quant_config is not None
            and self.quant_config.weight_quant_dtype == "mxfp4"
        ):
            w1 = w1.view(torch.float4_e2m1fn_x2)
            w2 = w2.view(torch.float4_e2m1fn_x2)
        # XpuFusedMoe relayouts the loaded [E, N, K] weights and scales for
        # this device's grouped GEMM, rebinding the parameters' storage.
        self.fused_moe_impl = XpuFusedMoe(
            w13=w1,
            w13_scales=self.w1_scale,
            w13_bias=self.w1_bias,
            w2=w2,
            w2_scales=self.w2_scale,
            w2_bias=self.w2_bias,
            n_experts_per_token=topk,
            activation=activation.value,
            num_experts=self.moe_config.num_local_experts,
            ep_rank=self.moe_config.ep_rank,
            ep_size=self.moe_config.ep_size,
            expert_map=expert_map,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
        )

    def _tier_impls(self, w1: torch.Tensor, w2: torch.Tensor, topk: int):
        assert self.tiers is not None
        assert self.w1_scale is not None and self.w2_scale is not None
        return self.tiers.impls(
            w1,
            self.w1_scale,
            w2,
            self.w2_scale,
            topk,
            self.moe_config.activation.value,
            self.gemm1_clamp_limit,
        )

    def eplb_expert_weights(self, layer: torch.nn.Module) -> list[ExpertWeight] | None:
        if self.tiers is None:
            return None
        return list(self.tiers.eplb_weights())

    def finalize_weight_storage(self, layer: torch.nn.Module) -> None:
        """Build the XpuFusedMoe now, so its relayout lands before EPLB (or
        anything else) takes views of the expert tensors."""
        w13 = getattr(layer, "w13_weight", None)
        w2 = getattr(layer, "w2_weight", None)
        if w13 is None or w2 is None:
            return
        if self.tiers is not None:
            self._tier_impls(w13, w2, self.moe_config.experts_per_token)
            return
        if self.fused_moe_impl is not None:
            return
        self._build_impl(
            w13,
            w2,
            self.moe_config.experts_per_token,
            self.moe_config.activation,
            getattr(layer, "expert_map", None),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if self.tiers is not None:
            self._apply_tiered(output, hidden_states, w1, w2, topk_weights, topk_ids)
            return
        if self.fused_moe_impl is None:
            # Methods that skip finalize_weight_storage build it on first use.
            self._build_impl(w1, w2, topk_ids.size(-1), activation, expert_map)
        assert self.fused_moe_impl is not None
        impl = staged_impl(self.fused_moe_impl, hidden_states.size(0), "experts")
        ep_kwargs: dict = {}
        if (
            self.moe_config.ep_size > 1
            and self.ep_contract is EPIdsContract.OP_HANDLES_IDS
        ):
            # The fused MoE reads topk_ids as a dense array; a router's strided
            # view must reach it contiguous (a no-op when it already is).
            topk_ids = topk_ids.contiguous()
        elif self.moe_config.ep_size > 1:
            # Avoid topk_ids = -1, remap them to an
            # expert not owned by this rank so they are skipped. The same
            # kernel zeroes rows_per_expert, and with every id valid the fused
            # MoE skips its padding-row mask.
            sentinel = global_num_experts - 1 if self.moe_config.ep_rank == 0 else 0
            if self.ep_contract is EPIdsContract.IDS_VALID:
                topk_ids, rows_per_expert = sentinel_ids_and_counts(
                    topk_ids, sentinel, impl.num_experts
                )
                ep_kwargs = {"ids_valid": True, "rows_per_expert": rows_per_expert}
            else:
                topk_ids = torch.where(topk_ids < 0, sentinel, topk_ids)
        if self.moe_config.ep_size > 1 and self._combine_mode(a1q_scale) == "exact":
            captured = _Captured()
            impl.apply(
                output=output,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                a1q_scale=a1q_scale,
                output_hook=captured,
                **self.impl_apply_kwargs,
                **ep_kwargs,
            )
            self._combine_exact(output, topk_weights, [captured.source()])
            return
        impl.apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1q_scale=a1q_scale,
            **self.impl_apply_kwargs,
            **ep_kwargs,
        )

    def _combine_mode(self, a1q_scale: torch.Tensor | None = None) -> str:
        """The VLLM_XPU_MOE_COMBINE mode this call runs; raises when a mode
        other than "split" cannot run here (it needs the fused MoE's
        output_hook, which takes unquantized activations and no W4A8)."""
        mode = self.combine_mode
        if mode == "split":
            return mode
        why = None
        if not fused_moe_has_output_hook():
            why = "the installed vllm-xpu-kernels' XpuFusedMoe.apply has no output_hook"
        elif self.impl_apply_kwargs.get("w4a8"):
            why = "the W4A8 experts path runs no output_hook"
        elif a1q_scale is not None:
            why = "the activations are quantized"
        elif mode == "exact" and getattr(self.moe_config, "dp_size", 1) > 1:
            why = "data parallelism: the ranks do not hold the same tokens"
        if why is not None:
            raise ValueError(f"VLLM_XPU_MOE_COMBINE={mode}: not available: {why}")
        logger.info_once(
            "XPU MoE combine: %s (VLLM_XPU_MOE_COMBINE). %s",
            mode,
            "One sum over both expert tiers of a rank, independent of the tiers."
            if mode == "rank"
            else "Per-expert outputs exchanged between the expert-parallel "
            "ranks and summed once in top-k order: independent of the rank "
            "and tier of every expert.",
        )
        return mode

    def _combine_exact(
        self,
        output: torch.Tensor,
        topk_weights: torch.Tensor,
        sources: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        if self.moe_config.ep_size > 1:
            from vllm.distributed.parallel_state import get_ep_group

            all_reduce = get_ep_group().all_reduce
        else:

            def all_reduce(t: torch.Tensor) -> torch.Tensor:
                return t

        combine_exact(
            output, topk_weights, sources, self.moe_config.ep_rank, all_reduce
        )

    def _apply_tiered(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        """Experts in the fast slots plus experts in the slow slots, summed;
        with host slots, plus the experts the CPU computes (their inputs go
        to the host before the card's experts are enqueued, and the host
        waits for those inputs only, so the CPU works while the card does)."""
        tiers = self.tiers
        assert tiers is not None
        fast, slow = self._tier_impls(w1, w2, topk_ids.size(-1))
        if self.moe_config.ep_size > 1:
            topk_ids = remap_unrouted(
                topk_ids, tiers.num_global, self.moe_config.ep_rank
            )
        if self.placement_trace is not None and self.placement_trace.active():
            self.placement_trace.record(tiers, topk_ids, topk_weights)
        mode = self._combine_mode()
        cpu = getattr(tiers, "cpu", None)
        pending = (
            None
            if cpu is None
            else cpu.post(tiers, hidden_states, topk_ids, topk_weights)
        )
        if mode != "split":
            if cpu is not None:
                logger.warning_once(
                    "XPU MoE combine %s with experts computed on the CPU: the "
                    "CPU computes each of its pairs as its own row, summed with "
                    "the card's rows in top-k order. The sum does not depend on "
                    "the rank or the card tier of any expert; an expert that "
                    "moves between the CPU and the card can change its own row "
                    "in the last bit of some values (the CPU kernel adds its "
                    "fp32 partial sums in another order than the card's).",
                    mode,
                )
            captured = (_Captured(), _Captured())
            fast.apply(
                output=output,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output_hook=captured[0],
                **self.impl_apply_kwargs,
            )
            # The slow call's own sum lands here and is not used.
            slow_out = torch.empty_like(output)
            staged_impl(slow, hidden_states.size(0), "slow").apply(
                output=slow_out,
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output_hook=captured[1],
                **self.impl_apply_kwargs,
            )
            sources = [c.source() for c in captured]
            if pending is not None:
                assert cpu is not None
                host = cpu.finish_pairs(pending, output.device)
                if host is not None:
                    sources.append((host.rows, host.row_map))
            if mode == "rank":
                sum_topk(output, topk_weights, sources)
            else:
                self._combine_exact(output, topk_weights, sources)
            return
        fast.apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            **self.impl_apply_kwargs,
        )
        slow_out = torch.empty_like(output)
        # Large steps read the slow tier from a device copy
        # (--cpu-offload-stage-min-tokens).
        staged_impl(slow, hidden_states.size(0), "slow").apply(
            output=slow_out,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            **self.impl_apply_kwargs,
        )
        output.add_(slow_out)
        if pending is not None:
            assert cpu is not None
            host_out = cpu.finish(pending, output.device)
            if host_out is not None:
                output.add_(host_out)


class XPUExpertsFp8(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsMxFp8(XPUExpertsFp8):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        assert quant_config.quant_dtype == "mxfp8"

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp8Static, None),
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsBlockFp8(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsWNA16(XPUExperts):
    """W4A16 INT4-symmetric MoE backed by `xpu_fused_moe(is_int4=True)`.

    Weight layout when `is_int4=True` (per `xpu_fused_moe` docstring):
        w13: [num_experts, 2*inter_size, hidden_size]   contiguous int4-packed
        w13_scales: [num_experts, 2*inter_size, hidden_size // group_size]
        w2:  [num_experts, hidden_size, inter_size]     contiguous int4-packed
        w2_scales:  [num_experts, hidden_size, inter_size // group_size]

    Pairs with `INCXPULinearMethod` for the linear layers; together they
    cover full-attn + MoE on Intel XPU end-to-end without IPEX.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kInt4Static, None),
            (kInt4Static32, None),
        )


def xpu_w4a8_moe_available() -> bool:
    """Whether vllm-xpu-kernels has the W4A8 grouped GEMM."""
    return current_platform.is_xpu() and hasattr(
        torch.ops._xpu_C, "cutlass_grouped_gemm_w4a8"
    )


class XPUExpertsW4A8Int(XPUExpertsWNA16):
    """INT4 symmetric g128 experts with dynamic per-token INT8 activations.

    Same weights and layout as `XPUExpertsWNA16`. vllm-xpu-kernels quantizes the
    tokens to int8 once before each grouped GEMM and runs the Xe2 s8 x s4 DPAS
    grouped GEMM on the int4 weights as stored (`XpuFusedMoe.apply(w4a8=True)`).
    Selected with `quantization_config.moe.activation: int8_per_token`. A layer
    the W4A8 GEMM cannot take (activation other than SiLU, a clamp, N or K not
    a multiple of 64) keeps the W4A16 path, with a warning.
    """

    impl_apply_kwargs = {"w4a8": True}

    @staticmethod
    def _supports_current_device() -> bool:
        return xpu_w4a8_moe_available()

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return activation_key == kInt8DynamicTokenSym and (
            XPUExpertsWNA16._supports_quant_scheme(weight_key, None)
        )

    def apply(self, *args, **kwargs):
        super().apply(*args, **kwargs)
        if getattr(self, "_w4a8_reported", False) or self.fused_moe_impl is None:
            return
        self._w4a8_reported = True
        # The kernels record on the first call why a layer cannot run W4A8
        # (None: it ran W4A8).
        reason = getattr(self.fused_moe_impl, "_w4a8_why", None)
        if reason is None:
            logger.info_once(
                "XPU MoE W4A8: int4 g128 experts with dynamic per-token int8 "
                "activations (s8 x s4 grouped GEMM)."
            )
        else:
            logger.warning_once(
                "XPU MoE W4A8 requested, but %s: this layer keeps W4A16.", reason
            )


class XPUExpertsMxFp4(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # K = a1q.size(-1). When activations are pre-quantized packed mxfp4,
        # K is the packed hidden_size (= logical / 2); the kernel output is at
        # logical hidden_size (2 * K). When unquantized (bf16), K is already
        # the logical size.
        logical_K = K if self.expects_unquantized_inputs else 2 * K
        return (0,), (0,), (M, logical_K)

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

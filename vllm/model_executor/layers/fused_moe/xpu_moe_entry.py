# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU MoE entry without repeated work, one switch per piece.

In a MoE layer the same input row is read by several INT8 consumers: the
shared expert's gate_up, the router gate and the shared-expert gate (when
the checkpoint carries them as INT8 W8A8), and the W4A8 routed experts. Each
quantized it itself, with the same call on the same tensor.

``VLLM_XPU_MOE_ENTRY_QUANT=1``: the row is quantized once
(``ops.scaled_int8_quant``, the call each INT8 linear makes) and every
consumer reads those int8 rows and scales (``entry_rows``); the W4A8
experts remap or gather them (vllm-xpu-kernels ``x_q``). Same bits: the
same op on the same tensor, so each consumer computes what it did.

``VLLM_XPU_MOE_GATES_MERGED=1``: an INT8 router gate and an INT8
shared-expert gate run as one GEMM over the concatenated weights and
per-row scales, writing each gate's output to its own tensor. Each element
is triton_scaled_mm's: the exact int32 dot, ``scale_a * acc`` then
``* scale_b`` in fp32, rounded to the activation dtype; it depends only on
its own row and column, so the split changes no bit. It runs where the gates'
INT8 linear runs its Triton GEMM (at most VLLM_XPU_INT8_ONEDNN_MIN_ROWS
rows on the oneDNN kernel); above that each gate keeps its own GEMM.
``=2`` runs the merged Triton GEMM at every row count. Above the split it
stands in for the gates' oneDNN GEMMs, whose epilogue (the order in which
it applies the two scales to the exact int32 dot) cannot be read on the
CPU: tier 2 there, at most one bf16 ulp per element when oneDNN applies
the same two fp32 scale products in another order; the card test measures
it. At or below the split it is ``=1``.

``VLLM_XPU_MOE_SHARED_IN_GATHER=1``: the gated shared expert returns its
output ungated, and the routed W4A8 experts' gather adds
``sigmoid(gate) * shared`` to the routed sum with the bf16 roundings of the
multiply and of the shared + routed add (vllm-xpu-kernels ``moe_gather``
with ``shared``). Same bits; the multiply and the add do not run. When a call
cannot fold (another experts path), the MoE op does the multiply and add
itself, as the layer did. ``=2`` (tier 2, different bits) rounds
``routed + gate * shared`` once in fp32.

``VLLM_XPU_MOE_SHARED_GATE_IN_GATHER=1`` (with the fold above; tier 2): the
shared expert hands the gather its gate's logits, and the gather computes
``T(1 / (1 + exp(-logit)))`` in fp32 with the device's ``exp`` and rounds it
to the activation dtype, where torch's sigmoid ran as its own kernel. The
gate can differ from torch's by one ulp where the two fp32 sigmoids round to
different sides of a bf16 boundary; nothing else moves. One launch fewer per
layer. Needs a vllm-xpu-kernels whose gather takes ``gate_logits``.

``VLLM_XPU_MOE_TOPK_ROUTE=1``: see ``topk_route``.

The layer decides at load (``resolve``) and logs one line naming what each
piece covers; the first forward logs which consumers read the rows."""

from contextlib import contextmanager

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import entry_rows
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


# --------------------------------------------------------------------- INT8
def int8_kernel(layer: torch.nn.Module | None):
    """The INT8 W8A8 kernel of ``layer`` when its activations are dynamic,
    symmetric, per token (the ones ``entry_rows`` serves), else None."""
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )

    kernel = getattr(getattr(layer, "scheme", None), "kernel", None)
    if not isinstance(kernel, TritonInt8ScaledMMLinearKernel):
        return None
    c = kernel.config
    if c.is_static_input_scheme or not c.input_symmetric:
        return None
    return kernel


def _native_quant() -> bool:
    # The engine's INT8 linear and the W4A8 experts both call the native op
    # when it is present; their fallbacks differ (the experts' torch rule
    # does not round the scale to bf16), so the shared rows need the op.
    if not current_platform.is_xpu():
        return _TEST_NATIVE
    from vllm._xpu_ops import native_int8_quant_available

    return native_int8_quant_available()


_TEST_NATIVE = False  # CPU tests stand in for the native op


def quantize_rows(x: torch.Tensor):
    """The call every consumer makes: (int8 [T, H], fp32 [T, 1])."""
    from vllm import _custom_ops as ops

    q, s, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    return q, s


# ------------------------------------------------------------ merged gates
@triton.jit
def _gates_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c1_ptr,
    c2_ptr,
    M,
    N,
    N1,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c1m,
    stride_c2m,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """triton_scaled_mm's scaled_mm_kernel (per-token scale_a, per-column
    scale_b, int32 accumulator, no bias), with the columns below N1 stored to
    c1 and the rest to c2."""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N
    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    a_ptrs = a_ptr + stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    b_ptrs = b_ptr + stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        a = tl.load(a_ptrs, mask=masks_am[:, None] & masks_k[None, :])
        b = tl.load(b_ptrs, mask=masks_k[:, None] & masks_bn[None, :])
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)
        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    scale_a = tl.load(scale_a_ptr + offsets_am[:, None], masks_am[:, None])
    accumulator = scale_a * accumulator.to(tl.float32)
    scale_b = tl.load(scale_b_ptr + offsets_bn[:, None], masks_bn[:, None])
    accumulator = scale_b.T * accumulator.to(tl.float32)
    c = accumulator.to(c1_ptr.type.element_ty)

    rows = offsets_am[:, None]
    cols = offsets_bn[None, :]
    in_c1 = (rows < M) & (cols < N1)
    tl.store(c1_ptr + stride_c1m * rows + cols, c, mask=in_c1)
    in_c2 = (rows < M) & (cols >= N1) & (cols < N)
    tl.store(c2_ptr + stride_c2m * rows + (cols - N1), c, mask=in_c2)


def gates_mm(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    w_cat: torch.Tensor,
    s_cat: torch.Tensor,
    n1: int,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x_q`` int8 [M, K], ``x_s`` fp32 [M, 1], ``w_cat`` int8 [N, K]
    (router rows, then shared-gate rows), ``s_cat`` fp32 [N]: the router
    output [M, n1] and the shared gate output [M, N - n1], both contiguous,
    each element the one triton_scaled_mm gives for its own gate."""
    M, K = x_q.shape
    N = w_cat.shape[0]
    c1 = torch.empty((M, n1), dtype=out_dtype, device=x_q.device)
    c2 = torch.empty((M, N - n1), dtype=out_dtype, device=x_q.device)
    if M == 0:
        return c1, c2
    # triton_scaled_mm's XPU tile for <= 64 rows; the bits do not depend on it.
    bm = max(1, triton.next_power_of_2(M)) if M <= 64 else 64
    bn, bk = 32, 128
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _gates_mm_kernel[grid](
        x_q,
        w_cat,
        x_s,
        s_cat,
        c1,
        c2,
        M,
        N,
        n1,
        K,
        x_q.stride(0),
        x_q.stride(1),
        w_cat.stride(1),
        w_cat.stride(0),
        c1.stride(0),
        c2.stride(0),
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        num_warps=8,
    )
    return c1, c2


def _gate_rows(layer: torch.nn.Module, kernel) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate's int8 weight as [N, K] and its scales as fp32 [N]."""
    w_q, w_s, _, _, _ = kernel._get_layer_params(layer)
    n = layer.output_size if hasattr(layer, "output_size") else None
    k = layer.input_size if hasattr(layer, "input_size") else None
    # After the kernel's load the weight is a [K, N] view of the [N, K] rows.
    w = w_q.t() if (n is not None and tuple(w_q.shape) == (k, n)) else w_q
    s = w_s.reshape(-1).to(torch.float32)
    if s.numel() == 1:
        s = s.expand(w.shape[0])
    return w, s


# ------------------------------------------------------------------- plan
class EntryPlan:
    """What the XPU MoE entry does for one MoE runner (decided at load)."""

    def __init__(self) -> None:
        self.quant = False
        self.gates: tuple[torch.nn.Module, torch.nn.Module] | None = None
        self.gates_row_limit: int | None = None
        # VLLM_XPU_MOE_GATES_MERGED=2: the row count above which the merged
        # GEMM stands in for the gates' oneDNN GEMMs (different bits there).
        self.gates_tier2_above: int | None = None
        self.fold_module: torch.nn.Module | None = None
        self.single_round = False
        # VLLM_XPU_MOE_SHARED_GATE_IN_GATHER: the gather takes the expert
        # gate's logits and applies the sigmoid itself.
        self.gate_logits = False
        self.logged_readers = False

    @property
    def active(self) -> bool:
        return self.quant or self.gates is not None or self.fold_module is not None

    @contextmanager
    def open(self, runner, x: torch.Tensor):
        """The entry scope of one forward of ``runner`` on ``x``."""
        rows = entry_rows.EntryRows(x, None, None)
        rows.share_rows = self.quant
        merged = (
            self.gates is not None
            and x.dim() == 2
            and (self.gates_row_limit is None or x.shape[0] <= self.gates_row_limit)
        )
        if self.quant or merged:
            rows.q, rows.s = quantize_rows(x)
        if merged:
            router, sgate = self.gates
            n1 = runner._xpu_gates_n1
            out_r, out_g = gates_mm(
                rows.q,
                rows.s,
                runner._xpu_gates_w,
                runner._xpu_gates_s,
                n1,
                x.dtype,
            )
            rows.outputs[id(router)] = out_r
            rows.outputs[id(sgate)] = out_g
        rows.fold_module = self.fold_module
        rows.single_round = self.single_round
        rows.gate_logits = self.gate_logits
        with entry_rows.scope(rows):
            yield rows
        if not self.logged_readers:
            self.logged_readers = True
            logger.info_once(
                "XPU MoE entry, first forward of %s: %s%s.",
                getattr(runner, "layer_name", "a MoE layer"),
                "int8 rows read by: " + ", ".join(rows.readers)
                if rows.readers
                else "no consumer read the entry rows",
                "; shared expert gated in the routed gather"
                if rows.folded
                else (
                    "; shared expert gated and added by the MoE op (the experts "
                    "did not fold it)"
                    if self.fold_module is not None
                    else ""
                ),
                scope="process",
            )


def _xpu_experts(runner):
    mk = getattr(runner._quant_method, "moe_kernel", None)
    impl = getattr(mk, "impl", None)
    return getattr(impl, "fused_experts", None)


def _w4a8_experts(runner):
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsW4A8Int,
    )

    fe = _xpu_experts(runner)
    return fe if isinstance(fe, XPUExpertsW4A8Int) else None


def kernels_take(name: str) -> bool:
    """Whether vllm-xpu-kernels' W4A8 apply takes ``x_q`` / ``shared``."""
    try:
        from vllm_xpu_kernels import moe_w4a8
    except ImportError:
        return False
    if name == "x_q":
        return bool(getattr(moe_w4a8, "TAKES_ENTRY_ROWS", False))
    if name == "gate_logits":
        probe = getattr(moe_w4a8, "gather_takes_logits", None)
        return probe is not None and bool(probe())
    return bool(getattr(moe_w4a8, "TAKES_SHARED", False)) and bool(
        moe_w4a8.gather_takes_shared()
    )


def resolve(runner, force_platform: bool = False) -> EntryPlan | None:
    """The entry plan of ``runner`` under the four switches, logged once per
    outcome; None when no piece applies."""
    want_quant = envs.VLLM_XPU_MOE_ENTRY_QUANT
    want_gates = envs.VLLM_XPU_MOE_GATES_MERGED
    want_fold = envs.VLLM_XPU_MOE_SHARED_IN_GATHER
    if not (want_quant or want_gates or want_fold):
        return None
    if not (current_platform.is_xpu() or force_platform):
        return None
    plan = EntryPlan()
    why: list[str] = []
    shared = runner._shared_experts
    shared_layer = getattr(shared, "_layer", None)
    expert_gate = getattr(shared_layer, "expert_gate", None)
    gate = runner.gate
    simple = (
        runner.routed_input_transform is None
        and not runner.do_naive_dispatch_combine
        and runner.moe_config.pcp_size == 1
    )
    if not simple:
        why.append("routed input transform, naive dispatch or PCP")

    if want_quant and simple:
        consumers = []
        if int8_kernel(gate) is not None:
            consumers.append("router gate")
        if int8_kernel(getattr(shared_layer, "gate_up_proj", None)) is not None:
            consumers.append("shared gate_up")
        if int8_kernel(expert_gate) is not None:
            consumers.append("shared-expert gate")
        if _w4a8_experts(runner) is not None and kernels_take("x_q"):
            consumers.append("W4A8 routed experts")
        if not _native_quant():
            why.append("entry quant: no native per-token int8 op")
        elif len(consumers) < 2:
            why.append(f"entry quant: {len(consumers)} INT8 consumer(s)")
        else:
            plan.quant = True
            plan.quant_consumers = consumers

    if want_gates and simple:
        kr, kg = int8_kernel(gate), int8_kernel(expert_gate)
        if kr is None or kg is None:
            why.append("merged gates: router and shared-expert gate not both INT8")
        elif (
            getattr(gate, "bias", None) is not None
            or getattr(expert_gate, "bias", None) is not None
        ):
            why.append("merged gates: a gate has a bias")
        else:
            from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
                XPU_INT8_ONEDNN_MIN_ROWS,
                XPUInt8ScaledMMLinearKernel,
            )
            from vllm.model_executor.utils import refill_derived

            wr, sr = _gate_rows(gate, kr)
            wg, sg = _gate_rows(expert_gate, kg)
            if wr.shape[1] != wg.shape[1] or sr.dtype != sg.dtype:
                why.append("merged gates: shapes differ")
            else:
                refill_derived(runner, "_xpu_gates_w", torch.cat([wr, wg], 0))
                refill_derived(runner, "_xpu_gates_s", torch.cat([sr, sg], 0))
                runner._xpu_gates_n1 = wr.shape[0]
                plan.gates = (gate, expert_gate)
                if isinstance(kr, XPUInt8ScaledMMLinearKernel) or isinstance(
                    kg, XPUInt8ScaledMMLinearKernel
                ):
                    if want_gates == 2:
                        # Tier 2 above the split: the merged Triton GEMM
                        # replaces the gates' oneDNN GEMMs there.
                        plan.gates_tier2_above = XPU_INT8_ONEDNN_MIN_ROWS
                    else:
                        plan.gates_row_limit = XPU_INT8_ONEDNN_MIN_ROWS

    if want_fold:
        from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
            ZeroExpertRouter,
        )
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP

        fe = _w4a8_experts(runner)
        cfg = runner.moe_config
        blockers = {
            "no gated Qwen3-MoE shared expert": not (
                isinstance(shared_layer, Qwen3MoeMLP) and expert_gate is not None
            ),
            "routed experts not XPU W4A8": fe is None,
            "vllm-xpu-kernels' gather takes no shared term": not kernels_take("shared"),
            "routed output reduced by its kernel": runner._fused_output_is_reduced,
            "routed scaling or output transform": runner.routed_scaling_factor != 1.0
            or runner.routed_output_transform is not None,
            "zero-expert router": isinstance(runner.router, ZeroExpertRouter),
            "sequence parallel, naive dispatch, PCP or DBO": cfg.is_sequence_parallel
            or not simple
            or runner.enable_dbo,
            "padded hidden size": shared_layer is not None
            and getattr(shared_layer.gate_up_proj, "input_size", cfg.hidden_dim)
            != cfg.hidden_dim,
            "exact combine or expert tiers": fe is not None
            and (fe.combine_mode != "split" or fe.tiers is not None),
        }
        hit = [k for k, v in blockers.items() if v]
        if hit:
            why.append("shared in gather: " + "; ".join(hit))
        else:
            plan.fold_module = shared_layer
            plan.single_round = want_fold == 2
            if envs.VLLM_XPU_MOE_SHARED_GATE_IN_GATHER:
                if kernels_take("gate_logits"):
                    plan.gate_logits = True
                else:
                    why.append(
                        "sigmoid in gather: vllm-xpu-kernels' gather takes no "
                        "gate logits"
                    )
    elif envs.VLLM_XPU_MOE_SHARED_GATE_IN_GATHER:
        why.append("sigmoid in gather: needs VLLM_XPU_MOE_SHARED_IN_GATHER")

    logger.info_once(
        "XPU MoE entry (VLLM_XPU_MOE_ENTRY_QUANT=%d, VLLM_XPU_MOE_GATES_MERGED=%d, "
        "VLLM_XPU_MOE_SHARED_IN_GATHER=%d): %s%s",
        int(want_quant),
        int(want_gates),
        int(want_fold),
        "; ".join(
            p
            for p in (
                "one per-token int8 quant of the MoE input for "
                + ", ".join(getattr(plan, "quant_consumers", []))
                if plan.quant
                else "",
                "router + shared-expert gate as one INT8 GEMM"
                + (
                    f" up to {plan.gates_row_limit} rows"
                    if plan.gates_row_limit is not None
                    else (
                        " at every row count (tier 2 above "
                        f"{plan.gates_tier2_above} rows: the merged Triton GEMM "
                        "replaces the gates' oneDNN GEMMs)"
                        if plan.gates_tier2_above is not None
                        else ""
                    )
                )
                if plan.gates is not None
                else "",
                (
                    "shared expert's gate and the shared + routed add in the "
                    "routed gather"
                    + (
                        " (one fp32 rounding, tier 2)"
                        if plan.single_round
                        else " (bf16 rounding points kept)"
                    )
                    + (
                        "; the gate's sigmoid in the gather (device exp, tier 2)"
                        if plan.gate_logits
                        else ""
                    )
                )
                if plan.fold_module is not None
                else "",
            )
            if p
        )
        or "nothing applies",
        (". Not applied: " + "; ".join(why)) if why else "",
        scope="process",
    )
    return plan if plan.active else None


# ----------------------------------------------------------- top-k route
def _topk_route_eligible(router) -> str | None:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )

    if type(router) is not FusedTopKRouter:
        return f"router {type(router).__name__}"
    if router.scoring_func != "softmax":
        return f"scoring {router.scoring_func}"
    if router.eplb_state is None:
        return "no EPLB"
    if rocm_aiter_ops.is_fused_moe_enabled():
        return "AITER"
    if _route_op() is None:
        return "vllm-xpu-kernels has no topk_softmax_route"
    if router.global_num_experts not in (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        192,
        320,
        384,
        448,
        576,
    ):
        return f"{router.global_num_experts} experts"
    return None


def _route_op():
    """vllm-xpu-kernels' ``_moe_C.topk_softmax_route``, or None."""
    try:
        return torch.ops._moe_C.topk_softmax_route
    except (AttributeError, RuntimeError):
        return None


def topk_route(
    router,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``VLLM_XPU_MOE_TOPK_ROUTE=1``: the router's softmax top-k, EPLB
    mapping + load record and (with VLLM_XPU_MOE_ROUTE_FUSED=1) the unowned
    sentinel as one kernel (vllm-xpu-kernels ``topk_softmax_route``):
    ``(topk_weights, physical topk_ids)`` as ``FusedTopKRouter`` followed by
    ``_apply_eplb_mapping`` gives them, or None to run those."""
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        get_moe_padding_mask,
    )

    why = getattr(router, "_xpu_topk_route_why", False)
    if why is False:
        why = None if envs.VLLM_XPU_MOE_TOPK_ROUTE else "off"
        if why is None and not current_platform.is_xpu() and not _TEST_NATIVE:
            why = "not XPU"
        if why is None:
            why = _topk_route_eligible(router)
        router._xpu_topk_route_why = why
        if envs.VLLM_XPU_MOE_TOPK_ROUTE:
            if why is None:
                logger.info_once(
                    "XPU MoE routing (VLLM_XPU_MOE_TOPK_ROUTE=1): softmax top-k, "
                    "EPLB map and load record%s in one kernel.",
                    " and the unowned sentinel"
                    if router._xpu_ids_sentinel() is not None
                    else "",
                    scope="process",
                )
            else:
                logger.warning_once(
                    "VLLM_XPU_MOE_TOPK_ROUTE=1, but %s: top-k and the EPLB "
                    "mapping run as before.",
                    why,
                    scope="process",
                )
    if why is not None:
        return None
    if (
        router.capture_fn is not None
        or indices_type not in (None, torch.int32)
        or router_logits.dtype not in (torch.bfloat16, torch.float32)
        or router_logits.dim() != 2
        or router_logits.shape[-1] != router.global_num_experts
    ):
        return None
    st = router.eplb_state
    m = router_logits.shape[0]
    dev = router_logits.device
    weights = torch.empty((m, router.top_k), dtype=torch.float32, device=dev)
    ids = torch.empty((m, router.top_k), dtype=torch.int32, device=dev)
    tei = torch.empty((m, router.top_k), dtype=torch.int32, device=dev)
    sentinel = router._xpu_ids_sentinel()
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id

    _route_op()(
        weights,
        ids,
        tei,
        router_logits.contiguous(),
        router.renormalize,
        get_moe_padding_mask(m),
        st.logical_to_physical_map,
        st.logical_replica_count,
        st.expert_load_view,
        st.should_record_tensor,
        st.num_unpadded_tokens_tensors[dbo_current_ubatch_id()],
        0 if sentinel is None else sentinel,
        sentinel is not None,
    )
    if sentinel is not None:
        ids.xpu_ids_sentinel = sentinel
    return weights, ids

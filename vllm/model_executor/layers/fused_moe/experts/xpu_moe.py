# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import torch

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


class XPUExperts(mk.FusedMoEExpertsModular):
    # Extra keyword arguments for every XpuFusedMoe.apply call.
    impl_apply_kwargs: dict = {}
    # Expert slots in device and host tiers, set by ExpertTiers.finalize.
    tiers: ExpertTiers | None = None
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
        if self.moe_config.ep_size > 1:
            # Avoid topk_ids = -1, remap them to an
            # expert not owned by this rank so they are skipped.
            sentinel = global_num_experts - 1 if self.moe_config.ep_rank == 0 else 0
            topk_ids = torch.where(topk_ids < 0, sentinel, topk_ids)
        staged_impl(self.fused_moe_impl, hidden_states.size(0), "experts").apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1q_scale=a1q_scale,
            **self.impl_apply_kwargs,
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
        """Experts in the fast slots plus experts in the slow slots, summed."""
        tiers = self.tiers
        assert tiers is not None
        fast, slow = self._tier_impls(w1, w2, topk_ids.size(-1))
        if self.moe_config.ep_size > 1:
            topk_ids = remap_unrouted(
                topk_ids, tiers.num_global, self.moe_config.ep_rank
            )
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

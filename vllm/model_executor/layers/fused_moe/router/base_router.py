# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.distributed.eplb.residency_mask import (
    DISPLACED_CALLS,
    DISPLACED_WEIGHT,
    ROUTED_CALLS,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.eplb_triton import eplb_map_and_record
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if current_platform.is_cuda_alike() or current_platform.is_xpu():

    @triton.jit
    def _eplb_map_and_record_i32_kernel(
        topk_ids_ptr,
        logical_replica_count_ptr,
        logical_to_physical_ptr,
        out_ids_ptr,
        out_ptr,
        record_enabled_ptr,
        num_unpadded_tokens_ptr,
        num_logical_experts,
        map_slots,
        out_size,
        numel,
        num_active_experts,
        sentinel,
        BLOCK_SIZE: tl.constexpr,
        SENTINEL: tl.constexpr = False,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int64)

        # TODO(bowen): When using `FusedMoEModularKernel`, this
        # can be done in a more unified way, since
        # `FusedMoEPrepareAndFinalize` will return the expert
        # token count, in some cases directly from the kernel.
        # However, now there are many code paths not using
        # the modular kernel, e.g. calling `fused_experts`,
        # so we decide to keep the logic here.
        #
        # If later refactor moved all the MoE kernel calls
        # to the modular kernel, we can move this logic there
        # to achieve better efficiency.
        physical_id = eplb_map_and_record(
            expert_id,
            offs // num_active_experts,
            mask,
            logical_to_physical_ptr,
            logical_replica_count_ptr,
            num_logical_experts,
            map_slots,
            out_ptr,
            out_size,
            record_enabled_ptr,
            num_unpadded_tokens_ptr,
        )
        if SENTINEL:
            # The XPU fused MoE's unowned sentinel for unrouted ids, as its
            # sentinel_ids_and_counts writes it (VLLM_XPU_MOE_ROUTE_FUSED).
            physical_id = tl.where(physical_id < 0, sentinel, physical_id)
        tl.store(out_ids_ptr + offs, physical_id, mask=mask)

    def _eplb_map_and_record_triton(
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        expert_load_view: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None,
        sentinel: int | None = None,
    ) -> torch.Tensor:
        topk_ids_in = topk_ids.contiguous().to(dtype=torch.int32)
        numel = topk_ids_in.numel()
        if numel == 0:
            return topk_ids
        num_active_experts = topk_ids_in.shape[-1]
        out_flat = torch.empty((numel,), device=topk_ids.device, dtype=topk_ids.dtype)
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        assert expert_load_view.is_contiguous()
        _eplb_map_and_record_i32_kernel[grid](
            topk_ids_in,
            logical_replica_count.contiguous(),
            logical_to_physical_map.contiguous(),
            out_flat,
            expert_load_view,
            record_enabled,
            num_unpadded_tokens,
            logical_replica_count.shape[0],
            logical_to_physical_map.shape[1],
            expert_load_view.shape[0],
            numel,
            num_active_experts,
            0 if sentinel is None else sentinel,
            BLOCK_SIZE=256,
            SENTINEL=sentinel is not None,
        )
        return out_flat.reshape(topk_ids.shape)

    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None = None,
        sentinel: int | None = None,
    ) -> torch.Tensor:
        # Fused triton implementation: mapping + optional recording in one kernel.
        return _eplb_map_and_record_triton(
            topk_ids=topk_ids,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_view=expert_load_view,
            record_enabled=record_enabled,
            num_unpadded_tokens=num_unpadded_tokens,
            sentinel=sentinel,
        )
else:

    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None = None,
        sentinel: int | None = None,
    ) -> torch.Tensor:
        return topk_ids


def get_moe_padding_mask(num_tokens: int) -> torch.Tensor | None:
    """Rows of the current batch that are CUDA graph or DP padding.

    Routers set these rows' expert ids to -1 so the dispatch and experts skip
    them; None when VLLM_MOE_SKIP_PADDING is off or the runner does not mark
    padding.
    """
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
        return is_padding[:num_tokens] if is_padding is not None else None
    return None


class BaseRouter(FusedMoERouter):
    """Base class providing common functionality for all router implementations.

    This class implements the template method pattern where select_experts() handles
    common pre-processing and post-processing, delegating the actual routing logic
    to the abstract _compute_routing() method.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState | None = None,
    ):
        """Args:
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts
        eplb_state: Optional EPLBLayerState for load balancing

        """
        super().__init__(eplb_state=eplb_state)
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.capture_fn: Callable[[torch.Tensor], None] | None = None

    def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None:
        """Set a capture callback for logical routed expert IDs."""
        self.capture_fn = capture_fn

    def _validate_eplb_state(self) -> None:
        """Validate that EPLB state is properly initialized if EPLB is enabled."""
        if self.eplb_state is not None:
            eplb_state = self.eplb_state
            if eplb_state.expert_load_view is None:
                raise ValueError("EPLB requires expert_load_view != None")
            if eplb_state.logical_to_physical_map is None:
                raise ValueError("EPLB requires logical_to_physical_map != None")
            if eplb_state.logical_replica_count is None:
                raise ValueError("EPLB requires logical_replica_count != None")
            if eplb_state.should_record_tensor is None:
                raise ValueError("EPLB requires should_record_tensor != None")
            if eplb_state.num_unpadded_tokens_tensors is None:
                raise ValueError("EPLB requires num_unpadded_tokens_tensors != None")

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Apply EPLB mapping to convert logical expert IDs to physical expert IDs."""
        if self.eplb_state is not None:
            eplb_state = self.eplb_state
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_to_physical_map is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            assert eplb_state.num_unpadded_tokens_tensors is not None
            sentinel = self._xpu_ids_sentinel()
            mapped = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                expert_load_view=eplb_state.expert_load_view,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ],
                sentinel=sentinel,
            )
            if sentinel is not None:
                # Tells the XPU fused MoE these ids carry its sentinel already,
                # so it skips its own sentinel kernel (it rewrites them to the
                # same values when the mark does not reach it).
                mapped.xpu_ids_sentinel = sentinel
            return mapped
        return topk_ids

    _XPU_SENTINEL_UNSET = object()

    def _xpu_ids_sentinel(self) -> int | None:
        """VLLM_XPU_MOE_ROUTE_FUSED: the unowned sentinel the XPU fused MoE
        writes for unrouted ids under expert parallelism with EPLB (its
        IDS_VALID contract), written by the mapping kernel instead; None when
        it does not apply. Fixed on the first call."""
        cached = getattr(self, "_xpu_sentinel", self._XPU_SENTINEL_UNSET)
        if cached is not self._XPU_SENTINEL_UNSET:
            return cached
        sentinel = None
        if current_platform.is_xpu() and envs.VLLM_XPU_MOE_ROUTE_FUSED:
            from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
                eplb_route_sentinel,
            )

            sentinel = eplb_route_sentinel(self.global_num_experts)
        self._xpu_sentinel = sentinel
        return sentinel

    @property
    def supports_residency_mask(self) -> bool:
        """Whether -inf logits keep an expert out of this router's top-k with
        the router's own weights over the rest: plain softmax or sigmoid
        scoring, no score-correction bias added after scoring (a masked
        expert's 0 + bias could still be chosen), no custom routing."""
        return (
            getattr(self, "e_score_correction_bias", None) is None
            and getattr(self, "custom_routing_function", None) is None
            and getattr(self, "scoring_func", "softmax") in ("softmax", "sigmoid")
        )

    def _apply_residency_mask(self, router_logits: torch.Tensor) -> torch.Tensor:
        """The logits with the layer's residency mask applied (-inf for the
        masked experts), counting what it displaced; the logits unchanged
        when the mask is off."""
        state = self.eplb_state
        mask = None if state is None else state.residency_mask
        if mask is None:
            return router_logits
        assert state is not None
        if not self.supports_residency_mask:
            raise NotImplementedError(
                f"{type(self).__name__} cannot keep masked experts out of its "
                "top-k (score-correction bias, custom routing or scoring "
                f"{getattr(self, 'scoring_func', None)!r}): the EPLB residency "
                "mask is not supported for this model"
            )
        if router_logits.shape[-1] != mask.shape[0]:
            raise ValueError(
                f"router logits over {router_logits.shape[-1]} experts, residency "
                f"mask over {mask.shape[0]}"
            )
        if state.residency_stats is not None:
            assert state.residency_shadow_load is not None
            self._count_displaced(router_logits, mask, state)
        return router_logits.masked_fill(mask, float("-inf"))

    def _count_displaced(
        self, router_logits: torch.Tensor, mask: torch.Tensor, state: EplbLayerState
    ) -> None:
        """Add this step's routing before the mask to the layer's counters:
        top-k calls to masked experts, all top-k calls, the renormalised
        weight of the displaced calls, and calls per expert. Padding rows are
        not counted. Fixed shapes and in-place adds only (graph-safe)."""
        stats = state.residency_stats
        shadow = state.residency_shadow_load
        assert stats is not None and shadow is not None
        num_tokens = router_logits.shape[0]
        values, ids = torch.topk(router_logits.float(), self.top_k, dim=-1)
        if getattr(self, "scoring_func", "softmax") == "sigmoid":
            weights = values.sigmoid()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        else:
            weights = values.softmax(dim=-1)
        if state.num_unpadded_tokens_tensors is not None:
            num_real = state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
            real = torch.arange(num_tokens, device=router_logits.device) < num_real
        else:
            real = torch.ones(num_tokens, dtype=torch.bool, device=router_logits.device)
        real = real.unsqueeze(-1).expand_as(ids)
        displaced = mask[ids] & real
        stats[DISPLACED_CALLS].add_(displaced.sum().to(stats.dtype))
        stats[ROUTED_CALLS].add_(real.sum().to(stats.dtype))
        stats[DISPLACED_WEIGHT].add_((weights * displaced).sum().to(stats.dtype))
        shadow.index_add_(0, ids.reshape(-1).long(), real.reshape(-1).to(shadow.dtype))

    def _convert_indices_dtype(
        self, topk_ids: torch.Tensor, indices_type: torch.dtype | None
    ) -> torch.Tensor:
        """Convert topk_ids to the desired dtype if needed."""
        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None
        return topk_ids

    @abstractmethod
    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the actual routing logic.

        This method must be implemented by subclasses to provide the specific
        routing algorithm (e.g., grouped_topk, fused_topk, custom routing, etc.).

        Args:
            hidden_states: Input hidden states
            router_logits: Router logits for expert selection
            indices_type: Desired dtype for expert indices (may be None)
            input_ids: Token ids, for routers that condition on them

        Returns:
            tuple of (topk_weights, topk_ids)

        """
        raise NotImplementedError

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route the input hidden states to the top-k experts based on the
        router logits.

        This method implements the template method pattern:
        1. Validates EPLB state
        2. Calls _compute_routing() to get topk_weights and topk_ids
        3. Applies EPLB mapping if enabled
        4. Converts indices dtype if needed

        Returns:
            (topk_weights, topk_ids)
            (tuple[torch.Tensor, torch.Tensor]):
            The weights and expert ids computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.

        """
        # Step 1: Validate EPLB state
        self._validate_eplb_state()

        # Step 1b: Keep the router away from masked experts (EPLB residency
        # mask; a no-op when it is off)
        router_logits = self._apply_residency_mask(router_logits)

        # VLLM_XPU_MOE_TOPK_ROUTE: steps 2 and 3 (and the XPU sentinel) in one
        # kernel, when this router and its call allow it.
        if envs.VLLM_XPU_MOE_TOPK_ROUTE and self.eplb_state is not None:
            from vllm.model_executor.layers.fused_moe.xpu_moe_entry import (
                topk_route,
            )

            routed = topk_route(self, router_logits, topk_indices_dtype)
            if routed is not None:
                return routed

        # Step 2: Compute routing (delegated to subclass)
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids
        )

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # Step 3: Apply EPLB mapping
        topk_ids = self._apply_eplb_mapping(topk_ids)

        # Step 4: Convert indices dtype
        topk_ids = self._convert_indices_dtype(topk_ids, topk_indices_dtype)

        return topk_weights, topk_ids

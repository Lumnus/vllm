# Modified by Wulf for B70 Flash-Next Disk; upstream terms are retained.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)

logger = init_logger(__name__)
# B70-V32-07: rows up to which the DFlash2 fixed-order HC down projection is used; wider batches (8-stream
# decode, prefill chunks) take the stock BF16 GEMM. The ordered Triton GEMM is 4.5x (64 rows) to 6.7x
# (4096 rows) slower than F.linear on B70. Default: unbounded (wu1ff behaviour).
_B70_HC_ORDERED_MAX_ROWS = int(os.environ.get("B70_HC_ORDERED_MAX_ROWS", str(1 << 30)))


def _is_xpu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_xpu()


# ---------------------------------------------------------------------------
# XPU HC down-GEMM determinism (K-split)
#
# The skinny BF16 HC down projection (K = hyper_hidden_size = 10240 -> N in
# {336 merged, 320 final mixer}) is run-to-run racy on XPU for row widths
# > 64: the oneDNN accumulation re-samples its split-K draw every execution,
# which is what made PIECEWISE-256 prefill replay diverge from eager
# (bitwise-equal inputs, few-ULP output flips). Replacing the single GEMM
# with two K=5120 GEMMs over contiguous weight halves is bitwise-deterministic
# in eager, fresh-clone and graph capture+replay at every width tested
# (M in {64, 128, 256}; 0/512 flips at M=256). See
# docs/HC_DOWN_GEMM_FIX_RECON.md for the offline campaign.
#
# The v0.32 Torch 2.14 BF16 path also changes valid outputs at M=1.
# DFlash2 therefore uses an explicitly ordered projection at every width.
# The legacy target-only split policy below remains unchanged.
# CUDA is untouched: the halves are prepared only on XPU
# (UnquantizedLinearMethod.process_weights_after_loading calls the
# ``hc_ksplit_prepare`` marker set below), so the split branch is never
# armed on non-XPU platforms.
# ---------------------------------------------------------------------------
_HC_KSPLIT_MAX_STOCK_WIDTH = 64
_HC_KSPLIT_WIDTHS_LOGGED: set[int] = set()


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output, it fuses its
    residual combine with the RMSNorm. A missing injection selects unit-weight
    combine. Final mixers use ``use_combine=False`` and do not produce a new
    injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine
        self._astra_hc_up_weight = None
        self._astra_hc_up_prepared = False
        spec = get_current_vllm_config().speculative_config
        self._ordered_hc = (
            spec is not None
            and spec.method == "dflash"
            and spec.draft_model_config.architectures == ["DFlash2DraftModel"]
            and config.params_dtype == torch.bfloat16
            and _is_xpu()
        )
        if self._ordered_hc:
            logger.info_once(
                "DFlash2 target HC uses fixed-order BF16 down projection "
                "with FP32 accumulation"
            )

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

        # -- XPU deterministic K-split state --------------------------------
        # Contiguous K=5120 halves of the down weight, prepared once at load
        # time on XPU only (see _prepare_kdown_ksplit). ``None`` everywhere
        # else, which keeps the stock single GEMM armed.
        self._kdown_lo: torch.Tensor | None = None
        self._kdown_hi: torch.Tensor | None = None
        self._kdown_k = 0
        down_linear = (
            self.input_mix_weight_down_block_inject
            if use_combine
            else self.input_mix_weight_down
        )
        # Marker consumed by
        # UnquantizedLinearMethod.process_weights_after_loading (XPU only).
        down_linear.hc_ksplit_prepare = self._prepare_kdown_ksplit

    def _prepare_kdown_ksplit(self) -> None:
        """One-time contiguous K-half prep for the deterministic XPU split."""
        if self._ordered_hc or self._kdown_lo is not None:
            return
        down_linear = (
            self.input_mix_weight_down_block_inject
            if self.use_combine
            else self.input_mix_weight_down
        )
        weight = down_linear.weight
        if weight is None or weight.ndim != 2 or weight.shape[1] % 2 != 0:
            logger.debug(
                "HC down GEMM %s: unexpected weight geometry %s; K-split left disarmed",
                type(down_linear).__name__,
                None if weight is None else tuple(weight.shape),
            )
            return
        self._kdown_k = weight.shape[1] // 2
        self._kdown_lo = weight.data[:, : self._kdown_k].contiguous()
        self._kdown_hi = weight.data[:, self._kdown_k :].contiguous()
        logger.debug(
            "HC down GEMM: K-split halves prepared (K=%d -> 2x%d, N=%d)",
            weight.shape[1],
            self._kdown_k,
            weight.shape[0],
        )

    def _mix_up(self, lora: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
        if self._ordered_hc and not self._astra_hc_up_prepared:
            from vllm_xpu_kernels.astra_structural import prepare_hc_weight

            # Resolve the supported checkpoint layout before graph capture.
            # This aliases the loaded weight and allocates no duplicate table.
            self._astra_hc_up_weight = prepare_hc_weight(self.input_mix_weight_up.weight)
            self._astra_hc_up_prepared = True
        if self._astra_hc_up_weight is not None and 1 <= lora.shape[0] <= 8:
            from vllm_xpu_kernels.astra_structural import hc_up_mix

            return hc_up_mix(lora, self._astra_hc_up_weight, normalized)
        return hc_gate_mix(normalized, self.input_mix_weight_up(lora), self.hc_count)

    def _down_gemm(self, xn: torch.Tensor) -> torch.Tensor:
        """Use ordered DFlash2 projection or the legacy target-only policy."""
        if self._ordered_hc and xn.shape[0] <= _B70_HC_ORDERED_MAX_ROWS:
            from .b70_hc_gemm import ordered_hc_gemm

            linear = (
                self.input_mix_weight_down_block_inject
                if self.use_combine
                else self.input_mix_weight_down
            )
            valid = self.lora_rank + (self.hc_count if self.use_combine else 0)
            return ordered_hc_gemm(xn, linear.weight, valid)
        if (
            self._kdown_lo is not None
            and _is_xpu()
            and xn.shape[0] > _HC_KSPLIT_MAX_STOCK_WIDTH
        ):
            width = xn.shape[0]
            if width not in _HC_KSPLIT_WIDTHS_LOGGED:
                _HC_KSPLIT_WIDTHS_LOGGED.add(width)
                logger.info(
                    "HC down GEMM: deterministic K-split engaged at row width "
                    "%d (stock single GEMM only at widths <= %d)",
                    width,
                    _HC_KSPLIT_MAX_STOCK_WIDTH,
                )
            return F.linear(xn[..., : self._kdown_k], self._kdown_lo) + F.linear(
                xn[..., self._kdown_k :], self._kdown_hi
            )
        if self.use_combine:
            return self.input_mix_weight_down_block_inject(xn)
        return self.input_mix_weight_down(xn)

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self._ordered_hc:
            from .b70_c1_graph import replay_hc

            return replay_hc(self, "mix", self._mix_eager, (hidden_states,))
        return self._mix_eager(hidden_states)

    def _mix_eager(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self._down_gemm(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self._down_gemm(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        block_input = self._mix_up(lora, xn)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self._ordered_hc:
            from .b70_c1_graph import replay_hc

            return replay_hc(
                self,
                "combine_and_mix",
                self._combine_and_mix_eager,
                (hidden_states, prev_block_output, prev_injection),
            )
        return self._combine_and_mix_eager(
            hidden_states, prev_block_output, prev_injection
        )

    def _combine_and_mix_eager(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm. A missing injection applies the block output
        to every stream with unit weight.
        """
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self._down_gemm(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self._down_gemm(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        block_input = self._mix_up(lora, xn)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )

from vllm.model_executor.kernels.linear import init_int8_linear_kernel
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    convert_to_int8_moe_kernel_format,
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    OnlineLinearBase,
    _is_tp_sharded,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    amax_for_moe_weight_quant,
    amax_for_tp_weight_quant,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    weight_amax,
)
from vllm.model_executor.utils import is_weights_pre_processed, replace_parameter

# Chunk so the fp32 intermediate of the divide stays under ~64 MB.
_QUANT_CHUNK_ELEMS = 16 * 1024 * 1024


def quantize_int8_per_channel(
    weight: torch.Tensor, amax: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[N, K]` float weight and its `[N, 1]` row amax -> (int8 `[N, K]`,
    fp32 `[N, 1]` scale): symmetric, round to nearest, in [-127, 127]."""
    vmax = torch.iinfo(torch.int8).max
    scale = (amax.to(torch.float32) / vmax).clamp_min(torch.finfo(torch.float32).tiny)
    qweight = torch.empty_like(weight, dtype=torch.int8)
    chunk_rows = max(1, _QUANT_CHUNK_ELEMS // max(1, weight.shape[-1]))
    for start in range(0, weight.shape[0], chunk_rows):
        rows = slice(start, start + chunk_rows)
        qweight[rows] = (
            (weight[rows].to(torch.float32) / scale[rows])
            .round_()
            .clamp_(-vmax, vmax)
            .to(torch.int8)
        )
    return qweight, scale


class Int8PerChannelOnlineLinearMethod(OnlineLinearBase):
    """Online W8A8 INT8 linear quantization.

    Per-output-channel symmetric weight scale + dynamic per-token symmetric
    activation scale, on the platform's INT8 scaled-mm kernel. Same layout as
    llmcompressor's W8A8 INT8 recipe with dynamic activations, so no
    pre-quantized checkpoint is required.
    """

    weight_quant_key = kInt8StaticChannelSym
    activation_quant_key: QuantKey | None = kInt8DynamicTokenSym
    accepts_activation_override = True
    # Per-row scales stay local to each vocab shard, and padded rows quantize
    # to zero, as with the FP8 per-channel method.
    supports_lm_head = True

    def __init__(self, activation_quant_key: QuantKey | None = kInt8DynamicTokenSym):
        super().__init__()
        self.activation_quant_key = self.resolve_activation_quant_key(
            activation_quant_key, explicit=True
        )

    @classmethod
    def resolve_activation_quant_key(
        cls, activation_quant_key: QuantKey | None, explicit: bool
    ) -> QuantKey | None:
        key = super().resolve_activation_quant_key(activation_quant_key, explicit)
        if key != kInt8DynamicTokenSym:
            raise NotImplementedError(
                f"activation override for online {cls.__name__} supports only "
                f"{kInt8DynamicTokenSym}, got {key}"
            )
        return key

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        self.int8_linear = init_int8_linear_kernel(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
            module_name=self.__class__.__name__,
            is_fused_module=len(output_partition_sizes) > 1,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if is_weights_pre_processed():
            raise RuntimeError("Online INT8 backend cannot use pre-processed weights")
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        weight = self.get_weight_for_quantization(layer)
        amax = weight_amax(weight, dim=-1, keepdim=True)
        amax = amax_for_tp_weight_quant(
            amax, _is_tp_sharded(layer, reduces_output_dim=False)
        )
        qweight, weight_scale = quantize_int8_per_channel(weight, amax)

        replace_parameter(layer, "weight", qweight)
        replace_parameter(layer, "weight_scale", weight_scale)
        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None

        self.int8_linear.process_weights_after_loading(layer)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.int8_linear.apply_weights(layer, x, bias)


class Int8OnlineMoEMethod(OnlineMoEMethodBase):
    """Online per-channel INT8 MoE quantization.
    Loads fp16/bf16 weights and quantizes them per-row to int8 during loading.
    """

    def __init__(
        self,
        *,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        w13_weight, w2_weight = self.get_weights_for_quantization(layer)
        self._quantize_weights(layer, w13_weight, w2_weight)
        self._setup_kernel(layer)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def _quantize_weights(
        self,
        layer: Module,
        w13_weight: torch.Tensor | None = None,
        w2_weight: torch.Tensor | None = None,
    ) -> None:
        w13_weight = layer.w13_weight if w13_weight is None else w13_weight
        w2_weight = layer.w2_weight if w2_weight is None else w2_weight
        vmax = torch.iinfo(torch.int8).max

        w13 = torch.empty_like(w13_weight, dtype=torch.int8)
        w2 = torch.empty_like(w2_weight, dtype=torch.int8)
        w13_scale = torch.zeros(
            layer.num_experts,
            w13_weight.shape[1],
            device=w13.device,
            dtype=torch.float32,
        )
        w2_scale = torch.zeros(
            layer.num_experts,
            w2_weight.shape[1],
            device=w2.device,
            dtype=torch.float32,
        )

        w2_amax = weight_amax(w2_weight, dim=-1)
        w2_amax = amax_for_moe_weight_quant(w2_amax, self.moe.tp_size)

        for expert in range(layer.local_num_experts):
            # w13: per-row quantization over hidden_size dim
            w = w13_weight[expert, :, :]
            scales = w.abs().amax(dim=1) / vmax
            q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
            w13[expert, :, :] = q.to(torch.int8)
            w13_scale[expert, :] = scales

            # w2: per-row quantization over intermediate_size dim
            w = w2_weight[expert, :, :]
            scales = w2_amax[expert] / vmax
            q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
            w2[expert, :, :] = q.to(torch.int8)
            w2_scale[expert, :] = scales

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_scale", w13_scale)
        replace_parameter(layer, "w2_scale", w2_scale)

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        w13, w2 = convert_to_int8_moe_kernel_format(
            int8_backend=self.int8_backend,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            layer=layer,
            w13_scale=layer.w13_scale,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_int8_moe_kernel(
            int8_backend=self.int8_backend,
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig | None":
        return make_int8_moe_quant_config(
            int8_backend=self.int8_backend,
            w1_scale=getattr(layer, "w13_scale", None),
            w2_scale=getattr(layer, "w2_scale", None),
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            per_out_ch_quant=True,
            layer=layer,
        )

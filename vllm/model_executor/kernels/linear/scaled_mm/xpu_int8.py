# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70: W8A8 INT8 linear through the oneDNN ``_xpu_C::int8_scaled_mm`` op.

Selected ahead of ``TritonInt8ScaledMMLinearKernel`` on XPU only when the
vllm-xpu-kernels wheel carries the op AND ``B70_XPU_INT8_ONEDNN=1``; otherwise
the Triton kernel stays the path. Symmetric activations only (the op has no
zero point); asymmetric layers fall through to Triton.
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import Int8ScaledMMLinearLayerConfig
from .triton import TritonInt8ScaledMMLinearKernel

logger = init_logger(__name__)


class XPUInt8ScaledMMLinearKernel(TritonInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "requires XPU."
        if os.environ.get("B70_XPU_INT8_ONEDNN", "0") != "1":
            return False, "B70_XPU_INT8_ONEDNN is not 1."
        if not hasattr(torch.ops._xpu_C, "int8_scaled_mm"):
            return False, "vllm-xpu-kernels has no _xpu_C.int8_scaled_mm op."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "asymmetric activations are not supported."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Same layout as Triton: w_q -> [K, N] view of the [N, K] weight
        # (oneDNN NT), channelwise scales for fused modules, scalar static
        # input scale.
        super().process_weights_after_loading(layer)
        logger.info_once("B70 int8 GEMM: oneDNN", scope="local")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, _, _ = self._get_layer_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        x_q, x_s, _ = ops.scaled_int8_quant(x_2d.contiguous(), i_s, None, True)
        out = torch.ops._xpu_C.int8_scaled_mm(
            x_q,
            x_s.reshape(-1).to(torch.float32),
            w_q,
            w_s.reshape(-1).to(torch.float32),
            x.dtype,
            bias,
        )
        return out.view(*x.shape[:-1], w_q.shape[1])

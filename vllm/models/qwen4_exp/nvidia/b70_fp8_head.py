# SPDX-License-Identifier: Apache-2.0
"""B70-0048/0053 (V32 port): the target's lm_head shard as FP8 W8A16 (E4M3, one scale per vocab row).

B70_HEAD_FP8=1 (XPU): built once at the first forward (the eager memory-profile run, before any activation) and the
BF16 shard released, so vLLM's profile sees the shard gone and the saving lands in KV. Every caller of the head
(compute_logits, astra greedy, the DFlash2 drafter sharing the module) goes through head.quant_method.apply."""
import os

import torch


class B70Fp8Head:
    """What LogitsProcessor needs of an lm_head (quant_method.apply), backed by an FP8 W8A16 copy of the shard."""

    def __init__(self, w8: torch.Tensor, scale: torch.Tensor) -> None:
        self.w8, self.scale = w8, scale

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        out = torch.ops._xpu_C.fp8_gemm_w8a16(x.reshape(-1, shape[-1]), self.w8, self.scale, None)
        if bias is not None:
            out = out + bias
        return out.reshape(*shape[:-1], -1)


def b70_fp8_head_of(lm_head, release_bf16: bool = False) -> "B70Fp8Head | None":
    """The FP8 copy of an lm_head shard, cached on the module. release_bf16: the BF16 weight goes to host memory and
    is released on the device before the FP8 copy is built (the two never coexist in VRAM)."""
    head8 = getattr(lm_head, "_b70_fp8_head", None)
    weight = getattr(lm_head, "weight", None)
    if head8 is None:
        if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16 or weight.numel() == 0:
            return None
        rows, k = weight.shape
        device = weight.device
        src = weight.data
        if release_bf16:
            src = weight.data.to("cpu")
            weight.data = torch.empty((0, k), dtype=weight.dtype, device=device)
        w8 = torch.empty((k, rows), dtype=torch.float8_e4m3fn, device=device)
        scale = torch.empty((1, rows), dtype=torch.float32, device=device)
        step = 8192
        for i in range(0, rows, step):
            chunk = src[i : i + step].to(device).float()
            s = chunk.abs().amax(dim=1).clamp(min=1e-12) / 448.0
            w8[:, i : i + step] = (chunk / s[:, None]).t().to(torch.float8_e4m3fn)
            scale[0, i : i + step] = s
            del chunk
        del src
        head8 = B70Fp8Head(w8, scale)
        lm_head._b70_fp8_head = head8
    if release_bf16 and weight is not None and weight.numel() != 0:
        weight.data = torch.empty((0, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    return head8


def maybe_prepare_head_fp8(lm) -> None:
    """Once per model: B70_HEAD_FP8=1 on XPU swaps lm.lm_head.quant_method for the FP8 copy and releases BF16."""
    if getattr(lm, "_b70_head_fp8_checked", False):
        return
    lm._b70_head_fp8_checked = True
    from vllm.platforms import current_platform

    if os.environ.get("B70_HEAD_FP8", "0") != "1" or not current_platform.is_xpu():
        return
    if torch.xpu.is_current_stream_capturing():
        lm._b70_head_fp8_checked = False
        return
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    head = lm.lm_head
    weight = head.weight
    released = weight.numel() * weight.element_size()
    alloc0 = torch.accelerator.memory_allocated(weight.device)
    head8 = b70_fp8_head_of(head, release_bf16=True)
    if head8 is None:
        logger.warning("B70 head FP8: no BF16 lm_head weight; off")
        return
    head.quant_method = head8
    alloc1 = torch.accelerator.memory_allocated(weight.device)
    logger.info("B70 head FP8: target lm_head %d x %d as FP8 W8A16 (%.0f MB), BF16 shard released (%.0f MB); "
                "torch allocated %.2f -> %.2f GiB", head8.w8.shape[1], head8.w8.shape[0], head8.w8.numel() / 1e6,
                released / 1e6, alloc0 / 2**30, alloc1 / 2**30)

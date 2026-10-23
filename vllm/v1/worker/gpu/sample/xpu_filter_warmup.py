# SPDX-License-Identifier: MIT
# B70-0060: from wu1ff/B70-LLM-Controller qwen3.8-flash-next-v1.0.5 integrated-serving.patch; Copyright (c) 2026 wu1ff, MIT (LICENSES/wu1ff-B70-LLM-Controller-MIT.txt).
# Generalised from DFlash-only to any spec method on Qwen4Exp + XPU (our MTP serving pays the same first-request JIT).
"""Bounded initialized warmup for the XPU top-k/top-p sampling filter."""
import time

import torch

from vllm.logger import init_logger


@torch.inference_mode()
def warmup_filter(runner):
    spec = runner.speculative_config
    if spec is None or runner.model_config.hf_text_config.model_type != "qwen4_exp_text":
        return
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

    started = time.perf_counter()
    before = torch.xpu.memory_allocated()
    # Dedicated initialized tensors; no live request, KV, RNG or sampler state.
    logits = torch.arange(runner.vocab_size, device=runner.device, dtype=torch.float32).unsqueeze(0) / runner.vocab_size
    k = torch.full((1,), 20, device=runner.device, dtype=torch.int32)
    p = torch.full((1,), 0.95, device=runner.device, dtype=torch.float32)
    for topk, topp in ((k, p), (k, None), (None, p)):
        apply_top_k_top_p_triton(logits.clone(), topk, topp)
    torch.xpu.synchronize()
    del logits, k, p
    init_logger(__name__).info(
        "B70-0060 initialized XPU filter warmup: %.3f s, retained tensor bytes=%d",
        time.perf_counter() - started, torch.xpu.memory_allocated() - before)

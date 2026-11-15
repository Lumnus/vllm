# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect loaded Qwen4Exp QSA modules to their kernel-owned warmup."""

import sys
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# `qsa_prepare` picks its query tile by this step size.
_PREPARE_SMALL_STEP_TOKENS = 4096


def qwen4_exp_qsa_triton_warmup(worker: "Worker") -> None:
    """Warm every reachable QSA specialization: indexer decode-query-length
    profiles plus the sparse attention split-K/merge configs."""
    qsa_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa_indexer")
    attn_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa")
    if qsa_module is None or attn_module is None:
        return
    indexer = None
    owner = None
    for layer in worker.get_model().modules():
        if indexer is None and isinstance(layer, qsa_module.QSAIndexer):
            indexer = layer
        elif owner is None and isinstance(layer, attn_module.Qwen4ExpQSAAttention):
            owner = layer
    if indexer is None or owner is None:
        return

    runner = worker.model_runner

    def block_table_for(prefix: str) -> torch.Tensor:
        group_id = next(
            i
            for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            if prefix in group.layer_names
        )
        return runner.block_tables.input_block_tables[group_id]

    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        warmup_qsa_mqa_paged_decode,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    assert k_cache.numel()
    profiles = warmup_qsa_mqa_paged_decode(
        k_cache,
        block_table_for(indexer.compressed_key_cache.prefix),
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
        max_decode_query_len=runner.decode_query_len,
        max_num_reqs=runner.max_num_reqs,
        max_num_batched_tokens=runner.max_num_tokens,
    )
    logger.info("Warmed up Qwen4Exp QSA decode kernels: %s.", profiles)

    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        warmup_qsa_sparse_paged_attention,
    )

    # A draft with its own KV dtype (speculative_config.kv_cache_dtype) specializes
    # the attention kernels differently from the target.
    owners = [owner]
    draft_model = getattr(getattr(runner, "drafter", None), "model", None)
    if draft_model is not None:
        for layer in draft_model.modules():
            if (
                isinstance(layer, attn_module.Qwen4ExpQSAAttention)
                and layer.kv_cache_dtype != owner.kv_cache_dtype
            ):
                owners.append(layer)
                break
    for attention in owners:
        kv_cache = attention.kv_cache
        assert kv_cache.numel()
        attention_profiles = warmup_qsa_sparse_paged_attention(
            kv_cache,
            block_table_for(attention.layer_name),
            num_query_heads=attention.num_heads,
            selection_width=indexer.output_width,
            head_dim=attention.head_dim,
            int8_dot=getattr(attention.impl, "int8_dot", 0),
        )
        logger.info(
            "Warmed up Qwen4Exp QSA sparse attention kernels (%s): %s.",
            attention.kv_cache_dtype,
            attention_profiles,
        )

    # The fused prepare kernel takes a wider query tile above 4,096 scheduled
    # tokens, and only a step that large reaches it: run one when the step
    # budget allows such steps.
    max_step_tokens = min(runner.max_num_tokens, worker.model_config.max_model_len)
    if max_step_tokens > _PREPARE_SMALL_STEP_TOKENS:
        from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

        if run_mixed_prefill_decode_warmup(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            max_step_tokens,
            req_id_prefix="_qsa_prepare_warmup",
        ):
            logger.info(
                "Warmed up the Qwen4Exp QSA prepare kernel on a step of %d tokens.",
                max_step_tokens,
            )

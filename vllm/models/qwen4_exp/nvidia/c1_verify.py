# SPDX-License-Identifier: Apache-2.0
"""Eager C1 DFlash2 verification with native speculative prefix-state kernels."""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group

from .b70_c1_graph import hc_graph_scope


def enabled(runner, batch) -> bool:
    spec = runner.speculative_config
    return (
        spec is not None
        and spec.method == "dflash"
        and spec.draft_model_config.architectures == ["DFlash2DraftModel"]
        and runner.device.type == "xpu"
        and runner.model_config.hf_text_config.model_type == "qwen4_exp_text"
        and batch.num_reqs == 1
        and not batch.has_prefill
        and 1 < batch.num_tokens <= 8
    )


@contextmanager
def canonical_reductions(width):
    """Keep C1's BF16 XCCL message width while batching target computation.

    XCCL changes the BF16 reduction order with the number of rows. Local GEMMs
    and recurrent kernels remain batched; only these small TP reductions retain
    the proven one-row collective. Draft execution uses its ordinary reductions.
    """
    comm = get_tp_group().device_communicator
    original = comm.all_reduce

    def reduce(input_):
        if (
            input_.dtype == torch.bfloat16
            and input_.ndim == 2
            and input_.shape[0] == width
        ):
            output = input_.clone()
            # XCCL keeps each row's arithmetic, while grouping submission and
            # completion. This collective remains outside every captured graph.
            with dist._coalescing_manager(
                group=comm.device_group, device=input_.device
            ):
                for row in output.split(1):
                    dist.all_reduce(row, group=comm.device_group)
            return output
        return original(input_)

    comm.all_reduce = reduce
    try:
        yield
    finally:
        comm.all_reduce = original


def verify_eager(runner, batch, metadata, model_inputs):
    """Run anchor plus proposals once and retain every native accepted prefix."""
    with canonical_reductions(batch.num_tokens), hc_graph_scope(batch.num_tokens):
        return runner.model(**model_inputs)


def verify(runner, batch, metadata, model_inputs):
    if batch.num_tokens == 8 and os.getenv("VLLM_XPU_C1_TARGET_GRAPH", "1") == "1":
        from .b70_target_graph import TargetGraphs

        graphs = getattr(runner, "_c1_target_graphs", None)
        if graphs is None:
            graphs = runner._c1_target_graphs = TargetGraphs()
        return graphs.run(runner, batch, metadata, model_inputs, verify_eager)
    return verify_eager(runner, batch, metadata, model_inputs)

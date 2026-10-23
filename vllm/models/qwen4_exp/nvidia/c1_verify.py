# SPDX-License-Identifier: Apache-2.0
"""Eager C1 DFlash2 verification with native speculative prefix-state kernels."""

from contextlib import contextmanager

import torch

from vllm.distributed import get_tp_group


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
        and 1 < batch.num_tokens <= 4
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
            return torch.cat([original(row) for row in input_.split(1)], dim=0)
        return original(input_)

    comm.all_reduce = reduce
    try:
        yield
    finally:
        comm.all_reduce = original


def verify(runner, batch, metadata, model_inputs):
    """Run anchor plus proposals once and retain every native accepted prefix."""
    with canonical_reductions(batch.num_tokens):
        return runner.model(**model_inputs)

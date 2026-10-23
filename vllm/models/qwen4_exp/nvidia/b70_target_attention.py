# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant native BF16 attention for the C1 DFlash2 target."""

import os

import torch


def fixed_splits(
    q,
    k,
    v,
    cu,
    used,
    blocks,
    max_query,
    causal,
    window,
    softcap,
    alibi,
    sink,
    return_lse,
    q_descale=None,
):
    value = os.getenv("VLLM_XPU_DFLASH_FIXED_SPLITS", "0")
    if value not in ("32", "64"):
        return None
    if (
        blocks is None
        or used is None
        or not causal
        or return_lse
        or alibi is not None
        or sink is not None
        or softcap not in (None, 0.0)
        or q_descale is not None
        or tuple(window) != (-1, -1)
        or q.dtype != torch.bfloat16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or cu.numel() != 2
        or used.numel() != 1
        or not 1 <= max_query <= 8
        or q.shape != (max_query, 6, 256)
        or k.ndim != 4
        or k.shape[1:] != (832, 1, 256)
        or v.shape != k.shape
        or blocks.dtype != torch.int32
    ):
        return None
    return int(value)


def forward(q, k, v, out, used, blocks, max_seq_len, scale, splits):
    # A fixed allocation and reduction schedule serves both one-token decode
    # and verification. Live per-position lengths retain causal masking and
    # the native short-sequence single-split rule in eager and graph replay.
    width = q.shape[0]
    if scale is None:
        scale = q.shape[-1] ** -0.5
    cu = torch.arange(width + 1, dtype=torch.int32, device=q.device)
    dummy = torch.zeros_like(cu)
    positions = torch.arange(width, dtype=used.dtype, device=q.device)
    lengths = (used.reshape(1) - (width - 1 - positions)).clamp_min(1).int()
    tables = blocks.repeat_interleave(width, dim=0)
    result, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q,
        k,
        v,
        out,
        cu,
        dummy,
        lengths,
        None,
        tables,
        None,
        1,
        max_seq_len,
        0.0,
        None,
        None,
        None,
        scale,
        None,
        False,
        False,
        -1,
        -1,
        0.0,
        False,
        None,
        splits,
        False,
        None,
        None,
    )
    return result

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int8_per_token_head KV on the XPU FlashAttention route: the per-token-head
writer through the split-layout views, the native INT8 paged decode
(single-token, uniform and ragged speculative-decode batches) and the
dequantized prefill, against an fp32 reference on the same quantized cache.

Run: pytest tests/kernels/attention/test_xpu_fa_int8_per_token_head.py
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.xpu_int8_per_token_head import (
    decode_max_query_len,
    forward_decode,
    forward_prefill,
    int8_kv_decode_supported,
    prefill_scratch_tokens,
    split_layout_views,
)
from vllm.v1.kv_cache_interface import KVQuantMode

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="int8_per_token_head on FlashAttention is XPU only.",
)

DEVICE = "xpu"
HEAD_SIZE = 256


def _content(head_size: int, padded: bool) -> int:
    side = head_size + 4
    return 2 * (-(-side // 16) * 16 if padded else side)


def _make_cache(num_blocks, nkv, block_size, head_size, padded):
    content = _content(head_size, padded)
    raw = torch.randint(
        -128, 127, (num_blocks, nkv, block_size, content), dtype=torch.int8
    ).to(DEVICE)
    return raw, split_layout_views(raw, nkv, head_size)


def _write(seq_lens, nkv, block_size, head_size, views, dtype):
    """Random K/V per sequence written by the per-token-head writer; returns
    the block table and the dequantized K/V read back from the views."""
    key_cache, value_cache, k_scale, v_scale = views
    num_blocks = key_cache.shape[0]
    nblk = [-(-L // block_size) for L in seq_lens]
    perm = torch.randperm(num_blocks - 1) + 1
    block_table = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32)
    slots, keys, values, c = [], [], [], 0
    for i, (L, n) in enumerate(zip(seq_lens, nblk)):
        block_table[i, :n] = perm[c : c + n]
        c += n
        pos = torch.arange(L)
        slots.append(
            block_table[i, pos // block_size].long() * block_size + pos % block_size
        )
        k = torch.randn(L, nkv, head_size) * 2
        k[..., 7] *= 20
        keys.append(k.to(DEVICE, dtype))
        values.append(torch.randn(L, nkv, head_size).to(DEVICE, dtype))
    triton_reshape_and_cache_flash_per_token_head_quant(
        torch.cat(keys),
        torch.cat(values),
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        torch.cat(slots).to(DEVICE),
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    )
    block_table = block_table.to(DEVICE)
    deq_k, deq_v = [], []
    for i, L in enumerate(seq_lens):
        pos = torch.arange(L, device=DEVICE)
        b = block_table[i, pos // block_size].long()
        o = pos % block_size
        deq_k.append(key_cache[b, o].float() * k_scale[b, o][..., None])
        deq_v.append(value_cache[b, o].float() * v_scale[b, o][..., None])
    return block_table, deq_k, deq_v


def _reference(q, deq_k, deq_v, q_lens, seq_lens, scale):
    out, q0 = [], 0
    for k, v, ql, L in zip(deq_k, deq_v, q_lens, seq_lens):
        g = q.shape[1] // k.shape[1]
        kk, vv = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        s = torch.einsum("qhd,khd->hqk", q[q0 : q0 + ql].float(), kk) * scale
        qpos = torch.arange(L - ql, L, device=DEVICE)
        mask = torch.arange(L, device=DEVICE)[None, :] <= qpos[:, None]
        s = s.masked_fill(~mask[None], float("-inf"))
        out.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vv))
        q0 += ql
    return torch.cat(out)


def test_kernels_support_int8_decode():
    assert int8_kv_decode_supported(HEAD_SIZE, torch.bfloat16) is None


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("nkv, gqa", [(1, 8), (2, 8), (4, 6)])
@pytest.mark.parametrize("block_size", [64, 832])
@pytest.mark.parametrize("mode", ["decode", "spec_uniform", "spec_ragged", "prefill"])
@torch.inference_mode()
def test_fa_int8_per_token_head(padded, nkv, gqa, block_size, mode):
    if mode.startswith("spec") and decode_max_query_len() < 5:
        pytest.skip("vllm-xpu-kernels small-q decode route disabled")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    seq_lens = [1, 37, 700, 2049, 64]
    q_lens = {
        "decode": [1] * 5,
        "spec_uniform": [1, 4, 4, 4, 4],
        "spec_ragged": [1, 3, 5, 2, 4],
        "prefill": [1, 37, 300, 129, 1],
    }[mode]
    if mode == "spec_uniform":
        seq_lens[0] = 4
        q_lens[0] = 4
    num_blocks = sum(-(-L // block_size) for L in seq_lens) + 4
    _, views = _make_cache(num_blocks, nkv, block_size, HEAD_SIZE, padded)
    block_table, deq_k, deq_v = _write(
        seq_lens, nkv, block_size, HEAD_SIZE, views, dtype
    )
    hq = nkv * gqa
    q = torch.randn(sum(q_lens), hq, HEAD_SIZE, device=DEVICE, dtype=dtype)
    out = torch.empty_like(q)
    qsl_cpu = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32
    )
    qsl = qsl_cpu.to(DEVICE)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    scale = HEAD_SIZE**-0.5
    key_cache, value_cache, k_scale, v_scale = views
    if mode == "prefill":
        nb = prefill_scratch_tokens(4096, block_size) // block_size
        ks = torch.empty((nb, block_size, nkv, HEAD_SIZE), dtype=dtype, device=DEVICE)
        vs = torch.empty_like(ks)
        forward_prefill(
            q,
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            out,
            ks,
            vs,
            qsl,
            qsl_cpu,
            seq,
            seq_lens,
            block_table,
            scale,
            True,
            [-1, -1],
            None,
        )
    else:
        forward_decode(
            q,
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            out,
            qsl,
            max(q_lens),
            seq,
            max(seq_lens),
            block_table,
            scale,
            [-1, -1],
            None,
        )
    ref = _reference(q, deq_k, deq_v, q_lens, seq_lens, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_fa_int8_prefill_groups_through_small_scratch():
    """More context than the scratch holds: several dequant groups."""
    torch.manual_seed(0)
    nkv, block_size, dtype = 2, 64, torch.bfloat16
    seq_lens, q_lens = [900, 1000, 700, 1100], [100, 1, 50, 1]
    num_blocks = sum(-(-L // block_size) for L in seq_lens) + 4
    _, views = _make_cache(num_blocks, nkv, block_size, HEAD_SIZE, True)
    block_table, deq_k, deq_v = _write(
        seq_lens, nkv, block_size, HEAD_SIZE, views, dtype
    )
    q = torch.randn(sum(q_lens), 16, HEAD_SIZE, device=DEVICE, dtype=dtype)
    out = torch.empty_like(q)
    qsl_cpu = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32
    )
    nb = prefill_scratch_tokens(1200, block_size) // block_size
    ks = torch.empty((nb, block_size, nkv, HEAD_SIZE), dtype=dtype, device=DEVICE)
    vs = torch.empty_like(ks)
    key_cache, value_cache, k_scale, v_scale = views
    forward_prefill(
        q,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        out,
        ks,
        vs,
        qsl_cpu.to(DEVICE),
        qsl_cpu,
        torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        seq_lens,
        block_table,
        HEAD_SIZE**-0.5,
        True,
        [-1, -1],
        None,
    )
    ref = _reference(q, deq_k, deq_v, q_lens, seq_lens, HEAD_SIZE**-0.5)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

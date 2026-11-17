# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for int8_per_token_head KV on the XPU FlashAttention backend:
backend selection, the KV spec, the split-layout views over the cache bytes
and the prefill grouping. Kernel numerics:
tests/kernels/attention/test_xpu_fa_int8_per_token_head.py (XPU)."""

from unittest.mock import patch

import pytest
import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.ops.xpu_int8_per_token_head import (
    plan_prefill_groups,
    prefill_scratch_tokens,
    split_layout_views,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

PLATFORM = "vllm.v1.attention.backends.flash_attn.current_platform"


def _spec(kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, block_size=64, nkv=2):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=nkv,
        head_size=256,
        dtype=torch.int8 if kv_quant_mode != KVQuantMode.NONE else torch.bfloat16,
        kv_quant_mode=kv_quant_mode,
    )


@pytest.mark.parametrize("is_xpu", [True, False])
def test_supports_kv_cache_dtype(is_xpu):
    with patch(PLATFORM) as platform:
        platform.is_xpu.return_value = is_xpu
        assert (
            FlashAttentionBackend.supports_kv_cache_dtype("int8_per_token_head")
            is is_xpu
        )
        assert not FlashAttentionBackend.supports_kv_cache_dtype("fp8_per_token_head")


@pytest.mark.parametrize("block_size", [64, 832])
@pytest.mark.parametrize("nkv", [1, 2, 4])
def test_spec_matches_triton(block_size, nkv):
    """Same bytes per block as TRITON_ATTN, so hybrid pages line up."""
    spec = _spec(block_size=block_size, nkv=nkv)
    with patch(PLATFORM) as platform:
        platform.is_xpu.return_value = True
        fa = FlashAttentionBackend.customize_spec(spec)
    assert fa == TritonAttentionBackend.customize_spec(spec)
    assert fa.state_content_bytes is not None
    assert (
        fa.page_size_bytes
        == TritonAttentionBackend.customize_spec(spec).page_size_bytes
    )


def test_spec_unchanged_elsewhere():
    spec = _spec()
    with patch(PLATFORM) as platform:
        platform.is_xpu.return_value = False
        assert FlashAttentionBackend.customize_spec(spec) == spec
    plain = _spec(KVQuantMode.NONE)
    with patch(PLATFORM) as platform:
        platform.is_xpu.return_value = True
        assert FlashAttentionBackend.customize_spec(plain) == plain


def test_cudagraph_support_uniform_batch():
    with patch(PLATFORM) as platform:
        platform.is_xpu.return_value = True
        support = FlashAttentionMetadataBuilder.get_cudagraph_support(
            None,  # type: ignore[arg-type]
            _spec(),
        )
    assert support == AttentionCGSupport.UNIFORM_BATCH


def _layer_cache(num_blocks, nkv, block_size, content, layout, pad=0):
    """Logical [B, H, N, C] int8 layer view over raw bytes, as the runner
    builds it: HND (H outer) or NHD (N outer) inside each block, optionally
    with a padded block stride. Returns (raw, view)."""
    page = nkv * block_size * content + pad
    raw = torch.zeros(num_blocks * page, dtype=torch.int8)
    if layout == "HND":
        strides = (page, block_size * content, content, 1)
    else:
        strides = (page, content, nkv * content, 1)
    return raw, torch.as_strided(raw, (num_blocks, nkv, block_size, content), strides)


@pytest.mark.parametrize("content", [2 * (256 + 4), 2 * 272])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("pad", [0, 1024])
@pytest.mark.parametrize("nkv", [1, 2])
def test_split_layout_byte_positions(content, layout, pad, nkv):
    hs, bs, nb = 256, 64, 3
    raw, cache = _layer_cache(nb, nkv, bs, content, layout, pad)
    key, value, k_scale, v_scale = split_layout_views(cache, nkv, hs)
    page = nkv * bs * content + pad
    assert key.shape == value.shape == (nb, bs, nkv, hs)
    assert k_scale.shape == v_scale.shape == (nb, bs, nkv)
    assert key.stride() == value.stride() == (page, nkv * 2 * hs, 2 * hs, 1)
    assert k_scale.dtype == torch.float32
    assert k_scale.stride() == v_scale.stride() == (page // 4, nkv, 1)
    base = raw.data_ptr()
    data = bs * nkv * 2 * hs
    for b, n, h in [(0, 0, 0), (2, bs - 1, nkv - 1), (1, 5, nkv // 2)]:
        cell = b * page + n * nkv * 2 * hs + h * 2 * hs
        assert key[b, n, h].data_ptr() - base == cell
        assert value[b, n, h].data_ptr() - base == cell + hs
        scale = b * page + data + (n * nkv + h) * 4
        assert k_scale[b, n, h].data_ptr() - base == scale
        assert v_scale[b, n, h].data_ptr() - base == scale + bs * nkv * 4
    # Everything the views cover stays inside each block's (H, N, C) bytes.
    assert data + 2 * bs * nkv * 4 <= nkv * bs * content
    # Kernel contract: 64-byte aligned K and V surface bases.
    assert page % 64 == 0 and (2 * hs) % 64 == 0 and hs % 64 == 0


def test_split_layout_round_trip_isolated():
    """Writing through the views touches only the split-layout bytes of the
    addressed block."""
    hs, nkv, bs, content = 256, 2, 64, 2 * 272
    raw, cache = _layer_cache(2, nkv, bs, content, "HND", pad=512)
    key, value, k_scale, v_scale = split_layout_views(cache, nkv, hs)
    key[1, 3, 1] = 7
    value[1, 3, 1] = -7
    k_scale[1, 3, 1] = 0.5
    v_scale[1, 3, 1] = 2.0
    assert torch.equal(key[1, 3, 1], torch.full((hs,), 7, dtype=torch.int8))
    assert torch.equal(value[1, 3, 1], torch.full((hs,), -7, dtype=torch.int8))
    assert k_scale[1, 3, 1].item() == 0.5 and v_scale[1, 3, 1].item() == 2.0
    page = nkv * bs * content + 512
    assert not raw[:page].any()  # block 0 untouched
    scale_bytes = int((torch.tensor([0.5, 2.0]).view(torch.int8) != 0).sum())
    assert int((raw[page:] != 0).sum()) == 2 * hs + scale_bytes


@pytest.mark.parametrize(
    "case, error",
    [
        ("head_size", "multiple of 64"),
        ("block_stride", "multiple of 64"),
        ("interleaved", "dense run"),
        ("dtype", "int8"),
    ],
)
def test_split_layout_refusals(case, error):
    hs, nkv, bs, content, pad = 256, 2, 64, 2 * 260, 0
    if case == "head_size":
        hs, content = 96, 2 * 100
    if case == "block_stride":
        pad = 8
    _, cache = _layer_cache(3, nkv, bs, content, "HND", pad=pad)
    if case == "interleaved":
        # Other layers' bytes between this layer's heads.
        page = 2 * nkv * bs * content
        raw = torch.zeros(3 * page, dtype=torch.int8)
        cache = torch.as_strided(
            raw, (3, nkv, bs, content), (page, 2 * bs * content, content, 1)
        )
    if case == "dtype":
        cache = cache.view(torch.uint8)
    with pytest.raises(ValueError, match=error):
        split_layout_views(cache, nkv, hs)


def test_plan_prefill_groups():
    # block 64, capacity 10 blocks
    assert plan_prefill_groups([64, 64, 64], 64, 10) == [(0, 3, 1)]
    # strides follow the longest sequence of each group
    assert plan_prefill_groups([300, 10, 10, 10], 64, 10) == [(0, 2, 5), (2, 4, 1)]
    assert plan_prefill_groups([640], 64, 10) == [(0, 1, 10)]
    # empty / zero-length rows still take one block
    assert plan_prefill_groups([0, 1], 64, 10) == [(0, 2, 1)]
    assert plan_prefill_groups([], 64, 10) == []
    with pytest.raises(ValueError, match="exceeds"):
        plan_prefill_groups([641], 64, 10)


# (max_model_len, layer cache block) of the QFN serving definitions: two cards
# 131072 in blocks of 6272 (full-attention block grown from 3136 at KV
# planning), four cards 262144 in blocks of 1600; plus edges around them.
_SERVING_GRID = [(131072, 6272), (131072, 3136), (262144, 1600), (131072, 1600)]
_GRID = _SERVING_GRID + [
    (max_len, bs)
    for max_len in (1, 999, 1000, 4096, 32768, 125440, 125441, 131071, 262144)
    for bs in (16, 64, 832, 1024, 1600, 3136, 4352, 6272, 12544)
]


@pytest.mark.parametrize("max_len, block_size", _GRID)
def test_prefill_scratch_holds_max_model_len(max_len, block_size):
    """In the layer's cache block, the scratch holds a max_model_len sequence
    (capacity >= need) and every shorter one, with no spare block."""
    tokens = prefill_scratch_tokens(max_len, block_size)
    capacity = tokens // block_size
    need = -(-max_len // block_size)
    assert capacity >= need
    assert tokens == need * block_size  # no spare block
    for seq_len in {1, max_len // 2 or 1, max_len - 1 or 1, max_len}:
        assert plan_prefill_groups([seq_len], block_size, capacity) == [
            (0, 1, max(1, -(-seq_len // block_size)))
        ]


@pytest.mark.parametrize("max_len, block_size", _SERVING_GRID)
def test_prefill_scratch_holds_max_model_len_in_dividing_blocks(max_len, block_size):
    """The same scratch holds the sequence in any block that divides the
    layer's block (a kernel block split from it)."""
    tokens = prefill_scratch_tokens(max_len, block_size)
    for kernel_bs in (64, block_size // 2, block_size):
        if block_size % kernel_bs == 0:
            assert plan_prefill_groups([max_len], kernel_bs, tokens // kernel_bs)


def test_check_prefill_scratch_refuses_a_short_scratch():
    from vllm.v1.attention.ops.xpu_int8_per_token_head import check_prefill_scratch

    check_prefill_scratch(21, 131072, 6272)
    check_prefill_scratch(164, 262144, 1600)
    with pytest.raises(ValueError, match="needs 21"):
        check_prefill_scratch(20, 131072, 6272)


class _MetaWorkspace:
    """Workspace manager stand-in: meta tensors of the requested shapes."""

    def get_simultaneous(self, *shapes_and_dtypes):
        return [torch.empty(s, dtype=d, device="meta") for s, d in shapes_and_dtypes]


@pytest.mark.parametrize(
    "max_len, init_bs, aligned_bs, layer_bs, prompt",
    [
        # Two cards: the layer is built at the default 64, the platform aligns
        # the hybrid block to 3136, KV planning grows full attention to 6272;
        # a 130,000-token prompt is 21 of those blocks.
        (131072, 64, 3136, 6272, 130000),
        (131072, 64, 3136, 6272, 131072),
        # Four cards: 64 -> 1600, a 262,144-token sequence is 164 blocks.
        (262144, 64, 1600, 1600, 262144),
    ],
)
def test_scratch_follows_the_layer_block_not_the_build_time_block(
    max_len, init_bs, aligned_bs, layer_bs, prompt
):
    """Regression: the scratch was sized with the block size seen when the
    layer was built (64), which is smaller than the layer's cache block after
    hybrid alignment and KV planning; 20 blocks of 6272 then held only
    125,440 tokens and a 130,000-token prompt raised 'a sequence of 21 blocks
    exceeds the INT8 KV prefill scratch of 20 blocks' in every worker."""
    from types import SimpleNamespace

    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv

    impl = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        alibi_slopes=None,
        logits_soft_cap=0,
        head_size=256,
        num_kv_heads=1,
        dcp_world_size=1,
    )
    cache_config = SimpleNamespace(block_size=init_bs)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_len, dtype=torch.bfloat16),
        cache_config=cache_config,
    )
    with (
        patch(PLATFORM) as platform,
        patch.object(int8_kv, "int8_kv_decode_supported", lambda *a: None),
        patch(
            "vllm.v1.attention.backends.flash_attn.current_workspace_manager",
            _MetaWorkspace,
        ),
    ):
        platform.is_xpu.return_value = True
        FlashAttentionImpl._init_int8_per_token_head_kv(impl, vllm_config)
        # After the layers are built: hybrid alignment, then KV planning.
        cache_config.block_size = aligned_bs
        key_scratch, value_scratch = FlashAttentionImpl._int8_kv_prefill_scratch(
            impl, layer_bs
        )
    assert key_scratch.shape == value_scratch.shape
    assert key_scratch.shape[1:] == (layer_bs, 1, 256)
    groups = plan_prefill_groups([prompt], layer_bs, key_scratch.shape[0])
    assert groups == [(0, 1, -(-prompt // layer_bs))]


def _triton_runs_here() -> bool:
    import os

    return os.environ.get("TRITON_INTERPRET") == "1"  # CPU tensors


def _paged_ref(q, k_pages, v_pages, block_table, seq_lens, cu_q, scale):
    """Causal paged attention reference over a bf16 paged cache."""
    outs = []
    for i in range(len(seq_lens)):
        q0, q1, L = int(cu_q[i]), int(cu_q[i + 1]), int(seq_lens[i])
        bs = k_pages.shape[1]
        pos = torch.arange(L)
        blocks = block_table[i, pos // bs].long()
        k = k_pages[blocks, pos % bs].float()
        v = v_pages[blocks, pos % bs].float()
        g = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        s = torch.einsum("qhd,khd->hqk", q[q0:q1].float(), k) * scale
        qpos = torch.arange(L - (q1 - q0), L)
        s = s.masked_fill(~(pos[None, :] <= qpos[:, None])[None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v))
    return torch.cat(outs)


@pytest.mark.skipif(not _triton_runs_here(), reason="needs TRITON_INTERPRET=1")
def test_forward_prefill_dequant_and_groups(monkeypatch):
    """Dequant kernel + grouping + scratch block tables, with a stand-in for
    the FA kernel: equals attention over the dequantized cache, and stale
    bytes past each sequence end (NaN scales included) never reach it."""
    import sys
    import types

    from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv

    calls = []

    def fake_fa(
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        out,
        block_table,
        seqused_k,
        window_size,
        s_aux,
    ):
        calls.append(block_table.shape)
        for i, L in enumerate(seqused_k.tolist()):
            # Every slot of every referenced block (tails included) is finite.
            used = block_table[i, : -(-L // k.shape[1])].long()
            assert torch.isfinite(k[used]).all() and torch.isfinite(v[used]).all()
        out.copy_(
            _paged_ref(q, k, v, block_table, seqused_k, cu_seqlens_q, softmax_scale).to(
                out.dtype
            )
        )

    stub = types.ModuleType("vllm._xpu_ops")
    stub.xpu_ops = types.SimpleNamespace(  # type: ignore[attr-defined]
        flash_attn_varlen_func=fake_fa
    )
    monkeypatch.setitem(sys.modules, "vllm._xpu_ops", stub)

    torch.manual_seed(0)
    hs, nkv, bs = 64, 2, 16
    seq_lens, q_lens = [40, 7, 33, 20], [5, 1, 33, 2]
    nb = sum(-(-L // bs) for L in seq_lens) + 2
    raw, cache = _layer_cache(nb, nkv, bs, 2 * (hs + 4), "HND")
    raw.copy_(torch.randint(-128, 127, raw.shape, dtype=torch.int8))
    key, value, k_scale, v_scale = split_layout_views(cache, nkv, hs)
    k_scale.copy_(torch.rand(k_scale.shape) * 0.05)
    v_scale.copy_(torch.rand(v_scale.shape) * 0.05)
    perm = torch.randperm(nb - 1) + 1
    bt = torch.zeros(
        len(seq_lens), max(-(-L // bs) for L in seq_lens), dtype=torch.int32
    )
    c = 0
    for i, L in enumerate(seq_lens):
        n = -(-L // bs)
        bt[i, :n] = perm[c : c + n]
        c += n
        tail = L % bs
        if tail:  # stale scales past the end of the sequence
            k_scale[bt[i, n - 1], tail:] = float("nan")
            v_scale[bt[i, n - 1], tail:] = float("inf")
    dq_k = key.float() * k_scale.nan_to_num(0, 0, 0)[..., None]
    dq_v = value.float() * v_scale.nan_to_num(0, 0, 0)[..., None]
    q = torch.randn(sum(q_lens), nkv * 4, hs, dtype=torch.bfloat16)
    cu = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32)
    out = torch.empty_like(q)
    # Six blocks force several groups; NaN shows any slot left unwritten.
    ks = torch.full((6, bs, nkv, hs), float("nan"), dtype=torch.bfloat16)
    vs = torch.full_like(ks, float("nan"))
    int8_kv.forward_prefill(
        q,
        key,
        value,
        k_scale,
        v_scale,
        out,
        ks,
        vs,
        cu,
        cu,
        torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens,
        bt,
        hs**-0.5,
        True,
        [-1, -1],
        None,
    )
    assert len(calls) > 1
    ref = _paged_ref(q, dq_k, dq_v, bt, seq_lens, cu, hs**-0.5)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

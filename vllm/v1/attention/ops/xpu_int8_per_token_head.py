# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 per-token-head KV cache on the XPU FlashAttention backend.

vllm-xpu-kernels decodes an `int8_per_token_head` cache natively in its
paged decode kernel (`flash_attn_varlen_func` with int8 K/V and the fp32
scale caches as `k_descale` / `v_descale`). The kernel loads K/V tiles with
2D block loads, which need 64-byte aligned surface bases, so it reads the block
in a block-local split layout::

    [int8 K|V for every (slot, head) | K scales fp32 | V scales fp32]

A FlashAttention layer keeps exactly the bytes per block that TRITON_ATTN's
per-token-head spec reserves (`TritonAttentionBackend.customize_spec`), so
page sizes and hybrid page unification are the same for both backends; only the
arrangement inside a block differs, and a layer's blocks are written and read by
that layer's backend alone. The split layout uses `H * N * (2 * D + 8)` bytes
of the `H * N * C` bytes of a block (`C = 2 * (D + 4)`, padded to 16 bytes
per side when the spec pads the slots); the rest stays unused.

The kernel has no INT8 prefill. Prefill and mixed batches dequantize the blocks
their sequences reference into a bf16/fp16 paged scratch (one layer, sized to
`max_model_len`, taken from the workspace manager) and run the stock kernel
over it. The grouping reads only host copies the scheduler already has. With
`AttentionConfig.int8_prefill_dot` set, causal prefill attends over the int8
cache in place instead, with int8 dots and no scratch
(`xpu_int8_prefill_attention`, Triton), or the vllm-xpu-kernels chunk-prefill
kernel reads it in place (`forward_prefill_sycl`, the "sycl_*" values). With
`AttentionConfig.int8_decode_dot` set, the
paged decode kernel quantizes the query itself (the same rule) and runs Q.K as
an int8 x int8 -> int32 dot (`forward_decode(..., int8_dot=True)`).
"""

import functools
from collections.abc import Sequence

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

# Alignment the kernel's 2D block loads need for every K/V surface base.
_SURFACE_ALIGN = 64


def check_split_layout(
    kv_cache: torch.Tensor, num_kv_heads: int, head_size: int
) -> None:
    """Raise ValueError unless `kv_cache` (one layer, logical
    `[num_blocks, H, N, C]` int8) can be read in the split layout."""
    if kv_cache.dtype != torch.int8 or kv_cache.dim() != 4:
        raise ValueError(
            "int8_per_token_head KV cache must be a 4-D int8 tensor, got "
            f"{kv_cache.dtype} {tuple(kv_cache.shape)}"
        )
    num_blocks, nkv, block_size, content = kv_cache.shape
    if nkv != num_kv_heads or content < 2 * (head_size + 4):
        raise ValueError(
            f"int8_per_token_head KV cache shape {tuple(kv_cache.shape)} does "
            f"not match num_kv_heads={num_kv_heads}, head_size={head_size}"
        )
    if head_size % _SURFACE_ALIGN != 0:
        raise ValueError(
            "int8_per_token_head on XPU FlashAttention needs a head size that "
            f"is a multiple of {_SURFACE_ALIGN}, got {head_size}"
        )
    page_bytes = nkv * block_size * content
    span = 1 + sum((kv_cache.shape[d] - 1) * kv_cache.stride(d) for d in (1, 2, 3))
    if kv_cache.stride(3) != 1 or span != page_bytes:
        raise ValueError(
            "int8_per_token_head on XPU FlashAttention needs each block's "
            f"(H, N, C) bytes to be one dense run; got strides {kv_cache.stride()} "
            f"for shape {tuple(kv_cache.shape)}"
        )
    page = kv_cache.stride(0) if num_blocks > 1 else page_bytes
    if page < page_bytes or page % _SURFACE_ALIGN != 0:
        raise ValueError(
            "int8_per_token_head on XPU FlashAttention needs a block stride "
            f"that is a multiple of {_SURFACE_ALIGN} bytes, got {page}"
        )
    if kv_cache.data_ptr() % _SURFACE_ALIGN != 0:
        raise ValueError(
            "int8_per_token_head on XPU FlashAttention needs a "
            f"{_SURFACE_ALIGN}-byte aligned KV cache base"
        )


def split_layout_views(
    kv_cache: torch.Tensor, num_kv_heads: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split-layout views over one layer's `[num_blocks, H, N, C]` int8
    cache.

    Returns `(key, value, k_scale, v_scale)`: key / value int8
    `[num_blocks, N, H, D]` with strides `(page, H * 2D, 2D, 1)` (value
    starts `D` bytes after key) and k_scale / v_scale fp32 `[num_blocks, N,
    H]` with strides `(page / 4, H, 1)` placed after the int8 data. `page`
    is the tensor's block stride, so padded pages work. This is the layout
    vllm-xpu-kernels checks in `check_int8_kv_cache`.
    """
    check_split_layout(kv_cache, num_kv_heads, head_size)
    num_blocks, nkv, block_size, content = kv_cache.shape
    page = kv_cache.stride(0) if num_blocks > 1 else nkv * block_size * content
    data_bytes = block_size * nkv * 2 * head_size
    scale_bytes = block_size * nkv * 4
    data = torch.as_strided(
        kv_cache,
        (num_blocks, block_size, nkv, 2 * head_size),
        (page, nkv * 2 * head_size, 2 * head_size, 1),
    )

    def scales(offset: int) -> torch.Tensor:
        raw = torch.as_strided(
            kv_cache,
            (num_blocks, scale_bytes),
            (page, 1),
            kv_cache.storage_offset() + offset,
        )
        # int8 [B, N*H*4] -> fp32 [B, N*H] (strides page/4, 1) -> [B, N, H]
        return raw.view(torch.float32).view(num_blocks, block_size, nkv)

    return (
        data[..., :head_size],
        data[..., head_size:],
        scales(data_bytes),
        scales(data_bytes + scale_bytes),
    )


def decode_max_query_len() -> int:
    """Largest query length per sequence that vllm-xpu-kernels runs through
    its paged decode kernel (causal batches above 1 take its small-q
    speculative-decode route); 1 when the bound is not exposed."""
    try:
        from vllm_xpu_kernels import flash_attn_interface
    except ImportError:
        return 1
    return max(1, int(getattr(flash_attn_interface, "_SPEC_DECODE_MAX_QLEN", 1)))


@functools.cache
def int8_kv_decode_supported(head_size: int, dtype: torch.dtype) -> str | None:
    """Run the INT8 per-token-head paged decode once on a one-block cache.
    Returns None when it works, else the reason (the vllm-xpu-kernels build
    predates INT8 KV support, or lacks this configuration)."""
    from vllm._xpu_ops import xpu_ops

    try:
        nkv, block_size = 1, 64
        content = 2 * (head_size + 4)
        dev = torch.device("xpu", torch.xpu.current_device())
        raw = torch.zeros((2, nkv, block_size, content), dtype=torch.int8, device=dev)
        key, value, k_scale, v_scale = split_layout_views(raw, nkv, head_size)
        q = torch.zeros((1, nkv, head_size), dtype=dtype, device=dev)
        out = torch.empty_like(q)
        xpu_ops.flash_attn_varlen_func(
            q,
            key,
            value,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=dev),
            max_seqlen_q=1,
            max_seqlen_k=1,
            softmax_scale=1.0,
            causal=True,
            out=out,
            block_table=torch.ones((1, 1), dtype=torch.int32, device=dev),
            seqused_k=torch.ones((1,), dtype=torch.int32, device=dev),
            k_descale=k_scale,
            v_descale=v_scale,
        )
        torch.xpu.synchronize()
    except Exception as e:  # noqa: BLE001 - any failure means unsupported
        return f"{type(e).__name__}: {e}"
    return None


@functools.cache
def int8_dot_decode_supported(head_size: int, dtype: torch.dtype) -> str | None:
    """Run the paged decode once with `int8_dot=True` on a one-block cache.
    Returns None when it works, else the reason (a vllm-xpu-kernels build
    without INT8 Q.K at decode, a query dtype other than bfloat16, or a head
    size the kernel does not cover)."""
    from vllm._xpu_ops import xpu_ops

    if dtype != torch.bfloat16:
        return f"query dtype {dtype} (bfloat16 only)"
    if head_size > 256 or head_size % 64:
        return f"head size {head_size} (a multiple of 64 up to 256)"
    try:
        from vllm_xpu_kernels import flash_attn_interface
    except ImportError as e:
        return f"{type(e).__name__}: {e}"
    if not getattr(flash_attn_interface, "INT8_DOT_DECODE", False):
        return "the installed vllm-xpu-kernels has no int8_dot decode"
    try:
        nkv, block_size = 1, 64
        content = 2 * (head_size + 4)
        dev = torch.device("xpu", torch.xpu.current_device())
        raw = torch.zeros((2, nkv, block_size, content), dtype=torch.int8, device=dev)
        key, value, k_scale, v_scale = split_layout_views(raw, nkv, head_size)
        q = torch.zeros((1, nkv, head_size), dtype=dtype, device=dev)
        out = torch.empty_like(q)
        xpu_ops.flash_attn_varlen_func(
            q,
            key,
            value,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=dev),
            max_seqlen_q=1,
            max_seqlen_k=1,
            softmax_scale=1.0,
            causal=True,
            out=out,
            block_table=torch.ones((1, 1), dtype=torch.int32, device=dev),
            seqused_k=torch.ones((1,), dtype=torch.int32, device=dev),
            k_descale=k_scale,
            v_descale=v_scale,
            int8_dot=True,
        )
        torch.xpu.synchronize()
    except Exception as e:  # noqa: BLE001 - any failure means unsupported
        return f"{type(e).__name__}: {e}"
    return None


# Rows of the probe: above the kernels' spec-decode bound, so the call takes the
# chunk-prefill kernel, not the paged decode route.
_SYCL_PROBE_ROWS = 40


@functools.cache
def int8_prefill_sycl_supported(
    head_size: int, dtype: torch.dtype, mode: int, block_size: int
) -> str | None:
    """Run the chunk-prefill kernel once with `int8_prefill=mode` (1, 2 or 3)
    on a small cache. Returns None when it works, else the reason (a
    vllm-xpu-kernels build without INT8 KV prefill, or a configuration it
    does not cover: bfloat16 query, head size 256, cache blocks of 32 or a
    multiple of 64 slots)."""
    from vllm._xpu_ops import xpu_ops

    if dtype != torch.bfloat16:
        return f"query dtype {dtype} (bfloat16 only)"
    if head_size != 256:
        return f"head size {head_size} (256 only)"
    if not (block_size == 32 or block_size % 64 == 0):
        return f"cache block of {block_size} slots (32 or a multiple of 64)"
    try:
        from vllm_xpu_kernels import flash_attn_interface
    except ImportError as e:
        return f"{type(e).__name__}: {e}"
    if mode not in getattr(flash_attn_interface, "INT8_PREFILL_MODES", {}):
        return f"the installed vllm-xpu-kernels has no int8_prefill={mode}"
    try:
        nkv, block = 1, 64
        content = 2 * (head_size + 4)
        dev = torch.device("xpu", torch.xpu.current_device())
        raw = torch.zeros((2, nkv, block, content), dtype=torch.int8, device=dev)
        key, value, k_scale, v_scale = split_layout_views(raw, nkv, head_size)
        rows = _SYCL_PROBE_ROWS
        q = torch.zeros((rows, nkv, head_size), dtype=dtype, device=dev)
        out = torch.empty_like(q)
        xpu_ops.flash_attn_varlen_func(
            q,
            key,
            value,
            cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32, device=dev),
            max_seqlen_q=rows,
            max_seqlen_k=rows,
            softmax_scale=1.0,
            causal=True,
            out=out,
            block_table=torch.ones((1, 1), dtype=torch.int32, device=dev),
            seqused_k=torch.tensor([rows], dtype=torch.int32, device=dev),
            k_descale=k_scale,
            v_descale=v_scale,
            int8_prefill=mode,
        )
        torch.xpu.synchronize()
    except Exception as e:  # noqa: BLE001 - any failure means unsupported
        return f"{type(e).__name__}: {e}"
    return None


def is_graph_capturing() -> bool:
    """Whether the current XPU stream is capturing a graph."""
    capturing = getattr(
        getattr(torch, "xpu", None), "is_current_stream_capturing", None
    )
    return bool(capturing()) if capturing is not None else False


# ---------------------------------------------------------------------------
# Prefill: dequantize the referenced blocks into a paged scratch
# ---------------------------------------------------------------------------


def plan_prefill_groups(
    seq_lens_upper_bound: Sequence[int],
    block_size: int,
    capacity_blocks: int,
) -> list[tuple[int, int, int]]:
    """Split sequences `[0, len(seq_lens_upper_bound))` into consecutive
    groups `(first, end, stride)` whose `(end - first) * stride` scratch
    blocks fit `capacity_blocks`; sequence `r` of a group gets scratch
    blocks `[(r - first) * stride, (r - first) * stride + stride)`.
    `stride` is the largest block count in the group. Host-only."""
    nblks = [max(1, cdiv(int(s), block_size)) for s in seq_lens_upper_bound]
    groups = []
    i, n = 0, len(nblks)
    while i < n:
        j, stride = i, 0
        while j < n and (j - i + 1) * max(stride, nblks[j]) <= capacity_blocks:
            stride = max(stride, nblks[j])
            j += 1
        if j == i:
            raise ValueError(
                f"a sequence of {nblks[i]} blocks exceeds the INT8 KV prefill "
                f"scratch of {capacity_blocks} blocks"
            )
        groups.append((i, j, stride))
        i = j
    return groups


# Prefill groups after the first pass row slices of the block table and of
# seq_lens, so their base pointers are not 16-byte aligned; those are scalar
# loads, and specialising on their alignment would only add compiled variants.
@triton.jit(do_not_specialize_on_alignment=["bt_ptr", "sl_ptr"])
def _dequant_blocks_kernel(
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    bt_ptr,
    sl_ptr,
    ok_ptr,
    ov_ptr,
    s_kb,
    s_kn,
    s_kh,
    s_vb,
    s_vn,
    s_vh,
    s_ksb,
    s_ksn,
    s_ksh,
    s_vsb,
    s_vsn,
    s_vsh,
    s_bt,
    stride_blocks,
    BS: tl.constexpr,
    NKV: tl.constexpr,
    HS: tl.constexpr,
    TOK: tl.constexpr,
    NKV_P2: tl.constexpr,
    HS_P2: tl.constexpr,
):
    # Program (r, c, t): tokens [t*TOK, (t+1)*TOK) of block c of sequence r
    # go to scratch block r * stride_blocks + c. Blocks past the sequence end
    # are skipped (the new block table never points at them); slots past the
    # sequence end inside its last block are written as zeros, so stale bytes
    # in the cache cannot reach the attention kernel as inf or NaN.
    r = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)
    seq_len = tl.load(sl_ptr + r)
    if c >= (seq_len + BS - 1) // BS:
        return
    src = tl.load(bt_ptr + r * s_bt + c).to(tl.int64)
    dst = (r * stride_blocks + c).to(tl.int64)
    tok = t * TOK + tl.arange(0, TOK)
    h = tl.arange(0, NKV_P2)
    d = tl.arange(0, HS_P2)
    m_t = tok < BS
    live = (c * BS + tok) < seq_len
    m3 = m_t[:, None, None] & (h < NKV)[None, :, None] & (d < HS)[None, None, :]
    m2 = m_t[:, None] & (h < NKV)[None, :]
    l3 = m3 & live[:, None, None]
    l2 = m2 & live[:, None]
    out_dtype = ok_ptr.dtype.element_ty
    o_off = (
        (dst * BS + tok)[:, None, None] * (NKV * HS)
        + h[None, :, None] * HS
        + d[None, None, :]
    )
    k = tl.load(
        k_ptr
        + src * s_kb
        + tok[:, None, None] * s_kn
        + h[None, :, None] * s_kh
        + d[None, None, :],
        mask=l3,
        other=0,
    )
    ks = tl.load(
        ks_ptr + src * s_ksb + tok[:, None] * s_ksn + h[None, :] * s_ksh,
        mask=l2,
        other=0.0,
    )
    tl.store(ok_ptr + o_off, (k.to(tl.float32) * ks[:, :, None]).to(out_dtype), mask=m3)
    v = tl.load(
        v_ptr
        + src * s_vb
        + tok[:, None, None] * s_vn
        + h[None, :, None] * s_vh
        + d[None, None, :],
        mask=l3,
        other=0,
    )
    vs = tl.load(
        vs_ptr + src * s_vsb + tok[:, None] * s_vsn + h[None, :] * s_vsh,
        mask=l2,
        other=0.0,
    )
    tl.store(ov_ptr + o_off, (v.to(tl.float32) * vs[:, :, None]).to(out_dtype), mask=m3)


def dequant_blocks(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    key_out: torch.Tensor,
    value_out: torch.Tensor,
    stride_blocks: int,
) -> None:
    """Dequantize the blocks sequence `r` of `block_table` references
    (`cdiv(seq_lens[r], N)` of them) into `key_out` / `value_out`
    (contiguous `[*, N, H, D]`) at blocks `r * stride_blocks + c`.
    `block_table` and `seq_lens` stay on the device."""
    _, block_size, nkv, head_size = key_cache.shape
    num_seqs = block_table.shape[0]
    if num_seqs == 0 or stride_blocks == 0:
        return
    tok = 16 if head_size * nkv <= 512 else 8
    _dequant_blocks_kernel[(num_seqs, stride_blocks, triton.cdiv(block_size, tok))](
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        block_table,
        seq_lens,
        key_out,
        value_out,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        block_table.stride(0),
        stride_blocks,
        BS=block_size,
        NKV=nkv,
        HS=head_size,
        TOK=tok,
        NKV_P2=triton.next_power_of_2(nkv),
        HS_P2=triton.next_power_of_2(head_size),
    )


def warmup_dequant_blocks(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    key_out: torch.Tensor,
    value_out: torch.Tensor,
) -> None:
    """Compile `_dequant_blocks_kernel` for every scratch stride class serving
    takes. Triton specialises the integer `stride_blocks` (a group's longest
    sequence, in blocks) on == 1 and on divisibility by 16; a warm-up prefill
    of short prompts reaches only the first. The block-table and sequence-length
    pointers are not specialised on alignment, so the row slices of later
    prefill groups need no variants of their own. Zero sequence lengths make
    every program return at once, so nothing is read or written."""
    seq_lens = torch.zeros(1, dtype=torch.int32, device=key_cache.device)
    for stride_blocks in (1, 2, 16):
        if stride_blocks <= key_out.shape[0]:
            dequant_blocks(
                key_cache,
                value_cache,
                k_scale,
                v_scale,
                block_table[:1],
                seq_lens,
                key_out,
                value_out,
                stride_blocks,
            )


def prefill_scratch_tokens(max_model_len: int, block_size: int) -> int:
    """Tokens of one layer's K (or V) prefill scratch: a `max_model_len`
    sequence in blocks of `block_size` or of any smaller kernel block
    size."""
    return (cdiv(max_model_len, block_size) + 1) * block_size


def forward_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    key_scratch: torch.Tensor,
    value_scratch: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_upper_bound_cpu: Sequence[int],
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: list[int] | None,
    sinks: torch.Tensor | None,
) -> None:
    """Attention for a batch with prefill rows over the INT8 cache: per group
    of sequences, one dequant launch into the scratch and one FA call over a
    uniform-stride block table built on the device. Host reads: only the
    metadata's host copies."""
    from vllm._xpu_ops import xpu_ops

    num_seqs = query_start_loc.numel() - 1
    block_size = key_cache.shape[1]
    qsl = query_start_loc_cpu[: num_seqs + 1].tolist()
    seq_ub = [int(s) for s in seq_lens_upper_bound_cpu[:num_seqs]]
    dev = query.device
    for first, end, stride in plan_prefill_groups(
        seq_ub, block_size, key_scratch.shape[0]
    ):
        g = end - first
        group_seq_lens = seq_lens[first:end]
        dequant_blocks(
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            block_table[first:end],
            group_seq_lens,
            key_scratch,
            value_scratch,
            stride,
        )
        scratch_table = (
            torch.arange(g, dtype=torch.int32, device=dev)[:, None] * stride
            + torch.arange(stride, dtype=torch.int32, device=dev)[None, :]
        )
        q0, q1 = qsl[first], qsl[end]
        xpu_ops.flash_attn_varlen_func(
            query[q0:q1],
            key_scratch,
            value_scratch,
            cu_seqlens_q=query_start_loc[first : end + 1] - q0,
            max_seqlen_q=max(qsl[r + 1] - qsl[r] for r in range(first, end)),
            max_seqlen_k=max(seq_ub[first:end]),
            softmax_scale=softmax_scale,
            causal=causal,
            out=output[q0:q1],
            block_table=scratch_table,
            seqused_k=group_seq_lens,
            window_size=window_size,
            s_aux=sinks,
        )


def forward_prefill_sycl(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_table: torch.Tensor,
    softmax_scale: float,
    mode: int,
    int8_dot: bool = False,
) -> None:
    """Causal batch with prefill rows over the INT8 cache in place: the
    vllm-xpu-kernels chunk-prefill kernel with `int8_prefill=mode` (1 BF16
    dots over the widened codes, 2 int8 Q.K, 3 int8 Q.K and P.V). Decode rows
    of a mixed batch take the paged decode kernel inside the same call
    (`int8_dot` as at decode). No scratch, no host reads."""
    from vllm._xpu_ops import xpu_ops

    xpu_ops.flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        out=output,
        block_table=block_table,
        seqused_k=seq_lens,
        window_size=[-1, -1],
        k_descale=k_scale,
        v_descale=v_scale,
        int8_dot=int8_dot,
        int8_prefill=mode,
    )


def forward_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_table: torch.Tensor,
    softmax_scale: float,
    window_size: list[int] | None,
    sinks: torch.Tensor | None,
    int8_dot: bool = False,
) -> None:
    """Causal batch with `max_query_len <= decode_max_query_len()`: the
    native INT8 paged decode (graph-safe, no host reads). `int8_dot`: the
    kernel quantizes Q per (token, head) and runs Q.K in int8."""
    from vllm._xpu_ops import xpu_ops

    xpu_ops.flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        out=output,
        block_table=block_table,
        seqused_k=seq_lens,
        window_size=window_size,
        k_descale=k_scale,
        v_descale=v_scale,
        s_aux=sinks,
        int8_dot=int8_dot,
    )

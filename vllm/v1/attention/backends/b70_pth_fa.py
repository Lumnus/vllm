# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-IP-K4: prefill over an int8 per-token-head KV cache through the XPU FA kernel.

On XPU the Triton unified attention kernel is ~100x slower than the SYCL
flash-attention kernel for prefill chunks (B70, hs=256: 4096 queries over 32K
context 2,705-3,034 ms vs 28.9 ms). Dequantizing the int8 context to bf16 costs
<1 ms per layer at 128K, so for batches that carry a real prefill we gather the
blocks each sequence references, dequantize them into a one-layer bf16 scratch,
and call ``flash_attn_varlen_func`` over a remapped block table. Decode-only
batches (and spec-decode verify batches, q <= B70_PTH_FA_MIN_Q) stay on the
Triton kernel and its CUDA graphs.

The scratch holds max_model_len tokens of one layer (K and V, bf16), shared by
all layers (they run in sequence), allocated in the first eager forward (the
memory-profile run) so the KV pool is sized around it. Sequences whose total
context exceeds it are processed in groups.

Env: B70_PTH_FA_PREFILL=0 turns it off; B70_PTH_FA_MIN_Q (default 64).

B70-IP-D2 (``B70_PTH_FA_DECODE=1``, default off): FLASH_ATTN on XPU accepts
``int8_per_token_head`` and decodes natively with the vllm-xpu-kernels INT8
per-token-head paged decode (wiesion, intel/llm-scaler#744). Its layers keep the
same bytes per block as TRITON_ATTN's inline layout (page sizes, hybrid unify
and B70-IP-K3 unchanged) but arrange them as the kernel's block-local split
layout ``[int8 K|V for all (slot, head) | K scales | V scales]``, so every K/V
surface base is 64-byte aligned. Each layer's pages are only ever touched by its
own backend, so the inline layout of TRITON_ATTN / QSA layers (drafter, QSA) is
untouched. Batches with max q <= the kernels' spec-decode bound run the native
decode (causal rows through the kernels' per-position split-K route,
non-causal rows as one decode row each); everything else goes through the K4
dequant-to-bf16 scratch, forced on (no B70_PTH_FA_MIN_Q threshold).
"""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_scratch: dict = {}
_logged = False


def _enabled() -> bool:
    return (
        current_platform.is_xpu() and os.environ.get("B70_PTH_FA_PREFILL", "1") != "0"
    )


def fa_decode_enabled() -> bool:
    """B70-IP-D2 gate: FLASH_ATTN serves int8_per_token_head on XPU."""
    return current_platform.is_xpu() and os.environ.get("B70_PTH_FA_DECODE", "0") == "1"


def _get_scratch(block_size, nkv, hs, dtype, device, max_len):
    key = (block_size, nkv, hs, dtype, str(device))
    s = _scratch.get(key)
    if s is None:
        nblk = -(-max_len // block_size) + 1
        k = torch.empty((nblk, block_size, nkv, hs), dtype=dtype, device=device)
        v = torch.empty_like(k)
        s = (k, v)
        _scratch[key] = s
        logger.info(
            "B70-IP-K4 int8 KV prefill via FA: bf16 scratch %d blocks x %d tokens "
            "(%.2f GiB)",
            nblk,
            block_size,
            2 * k.numel() * k.element_size() / 2**30,
        )
    return s


def _max_len(impl) -> int:
    # Captured in TritonAttentionImpl.__init__ (the config is not current at forward).
    v = getattr(impl, "_b70_max_model_len", None)
    return int(v) if v else 131072


def ensure_scratch(impl, kv_cache: torch.Tensor, dtype, force: bool = False) -> None:
    """Allocate at the first forward (profile run) so KV sizing sees it."""
    if not (force or _enabled()):
        return
    num_blocks, nkv, block_size, content = kv_cache.shape
    hs = impl.head_size
    _get_scratch(block_size, nkv, hs, dtype, kv_cache.device, _max_len(impl))


def _dequant_blocks(src_data, src_scale, ids, dst, hs, chunk=4):
    # src_data: (nb, bs, nkv, hs+pad) int8 view; src_scale: (nb, bs, nkv) fp32
    n = ids.numel()
    for i in range(0, n, chunk):
        sel = ids[i : i + chunk]
        d = src_data[sel][..., :hs].to(dst.dtype)
        s = src_scale[sel].to(dst.dtype).unsqueeze(-1)
        dst[i : i + sel.numel()] = d * s


def _dequant_fp8_blocks(src, scale, ids, dst, chunk=4):
    # src: (nb, bs, nkv, hs) fp8 view; scale: per-tensor fp32 scalar tensor
    sc = scale.reshape(-1)[0].to(dst.dtype)
    n = ids.numel()
    for i in range(0, n, chunk):
        sel = ids[i : i + chunk]
        dst[i : i + sel.numel()] = src[sel].to(dst.dtype) * sc


def maybe_fa_prefill(
    impl,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    attn_metadata,
    num_actual_tokens: int,
    fp8_scales: tuple | None = None,
    pth_scales: tuple | None = None,
    force: bool = False,
    window_size: list[int] | None = None,
) -> bool:
    """int8 per-token-head (default) or, with fp8_scales=(k_scale, v_scale),
    an fp8 per-tensor cache (B70-IP-K5, FLASH_ATTN backend).

    pth_scales=(k_scale, v_scale) overrides the impl's inline scale views (the
    B70-IP-D2 split layout). force=True (B70-IP-D2) ignores B70_PTH_FA_PREFILL
    and B70_PTH_FA_MIN_Q, since the INT8 FA kernel has no prefill: it also
    accepts non-causal batches, sinks and the caller's window_size."""
    global _logged
    if not force:
        if not _enabled():
            return False
        if attn_metadata.max_query_len <= int(os.environ.get("B70_PTH_FA_MIN_Q", "64")):
            return False
    if (
        impl.alibi_slopes is not None
        or (getattr(impl, "logits_soft_cap", 0) or 0) != 0
        or getattr(attn_metadata, "mm_prefix_range_tensor", None) is not None
        or getattr(attn_metadata, "mm_prefix_query_range_tensor", None) is not None
        or getattr(attn_metadata, "rswa_prefix_lens", None) is not None
        or not isinstance(attn_metadata.causal, bool)
    ):
        return False
    if not force and (
        getattr(impl, "sinks", None) is not None
        or tuple(impl.sliding_window or (-1, -1)) != (-1, -1)
        or not attn_metadata.causal
    ):
        return False
    causal = attn_metadata.causal if force else True
    sinks = getattr(impl, "sinks", None) if force else None
    if not force:
        window_size = None
    from vllm._xpu_ops import xpu_ops

    nb, bs, nkv, _ = key_cache.shape
    hs = impl.head_size
    ks, vs = _get_scratch(bs, nkv, hs, query.dtype, query.device, _max_len(impl))
    cap = ks.shape[0]
    seq_lens = attn_metadata.seq_lens.tolist()
    qsl = attn_metadata.query_start_loc.tolist()
    nseq = len(seq_lens)
    bt = attn_metadata.block_table
    nblks = [-(-s // bs) for s in seq_lens]
    i = 0
    while i < nseq:
        j, used = i, 0
        while j < nseq and used + nblks[j] <= cap:
            used += nblks[j]
            j += 1
        if j == i:  # one sequence longer than the scratch (max_model_len): never
            return False
        ids = torch.cat([bt[r, : nblks[r]] for r in range(i, j)]).to(torch.long)
        if fp8_scales is None:
            k_sc, v_sc = pth_scales or (impl._k_scale_cache, impl._v_scale_cache)
            _dequant_blocks(key_cache, k_sc, ids, ks, hs)
            _dequant_blocks(value_cache, v_sc, ids, vs, hs)
        else:
            _dequant_fp8_blocks(key_cache, fp8_scales[0], ids, ks)
            _dequant_fp8_blocks(value_cache, fp8_scales[1], ids, vs)
        width = max(nblks[i:j])
        new_bt = torch.zeros((j - i, width), dtype=torch.int32, device=query.device)
        off = 0
        for r in range(i, j):
            new_bt[r - i, : nblks[r]] = torch.arange(
                off, off + nblks[r], dtype=torch.int32, device=query.device
            )
            off += nblks[r]
        q0, q1 = qsl[i], qsl[j]
        cu = torch.tensor(
            [q - q0 for q in qsl[i : j + 1]], dtype=torch.int32, device=query.device
        )
        sk = torch.tensor(seq_lens[i:j], dtype=torch.int32, device=query.device)
        xpu_ops.flash_attn_varlen_func(
            query[q0:q1],
            ks,
            vs,
            cu_seqlens_q=cu,
            max_seqlen_q=max(qsl[r + 1] - qsl[r] for r in range(i, j)),
            max_seqlen_k=max(seq_lens[i:j]),
            softmax_scale=impl.scale,
            causal=causal,
            out=output[q0:q1],
            block_table=new_bt,
            seqused_k=sk,
            window_size=window_size,
            s_aux=sinks,
        )
        i = j
    if not _logged:
        _logged = True
        logger.info(
            "B70-IP-K4/K5 %s KV prefill via FA bf16: first batch %d seqs, max q %d, "
            "max ctx %d",
            "int8" if fp8_scales is None else "fp8",
            nseq,
            attn_metadata.max_query_len,
            max(seq_lens),
        )
    return True


# ---------------------------------------------------------------------------
# B70-IP-D2: native INT8 per-token-head decode on the XPU FLASH_ATTN backend
# ---------------------------------------------------------------------------
_d2_logged = False


def log_fa_decode_once(head_size: int, num_kv_heads: int) -> None:
    global _d2_logged
    if not _d2_logged:
        _d2_logged = True
        logger.info(
            "B70-IP-D2 int8_per_token_head KV on XPU FLASH_ATTN: native INT8 paged "
            "decode (vllm-xpu-kernels, q <= %d per row), prefill via dequant to a "
            "bf16 scratch (K4), block-local split layout; head_size %d, kv heads %d",
            spec_decode_max_qlen(),
            head_size,
            num_kv_heads,
        )


def spec_decode_max_qlen() -> int:
    """Largest max_query_len that vllm-xpu-kernels runs through its split-K
    decode kernel (the only kernel that reads INT8 KV); 1 when the module or
    the bound is missing."""
    try:
        from vllm_xpu_kernels import flash_attn_interface as fai
    except ImportError:
        return 1
    return max(1, int(getattr(fai, "_SPEC_DECODE_MAX_QLEN", 1)))


def split_layout_views(
    kv_cache: torch.Tensor, num_kv_heads: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-local split views over one layer's int8 per-token-head cache.

    ``kv_cache`` is the logical ``(B, H, N, 2 * (D + 4))`` int8 layer view vLLM
    allocates for the spec (same bytes as TRITON_ATTN's inline layout). Each
    block is re-read as ``[int8 K|V (N, H, 2D) | K scales fp32 (N, H) |
    V scales fp32 (N, H)]``. Returns key / value int8 ``[B, N, H, D]`` with
    strides ``(page, H*2D, 2D, 1)`` (value = key + D bytes) and k_scale /
    v_scale fp32 ``[B, N, H]`` with strides ``(page/4, H, 1)``: the contract of
    vllm-xpu-kernels ``check_int8_kv_cache``. ``page`` is the tensor's block
    stride, so padded pages work; the page itself must be one dense run of bytes
    (a block-compact KV cache layout, B outside H, N, C).
    """
    if kv_cache.dtype != torch.int8:
        raise ValueError(f"int8_per_token_head cache is int8, got {kv_cache.dtype}")
    num_blocks, nkv, block_size, content = kv_cache.shape
    if nkv != num_kv_heads or content != 2 * (head_size + 4):
        raise ValueError(
            f"int8_per_token_head cache shape {tuple(kv_cache.shape)} does not match "
            f"num_kv_heads={num_kv_heads}, head_size={head_size}"
        )
    page_bytes = block_size * num_kv_heads * content
    span = 1 + sum((kv_cache.shape[d] - 1) * kv_cache.stride(d) for d in (1, 2, 3))
    if kv_cache.stride(3) != 1 or span != page_bytes:
        raise ValueError(
            "B70-IP-D2 needs a block-compact KV cache layout (each block's "
            f"(H, N, C) bytes dense); got strides {kv_cache.stride()} for shape "
            f"{tuple(kv_cache.shape)}"
        )
    page = kv_cache.stride(0) if num_blocks > 1 else page_bytes
    if page < page_bytes or page % 64 != 0 or head_size % 64 != 0:
        raise ValueError(
            "B70-IP-D2: the INT8 decode kernel needs 64-byte aligned K/V surfaces "
            f"(block stride {page} B, head_size {head_size}); both must be "
            "multiples of 64"
        )
    data_bytes = block_size * num_kv_heads * 2 * head_size
    scale_bytes = block_size * num_kv_heads * 4
    data = torch.as_strided(
        kv_cache,
        (num_blocks, block_size, num_kv_heads, 2 * head_size),
        (page, num_kv_heads * 2 * head_size, 2 * head_size, 1),
    )

    def scales(offset: int) -> torch.Tensor:
        raw = torch.as_strided(
            kv_cache,
            (num_blocks, scale_bytes),
            (page, 1),
            kv_cache.storage_offset() + offset,
        )
        # int8 [B, N*H*4] -> fp32 [B, N*H] (stride page/4, 1) -> [B, N, H]
        return raw.view(torch.float32).view(num_blocks, block_size, num_kv_heads)

    return (
        data[..., :head_size],
        data[..., head_size:],
        scales(data_bytes),
        scales(data_bytes + scale_bytes),
    )


def pth_split_views(impl, kv_cache: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Cached split views for the bound cache. They live in ``_k_scale_cache``
    (a tuple here), which the runner clears after profiling, so no view keeps
    the profiling cache alive."""
    key = (kv_cache.data_ptr(), tuple(kv_cache.shape), tuple(kv_cache.stride()))
    cached = impl._k_scale_cache
    if not isinstance(cached, tuple) or cached[0] != key:
        views = split_layout_views(kv_cache, impl.num_kv_heads, impl.head_size)
        impl._k_scale_cache = cached = (key, views)
    return cached[1]


def _capturing() -> bool:
    fn = getattr(getattr(torch, "xpu", None), "is_current_stream_capturing", None)
    try:
        return bool(fn()) if fn is not None else False
    except Exception:  # noqa: BLE001
        return False


def _noncausal_rows_decode(
    query, key_cache, value_cache, k_scale, v_scale, output, attn_metadata, impl
) -> None:
    """Non-causal small-q rows (DFlash draft blocks): every query row attends
    to its request's whole context, i.e. one q_len-1 decode row each. Graph-safe
    (no host sync); padded rows past query_start_loc[-1] read one slot of block 0
    and are zeroed."""
    from vllm._xpu_ops import xpu_ops

    total = query.shape[0]
    qsl = attn_metadata.query_start_loc
    batch = qsl.numel() - 1
    cu = torch.arange(total + 1, dtype=torch.int32, device=query.device)
    pos = cu[:-1]
    req = torch.searchsorted(qsl[1:], pos, right=True).clamp_(max=batch - 1)
    is_pad = pos >= qsl[-1]
    used = (
        attn_metadata.seq_lens.index_select(0, req)
        .clamp(min=1)
        .masked_fill(is_pad, 1)
        .to(torch.int32)
    )
    bt = attn_metadata.block_table.index_select(0, req)
    bt.masked_fill_(is_pad[:, None], 0)
    xpu_ops.flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu,
        max_seqlen_q=1,
        max_seqlen_k=attn_metadata.max_seq_len,
        softmax_scale=impl.scale,
        causal=True,
        out=output,
        block_table=bt,
        seqused_k=used,
        k_descale=k_scale,
        v_descale=v_scale,
        s_aux=impl.sinks,
    )
    output.masked_fill_(is_pad[:, None, None], 0)


def fa_int8_forward(
    impl,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
    num_actual_tokens: int,
    window: tuple[int, int] | None,
) -> torch.Tensor:
    """B70-IP-D2 forward for one FLASH_ATTN layer over an int8 per-token-head
    cache. ``window`` is the impl's window after _maybe_symmetrize_window."""
    from vllm._xpu_ops import xpu_ops

    causal = attn_metadata.causal
    if (
        not isinstance(causal, bool)
        or getattr(attn_metadata, "mm_prefix_query_range_tensor", None) is not None
        or getattr(attn_metadata, "rswa_prefix_lens", None) is not None
    ):
        raise NotImplementedError(
            "B70-IP-D2 int8_per_token_head FLASH_ATTN: per-request causal, "
            "mm_prefix and R-SWA are not supported"
        )
    # The prefill scratch is allocated in the first forward (the profile run),
    # so the KV pool is sized around it.
    ensure_scratch(impl, kv_cache, query.dtype, force=True)
    n = num_actual_tokens
    q, out = query[:n], output[:n]
    key_cache, value_cache, k_scale, v_scale = pth_split_views(impl, kv_cache)
    window_size = list(window) if window is not None and window[0] >= 0 else None
    max_q = attn_metadata.max_query_len

    if max_q <= spec_decode_max_qlen() and (causal or max_q == 1):
        # q 1: the INT8 paged decode. 1 < q <= bound, causal: the kernels'
        # spec-decode route (one decode row per (request, position), exact
        # per-position causal lengths; ragged q lengths included).
        xpu_ops.flash_attn_varlen_func(
            q,
            key_cache,
            value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=max_q,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=impl.scale,
            causal=True,
            out=out,
            block_table=attn_metadata.block_table,
            seqused_k=attn_metadata.seq_lens,
            window_size=window_size,
            k_descale=k_scale,
            v_descale=v_scale,
            s_aux=impl.sinks,
        )
        return output
    if max_q <= spec_decode_max_qlen() and window_size is None:
        _noncausal_rows_decode(
            q, key_cache, value_cache, k_scale, v_scale, out, attn_metadata, impl
        )
        return output

    # Prefill or mixed batch: dequantize the referenced blocks into the K4
    # scratch and run FA bf16 (the INT8 kernel has no prefill path).
    if _capturing():
        raise RuntimeError(
            f"B70-IP-D2: a batch with max q {max_q} was captured into a full "
            "graph; the int8 prefill path syncs with the host. Use "
            "FULL_AND_PIECEWISE (full graphs for uniform decode only)."
        )
    if not maybe_fa_prefill(
        impl,
        q,
        key_cache,
        value_cache,
        out,
        attn_metadata,
        n,
        pth_scales=(k_scale, v_scale),
        force=True,
        window_size=window_size,
    ):
        raise NotImplementedError(
            "B70-IP-D2 int8_per_token_head FLASH_ATTN: this batch is neither a "
            "decode nor a dequantizable prefill (ALiBi / soft-cap / mm_prefix)"
        )
    return output

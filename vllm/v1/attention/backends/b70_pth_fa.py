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
"""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_scratch: dict = {}
_logged = False


def _enabled() -> bool:
    return current_platform.is_xpu() and os.environ.get("B70_PTH_FA_PREFILL", "1") != "0"


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


def ensure_scratch(impl, kv_cache: torch.Tensor, dtype) -> None:
    """Allocate at the first forward (profile run) so KV sizing sees it."""
    if not _enabled():
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
) -> bool:
    """int8 per-token-head (default) or, with fp8_scales=(k_scale, v_scale),
    an fp8 per-tensor cache (B70-IP-K5, FLASH_ATTN backend)."""
    global _logged
    if not _enabled():
        return False
    if attn_metadata.max_query_len <= int(os.environ.get("B70_PTH_FA_MIN_Q", "64")):
        return False
    if (
        impl.alibi_slopes is not None
        or getattr(impl, "sinks", None) is not None
        or (getattr(impl, "logits_soft_cap", 0) or 0) != 0
        or tuple(impl.sliding_window or (-1, -1)) != (-1, -1)
        or getattr(attn_metadata, "mm_prefix_range_tensor", None) is not None
        or getattr(attn_metadata, "mm_prefix_query_range_tensor", None) is not None
        or getattr(attn_metadata, "rswa_prefix_lens", None) is not None
        or not isinstance(attn_metadata.causal, bool)
        or not attn_metadata.causal
    ):
        return False
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
        if j == i:  # one sequence longer than the scratch: cannot happen (max_model_len)
            return False
        ids = torch.cat([bt[r, : nblks[r]] for r in range(i, j)]).to(torch.long)
        if fp8_scales is None:
            _dequant_blocks(key_cache, impl._k_scale_cache, ids, ks, hs)
            _dequant_blocks(value_cache, impl._v_scale_cache, ids, vs, hs)
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
            causal=True,
            out=output[q0:q1],
            block_table=new_bt,
            seqused_k=sk,
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int8 per-token-head main KV cache for the Qwen4Exp QSA layers, on CPU.

The first group needs no Triton: dtype acceptance, the cache spec (TRITON_ATTN's
inline scale packing, also as the hybrid block alignment sees it), the K/V and
scale views (TRITON_ATTN's byte offsets, nothing written) and the dequant math of
the kernels (K scale on the scores, V scale on P) against a dense reference.

The second group runs the Triton kernels on CPU tensors in the Triton
interpreter, in a child process with ``TRITON_INTERPRET=1`` (Triton reads it at
import): the per-token-head writer feeding the XPU attn2 kernel and split-K, the
BF16 launches unchanged, and the fused prepare's int8 writer against the writer
formula. Each layout runs twice: the tight cell (``2 * (head_dim + 4)`` bytes)
and the cell whose halves are rounded up to 16 bytes.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_indexer import QSAIndexerBackend
from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  (before qsa)
from vllm.models.qwen4_exp.nvidia import qsa as qsa_mod
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode, get_kv_quant_mode

HD = 256
PTH = "int8_per_token_head"
# Bytes of one K (or V) half of a cell: tight, and rounded up to 16 bytes.
HALVES = {"tight": HD + 4, "pad16": (HD + 4 + 15) // 16 * 16}


def _spec(dtype: torch.dtype, mode: str, nkv: int, block: int = 1):
    return FullAttentionSpec(
        block_size=block,
        num_kv_heads=nkv,
        head_size=HD,
        head_size_v=HD,
        dtype=dtype,
        kv_quant_mode=get_kv_quant_mode(mode),
    )


def _triton_half() -> int:
    """The half TRITON_ATTN packs in this tree."""
    spec = TritonAttentionBackend.customize_spec(_spec(torch.int8, PTH, 1))
    assert spec.state_content_bytes is not None
    return spec.state_content_bytes // 2


# --------------------------------------------------------------------------
# No Triton
# --------------------------------------------------------------------------


def test_qsa_accepts_int8_per_token_head() -> None:
    backend = qsa_mod.Qwen4ExpQSABackend
    for dtype in ("auto", "bfloat16", "fp8", "fp8_e4m3", PTH):
        assert backend.supports_kv_cache_dtype(dtype)
    assert not backend.supports_kv_cache_dtype("fp8_per_token_head")
    assert not backend.supports_kv_cache_dtype("int4_per_token_head")
    impl = qsa_mod.Qwen4ExpQSAImpl(12, HD, HD**-0.5, 1, PTH)
    assert impl.is_int8_pth
    assert not qsa_mod.Qwen4ExpQSAImpl(12, HD, HD**-0.5, 1, "fp8").is_int8_pth
    with pytest.raises(NotImplementedError):
        qsa_mod.Qwen4ExpQSAImpl(12, HD, HD**-0.5, 1, "fp8_per_token_head")


@pytest.mark.parametrize("nkv", [1, 2])
@pytest.mark.parametrize("block", [1, 16, 3328])
def test_qsa_spec_packs_inline_scales_like_triton(nkv: int, block: int) -> None:
    spec = _spec(torch.int8, PTH, nkv, block)
    qsa = qsa_mod.Qwen4ExpQSABackend.customize_spec(spec)
    assert qsa == TritonAttentionBackend.customize_spec(spec)
    assert qsa.state_content_bytes == 2 * _triton_half()
    assert _triton_half() in HALVES.values()
    assert qsa.page_size_bytes == block * nkv * qsa.state_content_bytes
    # The hybrid block alignment asks the first non-SSM backend, which in
    # Qwen4Exp is the indexer's: it must answer the same page.
    assert QSAIndexerBackend.customize_spec(spec) == qsa


@pytest.mark.parametrize("dtype,mode", [(torch.bfloat16, "auto"), (torch.uint8, "fp8")])
def test_qsa_spec_unchanged_for_bf16_and_fp8(dtype, mode) -> None:
    spec = _spec(dtype, mode, 2, 16)
    assert qsa_mod.Qwen4ExpQSABackend.customize_spec(spec) == spec
    assert QSAIndexerBackend.customize_spec(spec) is spec


def _pth_cache(nb: int, nkv: int, bs: int, layout: str, half: int) -> torch.Tensor:
    """Logical (B, H, N, C) int8 cache, HND or NHD physical, or inside a padded
    page (hybrid unify) at a nonzero storage offset."""
    c = 2 * half
    if layout == "hnd":
        return torch.randint(-128, 128, (nb, nkv, bs, c), dtype=torch.int8)
    if layout == "nhd":
        return torch.randint(-128, 128, (nb, bs, nkv, c), dtype=torch.int8).transpose(
            1, 2
        )
    page = bs * nkv * c + 64  # padded page, 4-byte multiple
    raw = torch.randint(-128, 128, (nb * page + 128,), dtype=torch.int8)
    return torch.as_strided(raw, (nb, nkv, bs, c), (page, c, nkv * c, 1), 128)


@pytest.mark.parametrize("cell", sorted(HALVES))
@pytest.mark.parametrize("layout", ["hnd", "nhd", "padded"])
@pytest.mark.parametrize("nkv", [1, 2])
def test_pth_views_address_the_cell_and_write_nothing(
    cell: str, layout: str, nkv: int
) -> None:
    h = HALVES[cell]
    cache = _pth_cache(5, nkv, 16, layout, h)
    before = cache.clone()
    k, v, ks, vs = qsa_ops.qsa_pth_kv_views(cache, HD)
    assert torch.equal(cache, before)
    assert k.dtype == v.dtype == torch.int8 and k.shape == (5, 16, nkv, HD)
    assert k.stride(-1) == v.stride(-1) == 1
    assert ks.shape == vs.shape == (5, 16, nkv) and ks.dtype == torch.float32
    if h == _triton_half():
        # Same scale views as TRITON_ATTN (byte offsets and strides).
        ref = TritonAttentionImpl.__new__(TritonAttentionImpl)
        ref._k_scale_cache = ref._v_scale_cache = None
        ref.head_size = HD
        ref._kv_quant_mode = KVQuantMode.INT8_PER_TOKEN_HEAD
        ref._set_kv_ptr_range = False
        ref._ensure_scale_caches(cache)
        for mine, theirs in ((ks, ref._k_scale_cache), (vs, ref._v_scale_cache)):
            assert mine.stride() == theirs.stride()
            assert mine.storage_offset() == theirs.storage_offset()
            assert torch.equal(mine.view(torch.int32), theirs.view(torch.int32))
    # K(hd) | ks | pad, then V(hd) | vs | pad.
    ks[3, 7, nkv - 1] = 2.5
    vs[1, 2, 0] = -0.75
    cell_bytes = cache[3, nkv - 1, 7]
    assert cell_bytes[HD : HD + 4].contiguous().view(torch.float32).item() == 2.5
    assert torch.equal(k[3, 7, nkv - 1], cell_bytes[:HD])
    cell_bytes = cache[1, 0, 2]
    scale = cell_bytes[h + HD : h + HD + 4].contiguous().view(torch.float32).item()
    assert scale == -0.75
    assert torch.equal(v[1, 2, 0], cell_bytes[h : h + HD])
    assert v.data_ptr() - k.data_ptr() == h


@pytest.mark.parametrize("content", [2 * HD, 2 * (HD + 4) + 2, 2 * (HD + 24)])
def test_pth_views_reject_other_content(content: int) -> None:
    with pytest.raises(ValueError):
        qsa_ops.qsa_pth_kv_views(torch.zeros(2, 1, 16, content, dtype=torch.int8), HD)
    with pytest.raises(ValueError):
        qsa_ops.qsa_pth_kv_views(torch.zeros(2, 1, 16, 2 * HD + 8), HD)


def test_impl_caches_views_and_follows_the_runner_reset() -> None:
    impl = qsa_mod.Qwen4ExpQSAImpl(12, HD, HD**-0.5, 1, PTH)
    cache = _pth_cache(3, 1, 16, "nhd", _triton_half())
    first = impl.pth_kv_views(cache)
    assert impl.pth_kv_views(cache)[2] is first[2]  # cached per bound cache
    impl._k_scale_cache = None  # the runner's post-profiling reset
    assert impl.pth_kv_views(cache)[2] is not first[2]
    other = _pth_cache(3, 1, 16, "hnd", _triton_half())
    assert impl.pth_kv_views(other)[0].data_ptr() == other.data_ptr()


def quantize_pth(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch replica of TRITON_ATTN's int8 per-token-head writer: per row of the
    last dim, scale = max(absmax / 127, 1e-6), x * (1 / scale), round half away
    from zero, clamp, truncate."""
    xf = x.float()
    scale = torch.clamp(xf.abs().amax(-1) / 127.0, min=1e-6)
    q = xf * (1.0 / scale)[..., None]
    q = torch.where(q >= 0, q + 0.5, q - 0.5).clamp(-128.0, 127.0)
    return q.trunc().to(torch.int8), scale


def dense_reference(q, k, v, scale) -> torch.Tensor:
    """fp32 softmax(q k^T * scale) v with GQA over one row of selected tokens."""
    rep = q.shape[0] // k.shape[1]
    kk = k.float().repeat_interleave(rep, dim=1)
    vv = v.float().repeat_interleave(rep, dim=1)
    s = torch.einsum("hd,khd->hk", q.float(), kk) * scale
    return torch.einsum("hk,khd->hd", torch.softmax(s, -1), vv)


def kernel_math_emulation(q, k_i8, ks, v_i8, vs, scale) -> torch.Tensor:
    """What the QSA kernels compute: BF16 dots over the exact int8 values, the K
    scale on the FP32 scores, the V scale on P (fp32, then BF16) before P.V."""
    rep = q.shape[0] // k_i8.shape[1]
    kb = k_i8.to(torch.bfloat16).float().repeat_interleave(rep, dim=1)
    vb = v_i8.to(torch.bfloat16).float().repeat_interleave(rep, dim=1)
    ksr = ks.repeat_interleave(rep, dim=1).T
    vsr = vs.repeat_interleave(rep, dim=1).T
    s = torch.einsum("hd,khd->hk", q.float(), kb) * ksr * scale
    pv = (torch.softmax(s, -1) * vsr).to(torch.bfloat16).float()
    return torch.einsum("hk,khd->hd", pv, vb)


def _metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a, b = a.float().flatten(), b.float().flatten()
    rel = ((a - b).norm() / b.norm()).item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return rel, cos


@pytest.mark.parametrize("hq,nkv", [(24, 2), (12, 1)])
@pytest.mark.parametrize("ntok", [17, 2051])
def test_dequant_math_matches_reference(hq: int, nkv: int, ntok: int) -> None:
    torch.manual_seed(ntok + hq)
    q = torch.randn(hq, HD, dtype=torch.bfloat16)
    # Per-token magnitudes vary (outlier tokens and heads), as real K/V do.
    k = (torch.randn(ntok, nkv, HD) * torch.rand(ntok, nkv, 1) * 4).to(torch.bfloat16)
    v = (torch.randn(ntok, nkv, HD) * torch.rand(ntok, nkv, 1) * 2).to(torch.bfloat16)
    k_i8, ks = quantize_pth(k)
    v_i8, vs = quantize_pth(v)
    scale = HD**-0.5
    emu = kernel_math_emulation(q, k_i8, ks, v_i8, vs, scale)
    deq = dense_reference(
        q, k_i8.float() * ks[..., None], v_i8.float() * vs[..., None], scale
    )
    rel_emu, cos_emu = _metrics(emu, deq)
    rel_q, cos_q = _metrics(deq, dense_reference(q, k, v, scale))
    # The scale placement is exact algebra; only P's BF16 rounding remains.
    assert rel_emu < 4e-3 and cos_emu > 0.99999
    # The int8 per-token-head storage floor against the BF16 cache.
    assert rel_q < 3e-2 and cos_q > 0.999


# --------------------------------------------------------------------------
# Triton kernels in the interpreter (child process)
# --------------------------------------------------------------------------


def _patch_interpreter() -> None:
    """Test harness only. The Triton interpreter keeps BF16 as raw uint16, so it
    computes tl.dot on BF16 operands and int<->BF16 casts on the bit patterns;
    route both through fp32, which it converts correctly. It also builds a
    scalar & vector mask as an integer array and then refuses it in its own
    masked load and store. Compiled kernels are not affected."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter

    builder = interpreter.InterpreterBuilder
    cast_impl, create_dot = builder.cast_impl, builder.create_dot
    load, store = builder.create_masked_load, builder.create_masked_store

    def cast(self, src, dst_type):
        s, d = src.dtype.scalar, dst_type.scalar
        if (s == tl.bfloat16) != (d == tl.bfloat16) and tl.float32 not in (s, d):
            return cast_impl(self, cast_impl(self, src, tl.float32), dst_type)
        return cast_impl(self, src, dst_type)

    def to_f32(handle):
        if handle.dtype.scalar != tl.bfloat16:
            return handle
        data = interpreter._convert_float(handle.data, tl.bfloat16, tl.float32, None)
        return interpreter.TensorHandle(data.view(np.float32), tl.float32)

    def dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        return create_dot(
            self, to_f32(a), to_f32(b), d, input_precision, max_num_imprecise_acc
        )

    def masked_load(self, ptrs, mask, *args):
        mask = interpreter.TensorHandle(mask.data != 0, tl.int1)
        return load(self, ptrs, mask, *args)

    def masked_store(self, ptrs, value, mask, *args):
        mask = interpreter.TensorHandle(mask.data != 0, tl.int1)
        return store(self, ptrs, value, mask, *args)

    builder.cast_impl = cast  # type: ignore[method-assign]
    builder.create_dot = dot  # type: ignore[method-assign]
    builder.create_masked_load = masked_load  # type: ignore[method-assign]
    builder.create_masked_store = masked_store  # type: ignore[method-assign]


def _paged_case(hq: int, nkv: int, half: int, rows: int, width: int, seed: int):
    torch.manual_seed(seed)
    nb, page = 6, 16
    tokens = nb * page - 3
    key = (torch.randn(tokens, nkv, HD) * torch.rand(tokens, nkv, 1) * 3).to(
        torch.bfloat16
    )
    value = (torch.randn(tokens, nkv, HD) * torch.rand(tokens, nkv, 1)).to(
        torch.bfloat16
    )
    block_table = torch.randperm(nb)[None, :].to(torch.int32).repeat(2, 1)
    slot_mapping = (
        block_table[0].long()[torch.arange(tokens) // page] * page
        + torch.arange(tokens) % page
    )
    bf16_cache = torch.zeros(nb, page, nkv, 2 * HD, dtype=torch.bfloat16).transpose(
        1, 2
    )
    kc, vc = bf16_cache.transpose(1, 2).split(HD, dim=-1)
    kc.reshape(-1, nkv, HD)[slot_mapping] = key
    vc.reshape(-1, nkv, HD)[slot_mapping] = value
    int8_cache = torch.zeros(nb, page, nkv, 2 * half, dtype=torch.int8).transpose(1, 2)
    indices = torch.full((rows, width + 1), -1, dtype=torch.int32)
    for r in range(rows):
        count = min(width, tokens - r)
        indices[r, :count] = torch.randperm(tokens)[:count].sort().values
        indices[r, width] = count
    return dict(
        key=key,
        value=value,
        slot_mapping=slot_mapping,
        bf16_cache=bf16_cache,
        int8_cache=int8_cache,
        q=torch.randn(rows, hq, HD, dtype=torch.bfloat16),
        indices=indices,
        block_table=block_table,
        token_to_req=torch.zeros(rows, dtype=torch.int32),
        gate=torch.full((rows, hq * HD), 30.0, dtype=torch.bfloat16),  # sigmoid ~ 1
    )


def _write_int8(case) -> tuple:
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    k, v, ks, vs = qsa_ops.qsa_pth_kv_views(case["int8_cache"], HD)
    triton_reshape_and_cache_flash_per_token_head_quant(
        case["key"],
        case["value"],
        k,
        v,
        ks,
        vs,
        case["slot_mapping"],
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    )
    return k, v, ks, vs


def _run(case, impl: str, int8: bool, int8_dot: int = 0) -> torch.Tensor:
    from vllm.models.qwen4_exp.xpu.ops import qsa_attn2
    from vllm.platforms import current_platform

    # grf_mode is a launch option of the XPU backend only.
    qsa_attn2._GRF_MODE = {}
    current_platform.is_xpu = lambda: impl == "attn2"  # type: ignore[method-assign]
    # The split-K merge does not run in the interpreter; one split stores
    # straight from the split-K kernel (the same int8 code).
    select = qsa_ops._select_config
    qsa_ops._select_config = lambda *a, **kw: (*select(*a, **kw)[:3], 1)
    out = torch.zeros_like(case["q"])
    if int8:
        k, v, ks, vs = qsa_ops.qsa_pth_kv_views(case["int8_cache"], HD)
        extra = dict(k_scale_cache=ks, v_scale_cache=vs)
    else:
        k, v = case["bf16_cache"].transpose(1, 2).split(HD, dim=-1)
        extra = {}
    try:
        qsa_ops.qsa_sparse_paged_attention(
            case["q"],
            k,
            v,
            case["indices"],
            case["block_table"],
            case["token_to_req"],
            False,
            out,
            output_gate=case["gate"],
            int8_dot=int8_dot,
            **extra,
        )
    finally:
        qsa_ops._select_config = select
    return out


def _reference_rows(case, k_rows, v_rows) -> torch.Tensor:
    """Dense per-row reference over token-indexed K/V [tokens, nkv, hd]."""
    out = []
    for r in range(case["q"].shape[0]):
        sel = case["indices"][r, : case["indices"][r, -1]].long()
        o = dense_reference(case["q"][r], k_rows[sel], v_rows[sel], HD**-0.5)
        out.append(o.to(torch.bfloat16).float() * torch.sigmoid(torch.tensor(30.0)))
    return torch.stack(out)


def _child_kernels(impl: str, hq: int, nkv: int, cell: str) -> None:
    _patch_interpreter()
    case = _paged_case(hq, nkv, HALVES[cell], rows=3, width=67, seed=hq + nkv)
    k, v, ks, vs = _write_int8(case)
    k_ref, ks_ref = quantize_pth(case["key"])
    v_ref, vs_ref = quantize_pth(case["value"])
    sm = case["slot_mapping"]
    assert torch.equal(k.reshape(-1, nkv, HD)[sm], k_ref)
    assert torch.equal(v.reshape(-1, nkv, HD)[sm], v_ref)
    assert torch.equal(ks.reshape(-1, nkv)[sm], ks_ref)
    assert torch.equal(vs.reshape(-1, nkv)[sm], vs_ref)

    out8 = _run(case, impl, True)
    out16 = _run(case, impl, False)
    deq = _reference_rows(
        case, k_ref.float() * ks_ref[..., None], v_ref.float() * vs_ref[..., None]
    )
    rel_k, cos_k = _metrics(out8, deq.view_as(out8))
    rel_q, cos_q = _metrics(out8, out16)
    rel_b, cos_b = _metrics(
        out16, _reference_rows(case, case["key"], case["value"]).view_as(out16)
    )
    print(
        f"{impl} hq={hq} nkv={nkv} {cell}: int8 vs dequant ref rel={rel_k:.2e} "
        f"cos={cos_k:.6f}; int8 vs bf16 cache rel={rel_q:.2e} cos={cos_q:.6f}; "
        f"bf16 vs ref rel={rel_b:.2e}"
    )
    assert rel_k < 1e-2 and cos_k > 0.9999
    assert rel_q < 3e-2 and cos_q > 0.999
    assert rel_b < 1e-2 and cos_b > 0.9999


def _patch_interpreter_int8_dot() -> None:
    """Test harness only: the interpreter multiplies int8 operands of tl.dot in
    int8; widen them to int32, the result type of the compiled dot."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter

    builder = interpreter.InterpreterBuilder
    create_dot = builder.create_dot

    def widen(handle):
        if handle.dtype.scalar != tl.int8:
            return handle
        return interpreter.TensorHandle(handle.data.astype(np.int32), tl.int32)

    def dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        return create_dot(
            self, widen(a), widen(b), d, input_precision, max_num_imprecise_acc
        )

    builder.create_dot = dot  # type: ignore[method-assign]


def _child_int8_dot(mode: int, hq: int, nkv: int, cell: str) -> None:
    _patch_interpreter()
    _patch_interpreter_int8_dot()
    from vllm.models.qwen4_exp.xpu.ops import qsa_attn2

    case = _paged_case(hq, nkv, HALVES[cell], rows=3, width=67, seed=mode + hq + nkv)
    k, v, _, _ = _write_int8(case)
    # The query quantization follows the KV writer's rounding.
    q_i8, q_scale = qsa_attn2.quantize_q_int8(case["q"])
    ref_i8, ref_scale = quantize_pth(case["q"])
    assert torch.equal(q_i8, ref_i8) and torch.equal(q_scale, ref_scale)
    gather = qsa_attn2.int8_dot_gather_kwargs(k, v, HD)
    assert gather["ALIGN8"] and gather["V_OFF"] == HALVES[cell]
    assert gather["KV_ALIGN"] == (16 if HALVES[cell] % 16 == 0 else 8)

    widened = _run(case, "attn2", True)
    out = _run(case, "attn2", True, int8_dot=mode)
    rel, cos = _metrics(out, widened)
    print(f"int8 dot mode {mode} hq={hq} nkv={nkv} {cell}: rel={rel:.2e} cos={cos:.6f}")
    assert rel < 3e-2 and cos > 0.999


def _child_fused_prepare(nkv: int, cell: str) -> None:
    _patch_interpreter()
    from vllm.models.qwen4_exp.nvidia.ops.qsa_prepare import qsa_prepare

    torch.manual_seed(nkv)
    T, HQI, DI, MHQ, page, nb = 9, 4, 128, 6 * nkv, 16, 3
    q = torch.randn(T, HQI * DI, dtype=torch.bfloat16)
    k = torch.randn(T, DI, dtype=torch.bfloat16)
    positions = torch.arange(T, dtype=torch.int64) + 5
    cos_sin = torch.randn(64, DI // 2, dtype=torch.float32)
    main_qkv = (torch.randn(T, 2 * (MHQ + nkv) * HD) * 2).to(torch.bfloat16)
    w = torch.randn(2, HD, dtype=torch.bfloat16) * 0.2
    slots = torch.randperm(nb * page)[:T]
    slots[-1] = -1

    def run(cache, pth):
        return qsa_prepare(
            q,
            k,
            positions,
            cos_sin,
            torch.zeros(DI, dtype=torch.bfloat16),
            torch.zeros(DI, dtype=torch.bfloat16),
            1e-6,
            torch.empty(T, HQI, DI, dtype=torch.bfloat16),
            torch.zeros(2, 4, DI, dtype=torch.bfloat16),
            torch.full((T,), -1, dtype=torch.int64),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([0, T], dtype=torch.int32),
            positions.clone(),
            torch.zeros(2, 4, DI, dtype=torch.bfloat16),
            torch.full((T,), -1, dtype=torch.int64),
            torch.zeros(0, 2, dtype=torch.int32),
            compress_ratio=4,
            mrope_section=None,
            rope_pos_offset=None,
            main_qkv=main_qkv,
            main_q_norm_weight=w[0],
            main_k_norm_weight=w[1],
            main_eps=1e-6,
            main_kv_cache=cache,
            main_slot_mapping=slots,
            main_k_scale=1.0,
            main_v_scale=1.0,
            main_kv_pth=pth,
        )

    h = HALVES[cell]
    bf16 = torch.zeros(nb, nkv, page, 2 * HD, dtype=torch.bfloat16).transpose(1, 2)
    i8 = torch.zeros(nb, nkv, page, 2 * h, dtype=torch.int8).transpose(1, 2)
    q16, g16 = run(bf16, False)
    q8, g8 = run(i8, True)
    assert torch.equal(q16, q8) and torch.equal(g16, g8)

    live = slots[slots >= 0]
    k16 = bf16.reshape(-1, nkv, 2 * HD)[live, :, :HD]
    v16 = bf16.reshape(-1, nkv, 2 * HD)[live, :, HD:]
    cells = i8.reshape(-1, nkv, 2 * h)[live]
    k_exp, ks_exp = quantize_pth(k16)
    v_exp, vs_exp = quantize_pth(v16)
    assert torch.equal(cells[..., :HD], k_exp)
    assert torch.equal(cells[..., h : h + HD], v_exp)
    ks = cells[..., HD : HD + 4].contiguous().view(torch.float32).squeeze(-1)
    vs = cells[..., h + HD : h + HD + 4].contiguous().view(torch.float32).squeeze(-1)
    assert torch.equal(ks, ks_exp) and torch.equal(vs, vs_exp)
    # The pad bytes after each scale are never written.
    assert not cells[..., HD + 4 : h].any() and not cells[..., h + HD + 4 :].any()
    # Unwritten slots stay zero (slot -1 is skipped).
    written = torch.zeros(nb * page, dtype=torch.bool)
    written[live] = True
    assert not i8.reshape(-1, nkv, 2 * h)[~written].any()


def _child_unfused_update(cell: str) -> None:
    _patch_interpreter()
    impl = qsa_mod.Qwen4ExpQSAImpl(12, HD, HD**-0.5, 1, PTH)
    h = HALVES[cell]
    cache = torch.zeros(2, 16, 1, 2 * h, dtype=torch.int8).transpose(1, 2)
    key = torch.full((8, 1, HD), float("nan"), dtype=torch.bfloat16)
    value = key.clone()
    key[:5] = torch.randn(5, 1, HD, dtype=torch.bfloat16)
    value[:5] = torch.randn(5, 1, HD, dtype=torch.bfloat16)
    slots = torch.tensor([3, 17, -1, 0, 30])
    impl.do_kv_cache_update(None, key, value, cache, slots)
    k, v, ks, vs = impl.pth_kv_views(cache)
    live = slots >= 0
    k_exp, ks_exp = quantize_pth(key[:5][live])
    v_exp, vs_exp = quantize_pth(value[:5][live])
    assert torch.equal(k.reshape(-1, 1, HD)[slots[live]], k_exp)
    assert torch.equal(v.reshape(-1, 1, HD)[slots[live]], v_exp)
    assert torch.equal(ks.reshape(-1, 1)[slots[live]], ks_exp)
    assert torch.equal(vs.reshape(-1, 1)[slots[live]], vs_exp)
    written = torch.zeros(32, dtype=torch.bool)
    written[slots[live]] = True
    flat = cache.transpose(1, 2).reshape(32, -1)
    assert not flat[~written].any()
    # The writer leaves the pad bytes of written cells alone.
    assert not flat[:, HD + 4 : h].any() and not flat[:, h + HD + 4 :].any()


def _in_interpreter(call: str) -> None:
    root = Path(__file__).parents[3]
    code = f"from tests.models.qwen4_exp import test_qsa_int8_kv as t\nt.{call}\n"
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=root,
        env={**os.environ, "TRITON_INTERPRET": "1"},
        timeout=900,
    )


requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")


@requires_triton
@pytest.mark.parametrize("cell", sorted(HALVES))
@pytest.mark.parametrize("hq,nkv", [(24, 2), (12, 1)])
@pytest.mark.parametrize("impl", ["attn2", "splitk"])
def test_kernels_read_int8_per_token_head(impl, hq, nkv, cell) -> None:
    _in_interpreter(f"_child_kernels({impl!r}, {hq}, {nkv}, {cell!r})")


@requires_triton
@pytest.mark.parametrize("cell", sorted(HALVES))
@pytest.mark.parametrize("nkv", [1, 2])
def test_fused_prepare_writes_the_writer_formula(nkv, cell) -> None:
    _in_interpreter(f"_child_fused_prepare({nkv}, {cell!r})")


@requires_triton
@pytest.mark.parametrize("cell", sorted(HALVES))
def test_unfused_update_writes_only_mapped_slots(cell) -> None:
    _in_interpreter(f"_child_unfused_update({cell!r})")


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("hq,nkv", [(24, 2), (12, 1)])
@pytest.mark.parametrize("cell", sorted(HALVES))
def test_attn2_int8_dot_matches_the_widened_path(mode, hq, nkv, cell) -> None:
    _in_interpreter(f"_child_int8_dot({mode}, {hq}, {nkv}, {cell!r})")


def test_int8_dot_launches_and_gathers() -> None:
    from vllm.models.qwen4_exp.xpu.ops import qsa_attn2

    assert qsa_attn2._select_launch(4096, True) is qsa_attn2._PREFILL
    assert qsa_attn2._select_launch(64, True) is qsa_attn2._DECODE
    for mode in (1, 2):
        prefill = qsa_attn2._select_launch(4096, True, mode)
        decode = qsa_attn2._select_launch(4096, False, mode)
        assert prefill is qsa_attn2._INT8_DOT_PREFILL[mode]
        assert decode is qsa_attn2._INT8_DOT_DECODE[mode]
        # An int8 DPAS dot consumes the head dim in steps of 32.
        assert HD // prefill.num_dchunk >= 32 and HD // decode.num_dchunk >= 32
    # Views that are not the inline cell keep the plain gathers.
    k = torch.zeros(2, 16, 1, HD, dtype=torch.int8)
    assert qsa_attn2.int8_dot_gather_kwargs(k, k.clone(), HD) == {"ALIGN8": False}

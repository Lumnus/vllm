# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the native INT8 prefill attention over an int8_per_token_head
KV cache (`vllm.v1.attention.ops.xpu_int8_prefill_attention`) and its routing
in the XPU FlashAttention backend (`AttentionConfig.int8_prefill_dot`).

The kernel tests run in a child process under TRITON_INTERPRET=1 (Triton reads
it at import) against three references computed here in float64:

1. quantized domain: the int8 query codes times their scales against the cached
   K/V codes times theirs (what the kernel computes before its own P rounding);
2. decode contract: the BF16 query against the cached K/V codes times their
   scales (what the native decode kernel computes on the same cache);
3. the dequantize-then-BF16 path the kernel replaces: the BF16 query against
   BF16(code * scale) K/V, which the existing prefill scratch holds.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config.attention import AttentionConfig
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.ops import xpu_int8_prefill_attention as i8p

# PAGE: the routing stubs' block size; the kernel cases take theirs.
HD, PAGE = 256, 64
# (seq_len, q_len): a chunk after cached context, a fresh prompt, a decode row
# of a mixed batch, a short chunk over a partly filled last block.
SEQS = [(150, 70), (33, 33), (90, 1), (61, 5)]


# --------------------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------------------
def quantize_pth(x: torch.Tensor, floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8 with the KV writer's rounding (half away from
    zero) over the last dimension."""
    x = x.float()
    scale = (x.abs().amax(-1) / 127.0).clamp_min(floor)
    y = x * (1.0 / scale)[..., None]
    y = torch.where(y >= 0, torch.floor(y + 0.5), torch.ceil(y - 0.5)).clamp(-127, 127)
    return y.to(torch.int8), scale


def _layer_cache(num_blocks: int, nkv: int, block: int):
    """A [B, H, N, C] int8 layer cache of `block` slots per block in the HND
    layout the runner builds (C = 2 * (HD + 4)), and its split-layout views."""
    from vllm.v1.attention.ops.xpu_int8_per_token_head import split_layout_views

    content = 2 * (HD + 4)
    page = nkv * block * content
    raw = torch.zeros(num_blocks * page, dtype=torch.int8)
    cache = torch.as_strided(
        raw, (num_blocks, nkv, block, content), (page, block * content, content, 1)
    )
    return raw, split_layout_views(cache, nkv, HD)


def _case(nkv: int, seed: int, block: int = 64) -> SimpleNamespace:
    torch.manual_seed(seed)
    hq = 6 * nkv
    nblk = [-(-L // block) for L, _ in SEQS]
    num_blocks = sum(nblk) + 3
    raw, (key, value, k_scale, v_scale) = _layer_cache(num_blocks, nkv, block)
    # Stale bytes everywhere first: garbage codes, NaN / inf scales.
    raw.copy_(torch.randint(-128, 128, raw.shape, dtype=torch.int8))
    k_scale.fill_(float("nan"))
    v_scale.fill_(float("inf"))
    perm = (torch.randperm(num_blocks - 1) + 1).to(torch.int32)
    table = torch.zeros(len(SEQS), max(nblk) + 2, dtype=torch.int32)
    k_rows, v_rows = [], []
    c = 0
    for i, (L, _) in enumerate(SEQS):
        table[i, : nblk[i]] = perm[c : c + nblk[i]]
        c += nblk[i]
        # Keys with a few large channels, values plain; BF16 as the model's.
        k = torch.randn(L, nkv, HD)
        k[..., :4] *= 8.0
        k = k.to(torch.bfloat16)
        v = torch.randn(L, nkv, HD).to(torch.bfloat16)
        kq, ks = quantize_pth(k, 1e-6)
        vq, vs = quantize_pth(v, 1e-6)
        pos = torch.arange(L)
        blk = table[i, pos // block].long()
        key[blk, pos % block] = kq
        value[blk, pos % block] = vq
        k_scale[blk, pos % block] = ks
        v_scale[blk, pos % block] = vs
        k_rows.append((kq, ks))
        v_rows.append((vq, vs))
    q_lens = [q for _, q in SEQS]
    cu = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32)
    # The query as the backend gets it: a [T, hq, HD] view into a wider qkv row.
    qkv = torch.randn(int(cu[-1]), (hq + 2 * nkv) * HD).to(torch.bfloat16)
    query = qkv[:, : hq * HD].view(-1, hq, HD)
    return SimpleNamespace(
        nkv=nkv,
        hq=hq,
        raw=raw,
        key=key,
        value=value,
        k_scale=k_scale,
        v_scale=v_scale,
        table=table,
        cu=cu,
        seq_lens=torch.tensor([L for L, _ in SEQS], dtype=torch.int32),
        query=query,
        k_rows=k_rows,
        v_rows=v_rows,
    )


def _attend(q_rows, k, v, scale) -> torch.Tensor:
    """Causal float64 attention of the last len(q_rows) positions; q [q, hq,
    D], k/v [L, nkv, D]."""
    ql, L = q_rows.shape[0], k.shape[0]
    g = q_rows.shape[1] // k.shape[1]
    k = k.repeat_interleave(g, 1).double()
    v = v.repeat_interleave(g, 1).double()
    s = torch.einsum("qhd,khd->hqk", q_rows.double(), k) * scale
    qpos = torch.arange(L - ql, L)
    s = s.masked_fill(~(torch.arange(L)[None, :] <= qpos[:, None])[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v)


def references(case) -> dict[str, torch.Tensor]:
    scale = HD**-0.5
    q_i8, q_s = quantize_pth(case.query, 1e-10)
    out = {"quantized": [], "decode": [], "scratch": []}
    for i, _ in enumerate(SEQS):
        q0, q1 = int(case.cu[i]), int(case.cu[i + 1])
        (kq, ks), (vq, vs) = case.k_rows[i], case.v_rows[i]
        k_deq = kq.double() * ks.double()[..., None]
        v_deq = vq.double() * vs.double()[..., None]
        q_deq = q_i8[q0:q1].double() * q_s[q0:q1].double()[..., None]
        out["quantized"].append(_attend(q_deq, k_deq, v_deq, scale))
        out["decode"].append(_attend(case.query[q0:q1], k_deq, v_deq, scale))
        k_bf = (kq.float() * ks[..., None]).to(torch.bfloat16)
        v_bf = (vq.float() * vs[..., None]).to(torch.bfloat16)
        scratch = _attend(case.query[q0:q1], k_bf, v_bf, scale)
        out["scratch"].append(scratch.to(torch.bfloat16).double())
    return {name: torch.cat(rows) for name, rows in out.items()}


def metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    out, ref = out.double().flatten(), ref.double().flatten()
    return {
        "rel": float((out - ref).norm() / ref.norm()),
        "max_abs": float((out - ref).abs().max()),
        "cos": float(torch.nn.functional.cosine_similarity(out, ref, dim=0)),
    }


# --------------------------------------------------------------------------------------
# interpreter children
# --------------------------------------------------------------------------------------
def _patch_interpreter() -> None:
    """Test harness only, for three interpreter gaps: it builds a scalar &
    vector mask as an integer array and refuses it in its own masked load and
    store; it multiplies int8 tl.dot operands in int8, where the compiled dot
    yields int32; and it keeps BF16 as raw uint16 bits, which its integer ->
    BF16 cast fills with the integer value, its FP32 -> BF16 cast truncates
    and its dot multiplies as integers."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter

    builder = interpreter.InterpreterBuilder
    load, store = builder.create_masked_load, builder.create_masked_store
    create_dot = builder.create_dot
    cast_impl = builder.cast_impl

    def cast(self, src, dst_type):
        if dst_type.scalar == tl.bfloat16 and (
            src.dtype.scalar.is_int() or src.dtype.scalar == tl.float32
        ):
            # Round to nearest even, as the device does (the interpreter
            # truncates).
            f32 = src.data.astype(np.float32)
            bits = f32.view(np.uint32)
            rne = (
                bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
            ) >> np.uint32(16)
            rne = np.where(np.isnan(f32), np.uint32(0x7FC0), rne).astype(np.uint16)
            return interpreter.TensorHandle(rne, tl.bfloat16)
        return cast_impl(self, src, dst_type)

    def bf16_as_fp32(handle):
        if handle.dtype.scalar != tl.bfloat16:
            return handle
        bits = handle.data.astype(np.uint32) << np.uint32(16)
        return interpreter.TensorHandle(bits.view(np.float32), tl.float32)

    def masked_load(self, ptrs, mask, *args, **kwargs):
        mask = interpreter.TensorHandle(mask.data != 0, tl.int1)
        return load(self, ptrs, mask, *args, **kwargs)

    def masked_store(self, ptrs, value, mask, *args, **kwargs):
        mask = interpreter.TensorHandle(mask.data != 0, tl.int1)
        return store(self, ptrs, value, mask, *args, **kwargs)

    def widen(handle):
        if handle.dtype.scalar != tl.int8:
            return bf16_as_fp32(handle)
        return interpreter.TensorHandle(handle.data.astype(np.int32), tl.int32)

    def dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        return create_dot(
            self, widen(a), widen(b), d, input_precision, max_num_imprecise_acc
        )

    builder.create_masked_load = masked_load  # type: ignore[method-assign]
    builder.create_masked_store = masked_store  # type: ignore[method-assign]
    builder.create_dot = dot  # type: ignore[method-assign]
    builder.cast_impl = cast  # type: ignore[method-assign]
    i8p._GRF_MODE = {}


def _child_kernel(launch: tuple, mode: int, nkv: int, block: int = 64) -> None:
    _patch_interpreter()
    case = _case(nkv, seed=mode + 10 * nkv, block=block)
    out = torch.full_like(case.query, float("nan")).contiguous()
    i8p.int8_prefill_attention(
        case.query,
        case.key,
        case.value,
        case.k_scale,
        case.v_scale,
        out,
        case.cu,
        case.seq_lens,
        case.table,
        HD**-0.5,
        pv_int8=mode == 2,
        launch=i8p.Int8PrefillLaunch(*launch),
    )
    refs = references(case)
    result = {name: metrics(out, ref) for name, ref in refs.items()}
    result["scratch_vs_decode"] = metrics(refs["scratch"], refs["decode"])
    result["quantized_vs_decode"] = metrics(refs["quantized"], refs["decode"])
    result["finite"] = bool(torch.isfinite(out).all())
    print("RESULT " + json.dumps(result))


def _child_quant() -> None:
    _patch_interpreter()
    case = _case(1, seed=3)
    q_i8, q_scale = i8p.quantize_query_int8(case.query)
    ref_i8, ref_scale = quantize_pth(case.query, 1e-10)
    assert torch.equal(q_i8, ref_i8)
    assert torch.equal(q_scale, ref_scale)
    print("RESULT " + json.dumps({"ok": True}))


def _child_warmup() -> None:
    _patch_interpreter()
    case = _case(1, seed=4)
    before = case.raw.clone()
    i8p.warmup_int8_prefill_attention(
        case.key,
        case.value,
        case.k_scale,
        case.v_scale,
        case.table,
        case.hq,
        torch.bfloat16,
        pv_int8=True,
    )
    assert torch.equal(case.raw, before)
    print("RESULT " + json.dumps({"ok": True}))


def _run_child(call: str) -> dict:
    root = Path(__file__).parents[3]
    module = "tests.v1.attention.test_xpu_int8_prefill_attention"
    code = f"import {module} as t\nt.{call}\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    result = json.loads(line[len("RESULT ") :])
    print(call, result)
    return result


needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")

# (block_m, block_n, qk_chunk, dv_chunks, dv_split, num_warps)
LAUNCHES = {
    "default": tuple(i8p._LAUNCH)[:6],
    "split2_chunks2": (16, 32, 32, 2, 2, 4),
    "small_m_wide_qk": (8, 32, 128, 4, 1, 1),
    "one_chunk_split4": (32, 64, 64, 1, 4, 4),
}

# Bounds on synthetic Gaussian activations with outlier key channels (the
# measured values are printed; see the module docstring for the references).
# The kernel against the quantized-domain reference: only its own P rounding
# (BF16 for "qk", int8 per tile row for "qk_pv").
BOUND_QUANTIZED = {1: 5e-3, 2: 2.5e-2}
# Against the decode contract (BF16 query): adds the int8 query rounding.
BOUND_DECODE = {1: 2.5e-2, 2: 3.5e-2}


@needs_triton
@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("launch", sorted(LAUNCHES))
def test_kernel_matches_references(launch, mode):
    result = _run_child(f"_child_kernel({LAUNCHES[launch]!r}, {mode}, 1)")
    assert result["finite"]
    assert result["quantized"]["rel"] < BOUND_QUANTIZED[mode]
    assert result["decode"]["rel"] < BOUND_DECODE[mode]
    assert result["decode"]["cos"] > 0.9995
    # The kernel is no further from the decode contract than the int8 query
    # rounding alone, plus its P rounding.
    assert (
        result["decode"]["rel"]
        < result["quantized_vs_decode"]["rel"] + BOUND_QUANTIZED[mode]
    )


@needs_triton
@pytest.mark.parametrize("mode", [1, 2])
def test_kernel_two_kv_heads_odd_block(mode):
    """Two KV heads, and a cache block of 96 slots: tiles of 32 (the served
    block of 1600 slots takes tiles of 64 the same way)."""
    result = _run_child(f"_child_kernel({LAUNCHES['default']!r}, {mode}, 2, 96)")
    assert result["finite"]
    assert result["quantized"]["rel"] < BOUND_QUANTIZED[mode]
    assert result["decode"]["rel"] < BOUND_DECODE[mode]


def test_tile_block_n():
    launch = i8p.Int8PrefillLaunch(block_n=128)
    assert [i8p.tile_block_n(launch, p) for p in (1600, 96, 256, 48)] == [
        64,
        32,
        128,
        16,
    ]
    with pytest.raises(ValueError, match="cache block 48"):
        i8p.check_launch(launch, HD, True, 48)
    i8p.check_launch(launch, HD, False, 48)


@needs_triton
def test_query_quantization_is_the_kv_writer_rounding():
    _run_child("_child_quant()")


@needs_triton
def test_warmup_reads_and_writes_nothing():
    _run_child("_child_warmup()")


@pytest.mark.parametrize(
    "launch, pv_int8, error",
    [
        ((64, 16, 64, 4, 1, 8), True, "block_n"),
        ((64, 64, 16, 4, 1, 8), False, "qk_chunk"),
        ((64, 64, 64, 3, 1, 8), False, "dv_chunks"),
        ((64, 64, 64, 4, 8, 8), False, "powers of two >= 16"),
        ((48, 64, 64, 4, 1, 8), False, "block_m"),
    ],
)
def test_check_launch_refusals(launch, pv_int8, error):
    with pytest.raises(ValueError, match=error):
        i8p.check_launch(i8p.Int8PrefillLaunch(*launch), HD, pv_int8)


def test_default_launch_is_legal():
    for pv_int8 in (False, True):
        i8p.check_launch(i8p._LAUNCH, HD, pv_int8)


# --------------------------------------------------------------------------------------
# config and backend routing
# --------------------------------------------------------------------------------------
def test_int8_prefill_dot_off_is_not_a_compile_factor():
    assert "int8_prefill_dot" not in AttentionConfig().compile_factors()
    factors = AttentionConfig(int8_prefill_dot="qk_pv").compile_factors()
    assert factors["int8_prefill_dot"] == "qk_pv"


def _impl_stub(
    mode: int, sliding_window=(-1, -1), sinks=None, block=PAGE
) -> SimpleNamespace:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    stub = SimpleNamespace(
        sliding_window=sliding_window,
        sinks=sinks,
        scale=HD**-0.5,
        num_heads=6,
        head_size=HD,
        kv_cache_dtype="int8_per_token_head",
        _int8_prefill_dot=mode,
        _int8_kv_scratch_dtype=torch.bfloat16,
        _int8_kv_block_size=block,
        scratch_claims=0,
    )

    def scratch(block_size):
        stub.scratch_claims += 1
        return None, None

    stub._int8_kv_prefill_scratch = scratch
    views = (
        torch.zeros(2, PAGE, 1, HD, dtype=torch.int8),
        torch.zeros(2, PAGE, 1, HD, dtype=torch.int8),
        torch.zeros(2, PAGE, 1),
        torch.zeros(2, PAGE, 1),
    )
    stub._int8_kv_split_views = lambda kv_cache: views
    stub._init_int8_prefill_dot = lambda cfg: FlashAttentionImpl._init_int8_prefill_dot(
        stub, cfg
    )
    stub._int8_decode_dot = False
    stub._init_int8_decode_dot = lambda cfg: FlashAttentionImpl._init_int8_decode_dot(
        stub, cfg
    )
    return stub


@pytest.mark.parametrize(
    "setting, window, sinks, block, expected",
    [
        ("off", (-1, -1), None, 1600, 0),
        ("qk", (-1, -1), None, 1600, 1),
        ("qk_pv", (-1, -1), None, 1600, 2),
        ("qk_pv", (127, 0), None, 1600, 0),
        ("qk", (-1, -1), torch.zeros(6), 1600, 0),
        # Tiles of 16 slots: too small for an int8 P.V, legal for a BF16 one.
        ("qk_pv", (-1, -1), None, 48, 0),
        ("qk", (-1, -1), None, 48, 1),
    ],
)
def test_init_selects_mode_per_layer(
    setting, window, sinks, block, expected, monkeypatch
):
    from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv

    monkeypatch.setattr(int8_kv, "decode_max_query_len", lambda: 16)
    stub = _impl_stub(-1, window, sinks, block)
    cfg = SimpleNamespace(attention_config=AttentionConfig(int8_prefill_dot=setting))
    stub._init_int8_prefill_dot(cfg)
    assert stub._int8_prefill_dot == expected


def _metadata(max_q: int, causal: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        causal=causal,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        num_actual_tokens=40,
        max_query_len=max_q,
        query_start_loc=torch.tensor([0, 40], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 40], dtype=torch.int32),
        seq_lens=torch.tensor([100], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([100], dtype=torch.int32),
        max_seq_len=100,
        block_table=torch.zeros(1, 8, dtype=torch.int32),
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_prefill_routing(mode, monkeypatch):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv

    calls = []
    monkeypatch.setattr(int8_kv, "decode_max_query_len", lambda: 16)
    monkeypatch.setattr(int8_kv, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(
        int8_kv, "forward_prefill", lambda *a, **kw: calls.append(("scratch", kw))
    )
    monkeypatch.setattr(
        int8_kv, "forward_decode", lambda *a, **kw: calls.append(("decode", kw))
    )
    monkeypatch.setattr(
        i8p,
        "int8_prefill_attention",
        lambda *a, **kw: calls.append(("int8", kw)),
    )
    stub = _impl_stub(mode)
    query = torch.zeros(40, 6, HD, dtype=torch.bfloat16)
    out = torch.zeros_like(query)
    FlashAttentionImpl._forward_int8_per_token_head(
        stub, query, None, _metadata(40), out
    )
    FlashAttentionImpl._forward_int8_per_token_head(
        stub, query, None, _metadata(8), out
    )
    expected = {0: "scratch", 1: "int8", 2: "int8"}[mode]
    assert [c[0] for c in calls] == [expected, "decode"]
    if mode:
        assert calls[0][1] == {"pv_int8": mode == 2}
        with pytest.raises(NotImplementedError, match="causal"):
            FlashAttentionImpl._forward_int8_per_token_head(
                stub, query, None, _metadata(40, causal=False), out
            )


@pytest.mark.parametrize("mode, claims", [(0, 1), (1, 0), (2, 0)])
def test_profile_run_claims_the_scratch_only_without_int8_prefill(mode, claims):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    stub = _impl_stub(mode)
    stub.vllm_flash_attn_version = 2
    out = torch.ones(4, 6 * HD)
    FlashAttentionImpl.forward(stub, None, out, None, None, None, None, out)
    assert stub.scratch_claims == claims
    assert not out.any()

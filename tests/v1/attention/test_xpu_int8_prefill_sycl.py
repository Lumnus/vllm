# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for INT8 prefill attention in the vllm-xpu-kernels chunk-prefill
kernel (`AttentionConfig.int8_prefill_dot` = "sycl_bf16" | "sycl_qk" |
"sycl_qk_pv") and its routing in the XPU FlashAttention backend.

The SYCL kernel cannot run on a CPU. What is checked here:

1. `sycl_model`, a float32 model of the kernel's arithmetic (32-slot KV tiles
   walked in order per query row, S = s32 * q_scale * k_scale or the BF16
   query against the widened K codes times k_scale, the causal mask on
   absolute positions, online softmax in base 2 with the scale folded into
   exp2, P * v_scale after the row sums, then P rounded to BF16 against the V
   codes, or for "sycl_qk_pv" P quantized per (row, tile) by the shared P
   rule and the s32 dot times the tile's P scale), against
   1. the engine's Triton kernel (interpreter), the numerical reference:
      "sycl_qk" against its "qk", "sycl_qk_pv" against its "qk_pv" with KV
      tiles of 32 slots (the same P quantization groups);
   2. A2's model of the paged decode kernel on the same cached tokens
      (`decode_model`): "sycl_qk" against int8_decode_dot, "sycl_bf16"
      against today's decode;
   3. the float64 references (quantized domain, decode contract);
2. the P rule: the Triton kernel's P codes and scales equal the exact rule
   (`p_rule`) bit for bit;
3. config, per-layer selection, routing, profile run and warm-up.

The card check is `tests/flash_attn/test_flash_attn_int8_kv.py -k prefill` in
vllm-xpu-kernels and `benchmarks/kernels/benchmark_xpu_int8_prefill_attention.py
--check` here.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention import test_xpu_int8_decode_dot as a2
from tests.v1.attention import test_xpu_int8_prefill_attention as a1
from vllm.config.attention import AttentionConfig
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.ops import xpu_int8_prefill_attention as i8p

HD = 256
SCALE = HD**-0.5
LOG2E = 1.4426950408889634
TILE = 32  # the chunk-prefill kernel's KV tile (chunk_policy_head256)


# --------------------------------------------------------------------------------------
# the shared P rule, exact reference
# --------------------------------------------------------------------------------------
def p_rule(pv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int8_attn_p_scale / int8_attn_query_code over the last dimension of
    P * v_scale (>= 0): scale = max(row max / 127, 1e-30), inv = 1 / scale
    (both IEEE fp32), codes min((trunc(2 y) + 1) >> 1, 127)."""
    x = pv.float().numpy()
    scale = np.maximum(
        a2._div_rn(x.max(-1), np.float32(127.0)), np.float32(1e-30)
    ).astype(np.float32)
    inv = a2._div_rn(np.float32(1.0), scale)
    y = (x * inv[..., None]).astype(np.float32)
    a2x = np.minimum(y * np.float32(2.0), np.float32(255.0))
    c = np.minimum((a2x.astype(np.int32) + 1) >> 1, 127)
    return torch.from_numpy(c.astype(np.float32)), torch.from_numpy(scale)


# --------------------------------------------------------------------------------------
# a float32 model of the chunk-prefill kernel's arithmetic
# --------------------------------------------------------------------------------------
def sycl_model(
    q_rows: torch.Tensor,
    k_codes: torch.Tensor,
    k_scale: torch.Tensor,
    v_codes: torch.Tensor,
    v_scale: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    """Output of `q_rows` [q, hq, D] (the last q positions of a sequence of
    L = k_codes.shape[0] tokens) as the chunk-prefill kernel computes it with
    int8_prefill = mode (1 BF16 dots, 2 int8 Q.K, 3 int8 Q.K and P.V): each
    query row walks the KV tiles of 32 slots in order with an fp32 online
    softmax; rows do not interact (a sub-group owns whole rows)."""
    ql, hq, _ = q_rows.shape
    L, nkv, _ = k_codes.shape
    g = hq // nkv
    eff = SCALE * LOG2E
    if mode >= 2:
        qc, qs = a2.query_rule(q_rows)
        qmat = qc.to(torch.float64)
    out = torch.empty(ql, hq, HD)
    for i in range(ql):
        seq = L - ql + i + 1  # keys this row sees
        for h in range(hq):
            kv = h // g
            m = -math.inf
            lsum = 0.0
            acc = torch.zeros(HD)
            for t0 in range(0, seq, TILE):
                cols = torch.arange(t0, min(t0 + TILE, seq))
                kc = k_codes[cols, kv]
                if mode >= 2:
                    dot = (kc.double() @ qmat[i, h]).float()  # exact int32
                    sc = dot * qs[i, h] * k_scale[cols, kv]
                else:
                    dot = kc.float() @ q_rows[i, h].float()
                    sc = dot * k_scale[cols, kv]
                m_new = max(m, eff * float(sc.max()))
                p = torch.exp2(sc * eff - m_new)
                resc = math.exp2(m - m_new) if m > -math.inf else 0.0
                lsum = lsum * resc + float(p.sum())
                pv = p * v_scale[cols, kv]
                if mode == 3:
                    codes, ps = p_rule(pv[None])
                    contrib = (codes[0] @ v_codes[cols, kv].float()) * float(ps[0])
                else:
                    contrib = a2._bf16(pv) @ v_codes[cols, kv].float()
                acc = acc * resc + contrib
                m = m_new
            out[i, h] = acc / lsum
    return out


# --------------------------------------------------------------------------------------
# interpreter children
# --------------------------------------------------------------------------------------
def _triton(case, pv_int8: bool, block_n: int) -> torch.Tensor:
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
        SCALE,
        pv_int8=pv_int8,
        launch=i8p.Int8PrefillLaunch(block_m=16, block_n=block_n, num_warps=4),
    )
    return out


def _models(case, mode: int) -> torch.Tensor:
    rows = []
    for i in range(len(case.k_rows)):
        r = slice(int(case.cu[i]), int(case.cu[i + 1]))
        (kq, ks), (vq, vs) = case.k_rows[i], case.v_rows[i]
        rows.append(sycl_model(case.query[r], kq, ks, vq, vs, mode))
    return torch.cat(rows)


def _child_against_triton(seed: int) -> None:
    a1._patch_interpreter()
    case = a1._case(1, seed=seed, block=64)
    m = a1.metrics
    refs = a1.references(case)
    s2, s3, s1 = _models(case, 2), _models(case, 3), _models(case, 1)
    t_qk = _triton(case, False, TILE)
    t_qkpv = _triton(case, True, TILE)
    result = {
        "finite": bool(torch.isfinite(t_qk).all() and torch.isfinite(t_qkpv).all()),
        "sycl_qk_vs_triton_qk": m(a2._bf16(s2), t_qk.float()),
        "sycl_qk_pv_vs_triton_qk_pv": m(a2._bf16(s3), t_qkpv.float()),
        "sycl_qk_vs_quantized": m(a2._bf16(s2), refs["quantized"]),
        "sycl_qk_pv_vs_quantized": m(a2._bf16(s3), refs["quantized"]),
        "sycl_qk_vs_decode_contract": m(a2._bf16(s2), refs["decode"]),
        "sycl_bf16_vs_decode_contract": m(a2._bf16(s1), refs["decode"]),
        "scratch_vs_decode_contract": m(refs["scratch"], refs["decode"]),
    }
    print("RESULT " + json.dumps(result))


def _child_p_rule() -> None:
    """The Triton kernel's P codes: run its P quantization on random tiles
    through a one-tile kernel copy is not possible from outside, so check the
    formula the kernel uses (same Triton ops) in the interpreter."""
    a1._patch_interpreter()
    import triton
    import triton.language as tl

    @triton.jit
    def _p_codes(pv_ptr, c_ptr, s_ptr, N: tl.constexpr):
        row = tl.program_id(0)
        pv = tl.load(pv_ptr + row * N + tl.arange(0, N))
        p_scale = tl.maximum(tl.math.div_rn(tl.max(pv, axis=0), 127.0), 1.0e-30)
        y = pv * tl.math.div_rn(1.0, p_scale)
        c = (tl.minimum(y * 2.0, 255.0).to(tl.int32) + 1) >> 1
        tl.store(c_ptr + row * N + tl.arange(0, N), tl.minimum(c, 127).to(tl.int8))
        tl.store(s_ptr + row, p_scale)

    torch.manual_seed(7)
    pv = torch.rand(256, TILE) ** 3 * torch.rand(256, 1) * 1e-2
    # rows on half-integer boundaries: max 127 -> scale 1, y == pv
    pv[:5] = 0.0
    for r, k in enumerate((0.5, 1.5, 2.5, 63.5, 126.5)):
        pv[r, 0] = 127.0
        bits = torch.tensor([k]).view(torch.int32)
        pv[r, 1:4] = torch.cat([bits, bits - 1, bits + 1]).view(torch.float32)
    pv[5] = 0.0  # all-zero row
    codes = torch.empty(256, TILE, dtype=torch.int8)
    scales = torch.empty(256)
    _p_codes[(256,)](pv, codes, scales, N=TILE)
    ref_c, ref_s = p_rule(pv)
    print(
        "RESULT "
        + json.dumps(
            {
                "code_mismatches": int((codes.float() != ref_c).sum()),
                "scale_mismatches": int((scales != ref_s).sum()),
                "boundary": [ref_c[r, 1:4].tolist() for r in range(5)],
            }
        )
    )


def _child_against_decode(seed: int) -> None:
    a1._patch_interpreter()
    case = a2._dec_case(seed)
    m = a1.metrics
    s2, s1, dec8, dec16 = [], [], [], []
    for i in range(len(a2.DEC_SEQS)):
        r = slice(int(case.cu[i]), int(case.cu[i + 1]))
        (kq, ks), (vq, vs) = case.k_rows[i], case.v_rows[i]
        args = (case.query[r], kq, ks, vq, vs)
        s2.append(sycl_model(*args, mode=2))
        s1.append(sycl_model(*args, mode=1))
        dec8.append(a2.decode_model(*args, int8_dot=True))
        dec16.append(a2.decode_model(*args, int8_dot=False))
    s2, s1 = torch.cat(s2), torch.cat(s1)
    dec8, dec16 = torch.cat(dec8), torch.cat(dec16)
    result = {
        "sycl_qk_vs_decode_int8": m(a2._bf16(s2), a2._bf16(dec8)),
        "sycl_bf16_vs_decode_today": m(a2._bf16(s1), a2._bf16(dec16)),
        "sycl_qk_vs_decode_today": m(a2._bf16(s2), a2._bf16(dec16)),
    }
    print("RESULT " + json.dumps(result))


def _run(call: str) -> dict:
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).parents[3]
    code = f"import tests.v1.attention.test_xpu_int8_prefill_sycl as t\nt.{call}\n"
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

# The BF16 output-rounding floor between two kernels that share every rounding
# of Q, K, V and the int32 dot: A2 measured 0.23-0.26 % (decode vs prefill).
BOUND_FLOOR = 4e-3
# qk_pv: the P codes can differ by one where P * inv sits next to a
# half-integer and the two kernels' fp32 P differ in the last bit.
BOUND_PV = 6e-3


@needs_triton
@pytest.mark.parametrize("seed", [31, 32])
def test_sycl_model_matches_the_triton_reference(seed):
    r = _run(f"_child_against_triton({seed})")
    assert r["finite"]
    assert r["sycl_qk_vs_triton_qk"]["rel"] < BOUND_FLOOR
    assert r["sycl_qk_vs_triton_qk"]["cos"] > 0.99999
    assert r["sycl_qk_pv_vs_triton_qk_pv"]["rel"] < BOUND_PV
    assert r["sycl_qk_vs_quantized"]["rel"] < 5e-3
    assert r["sycl_qk_pv_vs_quantized"]["rel"] < 2.5e-2
    # No int8 query: the decode contract up to the BF16 floor (today's
    # scratch sits there too; it rounds code * scale to BF16 once more).
    assert r["sycl_bf16_vs_decode_contract"]["rel"] < BOUND_FLOOR
    assert r["scratch_vs_decode_contract"]["rel"] < BOUND_FLOOR


@needs_triton
@pytest.mark.parametrize("seed", [21, 22])
def test_sycl_model_matches_the_decode_kernel(seed):
    r = _run(f"_child_against_decode({seed})")
    assert r["sycl_qk_vs_decode_int8"]["rel"] < BOUND_FLOOR
    assert r["sycl_bf16_vs_decode_today"]["rel"] < BOUND_FLOOR
    # The int8 query is what separates int8 prefill from today's decode.
    assert (
        r["sycl_qk_vs_decode_int8"]["rel"] < 0.5 * r["sycl_qk_vs_decode_today"]["rel"]
    )


@needs_triton
def test_triton_p_rule_is_the_exact_rule():
    r = _run("_child_p_rule()")
    assert r["code_mismatches"] == 0
    assert r["scale_mismatches"] == 0
    # k + 1/2 rounds up, its lower neighbour down, its upper neighbour up.
    assert r["boundary"] == [
        [
            min(math.floor(k + 0.5), 127),
            min(math.floor(k + 0.5), 127) - 1,
            min(math.floor(k + 0.5), 127),
        ]
        for k in (0.5, 1.5, 2.5, 63.5, 126.5)
    ]


# --------------------------------------------------------------------------------------
# config and backend routing
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("setting", ["sycl_bf16", "sycl_qk", "sycl_qk_pv"])
def test_sycl_values_are_compile_factors(setting):
    factors = AttentionConfig(int8_prefill_dot=setting).compile_factors()
    assert factors["int8_prefill_dot"] == setting
    assert "int8_prefill_dot" not in AttentionConfig().compile_factors()


def _stub(monkeypatch, setting, reason=None, window=(-1, -1), block=1600):
    from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv

    probes = []

    def probe(head, dtype, mode, blk):
        probes.append((head, dtype, mode, blk))
        return reason

    monkeypatch.setattr(int8_kv, "decode_max_query_len", lambda: 16)
    monkeypatch.setattr(int8_kv, "int8_prefill_sycl_supported", probe)
    monkeypatch.setattr(int8_kv, "int8_dot_decode_supported", lambda *a: None)
    stub = a1._impl_stub(-1, window, None, block)
    stub.probes = probes
    cfg = SimpleNamespace(
        attention_config=AttentionConfig(int8_prefill_dot=setting, int8_decode_dot="qk")
    )
    stub._init_int8_prefill_dot(cfg)
    return stub


@pytest.mark.parametrize(
    "setting, reason, window, expected, probed",
    [
        ("sycl_bf16", None, (-1, -1), 3, 1),
        ("sycl_qk", None, (-1, -1), 4, 2),
        ("sycl_qk_pv", None, (-1, -1), 5, 3),
        ("sycl_qk", "no int8_prefill=2", (-1, -1), 0, 2),
        ("sycl_qk", None, (127, 0), 0, None),
        ("qk", None, (-1, -1), 1, None),
    ],
)
def test_init_selects_the_sycl_kernel(
    setting, reason, window, expected, probed, monkeypatch
):
    stub = _stub(monkeypatch, setting, reason, window)
    assert stub._int8_prefill_dot == expected
    if probed is None:
        assert stub.probes == []
    else:
        assert stub.probes == [(HD, torch.bfloat16, probed, 1600)]


def test_boot_line(monkeypatch):
    from vllm.v1.attention.backends import flash_attn

    lines = []
    monkeypatch.setattr(
        flash_attn.logger, "info_once", lambda msg, *a: lines.append(msg % a)
    )
    _stub(monkeypatch, "sycl_qk")
    assert lines == [
        (
            "FlashAttention int8_per_token_head KV cache on XPU: native INT8 paged "
            "decode (query length <= 16 per sequence; Q.K int8 x int8 -> int32 "
            "(int8_decode_dot=qk)), native INT8 prefill attention in the "
            "vllm-xpu-kernels chunk-prefill kernel (int8_prefill_dot=sycl_qk: Q.K "
            "int8 x int8 -> int32, P.V BF16), no prefill scratch."
        )
    ]


@pytest.mark.parametrize(
    "setting, kmode", [("sycl_bf16", 1), ("sycl_qk", 2), ("sycl_qk_pv", 3)]
)
def test_prefill_routing(setting, kmode, monkeypatch):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv

    stub = _stub(monkeypatch, setting)
    calls = []
    monkeypatch.setattr(int8_kv, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(
        int8_kv, "forward_prefill", lambda *a, **kw: calls.append(("scratch", kw))
    )
    monkeypatch.setattr(
        int8_kv, "forward_decode", lambda *a, **kw: calls.append(("decode", kw))
    )
    monkeypatch.setattr(
        int8_kv,
        "forward_prefill_sycl",
        lambda *a, **kw: calls.append(("sycl", a[-1], kw)),
    )
    monkeypatch.setattr(
        i8p, "int8_prefill_attention", lambda *a, **kw: calls.append(("triton", kw))
    )
    query = torch.zeros(40, 6, HD, dtype=torch.bfloat16)
    out = torch.zeros_like(query)
    FlashAttentionImpl._forward_int8_per_token_head(
        stub, query, None, a1._metadata(40), out
    )
    FlashAttentionImpl._forward_int8_per_token_head(
        stub, query, None, a1._metadata(8), out
    )
    assert calls == [
        ("sycl", kmode, {"int8_dot": True}),
        ("decode", {"int8_dot": True}),
    ]
    with pytest.raises(NotImplementedError, match="causal"):
        FlashAttentionImpl._forward_int8_per_token_head(
            stub, query, None, a1._metadata(40, causal=False), out
        )


def test_profile_run_and_warmup_take_no_scratch(monkeypatch):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    stub = _stub(monkeypatch, "sycl_qk")
    stub.vllm_flash_attn_version = 2
    out = torch.ones(4, 6 * HD)
    FlashAttentionImpl.forward(stub, None, out, None, None, None, None, out)
    assert stub.scratch_claims == 0
    assert not out.any()
    monkeypatch.setattr(
        i8p,
        "warmup_int8_prefill_attention",
        lambda *a, **kw: pytest.fail("Triton warm-up on the SYCL path"),
    )
    FlashAttentionImpl.warmup_int8_kv_prefill(stub, None, torch.zeros(1, 8))
    assert stub.scratch_claims == 0


def test_forward_prefill_sycl_passes_the_kernel_mode(monkeypatch):
    import sys

    from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv

    seen = {}
    # A stand-in for vllm._xpu_ops (it imports vllm_xpu_kernels).
    fake = SimpleNamespace(
        xpu_ops=SimpleNamespace(flash_attn_varlen_func=lambda *a, **kw: seen.update(kw))
    )
    monkeypatch.setitem(sys.modules, "vllm._xpu_ops", fake)
    t = torch.zeros(1)
    int8_kv.forward_prefill_sycl(
        t, t, t, t, t, t, t, 40, t, 100, t, 0.0625, 3, int8_dot=True
    )
    assert seen["int8_prefill"] == 3 and seen["int8_dot"] is True
    assert seen["causal"] is True and seen["window_size"] == [-1, -1]

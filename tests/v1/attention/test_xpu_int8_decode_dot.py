# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for INT8 attention arithmetic at decode
(`AttentionConfig.int8_decode_dot`) and its agreement with the native INT8
prefill attention (`int8_prefill_dot="qk"`).

With int8_decode_dot the vllm-xpu-kernels paged decode kernel quantizes the
query per (token, query head) in-kernel and computes Q.K as an s8 x s8 -> s32
DPAS dot on the cached K codes. The kernel cannot run on a CPU; what is
checked here:

1. the query rule: the engine's `quantize_query_int8` (Triton, interpreter) is
   bit for bit the exact rule (a float64-backed reference) the decode kernel
   implements (`int8_attn_query_scale` / `int8_attn_query_code` in the kernels
   tree), including codes on and next to the half-integer boundaries;
2. the decode kernel's exactly rounded division (`int8_attn_div_rn`: a
   reciprocal estimate, two Newton steps, one FMA residual correction),
   emulated with reciprocal estimates off by up to 2^12 ulp, against IEEE
   division;
3. prefill / decode agreement on the same cached tokens: the prefill kernel
   ("qk", interpreter) against a float32 model of the decode kernel's
   arithmetic (`decode_model`: 64-slot KV tiles, four sub-groups of 16 columns
   each with its own online softmax, KV splits merged by log-sum-exp as
   ReduceSplitK does, P * v_scale rounded to BF16, V codes widened), at spec-
   decode verify shapes (1, 4, 5 and 16 query rows per sequence). The card
   check (`benchmarks/kernels/benchmark_xpu_int8_decode_attention.py --check`)
   compares the real kernel with the same model.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention import test_xpu_int8_prefill_attention as a1
from vllm.config.attention import AttentionConfig
from vllm.triton_utils import HAS_TRITON

HD = 256
SCALE = HD**-0.5
LOG2E = 1.4426950408889634
# (seq_len, q_len): decode rows and spec-decode verify rows over contexts
# spanning one to many KV tiles (block 128: two tiles of 64 per block).
DEC_SEQS = [(150, 1), (333, 4), (1100, 16), (61, 5)]


# --------------------------------------------------------------------------------------
# the shared query rule, exact reference
# --------------------------------------------------------------------------------------
def _div_rn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE fp32 division (float64 quotient rounded once more is exact for
    fp32 operands: 53 >= 2 * 24 + 2)."""
    return (a.astype(np.float64) / b.astype(np.float64)).astype(np.float32)


def query_rule(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The decode kernel's rule over the last dimension of a BF16 query:
    scale = max(absmax / 127, 1e-10), inv = 1 / scale (both IEEE), y = x * inv
    (fp32), code = sign(y) * min((trunc(2|y|) + 1) >> 1, 127)."""
    xf = x.float().numpy()
    absmax = np.abs(xf).max(-1)
    scale = np.maximum(_div_rn(absmax, np.float32(127.0)), np.float32(1e-10))
    inv = _div_rn(np.float32(1.0), scale)
    y = (xf * inv[..., None]).astype(np.float32)
    a2 = np.minimum(np.abs(y) * np.float32(2.0), np.float32(255.0))
    c = np.minimum((a2.astype(np.int32) + 1) >> 1, 127)
    code = np.where(y < 0, -c, c).astype(np.int8)
    return torch.from_numpy(code), torch.from_numpy(scale.astype(np.float32))


def _boundary_query() -> torch.Tensor:
    """Rows whose scaled values sit on and one ulp around half-integers, a
    zero row, a single-spike row and an all-negative row."""
    rows = []
    for k in (0.5, 1.5, 2.5, 63.5, 126.5):
        row = torch.zeros(HD, dtype=torch.bfloat16)
        row[0] = 127.0  # absmax 127 -> scale 1, inv 1: y == x
        bits = torch.tensor([k], dtype=torch.bfloat16).view(torch.int16)
        # k + 1/2 and its BF16 neighbours below and above.
        vals = torch.cat([bits, bits - 1, bits + 1]).view(torch.bfloat16)
        row[1:4] = vals
        row[4:7] = -vals
        rows.append(row)
    rows.append(torch.zeros(HD, dtype=torch.bfloat16))
    spike = torch.zeros(HD, dtype=torch.bfloat16)
    spike[17] = 3.0e-3
    rows.append(spike)
    rows.append(
        -torch.randn(HD, generator=torch.Generator().manual_seed(0))
        .abs()
        .to(torch.bfloat16)
    )
    return torch.stack(rows)[:, None, :]


# --------------------------------------------------------------------------------------
# a float32 model of the decode kernel's arithmetic
# --------------------------------------------------------------------------------------
def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def decode_model(
    q_rows: torch.Tensor,
    k_codes: torch.Tensor,
    k_scale: torch.Tensor,
    v_codes: torch.Tensor,
    v_scale: torch.Tensor,
    int8_dot: bool,
    num_splits: int = 1,
) -> torch.Tensor:
    """Decode output of `q_rows` [q, hq, D] (the last q positions of a sequence
    of L = k_codes.shape[0] tokens; row i sees keys [0, L - q + i]) as the
    paged decode kernel computes it: per KV tile of 64 slots, four sub-groups
    of 16 columns; S = s32 * q_scale * k_scale (int8_dot) or the BF16 query
    against the widened K codes, then * k_scale; online softmax in base 2 per
    sub-group with the scale folded into exp2; P * v_scale rounded to BF16
    times the V codes; sub-groups merged, then KV splits merged by
    log-sum-exp."""
    ql, hq, _ = q_rows.shape
    L, nkv, _ = k_codes.shape
    g = hq // nkv
    eff = SCALE * LOG2E
    if int8_dot:
        qc, qs = query_rule(q_rows)
        qmat = qc.to(torch.int64)
    out = torch.empty(ql, hq, HD)
    for i in range(ql):
        seq = L - ql + i + 1  # keys this row sees
        n_tiles = -(-seq // 64)
        per = -(-n_tiles // num_splits)
        splits = [(s * per, min(n_tiles, (s + 1) * per)) for s in range(num_splits)]
        splits = [(a, b) for a, b in splits if a < b]
        for h in range(hq):
            kv = h // g
            parts = []  # (m, l, O) per split, merged sub-groups
            for t0, t1 in splits:
                sg = []
                for s in range(4):
                    m = -math.inf
                    lsum = 0.0
                    acc = torch.zeros(HD)
                    for t in range(t0, t1):
                        cols = torch.arange(t * 64 + 16 * s, t * 64 + 16 * s + 16)
                        live = cols < seq
                        if not bool(live.any()):
                            continue
                        c = cols.clamp(max=L - 1)
                        kc = k_codes[c, kv]
                        if int8_dot:
                            # exact: |dot| <= 256 * 127^2 < 2^53
                            dot = (kc.double() @ qmat[i, h].double()).float()
                            sc = dot * qs[i, h] * k_scale[c, kv]
                        else:
                            dot = kc.float() @ q_rows[i, h].float()
                            sc = dot * k_scale[c, kv]
                        sc = torch.where(live, sc, torch.tensor(-math.inf))
                        m_new = max(m, eff * float(sc.max()))
                        p = torch.exp2(sc * eff - m_new)
                        resc = math.exp2(m - m_new) if m > -math.inf else 0.0
                        lsum = lsum * resc + float(p.sum())
                        pw = _bf16(p * v_scale[c, kv])
                        acc = acc * resc + pw @ v_codes[c, kv].float()
                        m = m_new
                    if m > -math.inf:
                        sg.append((m, lsum, acc))
                mm = max(x[0] for x in sg)
                ll = sum(x[1] * math.exp2(x[0] - mm) for x in sg)
                oo = sum(x[2] * math.exp2(x[0] - mm) for x in sg)
                parts.append((mm, ll, oo))
            mm = max(x[0] for x in parts)
            ll = sum(x[1] * math.exp2(x[0] - mm) for x in parts)
            oo = sum(x[2] * math.exp2(x[0] - mm) for x in parts)
            out[i, h] = oo / ll
    return out


# --------------------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------------------
def _dec_case(seed: int, block: int = 128) -> SimpleNamespace:
    """A1's case builder over DEC_SEQS (stale codes, NaN / inf scales outside
    the written slots, a random block order, a strided query view)."""
    saved = a1.SEQS
    a1.SEQS = DEC_SEQS
    try:
        return a1._case(1, seed=seed, block=block)
    finally:
        a1.SEQS = saved


def _quantized_ref(case, rows: slice, i: int) -> torch.Tensor:
    """float64 attention of the int8 query codes x scales against the cached
    codes x scales (the arithmetic both int8 kernels approximate)."""
    (kq, ks), (vq, vs) = case.k_rows[i], case.v_rows[i]
    qc, qs = query_rule(case.query[rows])
    q_deq = qc.double() * qs.double()[..., None]
    return a1._attend(
        q_deq,
        kq.double() * ks.double()[..., None],
        vq.double() * vs.double()[..., None],
        SCALE,
    )


# --------------------------------------------------------------------------------------
# interpreter children
# --------------------------------------------------------------------------------------
def _child_query_rule() -> None:
    a1._patch_interpreter()
    from vllm.v1.attention.ops import xpu_int8_prefill_attention as i8p

    torch.manual_seed(5)
    q = torch.randn(64, 6, HD)
    q[..., :3] *= 20.0  # outlier channels
    queries = [q.to(torch.bfloat16), _boundary_query()]
    mismatches = 0
    for x in queries:
        codes, scales = i8p.quantize_query_int8(x)
        ref_c, ref_s = query_rule(x)
        mismatches += int((codes != ref_c).sum()) + int((scales != ref_s).sum())
    # The boundary rows: k + 1/2 rounds away from zero, its lower neighbour
    # down, its upper neighbour up.
    bc, _ = query_rule(_boundary_query())
    expect = []
    for k in (0.5, 1.5, 2.5, 63.5, 126.5):
        hi = min(int(math.floor(k + 0.5)), 127)
        expect.append([hi, hi - 1, hi])
    got = [bc[r, 0, 1:4].tolist() for r in range(5)]
    neg = [bc[r, 0, 4:7].tolist() for r in range(5)]
    print(
        "RESULT "
        + json.dumps(
            {
                "mismatches": mismatches,
                "boundary_ok": got == expect
                and neg == [[-v for v in e] for e in expect],
                "zero_row_scale": float(query_rule(_boundary_query())[1][5, 0]),
            }
        )
    )


def _child_agreement(seed: int) -> None:
    a1._patch_interpreter()
    from vllm.v1.attention.ops import xpu_int8_prefill_attention as i8p

    case = _dec_case(seed)
    prefill = torch.full_like(case.query, float("nan")).contiguous()
    i8p.int8_prefill_attention(
        case.query,
        case.key,
        case.value,
        case.k_scale,
        case.v_scale,
        prefill,
        case.cu,
        case.seq_lens,
        case.table,
        SCALE,
        pv_int8=False,
    )
    dec, dec4, today, quant, contract = [], [], [], [], []
    for i in range(len(DEC_SEQS)):
        rows = slice(int(case.cu[i]), int(case.cu[i + 1]))
        (kq, ks), (vq, vs) = case.k_rows[i], case.v_rows[i]
        args = (case.query[rows], kq, ks, vq, vs)
        dec.append(decode_model(*args, int8_dot=True))
        dec4.append(decode_model(*args, int8_dot=True, num_splits=4))
        today.append(decode_model(*args, int8_dot=False))
        quant.append(_quantized_ref(case, rows, i))
        contract.append(
            a1._attend(
                case.query[rows],
                kq.double() * ks.double()[..., None],
                vq.double() * vs.double()[..., None],
                SCALE,
            )
        )
    dec, dec4, today = torch.cat(dec), torch.cat(dec4), torch.cat(today)
    quant, contract = torch.cat(quant), torch.cat(contract)
    m = a1.metrics
    result = {
        "finite": bool(torch.isfinite(prefill).all()),
        "prefill_vs_decode": m(prefill.float(), _bf16(dec)),
        "prefill_vs_decode_4splits": m(prefill.float(), _bf16(dec4)),
        "decode_vs_quantized": m(_bf16(dec), quant),
        "prefill_vs_quantized": m(prefill.float(), quant),
        "decode_int8_vs_contract": m(_bf16(dec), contract),
        "decode_today_vs_contract": m(_bf16(today), contract),
        "prefill_vs_decode_today": m(prefill.float(), _bf16(today)),
    }
    print("RESULT " + json.dumps(result))


def _run(call: str) -> dict:
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).parents[3]
    code = f"import tests.v1.attention.test_xpu_int8_decode_dot as t\nt.{call}\n"
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


@needs_triton
def test_prefill_query_quantization_is_the_decode_kernel_rule():
    result = _run("_child_query_rule()")
    assert result["mismatches"] == 0
    assert result["boundary_ok"]
    assert result["zero_row_scale"] == pytest.approx(1e-10)


# Bounds on synthetic Gaussian activations with outlier key channels (measured
# values print with -s). Prefill and decode share the int8 query and the exact
# int32 Q.K; they differ by fp32 order and by where the online softmax rounds
# P * v_scale to BF16 (64-slot tiles in prefill, 16-column sub-group slices and
# KV splits in decode).
BOUND_AGREE = 4e-3
BOUND_QUANTIZED = 5e-3


@needs_triton
@pytest.mark.parametrize("seed", [21, 22])
def test_prefill_and_decode_agree_on_cached_tokens(seed):
    r = _run(f"_child_agreement({seed})")
    assert r["finite"]
    for key in ("prefill_vs_decode", "prefill_vs_decode_4splits"):
        assert r[key]["rel"] < BOUND_AGREE, key
        assert r[key]["cos"] > 0.99999, key
    assert r["decode_vs_quantized"]["rel"] < BOUND_QUANTIZED
    assert r["prefill_vs_quantized"]["rel"] < BOUND_QUANTIZED
    # With int8 decode the two paths are much closer to each other than
    # prefill (int8 query) is to today's decode (BF16 query).
    assert r["prefill_vs_decode"]["rel"] < 0.5 * r["prefill_vs_decode_today"]["rel"]


def _div_rn_device_model(a: np.ndarray, b: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """int8_attn_div_rn with the reciprocal estimate r0; FMAs evaluated in
    float64 (exact products) and rounded once to fp32."""
    f32 = np.float32

    def fma(x, y, z):
        return (
            x.astype(np.float64) * y.astype(np.float64) + z.astype(np.float64)
        ).astype(f32)

    one = np.ones_like(b)
    r = r0.astype(f32)
    r = fma(fma(-b, r, one), r, r)
    r = fma(fma(-b, r, one), r, r)
    q = (a * r).astype(f32)
    e = fma(-b, q, a)
    return fma(e, r, q)


@pytest.mark.parametrize("ulps", [0, 1, 7, 255, 4095])
def test_decode_kernel_division_is_exactly_rounded(ulps):
    rng = np.random.default_rng(ulps)
    n = 200_000
    # absmax / 127 over the range activations take, and 1 / scale.
    absmax = np.exp(rng.uniform(np.log(1e-4), np.log(3e4), n)).astype(np.float32)
    b127 = np.full(n, 127.0, np.float32)
    scale = np.maximum(_div_rn(absmax, b127), np.float32(1e-10))
    for a, b in ((absmax, b127), (np.ones(n, np.float32), scale)):
        exact = _div_rn(np.float32(1.0), b)
        bits = exact.view(np.int32) + rng.integers(-ulps, ulps + 1, n).astype(np.int32)
        r0 = bits.view(np.float32)
        got = _div_rn_device_model(a, b, r0)
        assert np.array_equal(got, _div_rn(a, b)), int((got != _div_rn(a, b)).sum())


# --------------------------------------------------------------------------------------
# config and backend routing
# --------------------------------------------------------------------------------------
def test_int8_decode_dot_off_is_not_a_compile_factor():
    assert "int8_decode_dot" not in AttentionConfig().compile_factors()
    assert (
        AttentionConfig(int8_decode_dot="qk").compile_factors()["int8_decode_dot"]
        == "qk"
    )


def _stub(monkeypatch, setting: str, reason):
    from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv

    monkeypatch.setattr(int8_kv, "decode_max_query_len", lambda: 16)
    monkeypatch.setattr(int8_kv, "int8_dot_decode_supported", lambda *a: reason)
    stub = a1._impl_stub(-1)
    cfg = SimpleNamespace(attention_config=AttentionConfig(int8_decode_dot=setting))
    stub._init_int8_prefill_dot(cfg)
    return stub


@pytest.mark.parametrize(
    "setting, reason, expected",
    [("off", None, False), ("qk", None, True), ("qk", "no int8_dot decode", False)],
)
def test_init_and_decode_routing(setting, reason, expected, monkeypatch):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv

    stub = _stub(monkeypatch, setting, reason)
    assert stub._int8_decode_dot is expected
    calls = []
    monkeypatch.setattr(int8_kv, "forward_decode", lambda *a, **kw: calls.append(kw))
    query = torch.zeros(40, 6, HD, dtype=torch.bfloat16)
    for max_q in (1, 4, 16):
        FlashAttentionImpl._forward_int8_per_token_head(
            stub, query, None, a1._metadata(max_q), torch.zeros_like(query)
        )
    assert calls == [{"int8_dot": expected}] * 3


def test_boot_line_unchanged_when_off(monkeypatch):
    from vllm.v1.attention.backends import flash_attn

    lines = []
    monkeypatch.setattr(
        flash_attn.logger, "info_once", lambda msg, *a: lines.append(msg % a)
    )
    _stub(monkeypatch, "off", None)
    assert lines == [
        (
            "FlashAttention int8_per_token_head KV cache on XPU: native INT8 paged "
            "decode (query length <= 16 per sequence), prefill through a "
            "dequantized torch.bfloat16 scratch."
        )
    ]
    lines.clear()
    _stub(monkeypatch, "qk", None)
    assert lines == [
        (
            "FlashAttention int8_per_token_head KV cache on XPU: native INT8 paged "
            "decode (query length <= 16 per sequence; Q.K int8 x int8 -> int32 "
            "(int8_decode_dot=qk)), prefill through a dequantized torch.bfloat16 "
            "scratch."
        )
    ]

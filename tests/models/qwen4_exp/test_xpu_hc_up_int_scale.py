# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HC up fold's scale division in integers, on CPU (Triton interpreter).

``tl.div_rn`` reaches the XPU device as a plain, not correctly rounded
division. The HC up quant rule (``hc_gemm._i8_scale_bf16``) divided
``amax / 127`` with it; it now uses ``_ieee_over_127``, the integer form of the
HC down fold's ``_ieee_127_over``.

1. ``_ieee_over_127`` equals numpy's IEEE fp32 ``amax / 127`` bit for bit on
   every non-negative bf16 pattern (zero, denormals, normals, inf, NaN); a
   quotient below the fp32 normal range returns 0, which the 1e-5 floor of the
   scale replaces either way.
2. ``_i8_scale_bf16`` equals torch's ``clamp(amax / 127, min=1e-5)`` rounded to
   bf16, and the ``tl.div_rn`` form it replaces, on every pattern.
3. The finish kernel with the folded quant gives the same output, int8 rows and
   scales, bit for bit, as the same kernel on the ``tl.div_rn`` scale, on the
   shapes of ``test_xpu_hc_up_fused`` and on rows whose absmax sweeps the bf16
   patterns.
4. ``_i8_quantize`` keeps its ``tl.div_rn``: the quotient only picks the floor
   candidate and the exact compare against ``(f + 0.5) * scale`` decides, so a
   quotient off by several ulps gives the same code (numpy model of the rule).

The interpreter divides in IEEE, so (1)-(3) prove the integer form and the
composition; the device bits are the card check."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")

if HAS_TRITON:
    from vllm.models.qwen4_exp.xpu.hc_gemm import (
        _i8_quantize,
        _i8_scale_bf16,
        _ieee_over_127,
    )
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _i8_scale_div_rn(amax):
        """``_i8_scale_bf16`` as it was: the division on ``tl.div_rn``."""
        s = tl.div_rn(amax, 127.0)
        s = tl.where(s < 1e-5, 1e-5, s)
        u = s.to(tl.uint32, bitcast=True)
        u = u + 0x7FFF + ((u >> 16) & 1)
        r = ((u >> 16) << 16).to(tl.float32, bitcast=True)
        return tl.where(s != s, s, r)

    @triton.jit
    def _scales_of_bits(m_ptr, quo_ptr, new_ptr, old_ptr, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        m = tl.load(m_ptr + offs, mask, other=0)
        amax = (m.to(tl.int32) << 16).to(tl.float32, bitcast=True)
        tl.store(quo_ptr + offs, _ieee_over_127(amax), mask)
        tl.store(new_ptr + offs, _i8_scale_bf16(amax), mask)
        tl.store(old_ptr + offs, _i8_scale_div_rn(amax), mask)

    @triton.jit
    def _finish_silu_quant_div_rn(
        partials,
        output,
        q_out,
        s_out,
        M: tl.constexpr,
        N: tl.constexpr,
        TILES: tl.constexpr,
        LORA: tl.constexpr,
        HC: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``_hc_finish_silu_quant`` on the ``tl.div_rn`` scale."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        offsets = row * N + cols
        value = tl.full((BLOCK,), 0, tl.float32)
        for step in range(TILES):
            value += tl.load(partials + step * M * N + offsets, mask, other=0)
        stored = value.to(output.dtype.element_ty)
        tl.store(output + offsets, stored, mask)
        v = stored.to(tl.float32) / HC
        y = v * tl.sigmoid(v)
        lmask = cols < LORA
        x = y.to(tl.bfloat16).to(tl.float32)
        amax = tl.max(tl.where(lmask, tl.abs(x), 0.0), axis=0)
        scale = _i8_scale_div_rn(amax)
        tl.store(q_out + row * LORA + cols, _i8_quantize(x, scale), lmask)
        tl.store(s_out + row, scale)

    @triton.jit
    def _quant_rows(x, q, s, N: tl.constexpr, BLOCK: tl.constexpr, OLD: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        v = tl.load(x + row * N + cols, mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.where(mask, tl.abs(v), 0.0), axis=0)
        scale = _i8_scale_div_rn(amax) if OLD else _i8_scale_bf16(amax)
        tl.store(q + row * N + cols, _i8_quantize(v, scale), mask)
        tl.store(s + row, scale)


HC = 4
LOWRANK = 320
COLUMNS = 336  # the merged down GEMM: 320 lora + 4 injection + 12 padding rows
TILES = 80  # 10240 / 128
NAN_BITS = (0x7F81, 0x7FC0, 0x7FFF)  # bf16 NaNs (sign clear)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _faithful_bf16_cast() -> None:
    """The Triton interpreter casts fp32 -> bf16 by truncation; the device and
    torch round to nearest even. Patch that one direction (both kernels run
    with it) so CPU results mean what device results mean."""
    import triton.language as tl_
    from triton.runtime import interpreter as it

    if getattr(it, "_f2_bf16_patched", False):
        return
    orig = it._convert_float

    def conv(input, input_dtype, output_dtype, rounding_mode):
        if input_dtype == tl_.float32 and output_dtype == tl_.bfloat16:
            f = torch.from_numpy(np.ascontiguousarray(input).view(np.float32).copy())
            return f.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        return orig(input, input_dtype, output_dtype, rounding_mode)

    it._convert_float = conv
    it._f2_bf16_patched = True


def _all_patterns() -> torch.Tensor:
    """Zero, every denormal, normal and inf pattern (0x0000-0x7F80) and NaNs."""
    m = np.concatenate([np.arange(0, 0x7F81, dtype=np.int64), np.array(NAN_BITS)])
    return torch.from_numpy(m)


def _amax_of(m: torch.Tensor) -> np.ndarray:
    return (m.numpy().astype(np.uint32) << 16).view(np.float32)


def _ref_quotient(m: torch.Tensor) -> np.ndarray:
    """The numpy IEEE fp32 ``amax / 127`` (round to nearest even)."""
    with np.errstate(all="ignore"):
        return (_amax_of(m) / np.float32(127)).astype(np.float32)


def _ref_scale(m: torch.Tensor) -> torch.Tensor:
    """``bf16(clamp(amax / 127, min=1e-5))`` as an fp32 tensor: torch's IEEE
    division, clamp (NaN kept) and round to nearest even."""
    a = torch.from_numpy(_amax_of(m).copy())
    return torch.clamp(a / 127.0, min=1e-5).to(torch.bfloat16).to(torch.float32)


def _run_scales() -> tuple[torch.Tensor, ...]:
    m = _all_patterns()
    n = m.numel()
    quo, new, old = (torch.empty(n, dtype=torch.float32) for _ in range(3))
    _scales_of_bits[(triton.cdiv(n, 1024),)](m, quo, new, old, n, 1024)
    return m, quo, new, old


def _same_bits_or_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Indices where two fp32 tensors differ; any NaN equals any NaN."""
    both_nan = torch.isnan(a) & torch.isnan(b)
    return ((_bits(a) != _bits(b)) & ~both_nan).nonzero()[:, 0]


def _check_quotient_every_bf16() -> None:
    m, quo, _, _ = _run_scales()
    ref = torch.from_numpy(_ref_quotient(m).copy())
    tiny = ref.abs() < torch.finfo(torch.float32).tiny  # below the normal range
    want = torch.where(tiny, torch.zeros_like(ref), ref)
    bad = _same_bits_or_nan(quo, want)
    assert bad.numel() == 0, [(hex(int(m[i])), float(quo[i])) for i in bad[:8]]
    # the pins: zero, a denormal, inf, NaN, bf16 max, exact 127 and 254
    got = {int(k): float(v) for k, v in zip(m.tolist(), quo)}
    assert got[0] == 0.0 and got[0x0001] == 0.0 and got[0x7F80] == float("inf")
    assert got[0x7FC0] != got[0x7FC0]
    assert got[0x42FE] == 1.0 and got[0x437E] == 2.0  # 127 and 254
    n_exact = int((~tiny[:0x7F80]).sum())
    print(f"amax / 127: integer form bitwise to IEEE on {n_exact} bf16 values")


def _check_scale_every_bf16() -> None:
    m, _, new, old = _run_scales()
    ref = _ref_scale(m)
    bad = _same_bits_or_nan(new, ref)
    assert bad.numel() == 0, [(hex(int(m[i])), float(new[i])) for i in bad[:8]]
    bad = _same_bits_or_nan(new, old)
    assert bad.numel() == 0, [(hex(int(m[i])), float(new[i])) for i in bad[:8]]
    print(f"scale: integer form equals IEEE and tl.div_rn on {m.numel()} patterns")


def _check_finish_quant_before_after() -> None:
    from vllm.models.qwen4_exp.xpu import hc_gemm

    _faithful_bf16_cast()
    torch.manual_seed(0)
    for rows in (1, 2, 3, 7, 8, 13, 16, 32):
        partials = torch.randn(TILES, rows, COLUMNS) * 0.05
        if rows >= 4:
            partials[:, 0] = 0  # zero row: the 1e-5 scale floor
            partials[:, 1] *= 1e-6  # tiny row
            partials[:, 2] *= 1e3  # large row
        results = []
        for kernel in (hc_gemm._hc_finish_silu_quant, _finish_silu_quant_div_rn):
            out = torch.empty(rows, COLUMNS, dtype=torch.bfloat16)
            q = torch.empty(rows, LOWRANK, dtype=torch.int8)
            s = torch.empty(rows, 1, dtype=torch.float32)
            kernel[(rows,)](partials, out, q, s, rows, COLUMNS, TILES, LOWRANK, HC, 512)
            results.append((out.view(torch.int16), q, _bits(s)))
        for new, old in zip(*results):
            assert torch.equal(new, old), rows
        print(f"finish+quant rows {rows}: output, int8 rows, scales equal the old form")


def _check_quant_rows_sweep() -> None:
    """Rows whose absmax is each of a sweep of bf16 patterns (every denormal
    step, every normal binade at several mantissas, the corners), the rest of
    the row smaller, noise and exact ties of the scale."""
    _faithful_bf16_cast()
    torch.manual_seed(1)
    n = 320
    pats = sorted(
        {0, 1, 2, 63, 127, 128, 129, 0x00FF, 0x0100, 0x3F80, 0x7F7F}
        | set(range(0x0080, 0x7F80, 211))
        | {(e << 7) | f for e in range(1, 255) for f in (0, 1, 63, 64, 126, 127)}
    )
    amax = torch.from_numpy(np.array(pats, dtype=np.uint32) << 16).view(torch.float32)
    rows = amax.numel()
    x = torch.randn(rows, n) * amax.unsqueeze(1) * 0.3
    x[:, 0] = amax
    x[:, 1] = -amax
    x = x.to(torch.bfloat16)
    x[amax == 0] = 0
    got = []
    for old in (False, True):
        q = torch.empty(rows, n, dtype=torch.int8)
        s = torch.empty(rows, dtype=torch.float32)
        _quant_rows[(rows,)](x, q, s, n, 512, old)
        got.append((q, _bits(s)))
    assert torch.equal(got[0][0], got[1][0]), (got[0][0] != got[1][0]).sum()
    assert torch.equal(got[0][1], got[1][1])
    print(f"quant rows: {rows} absmax patterns, codes and scales equal the old form")


def _run(name: str) -> None:
    root = Path(__file__).parents[3]
    module = "tests.models.qwen4_exp.test_xpu_hc_up_int_scale"
    code = f"from {module} import {name}\n{name}()\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "TRITON_INTERPRET": "1"},
        timeout=2400,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stderr[-4000:] + result.stdout


def test_quotient_is_ieee_on_every_bf16() -> None:
    _run("_check_quotient_every_bf16")


def test_scale_is_ieee_and_equals_div_rn_on_every_bf16() -> None:
    _run("_check_scale_every_bf16")


def test_finish_quant_equals_the_div_rn_form() -> None:
    _run("_check_finish_quant_before_after")


def test_quant_rows_equal_the_div_rn_form_over_an_absmax_sweep() -> None:
    _run("_check_quant_rows_sweep")


def _q_model(x: np.ndarray, scale: np.ndarray, quotient: np.ndarray) -> np.ndarray:
    """``_i8_quantize`` in numpy with the quotient given."""
    f = np.floor(quotient)
    boundary = (f + np.float32(0.5)) * scale
    half_f = f * np.float32(0.5)
    tie = np.where(np.floor(half_f) == half_f, f, f + 1)
    q = np.where(x > boundary, f + 1, np.where(x < boundary, f, tie))
    return np.clip(q, -128, 127)


def test_quantize_does_not_depend_on_the_quotient_rounding() -> None:
    """The kept ``tl.div_rn`` of ``_i8_quantize`` may be several ulps off on the
    device: the code is the same for a quotient moved by up to 8 ulps either
    way, on random rows and on rows built on the half ties and next to them."""
    rng = np.random.default_rng(0)
    m = torch.from_numpy(rng.integers(0x3E00, 0x4400, 4000).astype(np.int64))
    scale = _ref_scale(m).numpy()[:, None]
    k = rng.integers(-127, 127, (4000, 1)).astype(np.float32)
    delta = rng.choice(
        np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32) / 256, (4000, 1)
    )
    noise = rng.standard_normal((4000, 1)).astype(np.float32) * 20
    x = np.concatenate([(k + 0.5 + delta) * scale, noise * scale, k * scale], axis=1)
    x = torch.from_numpy(x).to(torch.bfloat16).float().numpy()  # bf16 inputs
    exact = (x / scale).astype(np.float32)
    want = _q_model(x, scale, exact)
    # the kernel's own rule, against the reference rule (round half to even)
    ref = np.clip(np.rint(x.astype(np.float64) / scale.astype(np.float64)), -128, 127)
    assert np.array_equal(want, ref)
    for ulps in range(-8, 9):
        moved = exact.copy()
        for _ in range(abs(ulps)):
            moved = np.nextafter(
                moved, np.float32(np.inf if ulps > 0 else -np.inf)
            ).astype(np.float32)
        assert np.array_equal(_q_model(x, scale, moved), want), ulps

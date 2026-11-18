# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_XPU_HC_DOWN_FOLD on CPU, kernels in the Triton interpreter.

1. The folded combine + norm kernel writes ``out`` and ``xn`` bit for bit as
   ``hc_combine_norm`` does, and int8 rows and scales bit for bit as the IEEE
   form of the native quant rule (``int8_dense.cpp`` ``quant_rows_launch``, the
   calibration rule) gives on that ``xn``: 1-64 rows, QFN's and narrower
   streams, shared and per-stream norm weights, with and without injection
   logits (strided, as the down GEMM's slice hands them), both modes, every
   one of the 24 arrival orders of a row's four streams; the arrival words are
   zero after every launch (the next launch and graph replay start clean).
2. The same for the folded grouped norm against ``grouped_gemma_rmsnorm``.
3. The rule on rows built to sit on its corners: exact half ties, zero, tiny
   (127 / amax overflows), huge, NaN and inf elements; and its ``127 / amax``
   (integer arithmetic, since ``tl.div_rn`` is not IEEE on the XPU backend)
   bitwise to IEEE division on every positive bf16 amax.
4. The folded ops compose: the fused mix / combine-and-mix and the split ops
   on the pre-quantized GEMM equal the unfolded ops on the fused GEMM (whose
   CPU stand-in runs the same rule), bit for bit.
5. A HyperConnection resolves the fold at load and its mix routes through the
   folded ops; the fakes take the new arguments.

The device bits and the graph-replay reset are the card checks."""

import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.models.qwen4_exp.test_xpu_hc import _no_tp  # noqa: F401
from tests.models.qwen4_exp.test_xpu_hc_int8_up import ref_kernel  # noqa: F401
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")

if HAS_TRITON:
    from vllm.models.qwen4_exp.xpu.hc_down_fold import (
        _bf16_bits,
        _ieee_127_over,
        _q127,
        _row_scale,
        _stored_abs,
    )
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _rule_rows(x, q, s, N: tl.constexpr, BLOCK: tl.constexpr, INV127: tl.constexpr):
        """The fold's rule alone on bf16 rows (one stream per row)."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        v = tl.load(x + row * N + cols, mask, other=0.0).to(tl.float32)
        amax = tl.max(_stored_abs(v, mask), axis=0)
        full = _bf16_bits(amax) + (tl.full([], 1, tl.int64) << 60)
        inv, xs = _row_scale(full, 1, INV127)
        tl.store(q + row * N + cols, _q127(v, inv), mask)
        tl.store(s + row, xs)

    @triton.jit
    def _inv_of_bits(m_ptr, inv_ptr, div_ptr, N, BLOCK: tl.constexpr):
        """``127 / amax`` for bf16 bit patterns ``m``: the fold's integer form,
        and ``tl.div_rn`` beside it (what the fold used before; the device
        check prints how often the two differ)."""
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        m = tl.load(m_ptr + offs, mask, other=1)
        tl.store(inv_ptr + offs, _ieee_127_over(m), mask)
        amax = (m.to(tl.int32) << 16).to(tl.float32, bitcast=True)
        num = tl.full([BLOCK], 127.0, tl.float32)
        tl.store(div_ptr + offs, tl.div_rn(num, amax), mask)


HC = 4
QFN_EMBD = 2560


def rule_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The native per-token rule with IEEE division, in numpy float32:
    mx = max |x| (NaN ignored), inv = mx > 0 ? 127 / mx : 0, xs = mx * (1/127),
    q = clamp(rint(x * inv), +-127), NaN -> -127. Returns int8 rows and fp32
    [N, 1] scales."""
    import numpy as np

    xf = x.detach().float().numpy()
    f32 = np.float32
    with np.errstate(all="ignore"):
        a = np.where(np.isnan(xf), f32(0), np.abs(xf))
        mx = a.max(axis=1).astype(f32)
        pos = mx > 0
        inv = np.where(pos, f32(127) / np.where(pos, mx, f32(1)), f32(0)).astype(f32)
        xs = (mx * (f32(1) / f32(127))).astype(f32)
        p = (xf * inv[:, None]).astype(f32)
        r = np.rint(p)
        r = np.where(np.isnan(r), f32(-127), r)
        q = np.clip(r, -127, 127).astype(np.int8)
    return torch.from_numpy(q), torch.from_numpy(xs).reshape(-1, 1)


def _interpreter_casts() -> None:
    """The F2 cell's RNE fp32 -> bf16 patch, plus an exact bf16 -> fp32 widen:
    the interpreter's own widening mangles bf16 subnormals (0x000B came back
    as 2**-127), where the device and torch shift the bits."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as it

    from tests.models.qwen4_exp.test_xpu_hc_up_fused import _faithful_bf16_cast

    _faithful_bf16_cast()
    if getattr(it, "_kf_bf16_widen", False):
        return
    orig = it._convert_float

    def conv(input, input_dtype, output_dtype, rounding_mode):
        if input_dtype == tl.bfloat16 and output_dtype == tl.float32:
            return np.ascontiguousarray(input).view(np.uint16).astype(np.uint32) << 16
        return orig(input, input_dtype, output_dtype, rounding_mode)

    it._convert_float = conv
    it._kf_bf16_widen = True


def _order(perm) -> int:
    return sum(s << (2 * i) for i, s in enumerate(perm))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _inputs(rows, n_embd, seed, shared_w, w_dtype, inj):
    g = torch.Generator().manual_seed(seed)
    dim = HC * n_embd
    residual = (torch.randn(rows, dim, generator=g) * 2).to(torch.bfloat16)
    block = torch.randn(rows, n_embd, generator=g).to(torch.bfloat16)
    if rows >= 4:
        residual[1] *= 1e-30  # tiny rows
        block[1] *= 1e-30
        residual[2] *= 1e4  # large
    norm = (0.1 * torch.randn(n_embd if shared_w else dim, generator=g)).to(w_dtype)
    injection = None
    if inj:
        # The down GEMM's slice: [rows, HC] with the GEMM's row stride.
        down = torch.randn(rows, 336, generator=g).to(torch.bfloat16)
        injection = down[:, 320 : 320 + HC]
    return residual, block, injection, norm


def _check_combine_norm_q() -> None:
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine_norm
    from vllm.models.qwen4_exp.xpu import hc_down_fold

    _interpreter_casts()
    eps = 1e-6
    cases = [
        (rows, QFN_EMBD, True, torch.bfloat16, True) for rows in (1, 2, 5, 8, 13, 16)
    ]
    cases += [
        (64, QFN_EMBD, False, torch.bfloat16, True),
        (33, 640, True, torch.float32, True),
    ]
    cases += [
        (7, 600, False, torch.float32, True),
        (5, 640, True, torch.bfloat16, False),
    ]
    for i, (rows, n_embd, shared_w, w_dtype, inj) in enumerate(cases):
        residual, block, injection, norm = _inputs(
            rows, n_embd, i, shared_w, w_dtype, inj
        )
        out_ref, y_ref = hc_combine_norm(residual, block, injection, norm, eps, HC)
        q_ref, s_ref = rule_ref(y_ref)
        words = hc_down_fold.new_arrival_words(rows + 3, "cpu")
        orders = (
            list(itertools.permutations(range(HC)))
            if rows == 5
            else [(0, 1, 2, 3), (3, 2, 1, 0)]
        )
        for mode in (1, 2):
            for perm in orders:
                out, y, q, s = hc_down_fold.combine_norm_q(
                    residual, block, injection, norm, eps, HC, words, mode, _order(perm)
                )
                key = (rows, n_embd, shared_w, w_dtype, inj, mode, perm)
                assert torch.equal(_bits(out), _bits(out_ref)), key
                assert torch.equal(_bits(y), _bits(y_ref)), key
                assert torch.equal(q, q_ref), (key, int((q != q_ref).sum()))
                assert torch.equal(_bits(s), _bits(s_ref)), key
                assert int(words.abs().sum()) == 0, key
        print(
            f"combine+norm+quant rows {rows} embd {n_embd} shared_w {shared_w} "
            f"{w_dtype} inj {inj}: out, xn, int8 rows, scales bitwise; "
            f"{len(orders)} orders x 2 modes; words clean"
        )


def _check_norm_q() -> None:
    from vllm.models.qwen4_exp.nvidia.ops.hc import grouped_gemma_rmsnorm
    from vllm.models.qwen4_exp.xpu import hc_down_fold

    _interpreter_casts()
    eps = 1e-6
    for i, (rows, n_embd, shared_w) in enumerate(
        ((1, QFN_EMBD, True), (5, QFN_EMBD, False), (16, 640, True), (64, 600, False))
    ):
        x, _, _, norm = _inputs(rows, n_embd, 100 + i, shared_w, torch.bfloat16, False)
        y_ref = grouped_gemma_rmsnorm(x, norm, eps, HC)
        q_ref, s_ref = rule_ref(y_ref)
        words = hc_down_fold.new_arrival_words(rows, "cpu")
        orders = (
            list(itertools.permutations(range(HC)))
            if rows == 5
            else [(0, 1, 2, 3), (2, 0, 3, 1)]
        )
        for mode in (1, 2):
            for perm in orders:
                y, q, s = hc_down_fold.norm_q(
                    x, norm, eps, HC, words, mode, _order(perm)
                )
                key = (rows, n_embd, shared_w, mode, perm)
                assert torch.equal(_bits(y), _bits(y_ref)), key
                assert torch.equal(q, q_ref), (key, int((q != q_ref).sum()))
                assert torch.equal(_bits(s), _bits(s_ref)), key
                assert int(words.abs().sum()) == 0, key
        print(f"norm+quant rows {rows} embd {n_embd} shared_w {shared_w}: bitwise")


def _check_rule_corners() -> None:
    from vllm.models.qwen4_exp.xpu import hc_down_fold

    _interpreter_casts()
    n = 256
    rows = []
    # amax 127: inv = 1, so x = k + 0.5 is an exact tie for every k.
    ties = torch.tensor([k + 0.5 for k in range(-127, 127)] + [127.0, -127.0])
    rows += [ties, -ties]
    # amax 127/8: inv = 8, ties at (k + 0.5) / 8.
    rows.append(torch.tensor([(k + 0.5) / 8 for k in range(-127, 127)] + [127 / 8] * 2))
    rows.append(torch.zeros(n))
    rows.append(torch.full((n,), 1e-39))  # 127 / amax overflows to inf
    tiny = torch.full((n,), 1e-39)
    tiny[::3] = 0.0
    rows.append(tiny)
    rows.append(torch.linspace(-3e38, 3e38, n))
    nan_row = torch.randn(n)
    nan_row[5] = float("nan")
    rows.append(nan_row)
    inf_row = torch.randn(n)
    inf_row[7] = float("inf")
    inf_row[9] = float("-inf")
    rows.append(inf_row)
    torch.manual_seed(3)
    for _ in range(16):
        rows.append(torch.randn(n) * float(torch.empty(1).uniform_(-12, 12).exp()))
    x = torch.stack(rows).to(torch.bfloat16)
    q = torch.empty(x.shape, dtype=torch.int8)
    s = torch.empty(x.shape[0], 1, dtype=torch.float32)
    _rule_rows[(x.shape[0],)](x, q, s, n, n, hc_down_fold.INV127_BITS)
    ref_q, ref_s = rule_ref(x)
    assert torch.equal(q, ref_q), (q != ref_q).nonzero()[:8]
    bad = (s.view(torch.int32) != ref_s.view(torch.int32)).nonzero()[:, 0]
    assert bad.numel() == 0, (bad, s[bad], ref_s[bad])
    # ties to even: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, -0.5 -> 0
    assert q[0, 127].item() == 0 and q[0, 128].item() == 2 and q[0, 129].item() == 2
    assert (
        q[0, 126].item() == 0 and q[0, 254].item() == 127 and q[0, 255].item() == -127
    )
    assert q[3].abs().sum() == 0 and s[3].item() == 0.0
    assert q[7, 5].item() == -127  # NaN element
    # The constant is the device's 1.0f / 127.0f.
    import numpy as np

    c = np.array([np.float32(1) / np.float32(127)], dtype=np.float32)
    assert int(c.view(np.int32)[0]) == hc_down_fold.INV127_BITS
    print("rule corners: ties, zero, overflowing inv, huge, NaN, inf: bitwise")


def _ieee_inv_ref() -> tuple[torch.Tensor, torch.Tensor]:
    """Every positive bf16 pattern up to +inf (int64) and numpy's fp32
    ``127 / x`` of it (IEEE, round to nearest even), as int32 bits."""
    import numpy as np

    m = np.arange(1, 0x7F81, dtype=np.int64)
    x = (m.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(all="ignore"):
        ref = (np.float32(127) / x).astype(np.float32)
    return torch.from_numpy(m), torch.from_numpy(ref.view(np.int32))


def _check_inv_every_bf16() -> None:
    m, ref = _ieee_inv_ref()
    inv = torch.empty(m.numel(), dtype=torch.float32)
    div = torch.empty_like(inv)
    _inv_of_bits[(triton.cdiv(m.numel(), 1024),)](m, inv, div, m.numel(), 1024)
    bad = (inv.view(torch.int32) != ref).nonzero()[:, 0]
    assert bad.numel() == 0, [(hex(int(m[i])), float(inv[i])) for i in bad[:8]]
    print(f"127 / amax: integer form bitwise to IEEE on all {m.numel()} bf16 values")


_FUSED_SCHEMA = (
    "int8_dense_w8a8_fused_gemm(Tensor x, Tensor w, Tensor w_scale, Tensor? bias) "
    "-> Tensor"
)
_PREQUANT_SCHEMA = (
    "int8_dense_w8a8_gemm(Tensor xq, Tensor xs, Tensor w, Tensor w_scale, "
    "Tensor? bias) -> Tensor"
)
_SCHEMAS = (_FUSED_SCHEMA, _PREQUANT_SCHEMA)


def _gemm_from_rows(xq, xs, w, w_scale):
    out = (xq.double() @ w.double().t()) * xs.double().reshape(-1, 1)
    return (out * w_scale.double().reshape(1, -1)).to(torch.bfloat16)


def _cpu_int8_ops(lib, calls: list) -> None:
    """CPU stand-ins for the two vllm-xpu-kernels ops: the fused one runs the
    rule (IEEE form) then the GEMM from rows; the pre-quantized one only the
    latter. Both record their calls."""
    for schema in _SCHEMAS:
        lib.define(schema)

    def fused(x, w, w_scale, bias):
        calls.append("fused")
        q, s = rule_ref(x)
        return _gemm_from_rows(q, s, w, w_scale)

    def prequant(xq, xs, w, w_scale, bias):
        calls.append("prequant")
        return _gemm_from_rows(xq, xs, w, w_scale)

    lib.impl("int8_dense_w8a8_fused_gemm", fused, "CPU")
    lib.impl("int8_dense_w8a8_gemm", prequant, "CPU")


def _check_ops_compose() -> None:
    from tests.models.qwen4_exp.test_xpu_hc_up_fused import _ref_quant
    from vllm import _custom_ops as ops
    from vllm.models.qwen4_exp.xpu import hc_down_fold
    from vllm.models.qwen4_exp.xpu import hyperconnection as hcm

    _interpreter_casts()
    ops.scaled_int8_quant = lambda x, s, z, sym: (*_ref_quant(x), None)
    calls: list = []
    lib = torch.library.Library("_xpu_C", "FRAGMENT")
    _cpu_int8_ops(lib, calls)
    eps = 1e-6
    n_embd, lowrank = 640, 320
    dim = HC * n_embd
    g = torch.Generator().manual_seed(11)
    w_down = torch.randint(-127, 128, (336, dim), dtype=torch.int8, generator=g)
    w_down[lowrank + HC :] = 0
    s_down = torch.rand(336, 1, generator=g) * 1e-3 + 1e-4
    up = (torch.randn(dim, lowrank, generator=g) * lowrank**-0.5).to(torch.bfloat16)
    for rows in (1, 8, 13):
        residual, block, injection, norm = _inputs(
            rows, n_embd, rows, False, torch.bfloat16, True
        )
        words = hc_down_fold.new_arrival_words(rows, "cpu")
        args = (norm, w_down, up, eps, HC, lowrank, False, s_down)
        ref_mix = hcm.hc_mix(residual, *args)
        ref_cm = hcm.hc_combine_and_mix(residual, block, injection, *args)
        for mode in (1, 2):
            calls.clear()
            mix = hcm.hc_mix(residual, *args, mode, words)
            cm = hcm.hc_combine_and_mix(residual, block, injection, *args, mode, words)
            assert calls == ["prequant", "prequant"], calls
            for a, b in zip((*mix, *cm), (*ref_mix, *ref_cm)):
                assert torch.equal(_bits(a), _bits(b)), (rows, mode)
            # The split ops.
            combined, xn = hcm.hc_combine_then_norm(
                residual, block, injection, norm, eps, HC
            )
            c2, xn2, xq, xs = hcm.hc_combine_then_norm_q(
                residual, block, injection, norm, eps, HC, words, mode
            )
            assert torch.equal(_bits(c2), _bits(combined))
            assert torch.equal(_bits(xn2), _bits(xn))
            xn_b, xq_b, xs_b = hcm.hc_norm_q(residual, norm, eps, HC, words, mode)
            assert torch.equal(_bits(xn_b), _bits(hcm.hc_norm(residual, norm, eps, HC)))
            for f in (hcm.hc_down_lora, hcm.hc_down_lora_q):
                ref = f(xn, w_down, HC, lowrank, False, s_down)
                got = f(xn, w_down, HC, lowrank, False, s_down, xq, xs)
                for a, b in zip(got, ref):
                    assert torch.equal(a.view(torch.int8), b.view(torch.int8)), f
            assert int(words.abs().sum()) == 0
        print(f"ops rows {rows}: folded fused and split ops equal the unfolded ones")


def _run(name: str) -> None:
    root = Path(__file__).parents[3]
    code = (
        f"from tests.models.qwen4_exp.test_xpu_hc_down_fold import {name}\n{name}()\n"
    )
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


def test_combine_norm_folds_the_quant() -> None:
    _run("_check_combine_norm_q")


def test_norm_folds_the_quant() -> None:
    _run("_check_norm_q")


def test_rule_corners() -> None:
    _run("_check_rule_corners")


def test_inv_every_bf16_amax() -> None:
    _run("_check_inv_every_bf16")


def test_folded_ops_equal_unfolded() -> None:
    _run("_check_ops_compose")


def test_switch_default_off(monkeypatch) -> None:
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_XPU_HC_DOWN_FOLD", raising=False)
    assert envs.VLLM_XPU_HC_DOWN_FOLD == 0


def test_fakes_take_the_fold_arguments() -> None:
    from vllm.models.qwen4_exp.xpu import hyperconnection as hcm

    rows, dim = 3, HC * 64
    x = torch.zeros(rows, dim, dtype=torch.bfloat16)
    words = torch.zeros(rows, dtype=torch.int64)
    xn, xq, xs = hcm._hc_norm_q_fake(x, torch.zeros(dim), 1e-6, HC, words, 1)
    assert xn.shape == x.shape and xq.dtype == torch.int8 and xs.shape == (rows, 1)
    c, xn, xq, xs = hcm._hc_combine_then_norm_q_fake(
        x, x[:, :64], x[:, :HC], torch.zeros(dim), 1e-6, HC, words, 2
    )
    assert c.shape == xn.shape == xq.shape == x.shape and xs.dtype == torch.float32
    w = torch.zeros(16, dim, dtype=torch.int8)
    lora, inj = hcm._hc_down_lora_fake(x, w, HC, 10, False, None, xq, xs)
    assert lora.shape == (rows, 10) and inj.shape == (rows, HC)


@pytest.mark.usefixtures("ref_kernel")
@pytest.mark.parametrize("mode", ["0", "1", "2"])
def test_module_routes_through_the_folded_ops(monkeypatch, mode) -> None:
    """A checkpoint INT8 merged down (CT W8A8) on the fused GEMM with an INT8 up
    linear (the split route) and the final mixer: with the switch the merged
    down module folds at load and its mix and combine-and-mix call the folded
    norm ops and hand their rows to the pre-quantized GEMM; the mixer (down
    through its layer) does not fold and says why."""
    from tests.models.qwen4_exp.test_xpu_hc import N_EMBD, TOKENS, _torch_ops
    from tests.models.qwen4_exp.test_xpu_hc_int8_up import (
        _DOWN_SHARDS,
        _MIXER_UP,
        _UP,
        _checkpoint,
        _ct_config,
        _load,
        _Target,
        _weights,
    )
    from vllm.models.qwen4_exp.xpu import hyperconnection as hcm

    if hasattr(torch.ops._xpu_C, "int8_dense_w8a8_fused_gemm"):
        pytest.skip("a build with the INT8 ops is installed")
    monkeypatch.setenv("VLLM_XPU_HC_DOWN_FOLD", mode)
    _torch_ops(monkeypatch)
    seen: list = []

    def norm_q(x, w, eps, hc, words, down_fold):
        seen.append(("norm_q", down_fold, words is not None))
        xn = hcm.grouped_gemma_rmsnorm(x, w, eps, hc)
        return (xn, *rule_ref(xn))

    def combine_then_norm_q(r, b, inj, w, eps, hc, words, down_fold):
        seen.append(("combine_then_norm_q", down_fold, words is not None))
        out, xn = hcm.hc_combine_norm(r, b, inj, w, eps, hc)
        return (out, xn, *rule_ref(xn))

    monkeypatch.setattr(hcm, "_norm_q_op", norm_q)
    monkeypatch.setattr(hcm, "_combine_then_norm_q_op", combine_then_norm_q)
    calls: list = []
    with torch.library._scoped_library("_xpu_C", "FRAGMENT") as lib:
        _cpu_int8_ops(lib, calls)
        target = _Target(_ct_config([_UP, _MIXER_UP, _DOWN_SHARDS]))
        _load(
            target,
            _checkpoint(
                _weights(torch.float32, 7),
                int8_up=True,
                int8_mixer_down=False,
                int8_down=True,
            ),
        )
        hc = target.attn_hyper_connection
        mixer = target.hyper_connection_mixer
        assert hc.down_path == "fused_int8"
        x = torch.randn(TOKENS, HC, N_EMBD)
        mixed, inject = hc.mix(x)
        residual = torch.randn(TOKENS, HC, N_EMBD)
        block = torch.randn(TOKENS, N_EMBD)
        combined, mixed2, inject2 = hc.combine_and_mix(residual, block, inject)
    if mode == "0":
        assert hc._down_fold == 0 and hc._down_fold_reason == "off"
        assert seen == [] and calls == ["fused", "fused"]
    else:
        assert hc._down_fold == int(mode) and hc._fold_words is not None
        assert seen == [
            ("norm_q", int(mode), True),
            ("combine_then_norm_q", int(mode), True),
        ]
        assert calls == ["prequant", "prequant"]
        assert mixer._down_fold == 0
        assert mixer._down_fold_reason.startswith("down projection")
    assert mixed.shape == mixed2.shape == (TOKENS, N_EMBD)
    assert combined.shape == residual.shape and inject2.shape == (TOKENS, HC)


@pytest.mark.skipif(
    not (
        torch.xpu.is_available()
        and hasattr(torch.ops, "_xpu_C")
        and hasattr(torch.ops._xpu_C, "int8_dense_w8a8_gemm")
    ),
    reason="needs an XPU card and vllm-xpu-kernels' INT8 dense ops",
)
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 13, 16, 32, 64, 128, 1024])
@pytest.mark.parametrize("mode", [1, 2])
def test_device_bits(rows: int, mode: int) -> None:
    """The card check at QFN's shapes, for each mode:
    1. ``out`` and ``xn`` bitwise against ``hc_combine_norm`` (and ``xn``
       against ``grouped_gemma_rmsnorm`` for the norm kernel);
    2. the int8 rows and scales bitwise against the IEEE rule computed on the
       CPU from the device's own ``xn``: the recompute (mode 1) and the
       re-read (mode 2) must both hold the stored bits;
    3. against the native op (``int8_dense_w8a8_fused_gemm``: its own quant,
       the same GEMM): the count of differing outputs and the max difference
       are printed (the tier-2 measure; a code moves only where the native
       division differs from IEEE, expected well under 1e-4 of codes);
    4. 1,000 launches through an XPU graph replay with fresh inputs copied in
       before each replay: every replay's rows and scales bitwise against the
       rule and the arrival words zero at the end (a missed reset or a lost
       arrival shows as a wrong or missing row)."""
    from vllm.models.qwen4_exp.nvidia.ops.hc import (
        grouped_gemma_rmsnorm,
        hc_combine_norm,
    )
    from vllm.models.qwen4_exp.xpu import hc_down_fold

    dev = "xpu"
    eps = 1e-6
    torch.manual_seed(rows)
    residual, block, injection, norm = (
        t.to(dev) if t is not None else None
        for t in _inputs(rows, QFN_EMBD, rows, False, torch.bfloat16, True)
    )

    def ndiff(a, b) -> int:
        return int((_bits(a.cpu()) != _bits(b.cpu())).sum())

    def rows_off(a, b) -> int:
        return int((_bits(a.cpu()) != _bits(b.cpu())).any(dim=1).sum())

    words = hc_down_fold.new_arrival_words(max(rows, 8192), dev)
    out_ref, y_ref = hc_combine_norm(residual, block, injection, norm, eps, HC)
    out, y, q, s = hc_down_fold.combine_norm_q(
        residual, block, injection, norm, eps, HC, words, mode
    )
    torch.xpu.synchronize()
    q_ref, s_ref = rule_ref(y.cpu())
    found = {
        "out": ndiff(out, out_ref),
        "xn": ndiff(y, y_ref),
        "xn_rows": rows_off(y, y_ref),
        "codes": int((q.cpu() != q_ref).sum()),
        "scales": ndiff(s, s_ref),
        "words": int(words.abs().sum()),
    }
    yn_ref = grouped_gemma_rmsnorm(residual, norm, eps, HC)
    yn, qn, sn = hc_down_fold.norm_q(residual, norm, eps, HC, words, mode)
    torch.xpu.synchronize()
    qn_ref, sn_ref = rule_ref(yn.cpu())
    found.update(
        norm_xn=ndiff(yn, yn_ref),
        norm_xn_rows=rows_off(yn, yn_ref),
        norm_codes=int((qn.cpu() != qn_ref).sum()),
        norm_scales=ndiff(sn, sn_ref),
        norm_words=int(words.abs().sum()),
    )
    print(f"HCDOWNFOLD-BITS rows {rows} mode {mode}: {found}")

    # Against the native op through the same GEMM.
    w = torch.randint(-127, 128, (336, HC * QFN_EMBD), dtype=torch.int8, device=dev)
    ws = torch.rand(336, 1, device=dev) * 1e-3 + 1e-4
    native = torch.ops._xpu_C.int8_dense_w8a8_fused_gemm(y, w, ws, None)
    folded = hc_down_fold.int8_down_gemm_q(q, s, w, ws)
    diff = (native.float() - folded.float()).abs()
    print(
        f"HCDOWNFOLD rows {rows} mode {mode}: outputs differing "
        f"{int((diff > 0).sum())} of {diff.numel()}, max {diff.max().item()}"
    )

    # Graph replay with fresh inputs.
    g = torch.xpu.XPUGraph()
    r_in, b_in = residual.clone(), block.clone()
    with torch.xpu.graph(g):
        _, gy, gq, gs = hc_down_fold.combine_norm_q(
            r_in, b_in, injection, norm, eps, HC, words, mode
        )
    bad = 0
    for i in range(1000):
        r_in.copy_(residual * (1 + (i % 7) * 0.125))
        b_in.copy_(block * (1 - (i % 5) * 0.1))
        g.replay()
        if i % 50 == 0 or i == 999:
            torch.xpu.synchronize()
            rq, rs = rule_ref(gy.cpu())
            bad += int((gq.cpu() != rq).sum()) + int(
                (_bits(gs.cpu()) != _bits(rs)).sum()
            )
    torch.xpu.synchronize()
    left = int(words.abs().sum())
    print(
        f"HCDOWNFOLD-REPLAY rows {rows} mode {mode}: 1000 replays, "
        f"bad {bad}, words {left}"
    )
    found.update(replay_bad=bad, replay_words=left)
    assert all(v == 0 for v in found.values()), found


@pytest.mark.skipif(not torch.xpu.is_available(), reason="needs an XPU card")
def test_device_inv_every_bf16() -> None:
    """On the card: the fold's integer ``127 / amax`` equals IEEE division on
    every positive bf16 amax; and how many of them ``tl.div_rn`` (the fold's
    division before, a plain OpFDiv on this backend) gets wrong."""
    m, ref = _ieee_inv_ref()
    m = m.to("xpu")
    inv = torch.empty(m.numel(), dtype=torch.float32, device="xpu")
    div = torch.empty_like(inv)
    _inv_of_bits[(triton.cdiv(m.numel(), 1024),)](m, inv, div, m.numel(), 1024)
    torch.xpu.synchronize()
    bad = int((inv.cpu().view(torch.int32) != ref).sum())
    div_bad = int((div.cpu().view(torch.int32) != ref).sum())
    print(
        f"HCDOWNFOLD-INV integer form differs from IEEE on {bad} of {m.numel()}; "
        f"tl.div_rn differs on {div_bad}"
    )
    assert bad == 0, bad

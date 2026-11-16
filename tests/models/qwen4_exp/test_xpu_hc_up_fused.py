# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_XPU_HC_UP_FUSED on CPU, kernels in the Triton interpreter.

1. The finish kernel with the folded quant equals ``_hc_finish_silu`` and then
   the per-token int8 rule of ``ops.scaled_int8_quant`` on XPU (the rule of
   ``xpu_ops.dynamic_per_token_int8_quant_ref``) on its stored BF16 lora: output,
   int8 rows and scales bit for bit, with zero, tiny, large and tie rows.
2. The INT8 up GEMM with the gated mix in its epilogue equals
   ``triton_scaled_mm`` and then ``hc_gate_mix``, bit for bit, at 1-64 rows,
   with per-column and scalar weight scales.
3. The two ops a folded HyperConnection calls equal the unfolded pieces.

This proves the composition and the rule; the device bits are the card check."""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from tests.models.qwen4_exp.test_xpu_hc import _no_tp  # noqa: F401
from tests.models.qwen4_exp.test_xpu_hc_int8_up import ref_kernel  # noqa: F401
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")

if HAS_TRITON:
    from vllm.models.qwen4_exp.xpu.hc_gemm import _i8_quantize, _i8_scale_bf16
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _quant_rows(x, q, s, N: tl.constexpr, BLOCK: tl.constexpr):
        """The folded quant rule alone, on BF16 rows."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        v = tl.load(x + row * N + cols, mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.where(mask, tl.abs(v), 0.0), axis=0)
        scale = _i8_scale_bf16(amax)
        tl.store(q + row * N + cols, _i8_quantize(v, scale), mask)
        tl.store(s + row, scale)


HC = 4
LOWRANK = 320
COLUMNS = 336  # the merged down GEMM: 320 lora + 4 injection + 12 padding rows
TILES = 80  # 10240 / 128


def _load_ref():
    """``xpu_ops.dynamic_per_token_int8_quant_ref`` from the tree's source (the
    rule the native op mirrors; importing ``vllm._xpu_ops`` needs the wheel)."""
    root = Path(__file__).parents[3]
    src = (root / "vllm/_xpu_ops.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "dynamic_per_token_int8_quant_ref"
        ):
            ns = {"torch": torch}
            exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
            return ns["dynamic_per_token_int8_quant_ref"]
    raise AssertionError("reference not found")


def _ref_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference rule, with the fp32 [T, 1] scales ``ops.scaled_int8_quant``
    returns on XPU."""
    q, scale, _ = _load_ref()(x, True, 8)
    return q, scale.reshape(-1, 1).to(torch.float32)


def _faithful_bf16_cast() -> None:
    """The Triton interpreter casts fp32 -> bf16 by truncation; the device and
    torch round to nearest even. Patch that one direction (every kernel, old
    and new, runs with it) so CPU results mean what device results mean."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as it

    if getattr(it, "_f2_bf16_patched", False):
        return
    orig = it._convert_float

    def conv(input, input_dtype, output_dtype, rounding_mode):
        if input_dtype == tl.float32 and output_dtype == tl.bfloat16:
            f = torch.from_numpy(np.ascontiguousarray(input).view(np.float32).copy())
            return f.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        return orig(input, input_dtype, output_dtype, rounding_mode)

    it._convert_float = conv
    it._f2_bf16_patched = True


def _check_finish_quant() -> None:
    from vllm.models.qwen4_exp.xpu import hc_gemm
    from vllm.triton_utils import triton

    _faithful_bf16_cast()
    torch.manual_seed(0)
    for rows in (1, 2, 3, 7, 8, 13, 16, 32):
        partials = torch.randn(TILES, rows, COLUMNS) * 0.05
        if rows >= 4:
            partials[:, 0] = 0  # zero row: the 1e-5 scale floor
            partials[:, 1] *= 1e-6  # tiny row
            partials[:, 2] *= 1e3  # large row
        output = torch.empty(rows, COLUMNS, dtype=torch.bfloat16)
        lora = torch.empty(rows, LOWRANK, dtype=torch.bfloat16)
        hc_gemm._hc_finish_silu[(triton.cdiv(rows * COLUMNS, 256),)](
            partials, output, lora, rows, COLUMNS, TILES, LOWRANK, HC
        )
        ref_q, ref_s = _ref_quant(lora)
        out2 = torch.empty_like(output)
        q = torch.empty(rows, LOWRANK, dtype=torch.int8)
        s = torch.empty(rows, 1, dtype=torch.float32)
        hc_gemm._hc_finish_silu_quant[(rows,)](
            partials, out2, q, s, rows, COLUMNS, TILES, LOWRANK, HC, 512
        )
        assert torch.equal(out2.view(torch.int16), output.view(torch.int16)), rows
        assert torch.equal(q, ref_q), (rows, (q != ref_q).sum())
        assert torch.equal(s.view(torch.int32), ref_s.view(torch.int32)), rows
        print(f"finish+quant rows {rows}: output, int8 rows and scales bitwise")


def _check_quant_ties() -> None:
    """The rule on rows built to sit on its corners: exact half ties (half to
    even), the bf16 rounding of the scale, the 1e-5 floor and +-127."""
    _faithful_bf16_cast()
    n = 256
    rows = []
    # scale 1/8 exactly (amax 127/8); x = (k + 0.5) / 8 is a tie for every k.
    tie = torch.tensor([(k + 0.5) / 8 for k in range(-127, 127)] + [127 / 8] * 2)
    rows.append(tie)
    rows.append(-tie)
    rows.append(torch.zeros(n))
    rows.append(torch.full((n,), 1e-7))
    torch.manual_seed(3)
    for _ in range(12):
        rows.append(torch.randn(n) * float(torch.empty(1).uniform_(-8, 8).exp()))
    x = torch.stack(rows).to(torch.bfloat16)
    q = torch.empty(x.shape, dtype=torch.int8)
    s = torch.empty(x.shape[0], 1, dtype=torch.float32)
    _quant_rows[(x.shape[0],)](x, q, s, n, n)
    ref_q, ref_s = _ref_quant(x)
    assert torch.equal(ref_s[0], torch.tensor([0.125])), ref_s[0]
    assert torch.equal(q, ref_q), (q != ref_q).sum()
    assert torch.equal(s.view(torch.int32), ref_s.view(torch.int32))
    # ties went to even: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2
    assert q[0, 127].item() == 0 and q[0, 128].item() == 2 and q[0, 129].item() == 2
    print("quant rule: ties, scale rounding, floor: bitwise with the reference")


def _check_up_mix() -> None:
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
    from vllm.models.qwen4_exp.xpu import hc_gemm

    _faithful_bf16_cast()
    torch.manual_seed(4)
    n_embd = 640
    dim = HC * n_embd
    w_nk = torch.randint(-127, 128, (dim, LOWRANK), dtype=torch.int8)
    w_kn = w_nk.t()  # the INT8 linear's [K, N] view of [N, K] rows
    for per_col in (True, False):
        w_s = (
            torch.rand(dim, 1) * 1e-3 + 1e-4
            if per_col
            else torch.tensor([[3e-4]], dtype=torch.float32)
        )
        for rows in (1, 2, 5, 8, 16, 33, 64):
            lora = (torch.randn(rows, LOWRANK) * 0.3).to(torch.bfloat16)
            q, s = _ref_quant(lora)
            xn = torch.randn(rows, dim).to(torch.bfloat16)
            gate = triton_scaled_mm(q, w_kn, s, w_s, torch.bfloat16)
            ref = hc_gate_mix(xn, gate, HC)
            out = hc_gemm.int8_up_mix(q, s, w_kn, w_s, xn, HC)
            assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
                rows,
                per_col,
            )
        print(f"up GEMM + gated mix (per-column scale {per_col}): bitwise 1-64 rows")


def _check_ops() -> None:
    """The ops a folded module calls against the unfolded pieces (fp32 down,
    which takes the stock route; the BF16 split-K route is checked above)."""
    from vllm import _custom_ops as ops
    from vllm.models.qwen4_exp.xpu import hyperconnection as hcm

    _faithful_bf16_cast()
    ops.scaled_int8_quant = lambda x, s, z, sym: (*_ref_quant(x), None)
    torch.manual_seed(5)
    n_embd = 64
    dim = HC * n_embd
    lowrank = 16
    down = torch.randn(lowrank + HC + 12, dim) * dim**-0.5
    w_kn = torch.randint(-127, 128, (dim, lowrank), dtype=torch.int8).t()
    w_s = torch.rand(dim, 1) * 1e-2
    for rows in (1, 8, 70):
        xn = torch.randn(rows, dim)
        lora, inject = hcm.hc_down_lora(xn, down, HC, lowrank)
        q, s, inject_q = hcm.hc_down_lora_q(xn, down, HC, lowrank)
        rq, rs = _ref_quant(lora)
        assert torch.equal(q, rq) and torch.equal(s, rs)
        assert torch.equal(inject_q, inject)
        xb = xn.to(torch.bfloat16)
        for fused in (True, False):
            mixed = hcm.hc_up_mix_q(q, s, w_kn, w_s, xb, HC, 0, fused)
            gate = torch.ops.vllm.int8_scaled_mm_func(
                q, w_kn, s, w_s, torch.bfloat16, None
            )
            ref = hcm.hc_gate_mix(xb, gate, HC)
            assert torch.equal(mixed.view(torch.int16), ref.view(torch.int16)), (
                rows,
                fused,
            )
    print("ops: down + quant and up + mix equal the unfolded pieces")


def _run(name: str) -> None:
    root = Path(__file__).parents[3]
    code = f"from tests.models.qwen4_exp.test_xpu_hc_up_fused import {name}\n{name}()\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "TRITON_INTERPRET": "1"},
        timeout=900,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stderr[-4000:] + result.stdout


def test_finish_kernel_folds_the_quant() -> None:
    _run("_check_finish_quant")


def test_quant_rule_corners() -> None:
    _run("_check_quant_ties")


def test_up_gemm_with_gated_mix() -> None:
    _run("_check_up_mix")


def test_folded_ops_equal_pieces() -> None:
    _run("_check_ops")


@pytest.mark.usefixtures("ref_kernel")
def test_module_routes_through_the_folded_ops(monkeypatch):
    """A HyperConnection with an INT8 up linear: with the switch it resolves the
    fold when its weights load and its mix calls the two folded ops with the
    linear's weight and scales; without it, the layer path runs (the ops' bits
    are checked above against the pieces)."""
    from tests.models.qwen4_exp.test_xpu_hc_int8_up import (
        _MIXER_UP,
        _UP,
        _checkpoint,
        _ct_config,
        _load,
        _Target,
        _weights,
    )
    from vllm.models.qwen4_exp.xpu import hyperconnection as hcm

    class _Kernel:
        def _get_layer_params(self, layer):
            return layer.weight, layer.weight_scale, None, None, None

    monkeypatch.setattr(hcm, "_up_int8_kernel", lambda layer: (1, _Kernel()))
    calls = []

    def down_q(xn, down_weight, hc, lowrank, batch_invariant, down_scale):
        calls.append(("down", tuple(xn.shape), down_scale is None))
        rows = xn.shape[0]
        q = torch.zeros(rows, lowrank, dtype=torch.int8)
        return q, torch.ones(rows, 1), xn[:, :hc]

    def up_q(q, scale, up_weight, up_scale, xn, hc, gemm, fused):
        calls.append(("up", up_weight.data_ptr(), up_scale.data_ptr(), gemm, fused))
        return xn[:, : xn.shape[1] // hc]

    monkeypatch.setattr(hcm, "_down_lora_q_op", down_q)
    monkeypatch.setattr(hcm, "_up_mix_q_op", up_q)
    monkeypatch.setattr(hcm, "_norm_op", lambda x, w, eps, hc: x)
    for mode, folded in (("0", False), ("1", True), ("2", True)):
        monkeypatch.setenv("VLLM_XPU_HC_UP_FUSED", mode)
        target = _Target(_ct_config([_UP, _MIXER_UP]))
        w = _weights(torch.float32, seed=3)
        _load(target, _checkpoint(w, True, False, False))
        for module in (target.attn_hyper_connection, target.hyper_connection_mixer):
            assert (module._up_fold is not None) == folded, (mode, module)
            if not folded:
                assert module._up_fold_reason == "off"
                continue
            calls.clear()
            x = torch.randn(3, HC, module.n_embd)
            mixed, _ = module.mix(x)
            up = module.input_mix_weight_up
            assert calls[0] == ("down", (3, HC * module.n_embd), True)
            assert calls[1] == (
                "up",
                up.weight.data_ptr(),
                up.weight_scale.data_ptr(),
                1,
                mode == "1",
            )
            assert mixed.shape == (3, module.n_embd)


def _xpu_quant_available() -> bool:
    return (
        torch.xpu.is_available()
        and hasattr(torch.ops, "_C")
        and hasattr(torch.ops._C, "dynamic_per_token_scaled_int8_quant")
    )


@pytest.mark.skipif(
    not _xpu_quant_available(), reason="needs an XPU card and the native quant op"
)
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 13, 16, 32])
def test_device_bits(rows: int) -> None:
    """The card check: on the device, the folded finish kernel against
    _hc_finish_silu and the native quant op, and the fused up GEMM + mix
    against triton_scaled_mm and hc_gate_mix, at QFN's shapes, bitwise."""
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )
    from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
    from vllm.models.qwen4_exp.xpu import hc_gemm

    dev = "xpu"
    torch.manual_seed(rows)
    for scale in (0.02, 0.05, 1.0):
        partials = (torch.randn(TILES, rows, COLUMNS) * scale).to(dev)
        output = torch.empty(rows, COLUMNS, dtype=torch.bfloat16, device=dev)
        lora = torch.empty(rows, LOWRANK, dtype=torch.bfloat16, device=dev)
        hc_gemm._hc_finish_silu[(triton.cdiv(rows * COLUMNS, 256),)](
            partials, output, lora, rows, COLUMNS, TILES, LOWRANK, HC, num_warps=4
        )
        ref_q = torch.empty(rows, LOWRANK, dtype=torch.int8, device=dev)
        ref_s = torch.empty(rows, 1, dtype=torch.float32, device=dev)
        torch.ops._C.dynamic_per_token_scaled_int8_quant(ref_q, lora, ref_s)
        out2 = torch.empty_like(output)
        q = torch.empty_like(ref_q)
        s = torch.empty_like(ref_s)
        hc_gemm._hc_finish_silu_quant[(rows,)](
            partials, out2, q, s, rows, COLUMNS, TILES, LOWRANK, HC, 512, num_warps=4
        )
        assert torch.equal(out2.view(torch.int16), output.view(torch.int16))
        assert torch.equal(s.view(torch.int32), ref_s.view(torch.int32))
        assert int((q != ref_q).sum()) == 0, int((q != ref_q).sum())

    dim = HC * 2560
    w_kn = torch.randint(-127, 128, (dim, LOWRANK), dtype=torch.int8, device=dev).t()
    w_s = torch.rand(dim, 1, device=dev) * 1e-3 + 1e-4
    xn = torch.randn(rows, dim, device=dev).to(torch.bfloat16)
    ref = hc_gate_mix(xn, triton_scaled_mm(q, w_kn, s, w_s, torch.bfloat16), HC)
    out = hc_gemm.int8_up_mix(q, s, w_kn, w_s, xn, HC)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU MoE entry switches (fused_moe/xpu_moe_entry.py), CPU only.

1. One MoE layer through the engine's runner, shared experts, INT8 linears
   and XPU W4A8 experts (xpu_moe_entry_check.py): every switch alone, all
   together, and all together with the earlier fusion switches give today's
   output bit for bit (NaN padding rows included), over 1-16 rows, EP off,
   EP 4 and EP 4 with a moved placement, with and without INT8 gates in the
   checkpoint; with the quant, GEMM and fold counts of each piece.
2. The merged gate GEMM against triton_scaled_mm per gate, bitwise.
3. VLLM_XPU_MOE_TOPK_ROUTE against today's top-k + EPLB mapping kernel
   (xpu_topk_route_check.py).

The Triton kernels run under the interpreter in subprocesses, so the
interpreter setting stays out of this session."""

import json
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(script, *args, marker):
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    run = subprocess.run(
        [sys.executable, os.path.join(HERE, script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert run.returncode == 0 and marker in run.stdout, (
        run.stdout[-3000:] + run.stderr[-6000:]
    )
    return run.stdout


# (quant calls, Triton GEMMs, merged GEMMs, folded) per layer, by code.
# INT8 gates: today router, shared gate_up, shared-expert gate, shared down
# and the routed experts each quantize; four INT8 linear GEMMs.
EXPECT_I8 = {
    "base": (5, 4, 0, False),
    "entry": (2, 4, 0, False),
    "fold": (5, 4, 0, True),
    "merged": (4, 2, 1, False),
    "all": (2, 2, 1, True),
    "all+earlier": (2, 2, 1, True),
    "fold2": (5, 4, 0, True),
    "merged2": (4, 2, 1, False),
    "foldsig": (5, 4, 0, True),
    "foldsig2": (5, 4, 0, True),
}
# BF16 gates: gate_up, down and the routed experts quantize; two GEMMs.
EXPECT_BF = {
    "base": (3, 2, 0, False),
    "entry": (2, 2, 0, False),
    "fold": (3, 2, 0, True),
    "merged": (3, 2, 0, False),
    "all": (2, 2, 0, True),
    "all+earlier": (2, 2, 0, True),
    "fold2": (3, 2, 0, True),
    "merged2": (3, 2, 0, False),
    "foldsig": (3, 2, 0, True),
    "foldsig2": (3, 2, 0, True),
}
# Different bits by contract (tier 2): compared with a bound, not bitwise.
TIER2 = ("fold2", "foldsig", "foldsig2")
WIDE = ("r33", "r64", "r65", "r128")


def _bf16(bits):
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)


def test_moe_entry_layer_bit_identical(tmp_path):
    out = tmp_path / "entry.json"
    _run("xpu_moe_entry_check.py", str(out), marker="XPU-MOE-ENTRY-OK")
    with open(out) as f:
        res = json.load(f)
    tier2_gates = res.pop("merged2-vs-onednn")
    cases = sorted({k.split("|")[0] for k in res})
    # 2 checkpoints x 3 EP layouts x 7 row counts, + 4 merged-gate shapes
    assert len(cases) == 2 * 3 * 7 + 4, cases
    t2: dict[str, list] = {name: [] for name in TIER2}
    for case in cases:
        base = res[f"{case}|base"]
        expect = EXPECT_I8 if case.startswith("gi8") else EXPECT_BF
        for name in expect:
            key = f"{case}|{name}"
            if key not in res:
                continue
            got = res[key]
            if case.endswith(WIDE):
                continue  # counted and compared below
            if name not in TIER2:
                assert got["bits"] == base["bits"], key
            else:
                a, b = _bf16(base["bits"]).float(), _bf16(got["bits"]).float()
                fin = torch.isfinite(a)
                assert torch.equal(fin, torch.isfinite(b)), key
                d = (a[fin] - b[fin]).abs()
                rel = d / a[fin].abs().clamp_min(1e-3)
                t2[name].append((rel.max().item(), rel.mean().item()))
            q, g, gm, folded = expect[name]
            assert (got["quant"], got["gemm"], got["gates_mm"], got["folded"]) == (
                q,
                g,
                gm,
                folded,
            ), (key, got)
            if name in ("entry", "all", "all+earlier"):
                readers = " ".join(got["readers"])
                assert "W4A8 routed experts" in readers, (key, readers)
                assert "shared_expert.gate_up_proj" in readers, (key, readers)
                if case.startswith("gi8") and name == "entry":
                    assert "gate" in got["readers"], (key, readers)
                    assert "shared_expert_gate" in got["readers"], (key, readers)
    # 33 and 64 rows: the merged GEMM runs under =1 and =2 (Triton route of
    # the INT8 linear), bits equal today's.
    for m in (33, 64):
        base = res[f"gi8-ep1-r{m}|base"]
        for name in ("merged", "merged2", "all"):
            got = res[f"gi8-ep1-r{m}|{name}"]
            assert got["bits"] == base["bits"], (m, name)
            assert got["gates_mm"] == 1 and got["gemm"] == 2, (m, name, got)
    # 65 and 128 rows (the gates' oneDNN route): =1 keeps both gates' own
    # GEMMs (today's bits); =2 runs the merged Triton GEMM in their place.
    for m in (65, 128):
        base = res[f"gi8-ep1-r{m}|base"]
        merged = res[f"gi8-ep1-r{m}|merged"]
        merged2 = res[f"gi8-ep1-r{m}|merged2"]
        assert merged["bits"] == base["bits"] and merged["gates_mm"] == 0
        assert merged["onednn"] == base["onednn"]
        assert merged2["gates_mm"] == 1, merged2
        assert merged2["onednn"] == base["onednn"] - 2, (merged2, base)
        a, b = _bf16(base["bits"]).float(), _bf16(merged2["bits"]).float()
        assert torch.isfinite(b).all()
        rel = (a - b).norm() / a.norm()
        print(f"TIER2 merged2 r{m}: layer rel L2 {rel:.3e}")
    # The gate GEMMs themselves: within one bf16 ulp of the oneDNN stand-in.
    assert tier2_gates["worst_ulp"] <= 1, tier2_gates
    print(
        "TIER2 merged2 gates vs oneDNN stand-in: "
        f"{tier2_gates['differ']} of {tier2_gates['n']} differ, worst 1 ulp"
    )
    for name, rows in t2.items():
        mx, mean = max(r[0] for r in rows), sum(r[1] for r in rows) / len(rows)
        print(f"TIER2 {name}: cases {len(rows)} max_rel {mx:.3e} mean_rel {mean:.3e}")
        # The gather's sigmoid moves the gate by at most one bf16 ulp.
        if name == "foldsig":
            assert mean < 1e-3, (name, mean)


_GATES_CHECK = r"""
import torch
from vllm.model_executor.layers.fused_moe.xpu_moe_entry import gates_mm
from vllm.model_executor.layers.quantization.compressed_tensors import (
    triton_scaled_mm as tsm,
)
g = torch.Generator().manual_seed(1)
bf = torch.bfloat16
n = 0
for K, N1 in ((256, 64), (2560, 512)):
    for per_tensor_gate in (True, False):
        wr = torch.randint(-127, 128, (N1, K), dtype=torch.int8, generator=g)
        wg = torch.randint(-127, 128, (1, K), dtype=torch.int8, generator=g)
        sr = (torch.rand((N1,), generator=g) * 0.004 + 0.001).float()
        if per_tensor_gate:
            sg = torch.full((1,), 0.0031)
        else:
            sg = torch.rand((1,), generator=g) * 0.01
        for M in (1, 2, 3, 5, 8, 13, 16, 33, 64):
            if K == 2560 and M not in (1, 8, 16):
                continue
            x = torch.randint(-127, 128, (M, K), dtype=torch.int8, generator=g)
            xs = (torch.rand((M, 1), generator=g) * 0.02 + 1e-4).float()
            r_ref = tsm.triton_scaled_mm(x, wr.t(), xs, sr.view(-1, 1), bf)
            g_ref = tsm.triton_scaled_mm(x, wg.t(), xs, sg.view(1, 1), bf)
            w_cat = torch.cat([wr, wg])
            s_cat = torch.cat([sr, sg.reshape(-1)])
            r, gg = gates_mm(x, xs, w_cat, s_cat, N1, bf)
            assert r.is_contiguous() and gg.is_contiguous()
            assert torch.equal(r.view(torch.int16), r_ref.view(torch.int16))
            assert torch.equal(gg.view(torch.int16), g_ref.view(torch.int16))
            n += 1
print("GATES-MM-OK", n)
"""


def test_merged_gates_gemm_bitwise(tmp_path):
    script = tmp_path / "gates_check.py"
    script.write_text(_GATES_CHECK)
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    run = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert run.returncode == 0 and "GATES-MM-OK" in run.stdout, run.stderr[-4000:]


def test_topk_route_matches_topk_and_eplb_map():
    out = _run("xpu_topk_route_check.py", marker="XPU-TOPK-ROUTE-OK")
    n = int(out.split("XPU-TOPK-ROUTE-OK")[1].split()[0])
    assert n == 3 * 8 * 2 * 2 * 3, n


def test_switches_default_off():
    import vllm.envs as envs

    for name in (
        "VLLM_XPU_MOE_ENTRY_QUANT",
        "VLLM_XPU_MOE_SHARED_IN_GATHER",
        "VLLM_XPU_MOE_GATES_MERGED",
        "VLLM_XPU_MOE_TOPK_ROUTE",
        "VLLM_XPU_MOE_SHARED_GATE_IN_GATHER",
    ):
        if name in os.environ:
            pytest.skip(f"{name} set in the environment")
        assert not getattr(envs, name), name


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="XPU only"
)
@pytest.mark.parametrize("m", [1, 2, 4, 8, 13, 16, 33, 64])
def test_merged_gates_gemm_device_bits(m):
    """The card half of test_merged_gates_gemm_bitwise at QFN's shapes:
    router 512 x 2560 per-channel, shared-expert gate 1 x 2560."""
    from vllm.model_executor.layers.fused_moe.xpu_moe_entry import gates_mm
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        triton_scaled_mm as tsm,
    )

    g = torch.Generator().manual_seed(m)
    k, n1 = 2560, 512
    wr = torch.randint(-127, 128, (n1, k), dtype=torch.int8, generator=g)
    wg = torch.randint(-127, 128, (1, k), dtype=torch.int8, generator=g)
    sr = (torch.rand((n1,), generator=g) * 0.004 + 0.001).float()
    sg = torch.full((1,), 0.0031)
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, generator=g)
    xs = (torch.rand((m, 1), generator=g) * 0.02 + 1e-4).float()
    wr, wg, sr, sg, x, xs = (t.to("xpu") for t in (wr, wg, sr, sg, x, xs))
    bf = torch.bfloat16
    r_ref = tsm.triton_scaled_mm(x, wr.t(), xs, sr.view(-1, 1), bf)
    g_ref = tsm.triton_scaled_mm(x, wg.t(), xs, sg.view(1, 1), bf)
    r, gg = gates_mm(x, xs, torch.cat([wr, wg]), torch.cat([sr, sg]), n1, bf)
    assert torch.equal(r.view(torch.int16), r_ref.view(torch.int16))
    assert torch.equal(gg.view(torch.int16), g_ref.view(torch.int16))


@pytest.mark.skipif(
    not (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and hasattr(torch.ops, "_xpu_C")
        and hasattr(torch.ops._xpu_C, "int8_gemm_w8a8")
    ),
    reason="XPU with vllm-xpu-kernels' oneDNN INT8 GEMM only",
)
@pytest.mark.parametrize("m", [65, 96, 128, 256, 1024])
def test_merged_gates_vs_onednn_device(m):
    """VLLM_XPU_MOE_GATES_MERGED=2 on the card, QFN's gate shapes, above the
    oneDNN split: the merged Triton GEMM against each gate's own oneDNN GEMM.
    Prints how many elements differ; asserts at most one bf16 ulp. If none
    differ at every row count, =2 is bit-equal there and can be called
    tier 1."""
    from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
        _xpu_int8_scaled_mm_func,
    )
    from vllm.model_executor.layers.fused_moe.xpu_moe_entry import gates_mm

    g = torch.Generator().manual_seed(m)
    k, n1 = 2560, 512
    wr = torch.randint(-127, 128, (n1, k), dtype=torch.int8, generator=g)
    wg = torch.randint(-127, 128, (1, k), dtype=torch.int8, generator=g)
    sr = (torch.rand((n1,), generator=g) * 0.004 + 0.001).float()
    sg = torch.full((1,), 0.0031)
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, generator=g)
    xs = (torch.rand((m, 1), generator=g) * 0.02 + 1e-4).float()
    wr, wg, sr, sg, x, xs = (t.to("xpu") for t in (wr, wg, sr, sg, x, xs))
    bf = torch.bfloat16
    r, gg = gates_mm(x, xs, torch.cat([wr, wg]), torch.cat([sr, sg]), n1, bf)
    differ = 0
    for got, w, s in ((r, wr, sr), (gg, wg, sg)):
        ref = _xpu_int8_scaled_mm_func(x, w.t(), xs, s, bf)
        ulp = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
        differ += int((ulp != 0).sum())
        assert int(ulp.max()) <= 1, int(ulp.max())
    print(f"MERGED2-VS-ONEDNN rows {m}: {differ} of {m * (n1 + 1)} differ")

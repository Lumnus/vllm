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
}


def _bf16(bits):
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)


def test_moe_entry_layer_bit_identical(tmp_path):
    out = tmp_path / "entry.json"
    _run("xpu_moe_entry_check.py", str(out), marker="XPU-MOE-ENTRY-OK")
    with open(out) as f:
        res = json.load(f)
    cases = sorted({k.split("|")[0] for k in res})
    # 2 checkpoints x 3 EP layouts x 7 row counts, + 2 merged-gate shapes
    assert len(cases) == 2 * 3 * 7 + 2, cases
    t2 = []
    for case in cases:
        base = res[f"{case}|base"]
        expect = EXPECT_I8 if case.startswith("gi8") else EXPECT_BF
        for name in expect:
            key = f"{case}|{name}"
            if key not in res:
                continue
            got = res[key]
            if name != "fold2":
                assert got["bits"] == base["bits"], key
            else:
                a, b = _bf16(base["bits"]).float(), _bf16(got["bits"]).float()
                fin = torch.isfinite(a)
                assert torch.equal(fin, torch.isfinite(b)), key
                d = (a[fin] - b[fin]).abs()
                rel = d / a[fin].abs().clamp_min(1e-3)
                t2.append((rel.max().item(), rel.mean().item()))
            q, g, gm, folded = expect[name]
            if case.endswith(("r33", "r64")):
                continue  # counted below
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
    # 33 and 64 rows: the merged GEMM still runs (Triton route of the INT8
    # linear), bits as above.
    for m in (33, 64):
        assert res[f"gi8-ep1-r{m}|merged"]["gates_mm"] == 1
        assert res[f"gi8-ep1-r{m}|merged"]["gemm"] == 2
    mx, mean = max(r[0] for r in t2), sum(r[1] for r in t2) / len(t2)
    print(f"TIER2 fold2: cases {len(t2)} max_rel {mx:.3e} mean_rel {mean:.3e}")


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


def test_switches_default_off():
    import vllm.envs as envs

    for name in (
        "VLLM_XPU_MOE_ENTRY_QUANT",
        "VLLM_XPU_MOE_SHARED_IN_GATHER",
        "VLLM_XPU_MOE_GATES_MERGED",
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

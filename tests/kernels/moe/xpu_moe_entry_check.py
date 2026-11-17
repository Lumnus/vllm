# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU check of the XPU MoE entry (run by test_xpu_moe_entry.py in a
subprocess under the Triton interpreter).

One MoE layer of the engine's own code: MoERunner.forward, the real
SharedExperts around a real Qwen3MoeMLP (gate_up, SiLU-and-mul, down, the
sigmoid expert gate), the router gate, and XPUExpertsW4A8Int.apply; the
INT8 linears are the engine's XPUInt8ScaledMMLinearKernel / Triton kernel
(Triton GEMM interpreted). Stand-ins: the per-token int8 quant op (one
function, counted), the router's softmax top-k, and the vllm-xpu-kernels
W4A8 apply (int8 rows -> per-expert int8 GEMM, the gather's weighted fp32
sum, and with ``shared`` the kernel's formula
``T(float(T(acc)) + float(T(gate * shared)))``; it quantizes the token rows
itself unless given ``x_q``).

For every case it runs the layer with all switches off (today's path) and
under each switch alone, all together, and all together with the earlier
fusion switches, and writes per run the output bits, the quant calls, the
GEMM calls, whether the shared term was added in the gather and which
consumers read the entry rows."""

import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

SWITCHES = (
    "VLLM_XPU_MOE_ENTRY_QUANT",
    "VLLM_XPU_MOE_SHARED_IN_GATHER",
    "VLLM_XPU_MOE_GATES_MERGED",
)
TIER2_SWITCHES = ("VLLM_XPU_MOE_SHARED_GATE_IN_GATHER",)
EARLIER = {
    "VLLM_XPU_HC_UP_FUSED": "1",
    "VLLM_XPU_MOE_ROUTE_FUSED": "1",
    "VLLM_XPU_ATTN_META_ONCE": "1",
    "VLLM_XPU_W4A8_FUSED_MOE": "1",
}
VARIANTS = {
    "base": {},
    "entry": {"VLLM_XPU_MOE_ENTRY_QUANT": "1"},
    "fold": {"VLLM_XPU_MOE_SHARED_IN_GATHER": "1"},
    "merged": {"VLLM_XPU_MOE_GATES_MERGED": "1"},
    "all": {s: "1" for s in SWITCHES},
    "all+earlier": {**{s: "1" for s in SWITCHES}, **EARLIER},
    "fold2": {"VLLM_XPU_MOE_SHARED_IN_GATHER": "2"},
    "merged2": {"VLLM_XPU_MOE_GATES_MERGED": "2"},
    "foldsig": {
        "VLLM_XPU_MOE_SHARED_IN_GATHER": "1",
        "VLLM_XPU_MOE_SHARED_GATE_IN_GATHER": "1",
    },
    "foldsig2": {
        "VLLM_XPU_MOE_SHARED_IN_GATHER": "2",
        "VLLM_XPU_MOE_SHARED_GATE_IN_GATHER": "1",
    },
}
ALL_KEYS = (*SWITCHES, *TIER2_SWITCHES, *EARLIER)


def _onednn_lib(counts):
    """A CPU stand-in for vllm-xpu-kernels' oneDNN INT8 GEMM (the route above
    VLLM_XPU_INT8_ONEDNN_MIN_ROWS): the exact int32 dot, then the product of
    the two fp32 scales applied once (one order oneDNN may use; Triton's is
    (scale_a * acc) * scale_b), rounded to the output dtype."""
    lib = torch.library.Library("_xpu_C", "FRAGMENT")
    lib.define(
        "int8_gemm_w8a8(Tensor x, Tensor x_scale, Tensor? x_zp, Tensor w, "
        "Tensor w_scale, Tensor? w_zp, int group_size, Tensor? bias, "
        "ScalarType out_dtype) -> Tensor"
    )

    def impl(x, xs, xz, w, ws, wz, group, bias, out_dtype):
        counts["onednn"] += 1
        acc = (x.long() @ w.long()).float()
        scale = xs.float().reshape(-1, 1) * ws.float().reshape(1, -1)
        return (acc * scale).to(out_dtype)

    lib.impl("int8_gemm_w8a8", impl, "CPU")
    return lib


def main() -> None:
    for k in ALL_KEYS:
        os.environ.pop(k, None)

    import vllm._custom_ops as ops
    from vllm.model_executor.kernels.linear.scaled_mm import triton as ktriton
    from vllm.model_executor.kernels.linear.scaled_mm import xpu as kxpu
    from vllm.model_executor.layers.fused_moe import xpu_moe_entry
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP

    gen = torch.Generator().manual_seed(7)
    counts = {"quant": 0, "gemm": 0, "gates_mm": 0, "onednn": 0}
    onednn = _onednn_lib(counts)  # noqa: F841 (kept alive for the run)

    # ---- the per-token int8 quant op (one rule, counted) -----------------
    def quant(x, scale=None, azp=None, symmetric=True):
        assert scale is None and symmetric
        counts["quant"] += 1
        xf = x.float()
        s = (xf.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-5)
        s = s.to(torch.bfloat16).float()
        q = torch.round(xf / s).clamp(-128, 127).to(torch.int8)
        return q, s, None

    ops.scaled_int8_quant = quant
    xpu_moe_entry._TEST_NATIVE = True

    real_tsm = ktriton.triton_scaled_mm

    def tsm(*a, **k):
        counts["gemm"] += 1
        return real_tsm(*a, **k)

    ktriton.triton_scaled_mm = tsm
    kxpu.triton_scaled_mm = tsm
    real_gates = xpu_moe_entry.gates_mm

    def gates(*a, **k):
        counts["gates_mm"] += 1
        return real_gates(*a, **k)

    xpu_moe_entry.gates_mm = gates

    # ---- INT8 linears ------------------------------------------------------
    names = ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj")

    class Int8Linear(torch.nn.Module):
        def __init__(self, k, n, prefix, per_tensor=False):
            super().__init__()
            self.prefix, self.input_size, self.output_size = prefix, k, n
            w = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=gen)
            self.weight = w.t()  # the kernel's [K, N] view after load
            if per_tensor:
                ws = torch.full((1,), 0.003, dtype=torch.float32)
            else:
                ws = (torch.rand((n, 1), generator=gen) * 0.004 + 0.001).float()
            self.weight_scale = ws
            self.bias = None
            kern = object.__new__(kxpu.XPUInt8ScaledMMLinearKernel)
            kern.config = SimpleNamespace(
                is_static_input_scheme=False, input_symmetric=True
            )
            kern.layer_param_names = names
            self.scheme = SimpleNamespace(kernel=kern)

        def forward(self, x):
            return self.scheme.kernel.apply_weights(self, x, None), None

    class Bf16Linear(torch.nn.Module):
        def __init__(self, k, n, prefix):
            super().__init__()
            self.prefix, self.input_size, self.output_size = prefix, k, n
            self.weight = (torch.randn((n, k), generator=gen) / 16).to(torch.bfloat16)
            self.bias = None

        def forward(self, x):
            return F.linear(x, self.weight), None

    class SiluMul(torch.nn.Module):
        def forward(self, x):
            d = x.shape[-1] // 2
            return F.silu(x[..., :d]) * x[..., d:]

    # ---- the W4A8 apply stand-in ---------------------------------------------
    class Impl:
        def __init__(self, n_local, ep_rank, total, hidden, emap):
            self.num_experts = n_local
            self.total_experts_num = total
            self.ep_rank = ep_rank
            self.expert_map = emap
            self.w = torch.randint(
                -7, 8, (n_local, hidden, hidden), dtype=torch.int8, generator=gen
            )
            self.ws = (torch.rand((n_local, hidden), generator=gen) / 64).float()
            self._w4a8_why = None

        def local(self, g):
            if g < 0 or g >= self.total_experts_num:
                return -1
            e = g if self.expert_map is None else int(self.expert_map[g])
            return e if 0 <= e < self.num_experts else -1

        def apply(
            self,
            output,
            hidden_states,
            topk_weights,
            topk_ids,
            a1q_scale=None,
            w4a8=False,
            ids_valid=False,
            rows_per_expert=None,
            x_q=None,
            shared=None,
        ):
            assert w4a8
            if x_q is None:
                q, s, _ = quant(hidden_states)
            else:
                q, s = x_q
            s = s.reshape(-1)
            for t in range(hidden_states.shape[0]):
                acc = torch.zeros(hidden_states.shape[1], dtype=torch.float32)
                for k in range(topk_ids.shape[1]):
                    e = self.local(int(topk_ids[t, k]))
                    if e < 0:
                        continue
                    y = (q[t].int() @ self.w[e].int().t()).float()
                    y = (y * s[t] * self.ws[e]).to(output.dtype)
                    acc += y.float() * topk_weights[t, k].float()
                if shared is None:
                    output[t] = acc.to(output.dtype)
                    continue
                gate = shared[1].reshape(-1)[t].float()
                if len(shared) > 3 and shared[3]:
                    # The gather's sigmoid: fp32 1 / (1 + exp(-x)), rounded.
                    gate = (1.0 / (1.0 + torch.exp(-gate))).to(output.dtype).float()
                srow = shared[0][t].float()
                if len(shared) > 2 and shared[2]:
                    output[t] = (acc + gate * srow).to(output.dtype)
                else:
                    routed = acc.to(output.dtype).float()
                    gated = (gate * srow).to(output.dtype).float()
                    output[t] = (routed + gated).to(output.dtype)
            return shared is not None

    seen_rows = []
    real_open = xpu_moe_entry.EntryPlan.open

    def open_and_keep(self, runner, x):
        cm = real_open(self, runner, x)

        class Keep:
            def __enter__(self_):
                rows = cm.__enter__()
                seen_rows.append(rows)
                return rows

            def __exit__(self_, *exc):
                return cm.__exit__(*exc)

        return Keep()

    xpu_moe_entry.EntryPlan.open = open_and_keep
    xpu_moe._apply_takes_entry = lambda: True
    xpu_moe_entry.kernels_take = lambda name: True

    # ---- one layer ---------------------------------------------------------
    def build(int8_gates, ep, moved, n_global, hidden, inter, topk):
        n_local = n_global // ep
        emap = None
        if ep > 1:
            emap = torch.full((n_global,), -1, dtype=torch.int32)
            owned = (
                torch.randperm(n_global, generator=gen)[:n_local]
                if moved
                else torch.arange(n_local)
            )
            emap[owned] = torch.arange(n_local, dtype=torch.int32)
        mlp = object.__new__(Qwen3MoeMLP)
        torch.nn.Module.__init__(mlp)
        mlp.gate_up_proj = Int8Linear(hidden, 2 * inter, "shared_expert.gate_up_proj")
        mlp.down_proj = Int8Linear(inter, hidden, "shared_expert.down_proj")
        mlp.act_fn = SiluMul()
        if int8_gates:
            mlp.expert_gate = Int8Linear(hidden, 1, "shared_expert_gate", True)
            gate = Int8Linear(hidden, n_global, "gate")
        else:
            mlp.expert_gate = Bf16Linear(hidden, 1, "shared_expert_gate")
            gate = Bf16Linear(hidden, n_global, "gate")

        pcfg = SimpleNamespace(
            enable_eplb=False,
            all2all_backend="",
            use_fi_nvl_two_sided_kernels=False,
            use_all2all_kernels=False,
        )
        cfg = SimpleNamespace(
            is_sequence_parallel=False,
            pcp_size=1,
            dp_size=1,
            tp_size=1,
            ep_size=ep,
            ep_rank=0,
            hidden_dim=hidden,
            hidden_dim_unpadded=hidden,
            skip_final_all_reduce=ep > 1,
            moe_parallel_config=pcfg,
            use_ep=ep > 1,
            sp_size=1,
        )
        ex = object.__new__(xpu_moe.XPUExpertsW4A8Int)
        ex.tiers = None
        ex.fused_moe_impl = Impl(n_local, 0, n_global, hidden, emap)
        ex.moe_config = cfg
        ex.ep_contract = xpu_moe.EPIdsContract.OP_HANDLES_IDS
        ex.combine_mode = "split"
        ex.combine_explicit = False
        ex.placement_trace = None

        def forward_modular(
            x, topk_weights, topk_ids, shared_experts, shared_experts_input
        ):
            out = torch.empty_like(x)
            ex.apply(
                out,
                x,
                None,
                None,
                topk_weights,
                topk_ids,
                MoEActivation.SILU,
                n_global,
                emap,
                None,
                None,
                None,
                None,
                None,
                False,
            )
            return out

        qm = SimpleNamespace(
            is_monolithic=False,
            topk_indices_dtype=None,
            skip_forward_padding=True,
            has_unpadded_output=False,
            supports_internal_mk=True,
            mk_can_overlap_shared_experts=False,
            moe_kernel=SimpleNamespace(
                output_is_reduced=lambda: False,
                impl=SimpleNamespace(fused_experts=ex),
                can_overlap_shared_experts=False,
            ),
        )

        def select_experts(
            hidden_states, router_logits, topk_indices_dtype, input_ids=None
        ):
            p = router_logits.float().softmax(-1)
            w, ids = torch.topk(p, topk, dim=-1)
            w = w / w.sum(-1, keepdim=True)
            ids = ids.to(torch.int32)
            ids[::3, 1] = -1  # unrouted slots
            return w, ids

        r = object.__new__(MoERunner)
        torch.nn.Module.__init__(r)
        r.moe_config = cfg
        r.router = SimpleNamespace(
            select_experts=select_experts,
            residency_mask_active=False,
            masks_padding=True,
        )
        r.routed_input_transform = None
        r.routed_output_transform = None
        r.routed_scaling_factor = 1.0
        r.divide_shared_output_in_fp16 = False
        r.gate = gate
        r.shared_expert_gate = None
        r.routed_experts = SimpleNamespace(
            quant_method=qm,
            _ensure_moe_quant_config_init=lambda: None,
            forward_modular=forward_modular,
            supports_precomputed_routing=False,
        )
        r.enable_dbo = False
        r._fse_fuse_gate = False
        r._combined_gate_weight = None
        r._shared_experts = SharedExperts(
            mlp,
            moe_config=cfg,
            enable_dbo=False,
            mk_can_overlap_shared_experts=lambda: False,
        )
        r.layer_name = "layers.0.mlp.experts"
        r._xpu_entry = None
        r._xpu_shared_folded = False
        r._sequence_parallel_context = lambda: __import__("contextlib").nullcontext()
        return r

    def run(r, x):
        r._resolve_xpu_entry(force_platform=True)
        r._forward_entry = lambda hs, rl, sei, ids, name, hdu: r._forward_impl(
            hs, rl, sei, ids
        )
        for k in counts:
            counts[k] = 0
        seen_rows.clear()
        out = r.forward(x, x)
        rows = seen_rows[-1] if seen_rows else None
        info = {
            "folded": bool(rows is not None and rows.folded),
            "readers": [] if rows is None else list(rows.readers),
        }
        return out, dict(counts), info, r._xpu_entry

    results = {}
    shapes = (1, 2, 4, 7, 8, 13, 16)
    for int8_gates in (True, False):
        for ep, moved in ((1, False), (4, False), (4, True)):
            for m in shapes:
                tag = (
                    f"g{'i8' if int8_gates else 'bf'}-ep{ep}{'m' if moved else ''}-r{m}"
                )
                torch.manual_seed(0)
                r = build(int8_gates, ep, moved, 16, 64, 32, 4)
                x = (torch.randn((m + 2, 64), generator=gen) / 4).to(torch.bfloat16)
                x[m:] = float("nan")  # padding rows
                for name, env in VARIANTS.items():
                    for k in ALL_KEYS:
                        os.environ.pop(k, None)
                    os.environ.update(env)
                    out, c, info, plan = run(r, x)
                    results[f"{tag}|{name}"] = {
                        "bits": out.view(torch.int16).tolist(),
                        "quant": c["quant"],
                        "gemm": c["gemm"],
                        "gates_mm": c["gates_mm"],
                        **info,
                        "plan": None
                        if plan is None
                        else [
                            plan.quant,
                            plan.gates is not None,
                            plan.fold_module is not None,
                            plan.single_round,
                            plan.gate_logits,
                        ],
                    }
    # Merged gates around the Triton route's rows: 33 and 64 (Triton), 65 and
    # 128 (oneDNN stand-in): =1 keeps the gates' own GEMMs above 64, =2 runs
    # the merged Triton GEMM there.
    for m in (33, 64, 65, 128):
        r = build(True, 1, False, 16, 64, 32, 4)
        x = (torch.randn((m, 64), generator=gen) / 4).to(torch.bfloat16)
        for name in ("base", "merged", "merged2", "all"):
            for k in ALL_KEYS:
                os.environ.pop(k, None)
            os.environ.update(VARIANTS[name])
            out, c, info, plan = run(r, x)
            results[f"gi8-ep1-r{m}|{name}"] = {
                "bits": out.view(torch.int16).tolist(),
                "quant": c["quant"],
                "gemm": c["gemm"],
                "gates_mm": c["gates_mm"],
                "onednn": c["onednn"],
                **info,
                "plan": None,
            }
    # The merged Triton GEMM against the oneDNN stand-in per gate, 65-256
    # rows, router 512 x 2560 and the 1-column gate: the tier-2 distance.
    from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
        _xpu_int8_scaled_mm_func,
    )

    worst, differ, total = 0, 0, 0
    k, n1 = 256, 64
    wr = torch.randint(-127, 128, (n1, k), dtype=torch.int8, generator=gen)
    wg = torch.randint(-127, 128, (1, k), dtype=torch.int8, generator=gen)
    sr = (torch.rand((n1,), generator=gen) * 0.004 + 0.001).float()
    sg = torch.full((1,), 0.0031)
    for m in (65, 96, 128, 256):
        xq = torch.randint(-127, 128, (m, k), dtype=torch.int8, generator=gen)
        xs = (torch.rand((m, 1), generator=gen) * 0.02 + 1e-4).float()
        r_m, g_m = real_gates(
            xq, xs, torch.cat([wr, wg]), torch.cat([sr, sg]), n1, torch.bfloat16
        )
        for got, w, s in ((r_m, wr, sr), (g_m, wg, sg)):
            ref = _xpu_int8_scaled_mm_func(xq, w.t(), xs, s, torch.bfloat16)
            a = got.float().view(-1)
            b = ref.float().view(-1)
            ulp = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
            same_sign = (a >= 0) == (b >= 0)
            worst = max(worst, int(ulp[same_sign].max()) if same_sign.any() else 0)
            differ += int((ulp != 0).sum())
            total += ulp.numel()
    results["merged2-vs-onednn"] = {"worst_ulp": worst, "differ": differ, "n": total}
    with open(sys.argv[1], "w") as f:
        json.dump(results, f)
    print("XPU-MOE-ENTRY-OK", len(results))


if __name__ == "__main__":
    main()

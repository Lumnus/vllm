# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70 0040 on the device: VRAM experts + pinned-host experts (read through UVA
views carved from the 1 GiB arena) sum to exactly the all-in-VRAM result."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="needs an XPU")

E_LOCAL, E_GLOBAL, TOPK = 256, 512, 10
N, K, G = 640, 2560, 128  # Qwen3.8-Flash-Next routed expert, int4 g128, EP 2


def _moe(w13, w13_s, w2, w2_s, emap, n_exp):
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

    for w in (w13, w2):
        w.xpu_fused_moe = True  # already signed int4
    m = XpuFusedMoe(w13=w13, w13_scales=w13_s, w13_bias=None, w2=w2, w2_scales=w2_s,
                    w2_bias=None, n_experts_per_token=TOPK, activation="silu",
                    num_experts=n_exp, ep_rank=0, ep_size=2, expert_map=emap)
    m.w13, m.w2 = w13.view(torch.int8), w2.view(torch.int8)
    return m


def _map(slots: dict[int, int]) -> torch.Tensor:
    m = torch.full((E_GLOBAL,), -1, dtype=torch.int32)
    for g, s in slots.items():
        m[g] = s
    return m.to("xpu")


@pytest.mark.parametrize("rows", [1, 7, 64, 1024])
def test_hot_plus_cold_equals_all_in_vram(rows):
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    from vllm.model_executor.layers.fused_moe.b70_hotcold import _PinnedArena
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    torch.manual_seed(rows)
    w13 = implement_zp(torch.randint(0, 256, (E_LOCAL, 2 * N, K // 2), dtype=torch.uint8))
    w2 = implement_zp(torch.randint(0, 256, (E_LOCAL, K, N // 2), dtype=torch.uint8))
    w13_s = (torch.rand(E_LOCAL, 2 * N, K // G) * 0.002 + 0.001).half()
    w2_s = (torch.rand(E_LOCAL, K, N // G) * 0.002 + 0.001).half()
    perm = torch.randperm(E_LOCAL)
    hot_ids, cold_ids = perm[:150].sort().values, perm[150:].sort().values

    ref = _moe(w13.xpu(), w13_s.xpu(), w2.xpu(), w2_s.xpu(),
               _map({g: g for g in range(E_LOCAL)}), E_LOCAL)
    hot = _moe(w13[hot_ids].xpu(), w13_s[hot_ids].xpu(), w2[hot_ids].xpu(),
               w2_s[hot_ids].xpu(), _map({int(g): i for i, g in enumerate(hot_ids)}),
               len(hot_ids))
    arena = _PinnedArena()
    views = []
    for t in (w13, w13_s, w2, w2_s):
        pinned = arena.empty(t[cold_ids].shape, t.dtype)
        pinned.copy_(t[cold_ids])
        views.append(get_accelerator_view_from_cpu_tensor(pinned))
    cold = _moe(*views, _map({int(g): i for i, g in enumerate(cold_ids)}), len(cold_ids))

    x = torch.randn(rows, K, device="xpu", dtype=torch.bfloat16) * 0.5
    ti = torch.stack([torch.randperm(E_GLOBAL)[:TOPK] for _ in range(rows)]).to("xpu", torch.int32)
    tw = torch.softmax(torch.randn(rows, TOPK, device="xpu"), -1)
    outs = []
    for m in (ref, hot, cold):
        o = torch.full_like(x, 7.0)  # every row must be written, zeros included
        m.apply(o, x, tw, ti)
        outs.append(o)
    torch.testing.assert_close(outs[1] + outs[2], outs[0], rtol=0, atol=0)

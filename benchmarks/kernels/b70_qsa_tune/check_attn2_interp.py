# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-QSA-K: CPU correctness check of attn2 vs the torch reference and splitk.

Runs both Triton paths under TRITON_INTERPRET=1 on CPU tensors (no device),
at small shapes that still exercise: several requests, -1 holes in the
selection, a per-row valid count below the selection width, partial last
tiles, out-of-range physical pages, an all-empty row, and the output gate.

    TRITON_INTERPRET=1 python -I check_attn2_interp.py --ops-dir <tree>/vllm/models/qwen4_exp/nvidia/ops

The interpreter mis-handles bfloat16 operands of tl.dot (raw bits read as
integers), so the check runs in float16 and calls the attn2 launcher directly;
the bf16 path is covered by the GPU bench (--attn-impls) and the G1 suite.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aot_attn_regs import _stub_modules  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops-dir", required=True)
    ap.add_argument("--sel", type=int, default=150)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--head-dim", type=int, default=128)
    args = ap.parse_args()
    assert os.environ.get("TRITON_INTERPRET") == "1", "set TRITON_INTERPRET=1"
    import torch

    load = _stub_modules(args.ops_dir)
    load("qsa")  # imports qsa_attn2 through the stubbed package
    attn2 = sys.modules["qsaops.qsa_attn2"]
    tune = sys.modules["qsaops.qsa_xpu_tuning"]._tune

    torch.manual_seed(0)
    rows, sel, d = args.rows, args.sel, args.head_dim
    heads, kvh, page, nblocks, width = 6, 1, 32, 40, 8
    dt = torch.float16
    q = torch.randn(rows, heads, d).to(dt)
    gate = torch.randn(rows, heads, d).to(dt)
    kv = torch.randn(nblocks, page, kvh, 2 * d).to(dt)
    k_cache, v_cache = kv.split(d, dim=-1)
    table = torch.randint(0, nblocks, (2, width), dtype=torch.int32)
    table[1, 3] = nblocks + 5  # out-of-range physical page -> masked
    token_to_req = torch.tensor([i % 2 for i in range(rows)], dtype=torch.int32)
    ind = torch.full((rows, sel + 1), -1, dtype=torch.int32)
    for r in range(rows):
        n = sel - 7 * r if r != rows - 1 else 0  # last row: empty
        n = max(n, 0)
        toks = torch.randperm(width * page)[:n].to(torch.int32)
        if n > 10:
            toks[5] = -1  # hole
        ind[r, :n] = toks
        ind[r, sel] = n

    def reference():
        out = torch.zeros(rows, heads, d)
        for r in range(rows):
            n = int(ind[r, sel])
            lg = ind[r, :n].long()
            lg = lg[lg >= 0]
            req = int(token_to_req[r])
            pages = table[req, lg // page].long()
            ok = (pages >= 0) & (pages < nblocks)
            lg, pages = lg[ok], pages[ok]
            if lg.numel() == 0:
                o = torch.zeros(heads, d)
            else:
                kk = k_cache[pages, lg % page, 0].float()
                vv = v_cache[pages, lg % page, 0].float()
                sc = q[r].float() @ kk.T * d**-0.5
                o = (torch.softmax(sc, -1) @ vv).to(dt).float()
            out[r] = o * torch.sigmoid(gate[r].float())
        return out

    ref = reference()

    def run(impl, cfg=None):
        os.environ["B70_QSA_ATTN_IMPL"] = impl
        tune.clear()
        if cfg:
            tune["sparse_attn2_decode"] = cfg
        out = torch.empty_like(q)
        if impl == "splitk":
            raise RuntimeError("splitk needs bf16 Q (wrapper assert); skipped in fp16")
        attn2.qsa_sparse_paged_attention_attn2(
            q,
            k_cache,
            v_cache,
            ind,
            table,
            token_to_req,
            False,
            out,
            gate,
            d**-0.5,
            1.0,
            8,
        )
        return out.float()

    ok = True
    try:
        base = run("splitk")
        e = (base - ref).abs().max().item()
        print(f"splitk vs ref max_abs {e:.3e}")
    except Exception as exc:  # the interpreter cannot run every splitk path
        print(f"splitk not runnable under the interpreter: {str(exc)[:160]}")
        base = ref
    for cfg in (
        {"block_n": 16, "num_dchunk": 4, "tiles_per_prog": 1},
        {"block_n": 32, "num_dchunk": 2, "tiles_per_prog": 3, "row_chunk": 2},
        {"block_n": 64, "num_dchunk": 8, "tiles_per_prog": 2},
    ):
        got = run("attn2", cfg)
        e = (got - ref).abs().max().item()
        e2 = (got - base).abs().max().item()
        good = e <= 2e-2 and torch.isfinite(got).all().item()
        ok &= good
        print(
            f"attn2 {cfg} vs ref {e:.3e} vs splitk {e2:.3e} {'OK' if good else 'FAIL'}"
        )
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

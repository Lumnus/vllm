# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-QSA-D05: one-card identity + timing for the top-k levers (D01/D02).

Per shape (rows x compressed columns), on ReLU-shaped scores with real ties:
  identity: every top-k backend vs the torch selector, element for element
            (native: set + order mismatch counted, it is not exact by design)
  timing:   torch | native | native_det legacy | native_det select (torch ops)
            | native_det select-triton | radix  (wall time per call incl. launches)
  D02:      qsa_select_paged_decode per-row vs per-step (scorer + top-k), at
            decode_query_len 8 (DFlash K=7)

Usage (one card, engine stopped or another card):
  python benchmarks/kernels/b70_qsa_tune/bench_topk_levers.py --device xpu:1 \
      --rows 8,32,64 --ctx 4096,32768,131072,262144 --out topk.jsonl
CPU smoke: --device cpu --rows 8 --ctx 4096 --reps 1 (native/scorer skipped).
"""

import argparse
import json
import os
import time

import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qi

CR, TOKEN_TOPK, IDX_HEADS, IDX_DIM, PAGE = 4, 2048, 4, 128, 64


def sync(dev):
    if dev.type == "xpu":
        torch.xpu.synchronize()


def timeit(fn, dev, reps, warm=2):
    for _ in range(warm):
        fn()
    sync(dev)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        t.append((time.perf_counter() - t0) * 1e6)
    t.sort()
    return t[len(t) // 2]


def relu_logits(rows, width, positive_frac, gen, dev):
    x = torch.rand(rows, width, generator=gen)
    mask = torch.rand(rows, width, generator=gen) < positive_frac
    return torch.where(mask, x, 0.0).to(dev)


def run_topk(mode, impl, logits, visible, k):
    os.environ["B70_QSA_TOPK"] = mode
    os.environ["B70_QSA_TOPK_DET_IMPL"] = "legacy" if impl == "legacy" else "select"
    os.environ["B70_QSA_TOPK_SELECT_IMPL"] = (
        "triton" if impl == "select-triton" else "torch"
    )
    out = torch.empty(logits.shape[0], k, dtype=torch.int32, device=logits.device)
    ws = torch.empty(1 << 20, dtype=torch.uint8, device=logits.device)
    qi._topk(logits, visible, TOKEN_TOPK, CR, out, ws)
    return out


def bench_topk(args, dev, gen, rows, ctx, frac, emit):
    width = ctx // CR
    k = TOKEN_TOPK // CR
    logits = relu_logits(rows, width, frac, gen, dev)
    visible = torch.full((rows,), width, dtype=torch.int32, device=dev)
    visible[0] = max(1, k - 1)  # one short row
    want = run_topk("torch", "select", logits, visible, k)
    has_native = dev.type != "cpu" and hasattr(torch.ops._C, "top_k_per_row_decode")
    arms = [("torch", "select"), ("radix", "select")]
    if has_native:
        arms += [
            ("native", "select"),
            ("native_det", "legacy"),
            ("native_det", "select"),
            ("native_det", "select-triton"),
        ]
    for mode, impl in arms:
        got = run_topk(mode, impl, logits, visible, k)
        exact = bool(torch.equal(got, want))
        same_set = bool(torch.equal(got.sort(dim=1).values, want.sort(dim=1).values))
        us = timeit(
            lambda m=mode, i=impl: run_topk(m, i, logits, visible, k), dev, args.reps
        )
        emit(
            {
                "op": "topk",
                "mode": mode,
                "impl": impl,
                "rows": rows,
                "ctx": ctx,
                "positive_frac": frac,
                "exact": exact,
                "same_set": same_set,
                "us": us,
            }
        )


def bench_decode(args, dev, gen, num_requests, ctx, emit):
    dql = 8
    rows = num_requests * dql
    blocks = ctx // CR
    pages_per_req = -(-blocks // (PAGE // CR)) + 1
    comp_page = PAGE // CR
    pages = num_requests * pages_per_req + 1
    k_cache = torch.randn(pages, comp_page, 1, IDX_DIM, generator=gen).to(
        dev, torch.bfloat16
    )
    table = torch.randperm(pages, generator=gen)[: num_requests * pages_per_req]
    table = table.view(num_requests, pages_per_req).to(dev, torch.int32)
    pos = torch.cat([torch.arange(ctx - dql, ctx)] * num_requests)
    visible = ((pos + 1) // CR).to(dev, torch.int32)
    q = torch.randn(rows, IDX_HEADS, IDX_DIM, generator=gen).to(dev, torch.bfloat16)
    os.environ["B70_QSA_TOPK"] = args.decode_topk
    os.environ["B70_QSA_TOPK_DET_IMPL"] = "select"
    os.environ["B70_QSA_TOPK_SELECT_IMPL"] = args.select_impl
    res = {}
    for name, per_step in (("per_row", "0"), ("per_step", "1")):
        os.environ["B70_QSA_TOPK_PER_STEP"] = per_step
        out = torch.empty(rows, TOKEN_TOPK // CR, dtype=torch.int32, device=dev)

        def call(out=out):
            qi.qsa_select_paged_decode(
                q, k_cache, table, visible, TOKEN_TOPK, CR, dql, out
            )

        call()
        res[name] = out.clone()
        us = timeit(call, dev, args.reps)
        emit(
            {
                "op": "select_decode",
                "arm": name,
                "requests": num_requests,
                "rows": rows,
                "ctx": ctx,
                "topk": args.decode_topk,
                "us": us,
            }
        )
    os.environ["B70_QSA_TOPK_PER_STEP"] = "0"
    a = res["per_row"].cpu()
    b = res["per_step"].cpu()
    jac = []
    for r in range(rows):
        sa = set(x for x in a[r].tolist() if x >= 0)
        sb = set(x for x in b[r].tolist() if x >= 0)
        jac.append(len(sa & sb) / max(1, len(sa | sb)))
    emit(
        {
            "op": "per_step_overlap",
            "requests": num_requests,
            "ctx": ctx,
            "jaccard_min": min(jac),
            "jaccard_mean": sum(jac) / len(jac),
        }
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="xpu")
    p.add_argument("--rows", default="8,32,64")
    p.add_argument("--ctx", default="4096,32768,131072,262144")
    p.add_argument("--positive-frac", default="0.02,0.5")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--decode-topk", default="native_det")
    p.add_argument("--select-impl", default="triton", choices=["torch", "triton"])
    p.add_argument("--no-decode", action="store_true")
    p.add_argument("--out", default="")
    args = p.parse_args()
    dev = torch.device(args.device)
    if dev.type == "xpu":
        torch.xpu.set_device(dev)
    gen = torch.Generator().manual_seed(0)
    fh = open(args.out, "a") if args.out else None  # noqa: SIM115

    def emit(row):
        line = json.dumps(row)
        print(line, flush=True)
        if fh:
            fh.write(line + "\n")
            fh.flush()

    for ctx in [int(x) for x in args.ctx.split(",")]:
        for rows in [int(x) for x in args.rows.split(",")]:
            for frac in [float(x) for x in args.positive_frac.split(",")]:
                bench_topk(args, dev, gen, rows, ctx, frac, emit)
            if not args.no_decode and dev.type != "cpu" and rows % 8 == 0:
                bench_decode(args, dev, gen, rows // 8, ctx, emit)


if __name__ == "__main__":
    main()

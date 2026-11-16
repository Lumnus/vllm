# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-card bench: prefill attention over an int8_per_token_head KV cache on
XPU, the dequantize-then-BF16 path against the native INT8 kernel.

For each context length L, one sequence whose last `--chunk` tokens are the
prefill chunk (the cache already holds all L tokens, as in the engine: the
layer writes its K/V before it attends). Paths, per call of one layer:

1. `scratch`: `xpu_int8_per_token_head.forward_prefill`, the engine's path
   today (dequantize the referenced blocks into a BF16 scratch, FlashAttention
   of vllm-xpu-kernels over it);
2. `fa_only`: the same FlashAttention call over an already dequantized scratch
   (the dequantization's share);
3. `qk` / `qk_pv`: `xpu_int8_prefill_attention.int8_prefill_attention` (the
   engine's Triton kernel, the numerical reference); `qk_pv32`: the same with
   KV tiles of 32 slots, the P quantization groups of `sycl_qk_pv`;
4. `sycl_bf16` / `sycl_qk` / `sycl_qk_pv`: the vllm-xpu-kernels chunk-prefill
   kernel over the int8 cache in place
   (`xpu_int8_per_token_head.forward_prefill_sycl`, `int8_prefill` 1, 2, 3).

`--check` compares each path's output, on sampled query rows, with a float32
reference of the decode contract (BF16 query, K/V = cached codes x scales) and
with the native INT8 decode kernel run on the same rows (prefill and decode on
the same cached tokens; with int8_dot for the int8-query paths), and prints
the SYCL paths against their Triton references (`sycl_qk` vs `qk`,
`sycl_qk_pv` vs `qk_pv32`) over the whole chunk. `--sweep` times the Triton
int8 paths over launch configs.
`--full-prompt` times every chunk of an L-token prompt (the whole prefill's
attention for one layer) instead of the last chunk only.

Example (one card, the served shapes per card: 6 query heads, 1 KV head,
head_dim 256, cache block 1600):

    python benchmarks/kernels/benchmark_xpu_int8_prefill_attention.py \\
        --context 2048,8192,32768,131072 --chunk 4096 --check
"""

import argparse
import functools
import json
import statistics

import torch
from vllm._xpu_ops import xpu_ops
from vllm.v1.attention.ops import xpu_int8_per_token_head as int8_kv
from vllm.v1.attention.ops import xpu_int8_prefill_attention as i8p
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.kv_cache_interface import KVQuantMode

# --modes name -> int8_prefill of the vllm-xpu-kernels chunk-prefill kernel.
SYCL_MODES = {"sycl_bf16": 1, "sycl_qk": 2, "sycl_qk_pv": 3}

SWEEP = [
    (64, 64, 64, 4, 1, 16),
    (128, 64, 64, 4, 1, 32),
    (32, 64, 64, 4, 1, 8),
    (64, 64, 64, 4, 1, 8),
    (64, 64, 64, 2, 2, 8),
    (64, 64, 32, 4, 1, 16),
    (64, 32, 64, 4, 1, 16),
]


def make_cache(args, total: int, dev):
    """Split-layout int8 cache holding `total` tokens of one sequence, written
    by the engine's per-token-head writer from BF16 K/V."""
    hd, nkv, block = args.head_dim, args.kv_heads, args.block
    content = args.content or 2 * (hd + 4)
    nblocks = -(-total // block) + 2
    page = nkv * block * content
    raw = torch.zeros(nblocks * page, dtype=torch.int8, device=dev)
    cache = torch.as_strided(
        raw, (nblocks, nkv, block, content), (page, block * content, content, 1)
    )
    key, value, k_scale, v_scale = int8_kv.split_layout_views(cache, nkv, hd)
    # Blocks in a shuffled order, as a paged cache holds them.
    perm = torch.randperm(nblocks - 1, device=dev)[: -(-total // block)] + 1
    table = perm.to(torch.int32)[None, :]
    torch.manual_seed(args.seed)
    for s in range(0, total, 16384):
        e = min(total, s + 16384)
        k = torch.randn(e - s, nkv, hd, device=dev)
        k[..., : args.outliers] *= 8.0
        v = torch.randn(e - s, nkv, hd, device=dev)
        pos = torch.arange(s, e, device=dev)
        slots = (table[0, pos // block].long() * block + pos % block).to(torch.int64)
        triton_reshape_and_cache_flash_per_token_head_quant(
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            key,
            value,
            k_scale,
            v_scale,
            slots,
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        )
    return key, value, k_scale, v_scale, table


def timed(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    times = []
    for _ in range(iters):
        a = torch.xpu.Event(enable_timing=True)
        b = torch.xpu.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def reference_rows(q, key, value, k_scale, v_scale, table, block, rows, ctx, scale):
    """float32 decode-contract attention of query rows `rows` (positions
    ctx + r) over the cached codes x scales."""
    outs = []
    hq, nkv = q.shape[1], key.shape[2]
    g = hq // nkv
    end = ctx + int(rows.max()) + 1
    pos = torch.arange(end, device=q.device)
    blk = table[0, pos // block].long()
    k_all = key[blk, pos % block].float() * k_scale[blk, pos % block][..., None]
    v_all = value[blk, pos % block].float() * v_scale[blk, pos % block][..., None]
    for r in rows.tolist():
        L = ctx + r + 1
        qr = q[r].float().view(nkv, g, -1)
        s = torch.einsum("ngd,knd->ngk", qr, k_all[:L]) * scale
        o = torch.einsum("ngk,knd->ngd", torch.softmax(s, -1), v_all[:L])
        outs.append(o.reshape(hq, -1))
    return torch.stack(outs)


def metrics(out, ref):
    out, ref = out.double().flatten(), ref.double().flatten()
    return {
        "rel": float((out - ref).norm() / ref.norm()),
        "max_abs": float((out - ref).abs().max()),
        "cos": float(torch.nn.functional.cosine_similarity(out, ref, dim=0)),
    }


def run_context(args, L: int, dev) -> list[dict]:
    hq, hd = args.heads, args.head_dim
    scale = hd**-0.5
    key, value, k_scale, v_scale, table = make_cache(args, L, dev)
    chunks = (
        [(s, min(L, s + args.chunk)) for s in range(0, L, args.chunk)]
        if args.full_prompt
        else [(max(0, L - args.chunk), L)]
    )
    scratch_tokens = int8_kv.prefill_scratch_tokens(L, args.block)
    ks = torch.empty(
        scratch_tokens // args.block,
        args.block,
        args.kv_heads,
        hd,
        dtype=torch.bfloat16,
        device=dev,
    )
    vs = torch.empty_like(ks)
    results = []
    launches = (
        [i8p.Int8PrefillLaunch(*t) for t in SWEEP] if args.sweep else [i8p._LAUNCH]
    )
    totals: dict[str, float] = {}
    for s, e in chunks:
        q_len, ctx = e - s, s
        q = torch.randn(q_len, hq, hd, device=dev).to(torch.bfloat16)
        cu = torch.tensor([0, q_len], dtype=torch.int32, device=dev)
        seq_lens = torch.tensor([e], dtype=torch.int32, device=dev)
        outs = {}
        out = torch.empty_like(q)
        scratch = functools.partial(
            int8_kv.forward_prefill,
            q,
            key,
            value,
            k_scale,
            v_scale,
            out,
            ks,
            vs,
            cu,
            cu.cpu(),
            seq_lens,
            [e],
            table,
            scale,
            True,
            [-1, -1],
            None,
        )
        paths = {"scratch": scratch}
        # FlashAttention alone over the scratch the call above fills.
        scratch()
        stride = -(-e // args.block)
        paths["fa_only"] = functools.partial(
            xpu_ops.flash_attn_varlen_func,
            q,
            ks,
            vs,
            cu_seqlens_q=cu,
            max_seqlen_q=q_len,
            max_seqlen_k=e,
            softmax_scale=scale,
            causal=True,
            out=out,
            block_table=torch.arange(stride, dtype=torch.int32, device=dev)[None, :],
            seqused_k=seq_lens,
            window_size=[-1, -1],
            s_aux=None,
        )
        for launch in launches:
            for mode in args.modes.split(","):
                if mode not in ("qk", "qk_pv", "qk_pv32"):
                    continue
                name = mode if not args.sweep else f"{mode}{tuple(launch)}"
                outs[name] = torch.empty_like(q)
                paths[name] = functools.partial(
                    i8p.int8_prefill_attention,
                    q,
                    key,
                    value,
                    k_scale,
                    v_scale,
                    outs[name],
                    cu,
                    seq_lens,
                    table,
                    scale,
                    pv_int8=mode != "qk",
                    launch=launch._replace(block_n=32) if mode == "qk_pv32" else launch,
                )
        for mode in args.modes.split(","):
            if mode not in SYCL_MODES:
                continue
            outs[mode] = torch.empty_like(q)
            paths[mode] = functools.partial(
                int8_kv.forward_prefill_sycl,
                q,
                key,
                value,
                k_scale,
                v_scale,
                outs[mode],
                cu,
                q_len,
                seq_lens,
                e,
                table,
                scale,
                SYCL_MODES[mode],
                int8_dot=mode != "sycl_bf16",
            )
        ops = 4 * hq * hd * q_len * (ctx + (q_len + 1) / 2)
        for name, fn in paths.items():
            if name not in ("scratch", "fa_only") or "scratch" in args.modes:
                ms = timed(fn, args.iters, args.warmup)
                totals[name] = totals.get(name, 0.0) + ms
                if not args.full_prompt:
                    results.append(
                        {
                            "L": L,
                            "ctx": ctx,
                            "q": q_len,
                            "path": name,
                            "ms": ms,
                            "TOPS": ops / ms / 1e9,
                        }
                    )
        if args.check and (s, e) == chunks[-1]:
            scratch()
            outs["scratch"] = out.clone()
            rows = torch.linspace(0, q_len - 1, args.check_rows, device=dev).long()
            ref = reference_rows(
                q, key, value, k_scale, v_scale, table, args.block, rows, ctx, scale
            )
            # The native decode kernel on the same rows, BF16 query and
            # int8_dot (the int8-query paths' decode counterpart).
            dq = q[rows].contiguous()
            n = rows.numel()
            douts = {}
            for dot in (False, True):
                douts[dot] = torch.empty_like(dq)
                int8_kv.forward_decode(
                    dq,
                    key,
                    value,
                    k_scale,
                    v_scale,
                    douts[dot],
                    torch.arange(n + 1, dtype=torch.int32, device=dev),
                    1,
                    (ctx + rows + 1).to(torch.int32),
                    e,
                    table.expand(n, -1).contiguous(),
                    scale,
                    None,
                    None,
                    int8_dot=dot,
                )
            dout = douts[False]
            check = {
                "L": L,
                "decode_vs_ref": metrics(dout, ref),
                "decode_int8_vs_ref": metrics(douts[True], ref),
            }
            for name, o in outs.items():
                int8_q = name not in ("scratch", "sycl_bf16")
                check[f"{name}_vs_ref"] = metrics(o[rows], ref)
                check[f"{name}_vs_decode"] = metrics(o[rows], douts[int8_q])
                check[f"{name}_finite"] = bool(torch.isfinite(o).all())
            for a, b in (("sycl_qk", "qk"), ("sycl_qk_pv", "qk_pv32")):
                if a in outs and b in outs:
                    check[f"{a}_vs_{b}"] = metrics(outs[a], outs[b])
            print("CHECK " + json.dumps(check), flush=True)
    if args.full_prompt:
        for name, ms in totals.items():
            ops = 4 * hq * hd * L * (L + 1) / 2
            results.append(
                {"L": L, "path": name, "prompt_ms": ms, "TOPS": ops / ms / 1e9}
            )
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--context", default="2048,8192,32768,131072")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--block", type=int, default=1600)
    p.add_argument(
        "--content",
        type=int,
        default=0,
        help="cache bytes per (slot, head); default 2 * (head_dim + 4)",
    )
    p.add_argument(
        "--outliers",
        type=int,
        default=4,
        help="K channels scaled by 8 (outlier channels)",
    )
    p.add_argument(
        "--modes",
        default="scratch,qk,qk_pv,qk_pv32,sycl_bf16,sycl_qk,sycl_qk_pv",
        help="scratch (and fa_only), the Triton qk / qk_pv / qk_pv32, the "
        "SYCL sycl_bf16 / sycl_qk / sycl_qk_pv",
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--check-rows", type=int, default=32)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--full-prompt", action="store_true")
    args = p.parse_args()
    dev = torch.device("xpu", torch.xpu.current_device())
    for L in [int(x) for x in args.context.split(",")]:
        for row in run_context(args, L, dev):
            print("ROW " + json.dumps(row), flush=True)
        torch.xpu.empty_cache()


if __name__ == "__main__":
    main()

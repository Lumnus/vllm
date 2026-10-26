# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-QSA-T: bucket a torch-profiler trace (vLLM /start_profile, XPU activity) by QSA op.

usage: qsa_profile_summary.py <trace.json[.gz] | dir> [--steps N] [--top 30] [--kernels]

Reads Chrome-trace files written by vLLM's torch profiler
(--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'), keeps device
kernel events, and sums their duration per bucket (QSA ops, the dense pieces the
engine shares with the dense def, and "other"). With --steps the totals are also
given per engine step: pass the number of decode steps (or prefill chunks) that
fell inside the profiled window (e.g. --profiler-config max_iterations).

Reading it: QSA buckets are the sparse-only cost per window; compare their sum
against the dense def's flash-attention bucket over the same window. "host gaps"
= window wall time minus the union of device kernel intervals, per rank: if it
is large at decode, the step is launch/sync bound, not kernel bound.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re

BUCKETS = [
    ("qsa_prepare", r"_qsa_prepare_kernel"),
    ("qsa_metadata", r"_build_qsa_metadata|_logical_positions|token_to_req"),
    ("qsa_decode_score", r"_qsa_mqa_paged_uniform"),
    ("qsa_prefill_score", r"_qsa_mqa_paged_prefill"),
    ("qsa_topk_native", r"top_?k_per_row|topk_per_row|TopKPerRow"),
    ("qsa_expand", r"_expand_qsa"),
    ("qsa_sparse_attn", r"_qsa_sparse_paged_gqa"),
    ("qsa_merge", r"_qsa_merge_splitk"),
    ("qsa_store_compress", r"_store_qsa_rows|_compress_qsa_groups"),
    ("qsa_attn2", r"_qsa_attn2_tile"),
    ("qsa_attn2_merge", r"_qsa_attn2_merge"),
    ("qsa_select", r"_qsa_select_(keep|order)"),
    ("torch_sort_topk", r"(?i)sort|topk|radix|segmented|bitonic"),
    (
        "elementwise_fill_copy",
        r"(?i)fill|copy|elementwise|vectorized|where|bitwise|arange|cat_",
    ),
    (
        "dense_flash_attn",
        r"(?i)flash|fmha|varlen|chunk_prefill|paged_decode|_fwd_kernel",
    ),
    ("gdn", r"(?i)gdn|gated_delta|chunk_gated|fused_recurrent|causal_conv"),
    ("moe", r"(?i)moe|expert|grouped_gemm|int4"),
    ("hyper_conn", r"(?i)\bhc_|hyper|ordered_hc"),
    ("collective", r"(?i)ccl|allreduce|all_reduce|allgather"),
    ("gemm", r"(?i)gemm|matmul|onednn|xetla|linear"),
    ("norm_rope", r"(?i)rms|norm|rope|rotary"),
]
_RX = [(n, re.compile(p)) for n, p in BUCKETS]


def bucket(name: str) -> str:
    for n, rx in _RX:
        if rx.search(name):
            return n
    return "other"


def load(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def is_device_kernel(ev: dict) -> bool:
    cat = str(ev.get("cat", "")).lower()
    return ev.get("ph") == "X" and ("kernel" in cat or cat in ("gpu_op", "xpu_op"))


def summarise(path: str, steps: int, top: int, show_kernels: bool) -> None:
    trace = load(path)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    per_bucket = collections.Counter()
    calls = collections.Counter()
    per_kernel = collections.Counter()
    intervals = []
    for ev in events:
        if not is_device_kernel(ev):
            continue
        name = ev.get("name", "?")
        dur = float(ev.get("dur", 0.0))
        b = bucket(name)
        per_bucket[b] += dur
        calls[b] += 1
        per_kernel[name] += dur
        ts = float(ev.get("ts", 0.0))
        intervals.append((ts, ts + dur))
    if not intervals:
        print(
            f"{path}: no device kernel events (check torch_profiler_activities includes XPU)"
        )
        return
    intervals.sort()
    busy, cur_s, cur_e = 0.0, *intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    wall = intervals[-1][1] - intervals[0][0]
    total = sum(per_bucket.values())
    div = max(steps, 1)
    print(f"== {path}")
    print(
        f"window {wall / 1e3:.2f} ms, device busy (union) {busy / 1e3:.2f} ms, host gaps {(wall - busy) / 1e3:.2f} ms"
        + (
            f"  | per step: busy {busy / div / 1e3:.3f} ms, gaps {(wall - busy) / div / 1e3:.3f} ms"
            if steps
            else ""
        )
    )
    qsa = sum(
        v
        for k, v in per_bucket.items()
        if k.startswith("qsa_") or k == "torch_sort_topk"
    )
    print(
        f"QSA-attributable (qsa_* + torch_sort_topk) {qsa / 1e3:.2f} ms = {100 * qsa / total:.1f}% of kernel time"
        + (f", {qsa / div / 1e3:.3f} ms/step" if steps else "")
    )
    print(
        f"{'bucket':24s} {'ms':>10s} {'%':>6s} {'calls':>8s}"
        + ("   ms/step" if steps else "")
    )
    for b, v in per_bucket.most_common():
        print(
            f"{b:24s} {v / 1e3:10.3f} {100 * v / total:6.1f} {calls[b]:8d}"
            + (f" {v / div / 1e3:9.3f}" if steps else "")
        )
    if show_kernels:
        print("-- top kernels")
        for name, v in per_kernel.most_common(top):
            print(f"{v / 1e3:10.3f} ms  {bucket(name):20s} {name[:110]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--kernels", action="store_true")
    a = ap.parse_args()
    paths = [a.path]
    if os.path.isdir(a.path):
        paths = sorted(glob.glob(os.path.join(a.path, "**", "*.json*"), recursive=True))
    for p in paths:
        summarise(p, a.steps, a.top, a.kernels)


if __name__ == "__main__":
    main()

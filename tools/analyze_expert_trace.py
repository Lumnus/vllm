# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read an EPLB expert-load trace and say how local in time the routing is.

The trace is written with `EPLBConfig.trace_path` (format in
`vllm/distributed/eplb/eplb_trace.py`): per interval of N engine steps, per MoE
layer, the number of calls each logical expert received. This script needs only
the standard library and numpy.

    python tools/analyze_expert_trace.py TRACE [TRACE ...] [options]

TRACE is a file, or the `trace_path` prefix (then `<prefix>.rank*` is read).
The files of several ranks are added record by record (equal `step`). A saved
EPLB state file (`EPLBConfig.state_path`: a JSON object layer name -> list of
loads) gives the cumulative table only.

What is printed (a "call" is one token routed to one expert of one layer):

1. Cumulative: the share of all calls outside the n hottest experts of the
   layer (hot by the whole run), for each n in --top-n. The mean over layers,
   the worst layer, and calls per decoded token outside (share x --calls-per-
   token).
2. Per interval (trace only): the same n applied to each interval, as the mean
   over intervals, in two readings. `static`: the n experts hot over the whole
   run (what a fixed set would miss). `oracle`: the n hottest of that interval
   itself (the least any set of n changing every interval could miss).
3. Working set (trace only): over windows of W intervals, the number of
   distinct experts called and the number that carry 90 % of the window's
   calls (mean over layers and windows).
4. Turnover (trace only): the Jaccard index of the set of experts called
   in consecutive windows, and of the --hot hottest experts of consecutive
   windows. 1 = the same set, 0 = disjoint.
5. Replay (trace only): per layer, a fixed fast set of F experts (the F
   hottest of the first K intervals, never changing) and a pool of P further
   experts under LRU or count-decay eviction. An expert called in an interval
   and not resident at its start is a miss (a fetch); it is admitted at the
   end of the interval. Intervals carry no order inside, so every call to an
   expert that was not resident at the interval start counts as a missed call
   (a conservative reading when the interval is longer than a step). The
   first K intervals warm the pool and are not counted. Printed per (F, P,
   policy): hit share of calls, misses per 1,000 calls, misses per step (all
   layers together, a miss is one expert fetched for one layer in one
   interval), and the worst layer's hit share.
"""

import argparse
import glob
import json
import struct
import sys
import zlib
from dataclasses import dataclass

import numpy as np

MAGIC = b"EPLBTRC1"
_DATA_HEAD = struct.Struct("<BqdB")


@dataclass
class Trace:
    layers: list[str]
    interval: int  # engine steps per record
    steps: np.ndarray  # [T] step counter at the end of each record
    wall: np.ndarray  # [T] wall time
    counts: np.ndarray  # [T, layers, experts] uint32

    @property
    def num_intervals(self) -> int:
        return self.counts.shape[0]


def read_trace_file(path: str) -> Trace:
    """One rank's file. A truncated last record (a write in flight) is ignored."""
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw[: len(MAGIC)] != MAGIC:
        raise ValueError(f"{path}: not an expert trace")
    pos = len(MAGIC)
    meta: dict = {}
    steps, walls, bodies = [], [], []
    while pos + 4 <= len(raw):
        (length,) = struct.unpack_from("<I", raw, pos)
        if pos + 4 + length > len(raw):
            break
        payload = zlib.decompress(raw[pos + 4 : pos + 4 + length])
        pos += 4 + length
        if payload[0] == 0:
            meta = json.loads(payload[1:])
        elif payload[0] == 1:
            _, step, wall, itemsize = _DATA_HEAD.unpack_from(payload)
            plane = np.frombuffer(payload, dtype=np.uint8, offset=_DATA_HEAD.size)
            shuffled = plane.reshape(itemsize, -1)
            flat = shuffled.T.copy().reshape(-1).view(f"<u{itemsize}")
            steps.append(step)
            walls.append(wall)
            bodies.append(flat.astype(np.uint32))
    if not meta:
        raise ValueError(f"{path}: no meta record")
    num_layers = len(meta["layers"])
    num_experts = int(meta["num_logical_experts"])
    counts = (
        np.stack(bodies).reshape(len(bodies), num_layers, num_experts)
        if bodies
        else np.zeros((0, num_layers, num_experts), dtype=np.uint32)
    )
    return Trace(
        layers=list(meta["layers"]),
        interval=int(meta["interval"]),
        steps=np.array(steps, dtype=np.int64),
        wall=np.array(walls, dtype=np.float64),
        counts=counts,
    )


def read_trace(paths: list[str]) -> Trace:
    """Files given, or prefixes whose `.rank*` files are read; ranks are added
    on the steps they share."""
    files: list[str] = []
    for path in paths:
        found = sorted(glob.glob(path + ".rank*"))
        files.extend(found if found else [path])
    traces = [read_trace_file(f) for f in files]
    first = traces[0]
    if len(traces) == 1:
        return first
    common = set(first.steps.tolist())
    for t in traces[1:]:
        common &= set(t.steps.tolist())
    keep = sorted(common)
    total = None
    for t in traces:
        index = {s: i for i, s in enumerate(t.steps.tolist())}
        part = t.counts[[index[s] for s in keep]].astype(np.uint64)
        total = part if total is None else total + part
    assert total is not None
    idx = {s: i for i, s in enumerate(first.steps.tolist())}
    return Trace(
        layers=first.layers,
        interval=first.interval,
        steps=np.array(keep, dtype=np.int64),
        wall=first.wall[[idx[s] for s in keep]],
        counts=total.astype(np.uint32),
    )


def read_state_file(path: str) -> Trace | None:
    """A saved EPLB state: one cumulative 'interval', or None if `path` is not
    a JSON object of layer name -> loads."""
    try:
        with open(path) as handle:
            raw = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    layers = sorted(raw)
    counts = np.array([raw[k] for k in layers], dtype=np.uint32)[None]
    return Trace(layers, 0, np.array([0]), np.array([0.0]), counts)


# ---- the numbers


def outside_top(counts: np.ndarray, n: int) -> np.ndarray:
    """Share of the calls outside the n largest entries along the last axis."""
    total = counts.sum(axis=-1).astype(np.float64)
    if n >= counts.shape[-1]:
        return np.zeros_like(total)
    part = -np.partition(-counts.astype(np.int64), n - 1, axis=-1)[..., :n]
    kept = part.sum(axis=-1).astype(np.float64)
    return np.where(total > 0, 1.0 - kept / np.maximum(total, 1), 0.0)


def cumulative_table(trace: Trace, tops: list[int], calls_per_token: float):
    cum = trace.counts.astype(np.int64).sum(axis=0)  # [layers, experts]
    rows = []
    for n in tops:
        share = outside_top(cum, n)
        rows.append((n, share.mean(), share.max(), share.mean() * calls_per_token))
    return cum, rows


def per_interval_table(trace: Trace, cum: np.ndarray, tops: list[int]):
    rows = []
    hot_order = np.argsort(-cum, axis=-1, kind="stable")
    for n in tops:
        static_sum = oracle_sum = 0.0
        seen = 0
        for lo in range(0, trace.num_intervals, 128):
            c = trace.counts[lo : lo + 128].astype(np.int64)
            total = c.sum(axis=-1).astype(np.float64)
            mask = total > 0
            if n >= c.shape[-1]:
                seen += int(mask.sum())
                continue
            order = np.broadcast_to(hot_order[None, :, :n], (len(c), c.shape[1], n))
            kept = np.take_along_axis(c, order, axis=-1).sum(axis=-1)
            static = 1 - kept / np.maximum(total, 1)
            static_sum += float(static[mask].sum())
            oracle_sum += float(outside_top(c, n)[mask].sum())
            seen += int(mask.sum())
        rows.append((n, static_sum / max(seen, 1), oracle_sum / max(seen, 1)))
    return rows


def windows(trace: Trace, w: int) -> np.ndarray:
    """Counts summed over non-overlapping windows of w intervals [W, L, E]."""
    n = trace.num_intervals // w
    c = trace.counts[: n * w].astype(np.int64)
    return c.reshape(n, w, *c.shape[1:]).sum(axis=1)


def carrying(counts: np.ndarray, share: float) -> np.ndarray:
    """How many of the largest entries (last axis) carry `share` of the total."""
    srt = -np.sort(-counts, axis=-1)
    csum = np.cumsum(srt, axis=-1)
    total = csum[..., -1:]
    need = np.ceil(total * share)
    return (csum < need).sum(axis=-1) + (total[..., 0] > 0)


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = (a & b).sum(axis=-1)
    union = (a | b).sum(axis=-1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def hot_mask(counts: np.ndarray, m: int) -> np.ndarray:
    """Boolean mask of the m largest entries (those with a call) per row."""
    m = min(m, counts.shape[-1])
    idx = np.argsort(-counts, axis=-1, kind="stable")[..., :m]
    mask = np.zeros(counts.shape, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=-1)
    return mask & (counts > 0)


def working_set(trace: Trace, w: int, hot: int):
    win = windows(trace, w)
    if len(win) == 0:
        return None
    used = win > 0
    distinct = used.sum(axis=-1)  # [W, L]
    p90 = carrying(win, 0.9)
    out = {
        "windows": len(win),
        "distinct": distinct.mean(axis=0),  # per layer
        "p90": p90.mean(axis=0),
    }
    if len(win) > 1:
        out["jac_used"] = jaccard(used[:-1], used[1:]).mean(axis=0)
        hm = hot_mask(win, hot)
        out["jac_hot"] = jaccard(hm[:-1], hm[1:]).mean(axis=0)
    return out


def replay(
    trace: Trace, fixed: int, pool: int, policy: str, warm: int, half_life: float
):
    """Replay the trace through per-layer sets: `fixed` experts hot in the
    first `warm` intervals (never change) plus a pool of `pool` more."""
    counts = trace.counts.astype(np.int64)
    t_total, n_layers, n_exp = counts.shape
    warm = min(warm, t_total - 1) if t_total > 1 else 0
    first = counts[: max(warm, 1)].sum(axis=0)
    fast = np.zeros((n_layers, n_exp), dtype=bool)
    if fixed > 0:
        idx = np.argsort(-first, axis=-1, kind="stable")[:, :fixed]
        np.put_along_axis(fast, idx, True, axis=-1)
    in_pool = np.zeros((n_layers, n_exp), dtype=bool)
    score = np.full((n_layers, n_exp), -np.inf if policy == "lru" else 0.0)
    gamma = 0.5 ** (1.0 / half_life)
    rank_scale = 1.0 / (n_exp + 1)
    calls = np.zeros(n_layers)
    missed_calls = np.zeros(n_layers)
    miss_events = np.zeros(n_layers)
    for t in range(t_total):
        c = counts[t]
        used = c > 0
        resident = fast | in_pool
        if t >= warm:
            miss = used & ~resident
            calls += c.sum(axis=-1)
            missed_calls += (c * miss).sum(axis=-1)
            miss_events += miss.sum(axis=-1)
        if pool > 0:
            if policy == "lru":
                # recency: this interval, the heavier the more recent
                order = np.argsort(np.argsort(c, axis=-1, kind="stable"), axis=-1)
                stamp = t + order * rank_scale
                score = np.where(used, stamp, score)
            else:
                score = score * gamma + c
            candidate = (in_pool | used) & ~fast
            key = np.where(candidate, score, -np.inf)
            if pool >= n_exp:
                in_pool = candidate
            else:
                top = np.argpartition(-key, pool - 1, axis=-1)[:, :pool]
                keep = np.zeros_like(in_pool)
                np.put_along_axis(keep, top, True, axis=-1)
                in_pool = keep & candidate
    counted = max(t_total - warm, 0)
    steps = counted * trace.interval
    with np.errstate(invalid="ignore", divide="ignore"):
        hit = 1 - missed_calls / calls
        per_k = 1000 * miss_events / calls
    return {
        "hit": float(np.nanmean(hit)),
        "worst_hit": float(np.nanmin(hit)),
        "per_1000": float(np.nanmean(per_k)),
        "per_step": float(miss_events.sum() / steps) if steps else float("nan"),
        "intervals": counted,
    }


# ---- output


def pct(x: float) -> str:
    return f"{100 * x:6.2f}%"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="trace file(s), prefix(es), state JSON")
    ap.add_argument(
        "--top-n", type=int, nargs="+", default=[128, 196, 256, 384, 448, 480]
    )
    ap.add_argument(
        "--calls-per-token",
        type=float,
        default=480.0,
        help="expert calls per token over all layers (layers x calls per layer)",
    )
    ap.add_argument(
        "--window",
        type=int,
        nargs="+",
        default=[10, 60, 600],
        help="working-set windows, in intervals",
    )
    ap.add_argument("--hot", type=int, default=64, help="hot set size for turnover")
    ap.add_argument("--fixed", type=int, nargs="+", default=[0, 128], help="F values")
    ap.add_argument(
        "--warm",
        type=int,
        default=60,
        help="K: intervals that choose F and warm the pool, not counted",
    )
    ap.add_argument("--pool", type=int, nargs="+", default=[16, 32, 64, 128, 192])
    ap.add_argument(
        "--half-life",
        type=float,
        nargs="+",
        default=[30.0],
        help="count-decay half-life, in intervals",
    )
    ap.add_argument("--per-layer", action="store_true", help="working set per layer")
    args = ap.parse_args(argv)

    trace = read_state_file(args.inputs[0]) if len(args.inputs) == 1 else None
    state_only = trace is not None
    if trace is None:
        trace = read_trace(args.inputs)
    n_layers, n_exp = trace.counts.shape[1:]
    tops = [n for n in args.top_n if n <= n_exp]
    span = ""
    if not state_only:
        total_steps = trace.num_intervals * trace.interval
        span = f" x {trace.interval} steps = {total_steps} steps"
    print(f"layers {n_layers}, experts {n_exp}, intervals {trace.num_intervals}{span}")
    cum, rows = cumulative_table(trace, tops, args.calls_per_token)
    layer_calls = np.maximum(cum.sum(axis=-1), 1)
    print(f"\ncalls per layer (cumulative): median {int(np.median(layer_calls))}")
    print("\n1. Cumulative: share of calls outside the n hottest experts of the run")
    print("      n   mean over layers   worst layer   calls/token outside")
    for n, mean, worst, per_token in rows:
        print(f"{n:7d}   {pct(mean)}          {pct(worst)}      {per_token:8.2f}")
    srt = -np.sort(-cum, axis=-1)
    cs = np.cumsum(srt, axis=-1) / layer_calls[:, None]
    for share in (0.5, 0.9, 0.99, 0.999):
        k = (cs < share).sum(axis=-1) + 1
        print(
            f"   {share:.1%} of a layer's calls need {int(np.median(k))} experts "
            f"(median; {int(k.min())} to {int(k.max())})"
        )
    top_share = float((srt[:, 0] / layer_calls).mean())
    print(f"   the hottest expert of a layer carries {pct(top_share)} (mean)")
    if state_only or trace.num_intervals < 2:
        return 0

    print("\n2. Per interval: mean over intervals of the share outside the n hottest")
    print("      n   static (hot over the run)   oracle (hot in the interval)")
    for n, static, oracle in per_interval_table(trace, cum, tops):
        print(f"{n:7d}   {pct(static)}                    {pct(oracle)}")

    print(f"\n3-4. Working set and turnover (mean over layers/windows; hot={args.hot})")
    print("   W   windows   distinct   90% set   Jaccard used   Jaccard hot")
    for w in args.window:
        ws = working_set(trace, w, args.hot)
        if ws is None:
            print(f"{w:4d}   (trace shorter than a window)")
            continue
        ju = f"{ws['jac_used'].mean():.3f}" if "jac_used" in ws else "  -  "
        jh = f"{ws['jac_hot'].mean():.3f}" if "jac_hot" in ws else "  -  "
        print(
            f"{w:4d}   {ws['windows']:7d}   {ws['distinct'].mean():8.1f}   "
            f"{ws['p90'].mean():7.1f}   {ju:>12}   {jh:>11}"
        )
        if args.per_layer:
            for i, name in enumerate(trace.layers):
                d, p90 = ws["distinct"][i], ws["p90"][i]
                print(f"        {name}: distinct {d:.1f}, 90% {p90:.1f}")

    print(
        f"\n5. Replay (K = {args.warm} warm-up intervals not counted; "
        f"interval = {trace.interval} steps)"
    )
    print("   F     P  policy        hit   miss/1000 calls   miss/step   worst layer")
    for f in args.fixed:
        for p in args.pool:
            for policy, hl in [("lru", 0.0)] + [("decay", h) for h in args.half_life]:
                r = replay(trace, f, p, policy, args.warm, hl or 1.0)
                name = policy if policy == "lru" else f"decay{hl:g}"
                print(
                    f"{f:4d} {p:5d}  {name:<8}  {pct(r['hit'])}   "
                    f"{r['per_1000']:14.2f}   {r['per_step']:9.2f}   "
                    f"{pct(r['worst_hit'])}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())

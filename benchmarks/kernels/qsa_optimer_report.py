#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B70-QSA-P: read the B70_QSA_OPTIMER output (rank<r>.jsonl) and print the host/kernel split per rows-per-step.

usage: qsa_optimer_report.py <dir | file.jsonl ...> [--rank 0|all] [--json]

Per T (rows per step = streams x (1 + draft tokens), e.g. 8 for 1 stream, 32 for 4, 64 for 8 at K=7):
  step      mean wall time between two engine steps of the anchor layer (the whole step, not only QSA)
  per step  = window sum / steps in the window, summed over all QSA layers of one rank
  host      Python wall of the span, exclusive of nested spans (launch cost)
  dev       XPU event time of the span, exclusive of nested spans (kernel + idle gaps)   [events mode]
  kern      launch + drain wall of a leaf with the queue drained before and after          [sync mode]
  lead      device work queued before the span, drained by the leading sync                [sync mode]
  h/d       host / dev. Near 1.0: the device waited on the host (launch bound). Well below 1: kernel bound.
"""
import argparse
import collections
import glob
import json
import os
import sys


def load(paths, rank):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "rank*.jsonl"))) if os.path.isdir(p) else [p]
    out = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                if rank == "all" or str(r.get("rank")) == str(rank):
                    out.append(r)
    return out


def merge(recs):
    """T -> {steps, period_sum, period_n, spans: name -> sums, modes}; ranks merge by summing (means are per-rank
    averages once divided by the rank count)."""
    ranks = {r["rank"] for r in recs} or {0}
    agg = {}
    over = steps = 0.0
    for r in recs:
        over += r.get("timer_overhead_ms", 0.0)
        steps += r.get("window_steps", 0)
        for T, b in r["by_T"].items():
            a = agg.setdefault(int(T), dict(steps=0, period=[], over=0.0, spans=collections.defaultdict(
                lambda: collections.defaultdict(float)), mode=set()))
            a["steps"] += b["steps"]
            a["mode"].add(r["mode"])
            if b.get("period_ms_mean") is not None:
                a["period"].append((b["period_ms_mean"], b["steps"]))
            for n, s in b["spans"].items():
                for k, v in s.items():
                    a["spans"][n][k] += v
    return agg, len(ranks), (over / steps if steps else None)


def table(agg):
    rows = []
    for T in sorted(agg):
        a = agg[T]
        steps = a["steps"] or 1
        pw = sum(w for _, w in a["period"]) or 1
        period = sum(p * w for p, w in a["period"]) / pw if a["period"] else None
        res = dict(T=T, steps=a["steps"], step_ms=period, mode=sorted(a["mode"]), spans={})
        for n, s in a["spans"].items():
            d = dict(calls_per_step=s["n"] / steps, host_ms=s["host_excl_ms"] / steps,
                     host_incl_ms=s["host_ms"] / steps)
            if s.get("dev_ms"):
                d["dev_ms"] = s["dev_excl_ms"] / steps
                d["dev_incl_ms"] = s["dev_ms"] / steps
                d["host_over_dev"] = (s["host_excl_ms"] / s["dev_excl_ms"]) if s["dev_excl_ms"] > 0 else None
            if s.get("wall_ms"):
                d["kern_ms"] = s["wall_ms"] / steps
                d["lead_ms"] = s["lead_ms"] / steps
            res["spans"][n] = d
        rows.append(res)
    return rows


def fmt(v, w=8, p=3):
    return f"{'-':>{w}}" if v is None else f"{v:>{w}.{p}f}"


def render(rows, nranks, over=None):
    out = []
    for r in rows:
        out.append(f"\nT={r['T']} rows/step   steps={r['steps']}   mode={','.join(r['mode'])}   "
                   f"step wall {fmt(r['step_ms'], 0, 2).strip()} ms (per rank mean of {nranks} rank(s) summed windows)")
        out.append(f"  {'span':<16}{'calls/step':>11}{'host ms':>9}{'dev ms':>9}{'h/d':>7}{'kern ms':>9}{'lead ms':>9}")
        order = sorted(r["spans"].items(), key=lambda kv: -(kv[1].get("dev_ms") or kv[1].get("kern_ms") or kv[1]["host_ms"]))
        for n, d in order:
            out.append(f"  {n:<16}{d['calls_per_step']:>11.1f}{fmt(d['host_ms'])}{fmt(d.get('dev_ms'))}"
                       f"{fmt(d.get('host_over_dev'), 7, 2)}{fmt(d.get('kern_ms'))}{fmt(d.get('lead_ms'))}")
        top = r["spans"].get("run_qsa")
        if top and r["step_ms"]:
            inc = top.get("dev_incl_ms")
            out.append(f"  QSA body per step (inclusive): host {top['host_incl_ms']:.2f} ms"
                       + (f", dev {inc:.2f} ms" if inc is not None else "")
                       + f"  | step wall {r['step_ms']:.2f} ms"
                       + (f"  | dev share {100 * inc / r['step_ms']:.0f}%" if inc is not None else ""))
    if over is not None:
        out.append(f"\ntimer overhead: {over:.2f} ms per step per rank (all spans, both sides of every call); the host "
                   "columns above already exclude it from the parent spans, the step wall includes it.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--rank", default="0")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    recs = load(a.paths, a.rank)
    if not recs:
        print("no records", file=sys.stderr)
        return 1
    agg, nranks, over = merge(recs)
    rows = table(agg)
    print(json.dumps(rows, indent=1) if a.json else render(rows, nranks, over))
    return 0


if __name__ == "__main__":
    sys.exit(main())

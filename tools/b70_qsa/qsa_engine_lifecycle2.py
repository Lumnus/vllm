#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B70-QSA-P: prefix / offload lifecycle of a running sparse engine, judged against a same-run noise floor.

Replaces the fixed thresholds of qsa_engine_lifecycle.py (0.05 on compressed rows, 0.1 nat near-tie). Both engines
(sparse and dense) are run-to-run nondeterministic from about token 512, so a cold run and a warm run of the same
prompt differ no matter what the cache does, and the fixed thresholds fail every path (B2b §4). This driver
measures the noise first and asks a different question of every cache path: *is it distinguishable from a second
cold run?*

For every prompt it runs ``--floor-runs`` cold runs with fresh cache salts. Run 0 is the reference; all pairs of
cold runs give the noise floor. A cache path (GPU hit, CPU reload, junction, cancel + hit, interleave) is compared
with the reference by the same statistics:

  compressed rows (per dumped layer, per-row max |diff|)
    exact zone   rows [0, g) are copied from the reference run when the path reports g reused rows (hit tokens /
                 compress ratio): they must be bit-identical. With no reuse the zone is rows [0, onset), where onset
                 is the first row at which any pair of cold runs differs: before it the engine is deterministic,
                 so the path must be too.
    noise zone   the remaining rows: q50 and q90 of the per-row diff, and the last-row selection Jaccard, must not
                 be worse than the worst cold pair (plus --rel-margin and a small absolute term). max and q99 are
                 reported, and gate only with --strict-tail (single rows are too noisy for a few pairs).
    zero rows    an all-zero row that the cold runs do not have is a failure.
  greedy tokens
    identical up to the first divergence of the cold pairs (the path may not diverge earlier than noise does),
    then the mean top-1 logprob and the mean top-1/top-2 margin of the continuation must lie within the cold runs'
    spread (plus a margin). The old "near tie and in the base top-5" test is kept as information.

Arm comparison: ``--compare lifeA.json lifeB.json`` prints, per path, reused tokens, exactness and the noise-zone
statistics as a ratio to each arm's own floor, so "block drop on" and "off" can be read side by side.

Engine env (all default off): B70_QSA_DEBUG_DUMP=<dir> B70_QSA_DEBUG_DUMP_LAYERS=0,23,47 B70_QSA_TOPK=torch.
Exit status is non-zero if any path FAILs. A path with no moved prefix-cache metric is INCONCLUSIVE, not PASS.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import torch

# ------------------------------------------------------------------------------------------ http engine


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return r.read().decode()


METRIC = re.compile(r"^(vllm:(?:prefix_cache_\w+|external_prefix_cache_\w+|prompt_tokens_by_source\w*))"
                    r"(\{[^}]*\})?\s+([0-9.eE+-]+)$")


def metrics(base):
    out = {}
    for line in get(base, "/metrics").splitlines():
        m = METRIC.match(line)
        if m:
            labels = re.sub(r'(model_name|engine)="[^"]*",?', "", m.group(2) or "")
            out[m.group(1) + labels] = float(m.group(3))
    return out


def delta(after, before):
    return {k: after[k] - before.get(k, 0.0) for k in after if after[k] - before.get(k, 0.0)}


class Engine:
    def __init__(self, base, dump_dir):
        self.base, self.dump = base, Path(dump_dir)
        self.model = json.loads(get(base, "/v1/models"))["data"][0]["id"]

    def tokens(self, n, seed):
        words, i = [], 0
        while len(words) < n * 2:
            words.append(f"Record {seed}-{i}: item {i * 7919 % 10007} weighs {i * 104729 % 997} units "
                         f"and sits beside shelf {i % 61}.")
            i += 1
        ids = post(self.base, "/tokenize", {"model": self.model, "prompt": " ".join(words)})["tokens"]
        assert len(ids) >= n, (len(ids), n)
        return ids[:n]

    def complete(self, ids, salt, max_tokens=96):
        before_files = set(self.dump.glob("qsa-L*.pt")) if self.dump.exists() else set()
        before = metrics(self.base)
        res = post(self.base, "/v1/completions", {
            "model": self.model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 5,
            "cache_salt": salt, "ignore_eos": True})
        time.sleep(0.5)
        files = sorted(set(self.dump.glob("qsa-L*.pt")) - before_files)
        lp = res["choices"][0]["logprobs"]
        return {"tokens": lp["tokens"],
                "top": [sorted(d.items(), key=lambda kv: -kv[1]) for d in lp["top_logprobs"]],
                "metrics": delta(metrics(self.base), before),
                "dumps": self.load_dumps(files, len(ids))}

    @staticmethod
    def load_dumps(files, seq_len):
        found = {}
        for f in files:
            d = torch.load(f, weights_only=False)
            for r in d["requests"]:
                if r["seq_len"] == seq_len:
                    found[d["layer"]] = r  # last matching step wins
        return found

    def cancel_after_first_chunk(self, ids, salt):
        body = {"model": self.model, "prompt": ids, "max_tokens": 200, "stream": True, "cache_salt": salt}
        req = urllib.request.Request(self.base + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            r.readline()  # first chunk, then drop the connection


# ------------------------------------------------------------------------------------------ statistics


def _q(v: torch.Tensor, p: float) -> float:
    return float(torch.quantile(v.float(), p)) if v.numel() else 0.0


def first_over(diff: torch.Tensor, eps: float) -> int:
    idx = (diff > eps).nonzero().flatten()
    return int(idx[0]) if idx.numel() else int(diff.numel())


def row_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).abs().amax(dim=1)


def jaccard(a, b) -> float:
    sa, sb = set(a.tolist()) - {-1}, set(b.tolist()) - {-1}
    return len(sa & sb) / max(1, len(sa | sb))


def zone_stats(diff: torch.Tensor, start: int) -> dict:
    z = diff[start:]
    return {"rows": int(z.numel()), "q50": _q(z, 0.5), "q90": _q(z, 0.9), "q99": _q(z, 0.99),
            "max": float(z.max()) if z.numel() else 0.0}


def upper_bound(vals, rel, abs_):
    return max(vals) * (1 + rel) + abs_


def compare_layer(ref, cand, floor_runs, g, rel=0.5, abs_eps=1e-3, exact_eps=0.0, strict_tail=False):
    """ref/cand: dump dicts ({'compressed': [rows, dim], 'last_row_selected': [k]}); floor_runs: dumps of the other
    cold runs of the same prompt, ref first. g = rows reused from ref by this path (0 if none)."""
    runs = floor_runs
    if any(r["compressed"].shape != ref["compressed"].shape for r in runs) or \
            cand["compressed"].shape != ref["compressed"].shape:
        return {"ok": False, "why": "shape mismatch", "shapes": [tuple(r["compressed"].shape) for r in runs]
                + [tuple(cand["compressed"].shape)]}
    pairs = list(itertools.combinations(runs, 2))
    f_diff = [row_diff(a["compressed"], b["compressed"]) for a, b in pairs]
    onset = min((first_over(d, exact_eps) for d in f_diff), default=ref["compressed"].shape[0])
    c_diff = row_diff(ref["compressed"], cand["compressed"])
    c_onset = first_over(c_diff, exact_eps)
    n = c_diff.numel()
    need = g if g else onset              # rows that must be bit-identical
    out = {"rows": n, "floor_onset_row": onset, "cand_onset_row": c_onset, "exact_zone_rows": need,
           "reused_rows": g, "exact_ok": c_onset >= need}
    start = need
    cs = zone_stats(c_diff, start)
    fs = [zone_stats(d, start) for d in f_diff]
    out["cand"], out["floor"] = cs, fs
    zero_c = int((cand["compressed"].abs().amax(dim=1) == 0).sum())
    zero_f = max(int((r["compressed"].abs().amax(dim=1) == 0).sum()) for r in runs)
    cj = jaccard(ref["last_row_selected"], cand["last_row_selected"])
    fj = [jaccard(a["last_row_selected"], b["last_row_selected"]) for a, b in pairs]
    out["jaccard"], out["floor_jaccard"] = round(cj, 4), [round(x, 4) for x in fj]
    out["all_zero_rows"], out["floor_all_zero_rows"] = zero_c, zero_f
    checks = {"exact_zone": out["exact_ok"], "zero_rows": zero_c <= zero_f}
    ratios = {}
    for k in ("q50", "q90") + (("q99", "max") if strict_tail else ()):
        b = upper_bound([s[k] for s in fs], rel, abs_eps)
        checks[k] = cs[k] <= b
    for k in ("q50", "q90", "q99", "max"):
        m = max(s[k] for s in fs)
        ratios[k] = round(cs[k] / m, 3) if m > 0 else (0.0 if cs[k] == 0 else None)
    checks["jaccard"] = cj >= min(fj) - max(0.08, 0.5 * (max(fj) - min(fj)))
    out["ratio_to_worst_floor"] = ratios
    out["checks"] = checks
    out["ok"] = all(checks.values())
    return out


def _tok_stats(r):
    """(mean, standard error) of the top-1 logprob and of the top-1/top-2 margin over the generated tokens."""
    top1 = [t[0][1] for t in r["top"] if t]
    margin = [t[0][1] - t[1][1] for t in r["top"] if len(t) > 1]

    def ms(v):
        if not v:
            return 0.0, 0.0
        return statistics.fmean(v), (statistics.pstdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0)

    return ms(top1), ms(margin)


def first_div(a, b) -> int:
    for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
        if x != y:
            return i
    return min(len(a["tokens"]), len(b["tokens"]))


def compare_tokens(ref, cand, floor_runs, near_margin=0.1, rel=0.25, abs_eps=0.05, early_rel=0.25):
    """floor_runs: the cold runs of the same prompt, ref first. The divergence of the candidate from ref is compared
    with the divergence of the other cold runs from ref (the same kind of pair); with 2-4 samples a strict
    "not earlier than the earliest" would fail one run in n by chance, so the candidate fails only if it diverges
    before ``early_rel`` of the earliest cold divergence (minus two tokens): this catches a divergence at the very
    start of the continuation and nothing subtler. The continuation statistics carry the rest."""
    d_floor = [first_div(ref, r) for r in floor_runs[1:]]
    d_c = first_div(ref, cand)
    n = min(len(ref["tokens"]), len(cand["tokens"]))
    exact = d_c >= n and len(ref["tokens"]) == len(cand["tokens"])
    out = {"exact": exact, "first_divergence": None if exact else d_c, "floor_first_divergence": d_floor}
    if not exact:
        top = ref["top"][d_c]
        out["near_tie_vs_ref"] = len(top) > 1 and top[0][1] - top[1][1] < near_margin
        out["cand_token_in_ref_top5"] = cand["tokens"][d_c] in [t for t, _ in top]
    cm = _tok_stats(cand)
    fms = [_tok_stats(r) for r in floor_runs]
    checks = {"not_earlier_than_floor": exact or d_c >= early_rel * min(d_floor) - 2}
    for i, name in enumerate(("mean_top1_logprob", "mean_top1_top2_margin")):
        vals = [f[i][0] for f in fms]
        # a mean over a few dozen tokens has a sampling error; with 2-4 cold runs their range alone under-covers it
        se = max([f[i][1] for f in fms] + [cm[i][1]])
        tol = rel * (max(vals) - min(vals)) + 3 * se + abs_eps
        out[name] = {"cand": round(cm[i][0], 4), "floor": [round(v, 4) for v in vals], "se": round(se, 4)}
        checks[name] = min(vals) - tol <= cm[i][0] <= max(vals) + tol
    out["checks"] = checks
    out["ok"] = all(checks.values())
    return out


def hit_tokens(m):
    return int(sum(v for k, v in m.items() if "prompt_tokens_by_source" in k and ("cache" in k or "external" in k)))


# ------------------------------------------------------------------------------------------ the run


def run(eng, a):
    salt = f"qsaq-{uuid.uuid4().hex[:8]}"
    skip = set(filter(None, a.skip.split(",")))
    report = {"label": a.label, "prompt_tokens": a.prompt_tokens, "floor_runs": a.floor_runs, "paths": {}}
    P = eng.tokens(a.prompt_tokens, 1)

    def colds(ids, tag):
        """floor_runs cold runs with fresh salts; [0] is the reference."""
        return [eng.complete(ids, f"{salt}-{tag}{i}", a.max_tokens) for i in range(a.floor_runs)]

    def judge(name, ref, floor, cand, expect_hit=True):
        h = hit_tokens(cand["metrics"])
        g = h // a.compress_ratio
        dumps, layers = {}, sorted(set(ref["dumps"]) & set(cand["dumps"]) & set.intersection(
            *[set(f["dumps"]) for f in floor]))
        for L in layers:
            dumps[str(L)] = compare_layer(ref["dumps"][L], cand["dumps"][L], [f["dumps"][L] for f in floor], g,
                                          rel=a.rel_margin, strict_tail=a.strict_tail)
        dok = bool(layers) and all(d["ok"] for d in dumps.values())
        tk = compare_tokens(ref, cand, floor)
        entry = {"hit_tokens_reported": h, "metrics": cand["metrics"], "tokens": tk, "dumps": dumps,
                 "dumps_ok": dok}
        if not layers:
            entry["verdict"], entry["why"] = "INCONCLUSIVE", "no dump for this prompt (engine env?)"
        elif expect_hit and not cand["metrics"]:
            entry["verdict"], entry["why"] = "INCONCLUSIVE", "no prefix-cache metric moved; did the hit happen?"
        else:
            entry["verdict"] = "PASS" if (tk["ok"] and dok) else "FAIL"
        report["paths"][name] = entry
        brief = {"verdict": entry["verdict"], "hit": h, "tokens_ok": tk["ok"], "tokens": tk["checks"],
                 "dumps": {L: {"exact": d.get("exact_ok"), **(d.get("checks") or {})} for L, d in dumps.items()}}
        print(name, json.dumps(brief, default=str), flush=True)

    cold = colds(P, "cold")
    ref = cold[0]
    report["paths"]["cold"] = {"metrics": ref["metrics"], "dump_layers": sorted(ref["dumps"]),
                               "floor_first_divergence": [first_div(ref, x) for x in cold[1:]]}
    if "gpu_hit" not in skip:
        judge("gpu_hit", ref, cold, eng.complete(P, f"{salt}-cold0", a.max_tokens))
    if "cpu_reload" not in skip:
        n = 0
        while n < a.evict_tokens:
            eng.complete(eng.tokens(a.filler_tokens, 1000 + n), f"fill-{uuid.uuid4().hex}", max_tokens=1)
            n += a.filler_tokens
        judge("cpu_reload", ref, cold, eng.complete(P, f"{salt}-cold0", a.max_tokens))
    if "junction" not in skip:
        J = P[: a.junction_tokens] + eng.tokens(1500, 2)
        jc = colds(J, "jcold")
        # the junction prompt shares an unaligned prefix with P: run it under P's salt so it can hit P's blocks
        judge("junction", jc[0], jc, eng.complete(J, f"{salt}-cold0", a.max_tokens), expect_hit=False)
    if "cancel" not in skip:
        eng.cancel_after_first_chunk(eng.tokens(30000, 3), f"{salt}-cancel")
        judge("cancel_then_hit", ref, cold, eng.complete(P, f"{salt}-cold0", a.max_tokens))
    if "interleave" not in skip:
        Q = eng.tokens(a.prompt_tokens + 401, 4)
        stop = threading.Event()

        def long_decode():
            while not stop.is_set():
                eng.complete(eng.tokens(2000, 5), f"{salt}-bg", max_tokens=400)

        t = threading.Thread(target=long_decode, daemon=True)
        t.start()
        time.sleep(2)
        qc = colds(Q, "qcold")                      # floor under the same background load
        q_hit = eng.complete(Q, f"{salt}-qcold0", a.max_tokens)
        stop.set()
        judge("interleave_hit", qc[0], qc, q_hit)
    bad = [k for k, v in report["paths"].items() if v.get("verdict") == "FAIL"]
    unk = [k for k, v in report["paths"].items() if v.get("verdict") == "INCONCLUSIVE"]
    report["failed"], report["inconclusive"] = bad, unk
    return report


# ------------------------------------------------------------------------------------------ arm comparison


def compare_arms(ra: dict, rb: dict) -> str:
    """Side-by-side of two run reports (e.g. EAGLE block drop on vs off)."""
    lines = [f"{'path':<16}{'arm':<10}{'verdict':<13}{'hit tok':>8}{'exact':>7}{'q50/fl':>8}{'q90/fl':>8}{'jacc':>7}"
             f"{'div@':>6}"]
    for name in sorted(set(ra["paths"]) | set(rb["paths"])):
        for arm, r in ((ra["label"], ra), (rb["label"], rb)):
            p = r["paths"].get(name)
            if not p or "tokens" not in p:
                continue
            ds = list(p["dumps"].values())
            ex = "yes" if ds and all(d.get("exact_ok") for d in ds) else "NO"
            q50 = max((d.get("ratio_to_worst_floor", {}).get("q50") or 0) for d in ds) if ds else 0
            q90 = max((d.get("ratio_to_worst_floor", {}).get("q90") or 0) for d in ds) if ds else 0
            jac = min((d.get("jaccard", 1) for d in ds), default=1)
            dv = p["tokens"]["first_divergence"]
            lines.append(f"{name:<16}{arm:<10}{p['verdict']:<13}{p['hit_tokens_reported']:>8}{ex:>7}{q50:>8.2f}"
                         f"{q90:>8.2f}{jac:>7.3f}{'-' if dv is None else dv:>6}")
    lines.append("exact: reused rows bit-identical to the reference. q50/q90 ratio: noise-zone diff over the worst "
                 "cold pair (<= 1.5 is inside the floor). hit tok: cached prompt tokens the engine reported.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--dump-dir")
    ap.add_argument("--label")
    ap.add_argument("--prompt-tokens", type=int, default=6150)  # not 832-aligned
    ap.add_argument("--junction-tokens", type=int, default=4101)
    ap.add_argument("--evict-tokens", type=int, default=420000, help="filler volume; must exceed the engine's GPU KV tokens")
    ap.add_argument("--filler-tokens", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=96, help="more tokens, better token statistics")
    ap.add_argument("--floor-runs", type=int, default=3, help="cold runs per prompt (>= 2); pairs = n(n-1)/2")
    ap.add_argument("--rel-margin", type=float, default=0.5)
    ap.add_argument("--strict-tail", action="store_true", help="q99 and max also gate")
    ap.add_argument("--compress-ratio", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", default="", help="comma list: gpu_hit,cpu_reload,junction,cancel,interleave")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    a = ap.parse_args(argv)
    if a.compare:
        print(compare_arms(*(json.loads(Path(p).read_text()) for p in a.compare)))
        return 0
    if not (a.dump_dir and a.label):
        ap.error("--dump-dir and --label are required")
    if a.floor_runs < 2:
        ap.error("--floor-runs must be >= 2")
    report = run(Engine(a.base, a.dump_dir), a)
    text = json.dumps(report, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text)
    print("FAILED:" if report["failed"] else "ALL OK", report["failed"], "inconclusive:", report["inconclusive"])
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

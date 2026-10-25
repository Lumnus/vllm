#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""B70-QSA-Q test 2: prefix / offload lifecycle of a running sparse engine.

Drives a dev engine over HTTP and compares, for the same token prompt reached
through different cache paths, (a) the greedy continuation with top-5
logprobs and (b) the QSA compressed-key rows the engine dumps for that prompt
(``B70_QSA_DEBUG_DUMP``). Paths:

  cold        fresh cache_salt, nothing cached
  gpu_hit     same salt again: GPU prefix hit
  cpu_reload  after filler traffic evicts the GPU copy: CPU-tier reload
  junction    a prompt sharing an unaligned prefix with the cold one, vs its
              own cold run
  cancel      a streamed request cancelled after its first chunk, its slot
              reused, then the prompt again
  interleave  a long decode runs while a new prompt goes cold, then hits

Engine env for this script (all default off):
  B70_QSA_DEBUG_DUMP=<dir>  B70_QSA_DEBUG_DUMP_LAYERS=0,23,47
  B70_QSA_DEBUG_RING=warn   (check `journalctl -u ... | grep B70-QSA-DEBUG`)
  B70_QSA_TOPK=torch        (selection must be deterministic to compare)

Run it twice, once against the def with "disable_eagle_block_drop": true in
--speculative-config and once without, and pass --label accordingly. The
EAGLE trailing-block drop reaches every KV group on our tree (the DFlash draft
group is not flagged is_eagle_group), QSA ring and compressed groups included.
Exit status is non-zero if any comparison fails.
"""

import argparse
import json
import re
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import torch


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return r.read().decode()


METRIC = re.compile(
    r"^(vllm:(?:prefix_cache_\w+|external_prefix_cache_\w+|"
    r"prompt_tokens_by_source\w*))(\{[^}]*\})?\s+([0-9.eE+-]+)$"
)


def metrics(base):
    out = {}
    for line in get(base, "/metrics").splitlines():
        m = METRIC.match(line)
        if m:
            labels = re.sub(r'(model_name|engine)="[^"]*",?', "", m.group(2) or "")
            out[m.group(1) + labels] = float(m.group(3))
    return out


def delta(after, before):
    return {
        k: after[k] - before.get(k, 0.0) for k in after if after[k] - before.get(k, 0.0)
    }


class Engine:
    def __init__(self, base, dump_dir):
        self.base, self.dump = base, Path(dump_dir)
        self.model = json.loads(get(base, "/v1/models"))["data"][0]["id"]

    def tokens(self, n, seed):
        words = []
        i = 0
        while len(words) < n * 2:
            words.append(
                f"Record {seed}-{i}: item {i * 7919 % 10007} weighs "
                f"{i * 104729 % 997} units and sits beside shelf {i % 61}."
            )
            i += 1
        ids = post(
            self.base, "/tokenize", {"model": self.model, "prompt": " ".join(words)}
        )["tokens"]
        assert len(ids) >= n, (len(ids), n)
        return ids[:n]

    def complete(self, ids, salt, max_tokens=24):
        before_files = set(self.dump.glob("qsa-L*.pt")) if self.dump.exists() else set()
        before = metrics(self.base)
        body = {
            "model": self.model,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": 5,
            "cache_salt": salt,
            "ignore_eos": True,
        }
        res = post(self.base, "/v1/completions", body)
        time.sleep(0.5)
        files = sorted(set(self.dump.glob("qsa-L*.pt")) - before_files)
        choice = res["choices"][0]
        lp = choice["logprobs"]
        return {
            "tokens": lp["tokens"],
            "top": [
                sorted(d.items(), key=lambda kv: -kv[1]) for d in lp["top_logprobs"]
            ],
            "metrics": delta(metrics(self.base), before),
            "dumps": self.load_dumps(files, len(ids)),
        }

    @staticmethod
    def load_dumps(files, seq_len):
        found = {}
        for f in files:
            d = torch.load(f, weights_only=False)
            for r in d["requests"]:
                if r["seq_len"] == seq_len:
                    found[d["layer"]] = r  # last matching step wins
        return found


def compare_tokens(base, cand, margin=0.1):
    for i, (a, b) in enumerate(zip(base["tokens"], cand["tokens"])):
        if a == b:
            continue
        top = base["top"][i]
        near = len(top) > 1 and top[0][1] - top[1][1] < margin
        in_top = b in [t for t, _ in top]
        return {
            "exact": False,
            "first_divergence": i,
            "near_tie": near,
            "candidate_in_base_top5": in_top,
            "ok": near and in_top,
        }
    ok = len(base["tokens"]) == len(cand["tokens"])
    return {"exact": ok, "ok": ok}


def compare_dumps(base, cand, hit_tokens=0, atol=0.05):
    out, ok = {}, True
    if not base["dumps"] or not cand["dumps"]:
        return {"ok": False, "why": "no dump for this prompt (engine env?)"}
    for layer, b in base["dumps"].items():
        c = cand["dumps"].get(layer)
        if c is None:
            out[layer] = "missing"
            ok = False
            continue
        rb, rc = b["compressed"], c["compressed"]
        if rb.shape != rc.shape:
            out[layer] = f"shape {tuple(rb.shape)} vs {tuple(rc.shape)}"
            ok = False
            continue
        diff = (rb - rc).abs().amax(dim=1)
        g = hit_tokens // 4
        bad = (diff > atol).nonzero().flatten().tolist()
        zero_rows = int((rc.abs().amax(dim=1) == 0).sum())
        sb = set(b["last_row_selected"].tolist()) - {-1}
        sc = set(c["last_row_selected"].tolist()) - {-1}
        jac = len(sb & sc) / max(1, len(sb | sc))
        out[layer] = {
            "rows": rb.shape[0],
            "max_diff_reused_prefix": float(diff[:g].max()) if g else None,
            "max_diff_all": float(diff.max()),
            "rows_over_atol": bad[:10],
            "num_rows_over_atol": len(bad),
            "all_zero_rows": zero_rows,
            "last_row_selection_jaccard": round(jac, 4),
        }
        ok = ok and not bad and zero_rows == 0
    out["ok"] = ok
    return out


def hit_tokens(m):
    """Cached prompt tokens from the per-source counter, if exposed."""
    total = 0.0
    for k, v in m.items():
        if "prompt_tokens_by_source" in k and ("cache" in k or "external" in k):
            total += v
    return int(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=6150)  # not 832-aligned
    ap.add_argument("--junction-tokens", type=int, default=4101)
    ap.add_argument(
        "--evict-tokens",
        type=int,
        default=420000,
        help="filler volume; must exceed the engine's GPU KV tokens",
    )
    ap.add_argument("--filler-tokens", type=int, default=12000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", default="", help="comma list of paths to skip")
    a = ap.parse_args()
    eng = Engine(a.base, a.dump_dir)
    skip = set(a.skip.split(","))
    salt = f"qsaq-{uuid.uuid4().hex[:8]}"
    P = eng.tokens(a.prompt_tokens, 1)
    report = {"label": a.label, "prompt_tokens": len(P), "paths": {}}

    def record(name, base, cand, expect_hit=True):
        h = hit_tokens(cand["metrics"])
        entry = {
            "metrics": cand["metrics"],
            "hit_tokens_reported": h,
            "tokens": compare_tokens(base, cand),
            "dumps": compare_dumps(base, cand, h),
        }
        if expect_hit and not cand["metrics"]:
            entry["warning"] = "no prefix-cache metric moved; did the hit happen?"
        report["paths"][name] = entry
        print(
            name,
            json.dumps({k: v for k, v in entry.items() if k != "metrics"}, default=str)[
                :600
            ],
            flush=True,
        )

    cold = eng.complete(P, salt)
    report["paths"]["cold"] = {
        "metrics": cold["metrics"],
        "dump_layers": sorted(cold["dumps"]),
    }
    if "gpu_hit" not in skip:
        record("gpu_hit", cold, eng.complete(P, salt))
    if "cpu_reload" not in skip:
        n = 0
        while n < a.evict_tokens:
            eng.complete(
                eng.tokens(a.filler_tokens, 1000 + n),
                f"fill-{uuid.uuid4().hex}",
                max_tokens=1,
            )
            n += a.filler_tokens
        record("cpu_reload", cold, eng.complete(P, salt))
    if "junction" not in skip:
        J = P[: a.junction_tokens] + eng.tokens(1500, 2)
        j_cold = eng.complete(J, f"{salt}-jcold")
        record("junction", j_cold, eng.complete(J, salt))
    if "cancel" not in skip:
        body = {
            "model": eng.model,
            "prompt": eng.tokens(30000, 3),
            "max_tokens": 200,
            "stream": True,
            "cache_salt": f"{salt}-cancel",
        }
        req = urllib.request.Request(
            a.base + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1800) as r:
            r.readline()  # first chunk, then drop the connection
        record("cancel_then_hit", cold, eng.complete(P, salt))
    if "interleave" not in skip:
        Q = eng.tokens(a.prompt_tokens + 401, 4)
        stop = threading.Event()

        def long_decode():
            while not stop.is_set():
                eng.complete(eng.tokens(2000, 5), f"{salt}-bg", max_tokens=400)

        t = threading.Thread(target=long_decode, daemon=True)
        t.start()
        time.sleep(2)
        q_cold = eng.complete(Q, f"{salt}-q")
        q_hit = eng.complete(Q, f"{salt}-q")
        stop.set()
        record("interleave_hit", q_cold, q_hit)
    fails = [
        k
        for k, v in report["paths"].items()
        if isinstance(v, dict)
        and "tokens" in v
        and not (v["tokens"]["ok"] and v["dumps"].get("ok"))
    ]
    report["failed"] = fails
    text = json.dumps(report, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text)
    print("FAILED:" if fails else "ALL OK", fails)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""CPU tests for qsa_engine_lifecycle2.py with synthetic traces (no engine, no HTTP).

The fake engine reproduces what B2b measured: a cold run is bit-identical up to row ~132 (token ~528) and noisy
after it, and its greedy tokens diverge from another cold run within the first tens of tokens. A cache hit copies
the reused rows exactly and recomputes the rest with the same noise. Knobs inject real defects."""
import copy
import hashlib
import importlib.util
import json
import os
import random
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in ("qsa_engine_lifecycle2.py", "../../../tools/b70_qsa/qsa_engine_lifecycle2.py"):
    _p = os.path.join(_HERE, _c)
    if os.path.exists(_p):
        _spec = importlib.util.spec_from_file_location("qsa_engine_lifecycle2", _p)
        L = importlib.util.module_from_spec(_spec)
        sys.modules["qsa_engine_lifecycle2"] = L
        _spec.loader.exec_module(L)
        break
else:
    raise FileNotFoundError("qsa_engine_lifecycle2.py")

D = 16
ONSET = 132


def _seed(ids, extra=""):
    return int(hashlib.sha1((",".join(map(str, ids[:50])) + str(len(ids)) + extra).encode()).hexdigest()[:8], 16)


class FakeEngine:
    """layers dumped: 23, 47. noise: scale of the nondeterminism after ONSET."""

    model = "fake"

    def __init__(self, noise=0.5, seed=0, hit_noise=None, corrupt=None, early_onset=None, tok_noise=True,
                 zero_rows=False, no_dump=False, cache_hits=True, block=832):
        self.noise, self.rng = noise, random.Random(seed)
        self.hit_noise = noise if hit_noise is None else hit_noise
        self.corrupt, self.early_onset, self.tok_noise = corrupt, early_onset, tok_noise
        self.zero_rows, self.no_dump, self.cache_hits, self.block = zero_rows, no_dump, cache_hits, block
        self.cache = {}  # salt -> {tuple(prefix ids): {layer: rows}}
        self.log = []

    def tokens(self, n, seed):
        r = random.Random(seed)
        return [r.randrange(1000, 60000) for _ in range(n)]

    def _truth(self, ids, layer):
        g = torch.Generator().manual_seed(_seed(ids, f"L{layer}"))
        return torch.randn(len(ids) // 4 + 1, D, generator=g)

    def _tokens(self, ids, max_tokens, noisy):
        base = random.Random(_seed(ids, "tok"))
        div = self.rng.randrange(2, 30) if (noisy and self.tok_noise) else max_tokens
        toks, tops = [], []
        for i in range(max_tokens):
            top1 = f"t{base.randrange(50)}" if i < div else f"x{self.rng.randrange(50)}"
            lp1 = -(0.2 + 0.3 * self.rng.random())
            lp2 = lp1 - (0.5 + self.rng.random())
            toks.append(top1)
            tops.append([(top1, lp1), (f"y{i}", lp2), (f"z{i}", lp2 - 1), (f"w{i}", lp2 - 2), (f"v{i}", lp2 - 3)])
        return toks, tops

    def complete(self, ids, salt, max_tokens=96):
        hit = 0
        stored = self.cache.get(salt)
        if stored and self.cache_hits:
            for k, rows in stored.items():
                if tuple(ids[:len(k)]) == k and len(k) > hit:
                    hit = len(k)
        hit = hit // self.block * self.block
        g = hit // 4
        dumps = {}
        for layer in (23, 47):
            truth = self._truth(ids, layer)
            noise_scale = self.hit_noise if hit else self.noise
            on = self.early_onset or ONSET
            n = torch.randn_like(truth) * noise_scale
            n[:on] = 0
            rows = truth + n
            if hit:
                src = stored[next(k for k in stored if tuple(ids[:len(k)]) == k)][layer]
                rows[:g] = src[:g]
                if self.corrupt == "reused":
                    rows[700] += 0.5
            if self.corrupt == "tail" and hit:
                rows[g + 50:] += torch.randn(rows.shape[0] - g - 50, D) * 3
            if self.zero_rows:
                rows[-40:] = 0
            keep = set(range(120))
            for _ in range(self.rng.randint(4, 14) if noise_scale else 0):  # a few swapped selections (J ~ 0.9)
                keep.discard(self.rng.randrange(120))
                keep.add(self.rng.randrange(120, 400))
            sel = torch.tensor(sorted(keep))
            dumps[layer] = {"compressed": rows, "last_row_selected": sel, "seq_len": len(ids)}
        if self.cache_hits:
            self.cache.setdefault(salt, {})[tuple(ids)] = {L_: d["compressed"] for L_, d in dumps.items()}
        toks, tops = self._tokens(ids, max_tokens, noisy=True)
        m = {"vllm:prompt_tokens_by_source_total{source=\"local_cache_hit\"}": float(hit)} if hit else {}
        return {"tokens": toks, "top": tops, "metrics": m, "dumps": {} if self.no_dump else dumps}

    def cancel_after_first_chunk(self, ids, salt):
        self.log.append("cancel")


class Args:
    label = "t"
    prompt_tokens = 6150
    junction_tokens = 4101
    evict_tokens = 24000
    filler_tokens = 12000
    max_tokens = 64
    floor_runs = 3
    rel_margin = 0.5
    strict_tail = False
    compress_ratio = 4
    skip = "interleave"


def go(eng, **kw):
    a = Args()
    for k, v in kw.items():
        setattr(a, k, v)
    return L.run(eng, a)


def test_noise_only_passes_every_path():
    rep = go(FakeEngine(noise=0.5, seed=1))
    assert rep["failed"] == []
    for name, p in rep["paths"].items():
        if "verdict" in p:
            assert p["verdict"] in ("PASS", "INCONCLUSIVE"), (name, p)
    g = rep["paths"]["gpu_hit"]
    assert g["verdict"] == "PASS" and g["hit_tokens_reported"] == 5824
    lay = g["dumps"]["23"]
    assert lay["exact_ok"] and lay["reused_rows"] == 1456 and lay["floor_onset_row"] == ONSET


def test_old_fixed_thresholds_would_fail_the_same_traces():
    """The point of the floor: at noise 0.5 a cold-vs-cold pair is far over the old 0.05 atol."""
    e = FakeEngine(noise=0.5, seed=2)
    P = e.tokens(6150, 1)
    a, b = e.complete(P, "a"), e.complete(P, "b")
    diff = L.row_diff(a["dumps"][23]["compressed"], b["dumps"][23]["compressed"])
    assert int((diff > 0.05).sum()) > 1000


def test_corrupt_reused_row_fails():
    rep = go(FakeEngine(noise=0.5, seed=3, corrupt="reused"))
    p = rep["paths"]["gpu_hit"]
    assert p["verdict"] == "FAIL" and not p["dumps"]["23"]["exact_ok"]
    assert p["dumps"]["23"]["cand_onset_row"] == 700


def test_tail_noise_far_above_floor_fails():
    rep = go(FakeEngine(noise=0.3, seed=4, corrupt="tail"))
    p = rep["paths"]["gpu_hit"]
    assert p["verdict"] == "FAIL"
    assert not p["dumps"]["23"]["checks"]["q90"]


def test_zero_rows_fail():
    # zero rows only on the hit path: build floor runs clean, then switch the knob for the candidate
    e = FakeEngine(noise=0.5, seed=5)
    orig = e.complete
    calls = {"n": 0}

    def complete(ids, salt, max_tokens=96):
        calls["n"] += 1
        e.zero_rows = calls["n"] > 3  # the three cold runs are clean
        return orig(ids, salt, max_tokens)

    e.complete = complete
    p = go(e, skip="cpu_reload,junction,cancel,interleave")["paths"]["gpu_hit"]
    assert p["verdict"] == "FAIL" and not p["dumps"]["23"]["checks"]["zero_rows"]


def test_no_reuse_demands_identity_up_to_floor_onset():
    # a path with no reused rows that diverges at row 40, earlier than the cold runs do (132)
    e = FakeEngine(noise=0.5, seed=6)
    P = e.tokens(6150, 1)
    colds = [e.complete(P, f"c{i}") for i in range(3)]
    bad = FakeEngine(noise=0.5, seed=7, early_onset=40).complete(P, "x")
    lay = L.compare_layer(colds[0]["dumps"][23], bad["dumps"][23], [c["dumps"][23] for c in colds], 0)
    assert lay["exact_zone_rows"] == ONSET and lay["cand_onset_row"] == 40 and not lay["exact_ok"] and not lay["ok"]
    good = e.complete(P, "y")
    lay = L.compare_layer(colds[0]["dumps"][23], good["dumps"][23], [c["dumps"][23] for c in colds], 0)
    assert lay["exact_ok"] and lay["ok"]


def test_deterministic_engine_degenerates_to_exact():
    rep = go(FakeEngine(noise=0.0, seed=8, tok_noise=False))
    p = rep["paths"]["gpu_hit"]
    assert p["verdict"] == "PASS" and p["tokens"]["exact"]
    # and one wrong element in the noise zone is then a failure (floor is 0, bound is abs_eps only)
    e = FakeEngine(noise=0.0, seed=8, tok_noise=False, corrupt="tail")
    assert go(e)["paths"]["gpu_hit"]["verdict"] == "FAIL"


def test_token_stats_catch_a_degenerate_continuation():
    e = FakeEngine(noise=0.5, seed=9)
    P = e.tokens(6150, 1)
    colds = [e.complete(P, f"c{i}") for i in range(3)]
    cand = copy.deepcopy(e.complete(P, "c0"))
    # a candidate whose continuation is confidently wrong: very low top-1 logprob and tiny margins
    cand["top"] = [[(t, -3.0), ("a", -3.01), ("b", -3.5), ("c", -4), ("d", -5)] for t, _ in [x[0] for x in cand["top"]]]  # noqa: E501
    r = L.compare_tokens(colds[0], cand, colds)
    assert not r["ok"] and not r["checks"]["mean_top1_logprob"]
    assert L.compare_tokens(colds[0], colds[1], colds)["ok"]


def test_no_dump_and_no_hit_are_inconclusive_not_pass():
    rep = go(FakeEngine(no_dump=True, seed=10), skip="cpu_reload,junction,cancel,interleave")
    assert rep["paths"]["gpu_hit"]["verdict"] == "INCONCLUSIVE" and rep["failed"] == []
    rep = go(FakeEngine(cache_hits=False, seed=11), skip="cpu_reload,junction,cancel,interleave")
    assert rep["paths"]["gpu_hit"]["verdict"] == "INCONCLUSIVE"


def test_all_paths_run_and_json_roundtrips(tmp_path):
    rep = go(FakeEngine(noise=0.4, seed=12), skip="")  # interleave thread uses the fake as well
    assert set(rep["paths"]) >= {"cold", "gpu_hit", "cpu_reload", "junction", "cancel_then_hit", "interleave_hit"}
    json.loads(json.dumps(rep, default=str))


def test_compare_arms_table():
    a = go(FakeEngine(noise=0.5, seed=13), label="drop-on", skip="cpu_reload,junction,cancel,interleave")
    b = go(FakeEngine(noise=0.5, seed=14, block=416), label="drop-off", skip="cpu_reload,junction,cancel,interleave")
    t = L.compare_arms(a, b)
    assert "drop-on" in t and "drop-off" in t and "gpu_hit" in t and "q50/fl" in t


def test_cli_compare(tmp_path, capsys):
    a = go(FakeEngine(seed=15), label="A", skip="cpu_reload,junction,cancel,interleave")
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a, default=str))
    b = copy.deepcopy(a)
    b["label"] = "B"
    pb.write_text(json.dumps(b, default=str))
    assert L.main(["--compare", str(pa), str(pb)]) == 0
    assert "gpu_hit" in capsys.readouterr().out


def test_early_token_divergence_fails_and_late_does_not():
    mk = lambda toks: {"tokens": toks, "top": [[(x, -0.3), ("o", -1.5), ("p", -2), ("q", -3), ("r", -4)] for x in toks]}
    base = [f"t{i}" for i in range(40)]
    ref = mk(base)
    floor = [ref, mk(base[:25] + ["x"] * 15), mk(base[:30] + ["y"] * 10)]
    assert L.compare_tokens(ref, mk(base[:26] + ["z"] * 14), floor)["ok"]
    floor = [ref, mk(base[:30] + ["x"] * 10), mk(base[:35] + ["y"] * 5)]
    r = L.compare_tokens(ref, mk(["bad"] + base[1:]), floor)
    assert not r["ok"] and not r["checks"]["not_earlier_than_floor"]

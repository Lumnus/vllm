# SPDX-License-Identifier: Apache-2.0
"""CPU tests for qsa_optimer.py and qsa_optimer_report.py (no XPU, no vllm): a fake device with a work queue."""
import json
import os
import sys
import types

import pytest

import importlib.util  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, *candidates):
    """Load a sibling file, or the same file from the repo layout (tests/models/qwen4_exp -> vllm/..., benchmarks/...)."""
    for c in candidates:
        path = os.path.join(_HERE, c)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(name, path)
            m = importlib.util.module_from_spec(spec)
            sys.modules[name] = m
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(name)


qo = _load("qsa_optimer", "qsa_optimer.py", "../../../vllm/models/qwen4_exp/nvidia/qsa_optimer.py")
rep = _load("qsa_optimer_report", "qsa_optimer_report.py", "../../../benchmarks/kernels/qsa_optimer_report.py")


class FakeDev:
    """Host clock + one in-order device queue. host(ns) advances the host; launch(ns) enqueues device work."""

    def __init__(self):
        self.now = 0
        self.free = 0
        self.capture = False

    def clock(self):
        return self.now

    def host(self, ns):
        self.now += ns

    def launch(self, ns, host_ns=0):
        self.host(host_ns)
        self.free = max(self.free, self.now) + ns

    # backend API
    def event(self):
        return max(self.free, self.now)

    def elapsed_ms(self, a, b):
        return (b - a) / 1e6

    def sync(self):
        self.now = max(self.now, self.free)

    def capturing(self):
        return self.capture


def mk(dev, tmp_path, **kw):
    kinds = {"indexer": "container", "attn": "container", "select": "leaf", "attn_k": "leaf"}
    t = qo.OpTimer(dev, rank=0, out_dir=str(tmp_path), every=kw.pop("every", 4), warmup=kw.pop("warmup", 2),
                   clock=dev.clock, kinds=kinds, **kw)
    return t


def run_step(t, dev, T, body, layers=("L0", "L1", "L2")):
    for L in layers:
        timed = t.on_anchor(L, T)
        if timed:
            t.call(qo.ANCHOR, lambda: body(t, dev), (), {})
        else:
            body(t, dev)


def spans_of(tmp_path):
    return [json.loads(x) for x in open(tmp_path / "rank0.jsonl")]


def test_launch_bound_vs_kernel_bound(tmp_path):
    # span "select": 10 launches of 100us host, 50us device each -> the device waits for the host: dev ~= host
    dev = FakeDev()
    t = mk(dev, tmp_path)

    def body(t, d):
        t.call("select", lambda: [d.launch(50_000, host_ns=100_000) for _ in range(10)], (), {})

    for _ in range(2 + 4 + 1):  # +1: a window is written when the next one starts
        run_step(t, dev, 8, body)
    s = spans_of(tmp_path)[0]["by_T"]["8"]
    sel = s["spans"]["select"]
    assert s["steps"] == 4 and sel["n"] == 12
    assert sel["host_ms"] / sel["n"] == pytest.approx(1.0)           # 10 x 100us
    assert sel["dev_ms"] / sel["n"] == pytest.approx(1.0, abs=0.06)   # waited on the host
    # kernel bound: 10 us of host, 500 us of device per launch -> dev >> host
    dev = FakeDev()
    t = mk(dev, tmp_path / "k")

    def body2(t, d):
        t.call("select", lambda: [d.launch(500_000, host_ns=10_000) for _ in range(10)], (), {})
        d.sync()  # the next op needs the result: keeps the queue from running away between spans

    for _ in range(7):
        run_step(t, dev, 8, body2)
    sel = json.loads(open(tmp_path / "k" / "rank0.jsonl").readline())["by_T"]["8"]["spans"]["select"]
    assert sel["host_ms"] / sel["n"] == pytest.approx(0.1)
    assert sel["dev_ms"] / sel["n"] == pytest.approx(5.0, abs=0.1)


def test_nesting_exclusive(tmp_path):
    dev = FakeDev()
    t = mk(dev, tmp_path)

    def body(t, d):
        def indexer():
            d.host(200_000)  # python in the container
            t.call("select", lambda: d.launch(300_000, host_ns=50_000), (), {})
            d.host(100_000)

        t.call("indexer", indexer, (), {})

    for _ in range(7):
        run_step(t, dev, 8, body, layers=("L0",))
    sp = spans_of(tmp_path)[0]["by_T"]["8"]["spans"]
    n = sp["indexer"]["n"]
    assert sp["indexer"]["host_ms"] / n == pytest.approx(0.35)       # inclusive: 200+50+100
    assert sp["indexer"]["host_excl_ms"] / n == pytest.approx(0.30)  # without select's 50
    assert sp["select"]["host_excl_ms"] / n == pytest.approx(0.05)
    assert sp["indexer"]["dev_excl_ms"] == pytest.approx(sp["indexer"]["dev_ms"] - sp["select"]["dev_ms"])
    # run_qsa (the anchor) contains the indexer
    assert sp["run_qsa"]["host_excl_ms"] == pytest.approx(0.0, abs=1e-9)


def test_warmup_windows_T_buckets_and_period(tmp_path):
    dev = FakeDev()
    t = mk(dev, tmp_path, every=3, warmup=2)

    def body(t, d):
        d.host(10_000)

    for i in range(2 + 6 + 3):
        T = 8 if i < 6 else 32
        run_step(t, dev, T, body, layers=("L0", "L1"))
        dev.host(1_000_000)  # the rest of the step
    recs = spans_of(tmp_path)
    assert len(recs) >= 2
    first = recs[0]
    assert first["step"] == 5 and first["by_T"]["8"]["steps"] == 3
    # warmup steps are not counted: 2 warmup + windows of 3
    assert first["by_T"]["8"]["period_ms_mean"] == pytest.approx(1.02, abs=0.01)  # 2 layers x 10us + 1 ms
    assert any("32" in r["by_T"] for r in recs)
    # layers other than the anchor layer do not count as steps
    assert sum(r["by_T"][T]["steps"] for r in recs for T in r["by_T"]) <= 9


def test_capture_bypass_and_no_leak(tmp_path):
    dev = FakeDev()
    t = mk(dev, tmp_path)
    t.step = 10
    dev.capture = True
    assert t.call("select", lambda: 42, (), {}) == 42 and not t.pending
    dev.capture = False

    class Bad(FakeDev):
        def event(self):
            raise RuntimeError("no events")

    b = Bad()
    t2 = mk(b, tmp_path / "b")
    t2.step = 10
    for _ in range(7):  # the engine call always succeeds; the timer disables itself
        assert t2.call("select", lambda: 7, (), {}) == 7
    assert t2.dead
    # an exception from the op itself propagates and leaves the stack clean
    t3 = mk(FakeDev(), tmp_path / "c")
    t3.step = 10
    with pytest.raises(ValueError):
        t3.call("select", lambda: (_ for _ in ()).throw(ValueError("x")), (), {})
    assert t3._stack() == []


def test_sync_mode_leaf(tmp_path):
    dev = FakeDev()
    t = mk(dev, tmp_path, mode="sync")

    def body(t, d):
        d.launch(400_000)  # untracked work queued before the leaf: shows up as lead

        def sel():
            d.launch(300_000, host_ns=20_000)

        t.call("select", sel, (), {})

    for _ in range(7):
        run_step(t, dev, 8, body, layers=("L0",))
    sp = spans_of(tmp_path)[0]["by_T"]["8"]["spans"]["select"]
    n = sp["n"]
    assert sp["lead_ms"] / n == pytest.approx(0.4)
    assert sp["wall_ms"] / n == pytest.approx(0.32)   # launch 20us + drain; kernel 300us
    assert sp["host_ms"] / n == pytest.approx(0.02)
    assert "dev_ms" in sp and sp["dev_ms"] == 0.0     # events not used in sync mode


def test_wrap_run_qsa_and_install(tmp_path, monkeypatch):
    monkeypatch.setenv("B70_QSA_OPTIMER", "1")
    monkeypatch.setenv("B70_QSA_OPTIMER_DIR", str(tmp_path))
    monkeypatch.setenv("B70_QSA_OPTIMER_EVERY", "2")
    monkeypatch.setenv("B70_QSA_OPTIMER_WARMUP", "1")
    dev = FakeDev()
    monkeypatch.setattr(qo, "XpuBackend", lambda: dev)
    # a fake module with a function and a class method to wrap through B70_QSA_OPTIMER_SPANS
    m = types.ModuleType("fakeops")

    def sel(x):
        dev.launch(100_000, host_ns=10_000)
        return x + 1

    class K:
        def go(self, x):
            return sys.modules["fakeops"].sel(x) + 1

    m.sel, m.K = sel, K
    sys.modules["fakeops"] = m
    monkeypatch.setenv("B70_QSA_OPTIMER_SPANS", ",".join(f"{n}=-" for n in qo.DEFAULT_SPECS)
                       + ",f_sel=fakeops:sel,f_go=fakeops:K.go")
    monkeypatch.setattr(qo, "_installed", False)
    monkeypatch.setattr(qo, "_timer", None)
    import torch

    class Owner:
        layer_name = "model.layers.3.attn"

        @qo.wrap_run_qsa
        def _run_qsa(self, pqk, *rest):
            return sys.modules["fakeops"].K().go(1)

    o = Owner()
    pqk = torch.zeros(8, 4)
    for _ in range(7):
        assert o._run_qsa(pqk) == 3
    assert qo._timer is not None and qo._timer.anchor_layer == "model.layers.3.attn"
    recs = spans_of(tmp_path)
    assert recs and set(recs[0]["by_T"]["8"]["spans"]) == {"run_qsa", "f_go", "f_sel"}
    assert recs[0]["by_T"]["8"]["spans"]["f_sel"]["n"] >= 1
    # functions that were wrapped keep their name and are wrapped once
    assert sys.modules["fakeops"].sel.__name__ == "sel" and sys.modules["fakeops"].sel.__b70_optimer__
    qo._install_spans(print)
    assert not hasattr(sys.modules["fakeops"].sel.__wrapped__, "__b70_optimer__")


def test_off_is_identity(monkeypatch):
    monkeypatch.delenv("B70_QSA_OPTIMER", raising=False)

    def f(self, x):
        return x

    assert qo.wrap_run_qsa(f) is f


def test_report(tmp_path, capsys):
    dev = FakeDev()
    t = mk(dev, tmp_path, every=4, warmup=1)

    def body(t, d):
        t.call("select", lambda: d.launch(200_000, host_ns=20_000), (), {})

    for i in range(1 + 8 + 1):
        run_step(t, dev, 8 if i < 5 else 64, body, layers=("L0", "L1", "L2"))
        dev.host(500_000)
    assert rep.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "T=8 " in out and "T=64" in out and "select" in out and "run_qsa" in out
    assert rep.main([str(tmp_path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    r8 = next(r for r in rows if r["T"] == 8)
    assert r8["spans"]["select"]["calls_per_step"] == pytest.approx(3.0)
    assert r8["spans"]["select"]["host_ms"] == pytest.approx(0.06)   # 3 calls x 20us per step
    assert r8["spans"]["select"]["dev_ms"] >= 0.5                    # 3 x 200us


def test_overhead_recorded_and_excluded_from_parent(tmp_path):
    class SlowDev(FakeDev):
        def event(self):
            self.host(5_000)  # an event costs the host 5 us
            return super().event()

    dev = SlowDev()
    t = mk(dev, tmp_path)

    def body(t, d):
        def indexer():
            t.call("select", lambda: d.launch(100_000, host_ns=50_000), (), {})

        t.call("indexer", indexer, (), {})

    for _ in range(7):
        run_step(t, dev, 8, body, layers=("L0",))
    rec = spans_of(tmp_path)[0]
    sp = rec["by_T"]["8"]["spans"]
    n = sp["indexer"]["n"]
    assert rec["timer_overhead_ms"] > 0
    # indexer's own host time is zero: the child's 50 us and the child's event cost are both excluded
    assert sp["indexer"]["host_excl_ms"] / n == pytest.approx(0.0, abs=0.02)
    assert sp["select"]["host_excl_ms"] / n == pytest.approx(0.05)


def test_event_pool_is_recycled(tmp_path):
    class PoolBackend(FakeDev):
        released = 0

        def release(self, ev):
            PoolBackend.released += len(ev)

    dev = PoolBackend()
    t = mk(dev, tmp_path)

    def body(t, d):
        t.call("select", lambda: d.launch(10_000), (), {})

    for _ in range(7):
        run_step(t, dev, 8, body, layers=("L0",))
    assert PoolBackend.released > 0

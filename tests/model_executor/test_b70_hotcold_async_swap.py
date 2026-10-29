# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-Q27-0003: the hot/cold swapper, CPU only (no XPU device needed).

torch.xpu streams and events are stubbed (copies complete at once), so this checks the swapper's logic, not stream
concurrency: (1) with B70_MOE_HOTCOLD_ASYNC_SWAP unset the swapper is the synchronous one, step for step identical to
the module at B70_HC_REF (the trunk before this patch) on the same traffic: maps and weight bytes; (2) with it set,
after every runner step every global expert is routed to exactly one row (hot xor cold) whose bytes are that
expert's, the hot set follows a traffic shift, no step ever host-synchronizes, and the one-time log line fires.

Run: B70_HC_REF=/path/to/old/b70_hotcold.py python tests/model_executor/test_b70_hotcold_async_swap.py
(pytest also works; without B70_HC_REF the reference comparison is skipped)."""

import importlib.util
import logging
import os
import sys

import numpy as np
import torch

G, H, LAYERS, TICKS = 64, 24, 6, 160
KEYS = ("w13", "w2", "w13_scales", "w2_scales")
SHAPES = {"w13": (4, 8), "w2": (8, 2), "w13_scales": (2, 3), "w2_scales": (3, 2)}

_real_empty = torch.empty


def _empty(*a, pin_memory=False, **k):
    return _real_empty(*a, **k)


class _Stream:
    def wait_event(self, ev):
        pass

    def wait_stream(self, s):
        pass

    def synchronize(self):
        _SYNCS[0] += 1


class _Event:
    def record(self, stream=None):
        pass

    def query(self):
        return True


class _Ctx:
    def __init__(self, s):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_SYNCS = [0]
_FWD = _Stream()


def _patch_torch():
    torch.empty = _empty
    x = torch.xpu
    x.Stream = lambda *a, **k: _Stream()
    x.Event = lambda *a, **k: _Event()
    x.current_stream = lambda *a, **k: _FWD
    x.stream = _Ctx
    x.is_current_stream_capturing = lambda: False


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._ARENA.empty = lambda shape, dtype: _real_empty(shape, dtype=dtype)
    return mod


def _tag(g, key):
    dtype = torch.bfloat16 if "scales" in key else torch.uint8
    return torch.full(SHAPES[key], g, dtype=dtype)


class _Hot:
    pass


def _build(mod, seed):
    """LAYERS plans, all G experts local, H hot; expert g's bytes are all g."""
    rng = np.random.default_rng(seed)
    for layer in range(LAYERS):
        hot_local = sorted(rng.choice(G, H, replace=False).tolist())
        plan = mod.HotColdPlan(f"model.layers.{layer}.mlp.experts", list(range(G)), list(range(G)), G, hot_local)
        plan.spare = mod.spare_rows() if hasattr(mod, "spare_rows") else 0
        hot = _Hot()
        hot_t = {k: torch.stack([_tag(g, k) for g in plan.hot_local]) for k in KEYS}
        cold = {k: torch.stack([_tag(g, k) for g in plan.cold_local]) for k in KEYS}
        if plan.spare:
            cold = {k: mod._pin(t, plan.spare) for k, t in cold.items()}
        plan.cold = cold
        hot.w13, hot.w2 = hot_t["w13"].view(torch.int8), hot_t["w2"].view(torch.int8)
        hot.gemm1_wei_scales, hot.gemm2_wei_scales = hot_t["w13_scales"], hot_t["w2_scales"]
        hot_map = torch.full((G,), -1, dtype=torch.int32)
        cold_map = torch.full((G,), -1, dtype=torch.int32)
        for local_id, (is_cold, slot) in plan.slot.items():
            (cold_map if is_cold else hot_map)[local_id] = slot
        plan.hot_map, plan.cold_map = hot_map, cold_map
        plan.layer_idx = layer
        plan.route_counts = torch.zeros(2, G + 1, dtype=torch.int64)
        plan.count_graphs = True
        plan._impls = (hot, None)
        mod._PLANS[layer] = plan


def _traffic(rng, tick):
    """Per layer, decode routes: Zipf-like over one permutation, a different permutation after TICKS // 2."""
    perm = np.random.default_rng(1 if tick < TICKS // 2 else 2).permutation(G)
    p = 1.0 / (np.arange(G) + 1.0)
    p = p[np.argsort(perm)]
    p /= p.sum()
    return [rng.multinomial(400, p) for _ in range(LAYERS)]


def _feed(mod, counts):
    for layer, c in enumerate(counts):
        plan = mod._PLANS[layer]
        plan.route_counts[1, :G] += torch.from_numpy(c)
        plan.route_counts[0, :G] += torch.from_numpy(c // 4)


def _hot_bytes(mod):
    out = []
    for layer in range(LAYERS):
        hot, _ = mod._PLANS[layer]._impls
        out.append([hot.w13.clone(), hot.w2.clone(), hot.gemm1_wei_scales.clone(), hot.gemm2_wei_scales.clone()])
    return out


def _check_invariant(mod):
    for layer in range(LAYERS):
        plan = mod._PLANS[layer]
        hot, _ = plan._impls
        hot_t = {"w13": hot.w13.view(torch.uint8), "w2": hot.w2.view(torch.uint8),
                 "w13_scales": hot.gemm1_wei_scales, "w2_scales": hot.gemm2_wei_scales}
        hm, cm = plan.hot_map.numpy(), plan.cold_map.numpy()
        assert len(set(hm[hm >= 0].tolist())) == int((hm >= 0).sum()), "two experts in one hot slot"
        assert len(set(cm[cm >= 0].tolist())) == int((cm >= 0).sum()), "two experts in one cold row"
        for g in range(G):
            assert (hm[g] >= 0) != (cm[g] >= 0), f"layer {layer} expert {g}: hot {hm[g]} cold {cm[g]}"
            for k in KEYS:
                row = hot_t[k][hm[g]] if hm[g] >= 0 else plan.cold[k][cm[g]]
                assert torch.equal(row, _tag(g, k)), f"layer {layer} expert {g} {k}: wrong bytes"


def _overlap(mod):
    """Per layer, the share of the post-shift top-H experts that are hot."""
    want = set(np.random.default_rng(2).permutation(G)[:H].tolist())  # expert perm[r] has rank r
    return [len(want & set(np.nonzero(mod._PLANS[i].hot_map.numpy() >= 0)[0].tolist())) / H for i in range(LAYERS)]


_SYNC_OVERLAP: list = []


def _run(mod, seed=0, ticks=TICKS, check=None, after=None):
    mod._PLANS.clear()
    mod._SWAPPER = mod._Swapper()
    _build(mod, seed)
    rng = np.random.default_rng(seed + 100)
    mod.maybe_swap()  # setup
    steps = 0
    for t in range(ticks):
        _feed(mod, _traffic(rng, t))
        mod.maybe_swap()
        steps += 1
        if check:
            check(mod)
        # async: the runner steps on (no new period) until the cycle has flipped; in serving a cycle takes a few
        # steps (~0.1-0.3 s), far below the 5 s period, so each period's traffic meets one decision as in sync
        while getattr(mod._SWAPPER, "async_mode", False) and mod._SWAPPER.phase != "idle":
            mod.maybe_swap()
            steps += 1
            if check:
                check(mod)
        if after:
            after(mod, t)
    mod._steps = steps
    return mod


def _env(**kv):
    for k in ("B70_MOE_HOTCOLD_ASYNC_SWAP", "B70_MOE_HOTCOLD_ASYNC_SPARE"):
        os.environ.pop(k, None)
    os.environ.update({"B70_MOE_HOTCOLD": "1", "B70_MOE_HOTCOLD_SWAP": "1", "B70_MOE_HOTCOLD_SWAP_SOURCE": "decode",
                       "B70_MOE_ROUTE_COUNT": "1", "B70_MOE_HOTCOLD_SWAP_S": "0", "B70_MOE_HOTCOLD_SWAP_BUDGET": "4",
                       "B70_MOE_HOTCOLD_STATE": ""})
    os.environ.update(kv)


NEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "vllm", "model_executor", "layers", "fused_moe", "b70_hotcold.py")


def test_unset_is_the_synchronous_swap():
    _patch_torch()
    _env()
    ref_path = os.environ.get("B70_HC_REF", "")
    new = _load(NEW, "hc_new_sync")
    assert new.spare_rows() == 0 and not new.async_swap_enabled()
    trail_new = []
    _run(new, after=lambda m, t: trail_new.append((m._PLANS[0].hot_map.clone(), m._PLANS[5].cold_map.clone(),
                                                     _hot_bytes(m))), check=_check_invariant)
    assert not new._SWAPPER.async_mode and new._SWAPPER.total > 0
    assert all(p.spare == 0 and p.cold_rows == p.num_cold for p in new._PLANS.values())
    _SYNC_OVERLAP[:] = _overlap(new)
    if not ref_path:
        print("B70_HC_REF unset: reference comparison skipped")
        return
    ref = _load(ref_path, "hc_ref")
    trail_ref = []
    _run(ref, after=lambda m, t: trail_ref.append((m._PLANS[0].hot_map.clone(), m._PLANS[5].cold_map.clone(),
                                                     _hot_bytes(m))))
    for t, (a, b) in enumerate(zip(trail_new, trail_ref)):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), f"tick {t}: maps differ from the reference"
        for la, lb in zip(a[2], b[2]):
            assert all(torch.equal(x, y) for x, y in zip(la, lb)), f"tick {t}: hot bytes differ from the reference"
    for layer in range(LAYERS):
        for k in KEYS:
            assert torch.equal(new._PLANS[layer].cold[k], ref._PLANS[layer].cold[k])
    print(f"unset: identical to the reference over {TICKS} ticks, {new._SWAPPER.total} swaps, "
          f"{_SYNCS[0]} host synchronizes (both paths)")


def test_async_invariant_and_convergence():
    _patch_torch()
    _env(B70_MOE_HOTCOLD_ASYNC_SWAP="1")
    mod = _load(NEW, "hc_new_async")
    records = []

    class _H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    mod.logger.addHandler(_H())
    mod.logger.setLevel(logging.INFO)
    assert mod.spare_rows() == 4
    syncs0 = _SYNCS[0]
    _run(mod, check=_check_invariant)
    sw = mod._SWAPPER
    assert sw.async_mode and sw.total > 0, "async swapper never moved an expert"
    assert _SYNCS[0] == syncs0, "the async path synchronized the host"
    assert sum("async swap on" in m for m in records) == 1, "one-time log line"
    assert any("async:" in m for m in records)
    # convergence: after the shift the hot set follows the traffic as the synchronous swap's does
    got = _overlap(mod)
    assert min(got) >= min(_SYNC_OVERLAP or [0.9]) - 0.1, f"hot sets did not follow the shift: {got}"
    print(f"async: invariant held at every one of {mod._steps} runner steps ({TICKS} periods), {sw.total} swaps, "
          f"0 host synchronizes, "
          f"hot-set overlap with the new top-{H} after the shift {[round(x, 2) for x in got]} "
          f"(synchronous swap, same traffic: {[round(x, 2) for x in _SYNC_OVERLAP]})")


if __name__ == "__main__":
    test_unset_is_the_synchronous_swap()
    test_async_invariant_and_convergence()
    print("PASS")
    sys.exit(0)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70 0047: decode-aware swap scores and the learned hot-set state (CPU only)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import b70_hotcold

E, EP = 16, 2


class _Linear:
    def __init__(self, rank: int):
        self.rank = rank

    def map_global_to_local(self, g: int) -> int:
        per = E // EP
        return g - self.rank * per if self.rank * per <= g < (self.rank + 1) * per else -1


@pytest.fixture
def env(monkeypatch, tmp_path):
    rankings = tmp_path / "rank.json"
    rankings.write_text(json.dumps({"3": [9, 2, 15, 0, 7, 3, 10, 4, 5, 8, 11, 13, 14]}))
    monkeypatch.setenv("B70_MOE_HOTCOLD", "1")
    monkeypatch.setenv("B70_MOE_HOTCOLD_RANKINGS", str(rankings))
    monkeypatch.setenv("B70_MOE_HOT_EXPERTS", "3")
    monkeypatch.setenv("B70_MOE_HOTCOLD_STATE", str(tmp_path / "state"))
    monkeypatch.setattr("vllm.platforms.current_platform", SimpleNamespace(is_xpu=lambda: True))
    b70_hotcold._rankings.cache_clear()
    b70_hotcold._learned_state.cache_clear()
    yield tmp_path
    b70_hotcold._rankings.cache_clear()
    b70_hotcold._learned_state.cache_clear()


def _plan(rank: int):
    return b70_hotcold.maybe_plan("model.language_model.layers.3.mlp.experts",
                                  SimpleNamespace(num_experts=E), _Linear(rank), quant_config=object())


def _write_state(path, rank, periods, layers):
    (path / f"state-rank{rank}.json").write_text(json.dumps(
        {"version": 1, "source": "decode", "periods": periods, "layers": layers}))


def test_no_state_keeps_checkpoint_order(env):
    assert _plan(0).hot_local == [0, 2, 7]


def test_learned_state_places_hot_set(env):
    # rank 0: experts 5, 6 and 1 (never in the checkpoint ranking) are this host's hottest
    _write_state(env, 0, 20, {"3": {"5": 10.0, "6": 8.0, "1": 1.0}})
    _write_state(env, 1, 20, {"3": {"12": 3.0}})
    assert _plan(0).hot_local == [1, 5, 6]
    # rank 1: 12 first, then the checkpoint order (9, 15) as the tie-break
    assert _plan(1).hot_local == [1, 4, 7]


def test_state_with_too_few_periods_is_ignored(env):
    _write_state(env, 0, 3, {"3": {"5": 10.0, "6": 8.0, "1": 1.0}})
    assert _plan(0).hot_local == [0, 2, 7]


def _swapper(source, prefill_w=0.1):
    s = b70_hotcold._Swapper()
    s.source, s.prefill_w = source, prefill_w
    return s


def _stub(counts_prefill, counts_decode):
    n = len(counts_prefill)
    plan = SimpleNamespace(num_global=n, layer_idx=3)
    plan.route_counts = torch.tensor([counts_prefill + [0], counts_decode + [0]], dtype=torch.int64)
    plan.last_counts = np.zeros((2, n), np.int64)
    return plan


def test_combined_prefill_is_raw_prefill_delta():
    plan = _stub([4, 0, 6, 0], [0, 9, 0, 1])
    out = _swapper("prefill")._combined(plan, np)
    assert out.tolist() == [4, 0, 6, 0]
    # the next period sees only the new routes
    plan.route_counts[0, 1] += 2
    assert _swapper("prefill")._combined(plan, np).tolist() == [0, 2, 0, 0]


def test_combined_decode_normalises_per_layer():
    plan = _stub([4, 0, 6, 0], [0, 9, 0, 1])
    out = _swapper("decode", prefill_w=0.5)._combined(plan, np)
    np.testing.assert_allclose(out, [0.5 * 0.4, 0.9, 0.5 * 0.6, 0.1])


def test_save_then_load_roundtrip(env):
    s = _swapper("decode")
    s.state_path = str(env / "state")
    s.periods, s.prev_periods = 15, 0
    plan = SimpleNamespace(layer_idx=3, local_globals=np.arange(8), lt=np.zeros(E))
    plan.lt[[5, 6]] = [2.0, 1.0]
    s.plans = [plan]
    s._save_state(np)
    saved = json.loads((env / "state-rank0.json").read_text())
    assert saved["periods"] == 15 and saved["layers"] == {"3": {"5": 2.0, "6": 1.0}}
    b70_hotcold._learned_state.cache_clear()
    assert b70_hotcold._learned_state()["layers"][3] == {5: 2.0, 6: 1.0}

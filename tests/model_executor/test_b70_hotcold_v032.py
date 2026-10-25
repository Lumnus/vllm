# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-V32-HC1: the 0.32 port's additions to the hot/cold plan (CPU only).

Qwen3.8-Flash-Next shapes: 512 routed experts, linear EP placement, TP4 (128 per rank) and TP2 (256 per rank)."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import b70_hotcold

E = 512


class _Linear:
    def __init__(self, rank: int, ep: int):
        self.rank, self.per = rank, E // ep

    def map_global_to_local(self, g: int) -> int:
        lo = self.rank * self.per
        return g - lo if lo <= g < lo + self.per else -1


@pytest.fixture
def xpu(monkeypatch):
    for var in ("B70_MOE_HOT_EXPERTS", "B70_MOE_COLD_EXPERTS", "B70_MOE_HOTCOLD_RANKINGS",
                "B70_MOE_HOTCOLD_STATE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("B70_MOE_HOTCOLD", "1")
    monkeypatch.setattr("vllm.platforms.current_platform", SimpleNamespace(is_xpu=lambda: True))
    b70_hotcold._rankings.cache_clear()
    b70_hotcold._learned_state.cache_clear()
    yield monkeypatch
    b70_hotcold._rankings.cache_clear()
    b70_hotcold._learned_state.cache_clear()


def _plan(rank: int, ep: int, layer: int = 5):
    return b70_hotcold.maybe_plan(f"model.language_model.layers.{layer}.mlp.experts",
                                  SimpleNamespace(num_experts=E), _Linear(rank, ep), quant_config=object())


def test_off_unless_switched(xpu):
    xpu.delenv("B70_MOE_HOTCOLD")
    xpu.setenv("B70_MOE_COLD_EXPERTS", "16")
    assert _plan(0, 4) is None


def test_no_count_means_all_resident(xpu):
    # B70_MOE_HOTCOLD=1 alone keeps every expert in VRAM: the 4-card layout is unchanged
    assert _plan(0, 4) is None and _plan(1, 2) is None


@pytest.mark.parametrize("ep,cold", [(4, 8), (4, 16), (2, 138)])
def test_cold_count_is_per_rank(xpu, ep, cold):
    xpu.setenv("B70_MOE_COLD_EXPERTS", str(cold))
    for rank in range(ep):
        p = _plan(rank, ep)
        assert p.num_cold == cold and p.num_hot == E // ep - cold
        assert sorted(p.hot_local + p.cold_local) == list(range(E // ep))


def test_cold_wins_over_hot(xpu):
    xpu.setenv("B70_MOE_HOT_EXPERTS", "100")
    xpu.setenv("B70_MOE_COLD_EXPERTS", "16")
    assert _plan(0, 4).num_cold == 16


def test_cold_zero_is_no_plan_and_clamps(xpu):
    xpu.setenv("B70_MOE_COLD_EXPERTS", "0")
    assert _plan(0, 4) is None
    xpu.setenv("B70_MOE_COLD_EXPERTS", "1000")
    assert _plan(0, 4).num_hot == 1  # never zero hot


def test_hot_set_follows_rankings_at_ep4(xpu, tmp_path):
    # rank 2 owns 256..383; the ranking lists other ranks' experts too (they are ignored)
    order = [300, 5, 383, 256, 470, 257]
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"5": order}))
    xpu.setenv("B70_MOE_HOTCOLD_RANKINGS", str(path))
    xpu.setenv("B70_MOE_COLD_EXPERTS", str(128 - 3))
    p = _plan(2, 4)
    assert p.hot_local == sorted([300 - 256, 383 - 256, 0])


def test_remap_unrouted():
    ids = torch.tensor([[3, -1], [-1, 7]], dtype=torch.int32)
    r0 = b70_hotcold.remap_unrouted(ids, 512, 0)
    r1 = b70_hotcold.remap_unrouted(ids, 512, 3)
    assert r0.dtype == ids.dtype and r0.tolist() == [[3, 511], [511, 7]]
    assert r1.tolist() == [[3, 0], [0, 7]]
    # the sentinel is never owned by the remapping rank (linear placement), so it is -1 in both maps
    assert _Linear(0, 4).map_global_to_local(511) == -1 and _Linear(3, 4).map_global_to_local(0) == -1


def test_learned_state_is_per_engine(xpu, tmp_path):
    # two engines on one host: each def sets its own B70_MOE_HOTCOLD_STATE prefix; files never mix
    for eng, score in (("tp2a", 9.0), ("tp2b", 1.0)):
        for rank in range(2):
            (tmp_path / f"{eng}-rank{rank}.json").write_text(
                json.dumps({"periods": 20, "layers": {"5": {str(rank * 256 + 1): score}}}))
    xpu.setenv("B70_MOE_HOTCOLD_STATE", str(tmp_path / "tp2a"))
    a = b70_hotcold._learned_state()
    b70_hotcold._learned_state.cache_clear()
    xpu.setenv("B70_MOE_HOTCOLD_STATE", str(tmp_path / "tp2b"))
    b = b70_hotcold._learned_state()
    assert a["layers"][5] == {1: 9.0, 257: 9.0}
    assert b["layers"][5] == {1: 1.0, 257: 1.0}

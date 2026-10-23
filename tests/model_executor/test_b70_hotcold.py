# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70 0040: hot/cold expert plan (CPU only; the kernel side is in
tests/kernels/moe/test_b70_hotcold_xpu.py)."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import b70_hotcold

E, EP = 16, 2


class _Linear:
    """ExpertMapManager stand-in: linear placement, E // EP experts per rank."""

    def __init__(self, rank: int):
        self.rank = rank

    def map_global_to_local(self, g: int) -> int:
        per = E // EP
        return g - self.rank * per if self.rank * per <= g < (self.rank + 1) * per else -1


@pytest.fixture
def env(monkeypatch, tmp_path):
    rankings = tmp_path / "rank.json"
    # layer 3: hottest first; experts 1, 6 and 12 never observed
    rankings.write_text(json.dumps({"3": [9, 2, 15, 0, 7, 3, 10, 4, 5, 8, 11, 13, 14]}))
    monkeypatch.setenv("B70_MOE_HOTCOLD", "1")
    monkeypatch.setenv("B70_MOE_HOTCOLD_RANKINGS", str(rankings))
    monkeypatch.setenv("B70_MOE_HOT_EXPERTS", "3")
    monkeypatch.setattr("vllm.platforms.current_platform",
                        SimpleNamespace(is_xpu=lambda: True))
    b70_hotcold._rankings.cache_clear()
    yield
    b70_hotcold._rankings.cache_clear()


def _plan(rank: int, name: str = "model.language_model.layers.3.mlp.experts"):
    return b70_hotcold.maybe_plan(name, SimpleNamespace(num_experts=E), _Linear(rank),
                                  quant_config=object())


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("B70_MOE_HOTCOLD", raising=False)
    assert _plan(0) is None


def test_hot_set_follows_rankings_per_rank(env):
    p0, p1 = _plan(0), _plan(1)
    # rank 0 owns 0..7: its ranked order is 2, 0, 7, 3, 4, 5 -> hot 2, 0, 7 (local ids)
    assert p0.hot_local == [0, 2, 7]
    assert p0.cold_local == [1, 3, 4, 5, 6]
    # rank 1 owns 8..15: ranked 9, 15, 10, 8, 11, 13, 14 -> hot 9, 15, 10 = local 1, 7, 2
    assert p1.hot_local == [1, 2, 7]


def test_unranked_experts_fill_by_id(env, monkeypatch):
    monkeypatch.setenv("B70_MOE_HOT_EXPERTS", "7")
    p0 = _plan(0)  # ranked 2, 0, 7, 3, 4, 5 then unranked 1, 6
    assert p0.hot_local == [0, 1, 2, 3, 4, 5, 7]
    assert p0.cold_local == [6]


def test_slots_cover_local_experts_once(env):
    p = _plan(0)
    hot = sorted(s for c, s in p.slot.values() if not c)
    cold = sorted(s for c, s in p.slot.values() if c)
    assert hot == list(range(p.num_hot)) and cold == list(range(p.num_cold))
    assert sorted(p.slot) == list(range(E // EP))


def test_no_plan_when_everything_fits_or_mtp(env, monkeypatch):
    assert _plan(0, "mtp.layers.0.mlp.experts") is None
    monkeypatch.setenv("B70_MOE_HOT_EXPERTS", str(E // EP))
    assert _plan(0) is None


def test_loader_routes_cold_experts_to_twins(env):
    p = _plan(0)
    layer = torch.nn.Module()
    w = torch.nn.Parameter(torch.zeros(p.num_hot, 4, 6, dtype=torch.int32), requires_grad=False)
    w.is_transposed = True
    layer.register_parameter("w13_qweight", w)
    p.attach_twins(layer)
    twin = p.twins["w13_qweight"]
    assert twin.shape == (p.num_cold, 4, 6) and twin.device.type == "cpu"
    assert twin.is_transposed is True
    param, slot = p.route(w, 2)  # local 2 is hot, slot 1
    assert param is w and slot == 1
    param, slot = p.route(w, 4)  # local 4 is cold, slot 2
    assert param is twin and slot == 2

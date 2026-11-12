# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`EPLBConfig.trace_path`: the routed expert load over time, and the script
that reads it."""

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_trace import (
    TRACE_MAGIC,
    ExpertLoadTracer,
    encode_counts,
)

from .eplb_single_rank import (
    FakeMoEModel,
    add_model,
    physical_rows_hold,
    single_rank_eplb,
)

_SCRIPT = pathlib.Path(__file__).parents[2] / "tools" / "analyze_expert_trace.py"
_spec = importlib.util.spec_from_file_location("analyze_expert_trace", _SCRIPT)
assert _spec is not None and _spec.loader is not None
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


def _tracer_state(num_layers: int, num_experts: int):
    """A model state as the tracer reads it: one rank, identity placement."""
    p2l = torch.arange(num_experts, dtype=torch.int32).repeat(num_layers, 1)
    return SimpleNamespace(
        expert_load_pass=torch.zeros(num_layers, num_experts, dtype=torch.int32),
        physical_to_logical_map=p2l,
        model=SimpleNamespace(
            num_logical_experts=num_experts,
            moe_layers=[
                SimpleNamespace(layer_name=f"layers.{i}.experts")
                for i in range(num_layers)
            ],
        ),
    )


def _write_trace(path, counts: np.ndarray, interval: int = 1, rank: int = 0) -> None:
    """Feed `counts` [T, layers, experts] (one interval per entry, as the sum
    of `interval` equal steps' worth, spread on the first step) to a tracer."""
    state = _tracer_state(*counts.shape[1:])
    tracer = ExpertLoadTracer(str(path), interval, rank=rank)
    for chunk in counts:
        for k in range(interval):
            if k == 0:
                state.expert_load_pass.copy_(torch.from_numpy(chunk.astype(np.int32)))
            tracer.accumulate([state])
            state.expert_load_pass.zero_()
            tracer.end_step([state])
    tracer.drain()


def _zipf(n: int, s: float) -> np.ndarray:
    p = 1.0 / np.arange(1, n + 1) ** s
    return p / p.sum()


def test_off_by_default_and_out_of_the_compile_hash():
    assert EPLBConfig().trace_path is None
    assert (
        EPLBConfig(trace_path="/tmp/x", trace_interval=5).compile_factors()
        == EPLBConfig().compile_factors()
    )
    with pytest.raises(ValueError):
        EPLBConfig(trace_interval=0)


def test_the_encoding_picks_the_smallest_integer_and_survives_a_round_trip(tmp_path):
    for largest, size in [(255, 1), (256, 2), (65535, 2), (65536, 4)]:
        counts = np.array([[0, largest, 3]])
        assert encode_counts(counts)[0] == size
    big = np.random.default_rng(0).integers(0, 70000, size=(3, 40))
    path = tmp_path / "t"
    _write_trace(path, big[None])
    trace = analyze.read_trace([str(path)])
    assert np.array_equal(trace.counts[0], big)
    assert (tmp_path / "t.rank0").read_bytes().startswith(TRACE_MAGIC)


def test_a_cut_last_record_is_ignored(tmp_path):
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 50, size=(4, 2, 16))
    _write_trace(tmp_path / "t", counts)
    file = tmp_path / "t.rank0"
    file.write_bytes(file.read_bytes()[:-5])
    trace = analyze.read_trace([str(file)])
    assert trace.num_intervals == 3
    assert np.array_equal(trace.counts, counts[:3])


def test_ranks_are_added_on_their_common_steps(tmp_path):
    a = np.full((3, 1, 4), 2)
    b = np.full((3, 1, 4), 5)
    _write_trace(tmp_path / "m", a, rank=0)
    _write_trace(tmp_path / "m", b, rank=1)
    trace = analyze.read_trace([str(tmp_path / "m")])
    assert (trace.counts == 7).all()


def test_interval_sums_steps_and_maps_physical_to_logical(tmp_path):
    state = _tracer_state(1, 3)
    # four physical slots: logical 0, 1, 2 and a replica of 1; slot 3 of the
    # buffer is empty (-1)
    state.expert_load_pass = torch.zeros(1, 5, dtype=torch.int32)
    state.physical_to_logical_map = torch.tensor([[0, 1, 2, 1, -1]], dtype=torch.int32)
    tracer = ExpertLoadTracer(str(tmp_path / "t"), 2, rank=0)
    for load in ([1, 2, 3, 4, 9], [10, 20, 30, 40, 9], [5, 0, 0, 0, 0]):
        state.expert_load_pass.copy_(torch.tensor([load], dtype=torch.int32))
        tracer.accumulate([state])
        state.expert_load_pass.zero_()
        tracer.end_step([state])
    tracer.drain()
    trace = analyze.read_trace([str(tmp_path / "t")])
    # one full interval of two steps; the third step is still open
    assert trace.counts.shape == (1, 1, 3)
    assert trace.counts[0, 0].tolist() == [11, 22 + 44, 33]
    assert trace.steps.tolist() == [2]
    assert trace.interval == 2


def test_with_the_trace_on_eplb_records_every_step_and_never_moves(
    monkeypatch, tmp_path
):
    path = str(tmp_path / "trace")
    state = single_rank_eplb(
        monkeypatch,
        trace_path=path,
        trace_interval=3,
        step_interval=2_000_000_000,
        window_size=1,
    )
    model = FakeMoEModel(num_layers=2, num_logical=4)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    before = model_state.physical_to_logical_map.clone()
    rng = np.random.default_rng(2)
    expected = []
    for _ in range(4):
        total = np.zeros((2, 4), dtype=np.int64)
        for _ in range(3):
            load = rng.integers(0, 9, size=(2, 4))
            model_state.expert_load_pass.copy_(torch.from_numpy(load.astype(np.int32)))
            state.step()
            total += load
        expected.append(total)
    assert state._trace is not None
    state._trace.drain()
    trace = analyze.read_trace([path])
    assert trace.layers == ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"]
    assert trace.steps.tolist() == [3, 6, 9, 12]
    assert np.array_equal(trace.counts, np.stack(expected))
    assert torch.equal(model_state.physical_to_logical_map, before)
    assert physical_rows_hold(model, state)


def test_with_the_trace_off_nothing_changes(monkeypatch, tmp_path):
    state = single_rank_eplb(monkeypatch, step_interval=100, window_size=5)
    add_model(state, FakeMoEModel(num_layers=1, num_logical=4))
    assert state._trace is None
    # recording is only on in the last window_size steps before a rearrangement
    assert not state._should_record_current_step()
    state.expert_rearrangement_step = 96
    assert state._should_record_current_step()
    assert list(tmp_path.iterdir()) == []


def test_dummy_steps_are_not_traced(monkeypatch, tmp_path):
    path = str(tmp_path / "trace")
    state = single_rank_eplb(
        monkeypatch, trace_path=path, trace_interval=1, step_interval=2_000_000_000
    )
    add_model(state, FakeMoEModel(num_layers=1, num_logical=4))
    model_state = next(iter(state.model_states.values()))
    model_state.expert_load_pass.fill_(7)
    state.step(is_dummy=True)
    model_state.expert_load_pass.copy_(torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))
    state.step()
    assert state._trace is not None
    state._trace.drain()
    trace = analyze.read_trace([path])
    assert trace.counts[:, 0].tolist() == [[1, 2, 3, 4]]


# ---- the analyzer on synthetic loads

N_EXP = 128
CALLS = 4000  # calls per interval per layer


def _draw(rng, probs: np.ndarray) -> np.ndarray:
    return rng.multinomial(CALLS, probs)


def test_a_stationary_zipf_covers_as_the_tail_says(tmp_path):
    rng = np.random.default_rng(3)
    p = _zipf(N_EXP, 1.0)
    counts = np.stack([np.stack([_draw(rng, p)] * 1) for _ in range(300)])
    _write_trace(tmp_path / "z", counts)
    trace = analyze.read_trace([str(tmp_path / "z")])
    cum, rows = analyze.cumulative_table(trace, [16, 64], 1.0)
    for n, mean, _worst, _ in rows:
        assert mean == pytest.approx(1 - p[:n].sum(), abs=0.01)
    # per interval: a fixed set does as well as the oracle, up to noise
    static, oracle = analyze.per_interval_table(trace, cum, [16])[0][1:]
    assert static == pytest.approx(1 - p[:16].sum(), abs=0.01)
    assert static - oracle < 0.02
    # a fast set of the 16 hottest plus a pool of 48: the pool covers the next
    # ranks, the tail beyond it is the miss
    r = analyze.replay(trace, 16, 48, "lru", 20, 1.0)
    assert 1 - r["hit"] == pytest.approx(1 - p[:16].sum() - p[16:64].sum(), abs=0.05)
    # no fast set, a pool as large as the experts: after warm-up nothing misses
    full = analyze.replay(trace, 0, N_EXP, "lru", 20, 1.0)
    assert full["hit"] > 0.999
    ws = analyze.working_set(trace, 10, 16)
    assert ws is not None and ws["jac_hot"].mean() > 0.75


def test_a_rotating_hot_set_shows_in_the_working_set_and_the_replay(tmp_path):
    rng = np.random.default_rng(4)
    hot = 16
    period = 25
    intervals = 400
    counts = []
    for t in range(intervals):
        start = (t // period) * hot % N_EXP
        p = np.full(N_EXP, 0.02 / (N_EXP - hot))
        p[np.arange(start, start + hot) % N_EXP] = 0.98 / hot
        counts.append(_draw(rng, p)[None])
    _write_trace(tmp_path / "d", np.stack(counts))
    trace = analyze.read_trace([str(tmp_path / "d")])
    cum, rows = analyze.cumulative_table(trace, [16], 1.0)
    # over the whole run the hot sets average out: the 16 hottest of the run
    # miss most calls
    assert rows[0][1] > 0.7
    # within an interval, 16 experts carry nearly everything
    static, oracle = analyze.per_interval_table(trace, cum, [16])[0][1:]
    assert static > 0.7
    assert oracle < 0.06
    # a fixed set of 16 chosen early misses most; a pool of 16 follows the rotation
    fixed = analyze.replay(trace, 16, 0, "lru", 20, 1.0)
    pool_lru = analyze.replay(trace, 0, 16, "lru", 20, 1.0)
    # a decay of one interval's half-life follows the rotation; a slow one
    # keeps the old hot set while its score outweighs the new
    pool_decay = analyze.replay(trace, 0, 16, "decay", 20, 1.0)
    pool_slow = analyze.replay(trace, 0, 16, "decay", 20, 20.0)
    assert fixed["hit"] < 0.3
    assert pool_lru["hit"] > 0.85
    assert pool_decay["hit"] > 0.8
    assert pool_slow["hit"] < pool_lru["hit"] - 0.1
    # the sets of the windows of 5 intervals mostly repeat; across a
    # rotation they change
    ws = analyze.working_set(trace, 5, 16)
    assert ws is not None
    assert 0.5 < ws["jac_hot"].mean() < 0.98
    assert ws["distinct"].mean() > hot
    assert pool_lru["per_step"] > 0


def test_the_state_file_gives_the_cumulative_table(tmp_path, capsys):
    rng = np.random.default_rng(5)
    loads = rng.multinomial(100000, _zipf(N_EXP, 1.0), size=3)
    import json

    path = tmp_path / "state.json"
    path.write_text(json.dumps({f"l{i}": loads[i].tolist() for i in range(3)}))
    assert analyze.main([str(path), "--top-n", "16", "64"]) == 0
    out = capsys.readouterr().out
    assert "Cumulative" in out and "Replay" not in out
    trace = analyze.read_state_file(str(path))
    assert trace is not None and trace.counts.shape == (1, 3, N_EXP)


def test_the_script_runs_end_to_end_on_a_trace(tmp_path, capsys):
    rng = np.random.default_rng(6)
    counts = np.stack([rng.multinomial(500, _zipf(64, 0.8), size=2) for _ in range(80)])
    _write_trace(tmp_path / "e", counts, interval=2)
    rc = analyze.main(
        [
            str(tmp_path / "e"),
            "--top-n",
            "8",
            "32",
            "--window",
            "5",
            "20",
            "--hot",
            "8",
            "--fixed",
            "0",
            "8",
            "--warm",
            "10",
            "--pool",
            "8",
            "16",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    for head in ("Cumulative", "Per interval", "Working set", "Replay"):
        assert head in out
    assert "80 x 2 steps = 160 steps" in out

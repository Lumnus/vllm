# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`EPLBConfig.trace_path`: the routed expert load over time, and the script
that reads it."""

import importlib.util
import pathlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

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
            tracer.accumulate([state], True)
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
        tracer.accumulate([state], True)
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
            model_state.expert_load_pass += torch.from_numpy(load.astype(np.int32))
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
    model_state.expert_load_pass += torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
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


# ---- the cost of the trace on the stepping thread, and what it attributes


class _Counter(TorchDispatchMode):
    """Counts the device operations a block issues (views and metadata are
    not submissions) and notes any read of a value on the host."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []
        self.host_reads: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        returns = func._schema.returns
        is_view = any(
            r.alias_info is not None and not r.alias_info.is_write for r in returns
        )
        if func._schema.name in ("aten::_local_scalar_dense", "aten::item"):
            self.host_reads.append(func._schema.name)
        if not is_view:
            self.ops.append(func._schema.name)
        return func(*args, **(kwargs or {}))


def _ops_per_step(monkeypatch, tmp_path, layers: int, steps: int, **eplb_kwargs):
    """The operations each `EplbState.step` issues, with the router kernel's
    adds (outside the count) feeding the pass whenever the flag asks for it."""
    state = single_rank_eplb(monkeypatch, **eplb_kwargs)
    add_model(state, FakeMoEModel(num_layers=layers, num_logical=16))
    model_state = next(iter(state.model_states.values()))
    state.should_record_tensor = torch.ones((), dtype=torch.bool)
    state.expert_rearrangement_step = 0  # EPLB starts at 3/4 of the interval
    per_step = []
    for _ in range(steps):
        if bool(state.should_record_tensor):
            model_state.expert_load_pass += 1
        with _Counter() as counter:
            state.step()
        assert counter.host_reads == []
        per_step.append(len(counter.ops))
    return state, per_step


@pytest.mark.parametrize("layers", [2, 48])
def test_the_trace_adds_no_operation_to_a_step_outside_the_window(
    monkeypatch, tmp_path, layers
):
    common = dict(step_interval=2_000_000_000, window_size=1, trace_interval=4)
    _, off = _ops_per_step(monkeypatch, tmp_path, layers, 12, **common)
    state, on = _ops_per_step(
        monkeypatch, tmp_path, layers, 12, trace_path=str(tmp_path / "t"), **common
    )
    assert state._trace is not None
    assert off[1:] == [0] * 11  # today: nothing between window steps
    # the first step sets the flag; the interval ends (steps 4, 8, 12) cost a
    # fixed handful of operations per model, whatever the layer count
    ends = [i for i in range(12) if (i + 1) % 4 == 0]
    assert [n for i, n in enumerate(on) if i not in ends and i > 0] == [0] * 8
    assert all(0 < on[i] <= 14 for i in ends)


def test_the_cost_per_step_does_not_grow_with_the_layers(monkeypatch, tmp_path):
    common = dict(step_interval=100, window_size=100, trace_interval=4)
    runs = {}
    for layers in (2, 48):
        _, off = _ops_per_step(monkeypatch, tmp_path, layers, 9, **common)
        _, on = _ops_per_step(
            monkeypatch,
            tmp_path,
            layers,
            9,
            trace_path=str(tmp_path / f"t{layers}"),
            **common,
        )
        runs[layers] = (off, on)
    # the window records every step: its own copy and zero, plus one add
    for off, on in runs.values():
        added = [a - b for a, b in zip(on[1:], off[1:])]
        assert set(added) <= {0, 1} | set(range(2, 15))
        assert [d for i, d in enumerate(added, 1) if (i + 1) % 4 != 0] == [1] * 6
    assert runs[2][1] == runs[48][1]  # operation for operation the same


def test_with_the_trace_off_the_step_is_todays(monkeypatch, tmp_path):
    state, per_step = _ops_per_step(
        monkeypatch, tmp_path, 3, 6, step_interval=10, window_size=3
    )
    assert state._trace is None
    assert all(m.map_change_hook is None for m in state.model_states.values())
    # not near a rearrangement: no operation (the first step sets the flag
    # off); in the window: the pass is copied into it and zeroed, and the flag
    # goes on for the next step
    assert per_step == [1, 0, 0, 0, 0, 0]
    state.expert_rearrangement_step = 7
    model_state = next(iter(state.model_states.values()))
    model_state.expert_load_pass += 1
    with _Counter() as counter:
        state.step()
    assert counter.ops == ["aten::copy_", "aten::zero_", "aten::fill_"]
    assert model_state.expert_load_pass.sum() == 0


class _ThreadEvent:
    """A device event that notes the thread it is waited on from."""

    waits: list[int] = []
    gate: threading.Event | None = None

    def record(self) -> None:
        pass

    def synchronize(self) -> None:
        type(self).waits.append(threading.get_ident())
        if type(self).gate is not None:
            type(self).gate.wait(10)


def test_only_the_writer_thread_waits_and_a_slow_writer_drops_not_blocks(
    monkeypatch, tmp_path
):
    _ThreadEvent.waits = []
    _ThreadEvent.gate = threading.Event()
    state = single_rank_eplb(
        monkeypatch,
        trace_path=str(tmp_path / "t"),
        trace_interval=1,
        step_interval=2_000_000_000,
        window_size=1,
    )
    add_model(state, FakeMoEModel(num_layers=1, num_logical=4))
    tracer = state._trace
    assert tracer is not None
    tracer._event_factory = _ThreadEvent
    model_state = next(iter(state.model_states.values()))
    for _ in range(20):  # the writer waits on its first event: nothing blocks
        model_state.expert_load_pass += 1
        state.step()
    assert tracer.dropped > 0
    _ThreadEvent.gate.set()
    tracer.drain()
    assert _ThreadEvent.waits
    assert threading.get_ident() not in _ThreadEvent.waits
    trace = analyze.read_trace([str(tmp_path / "t")])
    assert trace.num_intervals == 20 - tracer.dropped


def test_a_trace_that_cannot_be_written_is_dropped_and_the_pass_returns(
    monkeypatch, tmp_path
):
    state = single_rank_eplb(
        monkeypatch,
        trace_path=str(tmp_path / "missing" / "t"),
        trace_interval=1,
        step_interval=2_000_000_000,
        window_size=1,
    )
    add_model(state, FakeMoEModel(num_layers=1, num_logical=4))
    model_state = next(iter(state.model_states.values()))
    model_state.expert_load_pass += 1
    state.step()
    assert state._trace is not None
    state._trace.drain()
    model_state.expert_load_pass += 1
    state.step()  # sees the failure
    assert state._trace is None
    assert model_state.map_change_hook is None
    assert model_state.expert_load_pass.sum() == 0


def _true_logical_loads(state, model_state, hot: list[int], rng) -> np.ndarray:
    """Add one step's routing for logical experts `hot` (skewed) to the pass,
    on whatever slots the current placement gives them; return the logical
    counts."""
    layers, physical = model_state.expert_load_pass.shape
    num_logical = model_state.model.num_logical_experts
    truth = np.zeros((layers, num_logical), dtype=np.int64)
    p2l = model_state.physical_to_logical_map
    for layer in range(layers):
        for expert, calls in zip(hot, (40, 20, 10, 5)):
            slots = (p2l[layer] == expert).nonzero().flatten().tolist()
            for k in range(calls):
                model_state.expert_load_pass[layer, slots[k % len(slots)]] += 1
            truth[layer, expert] += calls
    return truth


@pytest.mark.parametrize("hook", [True, False])
def test_the_counts_are_logical_across_a_rearrangement(monkeypatch, tmp_path, hook):
    path = str(tmp_path / "trace")
    state = single_rank_eplb(
        monkeypatch,
        trace_path=path,
        trace_interval=4,
        step_interval=6,
        window_size=6,
        policy="default",
    )
    model = FakeMoEModel(num_layers=2, num_logical=4, num_redundant=2)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    state.expert_rearrangement_step = 0
    if not hook:
        model_state.map_change_hook = None  # what attributing at the end does
    rng = np.random.default_rng(0)
    before = model_state.physical_to_logical_map.clone()
    truth = []
    for _ in range(12):
        truth.append(_true_logical_loads(state, model_state, [0, 1, 2, 3], rng))
        state.step()
    assert not torch.equal(model_state.physical_to_logical_map, before)
    assert physical_rows_hold(model, state)
    tracer = state._trace
    assert tracer is not None
    tracer.drain()
    trace = analyze.read_trace([path])
    expected = np.stack([sum(truth[i : i + 4]) for i in (0, 4, 8)])
    assert trace.steps.tolist() == [4, 8, 12]
    if hook:
        assert np.array_equal(trace.counts, expected)
    else:
        # the interval that spans the move is attributed with the new placement
        assert not np.array_equal(trace.counts, expected)
        assert trace.counts.sum() == expected.sum()


def test_sampling_counts_only_the_first_steps_of_each_interval(monkeypatch, tmp_path):
    path = str(tmp_path / "trace")
    state = single_rank_eplb(
        monkeypatch,
        trace_path=path,
        trace_interval=5,
        trace_sample=2,
        step_interval=2_000_000_000,
        window_size=1,
    )
    add_model(state, FakeMoEModel(num_layers=1, num_logical=4))
    model_state = next(iter(state.model_states.values()))
    state.should_record_tensor = torch.ones((), dtype=torch.bool)
    flags = []
    for step in range(15):
        flags.append(bool(state.should_record_tensor))
        if flags[-1]:
            model_state.expert_load_pass += step + 1
        state.step()
    assert flags == [True, True, False, False, False] * 3
    assert state._trace is not None
    state._trace.drain()
    trace = analyze.read_trace([path])
    assert trace.sample == 2 and trace.interval == 5 and trace.steps_per_record == 2
    assert trace.counts[:, 0, 0].tolist() == [1 + 2, 6 + 7, 11 + 12]


def test_a_dirty_pass_is_not_copied_into_the_window(monkeypatch, tmp_path):
    state = single_rank_eplb(
        monkeypatch,
        trace_path=str(tmp_path / "trace"),
        trace_interval=100,
        step_interval=10,
        window_size=2,
    )
    add_model(state, FakeMoEModel(num_layers=1, num_logical=4))
    model_state = next(iter(state.model_states.values()))
    state.expert_rearrangement_step = 0
    for _ in range(7):  # outside the window: the pass keeps adding
        model_state.expert_load_pass += 1
        state.step()
    assert model_state.expert_load_pass.sum() == 7 * 4
    model_state.expert_load_pass += 1
    state.step()  # the next step is the window's first: the pass is taken
    assert model_state.expert_load_pass.sum() == 0
    model_state.expert_load_pass += 1
    state.step()
    assert model_state.expert_load_window[0].sum() == 4  # this step alone
    assert state._trace is not None
    state._trace.drain()


def test_files_rotate_by_size_keep_the_newest_and_read_as_one(tmp_path):
    state = _tracer_state(2, 8)
    tracer = ExpertLoadTracer(str(tmp_path / "r"), 1, rank=0, rotate_bytes=300, keep=3)
    rng = np.random.default_rng(1)
    loads = []
    for _ in range(40):
        load = rng.integers(0, 200, size=(2, 8))
        loads.append(load)
        state.expert_load_pass.copy_(torch.from_numpy(load.astype(np.int32)))
        tracer.accumulate([state], True)
        state.expert_load_pass.zero_()
        tracer.end_step([state])
        tracer.drain()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "r.rank0" in names and len(names) == 4  # live file + 3 kept
    for path in tmp_path.iterdir():
        assert path.read_bytes().startswith(TRACE_MAGIC)
        assert path.stat().st_size <= 300 + 200  # a record may overshoot once
    trace = analyze.read_trace([str(tmp_path / "r")])
    assert trace.steps.tolist() == sorted(trace.steps.tolist())
    assert np.array_equal(trace.counts, np.stack(loads)[-trace.num_intervals :])
    assert trace.steps[-1] == 40


def test_a_restart_starts_a_new_file_and_the_steps_continue(tmp_path):
    state = _tracer_state(1, 4)
    for _ in range(2):  # two runs write the same prefix
        tracer = ExpertLoadTracer(str(tmp_path / "r"), 1, rank=0)
        for _ in range(3):
            state.expert_load_pass += 1
            tracer.accumulate([state], True)
            state.expert_load_pass.zero_()
            tracer.end_step([state])
        tracer.drain()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.rank0", "r.rank0.1"]
    trace = analyze.read_trace([str(tmp_path / "r")])
    assert trace.steps.tolist() == [1, 2, 3, 4, 5, 6]
    assert (trace.counts == 1).all()

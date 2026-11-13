# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where a KV cache resize runs among the engine core's executor calls."""

import queue
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCore, EngineCoreProc

UTILITY = EngineCoreRequestType.UTILITY


class PollableFuture(Future):
    """Done on the `ready_after`-th poll; on_poll is called first."""

    def __init__(self, ready_after: int, on_poll=None):
        super().__init__()
        self.ready_after = ready_after
        self.on_poll = on_poll
        self.polls = 0

    def poll(self) -> bool:
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self.polls)
        if self.polls >= self.ready_after and not self.done():
            self.set_result("output")
        return self.done()

    def wait_briefly(self, pending, deadline) -> None:
        pass

    def result(self, timeout=None):
        if not self.done():
            self.set_result("output")
        return super().result(timeout)


class RecordingExecutor:
    """Records the engine core's calls in the order the workers would see them
    (one broadcast queue, each worker runs its calls one at a time)."""

    def __init__(self, resize_to: int | None = None) -> None:
        self.calls: list[str] = []
        self.resize_to = resize_to
        self.is_sleeping = False
        self.futures: dict[str, PollableFuture] = {}

    def execute_model(self, scheduler_output, non_block=False):
        self.calls.append(f"execute_model:{scheduler_output.name}")
        future: Future = Future()
        future.set_result(None)
        return future

    def sample_tokens(self, grammar_output, non_block=False):
        # The future of the batch whose sampling was just queued.
        name = self.sampled_name
        self.calls.append(f"sample_tokens:{name}")
        return self.futures[name]

    def resize_kv_cache(self, target_num_blocks: int) -> int:
        self.calls.append(f"resize_kv_cache:{target_num_blocks}")
        return self.resize_to if self.resize_to is not None else target_num_blocks


class FakeScheduler:
    """Hands out batches by name; a batch may defer its sampling, as batches
    with structured output do until the previous output is processed."""

    def __init__(self, executor: RecordingExecutor, batches: list[SimpleNamespace]):
        self.executor = executor
        self.batches = deque(batches)
        self.kv_cache_config = SimpleNamespace(num_blocks=10)
        self.grown_to: list[int] = []

    def has_requests(self) -> bool:
        return bool(self.batches)

    def schedule(self, throttle_prefills=False):
        batch = self.batches.popleft()
        self.executor.sampled_name = batch.name
        return batch

    def get_grammar_bitmask(self, scheduler_output):
        # Sampling is queued right after this call.
        self.executor.sampled_name = scheduler_output.name
        return None

    def update_from_output(self, scheduler_output, model_output):
        return {}

    def grow_kv_cache(self, num_blocks: int) -> None:
        self.grown_to.append(num_blocks)
        self.kv_cache_config.num_blocks = num_blocks


def _batch(name: str, defer_sampling: bool) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=defer_sampling,
        has_structured_output_spec_decode=False,
    )


def _make_core(executor, scheduler, *, resize_to=None) -> EngineCoreProc:
    core = object.__new__(EngineCoreProc)
    core.model_executor = executor
    core.scheduler = scheduler
    core.input_queue = queue.Queue()
    core._deferred_inputs = deque()
    core.batch_queue_size = 2
    core.batch_queue = deque(maxlen=2)
    core.is_pooling_model = False
    core.is_mm_encoder_only = False
    core.num_executed_steps = 0
    core.log_stats = False
    core.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_logging_iteration_details=False),
        cache_config=SimpleNamespace(
            enable_extensible_kv_cache=True, num_gpu_blocks=10, kv_cache_size_tokens=0
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1, distributed_executor_backend="mp"
        ),
    )
    core.has_work = lambda: True  # type: ignore[method-assign]
    core._should_throttle_prefills = lambda: False  # type: ignore[method-assign]
    core._process_aborts_queue = lambda: None  # type: ignore[method-assign]
    core._attach_iteration_details = lambda *a: None  # type: ignore[method-assign]
    core._attach_weight_version = lambda *a: None  # type: ignore[method-assign]
    core.log_error_detail = lambda so: nullcontext()  # type: ignore[method-assign]
    return core


def test_resize_waits_for_the_batch_between_execute_and_sample():
    """A batch with deferred sampling has its `execute_model` sent while the
    previous batch's output is awaited. A resize request that arrives in that
    wait is held until the step completes, so the workers see
    `execute_model`, `sample_tokens` of every launched batch before the resize.
    """
    executor = RecordingExecutor(resize_to=40)
    scheduler = FakeScheduler(
        executor, [_batch("A", defer_sampling=False), _batch("B", defer_sampling=True)]
    )
    core = _make_core(executor, scheduler)
    executor.futures["A"] = PollableFuture(ready_after=3)
    executor.futures["B"] = PollableFuture(ready_after=1)
    handled: list[Any] = []

    def on_poll(n: int) -> None:
        if n == 1:
            core.input_queue.put((UTILITY, (0, 1, "resize_kv_cache", (40,))))

    executor.futures["A"].on_poll = on_poll

    def handle(request_type, request):
        handled.append(request)
        EngineCore.resize_kv_cache(core, request[3][0])

    core._handle_client_request = handle  # type: ignore[method-assign]

    # Step 1 launches A and returns with the batch queue not full.
    outputs, executed = core.step_with_batch_queue()
    assert outputs is None and executed
    assert executor.calls == ["execute_model:A", "sample_tokens:A"]

    # Step 2 launches B (sampling deferred), waits for A, then samples B.
    core.step_with_batch_queue()
    assert executor.calls == [
        "execute_model:A",
        "sample_tokens:A",
        "execute_model:B",
        "sample_tokens:B",
    ]
    assert not handled
    assert len(core._deferred_inputs) == 1

    # The loop's next pass handles the held request: after every launched batch.
    core._process_input_queue()
    assert executor.calls[-1] == "resize_kv_cache:40"
    assert executor.calls.count("resize_kv_cache:40") == 1
    assert scheduler.grown_to == [40]


def test_resize_between_steps_follows_the_batches_in_flight():
    """With a batch still in the queue (async scheduling), a resize between two
    steps is queued behind its `execute_model` and `sample_tokens`."""
    executor = RecordingExecutor(resize_to=33)
    scheduler = FakeScheduler(
        executor, [_batch("A", False), _batch("B", False), _batch("C", False)]
    )
    core = _make_core(executor, scheduler)
    for name in "ABC":
        executor.futures[name] = PollableFuture(ready_after=1)

    core.step_with_batch_queue()
    assert len(core.batch_queue) == 1  # A is in flight
    EngineCore.resize_kv_cache(core, 33)
    core.step_with_batch_queue()

    assert executor.calls == [
        "execute_model:A",
        "sample_tokens:A",
        "resize_kv_cache:33",
        "execute_model:B",
        "sample_tokens:B",
    ]
    assert scheduler.grown_to == [33]


def test_resize_only_grows_the_scheduler_when_the_workers_did():
    executor = RecordingExecutor(resize_to=10)
    scheduler = FakeScheduler(executor, [])
    core = _make_core(executor, scheduler)
    record = EngineCore.resize_kv_cache(core, 25)
    assert record["num_blocks"] == 10 and record["state"] == "unchanged"
    assert scheduler.grown_to == []
    assert core.vllm_config.cache_config.num_gpu_blocks == 10

    executor.resize_to = 25
    record = EngineCore.resize_kv_cache(core, 25)
    assert record["num_blocks"] == 25 and record["state"] == "finished"
    assert scheduler.grown_to == [25]


def _report(**overrides) -> dict[str, Any]:
    report = {
        "target_blocks": 64,
        "committed_blocks": 10,
        "capacity_blocks": 64,
        "wanted_blocks": 64,
        "measured_blocks": 64,
        "floor_blocks": 64,
        "vote_blocks": 64,
        "floor_bytes": 1 << 30,
        "floor_source": "transient",
        "torch_allocated_bytes": 19 << 30,
        "torch_reserved_bytes": 20 << 30,
        "headroom_after_bytes": 0,
        "cap_after_bytes": 20 << 30,
        "free_after_bytes": 1 << 20,
    }
    report.update(overrides)
    return report


def test_a_grow_short_of_its_target_says_why():
    executor = RecordingExecutor(resize_to=20)
    executor.kv_cache_grow_reports = lambda: [  # type: ignore[attr-defined]
        _report(),
        _report(floor_blocks=20, vote_blocks=20),
    ]
    scheduler = FakeScheduler(executor, [])
    core = _make_core(executor, scheduler)

    record = EngineCore.resize_kv_cache(core, 64)
    assert record["state"] == "finished" and record["num_blocks"] == 20
    assert record["message"].startswith("Grew from 10 to 20 blocks of the 64 asked")
    assert "rank 1" in record["message"]
    assert "--kv-cache-grow-torch-floor-mib" in record["message"]

    # Nothing more keeps the floor: unchanged, and said. (The real scheduler
    # sets num_gpu_blocks when it grows; the fake does not.)
    core.vllm_config.cache_config.num_gpu_blocks = 20
    record = EngineCore.resize_kv_cache(core, 64)
    assert record["state"] == "unchanged"
    assert record["message"].startswith("Not grown from 20 blocks of the 64 asked")

    # The measurement set the count.
    executor.resize_to = 30
    executor.kv_cache_grow_reports = lambda: [  # type: ignore[attr-defined]
        _report(measured_blocks=30, wanted_blocks=30, floor_blocks=40, vote_blocks=30)
    ]
    record = EngineCore.resize_kv_cache(core, 64)
    assert "memory measurement on rank 0 fits 30 blocks" in record["message"]

    # A grow that reached its target says nothing.
    executor.resize_to = 64
    record = EngineCore.resize_kv_cache(core, 64)
    assert record["num_blocks"] == 64 and record["message"] == ""


def test_a_grow_short_of_its_target_without_reports_points_at_the_logs():
    executor = RecordingExecutor(resize_to=12)
    core = _make_core(executor, FakeScheduler(executor, []))
    record = EngineCore.resize_kv_cache(core, 64)
    assert record["num_blocks"] == 12
    assert "KV cache grow floor" in record["message"]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda c: setattr(c.cache_config, "enable_extensible_kv_cache", False),
            "extensible",
        ),
        (
            lambda c: setattr(c.parallel_config, "data_parallel_size", 2),
            "data parallelism",
        ),
        (
            lambda c: setattr(
                c.parallel_config, "distributed_executor_backend", "external_launcher"
            ),
            "external launcher",
        ),
    ],
)
def test_resize_refuses_where_it_cannot_run(change, match):
    executor = RecordingExecutor()
    core = _make_core(executor, FakeScheduler(executor, []))
    change(core.vllm_config)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        EngineCore.resize_kv_cache(core, 25)
    assert executor.calls == []


def test_resize_refuses_while_asleep():
    executor = RecordingExecutor()
    executor.is_sleeping = True
    core = _make_core(executor, FakeScheduler(executor, []))
    with pytest.raises(RuntimeError, match="asleep"):
        EngineCore.resize_kv_cache(core, 25)
    assert executor.calls == []


def test_resize_that_times_out_fails_the_engine():
    """Workers that did not answer in time may answer later, into the next
    call: the engine core gives the executor up instead of going on."""
    executor = RecordingExecutor()

    def resize_kv_cache(target_num_blocks: int) -> int:
        raise TimeoutError("RPC call to resize_kv_cache timed out.")

    executor.resize_kv_cache = resize_kv_cache  # type: ignore[method-assign]
    scheduler = FakeScheduler(executor, [])
    core = _make_core(executor, scheduler)

    with pytest.raises(TimeoutError):
        core.resize_kv_cache(25)

    request_type, _ = core.input_queue.get_nowait()
    assert request_type == EngineCoreRequestType.EXECUTOR_FAILED
    assert scheduler.grown_to == []


def test_executor_bounds_the_wait_for_the_workers():
    from vllm.v1.executor.abstract import Executor

    seen: dict[str, Any] = {}

    def collective_rpc(method, timeout=None, args=(), kwargs=None, non_block=False):
        seen.update(method=method, timeout=timeout, args=args)
        return [12, 12]

    executor = SimpleNamespace(collective_rpc=collective_rpc)
    assert Executor.resize_kv_cache(executor, 12) == 12  # type: ignore[arg-type]
    assert seen["method"] == "resize_kv_cache"
    assert seen["args"] == (12,)
    assert seen["timeout"] is not None and seen["timeout"] > 0

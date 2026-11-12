# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine core's KV cache shrink: a state machine advanced between steps,
with the workers' calls recorded in the order the workers would run them."""

import time

import pytest

from tests.v1.engine.test_kv_cache_resize import (
    PollableFuture,
    _batch,
    _make_core,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCore

MIB = 1 << 20


class ShrinkExecutor:
    """Records execute/sample/plan/shrink calls; a shrink may fail."""

    def __init__(self, num_blocks: int = 100, chunked: bool = True) -> None:
        self.calls: list[str] = []
        self.is_sleeping = False
        self.futures: dict[str, PollableFuture] = {}
        self.committed = num_blocks
        self.chunked = chunked
        # Each entry: an exception to raise from the next shrink, and whether
        # the workers changed anything before raising.
        self.shrink_failures: list[tuple[Exception, bool]] = []
        self.sleeps: list[int] = []

    def execute_model(self, scheduler_output, non_block=False):
        from concurrent.futures import Future

        self.calls.append(f"execute_model:{scheduler_output.name}")
        future: Future = Future()
        future.set_result(None)
        return future

    def sample_tokens(self, grammar_output, non_block=False):
        name = self.sampled_name
        self.calls.append(f"sample_tokens:{name}")
        return self.futures[name]

    def kv_cache_shrink_plan(self, num_blocks: int) -> list[dict]:
        self.calls.append(f"plan:{num_blocks}")
        releasable = (self.committed - num_blocks) * MIB if self.chunked else 0
        return [
            {
                "num_committed_blocks": self.committed,
                "releasable_bytes": releasable,
                "release_chunk_bytes": 4 * MIB if self.chunked else None,
            }
        ] * 2

    def kv_cache_blocks_releasable(self, num_bytes: int) -> list[int | None]:
        return [-(-num_bytes // MIB), -(-num_bytes // MIB) + 2]

    def shrink_kv_cache(self, num_blocks: int) -> list[int]:
        self.calls.append(f"shrink:{num_blocks}")
        if self.shrink_failures:
            error, changed = self.shrink_failures.pop(0)
            if changed:
                self.committed = num_blocks
            raise error
        released = (self.committed - num_blocks) * MIB
        self.committed = num_blocks
        return [released, released + MIB]

    def resize_kv_cache(self, target_num_blocks: int) -> int:
        self.calls.append(f"grow:{target_num_blocks}")
        return target_num_blocks

    def sleep(self, level: int) -> None:
        self.sleeps.append(level)


class ShrinkScheduler:
    """Batches by name, and a fence whose held blocks the test sets."""

    def __init__(self, executor: ShrinkExecutor, cache_config, batches=()):
        self.executor = executor
        self.cache_config = cache_config
        self.batches = list(batches)
        self.fenced_to: int | None = None
        self.held = 0
        self.floor = 10
        self.preempted = 0
        self.events: list[str] = []

    def has_requests(self) -> bool:
        return bool(self.batches)

    def schedule(self, throttle_prefills=False):
        batch = self.batches.pop(0)
        self.executor.sampled_name = batch.name
        return batch

    def get_grammar_bitmask(self, scheduler_output):
        self.executor.sampled_name = scheduler_output.name
        return None

    def update_from_output(self, scheduler_output, model_output):
        return {}

    def min_kv_cache_blocks(self) -> int:
        return self.floor

    def begin_kv_cache_shrink(self, num_blocks: int) -> None:
        if num_blocks < self.floor:
            raise ValueError("below the max_model_len floor")
        assert self.fenced_to is None
        self.fenced_to = num_blocks
        self.events.append(f"fence:{num_blocks}")

    def kv_cache_shrink_progress(self) -> tuple[int, int]:
        assert self.fenced_to is not None
        return self.cache_config.num_gpu_blocks - self.fenced_to, self.held

    def preempt_kv_cache_shrink_holders(self) -> int:
        self.events.append("preempt")
        self.preempted += 1
        self.held = 0
        return 1

    def finish_kv_cache_shrink(self) -> None:
        assert self.fenced_to is not None and self.held == 0
        self.events.append(f"finish:{self.fenced_to}")
        self.cache_config.num_gpu_blocks = self.fenced_to
        self.fenced_to = None

    def abort_kv_cache_shrink(self) -> None:
        self.events.append("abort")
        self.fenced_to = None

    def grow_kv_cache(self, num_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_blocks

    def set_pause_state(self, state) -> None:
        self.events.append(f"pause:{state.name}")


def _core(num_blocks=100, chunked=True, batches=()):
    executor = ShrinkExecutor(num_blocks, chunked)
    core = _make_core(executor, None)
    core.vllm_config.cache_config.num_gpu_blocks = num_blocks
    core.vllm_config.cache_config.kv_cache_size_tokens = 0
    scheduler = ShrinkScheduler(executor, core.vllm_config.cache_config, batches)
    core.scheduler = scheduler
    return core, executor, scheduler


def test_an_idle_engine_shrinks_within_the_call():
    core, executor, scheduler = _core()
    record = core.resize_kv_cache(60)
    assert record["state"] == "finished"
    assert record["num_blocks"] == 60
    assert record["released_bytes_per_rank"] == 40 * MIB
    assert executor.calls == ["plan:60", "shrink:60"]
    assert scheduler.events == ["fence:60", "finish:60"]
    assert core.get_kv_cache_resize_state()["state"] == "finished"


def test_no_worker_call_before_the_drain_and_none_inside_a_step():
    """The shrink's worker call comes only between steps once no request holds
    a fenced block: after `execute_model` and `sample_tokens` of every
    launched batch, also one whose sampling was deferred."""
    core, executor, scheduler = _core(
        batches=[_batch("A", False), _batch("B", True), _batch("C", False)]
    )
    for name in "ABC":
        executor.futures[name] = PollableFuture(ready_after=1)
    scheduler.held = 5

    assert core.resize_kv_cache(60, deadline_s=0)["state"] == "draining"
    assert executor.calls == ["plan:60"]

    core.step_with_batch_queue()  # A
    core._advance_kv_resize()
    core.step_with_batch_queue()  # B launched, A awaited, B sampled
    assert "shrink:60" not in executor.calls
    # The last holder finished during that step.
    scheduler.held = 0
    core._advance_kv_resize()
    core.step_with_batch_queue()

    assert executor.calls == [
        "plan:60",
        "execute_model:A",
        "sample_tokens:A",
        "execute_model:B",
        "sample_tokens:B",
        "shrink:60",
        "execute_model:C",
        "sample_tokens:C",
    ]
    assert core.get_kv_cache_resize_state()["state"] == "finished"


def test_the_busy_loop_advances_the_shrink_between_steps():
    core, executor, scheduler = _core()
    scheduler.held = 1
    core.resize_kv_cache(60)
    order: list[str] = []
    calls = iter(range(3))

    def handle_shutdown() -> bool:
        return next(calls) < 2

    core._handle_shutdown = handle_shutdown  # type: ignore[method-assign]
    core._process_input_queue = lambda: order.append("inputs")  # type: ignore[method-assign]
    core._maybe_publish_request_counts = lambda: None  # type: ignore[method-assign]

    def step() -> bool:
        order.append("step")
        scheduler.held = 0
        return True

    core._process_engine_step = step  # type: ignore[method-assign]
    advance = core._advance_kv_resize

    def recorded_advance() -> None:
        order.append("advance")
        advance()

    core._advance_kv_resize = recorded_advance  # type: ignore[method-assign]
    core.enable_fault_tolerance = False
    with pytest.raises(SystemExit):
        type(core).run_busy_loop(core)
    assert order == ["advance", "inputs", "step", "advance", "inputs", "step"]
    assert executor.calls[-1] == "shrink:60"


def test_deadline_aborts_by_default():
    core, executor, scheduler = _core()
    scheduler.held = 3
    core.resize_kv_cache(60, deadline_s=0.01)
    time.sleep(0.02)
    record = core.get_kv_cache_resize_state()
    assert record["state"] == "aborted"
    assert "deadline" in record["message"]
    assert record["num_blocks"] == 100
    assert scheduler.events == ["fence:60", "abort"]
    assert "shrink:60" not in executor.calls


def test_deadline_preempts_the_holders_when_asked():
    core, executor, scheduler = _core()
    scheduler.held = 3
    core.resize_kv_cache(60, deadline_s=0.01, preempt=True)
    time.sleep(0.02)
    record = core.get_kv_cache_resize_state()
    assert record["state"] == "finished"
    assert record["preempted_requests"] == 1
    assert scheduler.events == ["fence:60", "preempt", "finish:60"]


def test_one_resize_at_a_time():
    core, executor, scheduler = _core()
    scheduler.held = 3
    first = core.resize_kv_cache(60)
    assert first["state"] == "draining"
    # The same request again reads the shrink in progress.
    assert core.resize_kv_cache(60)["state"] == "draining"
    with pytest.raises(RuntimeError, match="one resize at a time"):
        core.resize_kv_cache(50)
    with pytest.raises(RuntimeError, match="one resize at a time"):
        core.resize_kv_cache(200)  # a grow
    assert scheduler.events == ["fence:60"]
    assert executor.calls == ["plan:60"]


def test_refusals_before_anything_is_fenced():
    core, executor, scheduler = _core(chunked=False)
    with pytest.raises(ValueError, match="kv-cache-release-chunk-mib"):
        core.resize_kv_cache(60)
    assert scheduler.events == []

    core, executor, scheduler = _core()
    with pytest.raises(ValueError, match="floor"):
        core.resize_kv_cache(5)
    assert scheduler.events == []


def test_a_release_the_workers_refused_aborts():
    core, executor, scheduler = _core()
    executor.shrink_failures = [(RuntimeError("no rank shrank"), False)]
    record = core.resize_kv_cache(60)
    assert record["state"] == "aborted"
    assert "refused" in record["message"]
    assert scheduler.events == ["fence:60", "abort"]
    assert core.vllm_config.cache_config.num_gpu_blocks == 100


def test_a_release_past_the_point_of_no_return_keeps_the_fence_and_retries(
    monkeypatch,
):
    core, executor, scheduler = _core()
    monkeypatch.setattr(EngineCore, "KV_RESIZE_RETRY_INTERVAL_S", 0.01)
    executor.shrink_failures = [(RuntimeError("unmap failed on rank 1"), True)]
    record = core.resize_kv_cache(60)
    assert record["state"] == "unfinished"
    assert scheduler.fenced_to == 60  # workers and scheduler stay together
    with pytest.raises(RuntimeError, match="cannot be aborted"):
        core.abort_kv_cache_resize()
    # Retried no sooner than the interval.
    assert core.get_kv_cache_resize_state()["state"] == "unfinished"
    assert executor.calls.count("shrink:60") == 1
    time.sleep(0.02)
    assert core.get_kv_cache_resize_state()["state"] == "finished"
    assert executor.calls.count("shrink:60") == 2
    assert scheduler.events == ["fence:60", "finish:60"]


def test_a_timeout_fails_the_engine():
    core, executor, scheduler = _core()

    def shrink(num_blocks):
        raise TimeoutError("RPC call to shrink_kv_cache timed out.")

    executor.shrink_kv_cache = shrink  # type: ignore[method-assign]
    record = core.resize_kv_cache(60)
    assert record["state"] == "unfinished"
    request_type, _ = core.input_queue.get_nowait()
    assert request_type == EngineCoreRequestType.EXECUTOR_FAILED


def _pausable(core):
    from vllm.v1.engine.core import CheckpointState

    core._finish_pause = lambda *a: None  # type: ignore[method-assign]
    core._checkpoint_state = CheckpointState.RUNNING
    core._idle_state_callbacks = []
    core._pause_complete = lambda: True  # type: ignore[method-assign]
    return core


@pytest.mark.parametrize("call", ["pause", "sleep"])
def test_pause_and_sleep_abort_a_draining_shrink(call):
    core, executor, scheduler = _core()
    _pausable(core)
    scheduler.held = 2
    core.resize_kv_cache(60)
    if call == "pause":
        core.pause_scheduler(mode="keep", clear_cache=False)
    else:
        EngineCore.sleep(core, level=1, mode="keep")
    assert core.get_kv_cache_resize_state()["state"] == "aborted"
    assert scheduler.events[:2] == ["fence:60", "abort"]


def test_sleep_waits_for_an_unfinished_shrink_and_pause_keeps_it():
    core, executor, scheduler = _core()
    _pausable(core)
    executor.shrink_failures = [
        (RuntimeError("unmap failed"), True),
        (RuntimeError("unmap failed again"), True),
        (RuntimeError("and again"), True),
    ]
    assert core.resize_kv_cache(60)["state"] == "unfinished"

    # A pause changes no memory: the fence stays, one more try is made.
    core.pause_scheduler(mode="keep", clear_cache=False)
    assert core.get_kv_cache_resize_state()["state"] == "unfinished"
    # A sleep would drop the KV cache on ranks in different states: refused.
    with pytest.raises(RuntimeError, match="must finish before sleeping"):
        EngineCore.sleep(core, level=1, mode="keep")
    assert executor.sleeps == []
    # Once the release goes through, sleep proceeds.
    EngineCore.sleep(core, level=1, mode="keep")
    assert executor.sleeps == [1]
    assert core.get_kv_cache_resize_state()["state"] == "finished"


class StubHolder:
    """A holder of other device memory, for tests only."""

    def __init__(self) -> None:
        self.told: list[tuple[int, int]] = []

    def on_kv_cache_released(self, bytes_per_rank: int, num_blocks: int) -> None:
        self.told.append((bytes_per_rank, num_blocks))


def test_holders_hear_what_a_shrink_released_and_may_ask_for_one():
    core, executor, scheduler = _core()
    holder = StubHolder()
    core.register_kv_cache_memory_holder(holder)

    # 10 MiB per rank: rank 1 needs 12 tail blocks for it, so 12 go.
    record = core.request_kv_cache_release(10 * MIB)

    assert record["state"] == "finished"
    assert record["num_blocks"] == 88
    assert holder.told == [(12 * MIB, 88)]


def test_state_read_when_nothing_happened():
    core, executor, scheduler = _core()
    assert core.get_kv_cache_resize_state() == {"state": "idle", "num_blocks": 100}
    assert core.resize_kv_cache(100)["state"] == "unchanged"
    assert executor.calls == []

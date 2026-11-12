# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV cache gauges (`vllm:kv_cache_num_blocks`, `..._fenced_blocks`) are
fed by the scheduler's stats. A resize happens between steps, so the engine
core sends the stats when the resize ends (grow, finished shrink, abort) and
when a shrink fences, instead of with the next step's output."""

import queue

from tests.v1.engine.test_kv_cache_shrink import _core
from vllm.v1.metrics.stats import SchedulerStats


def _with_stats(num_blocks=100):
    core, executor, scheduler = _core(num_blocks)

    def make_stats():
        fenced = (
            0
            if scheduler.fenced_to is None
            else scheduler.cache_config.num_gpu_blocks - scheduler.fenced_to
        )
        return SchedulerStats(
            num_gpu_blocks=scheduler.cache_config.num_gpu_blocks,
            num_fenced_gpu_blocks=fenced,
        )

    scheduler.make_stats = make_stats
    core.output_queue = queue.Queue()
    return core, scheduler


def _sent(core) -> list[tuple[int, int]]:
    sent = []
    while not core.output_queue.empty():
        _, outputs = core.output_queue.get_nowait()
        stats = outputs.scheduler_stats
        sent.append((stats.num_gpu_blocks, stats.num_fenced_gpu_blocks))
    return sent


def test_a_grow_sends_the_new_size_at_once():
    core, _ = _with_stats()
    core.resize_kv_cache(140)
    assert _sent(core) == [(140, 0)]


def test_an_unchanged_resize_sends_nothing():
    core, _ = _with_stats()
    core.resize_kv_cache(100)
    assert _sent(core) == []


def test_a_shrink_sends_the_fence_and_then_the_new_size():
    core, scheduler = _with_stats()
    scheduler.held = 2
    core.resize_kv_cache(60)
    assert _sent(core) == [(100, 40)]
    scheduler.held = 0
    core._advance_kv_resize()
    assert _sent(core) == [(60, 0)]


def test_an_aborted_shrink_sends_the_pool_without_the_fence():
    core, scheduler = _with_stats()
    scheduler.held = 2
    core.resize_kv_cache(60)
    _sent(core)
    core.abort_kv_cache_resize("test")
    assert _sent(core) == [(100, 0)]


def test_an_engine_core_without_an_output_queue_sends_nothing():
    core, _, _ = _core()
    record = core.resize_kv_cache(140)  # no output queue: nothing to send
    assert record["num_blocks"] == 140

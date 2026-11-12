# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A collective RPC that one rank fails takes every rank's reply off its
queue before it raises, so the next RPC reads its own replies."""

from collections import deque
from types import SimpleNamespace

import pytest

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc

pytestmark = pytest.mark.cpu_test

OK = WorkerProc.ResponseStatus.SUCCESS
FAILED = WorkerProc.ResponseStatus.FAILURE


class FakeQueue:
    """A rank's response queue; `script` holds that rank's replies to the
    calls in order."""

    def __init__(self, script: list) -> None:
        self.script = deque(script)
        self.ready: deque = deque()

    def dequeue(self, timeout=None):
        if not self.ready:
            raise TimeoutError
        return self.ready.popleft()

    def readable(self) -> bool:
        return bool(self.ready)


class FakeBroadcast:
    def __init__(self, queues: list[FakeQueue]) -> None:
        self.queues = queues

    def enqueue(self, message) -> None:
        # Each rank answers the call (a rank whose script says None never does).
        for q in self.queues:
            reply = q.script.popleft()
            if reply is not None:
                q.ready.append(reply)


def _executor(scripts: list[list]) -> MultiprocExecutor:
    executor = object.__new__(MultiprocExecutor)
    queues = [FakeQueue(script) for script in scripts]
    executor.rpc_broadcast_mq = FakeBroadcast(queues)
    executor.response_mqs = queues
    executor.futures_queue = deque()
    executor.is_failed = False
    executor.failure_callback = None
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_fault_tolerance=False)
    )
    return executor


def test_the_next_call_reads_its_own_replies_after_a_failed_one():
    executor = _executor(
        [
            [(FAILED, "boom 0"), (OK, "release 0"), (OK, "grow 0")],
            [(FAILED, "boom 1"), (OK, "release 1"), (OK, "grow 1")],
            [(OK, "commit 2"), (OK, "release 2"), (OK, "grow 2")],
        ]
    )
    with pytest.raises(RuntimeError, match="boom 0"):
        executor.collective_rpc("commit")
    assert executor.collective_rpc("release") == [
        "release 0",
        "release 1",
        "release 2",
    ]
    assert executor.collective_rpc("grow") == ["grow 0", "grow 1", "grow 2"]
    assert not executor.is_failed


def test_a_rank_that_never_replies_after_a_failure_fails_the_executor():
    executor = _executor([[(FAILED, "boom 0")], [None]])
    executor.DRAIN_AFTER_FAILURE_S = 0.0
    called = []
    executor.failure_callback = lambda: called.append(True)
    with pytest.raises(RuntimeError, match="boom 0"):
        executor.collective_rpc("commit")
    assert executor.is_failed
    assert called == [True]


def test_a_single_rank_reply_is_unchanged():
    executor = _executor([[(OK, "a")], [(OK, "b")]])
    assert executor.collective_rpc("x", unique_reply_rank=1) == "b"

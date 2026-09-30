# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-0014a: the env-gated offload trace observes and never changes results."""

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_scheduler import (
    _hybrid_lookup,
    _make_partial_tail_request,
    _make_sparse_retention_scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    b70_offload as b70,
)


def _capture(monkeypatch) -> list[str]:
    lines: list[str] = []
    monkeypatch.setattr(b70, "log", lambda fmt, *args: lines.append(fmt % args))
    return lines


@pytest.mark.parametrize("level", ["", "1", "2"])
def test_trace_logs_junction_and_zeroed_lookup(monkeypatch, level):
    monkeypatch.setenv("B70_OFFLOAD_TRACE", level)
    monkeypatch.setenv("B70_OFFLOAD_TRACE_PERIOD_S", "3600")
    lines = _capture(monkeypatch)
    scheduler = _make_sparse_retention_scheduler()
    request = _make_partial_tail_request(scheduler)
    request.shared_prefix_boundary = 0
    scheduler.manager.lookup.side_effect = _hybrid_lookup(set())

    # Same result and junction as without the trace.
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.shared_prefix_boundary == 16

    if not level:
        assert lines == []
        return
    assert any(line.startswith("enabled scheduler trace: level=") for line in lines)
    assert any(
        line.startswith("junction-set req=req") and "boundary=16" in line
        for line in lines
    )
    assert any(
        line.startswith("zeroed_by_sparse_group req=req")
        and "full_attention_hit=16" in line
        for line in lines
    )
    assert any(line.startswith("lookup req=req") for line in lines) == (
        level == "2"
    )

    # A second lookup of the same request logs no repeated event.
    before = len([x for x in lines if not x.startswith("lookup ")])
    scheduler.get_num_new_matched_tokens(request, 0)
    after = len([x for x in lines if not x.startswith("lookup ")])
    assert after == before

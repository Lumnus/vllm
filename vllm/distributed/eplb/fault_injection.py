# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation failures on demand, to exercise EPLB's skip paths on a serving
engine (the runtime control surface's `fault.arm`, development mode only).

Nothing is armed unless asked, and an unarmed `maybe_fail` is one dictionary
test. A fault is armed in one process (one EP rank) for one stage and a
number of shots; each shot raises `torch.OutOfMemoryError` at the stage's
first allocation, the error the device raises when it is full:

- `load`: the logical load window of a rearrangement (phase 1 of
  `EplbState.rearrange`, before the first collective);
- `plan`: the plan of a rearrangement (phase 2, before any weight moves);
- `copy_rows`: the row index upload of `copy_rows`, which copies received
  experts from the transfer buffer into the weights (after a transfer, on
  every layer of a synchronous rearrangement and at an async layer's commit).
"""

import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

STAGES = ("load", "plan", "copy_rows")

_lock = threading.Lock()
_armed: dict[str, int] = {}
_fired: dict[str, int] = {}


def arm(stage: str, count: int = 1) -> None:
    """Fail the next `count` allocations of `stage` in this process."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; one of {', '.join(STAGES)}")
    if count < 1:
        raise ValueError("count must be at least 1")
    with _lock:
        _armed[stage] = count


def clear(stage: str | None = None) -> None:
    """Disarm `stage`, or every stage."""
    with _lock:
        if stage is None:
            _armed.clear()
        else:
            _armed.pop(stage, None)


def armed() -> dict[str, int]:
    """The shots left per armed stage."""
    with _lock:
        return dict(_armed)


def fired() -> dict[str, int]:
    """The shots fired per stage since the process started."""
    with _lock:
        return dict(_fired)


def maybe_fail(stage: str) -> None:
    """Raise the injected allocation failure if `stage` is armed here."""
    if not _armed:
        return
    with _lock:
        left = _armed.get(stage)
        if not left:
            return
        if left == 1:
            del _armed[stage]
        else:
            _armed[stage] = left - 1
        _fired[stage] = _fired.get(stage, 0) + 1
    logger.warning(
        "EPLB fault injection: failing the allocation of stage %s (%d more armed).",
        stage,
        left - 1,
    )
    raise torch.OutOfMemoryError(
        f"injected allocation failure at EPLB stage {stage} (runtime control "
        "fault injection)"
    )

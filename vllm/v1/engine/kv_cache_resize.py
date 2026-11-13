# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine core's record of a KV cache resize, and the interface for a
holder of other device memory that takes what a shrink gives back."""

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class KVCacheResizeState(enum.StrEnum):
    """Where a resize is. A grow goes straight to FINISHED (or UNCHANGED).

    A shrink: DRAINING (the tail of the pool is fenced, requests holding
    fenced blocks finish or are preempted), then RELEASING (the workers give
    the memory back; within one step boundary), then FINISHED. From
    DRAINING it may be ABORTED (deadline, pause, sleep, a refusal before the
    workers changed anything); from RELEASING it may become UNFINISHED (a
    worker failed after the point of no return: the fence stays, and the
    release is repeated until every worker is done).
    """

    UNCHANGED = "unchanged"
    DRAINING = "draining"
    RELEASING = "releasing"
    UNFINISHED = "unfinished"
    FINISHED = "finished"
    ABORTED = "aborted"

    @property
    def active(self) -> bool:
        return self in (
            KVCacheResizeState.DRAINING,
            KVCacheResizeState.RELEASING,
            KVCacheResizeState.UNFINISHED,
        )


@dataclass
class KVCacheResize:
    direction: str  # "grow" or "shrink"
    from_blocks: int
    to_blocks: int
    state: KVCacheResizeState
    deadline_s: float = 0.0
    preempt: bool = False
    started: float = field(default_factory=time.monotonic)
    # Blocks the fence takes out, and those still held by requests.
    fenced_blocks: int = 0
    held_blocks: int = 0
    preempted_requests: int = 0
    # Bytes each rank released, once released.
    released_bytes: list[int] = field(default_factory=list)
    message: str = ""
    # When the release was last tried (UNFINISHED retries are spaced).
    last_release_attempt: float = 0.0

    def deadline_passed(self, now: float) -> bool:
        return self.deadline_s > 0 and now - self.started >= self.deadline_s

    def record(self, num_blocks: int) -> dict[str, Any]:
        """What the engine reports: num_blocks is the pool size in force."""
        return {
            "state": self.state.value,
            "direction": self.direction,
            "num_blocks": num_blocks,
            "from_blocks": self.from_blocks,
            "to_blocks": self.to_blocks,
            "fenced_blocks": self.fenced_blocks,
            "held_blocks": self.held_blocks,
            "preempted_requests": self.preempted_requests,
            "released_bytes_per_rank": (
                min(self.released_bytes) if self.released_bytes else 0
            ),
            "elapsed_s": round(time.monotonic() - self.started, 3),
            "message": self.message,
        }


def _gib(num_bytes: int | None) -> str:
    return "none" if num_bytes is None else f"{num_bytes / 2**30:.2f} GiB"


def grow_message(
    target: int,
    old: int,
    new: int,
    reports: list[dict[str, Any] | None] | None,
) -> str:
    """Why a grow toward target stopped at new blocks (from old), from the
    workers' reports of their vote (`GPUWorker.kv_cache_grow_report`); empty
    when it reached the target."""
    if new >= target:
        return ""
    head = (
        f"Not grown from {old} blocks"
        if new <= old
        else f"Grew from {old} to {new} blocks"
    ) + f" of the {target} asked"
    ranked = [(r, rep) for r, rep in enumerate(reports or []) if rep]
    if not ranked:
        return (
            f"{head}; see the workers' 'KV cache grow floor' and 'KV cache resize "
            "measurement' log lines."
        )
    rank, rep = min(ranked, key=lambda item: item[1]["vote_blocks"])
    if rep["vote_blocks"] > new:
        return (
            f"{head}: every rank voted for at least {rep['vote_blocks']} blocks, "
            "but memory could not be mapped on every rank (see the workers' "
            "'Could not map memory' lines)."
        )
    if rep["floor_blocks"] < rep["wanted_blocks"]:
        return (
            f"{head}: rank {rank} keeps torch able to reserve "
            f"{_gib(rep['floor_bytes'])} more ({rep['floor_source']}); at "
            f"{rep['wanted_blocks']} blocks it would have "
            f"{_gib(rep['headroom_after_bytes'])} (torch reserved "
            f"{_gib(rep['torch_reserved_bytes'])}, allocated "
            f"{_gib(rep['torch_allocated_bytes'])}; cap after the grow "
            f"{_gib(rep['cap_after_bytes'])}, device free after "
            f"{_gib(rep['free_after_bytes'])}), so at most "
            f"{rep['floor_blocks']} blocks. --kv-cache-grow-torch-floor-mib sets "
            "the floor."
        )
    if rep["measured_blocks"] < min(rep["target_blocks"], rep["capacity_blocks"]):
        return (
            f"{head}: the memory measurement on rank {rank} fits "
            f"{rep['measured_blocks']} blocks."
        )
    return f"{head}: the reservation holds {rep['capacity_blocks']} blocks."


class KVCacheMemoryHolder(Protocol):
    """Something in the engine that holds device memory the KV cache can
    make room for (expert rows, a vision tower). It is told when a shrink
    has given memory back, and may ask the engine core for a shrink with
    `EngineCore.request_kv_cache_release`. How it uses the memory is its own
    business; nothing here sets policy."""

    def on_kv_cache_released(self, bytes_per_rank: int, num_blocks: int) -> None:
        """A shrink finished: every rank gave back at least bytes_per_rank;
        the KV cache now holds num_blocks blocks."""

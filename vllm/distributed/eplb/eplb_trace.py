# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A trace of the routed expert load over time (`EPLBConfig.trace_path`).

EPLB keeps the load per physical expert of the current step in
`expert_load_pass`, written by the router's recording kernel (inside the
captured graph). The trace does not add a second accumulator beside it: on the
steps where EPLB does not reset `expert_load_pass` for its own window, the
kernel simply keeps adding into it, so those steps cost no device operation
outside the graph at all. Where the pass is reset for the window, one add
folds it into the trace's physical accumulator first (one operation for all
layers of a model, not one per layer).

The physical counts are mapped to logical experts when they are about to lose
their meaning: every `EPLBConfig.trace_interval` steps, and before a committed
placement change (`EplbModelState.map_change_hook`), so an interval that spans
a rearrangement is attributed with the placement each of its steps ran on. At
the end of an interval the logical counts are copied to a reusable pinned host
buffer with a non-blocking copy and an event, and put on a queue. A background
thread waits for the event (its own wait, not the stepping thread's), encodes
the record and appends it to the file, rotating by size. No step waits for the
device, nothing is added to a captured graph, and the stepping thread does no
compression or file work.

`EPLBConfig.trace_sample` lowers the cost inside the graph: the recording
kernel only counts the first `trace_sample` steps of each `trace_interval`
steps (the flag it reads flips twice per interval). A record then sums the
sampled steps; the meta record says how many.

The loads are the counts of this rank before EPLB's all-reduce, so each rank
writes its own files, `<trace_path>.rank<R>` (the newest) and, rotated,
`<trace_path>.rank<R>.<n>` (n = 1, 2, ... oldest first); a reader that wants
the global load adds the ranks record by record (same `step`).

File format (append-only, one file per rank and segment):

    file   := MAGIC record*
    record := u32le length, zlib(payload)
    payload (meta, kind 0)  := 0x00, JSON object
    payload (data, kind 1)  := <BqdB: 1, step, wall time, itemsize>, body
    body   := the counts [layers, logical experts] of the interval as
              unsigned little-endian integers of `itemsize` bytes (1, 2 or
              4: the smallest that holds the largest count), byte-plane
              shuffled (all the first bytes, then all the second bytes, ...)

The meta record is the first record of every file and has `version` (2),
`layers` (the layer names, rows of the counts), `num_logical_experts`,
`interval`, `sample` (steps counted per record; `interval` when every step is),
`rank`. `step` is the number of decoded engine steps traced so far when the
record's interval ended; `wall` is `time.time()` at that moment. Version 1
files (no `sample`) are read as sampling every step.
"""

import atexit
import json
import os
import queue
import re
import struct
import threading
import time
import zlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY

logger = init_logger(__name__)

TRACE_MAGIC = b"EPLBTRC1"
TRACE_VERSION = 2
KIND_META = 0
KIND_DATA = 1
_DATA_HEAD = struct.Struct("<BqdB")
_LENGTH = struct.Struct("<I")
_MAX_QUEUED = 8  # intervals waiting for the writer; more are dropped
_CLOSE_TIMEOUT_S = 10.0


def encode_counts(counts: np.ndarray) -> tuple[int, bytes]:
    """The counts as (itemsize, byte-plane shuffled little-endian body)."""
    largest = int(counts.max()) if counts.size else 0
    itemsize = 1 if largest < 1 << 8 else 2 if largest < 1 << 16 else 4
    flat = np.ascontiguousarray(counts, dtype=f"<u{itemsize}").reshape(-1)
    return itemsize, flat.view(np.uint8).reshape(-1, itemsize).T.tobytes()


def encode_record(payload: bytes) -> bytes:
    packed = zlib.compress(payload, 6)
    return _LENGTH.pack(len(packed)) + packed


def encode_meta(meta: dict[str, Any]) -> bytes:
    return encode_record(bytes([KIND_META]) + json.dumps(meta).encode())


def encode_data(step: int, wall: float, counts: np.ndarray) -> bytes:
    itemsize, body = encode_counts(counts)
    head = _DATA_HEAD.pack(KIND_DATA, step, wall, itemsize)
    return encode_record(head + body)


def segment_paths(path: str) -> list[str]:
    """The rotated segments of `path` (`<path>.<n>`), oldest first."""
    directory, base = os.path.split(path)
    found = []
    for name in os.listdir(directory or "."):
        match = re.fullmatch(re.escape(base) + r"\.(\d+)", name)
        if match:
            found.append((int(match.group(1)), os.path.join(directory, name)))
    return [p for _, p in sorted(found)]


class _Slot:
    """The trace's device state for one model."""

    def __init__(self, load: torch.Tensor, num_logical: int) -> None:
        self.acc = torch.zeros_like(load)
        """Physical counts since the last flush."""
        self.logical = torch.zeros(
            (load.shape[0], num_logical + 1), dtype=load.dtype, device=load.device
        )
        """Logical counts of the open interval; the last column takes the
        empty slots (-1) and is dropped."""
        self.dirty = False
        """`expert_load_pass` holds counts the trace has not taken."""


class ExpertLoadTracer:
    """Sums the routed load and appends it, every `interval` steps, to
    `<path>.rank<rank>`. Held by `EplbState`; see the module docstring."""

    def __init__(
        self,
        path: str,
        interval: int,
        rank: int,
        *,
        sample: int | None = None,
        rotate_bytes: int = 0,
        keep: int = 0,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.path = f"{path}.rank{rank}"
        self.interval = interval
        self.sample = interval if sample is None else sample
        assert 0 < self.sample <= interval
        self.rank = rank
        self.rotate_bytes = rotate_bytes
        self.keep = keep
        self.steps = 0
        self.dropped = 0
        self._event_factory = event_factory
        self._slots: dict[int, _Slot] = {}
        self._meta: dict[str, Any] | None = None
        self._shape: tuple[int, int] | None = None
        self._free: queue.SimpleQueue[torch.Tensor] = queue.SimpleQueue()
        self._allocated = 0
        self._queue: queue.Queue[tuple[int, float, Any, torch.Tensor] | None] = (
            queue.Queue()
        )
        self._thread: threading.Thread | None = None
        self._failed = False
        self._size = 0
        self._begin_file()

    @property
    def failed(self) -> bool:
        """The writer gave up (the file cannot be written)."""
        return self._failed

    # ---- the stepping thread: device work only, a few operations per step

    def records_next_step(self) -> bool:
        """Whether the recording kernel should count the step about to run."""
        return self.steps % self.interval < self.sample

    def _slot(self, model_state: Any) -> _Slot:
        load = model_state.expert_load_pass
        slot = self._slots.get(id(model_state))
        if slot is None or slot.acc.shape != load.shape:
            slot = _Slot(load, model_state.model.num_logical_experts)
            self._slots[id(model_state)] = slot
        return slot

    def accumulate(self, model_states: Sequence[Any], pass_reset: bool) -> None:
        """The hook before EPLB's window step. With `pass_reset` the window is
        about to copy and zero `expert_load_pass`: one add takes it first.
        Without, the pass keeps accumulating in the graph and nothing is
        done."""
        for model_state in model_states:
            slot = self._slot(model_state)
            if pass_reset:
                slot.acc += model_state.expert_load_pass
                slot.dirty = False
            else:
                slot.dirty = True

    def fold_pending(self, model_states: Sequence[Any]) -> None:
        """Take what `expert_load_pass` has accumulated and clear it; call
        before the window starts to copy the pass (it would see old steps)."""
        for model_state in model_states:
            self._fold(model_state, self._slot(model_state))

    def _fold(self, model_state: Any, slot: _Slot) -> None:
        if slot.dirty:
            slot.acc += model_state.expert_load_pass
            model_state.expert_load_pass.zero_()
            slot.dirty = False

    def _flush(self, model_state: Any, slot: _Slot) -> None:
        """Map the physical counts to logical experts with the placement they
        ran on and add them to the interval's logical counts."""
        p2l = model_state.physical_to_logical_map
        num_logical = slot.logical.shape[1] - 1
        slot.logical.scatter_add_(
            dim=-1,
            index=p2l.masked_fill(p2l < 0, num_logical).long(),
            src=slot.acc,
        )
        slot.acc.zero_()

    def before_map_change(self, model_state: Any) -> None:
        """Called before a placement is committed to `model_state`: the counts
        so far belong to the placement still in force."""
        if self._failed:
            return
        slot = self._slot(model_state)
        self._fold(model_state, slot)
        self._flush(model_state, slot)

    def end_step(self, model_states: Sequence[Any]) -> None:
        """Count a traced step; at the end of an interval queue its copy to
        the host."""
        self.steps += 1
        if self.steps % self.interval == 0:
            self._queue_interval(model_states)

    def _queue_interval(self, model_states: Sequence[Any]) -> None:
        slots = []
        for model_state in model_states:
            slot = self._slot(model_state)
            self._fold(model_state, slot)
            self._flush(model_state, slot)
            slots.append(slot)
        shape = (
            sum(s.logical.shape[0] for s in slots),
            slots[0].logical.shape[1],
        )
        if any(s.logical.shape[1] != shape[1] for s in slots):
            logger.warning("EPLB trace: models differ in logical experts; stopped.")
            self._failed = True
            return
        host = self._take_host(shape, slots[0].logical)
        if host is None:
            self.dropped += 1
            if self.dropped == 1:
                logger.warning(
                    "EPLB trace: the writer is %d intervals behind; dropping.",
                    _MAX_QUEUED,
                )
        else:
            row = 0
            for slot in slots:
                host[row : row + slot.logical.shape[0]].copy_(
                    slot.logical, non_blocking=True
                )
                row += slot.logical.shape[0]
            event = None
            if self._event_factory is not None:
                event = self._event_factory()
            elif slots[0].logical.device.type != "cpu":
                event = torch.Event()
            if event is not None:
                event.record()
            if self._meta is None:
                self._meta = {
                    "version": TRACE_VERSION,
                    "layers": [
                        layer.layer_name
                        for model_state in model_states
                        for layer in model_state.model.moe_layers
                    ],
                    "num_logical_experts": shape[1] - 1,
                    "interval": self.interval,
                    "sample": self.sample,
                    "rank": self.rank,
                }
            self._start_writer()
            self._queue.put((self.steps, time.time(), event, host))
        for slot in slots:
            slot.logical.zero_()

    def _take_host(
        self, shape: tuple[int, int], like: torch.Tensor
    ) -> torch.Tensor | None:
        """A reusable pinned buffer, or None when the writer is behind."""
        if self._shape not in (None, shape):
            self._failed = True
            return None
        self._shape = shape
        try:
            return self._free.get_nowait()
        except queue.Empty:
            pass
        if self._allocated >= _MAX_QUEUED:
            return None
        self._allocated += 1
        return torch.empty(
            shape, dtype=like.dtype, pin_memory=PIN_MEMORY and like.device.type != "cpu"
        )

    # ---- the writer thread

    def _start_writer(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="eplb-trace-writer", daemon=True
            )
            self._thread.start()
            atexit.register(self.close)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                step, wall, event, host = item
                if event is not None:
                    event.synchronize()
                if not self._failed:
                    self._append(encode_data(step, wall, host.numpy()[:, :-1]))
                self._free.put(host)
            except Exception as err:  # the writer must not die silently
                logger.warning("EPLB: the expert trace writer stopped: %s", err)
                self._failed = True
            finally:
                self._queue.task_done()

    def drain(self) -> None:
        """Wait until every queued interval is written. For tests and
        shutdown; the serving path never calls it."""
        self._queue.join()

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._thread = None
        self._queue.put(None)
        thread.join(_CLOSE_TIMEOUT_S)

    # ---- files

    def _begin_file(self) -> None:
        """A file left by an earlier run is rotated aside, so every file
        starts with its own meta record."""
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                self._rotate()
            else:
                self._size = 0
        except OSError as err:
            self._fail(err)

    def _rotate(self) -> None:
        segments = segment_paths(self.path)
        last = int(segments[-1].rsplit(".", 1)[1]) if segments else 0
        os.replace(self.path, f"{self.path}.{last + 1}")
        self._size = 0
        if self.keep > 0:
            for old in segment_paths(self.path)[: -self.keep]:
                os.remove(old)

    def _append(self, record: bytes) -> None:
        try:
            full = self._size + len(record) > self.rotate_bytes
            if self.rotate_bytes and full and self._size > 0:
                self._rotate()
            with open(self.path, "ab") as handle:
                if self._size == 0:
                    assert self._meta is not None
                    head = TRACE_MAGIC + encode_meta(self._meta)
                    handle.write(head)
                    self._size += len(head)
                handle.write(record)
                self._size += len(record)
        except OSError as err:
            self._fail(err)

    def _fail(self, err: OSError) -> None:
        logger.warning("EPLB: cannot write the expert trace %s: %s", self.path, err)
        self._failed = True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A trace of the routed expert load over time (`EPLBConfig.trace_path`).

EPLB keeps the load per physical expert of the current step in
`expert_load_pass`, written by the router's recording kernel (inside the
captured graph). The trace sums those passes on the device for
`EPLBConfig.trace_interval` steps, maps the sum to logical experts, copies it
to pinned host memory with a non-blocking copy and an event, and writes it out
a later step, once the event has completed. No step waits for the device and
nothing is added to a captured graph.

The loads are the counts of this rank before EPLB's all-reduce, so each rank
writes its own file, `<trace_path>.rank<R>`, and a reader that wants the global
load adds the files record by record (same `step`).

File format (append-only, one file per rank):

    file   := MAGIC record*
    record := u32le length, zlib(payload)
    payload (meta, kind 0)  := 0x00, JSON object
    payload (data, kind 1)  := <BqdB: 1, step, wall time, itemsize>, body
    body   := the counts [layers, logical experts] of the interval as
              unsigned little-endian integers of `itemsize` bytes (1, 2 or
              4: the smallest that holds the largest count), byte-plane
              shuffled (all the first bytes, then all the second bytes, ...)

The meta record is the first record and has `version`, `layers` (the layer
names, rows of the counts), `num_logical_experts`, `interval`, `rank`.
`step` is the number of decoded engine steps traced so far when the record's
interval ended; `wall` is `time.time()` at that moment.
"""

import json
import os
import struct
import time
import zlib
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY

logger = init_logger(__name__)

TRACE_MAGIC = b"EPLBTRC1"
TRACE_VERSION = 1
KIND_META = 0
KIND_DATA = 1
_DATA_HEAD = struct.Struct("<BqdB")
_LENGTH = struct.Struct("<I")


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


class ExpertLoadTracer:
    """Sums the per-step routed load and appends it, every `interval` steps,
    to `<path>.rank<rank>`. Held by `EplbState`; see the module docstring."""

    def __init__(self, path: str, interval: int, rank: int) -> None:
        self.path = f"{path}.rank{rank}"
        self.interval = interval
        self.rank = rank
        self.steps = 0
        self._acc: list[torch.Tensor] = []
        self._pending: deque[tuple[int, float, Any, torch.Tensor]] = deque()
        self._failed = False
        self._meta: dict[str, Any] | None = None

    def accumulate(self, model_states: Sequence[Any]) -> None:
        """Add the load of this step; call before `expert_load_pass` is reset.
        Device work only."""
        if self._failed:
            return
        for i, model_state in enumerate(model_states):
            load = model_state.expert_load_pass
            if i >= len(self._acc):
                self._acc.append(torch.zeros_like(load))
            elif self._acc[i].shape != load.shape:
                self._acc[i] = torch.zeros_like(load)
            self._acc[i] += load

    def end_step(self, model_states: Sequence[Any]) -> None:
        """Count a traced step; at the end of an interval queue its copy to
        the host; write the intervals whose copies have completed."""
        if self._failed:
            return
        self._write_ready()
        self.steps += 1
        if self.steps % self.interval == 0:
            self._queue_interval(model_states)

    def _queue_interval(self, model_states: Sequence[Any]) -> None:
        logical_loads = []
        for model_state, acc in zip(model_states, self._acc):
            num_logical = model_state.model.num_logical_experts
            p2l = model_state.physical_to_logical_map
            logical = torch.zeros(
                (acc.shape[0], num_logical + 1), dtype=acc.dtype, device=acc.device
            )
            logical.scatter_add_(
                dim=-1,
                index=p2l.masked_fill(p2l < 0, num_logical).long(),
                src=acc,
            )
            acc.zero_()
            logical_loads.append(logical[:, :-1])
        device_loads = torch.cat(logical_loads, dim=0)
        on_device = device_loads.device.type != "cpu"
        host = torch.empty(
            device_loads.shape,
            dtype=device_loads.dtype,
            pin_memory=PIN_MEMORY and on_device,
        )
        host.copy_(device_loads, non_blocking=True)
        event = None
        if on_device:
            event = torch.Event()
            event.record()
        if self._meta is None:
            self._meta = {
                "version": TRACE_VERSION,
                "layers": [
                    layer.layer_name
                    for model_state in model_states
                    for layer in model_state.model.moe_layers
                ],
                "num_logical_experts": int(device_loads.shape[1]),
                "interval": self.interval,
                "rank": self.rank,
            }
        self._pending.append((self.steps, time.time(), event, host))

    def _write_ready(self) -> None:
        while self._pending:
            step, wall, event, host = self._pending[0]
            if event is not None and not event.query():
                return
            self._pending.popleft()
            self._append(encode_data(step, wall, host.numpy()))

    def drain(self) -> None:
        """Write every queued interval, waiting for its copy. For tests and
        shutdown; the serving path never calls it."""
        for _, _, event, _ in self._pending:
            if event is not None:
                event.synchronize()
        self._write_ready()

    def _append(self, record: bytes) -> None:
        try:
            fresh = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
            with open(self.path, "ab") as handle:
                if fresh:
                    assert self._meta is not None
                    handle.write(TRACE_MAGIC + encode_meta(self._meta))
                handle.write(record)
        except OSError as err:
            logger.warning("EPLB: cannot write the expert trace %s: %s", self.path, err)
            self._failed = True
            self._pending.clear()

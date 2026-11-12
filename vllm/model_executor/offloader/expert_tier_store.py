# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert store of one engine process, as the expert tiers use it.

:class:`StoreSession` decides at load whether the store is built or only read
(from its manifest and an identity string over the checkpoint, the kernel
layout and the schema), writes the units a rank owns while its layers come up,
finalizes the store once every layer on every rank has written, and then
opens the reader the layers' host rows are filled from and the rearrangements
read through.
"""

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from vllm.config.expert_store import ExpertStoreConfig
from vllm.logger import init_logger
from vllm.model_executor.offloader.expert_store import (
    ExpertStore,
    ExpertStoreWriter,
    StoreSchema,
    TensorSpec,
    _Layout,
)

logger = init_logger(__name__)

# Changes whenever the bytes of a unit would change for the same checkpoint
# (the layout transform of the kernels, the order of the tensors).
LAYOUT_VERSION = "expert-tiers-int4-sm4/1"

_SUFFIXES = (".safetensors", ".json", ".bin", ".pt", ".model")


def checkpoint_identity(model_config: Any) -> str:
    """A string that changes when the checkpoint's bytes do: the model path
    or id, revision and quantization, and for a local directory the name,
    size and modification time of each weight file."""
    parts: list[Any] = [
        str(getattr(model_config, "model", "")),
        str(getattr(model_config, "revision", None)),
        str(getattr(model_config, "quantization", None)),
        str(getattr(model_config, "dtype", None)),
    ]
    path = parts[0]
    if os.path.isdir(path):
        files = []
        for name in sorted(os.listdir(path)):
            if name.endswith(_SUFFIXES):
                st = os.stat(os.path.join(path, name))
                files.append((name, st.st_size, st.st_mtime_ns))
        parts.append(files)
    return json.dumps(parts, sort_keys=True)


def build_schema(
    num_layers: int, num_experts: int, tensor_bytes: Mapping[str, int]
) -> StoreSchema:
    """One schema for every layer. Every tensor is stored as a flat run of
    bytes (``uint8``), so a unit is the kernel's bytes whatever their dtype;
    ``tensor_bytes`` gives each tensor's size, in the order of the unit."""
    specs = [TensorSpec(name, "uint8", (n,)) for name, n in tensor_bytes.items()]
    return StoreSchema.uniform(num_layers, num_experts, specs)


def store_identity(schema: StoreSchema, checkpoint: str) -> str:
    """The identity string of a store: checkpoint, layout version, schema."""
    body = {
        "layout": LAYOUT_VERSION,
        "checkpoint": checkpoint,
        "layers": schema.layers,
        "experts": schema.experts_per_layer,
        "align": schema.tensor_align,
        "tensors": [
            [t.name, t.dtype, list(t.shape)]
            for group in schema.groups
            for t in group.tensors
        ],
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return f"sha256:{digest}"


class StoreSession:
    """Build or fill, write, finalize, open: for one process (one rank)."""

    def __init__(
        self,
        config: ExpertStoreConfig,
        schema: StoreSchema,
        identity: str,
        *,
        rank: int,
        num_ranks: int,
        barrier: Callable[[], None],
        pin: bool = True,
    ):
        self.config = config
        self.schema = schema
        self.identity = identity
        self.rank = rank
        self.num_ranks = num_ranks
        self._barrier = barrier
        self.pin = pin
        status = ExpertStore.status(config.dirs, identity)
        self.status = status
        if config.build == "rebuild":
            self.mode = "build"
        elif status == "complete":
            self.mode = "fill"
        elif config.build == "never":
            raise ValueError(
                f"--expert-store-config build=never, but the store in "
                f"{config.dirs} is {status} for this checkpoint and layout"
            )
        else:
            self.mode = "build"
        self._wipe = self.mode == "build" and (
            config.build == "rebuild" or status in ("complete", "mismatch")
        )
        self.registered: set[str] = set()
        self.built: set[str] = set()
        self._writer: ExpertStoreWriter | None = None
        self._store: ExpertStore | None = None
        self._stats_stop: threading.Event | None = None
        self._lock = threading.RLock()
        self._t_start = time.perf_counter()
        self.build_seconds = 0.0
        self.units_written = 0
        self.pool_layers = 0
        self.pool_bytes_pinned = 0
        self.pool_rows = 0
        self.pool_slow_slots = 0

    # -- registration -----------------------------------------------------

    def register(self, layer_name: str) -> None:
        self.registered.add(layer_name)

    @property
    def slot_bytes(self) -> int:
        """Bytes of one unit as stored (padded)."""
        return _Layout(self.schema, len(self.config.dirs), self.identity).slot_bytes

    def note_pool(self, rows: int, slot_bytes: int, slow_slots: int) -> None:
        """A layer's pinned pool rows, for the one summary line."""
        with self._lock:
            self.pool_layers += 1
            self.pool_rows = rows
            self.pool_slow_slots = slow_slots
            self.pool_bytes_pinned += rows * slot_bytes

    # -- reader -----------------------------------------------------------

    @property
    def store(self) -> ExpertStore:
        """The reader; in build mode it exists once every layer is written."""
        if self._store is None:
            raise RuntimeError("the expert store is not open yet")
        return self._store

    @property
    def is_open(self) -> bool:
        return self._store is not None

    def open_reader(self) -> ExpertStore:
        with self._lock:
            if self._store is not None:
                return self._store
            cfg = self.config
            slot_bytes = self.slot_bytes
            pool_bytes = max(
                int(cfg.pool_gib * 2**30) // max(self.num_ranks, 1), 4 * slot_bytes
            )
            t0 = time.perf_counter()
            self._store = ExpertStore.open(
                cfg.dirs,
                self.identity,
                pool_bytes,
                io_threads=cfg.io_threads,
                pin=self.pin,
                direct=cfg.direct,
                verify=cfg.verify,
                decay_interval=cfg.decay_interval,
                read_mode=cfg.read_mode,
            )
            layout = self._store.layout
            self._start_stats_thread()
            logger.info(
                "Expert store opened (%s): %d units of %d B, %.2f GiB in %d "
                "stripe(s) %s, staging pool %d slots (%.2f GiB), read mode %s, "
                "%.2f s.",
                self.mode,
                layout.total_units,
                layout.slot_bytes,
                sum(layout.stripe_bytes) / 2**30,
                layout.num_stripes,
                cfg.dirs,
                self._store.num_slots,
                self._store.num_slots * slot_bytes / 2**30,
                cfg.read_mode,
                time.perf_counter() - t0,
            )
            return self._store

    def _start_stats_thread(self) -> None:
        interval = self.config.stats_interval_s
        if interval <= 0 or self._stats_stop is not None:
            return
        stop = self._stats_stop = threading.Event()
        store = self._store

        def loop() -> None:
            while not stop.wait(interval):
                st = store.stats()  # type: ignore[union-attr]
                logger.info(
                    "Expert store (rank %d): %d reads (into rows: %d direct, %d "
                    "buffered, %d bounce), %.0f MiB, read p50 %.2f ms p95 %.2f "
                    "ms, staging hits %d misses %d evictions %d.",
                    self.rank,
                    st["reads"],
                    st["into_reads"]["direct"],
                    st["into_reads"]["buffered"],
                    st["into_reads"]["bounce"],
                    st["bytes_read"] / 2**20,
                    st["read_ms"]["p50"],
                    st["read_ms"]["p95"],
                    st["hits"],
                    st["misses"],
                    st["evictions"],
                )

        threading.Thread(target=loop, name="expert-store-stats", daemon=True).start()

    # -- writing ----------------------------------------------------------

    def _ensure_writer(self) -> ExpertStoreWriter:
        if self._writer is not None:
            return self._writer
        cfg = self.config
        writer_id = f"rank{self.rank}"
        # Rank 0 opens first, clearing an old store when asked to; the others
        # join the new one. Every rank passes the barrier, whatever it found
        # in the manifests.
        if self.rank == 0:
            self._writer = ExpertStoreWriter(
                cfg.dirs, self.schema, self.identity,
                writer_id=writer_id, rebuild=self._wipe,
            )  # fmt: skip
        self._barrier()
        if self._writer is None:
            self._writer = ExpertStoreWriter(
                cfg.dirs, self.schema, self.identity, writer_id=writer_id
            )
        logger.info(
            "Expert store: building (%s; found %s) %d layers x %d experts, "
            "%d B per unit, in %s.",
            "from scratch" if self._wipe else "starting or resuming",
            self.status,
            self.schema.layers,
            self.schema.experts_per_layer,
            self.slot_bytes,
            cfg.dirs,
        )
        return self._writer

    def write_units(
        self, store_layer: int, units: Iterable[tuple[int, Mapping[str, Any]]]
    ) -> None:
        """Write units (logical expert, tensors by name) of a layer."""
        assert self.mode == "build"
        t0 = time.perf_counter()
        writer = self._ensure_writer()
        n = 0
        for logical, tensors in units:
            writer.write(store_layer, logical, tensors)
            n += 1
        self.units_written += n
        self.build_seconds += time.perf_counter() - t0

    def layer_done(self, layer_name: str) -> None:
        """A layer's units are written (build) or its rows are filled (fill).
        The last layer ends the build: finalize across ranks, open the reader."""
        with self._lock:
            self.built.add(layer_name)
            if self.built != self.registered:
                return
            logger.info(
                "Expert store tiers: %d layers, %d pool rows each (%d host "
                "slots + %d spare), %.2f GiB pinned per rank for the rows.",
                self.pool_layers,
                self.pool_rows,
                self.pool_slow_slots,
                self.pool_rows - self.pool_slow_slots,
                self.pool_bytes_pinned / 2**30,
            )
            if self.mode == "build":
                self._finish_build()
            elif self._store is None:
                self.open_reader()

    def _finish_build(self) -> None:
        writer = self._ensure_writer()
        t0 = time.perf_counter()
        writer.close()
        self._barrier()
        if self.rank == 0:
            writer.finalize(checksum=True)
        self._barrier()
        self.build_seconds += time.perf_counter() - t0
        logger.info(
            "Expert store built: rank %d wrote %d units in %.1f s (including "
            "the checksum pass on rank 0).",
            self.rank,
            self.units_written,
            self.build_seconds,
        )
        self.open_reader()

    def close(self) -> None:
        with self._lock:
            if self._stats_stop is not None:
                self._stats_stop.set()
                self._stats_stop = None
            if self._store is not None:
                self._store.close()
                self._store = None
            if self._writer is not None:
                self._writer.close()
                self._writer = None


_SESSIONS: dict[tuple[str, tuple[str, ...]], StoreSession] = {}
_SESSIONS_LOCK = threading.Lock()


def get_session(
    key: tuple[str, tuple[str, ...]], make: Callable[[], StoreSession]
) -> StoreSession:
    """The process's session for a store, created by ``make`` on first use."""
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            session = _SESSIONS[key] = make()
        return session


def reset_sessions() -> None:
    """Close and forget every session (tests, process teardown)."""
    with _SESSIONS_LOCK:
        for session in _SESSIONS.values():
            session.close()
        _SESSIONS.clear()

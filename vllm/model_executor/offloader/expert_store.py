# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experts on NVMe behind a bounded pinned host pool.

Weights never change, so the store is read-only once built. Every expert of
every MoE layer is one *unit*: its tensors concatenated in a fixed order, padded
to a multiple of 4096 bytes. A unit's place in a stripe file follows from
(layer, expert) and the manifest alone. A store is one directory per drive, each
holding one stripe file (``experts.bin``) and the same ``manifest.json``.

A read always goes drive -> pool slot (pinned host memory, O_DIRECT where the
filesystem takes it) and a consumer then copies, or reads through UVA, from the
slot. The module holds no device code: the pool is host memory, and the
tensors it hands out are views of it.

Writing: :class:`ExpertStoreWriter`. Several processes may write disjoint sets
of units; ``finalize`` marks the store complete once every unit is recorded.
Reading: :class:`ExpertStore`.
"""

import contextlib
import errno
import hashlib
import heapq
import json
import math
import mmap
import os
import secrets
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

FORMAT = "expert-store/v1"
PAGE = 4096
DATA_OFFSET = PAGE
DATA_FILE = "experts.bin"
MANIFEST_FILE = "manifest.json"
WRITERS_DIR = "writers"
SIDECAR_SUFFIX = ".sha256"
PART_SUFFIX = ".part"
# Pinned chunks of this size are the largest the pinned host allocator takes
# without rounding (it rounds a request up to a power of two) and stay below the
# size XPU refuses to pin in one allocation.
CHUNK_BYTES = 1 << 30
# Retry backoff for a failed or short read, seconds (then raise).
RETRY_BACKOFF_S = (0.001, 0.010, 0.100)
DEFAULT_TENSOR_ALIGN = 64

_ITEMSIZE = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}

Key = tuple[int, int]


class ExpertStoreError(Exception):
    """Base class of every error this module raises about a store."""


class StoreIncompleteError(ExpertStoreError):
    """The store is not complete (units missing, or never finalized)."""


class StoreMismatchError(ExpertStoreError):
    """The store is not the one asked for (identity, layout, stripes)."""


class StoreCorruptError(ExpertStoreError):
    """A file does not match its recorded size or checksum."""


class StoreClosedError(ExpertStoreError):
    """The store was closed while the call was pending."""


class EntryEvictedError(ExpertStoreError):
    """A pool entry was evicted before it was leased."""


def _round_up(value: int, unit: int) -> int:
    return -(-value // unit) * unit


# --------------------------------------------------------------------------
# Schema and layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """One tensor of a unit: name, dtype name (``"int8"``, ``"bfloat16"``) and shape."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        if self.dtype not in _ITEMSIZE:
            raise ValueError(f"unsupported dtype {self.dtype!r} for {self.name!r}")
        return math.prod(self.shape) * _ITEMSIZE[self.dtype]


@dataclass(frozen=True)
class LayerGroup:
    """Consecutive layers whose units share one tensor schema (hence one size)."""

    first_layer: int
    num_layers: int
    tensors: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class StoreSchema:
    """What a store holds: layers, experts per layer and the unit schema."""

    layers: int
    experts_per_layer: int
    groups: tuple[LayerGroup, ...]
    tensor_align: int = DEFAULT_TENSOR_ALIGN

    @classmethod
    def uniform(
        cls,
        layers: int,
        experts_per_layer: int,
        tensors: Sequence[TensorSpec],
        tensor_align: int = DEFAULT_TENSOR_ALIGN,
    ) -> "StoreSchema":
        """One schema for every layer."""
        group = LayerGroup(0, layers, tuple(tensors))
        return cls(layers, experts_per_layer, (group,), tensor_align)


@dataclass(frozen=True)
class _TensorPlace:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class _GroupLayout:
    first_layer: int
    num_layers: int
    payload_bytes: int
    unit_bytes: int
    tensors: tuple[_TensorPlace, ...]


class _Layout:
    """Everything derived from a schema, a stripe count and an identity."""

    def __init__(
        self,
        schema: StoreSchema,
        num_stripes: int,
        identity: str,
        stripe_fn: Mapping[str, Any] | None = None,
    ):
        if num_stripes < 1:
            raise ValueError("num_stripes must be >= 1")
        if schema.layers < 1 or schema.experts_per_layer < 1:
            raise ValueError("layers and experts_per_layer must be >= 1")
        if schema.tensor_align < 1 or schema.tensor_align & (schema.tensor_align - 1):
            raise ValueError("tensor_align must be a power of two")
        self.schema = schema
        self.num_stripes = num_stripes
        self.identity = identity
        self.stripe_fn = dict(stripe_fn or {"kind": "modulo"})
        if self.stripe_fn.get("kind") != "modulo":
            raise StoreMismatchError(
                f"unsupported stripe function {self.stripe_fn!r}; only 'modulo' is "
                "implemented (an explicit assignment table is reserved)"
            )
        self.layers = schema.layers
        self.experts = schema.experts_per_layer
        self.total_units = self.layers * self.experts
        groups: list[_GroupLayout] = []
        expect = 0
        for group in schema.groups:
            if group.first_layer != expect or group.num_layers < 1:
                raise ValueError(
                    "layer groups must be consecutive and cover all layers"
                )
            expect += group.num_layers
            names = [t.name for t in group.tensors]
            if not names or len(set(names)) != len(names):
                raise ValueError("a unit needs at least one tensor, with unique names")
            places = []
            offset = 0
            for spec in group.tensors:
                offset = _round_up(offset, schema.tensor_align)
                places.append(
                    _TensorPlace(
                        spec.name, spec.dtype, tuple(spec.shape), offset, spec.nbytes
                    )
                )
                offset += spec.nbytes
            groups.append(
                _GroupLayout(
                    group.first_layer,
                    group.num_layers,
                    offset,
                    _round_up(offset, PAGE),
                    tuple(places),
                )
            )
        if expect != schema.layers:
            raise ValueError("layer groups must cover all layers")
        self.groups = tuple(groups)
        self.slot_bytes = max(g.unit_bytes for g in self.groups)
        # Per group and stripe: first byte after the groups before it.
        self._base: list[list[int]] = []
        totals = [DATA_OFFSET] * num_stripes
        for g in self.groups:
            first = g.first_layer * self.experts
            last = first + g.num_layers * self.experts
            self._base.append(list(totals))
            for s in range(num_stripes):
                totals[s] += (
                    self._below(last, s) - self._below(first, s)
                ) * g.unit_bytes
        self.stripe_bytes = tuple(totals)
        self._group_of_layer = [0] * self.layers
        for gi, g in enumerate(self.groups):
            for layer in range(g.first_layer, g.first_layer + g.num_layers):
                self._group_of_layer[layer] = gi

    def _below(self, bound: int, stripe: int) -> int:
        """Units with a global index below ``bound`` that land on ``stripe``."""
        return max(0, (bound - stripe + self.num_stripes - 1) // self.num_stripes)

    def group_of(self, layer: int) -> _GroupLayout:
        return self.groups[self._group_of_layer[layer]]

    def unit_index(self, layer: int, expert: int) -> int:
        if not (0 <= layer < self.layers and 0 <= expert < self.experts):
            raise ValueError(
                f"unit ({layer}, {expert}) is outside the store "
                f"({self.layers} layers x {self.experts} experts)"
            )
        return layer * self.experts + expert

    def locate(self, layer: int, expert: int) -> tuple[int, int, int]:
        """(stripe, byte offset in the stripe file, unit bytes) of a unit."""
        index = self.unit_index(layer, expert)
        gi = self._group_of_layer[layer]
        g = self.groups[gi]
        stripe = index % self.num_stripes
        first = g.first_layer * self.experts
        local = self._below(index, stripe) - self._below(first, stripe)
        return stripe, self._base[gi][stripe] + local * g.unit_bytes, g.unit_bytes

    def manifest_core(self) -> dict[str, Any]:
        """The part of the manifest fixed by the layout (hashed, never changes)."""
        return {
            "format": FORMAT,
            "identity": self.identity,
            "layers": self.layers,
            "experts_per_layer": self.experts,
            "tensor_align": self.schema.tensor_align,
            "groups": [
                {
                    "first_layer": g.first_layer,
                    "num_layers": g.num_layers,
                    "payload_bytes": g.payload_bytes,
                    "unit_bytes": g.unit_bytes,
                    "tensors": [
                        {
                            "name": t.name,
                            "dtype": t.dtype,
                            "shape": list(t.shape),
                            "offset": t.offset,
                            "nbytes": t.nbytes,
                        }
                        for t in g.tensors
                    ],
                }
                for g in self.groups
            ],
            "num_stripes": self.num_stripes,
            "stripe_fn": self.stripe_fn,
            "data_offset": DATA_OFFSET,
            "stripe_bytes": list(self.stripe_bytes),
        }

    def layout_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.manifest_core())).hexdigest()

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "_Layout":
        try:
            if manifest["format"] != FORMAT:
                raise StoreMismatchError(
                    f"format {manifest['format']!r}, this reader takes {FORMAT!r}"
                )
            groups = tuple(
                LayerGroup(
                    int(g["first_layer"]),
                    int(g["num_layers"]),
                    tuple(
                        TensorSpec(t["name"], t["dtype"], tuple(t["shape"]))
                        for t in g["tensors"]
                    ),
                )
                for g in manifest["groups"]
            )
            schema = StoreSchema(
                int(manifest["layers"]),
                int(manifest["experts_per_layer"]),
                groups,
                int(manifest["tensor_align"]),
            )
            layout = cls(
                schema,
                int(manifest["num_stripes"]),
                str(manifest["identity"]),
                manifest["stripe_fn"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreCorruptError(f"manifest is malformed: {exc!r}") from exc
        core = layout.manifest_core()
        for key, value in core.items():
            if manifest.get(key) != value:
                raise StoreCorruptError(
                    f"manifest field {key!r} disagrees with the layout it implies"
                )
        return layout


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _header_page(stripe: int, layout: _Layout) -> bytes:
    body = _canonical(
        {
            "format": FORMAT,
            "stripe": stripe,
            "num_stripes": layout.num_stripes,
            "layout_sha256": layout.layout_sha256(),
        }
    )
    return body + b" " * (DATA_OFFSET - len(body))


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path: str, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    buf = bytearray(block)
    view = memoryview(buf)
    with open(path, "rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buf)
            if not count:
                break
            digest.update(view[:count])
    return digest.hexdigest()


def _read_manifest(directory: str) -> dict[str, Any] | None:
    path = os.path.join(directory, MANIFEST_FILE)
    try:
        with open(path, "rb") as handle:
            return json.loads(handle.read())
    except FileNotFoundError:
        return None
    except ValueError as exc:
        raise StoreCorruptError(f"{path} is not valid JSON") from exc


def _read_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        page = handle.read(PAGE).rstrip(b" \x00")
    if not page:
        return {}
    try:
        return json.loads(page)
    except ValueError as exc:
        raise StoreCorruptError(f"{path}: header is not valid JSON") from exc


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


def _as_bytes(value: Any) -> np.ndarray:
    """A tensor, array or bytes-like as a flat uint8 array (no copy if possible)."""
    if hasattr(value, "numpy") and hasattr(value, "is_cuda"):  # torch tensor
        if value.device.type != "cpu":
            raise ValueError("tensors given to the writer must be on the CPU")
        import torch

        flat = value.detach().contiguous().reshape(-1).view(torch.uint8)
        return flat.numpy()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).reshape(-1).view(np.uint8)
    return np.frombuffer(value, dtype=np.uint8)


def _dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


class ExpertStoreWriter:
    """Writes units into a store; several processes may write disjoint units.

    ``dirs`` are the drives' directories in stripe order (the same order for
    every writer). Opening an incomplete store of the same layout and identity
    resumes it; any other existing store is refused unless ``rebuild`` is set
    (one coordinating writer should rebuild before the others open).

    Each writer keeps a record of the units it wrote in ``writers/<id>.json``
    of the first directory, saved by :meth:`flush` and :meth:`close`.
    :meth:`finalize` (any writer, after the others closed) checks that the
    records cover every unit exactly once, then publishes the stripes, the
    checksum sidecars and the completed manifest.
    """

    def __init__(
        self,
        dirs: Sequence[str],
        schema: StoreSchema,
        identity: str,
        *,
        writer_id: str | None = None,
        rebuild: bool = False,
    ):
        if not dirs:
            raise ValueError("at least one directory is needed")
        self.dirs = [os.fspath(d) for d in dirs]
        self.layout = _Layout(schema, len(self.dirs), identity)
        self.writer_id = writer_id or f"w{os.getpid()}-{secrets.token_hex(3)}"
        if not self.writer_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("writer_id may hold letters, digits, '-' and '_'")
        self._lock = threading.Lock()
        self._bitmap = bytearray(-(-self.layout.total_units // 8))
        self._count = 0
        self._closed = False
        self._fds: list[int] = []
        self._core = self.layout.manifest_core()
        self._core_sha = self.layout.layout_sha256()
        try:
            self._prepare_dirs(rebuild)
        except BaseException:
            self._close_fds()
            raise
        self._record_path = os.path.join(
            self.dirs[0], WRITERS_DIR, f"{self.writer_id}.json"
        )
        self._load_own_record()

    # -- opening -----------------------------------------------------------

    def _prepare_dirs(self, rebuild: bool) -> None:
        layout = self.layout
        for directory in self.dirs:
            os.makedirs(os.path.join(directory, WRITERS_DIR), exist_ok=True)
            manifest = _read_manifest(directory)
            if rebuild:
                self._wipe(directory)
            elif manifest is not None:
                if manifest.get("layout_sha256") != self._core_sha:
                    raise StoreMismatchError(
                        f"{directory} holds a store with another layout or "
                        f"identity ({manifest.get('identity')!r}); pass rebuild=True"
                    )
                if manifest.get("complete"):
                    raise StoreMismatchError(
                        f"{directory} already holds a complete store; "
                        "pass rebuild=True to replace it"
                    )
        for index, directory in enumerate(self.dirs):
            path = os.path.join(directory, DATA_FILE + PART_SUFFIX)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            self._fds.append(fd)
            size = layout.stripe_bytes[index]
            if os.fstat(fd).st_size != size:
                os.ftruncate(fd, size)
            header = _read_header(path) if os.fstat(fd).st_size >= PAGE else {}
            if header.get("layout_sha256") not in (None, self._core_sha) or (
                header.get("stripe") not in (None, index)
            ):
                self._close_fds()
                raise StoreMismatchError(
                    f"{path} belongs to another layout or stripe position; "
                    "directories must be given in stripe order, or pass rebuild=True"
                )
            os.pwrite(fd, _header_page(index, layout), 0)
            manifest_doc = dict(self._core)
            manifest_doc.update(
                layout_sha256=self._core_sha, complete=False, writers=[]
            )
            if _read_manifest(directory) is None:
                _write_atomic(
                    os.path.join(directory, MANIFEST_FILE),
                    json.dumps(manifest_doc, indent=1, sort_keys=True).encode(),
                )

    @staticmethod
    def _wipe(directory: str) -> None:
        for name in (
            MANIFEST_FILE,
            DATA_FILE,
            DATA_FILE + PART_SUFFIX,
            DATA_FILE + SIDECAR_SUFFIX,
        ):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(os.path.join(directory, name))
        shutil.rmtree(os.path.join(directory, WRITERS_DIR), ignore_errors=True)
        os.makedirs(os.path.join(directory, WRITERS_DIR), exist_ok=True)

    def _load_own_record(self) -> None:
        try:
            with open(self._record_path, "rb") as handle:
                record = json.loads(handle.read())
        except FileNotFoundError:
            return
        if record.get("layout_sha256") != self._core_sha:
            raise StoreMismatchError(f"{self._record_path} is from another layout")
        for start, end in record["runs"]:
            for index in range(start, end):
                self._set_bit(index)

    def _close_fds(self) -> None:
        for fd in self._fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        self._fds = []

    # -- writing -----------------------------------------------------------

    def _set_bit(self, index: int) -> None:
        mask = 1 << (index & 7)
        if not self._bitmap[index >> 3] & mask:
            self._bitmap[index >> 3] |= mask
            self._count += 1

    def write(self, layer: int, expert: int, tensors: Mapping[str, Any]) -> None:
        """Write one unit; ``tensors`` maps every schema name to its data.

        Thread-safe; writing a unit again overwrites it.
        """
        if self._closed:
            raise ExpertStoreError("writer is closed")
        layout = self.layout
        stripe, offset, unit_bytes = layout.locate(layer, expert)  # checks range
        group = layout.group_of(layer)
        if set(tensors) != {t.name for t in group.tensors}:
            raise ValueError(
                f"unit ({layer}, {expert}) needs tensors "
                f"{[t.name for t in group.tensors]}, got {sorted(tensors)}"
            )
        buf = np.zeros(unit_bytes, dtype=np.uint8)
        for place in group.tensors:
            value = tensors[place.name]
            name = _dtype_name(value)
            if name is not None and name != place.dtype:
                raise ValueError(
                    f"{place.name}: dtype {name}, schema says {place.dtype}"
                )
            shape = getattr(value, "shape", None)
            if shape is not None and tuple(shape) != place.shape:
                raise ValueError(
                    f"{place.name}: shape {tuple(shape)}, schema says {place.shape}"
                )
            raw = _as_bytes(value)
            if raw.size != place.nbytes:
                raise ValueError(
                    f"{place.name}: {raw.size} bytes, schema says {place.nbytes}"
                )
            buf[place.offset : place.offset + place.nbytes] = raw
        view = memoryview(buf)
        done = 0
        while done < unit_bytes:
            done += os.pwrite(self._fds[stripe], view[done:], offset + done)
        with self._lock:
            self._set_bit(layout.unit_index(layer, expert))

    @property
    def units_written(self) -> int:
        return self._count

    def _runs(self) -> list[list[int]]:
        runs: list[list[int]] = []
        start = None
        for index in range(self.layout.total_units + 1):
            on = index < self.layout.total_units and bool(
                self._bitmap[index >> 3] & (1 << (index & 7))
            )
            if on and start is None:
                start = index
            elif not on and start is not None:
                runs.append([start, index])
                start = None
        return runs

    def flush(self, *, done: bool = False) -> None:
        """Make what was written durable, then save this writer's record."""
        with self._lock:
            runs = self._runs()
            count = self._count
        for fd in self._fds:
            os.fdatasync(fd)
        record = {
            "writer": self.writer_id,
            "layout_sha256": self._core_sha,
            "units": count,
            "runs": runs,
            "done": done,
            "saved_unix": int(time.time()),
        }
        _write_atomic(self._record_path, json.dumps(record).encode())

    def close(self) -> None:
        """Flush, save the record as done, close the files."""
        if self._closed:
            return
        self.flush(done=True)
        self._closed = True
        self._close_fds()

    def __enter__(self) -> "ExpertStoreWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        if exc[0] is None:
            self.close()
        else:
            self._closed = True
            self._close_fds()

    # -- finalizing --------------------------------------------------------

    def _collect_records(self) -> tuple[bytearray, list[dict[str, Any]]]:
        total = self.layout.total_units
        union = bytearray(-(-total // 8))
        overlap = 0
        records = []
        folder = os.path.join(self.dirs[0], WRITERS_DIR)
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(folder, name), "rb") as handle:
                record = json.loads(handle.read())
            if record.get("layout_sha256") != self._core_sha:
                raise StoreMismatchError(f"writer record {name} is from another layout")
            for start, end in record["runs"]:
                for index in range(start, end):
                    mask = 1 << (index & 7)
                    if union[index >> 3] & mask:
                        overlap += 1
                    union[index >> 3] |= mask
            records.append(
                {
                    "id": record["writer"],
                    "units": record["units"],
                    "done": record["done"],
                }
            )
        if overlap:
            raise ExpertStoreError(
                f"{overlap} units are recorded by more than one writer; "
                "writers must hold disjoint sets"
            )
        return union, records

    def finalize(self, *, checksum: bool = True) -> None:
        """Mark the store complete if every unit is recorded, else raise.

        Call after the other writers closed. Publishes the stripes (rename of
        ``experts.bin.part``), writes the sha256 sidecars (a full read of every
        stripe; ``checksum=False`` skips them) and the completed manifest.
        """
        self.flush(done=self._closed)
        union, records = self._collect_records()
        total = self.layout.total_units
        missing = [i for i in range(total) if not union[i >> 3] & (1 << (i & 7))]
        if missing:
            first = [divmod(i, self.layout.experts) for i in missing[:5]]
            raise StoreIncompleteError(
                f"{len(missing)} of {total} units are not recorded by any writer "
                f"(first: layer/expert {first})"
            )
        if not self._closed:
            for fd in self._fds:
                os.fsync(fd)
        for directory in self.dirs:
            part = os.path.join(directory, DATA_FILE + PART_SUFFIX)
            final = os.path.join(directory, DATA_FILE)
            if os.path.exists(part):
                os.replace(part, final)
            elif not os.path.exists(final):
                raise StoreIncompleteError(f"{directory}: no stripe file to publish")
        sums: list[str | None] = [None] * len(self.dirs)
        if checksum:
            with ThreadPoolExecutor(len(self.dirs)) as pool:
                futures = [
                    pool.submit(_sha256_file, os.path.join(d, DATA_FILE))
                    for d in self.dirs
                ]
                sums = [f.result() for f in futures]
            for directory, digest in zip(self.dirs, sums):
                _write_atomic(
                    os.path.join(directory, DATA_FILE + SIDECAR_SUFFIX),
                    f"{digest}  {DATA_FILE}\n".encode(),
                )
        manifest = dict(self._core)
        manifest.update(
            layout_sha256=self._core_sha,
            complete=True,
            checksummed=checksum,
            writers=records,
            finalized_unix=int(time.time()),
        )
        payload = json.dumps(manifest, indent=1, sort_keys=True).encode()
        for directory in self.dirs:
            _write_atomic(os.path.join(directory, MANIFEST_FILE), payload)
            _fsync_dir(directory)


# --------------------------------------------------------------------------
# Eviction
# --------------------------------------------------------------------------


class EvictionPolicy(Protocol):
    """What the pool needs from a replacement policy (all calls under its lock)."""

    def tick(self) -> None:
        """One ``request`` call happened."""

    def record_use(self, key: Key) -> None:
        """``key`` was asked for."""

    def make_evictable(self, key: Key) -> None:
        """``key`` is resident with no lease: it may be chosen."""

    def make_pinned(self, key: Key) -> None:
        """``key`` is leased or loading: it may not be chosen."""

    def remove(self, key: Key) -> None:
        """``key`` left the pool."""

    def pop_victim(self) -> Key | None:
        """The key to evict (and forget as evictable), or None."""


class CountDecayPolicy:
    """Use counts that halve every ``decay_interval`` requests; lowest count goes.

    A count is kept for every key ever asked for, resident or not, so an expert
    that comes back keeps its history. Ties go to the least recently used.
    The decay is continuous (half-life ``decay_interval`` requests), and kept
    lazily: ``log2(count) + t / interval`` does not change between uses, so
    victims come out of a heap in O(log n).
    """

    def __init__(self, decay_interval: int = 256):
        if decay_interval < 1:
            raise ValueError("decay_interval must be >= 1")
        self.interval = float(decay_interval)
        self._now = 0
        self._seq = 0
        self._count: dict[Key, tuple[float, int]] = {}
        self._last: dict[Key, int] = {}
        self._evictable: dict[Key, tuple[float, int]] = {}
        self._heap: list[tuple[float, int, Key]] = []

    def tick(self) -> None:
        self._now += 1

    def _priority(self, key: Key) -> tuple[float, int]:
        entry = self._count.get(key)
        prio = (
            -math.inf
            if entry is None
            else math.log2(entry[0]) + entry[1] / self.interval
        )
        return prio, self._last.get(key, -1)

    def _push(self, key: Key) -> None:
        prio, last = self._priority(key)
        self._evictable[key] = (prio, last)
        heapq.heappush(self._heap, (prio, last, key))

    def record_use(self, key: Key) -> None:
        value, stamp = self._count.get(key, (0.0, self._now))
        value = value * 2.0 ** (-(self._now - stamp) / self.interval) + 1.0
        self._count[key] = (value, self._now)
        self._seq += 1
        self._last[key] = self._seq
        if key in self._evictable:
            self._push(key)

    def count(self, key: Key) -> float:
        """The decayed count of ``key`` now."""
        value, stamp = self._count.get(key, (0.0, self._now))
        return value * 2.0 ** (-(self._now - stamp) / self.interval)

    def make_evictable(self, key: Key) -> None:
        self._push(key)

    def make_pinned(self, key: Key) -> None:
        self._evictable.pop(key, None)

    def remove(self, key: Key) -> None:
        self._evictable.pop(key, None)

    def pop_victim(self) -> Key | None:
        while self._heap:
            prio, last, key = heapq.heappop(self._heap)
            if self._evictable.get(key) == (prio, last):
                del self._evictable[key]
                return key
        return None


# --------------------------------------------------------------------------
# Pool and reader
# --------------------------------------------------------------------------


class _Chunk:
    """One allocation of pool memory, page aligned, with its numpy view."""

    def __init__(self, nbytes: int, pin: bool):
        self.torch: Any = None
        self._owner: Any
        if pin:
            import torch

            self.torch = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            self._owner = self.torch
            self.array = self.torch.numpy()
        else:
            self._owner = mmap.mmap(-1, nbytes)
            self.array = np.frombuffer(self._owner, dtype=np.uint8)
        if self.array.ctypes.data % PAGE:
            raise ExpertStoreError(
                "pool memory is not page aligned, which direct reads need"
            )

    def torch_tensor(self) -> Any:
        if self.torch is None:
            import torch

            self.torch = torch.from_numpy(self.array)
        return self.torch


@dataclass
class _Slot:
    index: int
    chunk: _Chunk
    start: int
    state: str = "free"  # free | loading | ready
    key: Key | None = None
    leases: int = 0
    gen: int = 0


@dataclass
class _KeyState:
    state: str  # queued | loading | ready
    slot: int = -1
    waiters: list[tuple[Future, bool]] | None = None


class PoolEntry:
    """A unit in a pool slot: tensors as views of the slot, and a lease.

    A lease keeps the slot from being evicted. Use the views only while the
    entry is leased (``with entry:`` or ``acquire``/``release``); an unleased
    entry may be evicted at any moment, after which :meth:`acquire` raises
    :class:`EntryEvictedError`. Each handle holds at most one lease.
    """

    def __init__(self, store: "ExpertStore", key: Key, slot: _Slot, held: bool):
        self._store = store
        self._slot = slot
        self._gen = slot.gen
        self._held = held
        self.key = key
        self._group = store.layout.group_of(key[0])

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> None:
        """Take a lease; raises :class:`EntryEvictedError` if the slot moved on."""
        if self._held:
            raise ExpertStoreError("entry is already leased by this handle")
        self._store._acquire(self._slot, self._gen)
        self._held = True

    def release(self) -> None:
        """Give the lease back; the slot may then be evicted."""
        if not self._held:
            raise ExpertStoreError("entry is not leased by this handle")
        self._held = False
        self._store._release(self._slot)

    def __enter__(self) -> "PoolEntry":
        if not self._held:
            self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._held:
            self.release()

    def valid(self) -> bool:
        """Whether the slot still holds this unit (always true while leased)."""
        return self._slot.gen == self._gen and self._slot.state == "ready"

    @property
    def payload_bytes(self) -> int:
        return self._group.payload_bytes

    @property
    def raw(self) -> np.ndarray:
        """The unit's tensor bytes as a uint8 view of the slot (no padding)."""
        slot = self._slot
        return slot.chunk.array[slot.start : slot.start + self._group.payload_bytes]

    def slot_tensor(self) -> Any:
        """The whole slot as a uint8 torch tensor (pinned if the pool is)."""
        slot = self._slot
        size = self._store.layout.slot_bytes
        return slot.chunk.torch_tensor()[slot.start : slot.start + size]

    def tensors(self) -> dict[str, Any]:
        """Every tensor of the unit as a torch view of the slot (zero copy)."""
        import torch

        base = self.slot_tensor()
        out = {}
        for place in self._group.tensors:
            dtype = getattr(torch, place.dtype)
            flat = base[place.offset : place.offset + place.nbytes]
            out[place.name] = flat.view(dtype).view(place.shape)
        return out

    def tensor_places(self) -> tuple[_TensorPlace, ...]:
        """Name, dtype, shape, offset in the slot and bytes of each tensor."""
        return self._group.tensors


class _Stripe:
    def __init__(self, path: str, want_direct: bool):
        self.path = path
        self.lock = threading.Lock()
        self.retired: list[int] = []
        self.direct = False
        self.fd = -1
        if want_direct:
            try:
                self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                self.direct = True
            except OSError:
                self.fd = -1
        if self.fd < 0:
            self.fd = os.open(path, os.O_RDONLY)

    def fall_back(self, failed_fd: int) -> None:
        with self.lock:
            if self.fd == failed_fd and self.direct:
                self.retired.append(self.fd)
                self.fd = os.open(self.path, os.O_RDONLY)
                self.direct = False

    def close(self) -> None:
        for fd in [self.fd, *self.retired]:
            with contextlib.suppress(OSError):
                os.close(fd)


class ExpertStore:
    """A read-only expert store behind a bounded pinned pool.

    Build with :meth:`open`. ``request`` schedules reads and returns at once;
    ``get`` returns a (by default leased) :class:`PoolEntry`, blocking only if
    the read is not done. Thread-safe; ``close`` joins the I/O threads.
    """

    def __init__(
        self,
        stripes: list[_Stripe],
        layout: _Layout,
        pool_bytes: int,
        io_threads: int,
        pin: bool,
        policy: EvictionPolicy,
        preadv: Callable[..., int],
    ):
        self.layout = layout
        self._stripes = stripes
        self._policy = policy
        self._preadv = preadv
        slot_bytes = layout.slot_bytes
        slots = pool_bytes // slot_bytes
        if slots < 1:
            raise ValueError(
                f"pool_bytes={pool_bytes} is smaller than one slot ({slot_bytes} B)"
            )
        self.num_slots = slots
        per_chunk = max(1, CHUNK_BYTES // slot_bytes)
        self._slots: list[_Slot] = []
        while len(self._slots) < slots:
            n = min(per_chunk, slots - len(self._slots))
            chunk = _Chunk(n * slot_bytes, pin)
            for i in range(n):
                self._slots.append(_Slot(len(self._slots), chunk, i * slot_bytes))
        self.pinned = pin
        self._free = list(range(slots - 1, -1, -1))
        self._keys: dict[Key, _KeyState] = {}
        self._queue: deque[Key] = deque()
        self._lock = threading.RLock()
        self._closed = False
        self._executor = ThreadPoolExecutor(io_threads, thread_name_prefix="expert-io")
        # statistics
        self._hits = self._misses = self._joins = self._reads = 0
        self._bytes_read = self._evictions = self._errors = 0
        self._reads_direct = self._reads_buffered = 0
        self._waits = 0
        self._wait_s = 0.0
        self._latencies: deque[float] = deque(maxlen=65536)
        self._latency_max = 0.0

    @classmethod
    def open(
        cls,
        dirs: Sequence[str],
        identity: str,
        pool_bytes: int,
        io_threads: int = 16,
        pin: bool = True,
        *,
        direct: bool = True,
        verify: bool = False,
        decay_interval: int = 256,
        policy: EvictionPolicy | None = None,
        preadv: Callable[..., int] = os.preadv,
    ) -> "ExpertStore":
        """Open a complete store whose identity is ``identity``.

        ``dirs`` may be in any order. Refuses, with the reason, a store that is
        incomplete, of another identity or layout, missing a stripe, or with a
        stripe of the wrong size; ``verify`` also hashes every stripe against
        its sidecar (a full read).
        """
        dirs = [os.fspath(d) for d in dirs]
        if not dirs:
            raise ValueError("at least one directory is needed")
        manifests = []
        for directory in dirs:
            manifest = _read_manifest(directory)
            if manifest is None:
                raise StoreIncompleteError(
                    f"{directory}: no {MANIFEST_FILE} (store never started)"
                )
            manifests.append(manifest)
        first = manifests[0]
        if not first.get("complete"):
            n_records = 0
            with contextlib.suppress(OSError):
                n_records = len(os.listdir(os.path.join(dirs[0], WRITERS_DIR)))
            raise StoreIncompleteError(
                f"{dirs[0]}: store is not complete (not finalized; "
                f"{n_records} writer records); finish it with ExpertStoreWriter "
                "or rebuild it"
            )
        if first.get("identity") != identity:
            raise StoreMismatchError(
                f"store identity {first.get('identity')!r} is not {identity!r}"
            )
        layout = _Layout.from_manifest(first)
        if first.get("layout_sha256") != layout.layout_sha256():
            raise StoreCorruptError("manifest layout hash does not match its content")
        if len(dirs) != layout.num_stripes:
            raise StoreMismatchError(
                f"store has {layout.num_stripes} stripes, {len(dirs)} directories given"
            )
        ordered: list[str | None] = [None] * layout.num_stripes
        for directory, manifest in zip(dirs, manifests):
            if manifest.get("layout_sha256") != first[
                "layout_sha256"
            ] or not manifest.get("complete"):
                raise StoreMismatchError(
                    f"{directory}: manifest differs from {dirs[0]}"
                )
            path = os.path.join(directory, DATA_FILE)
            try:
                size = os.stat(path).st_size
            except FileNotFoundError as exc:
                raise StoreIncompleteError(f"{path} is missing") from exc
            header = _read_header(path)
            index = header.get("stripe")
            if (
                header.get("format") != FORMAT
                or header.get("layout_sha256") != first["layout_sha256"]
                or not isinstance(index, int)
                or not 0 <= index < layout.num_stripes
            ):
                raise StoreMismatchError(f"{path}: header does not match the manifest")
            if ordered[index] is not None:
                raise StoreMismatchError(f"stripe {index} given twice")
            if size != layout.stripe_bytes[index]:
                raise StoreCorruptError(
                    f"{path}: {size} B on disk, manifest says "
                    f"{layout.stripe_bytes[index]} B (truncated or extended)"
                )
            ordered[index] = directory
        if verify:
            cls._verify_dirs([d for d in ordered if d is not None], first)
        stripes: list[_Stripe] = []
        try:
            for stripe_dir in ordered:
                assert stripe_dir is not None
                stripes.append(_Stripe(os.path.join(stripe_dir, DATA_FILE), direct))
            return cls(
                stripes,
                layout,
                pool_bytes,
                io_threads,
                pin,
                policy or CountDecayPolicy(decay_interval),
                preadv,
            )
        except BaseException:
            for stripe in stripes:
                stripe.close()
            raise

    @staticmethod
    def _verify_dirs(ordered: list[str], manifest: Mapping[str, Any]) -> None:
        if not manifest.get("checksummed"):
            raise StoreIncompleteError("store was finalized without checksums")

        def check(directory: str) -> None:
            path = os.path.join(directory, DATA_FILE)
            with open(path + SIDECAR_SUFFIX, "rb") as handle:
                want = handle.read().split()[0].decode()
            if _sha256_file(path) != want:
                raise StoreCorruptError(f"{path}: sha256 does not match its sidecar")

        with ThreadPoolExecutor(len(ordered)) as pool:
            for future in [pool.submit(check, d) for d in ordered]:
                future.result()

    # -- requests ----------------------------------------------------------

    def request(self, keys: Iterable[Key], lease: bool = False) -> list[Future]:
        """Schedule reads for the units not in the pool; return at once.

        One future per key, in order, resolving to a :class:`PoolEntry` (leased
        if ``lease``; the caller then releases it). A unit already loading is
        joined, a unit already resident is a hit. Keys that find no free or
        evictable slot wait and are read as slots come free. All reads of one
        call are issued together, interleaved over the stripes.
        """
        keys = [(int(layer), int(expert)) for layer, expert in keys]
        for layer, expert in keys:
            self.layout.unit_index(layer, expert)
        futures: list[Future] = []
        with self._lock:
            if self._closed:
                raise StoreClosedError("store is closed")
            self._policy.tick()
            for key in keys:
                self._policy.record_use(key)
                futures.append(self._attach(key, lease, count=True))
            batch = self._pump_locked()
        self._submit(batch)
        return futures

    def get(
        self, key: Key, timeout: float | None = None, lease: bool = True
    ) -> PoolEntry:
        """The unit's entry, leased by default; blocks only if its read is pending.

        A key never asked for is requested here (and counted as a use); a key
        already requested, loading or resident, is only claimed, so
        ``request`` then ``get`` is one use. Raises ``TimeoutError`` after
        ``timeout`` seconds (the read goes on; a lease it would have taken is
        given back).
        """
        key = (int(key[0]), int(key[1]))
        self.layout.unit_index(*key)
        batch: list[tuple[Key, int, int]] = []
        with self._lock:
            if self._closed:
                raise StoreClosedError("store is closed")
            if key in self._keys:
                future = self._attach(key, lease, count=False)
            else:
                self._policy.tick()
                self._policy.record_use(key)
                future = self._attach(key, lease, count=True)
                batch = self._pump_locked()
        self._submit(batch)
        if not future.done():
            start = time.perf_counter()
            try:
                return future.result(timeout)
            except FuturesTimeout:
                # The read goes on. Do not leave a lease nobody will release.
                if not future.cancel() and lease:
                    future.add_done_callback(self._drop_late_lease)
                raise TimeoutError(f"unit {key} not ready after {timeout} s") from None
            finally:
                with self._lock:
                    self._waits += 1
                    self._wait_s += time.perf_counter() - start
        return future.result()

    @staticmethod
    def _drop_late_lease(future: Future) -> None:
        if not future.cancelled() and future.exception() is None:
            entry = future.result()
            if entry.held:
                entry.release()

    def _attach(self, key: Key, lease: bool, count: bool) -> Future:
        """A future for ``key``, creating its state if needed (lock held)."""
        future: Future = Future()
        state = self._keys.get(key)
        if state is None:
            if count:
                self._misses += 1
            state = _KeyState("queued", waiters=[])
            self._keys[key] = state
            self._queue.append(key)
        elif state.state == "ready":
            if count:
                self._hits += 1
            self._resolve(key, state, future, lease)
            return future
        elif count:
            self._joins += 1
        assert state.waiters is not None
        state.waiters.append((future, lease))
        return future

    def _resolve(self, key: Key, state: _KeyState, future: Future, lease: bool) -> None:
        """Hand a ready slot to one waiter (lock held)."""
        slot = self._slots[state.slot]
        entry = PoolEntry(self, key, slot, held=lease)
        if lease:
            slot.leases += 1
            self._policy.make_pinned(key)
        try:
            future.set_result(entry)
        except InvalidStateError:  # cancelled by the caller
            if lease:  # no pump here: the caller of _resolve is mid-update
                entry._held = False
                slot.leases -= 1
                if slot.leases == 0:
                    self._policy.make_evictable(key)

    def _pump_locked(self) -> list[tuple[Key, int, int]]:
        """Give queued keys slots (free, else evicted); lock held."""
        batch: list[tuple[Key, int, int]] = []
        while self._queue:
            index = self._take_slot_locked()
            if index is None:
                break
            key = self._queue.popleft()
            slot = self._slots[index]
            slot.state = "loading"
            slot.key = key
            slot.leases = 0
            slot.gen += 1
            state = self._keys[key]
            state.state = "loading"
            state.slot = index
            self._policy.make_pinned(key)
            batch.append((key, index, slot.gen))
        return batch

    def _take_slot_locked(self) -> int | None:
        if self._free:
            return self._free.pop()
        victim = self._policy.pop_victim()
        if victim is None:
            return None
        state = self._keys.pop(victim)
        slot = self._slots[state.slot]
        slot.state = "free"
        slot.key = None
        slot.gen += 1
        self._policy.remove(victim)
        self._evictions += 1
        return slot.index

    def _submit(self, batch: list[tuple[Key, int, int]]) -> None:
        if not batch:
            return
        # Interleave over the stripes so that every drive has work at once.
        lanes: dict[int, list[tuple[int, Key, int, int]]] = {}
        for key, index, gen in batch:
            stripe, offset, _ = self.layout.locate(*key)
            lanes.setdefault(stripe, []).append((offset, key, index, gen))
        for lane in lanes.values():
            lane.sort()
        order = []
        depth = max(len(lane) for lane in lanes.values())
        for i in range(depth):
            for stripe in sorted(lanes):
                if i < len(lanes[stripe]):
                    order.append(lanes[stripe][i])
        for _, key, index, gen in order:
            try:
                self._executor.submit(self._load, key, index, gen)
            except RuntimeError:  # executor shut down by close()
                self._fail(key, index, StoreClosedError("store is closed"))

    # -- loading -----------------------------------------------------------

    def _load(self, key: Key, index: int, gen: int) -> None:
        slot = self._slots[index]
        try:
            stripe_no, offset, size = self.layout.locate(*key)
            start = time.perf_counter()
            direct = self._read_unit(
                self._stripes[stripe_no],
                offset,
                slot.chunk.array[slot.start : slot.start + size],
            )
            latency = time.perf_counter() - start
        except BaseException as exc:  # noqa: BLE001 - delivered to the waiters
            self._fail(key, index, exc)
            return
        with self._lock:
            self._latencies.append(latency)
            self._reads += 1
            self._bytes_read += size
            if direct:
                self._reads_direct += 1
            else:
                self._reads_buffered += 1
            self._latency_max = max(self._latency_max, latency)
            state = self._keys.get(key)
            if self._closed or state is None or slot.gen != gen:
                return
            state.state = "ready"
            slot.state = "ready"
            waiters = state.waiters or []
            state.waiters = None
            for future, lease in waiters:
                if not future.done():
                    self._resolve(key, state, future, lease)
            if slot.leases == 0:
                self._policy.make_evictable(key)
            batch = self._pump_locked()
        self._submit(batch)

    def _fail(self, key: Key, index: int, exc: BaseException) -> None:
        with self._lock:
            self._errors += 1
            state = self._keys.pop(key, None)
            slot = self._slots[index]
            if slot.key == key:
                slot.state = "free"
                slot.key = None
                slot.gen += 1
                self._free.append(index)
            self._policy.remove(key)
            waiters = (state.waiters if state else None) or []
            for future, _ in waiters:
                if not future.done():
                    future.set_exception(exc)
            batch = [] if self._closed else self._pump_locked()
        self._submit(batch)

    def _read_unit(self, stripe: _Stripe, offset: int, out: np.ndarray) -> bool:
        """Fill ``out`` from the stripe at ``offset``; True if O_DIRECT was used."""
        view = memoryview(out)
        need = len(view)
        got = 0
        attempt = 0
        while got < need:
            fd, direct = stripe.fd, stripe.direct
            error = ""
            try:
                count = self._preadv(fd, [view[got:]], offset + got)
                if count > 0:
                    got += count
                    if direct and got < need:
                        got &= ~(PAGE - 1)  # O_DIRECT resumes on a page boundary
                    continue
                error = f"short read ({got} < {need} B)"
            except OSError as exc:
                if exc.errno == errno.EINVAL and direct:
                    stripe.fall_back(fd)
                    continue
                error = f"errno {exc.errno} ({exc.strerror})"
            if attempt >= len(RETRY_BACKOFF_S):
                raise ExpertStoreError(
                    f"read failed at {offset + got} of {stripe.path}: {error}"
                )
            time.sleep(RETRY_BACKOFF_S[attempt])
            attempt += 1
        return stripe.direct

    # -- leases ------------------------------------------------------------

    def _acquire(self, slot: _Slot, gen: int) -> None:
        with self._lock:
            if slot.gen != gen or slot.state != "ready" or slot.key is None:
                raise EntryEvictedError("the unit was evicted from its slot")
            slot.leases += 1
            self._policy.make_pinned(slot.key)

    def _release(self, slot: _Slot) -> None:
        with self._lock:
            batch = self._release_locked(slot)
        self._submit(batch)

    def _release_locked(self, slot: _Slot) -> list[tuple[Key, int, int]]:
        slot.leases -= 1
        if slot.leases == 0 and slot.state == "ready" and slot.key is not None:
            self._policy.make_evictable(slot.key)
            return self._pump_locked()
        return []

    # -- inspection and closing -------------------------------------------

    def resident(self, key: Key) -> bool:
        """Whether the unit is in the pool now (a hint: it may be evicted)."""
        with self._lock:
            state = self._keys.get((int(key[0]), int(key[1])))
            return state is not None and state.state == "ready"

    def stats(self) -> dict[str, Any]:
        """Counters since open (a snapshot)."""
        with self._lock:
            lat = np.array(self._latencies, dtype=np.float64)
            used = sum(1 for s in self._slots if s.state != "free")
            ready = sum(1 for s in self._slots if s.state == "ready")
            leased = sum(1 for s in self._slots if s.leases > 0)
            out = {
                "hits": self._hits,
                "misses": self._misses,
                "joins": self._joins,
                "reads": self._reads,
                "bytes_read": self._bytes_read,
                "read_errors": self._errors,
                "evictions": self._evictions,
                "reads_direct": self._reads_direct,
                "reads_buffered": self._reads_buffered,
                "direct": [s.direct for s in self._stripes],
                "waits": self._waits,
                "wait_seconds": self._wait_s,
                "slots": self.num_slots,
                "slots_used": used,
                "slots_ready": ready,
                "slots_leased": leased,
                "slots_queued_keys": len(self._queue),
                "occupancy": used / self.num_slots,
                "pool_bytes": self.num_slots * self.layout.slot_bytes,
                "pinned": self.pinned,
            }
        if lat.size:
            out["read_ms"] = {
                "p50": float(np.percentile(lat, 50)) * 1e3,
                "p95": float(np.percentile(lat, 95)) * 1e3,
                "max": self._latency_max * 1e3,
            }
        else:
            out["read_ms"] = {"p50": 0.0, "p95": 0.0, "max": 0.0}
        return out

    def close(self) -> None:
        """Stop the I/O threads (after the reads under way), fail pending gets."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            error = StoreClosedError("store was closed")
            for state in self._keys.values():
                for future, _ in state.waiters or []:
                    if not future.done():
                        future.set_exception(error)
                state.waiters = None
            self._queue.clear()
        for stripe in self._stripes:
            stripe.close()

    def __enter__(self) -> "ExpertStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

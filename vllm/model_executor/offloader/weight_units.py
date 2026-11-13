# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named weight components on NVMe: the expert store's machinery for any weights
that never change (a vision tower, an audio tower, later others).

A *component* is a named set of tensors, typically a module's state dict. It is
stored as a byte stream (tensors back to back, each start aligned) cut into
*extents*: the units of the expert store, padded to 4096 bytes, striped over the
drives, read with one ``preadv`` into a pool slot. How the stream is cut is a
setting of the component:

* ``"whole"``: as few extents as the extent cap allows (one, up to the cap);
* an integer: extents of that many bytes (a multiple of 4096);
* ``"tensor"``: every tensor starts a new extent (a tensor above the cap is
  cut into several).

No extent is larger than ``max_extent_bytes`` (at most the pinned pool's 1 GiB
chunk), so a component of any size never needs one contiguous allocation above
that: a "unit" of 3 GiB is several extents read into several slots.

Every component has its own pool, whose slot is the component's largest extent:
a 2.5 MB component and a 100 MB component in one process do not share a slot
size. The pools are the expert store's own (:class:`ExpertStore`: leases,
eviction, direct reads, stripes, optional per-unit checks), one instance per
component over the same stripe files, opened when a component is first used.

Component stores are siblings of expert stores, not rows of one: their own
directories (or a subdirectory of the same drives), their own manifest
(``weight-units/v1``), the same stripe file format (header page, 4096-byte
units) and writer protocol (records, finalize, per-unit digests). A component
store has no closed-form offset: its extents differ in size, so offsets come from
a plan derived from the manifest (greedy: each extent goes to the stripe with
the fewest bytes so far).
"""

import hashlib
import os
import threading
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm.model_executor.offloader import expert_store as es
from vllm.model_executor.offloader.expert_store import (
    CHUNK_BYTES,
    DATA_FILE,
    DATA_OFFSET,
    PAGE,
    ExpertStore,
    ExpertStoreError,
    ExpertStoreWriter,
    PoolEntry,
    StoreCorruptError,
    StoreIncompleteError,
    StoreMismatchError,
    TensorSpec,
)

FORMAT = "weight-units/v1"
DEFAULT_MAX_EXTENT = 256 << 20
SPLIT_WHOLE = "whole"
SPLIT_TENSOR = "tensor"

Key = tuple[int, int]  # (component index, extent index)


# --------------------------------------------------------------------------
# Schema and plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    """One component: a name, its tensors in stream order, and how to cut it."""

    name: str
    tensors: tuple[TensorSpec, ...]
    split: int | str = SPLIT_TENSOR

    @classmethod
    def from_state_dict(
        cls, name: str, state_dict: Mapping[str, Any], split: int | str = SPLIT_WHOLE
    ) -> "ComponentSpec":
        """The spec of a module's state dict (tensors in dict order)."""
        specs = []
        for tensor_name, value in state_dict.items():
            dtype = str(value.dtype).removeprefix("torch.")
            specs.append(TensorSpec(tensor_name, dtype, tuple(value.shape)))
        return cls(name, tuple(specs), split)


@dataclass(frozen=True)
class ComponentSchema:
    """The components of a store and the settings they share."""

    components: tuple[ComponentSpec, ...]
    max_extent_bytes: int = DEFAULT_MAX_EXTENT
    tensor_align: int = es.DEFAULT_TENSOR_ALIGN


@dataclass(frozen=True)
class Piece:
    """A byte range of one tensor and where it sits inside an extent."""

    tensor: int  # index into the component's tensors
    tensor_start: int  # first byte of the tensor held here
    nbytes: int
    unit_offset: int  # offset inside the extent


@dataclass(frozen=True)
class _Extent:
    payload_bytes: int
    unit_bytes: int
    pieces: tuple[Piece, ...]


@dataclass(frozen=True)
class _Plan:
    spec: ComponentSpec
    offsets: tuple[int, ...]  # stream offset of each tensor
    stream_bytes: int
    extents: tuple[_Extent, ...]

    @property
    def tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.spec.tensors)


def _round_up(value: int, unit: int) -> int:
    return -(-value // unit) * unit


def _make_plan(spec: ComponentSpec, align: int, max_extent: int) -> _Plan:
    names = [t.name for t in spec.tensors]
    if not spec.name or not names or len(set(names)) != len(names):
        raise ValueError(
            f"component {spec.name!r}: needs a name and tensors with unique names"
        )
    starts: list[int] = []
    off = 0
    for t in spec.tensors:
        off = _round_up(off, align)
        starts.append(off)
        off += t.nbytes
    total = off
    ranges: list[tuple[int, int]] = []
    split = spec.split
    if split == SPLIT_TENSOR:
        for start, t in zip(starts, spec.tensors):
            for a in range(0, t.nbytes, max_extent):
                ranges.append((start + a, start + min(t.nbytes, a + max_extent)))
    else:
        if split == SPLIT_WHOLE:
            size = max_extent
        elif isinstance(split, int) and not isinstance(split, bool):
            size = split
            if size < PAGE or size % PAGE or size > max_extent:
                raise ValueError(
                    f"component {spec.name!r}: split {split} must be a multiple of "
                    f"{PAGE} and at most max_extent_bytes ({max_extent})"
                )
        else:
            raise ValueError(f"component {spec.name!r}: unknown split {split!r}")
        for a in range(0, total, size):
            ranges.append((a, min(total, a + size)))
    if not ranges:
        raise ValueError(f"component {spec.name!r} holds no bytes")
    extents = []
    for a, b in ranges:
        pieces = []
        i = max(0, bisect_right(starts, a) - 1)
        while i < len(starts) and starts[i] < b:
            s, e = starts[i], starts[i] + spec.tensors[i].nbytes
            lo, hi = max(a, s), min(b, e)
            if hi > lo:
                pieces.append(Piece(i, lo - s, hi - lo, lo - a))
            i += 1
        extents.append(_Extent(b - a, _round_up(b - a, PAGE), tuple(pieces)))
    return _Plan(spec, tuple(starts), total, tuple(extents))


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ViewGroup:
    payload_bytes: int
    tensors: tuple[Any, ...] = ()


class _ComponentView:
    """One component as the layout an :class:`ExpertStore` pool expects."""

    def __init__(self, layout: "CatalogLayout", comp: int):
        self._layout = layout
        self.comp = comp
        plan = layout.plans[comp]
        self.num_units = len(plan.extents)
        self.slot_bytes = max(e.unit_bytes for e in plan.extents)
        self._group = _ViewGroup(self.slot_bytes)
        self.layers = comp + 1
        self.experts = self.num_units

    def unit_index(self, layer: int, expert: int) -> int:
        if layer != self.comp or not 0 <= expert < self.num_units:
            raise ValueError(
                f"unit ({layer}, {expert}) is not an extent of component {self.comp}"
            )
        return self._layout.base[self.comp] + expert

    def locate(self, layer: int, expert: int) -> tuple[int, int, int]:
        return self._layout.locate(layer, expert)

    def group_of(self, layer: int) -> _ViewGroup:
        return self._group


class CatalogLayout:
    """Where every extent of every component sits, derived from a schema."""

    def __init__(
        self,
        schema: ComponentSchema,
        num_stripes: int,
        identity: str,
        stripe_fn: Mapping[str, Any] | None = None,
    ):
        if num_stripes < 1:
            raise ValueError("num_stripes must be >= 1")
        if schema.tensor_align < 1 or schema.tensor_align & (schema.tensor_align - 1):
            raise ValueError("tensor_align must be a power of two")
        cap = schema.max_extent_bytes
        if cap < PAGE or cap % PAGE or cap > CHUNK_BYTES:
            raise ValueError(
                f"max_extent_bytes must be a multiple of {PAGE} up to {CHUNK_BYTES}"
            )
        self.stripe_fn = dict(stripe_fn or {"kind": "greedy-bytes"})
        if self.stripe_fn.get("kind") != "greedy-bytes":
            raise StoreMismatchError(
                f"unsupported stripe function {self.stripe_fn!r}; only 'greedy-bytes'"
            )
        names = [c.name for c in schema.components]
        if not names or len(set(names)) != len(names):
            raise ValueError("a store needs components with unique names")
        self.schema = schema
        self.num_stripes = num_stripes
        self.identity = identity
        self.plans = [
            _make_plan(c, schema.tensor_align, cap) for c in schema.components
        ]
        self.names = names
        self.base: list[int] = []
        sizes: list[int] = []
        for plan in self.plans:
            self.base.append(len(sizes))
            sizes.extend(e.unit_bytes for e in plan.extents)
        self.total_units = len(sizes)
        self.layers = len(self.plans)
        self.experts = max(len(p.extents) for p in self.plans)
        self.slot_bytes = max(sizes)
        totals = [DATA_OFFSET] * num_stripes
        self._stripe = np.zeros(self.total_units, dtype=np.int64)
        self._offset = np.zeros(self.total_units, dtype=np.int64)
        self._size = np.array(sizes, dtype=np.int64)
        for i, size in enumerate(sizes):
            s = min(range(num_stripes), key=lambda k: totals[k])
            self._stripe[i] = s
            self._offset[i] = totals[s]
            totals[s] += size
        self.stripe_bytes = tuple(totals)

    # -- addressing --

    def unit_index(self, comp: int, unit: int) -> int:
        if not (0 <= comp < self.layers and 0 <= unit < len(self.plans[comp].extents)):
            raise ValueError(f"unit ({comp}, {unit}) is outside the store")
        return self.base[comp] + unit

    def locate(self, comp: int, unit: int) -> tuple[int, int, int]:
        """(stripe, byte offset in the stripe file, unit bytes) of an extent."""
        i = self.unit_index(comp, unit)
        return int(self._stripe[i]), int(self._offset[i]), int(self._size[i])

    def key_of(self, index: int) -> Key:
        comp = bisect_right(self.base, index) - 1
        return comp, index - self.base[comp]

    def extent(self, comp: int, unit: int) -> _Extent:
        return self.plans[comp].extents[unit]

    def view(self, comp: int) -> _ComponentView:
        return _ComponentView(self, comp)

    # -- manifest --

    def manifest_core(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "identity": self.identity,
            "tensor_align": self.schema.tensor_align,
            "max_extent_bytes": self.schema.max_extent_bytes,
            "components": [
                {
                    "name": p.spec.name,
                    "split": p.spec.split,
                    "stream_bytes": p.stream_bytes,
                    "extents": len(p.extents),
                    "tensors": [
                        {
                            "name": t.name,
                            "dtype": t.dtype,
                            "shape": list(t.shape),
                            "offset": off,
                            "nbytes": t.nbytes,
                        }
                        for t, off in zip(p.spec.tensors, p.offsets)
                    ],
                }
                for p in self.plans
            ],
            "num_stripes": self.num_stripes,
            "stripe_fn": self.stripe_fn,
            "data_offset": DATA_OFFSET,
            "stripe_bytes": list(self.stripe_bytes),
        }

    def layout_sha256(self) -> str:
        return hashlib.sha256(es._canonical(self.manifest_core())).hexdigest()

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "CatalogLayout":
        try:
            if manifest["format"] != FORMAT:
                raise StoreMismatchError(
                    f"format {manifest['format']!r}, this reader takes {FORMAT!r}"
                )
            comps = tuple(
                ComponentSpec(
                    c["name"],
                    tuple(
                        TensorSpec(t["name"], t["dtype"], tuple(t["shape"]))
                        for t in c["tensors"]
                    ),
                    c["split"],
                )
                for c in manifest["components"]
            )
            schema = ComponentSchema(
                comps,
                int(manifest["max_extent_bytes"]),
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
        for key, value in layout.manifest_core().items():
            if manifest.get(key) != value:
                raise StoreCorruptError(
                    f"manifest field {key!r} disagrees with the layout it implies"
                )
        return layout


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


class ComponentStoreWriter(ExpertStoreWriter):
    """Writes components; several processes may write disjoint components or extents.

    The protocol is the expert store's (``flush``, ``close``, ``finalize``,
    resume by writer id, ``rebuild``). A unit is an extent: ``write_component``
    writes all of a component's extents, skipping those this writer already
    recorded.
    """

    def __init__(
        self,
        dirs: Sequence[str],
        schema: ComponentSchema,
        identity: str,
        *,
        writer_id: str | None = None,
        rebuild: bool = False,
    ):
        super().__init__(
            dirs,
            schema,  # type: ignore[arg-type]
            identity,
            writer_id=writer_id,
            rebuild=rebuild,
        )
        self.catalog: CatalogLayout = self.layout

    @staticmethod
    def _make_layout(schema: Any, num_stripes: int, identity: str) -> Any:
        return CatalogLayout(schema, num_stripes, identity)

    def _unit_key(self, index: int) -> Any:
        comp, unit = self.catalog.key_of(index)
        return self.catalog.names[comp], unit

    def write(self, layer: int, expert: int, tensors: Mapping[str, Any]) -> None:
        raise ExpertStoreError("a component store takes write_component()")

    def _component(self, name: str) -> int:
        try:
            return self.catalog.names.index(name)
        except ValueError:
            raise ValueError(
                f"no component {name!r} (have {self.catalog.names})"
            ) from None

    def has_component(self, name: str) -> bool:
        """Whether every extent of the component is recorded by this writer."""
        comp = self._component(name)
        return all(
            self.has_unit(comp, u) for u in range(len(self.catalog.plans[comp].extents))
        )

    def write_component(
        self, name: str, tensors: Mapping[str, Any], *, workers: int = 1
    ) -> int:
        """Write a component from its tensors (name -> torch CPU tensor, ndarray
        or bytes); returns the number of extents written (those already recorded
        by this writer are skipped). Thread-safe across components."""
        if self._closed:
            raise ExpertStoreError("writer is closed")
        comp = self._component(name)
        plan = self.catalog.plans[comp]
        if set(tensors) != {t.name for t in plan.spec.tensors}:
            raise ValueError(
                f"component {name!r} needs tensors "
                f"{[t.name for t in plan.spec.tensors]}, got {sorted(tensors)}"
            )
        raw: list[np.ndarray] = []
        for spec in plan.spec.tensors:
            value = tensors[spec.name]
            got = es._dtype_name(value)
            if got is not None and got != spec.dtype:
                raise ValueError(f"{spec.name}: dtype {got}, schema says {spec.dtype}")
            shape = getattr(value, "shape", None)
            if shape is not None and tuple(shape) != spec.shape:
                raise ValueError(
                    f"{spec.name}: shape {tuple(shape)}, schema says {spec.shape}"
                )
            flat = es._as_bytes(value)
            if flat.size != spec.nbytes:
                raise ValueError(
                    f"{spec.name}: {flat.size} bytes, schema says {spec.nbytes}"
                )
            raw.append(flat)
        todo = [u for u in range(len(plan.extents)) if not self.has_unit(comp, u)]

        def one(unit: int) -> None:
            self._write_extent(comp, unit, raw)

        if workers <= 1 or len(todo) <= 1:
            for u in todo:
                one(u)
        else:
            with ThreadPoolExecutor(workers) as pool:
                for f in [pool.submit(one, u) for u in todo]:
                    f.result()
        return len(todo)

    def _write_extent(self, comp: int, unit: int, raw: list[np.ndarray]) -> None:
        layout = self.catalog
        stripe, offset, unit_bytes = layout.locate(comp, unit)
        buf = np.zeros(unit_bytes, dtype=np.uint8)
        for piece in layout.extent(comp, unit).pieces:
            src = raw[piece.tensor]
            buf[piece.unit_offset : piece.unit_offset + piece.nbytes] = src[
                piece.tensor_start : piece.tensor_start + piece.nbytes
            ]
        view = memoryview(buf)
        index = layout.unit_index(comp, unit)
        self._crc.set(index, es._integrity.unit_digest(view, index))
        done = 0
        while done < unit_bytes:
            done += os.pwrite(self._fds[stripe], view[done:], offset + done)
        with self._lock:
            self._set_bit(index)


# --------------------------------------------------------------------------
# Host destinations
# --------------------------------------------------------------------------


def _torch_dtype(name: str) -> Any:
    import torch

    return getattr(torch, name)


def allocate_host(
    specs: Sequence[TensorSpec], pin: bool = False, align: int = 64
) -> tuple[dict[str, Any], list[Any]]:
    """Empty torch tensors for ``specs`` in host memory.

    Pageable (``pin=False``): one allocation per tensor. Pinned: tensors are
    carved from pinned chunks of at most 1 GiB each (a power-of-two size, which
    the pinned allocator would round to anyway), so no allocation is above the
    size the XPU refuses to pin; a single tensor above 1 GiB cannot be pinned.
    Returns the tensors and the objects that own their memory.
    """
    import torch

    if not pin:
        return {
            s.name: torch.empty(s.shape, dtype=_torch_dtype(s.dtype), device="cpu")
            for s in specs
        }, []
    chunks: list[list[tuple[TensorSpec, int]]] = [[]]
    used = 0
    for s in specs:
        if s.nbytes > CHUNK_BYTES:
            raise ValueError(
                f"{s.name}: {s.nbytes} B is above the {CHUNK_BYTES} B pinned chunk; "
                "a tensor that large cannot be pinned (use pin=False)"
            )
        start = _round_up(used, align)
        if start + s.nbytes > CHUNK_BYTES:
            chunks.append([])
            start = 0
        chunks[-1].append((s, start))
        used = start + s.nbytes
    tensors: dict[str, Any] = {}
    owners = []
    for members in chunks:
        if not members:
            continue
        end = max(start + s.nbytes for s, start in members)
        size = max(PAGE, 1 << (end - 1).bit_length())
        base = torch.empty(
            min(size, CHUNK_BYTES), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        owners.append(base)
        for s, start in members:
            flat = base[start : start + s.nbytes].view(_torch_dtype(s.dtype))
            tensors[s.name] = flat.view(s.shape)
    return {s.name: tensors[s.name] for s in specs}, owners


class _Dest:
    """A destination tensor seen as flat bytes, for piecewise copies."""

    def __init__(self, tensor: Any, spec: TensorSpec):
        import torch

        if isinstance(tensor, np.ndarray):
            if tensor.dtype.itemsize * tensor.size != spec.nbytes or not (
                tensor.flags["C_CONTIGUOUS"] and tensor.flags["WRITEABLE"]
            ):
                raise ValueError(f"{spec.name}: destination is not a matching array")
            self.np: np.ndarray | None = tensor.reshape(-1).view(np.uint8)
            self.torch: Any = None
            return
        if tuple(tensor.shape) != spec.shape or (
            str(tensor.dtype).removeprefix("torch.") != spec.dtype
        ):
            raise ValueError(
                f"{spec.name}: destination is {tuple(tensor.shape)} "
                f"{tensor.dtype}, component has {spec.shape} {spec.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{spec.name}: destination must be contiguous")
        flat = tensor.reshape(-1).view(torch.uint8)
        if tensor.device.type == "cpu":
            self.np = flat.numpy()
            self.torch = None
        else:
            self.np = None
            self.torch = flat

    def copy(self, start: int, src: np.ndarray) -> None:
        if self.np is not None:
            self.np[start : start + src.size] = src
        else:
            import torch

            self.torch[start : start + src.size].copy_(torch.from_numpy(src))


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


class Extent:
    """A leased extent: byte ranges of the component's tensors in a pool slot.

    The lease rules are the expert store's: use the views only while leased,
    release exactly once, and for an asynchronous copy to a card release after
    the copy has completed (an event), not after the call returns.
    """

    def __init__(self, component: "Component", unit: int, entry: PoolEntry):
        self.component = component
        self.unit = unit
        self.entry = entry
        self._extent = component._plan.extents[unit]

    def pieces(self) -> list[tuple[str, int, np.ndarray]]:
        """(tensor name, first byte of the tensor held here, uint8 view of it)."""
        data = self.entry.raw
        spec = self.component._plan.spec
        return [
            (
                spec.tensors[p.tensor].name,
                p.tensor_start,
                data[p.unit_offset : p.unit_offset + p.nbytes],
            )
            for p in self._extent.pieces
        ]

    def release(self) -> None:
        self.entry.release()

    def __enter__(self) -> "Extent":
        return self

    def __exit__(self, *exc: object) -> None:
        if self.entry.held:
            self.entry.release()


class LoadedComponent:
    """A component's tensors in host memory (or in destinations the caller gave)."""

    def __init__(
        self,
        component: "Component",
        tensors: dict[str, Any],
        owners: list[Any],
        owned: bool,
        pinned: bool,
        timings: dict[str, Any],
    ):
        self.component = component
        self.tensors: dict[str, Any] = tensors
        self._owners = owners
        self.owned = owned
        self.pinned = pinned
        self.timings = timings
        self.released = False

    @property
    def nbytes(self) -> int:
        return self.component.nbytes

    def release(self) -> None:
        """Drop the tensors. Memory this object allocated goes back to the
        allocator once nothing else refers to it; destinations the caller gave
        are only forgotten."""
        self.tensors = {}
        self._owners = []
        self.released = True
        self.component._forget(self)

    def __enter__(self) -> "LoadedComponent":
        return self

    def __exit__(self, *exc: object) -> None:
        if not self.released:
            self.release()


class Component:
    """One component of an open :class:`ComponentStore`."""

    def __init__(self, store: "ComponentStore", index: int):
        self._cs = store
        self.index = index
        self._plan = store.layout.plans[index]
        self.name = self._plan.spec.name
        self.tensor_specs = self._plan.spec.tensors
        self.num_extents = len(self._plan.extents)
        self._loaded: LoadedComponent | None = None
        self._lock = threading.RLock()
        self._sub: ExpertStore | None = None
        self._users = 0
        self._keep_pool = False
        self.last_timings: dict[str, Any] = {}

    # -- facts --

    @property
    def nbytes(self) -> int:
        """Bytes of tensor data (without alignment and page padding)."""
        return self._plan.tensor_bytes

    @property
    def stored_bytes(self) -> int:
        """Bytes on the drives (extents padded to 4096)."""
        return sum(e.unit_bytes for e in self._plan.extents)

    @property
    def slot_bytes(self) -> int:
        return self._cs.layout.view(self.index).slot_bytes

    @property
    def state(self) -> str:
        """``"host"`` while a loaded copy is held, else ``"disk"``."""
        return "host" if self._loaded is not None else "disk"

    @property
    def loaded(self) -> LoadedComponent | None:
        return self._loaded

    def _forget(self, loaded: LoadedComponent) -> None:
        with self._lock:
            if self._loaded is loaded:
                self._loaded = None

    # -- pool --

    def _pool_acquire(self) -> ExpertStore:
        with self._lock:
            if self._sub is None:
                self._sub = self._cs._open_sub(self.index)
            self._users += 1
            return self._sub

    def _pool_release(self) -> None:
        with self._lock:
            self._users -= 1
            if self._users == 0 and not self._keep_pool and self._sub is not None:
                sub, self._sub = self._sub, None
                sub.close()

    def close_pool(self) -> None:
        """Free this component's pool (after the readers using it are done)."""
        with self._lock:
            self._keep_pool = False
            if self._users == 0 and self._sub is not None:
                sub, self._sub = self._sub, None
                sub.close()

    def pool_stats(self) -> dict[str, Any] | None:
        sub = self._sub
        return sub.stats() if sub is not None else None

    # -- reading --

    def extents(self, *, prefetch: int = 4) -> Iterator[Extent]:
        """Leased extents in order, ``prefetch`` reads ahead.

        The caller releases each extent (``with`` or ``release``); the pool
        must hold at least ``prefetch`` slots or reads beyond it wait for
        releases. The pool stays open until :meth:`close_pool`.
        """
        sub = self._pool_acquire()
        with self._lock:
            self._keep_pool = True
        depth = max(1, min(prefetch, sub.num_slots))
        window: deque[tuple[int, Future]] = deque()
        nxt = 0
        try:
            while nxt < self.num_extents or window:
                while nxt < self.num_extents and len(window) < depth:
                    future = sub.request([(self.index, nxt)], lease=True)[0]
                    window.append((nxt, future))
                    nxt += 1
                unit, future = window.popleft()
                yield Extent(self, unit, future.result())
        finally:
            for _, future in window:
                if not future.cancel():
                    future.add_done_callback(ExpertStore._drop_late_lease)
            self._pool_release()

    def load(
        self,
        dest: Mapping[str, Any] | None = None,
        *,
        pin: bool = False,
        threads: int | None = None,
        keep_pool: bool = False,
    ) -> LoadedComponent:
        """Bring the component from disk into host memory (or into ``dest``).

        ``dest`` maps tensor names to existing tensors (host pageable, host
        pinned, or on a device: the copy out of the pool slot is synchronous,
        so the slot is released only after the bytes arrived). Without it,
        host tensors are allocated (``pin`` for pinned). Already loaded and no
        ``dest``: returns the held copy. ``threads`` is the number of copy
        workers (default: the reader threads of the store); reads run on the
        store's I/O threads, bounded by the slots of the component's pool.
        ``keep_pool`` leaves the staging pool allocated for the next call.
        Timings of the call are in ``loaded.timings`` and ``last_timings``.
        """
        with self._lock:
            if dest is None and self._loaded is not None:
                return self._loaded
        t0 = time.perf_counter()
        owners: list[Any] = []
        if dest is None:
            tensors, owners = allocate_host(self.tensor_specs, pin)
        else:
            if set(dest) != {t.name for t in self.tensor_specs}:
                raise ValueError(
                    f"dest needs tensors {[t.name for t in self.tensor_specs]}"
                )
            tensors = dict(dest)
        sinks = [_Dest(tensors[t.name], t) for t in self.tensor_specs]
        t_alloc = time.perf_counter()
        sub = self._pool_acquire()
        if keep_pool:
            with self._lock:
                self._keep_pool = True
        t_pool = time.perf_counter()
        try:
            self._read_all(sub, sinks, threads or self._cs.io_threads)
        finally:
            self._pool_release()
        t_end = time.perf_counter()
        total = t_end - t0
        timings = {
            "component": self.name,
            "bytes": self.nbytes,
            "extents": self.num_extents,
            "threads": threads or self._cs.io_threads,
            "alloc_s": t_alloc - t0,
            "pool_open_s": t_pool - t_alloc,
            "read_copy_s": t_end - t_pool,
            "total_s": total,
            "mib_s": self.nbytes / total / (1 << 20) if total else 0.0,
            "pinned": bool(pin and dest is None),
        }
        loaded = LoadedComponent(
            self, tensors, owners, dest is None, bool(pin and dest is None), timings
        )
        with self._lock:
            self.last_timings = timings
            if dest is None:
                self._loaded = loaded
        return loaded

    def _read_all(self, sub: ExpertStore, sinks: list[_Dest], threads: int) -> None:
        def copy_out(unit: int, entry: PoolEntry) -> None:
            try:
                data = entry.raw
                for p in self._plan.extents[unit].pieces:
                    sinks[p.tensor].copy(
                        p.tensor_start, data[p.unit_offset : p.unit_offset + p.nbytes]
                    )
            finally:
                entry.release()

        # A window of at most as many leased reads as the pool has slots: a read
        # for a unit already resident takes its lease at once, so asking for more
        # than the slots hold could leave the oldest read without a slot while
        # the later ones, not yet consumed, hold them all.
        depth = max(1, min(sub.num_slots, self.num_extents))
        window: deque[tuple[int, Future]] = deque()
        jobs: list[Future] = []
        pool = ThreadPoolExecutor(max(1, threads), thread_name_prefix="component-copy")
        failure: BaseException | None = None
        nxt = 0
        try:
            while failure is None and (nxt < self.num_extents or window):
                while nxt < self.num_extents and len(window) < depth:
                    future = sub.request([(self.index, nxt)], lease=True)[0]
                    window.append((nxt, future))
                    nxt += 1
                unit, future = window.popleft()
                try:
                    entry = future.result()
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    failure = exc
                    break
                jobs.append(pool.submit(copy_out, unit, entry))
            for _, future in window:
                if not future.cancel():
                    future.add_done_callback(ExpertStore._drop_late_lease)
        finally:
            pool.shutdown(wait=True)
        for job in jobs:
            job.result()
        if failure is not None:
            raise failure

    def release(self) -> None:
        """Drop the host copy: the component is on disk only again."""
        with self._lock:
            loaded = self._loaded
        if loaded is not None:
            loaded.release()


class ComponentStore:
    """The reader of a component store: one lazily opened pool per component."""

    def __init__(
        self,
        stripe_paths: list[str],
        layout: CatalogLayout,
        pool_bytes: int | Mapping[str, int],
        io_threads: int,
        pin: bool,
        direct: bool,
        decay_interval: int,
        checker: Any,
        read_mode: str = "auto",
    ):
        self.layout = layout
        self._paths = stripe_paths
        self._pool_bytes = pool_bytes
        self.io_threads = io_threads
        self._pin = pin
        self._direct = direct
        self._decay = decay_interval
        self._checker = checker
        self._read_mode = read_mode
        self._closed = False
        self.components: dict[str, Component] = {
            name: Component(self, i) for i, name in enumerate(layout.names)
        }

    @classmethod
    def open(
        cls,
        dirs: Sequence[str],
        identity: str,
        pool_bytes: int | Mapping[str, int] = 256 << 20,
        io_threads: int = 8,
        pin: bool = True,
        *,
        direct: bool = True,
        verify: bool = False,
        decay_interval: int = 256,
        check_every: int = 0,
        check_threads: int = 4,
        read_mode: str = "auto",
    ) -> "ComponentStore":
        """Open a complete component store of this identity.

        ``pool_bytes`` is the staging pool of each component (an int for all,
        or a mapping by name); a pool is at least one slot and never larger
        than its component. Pools are allocated when a component is first read.
        The refusals are the expert store's (incomplete, other identity,
        missing or truncated stripe); ``check_every`` as there.
        """
        dirs = [os.fspath(d) for d in dirs]
        if not dirs:
            raise ValueError("at least one directory is needed")
        manifests = []
        for directory in dirs:
            manifest = es._read_manifest(directory)
            if manifest is None:
                raise StoreIncompleteError(
                    f"{directory}: no {es.MANIFEST_FILE} (store never started)"
                )
            manifests.append(manifest)
        first = manifests[0]
        if not first.get("complete"):
            raise StoreIncompleteError(
                f"{dirs[0]}: store is not complete (not finalized)"
            )
        if first.get("identity") != identity:
            raise StoreMismatchError(
                f"store identity {first.get('identity')!r} is not {identity!r}"
            )
        layout = CatalogLayout.from_manifest(first)
        if first.get("layout_sha256") != layout.layout_sha256():
            raise StoreCorruptError("manifest layout hash does not match its content")
        if len(dirs) != layout.num_stripes:
            raise StoreMismatchError(
                f"store has {layout.num_stripes} stripes, {len(dirs)} directories given"
            )
        ordered: list[str | None] = [None] * layout.num_stripes
        for directory, manifest in zip(dirs, manifests):
            if manifest.get("layout_sha256") != first["layout_sha256"] or not (
                manifest.get("complete")
            ):
                raise StoreMismatchError(
                    f"{directory}: manifest differs from {dirs[0]}"
                )
            path = os.path.join(directory, DATA_FILE)
            try:
                size = os.stat(path).st_size
            except FileNotFoundError as exc:
                raise StoreIncompleteError(f"{path} is missing") from exc
            header = es._read_header(path)
            index = header.get("stripe")
            if (
                header.get("format") != es.FORMAT
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
        stripe_dirs = [str(d) for d in ordered]
        if verify:
            ExpertStore._verify_dirs(stripe_dirs, first)
        checker = es._make_checker(stripe_dirs[0], first, check_every, check_threads)
        return cls(
            [os.path.join(d, DATA_FILE) for d in stripe_dirs],
            layout,
            pool_bytes,
            io_threads,
            pin,
            direct,
            decay_interval,
            checker,
            read_mode,
        )

    # -- access --

    def component(self, name: str) -> Component:
        try:
            return self.components[name]
        except KeyError:
            raise KeyError(
                f"no component {name!r} (have {list(self.components)})"
            ) from None

    def __getitem__(self, name: str) -> Component:
        return self.component(name)

    def _pool_size(self, comp: Component) -> int:
        view = self.layout.view(comp.index)
        if isinstance(self._pool_bytes, Mapping):
            want = int(self._pool_bytes.get(comp.name, view.slot_bytes))
        else:
            want = int(self._pool_bytes)
        want = min(want, comp.stored_bytes)
        return max(view.slot_bytes, want)

    def _open_sub(self, index: int) -> ExpertStore:
        if self._closed:
            raise es.StoreClosedError("store is closed")
        comp = self.components[self.layout.names[index]]
        stripes = [es._Stripe(p, self._direct) for p in self._paths]
        try:
            sub = ExpertStore(
                stripes,
                self.layout.view(index),  # type: ignore[arg-type]
                self._pool_size(comp),
                self.io_threads,
                self._pin,
                es.CountDecayPolicy(self._decay),
                os.preadv,
            )
        except BaseException:
            for stripe in stripes:
                stripe.close()
            raise
        sub._integrity = self._checker
        sub.set_read_mode(self._read_mode)
        return sub

    def stats(self) -> dict[str, Any]:
        """Per component: its pool's stats while the pool is open, else None."""
        return {name: c.pool_stats() for name, c in self.components.items()}

    def slot_bytes(self) -> dict[str, int]:
        """The slot size of each component's pool."""
        return {name: c.slot_bytes for name, c in self.components.items()}

    def close(self) -> None:
        self._closed = True
        for comp in self.components.values():
            comp.close_pool()

    def __enter__(self) -> "ComponentStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

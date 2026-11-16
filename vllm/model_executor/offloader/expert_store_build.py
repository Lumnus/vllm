# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build, verify and describe weight stores outside the engine.

::

    B="python -m vllm.model_executor.offloader.expert_store_build"
    $B build --dirs /nvme0/store /nvme1/store --source pkg.mod:factory \\
        [--arg k=v ...] [--part 0/4] [--workers 4] [--finalize]
    $B finalize  --dirs ...
    $B verify    --dirs ... [--source ... --sample 64]
    $B info      --dirs ... [--json]
    $B checksums --dirs ...

The store is either an expert store (``(layer, expert)`` units, see
``expert_store``) or a component store (named components, see ``weight_units``);
the source says which. The step that turns a checkpoint into kernel-layout
tensors is the *source* (:class:`UnitSource`), a small object supplied by
``module:callable``. Built in: ``--source checkpoint`` (a GPTQ / AutoRound
int4 MoE checkpoint in safetensors, see ``expert_store_ckpt``; the expert store
of an engine, built before the first boot), and ``synthetic`` and
``synthetic-components`` for tests and for trying the tool::

    $B build --source checkpoint --model /models/m --engine-quantization inc \\
        --dirs /nvme0/store /nvme1/store [--part 0/4]
    $B info --source checkpoint --model /models/m --engine-quantization inc

The tool runs on the CPU only (``card_guard``): run it as
``python <tree>/vllm/model_executor/offloader/card_guard.py <command> ...`` to
guard the imports too.

``build`` is resumable (the writer's records say which units are durable; a
rerun with the same ``--part`` skips them) and multi-process friendly: with
``--part i/n`` a process takes the keys ``i, i+n, ...`` of the source's key
list, so n processes write disjoint units; ``finalize`` (or ``--finalize`` on a
single-part build) is run once, after the others exited.
"""

import argparse
import importlib
import json
import os
import sys
import time
import zlib
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from vllm.model_executor.offloader import expert_store as es
from vllm.model_executor.offloader import expert_store_integrity as integrity
from vllm.model_executor.offloader import weight_units as wu


class UnitSource(Protocol):
    """What the build tool needs from a checkpoint reader.

    ``schema`` is a ``StoreSchema`` (expert store) or a ``ComponentSchema``
    (component store); ``identity`` is the string the store is opened with (it
    must change whenever the bytes would); ``unit_keys()`` lists every key once, in a
    stable order (``(layer, expert)`` or a component name); ``read(key)``
    returns that unit's tensors in kernel layout, ``name -> tensor``. ``read``
    is called from several threads.
    """

    identity: str
    schema: Any

    def unit_keys(self) -> Sequence[Any]: ...

    def read(self, key: Any) -> Mapping[str, Any]: ...


# --------------------------------------------------------------------------
# Synthetic sources
# --------------------------------------------------------------------------


def _synth(seed: int, a: int, b: int, name: str, nbytes: int) -> bytes:
    rng = np.random.default_rng([seed, a, b, zlib.crc32(name.encode())])
    return rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()


def _parse_tensors(text: str) -> list[es.TensorSpec]:
    out = []
    for part in text.split(","):
        name, dtype, shape = part.split(":")
        out.append(es.TensorSpec(name, dtype, tuple(int(x) for x in shape.split("x"))))
    return out


class SyntheticExperts:
    """Deterministic random expert units, for tests and for trying the tool."""

    def __init__(
        self,
        layers: int = 4,
        experts: int = 16,
        tensors: str = "w13:int8:32x64,w2:int8:64x16,sc:bfloat16:32x4",
        seed: int = 0,
        delay_ms: float = 0.0,
        identity: str = "synthetic-experts",
    ):
        self.layers, self.experts, self.seed = int(layers), int(experts), int(seed)
        self.delay = float(delay_ms) / 1e3
        self.identity = identity
        self.specs = _parse_tensors(tensors)
        self.schema = es.StoreSchema.uniform(self.layers, self.experts, self.specs)

    def unit_keys(self) -> list[tuple[int, int]]:
        return [(layer, e) for layer in range(self.layers) for e in range(self.experts)]

    def read(self, key: Any) -> dict[str, bytes]:
        if self.delay:
            time.sleep(self.delay)
        layer, expert = key
        return {
            s.name: _synth(self.seed, layer, expert, s.name, s.nbytes)
            for s in self.specs
        }


class SyntheticComponents:
    """Deterministic random components: ``components="name:MiB:split;..."``.

    Each component is six tensors of mixed dtype whose sizes grow by a factor of
    two, together about ``MiB`` mebibytes; ``split`` is ``whole``, ``tensor`` or
    a number of bytes.
    """

    def __init__(
        self,
        components: str = "tower:6:1048576;small:0.25:tensor",
        seed: int = 0,
        max_extent: int = 4 << 20,
        delay_ms: float = 0.0,
        identity: str = "synthetic-components",
    ):
        self.seed = int(seed)
        self.delay = float(delay_ms) / 1e3
        self.identity = identity
        specs = []
        for item in components.split(";"):
            name, mib, split = item.split(":")
            total = int(float(mib) * (1 << 20))
            dtypes = ["bfloat16", "float32", "int8"]
            tensors = []
            for i in range(6):
                dt = dtypes[i % 3]
                size = max(es._ITEMSIZE[dt], total * 2**i // 63)
                tensors.append(es.TensorSpec(f"t{i}", dt, (size // es._ITEMSIZE[dt],)))
            specs.append(
                wu.ComponentSpec(
                    name, tuple(tensors), split if split.isalpha() else int(split)
                )
            )
        self.schema = wu.ComponentSchema(tuple(specs), int(max_extent))

    def unit_keys(self) -> list[str]:
        return [c.name for c in self.schema.components]

    def read(self, key: Any) -> dict[str, bytes]:
        if self.delay:
            time.sleep(self.delay)
        spec = next(c for c in self.schema.components if c.name == key)
        return {
            t.name: _synth(self.seed, 0, 0, f"{key}/{t.name}", t.nbytes)
            for t in spec.tensors
        }


def _coerce(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def load_source(spec: str, args: Iterable[str] = ()) -> UnitSource:
    """``checkpoint``, ``synthetic``, ``synthetic-components`` or
    ``module:callable``; ``args`` are ``key=value`` pairs passed to it as keyword
    arguments."""
    kwargs = {}
    for item in args:
        key, _, value = item.partition("=")
        kwargs[key] = _coerce(value)
    if spec == "synthetic":
        return SyntheticExperts(**kwargs)
    if spec == "synthetic-components":
        return SyntheticComponents(**kwargs)
    if spec == "checkpoint":
        from vllm.model_executor.offloader.expert_store_ckpt import CheckpointExperts

        return CheckpointExperts(**kwargs)
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(
            "--source is 'checkpoint', 'synthetic', 'synthetic-components' or "
            "module:callable"
        )
    factory = getattr(importlib.import_module(module), attr)
    return factory(**kwargs)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def open_writer(
    dirs: Sequence[str],
    schema: Any,
    identity: str,
    *,
    writer_id: str | None = None,
    rebuild: bool = False,
) -> es.ExpertStoreWriter:
    """The right writer for ``schema`` (experts or components)."""
    if isinstance(schema, wu.ComponentSchema):
        return wu.ComponentStoreWriter(
            dirs, schema, identity, writer_id=writer_id, rebuild=rebuild
        )
    return es.ExpertStoreWriter(
        dirs, schema, identity, writer_id=writer_id, rebuild=rebuild
    )


def item_done(writer: es.ExpertStoreWriter, key: Any) -> bool:
    """Whether the writer already recorded ``key`` (a resumed build skips it)."""
    if isinstance(writer, wu.ComponentStoreWriter):
        return writer.has_component(key)
    return writer.has_unit(*key)


def write_item(
    writer: es.ExpertStoreWriter, key: Any, tensors: Mapping[str, Any]
) -> None:
    if isinstance(writer, wu.ComponentStoreWriter):
        writer.write_component(key, tensors)
    else:
        writer.write(key[0], key[1], tensors)


@dataclass
class BuildResult:
    written: int = 0
    skipped: int = 0
    seconds: float = 0.0


def build_from_iter(
    writer: es.ExpertStoreWriter,
    items: Iterable[tuple[Any, Mapping[str, Any]]],
    *,
    skip_done: bool = True,
    flush_every: int = 64,
    workers: int = 1,
) -> BuildResult:
    """Write ``(key, tensors)`` pairs; the caller closes the writer.

    Keys are ``(layer, expert)`` for an expert store and a component name for a
    component store. With ``skip_done`` a pair whose key this writer already
    recorded is not written again (the pair was produced already; pass a lazy
    iterator or use :func:`build_source` to skip the production as well).
    ``workers`` > 1 writes in threads (the iterator is consumed by the caller).
    """
    result = BuildResult()
    start = time.perf_counter()
    pending = []
    pool = ThreadPoolExecutor(workers) if workers > 1 else None
    try:
        for key, tensors in items:
            if skip_done and item_done(writer, key):
                result.skipped += 1
                continue
            if pool is None:
                write_item(writer, key, tensors)
            else:
                pending.append(pool.submit(write_item, writer, key, tensors))
                if len(pending) >= 2 * workers:
                    pending.pop(0).result()
            result.written += 1
            if flush_every and result.written % flush_every == 0:
                writer.flush()
        for future in pending:
            future.result()
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    result.seconds = time.perf_counter() - start
    return result


def _unit_payload_bytes(source: Any) -> int:
    """Payload bytes of one unit of an expert source (0 when not known)."""
    schema = getattr(source, "schema", None)
    groups = getattr(schema, "groups", None)
    if not groups:
        return 0
    return sum(t.nbytes for t in groups[0].tensors)


def parse_part(text: str) -> tuple[int, int]:
    index, _, count = text.partition("/")
    i, n = int(index), int(count or 1)
    if not 0 <= i < n:
        raise ValueError(f"--part {text!r}: need 0 <= i < n")
    return i, n


def build_source(
    dirs: Sequence[str],
    source: UnitSource,
    *,
    part: tuple[int, int] = (0, 1),
    workers: int = 4,
    flush_every: int = 64,
    rebuild: bool = False,
    finalize: bool = False,
    checksum: bool = True,
    log: Any = None,
    expect_identity: str | None = None,
) -> BuildResult:
    """Build (or resume) this process's share of a store from ``source``.

    ``expect_identity``: refuse (before touching the directories) unless the
    source's identity is this string, e.g. the identity an engine reported.

    Reads and writes run in ``workers`` threads. The writer id is
    ``part<i>of<n>``, so a rerun resumes the same record. ``finalize`` also
    publishes the store (only for a single-part build; with several parts run
    :func:`finalize_store` once after all exited).
    """
    index, count = part
    if expect_identity is not None and source.identity != expect_identity:
        raise ValueError(
            f"the source's identity is {source.identity}, expected "
            f"{expect_identity}: the engine would not accept this store"
        )
    if finalize and count != 1:
        raise ValueError(
            "--finalize with several parts: run `finalize` once afterwards"
        )
    writer = open_writer(
        dirs,
        source.schema,
        source.identity,
        writer_id=f"part{index}of{count}",
        rebuild=rebuild,
    )
    keys = list(source.unit_keys())[index::count]
    todo = [k for k in keys if not item_done(writer, k)]
    result = BuildResult(skipped=len(keys) - len(todo))
    start = time.perf_counter()
    done = 0
    unit_bytes = _unit_payload_bytes(source)
    if log:
        log(
            f"part {index}/{count}: {len(todo)} units to write, "
            f"{result.skipped} already done, {max(1, workers)} workers"
        )

    def one(key: Any) -> None:
        write_item(writer, key, source.read(key))

    def progress() -> str:
        elapsed = max(time.perf_counter() - start, 1e-9)
        rate = done / elapsed
        text = f"{done}/{len(todo)} units, {rate:.1f} units/s"
        if unit_bytes:
            text += f", {rate * unit_bytes / (1 << 20):.1f} MiB/s"
        if rate > 0:
            text += (
                f", {elapsed:.0f} s elapsed, ~{(len(todo) - done) / rate:.0f} s left"
            )
        return text

    try:
        # At most `workers` units are in flight (read, converted, written);
        # the queue holds only keys.
        with ThreadPoolExecutor(max(1, workers)) as pool:
            futures = [pool.submit(one, k) for k in todo]
            for future in futures:
                future.result()
                done += 1
                if flush_every and done % flush_every == 0:
                    writer.flush()
                    if log:
                        log(progress())
        if log and done % max(flush_every, 1):
            log(progress())
        writer.close()
    except BaseException:
        writer.__exit__(RuntimeError, None, None)
        raise
    result.written = len(todo)
    result.seconds = time.perf_counter() - start
    if finalize:
        finalize_store(dirs, checksum=checksum)
    return result


# --------------------------------------------------------------------------
# Reading a store's description
# --------------------------------------------------------------------------


def _load_layout(manifest: Mapping[str, Any]) -> Any:
    fmt = manifest.get("format")
    if fmt == es.FORMAT:
        return es._Layout.from_manifest(manifest)
    if fmt == wu.FORMAT:
        return wu.CatalogLayout.from_manifest(manifest)
    raise es.StoreMismatchError(f"unknown store format {fmt!r}")


def _manifest(dirs: Sequence[str]) -> dict[str, Any]:
    manifest = es._read_manifest(os.fspath(dirs[0]))
    if manifest is None:
        raise es.StoreIncompleteError(f"{dirs[0]}: no manifest (store never started)")
    return manifest


def _key_of(layout: Any, index: int) -> Any:
    if hasattr(layout, "key_of"):
        return layout.key_of(index)
    return divmod(index, layout.experts)


def _locate(layout: Any, index: int) -> tuple[int, int, int]:
    return layout.locate(*_key_of(layout, index))


def _ordered_dirs(dirs: Sequence[str], layout: Any) -> list[str]:
    """The directories in stripe order, by reading each stripe header."""
    ordered: list[str | None] = [None] * layout.num_stripes
    for directory in dirs:
        header = es._read_header(os.path.join(directory, es.DATA_FILE))
        i = header.get("stripe")
        if not isinstance(i, int) or not 0 <= i < layout.num_stripes or ordered[i]:
            raise es.StoreMismatchError(f"{directory}: stripe position {i!r} unusable")
        ordered[i] = directory
    if None in ordered:
        raise es.StoreMismatchError("a stripe is missing")
    return [str(d) for d in ordered]


def info_store(dirs: Sequence[str]) -> dict[str, Any]:
    """Schema, identity, sizes and stripe layout of a store (complete or not)."""
    dirs = [os.fspath(d) for d in dirs]
    manifest = _manifest(dirs)
    layout = _load_layout(manifest)
    kind = "components" if isinstance(layout, wu.CatalogLayout) else "experts"
    out: dict[str, Any] = {
        "kind": kind,
        "format": manifest["format"],
        "identity": manifest["identity"],
        "complete": bool(manifest.get("complete")),
        "units": layout.total_units,
        "num_stripes": layout.num_stripes,
        "stripe_fn": layout.stripe_fn,
        "stripe_bytes": list(layout.stripe_bytes),
        "total_bytes": sum(layout.stripe_bytes) - layout.num_stripes * es.DATA_OFFSET,
        "slot_bytes": layout.slot_bytes,
        "unit_checksums": manifest.get("unit_checksums"),
        "whole_file_sha256": bool(manifest.get("checksummed")),
    }
    if kind == "experts":
        out["layers"] = layout.layers
        out["experts_per_layer"] = layout.experts
        out["groups"] = [
            {
                "first_layer": g.first_layer,
                "num_layers": g.num_layers,
                "payload_bytes": g.payload_bytes,
                "unit_bytes": g.unit_bytes,
                "padding_bytes_per_unit": g.unit_bytes - g.payload_bytes,
                "tensors": [t.name for t in g.tensors],
            }
            for g in layout.groups
        ]
    else:
        out["max_extent_bytes"] = layout.schema.max_extent_bytes
        out["components"] = [
            {
                "name": p.spec.name,
                "split": p.spec.split,
                "tensors": len(p.spec.tensors),
                "tensor_bytes": p.tensor_bytes,
                "extents": len(p.extents),
                "largest_extent": max(e.unit_bytes for e in p.extents),
                "stored_bytes": sum(e.unit_bytes for e in p.extents),
            }
            for p in layout.plans
        ]
    files = []
    for directory in dirs:
        row: dict[str, Any] = {"dir": directory}
        for name in (es.DATA_FILE, es.DATA_FILE + es.PART_SUFFIX):
            path = os.path.join(directory, name)
            if os.path.exists(path):
                row["file"] = name
                row["size"] = os.stat(path).st_size
        files.append(row)
    out["directories"] = files
    folder = os.path.join(dirs[0], es.WRITERS_DIR)
    if os.path.isdir(folder):
        records = []
        for name in sorted(os.listdir(folder)):
            if name.endswith(".json"):
                with open(os.path.join(folder, name), "rb") as handle:
                    rec = json.loads(handle.read())
                records.append(
                    {"id": rec["writer"], "units": rec["units"], "done": rec["done"]}
                )
        out["writers"] = records
        out["units_recorded"] = sum(r["units"] for r in records)
    return out


def format_info(info: Mapping[str, Any]) -> str:
    gib = 1 << 30
    lines = [
        f"kind        {info['kind']} ({info['format']})",
        f"identity    {info['identity']}",
        f"state       {'complete' if info['complete'] else 'INCOMPLETE'}"
        + (
            f", {info.get('units_recorded', '?')} of {info['units']} units recorded"
            if not info["complete"]
            else f", {info['units']} units"
        ),
        (
            f"size        {info['total_bytes'] / gib:.3f} GiB in "
            f"{info['num_stripes']} stripe(s), largest unit {info['slot_bytes']} B"
        ),
        f"stripes     {info['stripe_fn']} bytes {info['stripe_bytes']}",
        (
            "checksums   unit table: "
            + (info["unit_checksums"]["algo"] if info["unit_checksums"] else "none")
            + "; whole-file sha256: "
            + ("yes" if info["whole_file_sha256"] else "no")
        ),
    ]
    if info["kind"] == "experts":
        lines.append(
            f"layers      {info['layers']} x {info['experts_per_layer']} experts"
        )
        for g in info["groups"]:
            lines.append(
                f"  group layers {g['first_layer']}.."
                f"{g['first_layer'] + g['num_layers'] - 1}:"
                f" unit {g['unit_bytes']} B (payload {g['payload_bytes']}, padding"
                f" {g['padding_bytes_per_unit']}), tensors {g['tensors']}"
            )
    else:
        lines.append(f"extent cap  {info['max_extent_bytes']} B")
        for c in info["components"]:
            lines.append(
                f"  {c['name']}: {c['tensors']} tensors, {c['tensor_bytes']} B,"
                f" split {c['split']}, {c['extents']} extent(s), largest"
                f" {c['largest_extent']} B"
            )
    for d in info["directories"]:
        lines.append(
            f"  {d['dir']}: {d.get('file', 'no stripe file')} {d.get('size', '')}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Finalize, verify, checksums
# --------------------------------------------------------------------------


def finalize_store(dirs: Sequence[str], *, checksum: bool = True) -> None:
    """Publish a store whose writers all closed (the schema is read back from the
    manifest, so no source is needed)."""
    dirs = [os.fspath(d) for d in dirs]
    manifest = _manifest(dirs)
    layout = _load_layout(manifest)
    with open_writer(
        dirs, layout.schema, manifest["identity"], writer_id="finalizer"
    ) as writer:
        writer.finalize(checksum=checksum)


def _scan_units(
    dirs: Sequence[str], layout: Any, fn: Any, threads: int | None = None
) -> None:
    """Call ``fn(index, data)`` for every unit, one thread per stripe, each stripe
    read front to back (buffered reads; this is a maintenance pass)."""
    ordered = _ordered_dirs(dirs, layout)
    per: list[list[tuple[int, int, int]]] = [[] for _ in ordered]
    for index in range(layout.total_units):
        stripe, offset, size = _locate(layout, index)
        per[stripe].append((offset, index, size))

    def run(stripe: int) -> None:
        path = os.path.join(ordered[stripe], es.DATA_FILE)
        with open(path, "rb", buffering=0) as handle:
            buf = bytearray(layout.slot_bytes)
            for offset, index, size in sorted(per[stripe]):
                view = memoryview(buf)[:size]
                handle.seek(offset)
                got = 0
                while got < size:
                    n = handle.readinto(view[got:])
                    if not n:
                        raise es.StoreCorruptError(f"{path}: short read at {offset}")
                    got += n
                fn(index, view)

    with ThreadPoolExecutor(threads or len(ordered)) as pool:
        for future in [pool.submit(run, s) for s in range(len(ordered))]:
            future.result()


def add_checksums(dirs: Sequence[str], *, force: bool = False) -> int:
    """Give a complete store without a unit checksum table one (a full read).

    The stripes are not touched; the table goes next to them and the manifests
    get an ``unit_checksums`` entry. Returns the number of units hashed.
    """
    dirs = [os.fspath(d) for d in dirs]
    manifest = _manifest(dirs)
    if not manifest.get("complete"):
        raise es.StoreIncompleteError("store is not complete")
    if manifest.get("unit_checksums") and not force:
        raise es.ExpertStoreError("store already has unit checksums (--force to redo)")
    layout = _load_layout(manifest)
    table = np.zeros(layout.total_units, dtype=np.dtype("<u8"))

    def record(index: int, data: Any) -> None:
        table[index] = integrity.unit_digest(data, index)

    _scan_units(dirs, layout, record)
    info = integrity.publish_table(list(dirs), table)
    for directory in dirs:
        doc = es._read_manifest(directory)
        assert doc is not None
        doc["unit_checksums"] = info
        es._write_atomic(
            os.path.join(directory, es.MANIFEST_FILE),
            json.dumps(doc, indent=1, sort_keys=True).encode(),
        )
        es._fsync_dir(directory)
    return layout.total_units


def _expected_unit(layout: Any, key: Any, tensors: Mapping[str, Any]) -> np.ndarray:
    """The unit bytes a writer would lay down for ``tensors`` at ``key``."""
    if isinstance(layout, wu.CatalogLayout):
        comp, unit = key
        plan = layout.plans[comp]
        buf = np.zeros(layout.locate(comp, unit)[2], dtype=np.uint8)
        for p in layout.extent(comp, unit).pieces:
            name = plan.spec.tensors[p.tensor].name
            raw = es._as_bytes(tensors[name])
            buf[p.unit_offset : p.unit_offset + p.nbytes] = raw[
                p.tensor_start : p.tensor_start + p.nbytes
            ]
        return buf
    group = layout.group_of(key[0])
    buf = np.zeros(group.unit_bytes, dtype=np.uint8)
    for place in group.tensors:
        buf[place.offset : place.offset + place.nbytes] = es._as_bytes(
            tensors[place.name]
        )
    return buf


def verify_store(
    dirs: Sequence[str],
    *,
    sha: bool = True,
    units: bool = True,
    source: UnitSource | None = None,
    sample: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Check a complete store: whole-file sha256 against the sidecars, every unit
    against its digest, and (with ``source``) ``sample`` random units against
    the source's tensors. Returns a report; ``report["ok"]`` is the verdict."""
    dirs = [os.fspath(d) for d in dirs]
    manifest = _manifest(dirs)
    report: dict[str, Any] = {"ok": True, "problems": [], "notes": []}
    if not manifest.get("complete"):
        report["ok"] = False
        report["problems"].append("store is not complete")
        return report
    layout = _load_layout(manifest)
    ordered = _ordered_dirs(dirs, layout)
    for stripe, directory in enumerate(ordered):
        size = os.stat(os.path.join(directory, es.DATA_FILE)).st_size
        if size != layout.stripe_bytes[stripe]:
            report["ok"] = False
            report["problems"].append(
                f"stripe {stripe}: {size} B on disk, manifest says "
                f"{layout.stripe_bytes[stripe]}"
            )
    if sha:
        if manifest.get("checksummed"):
            try:
                es.ExpertStore._verify_dirs(ordered, manifest)
                report["sha256"] = "ok"
            except es.ExpertStoreError as exc:
                report["ok"] = False
                report["problems"].append(str(exc))
        else:
            report["notes"].append("no whole-file sha256 recorded")
    info = manifest.get("unit_checksums")
    if units:
        if not info:
            report["notes"].append("no unit checksum table (run `checksums`)")
        else:
            table = integrity.load_table(ordered[0], info)
            bad: list[Any] = []

            def check(index: int, data: Any) -> None:
                if integrity.unit_digest(data, index) != int(table[index]):
                    bad.append(_key_of(layout, index))

            _scan_units(ordered, layout, check)
            report["units_checked"] = layout.total_units
            if bad:
                report["ok"] = False
                report["problems"].append(
                    f"{len(bad)} unit(s) do not match their digest, first: {bad[:5]}"
                )
    if source is not None:
        keys = list(source.unit_keys())
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(keys), size=min(sample, len(keys)), replace=False)
        mismatches = []
        for pick in sorted(picks):
            key = keys[int(pick)]
            tensors = source.read(key)
            if isinstance(layout, wu.CatalogLayout):
                comp = layout.names.index(key)
                unit_keys = [(comp, u) for u in range(len(layout.plans[comp].extents))]
            else:
                unit_keys = [key]
            for ukey in unit_keys:
                stripe, offset, size = layout.locate(*ukey)
                with open(os.path.join(ordered[stripe], es.DATA_FILE), "rb") as handle:
                    handle.seek(offset)
                    data = np.frombuffer(handle.read(size), dtype=np.uint8)
                if not np.array_equal(data, _expected_unit(layout, ukey, tensors)):
                    mismatches.append(ukey)
        report["source_sampled"] = len(picks)
        if mismatches:
            report["ok"] = False
            report["problems"].append(
                f"{len(mismatches)} sampled unit(s) differ from the source, "
                f"first: {mismatches[:5]}"
            )
    return report


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.model_executor.offloader.expert_store_build",
        description=(__doc__ or "").split("::")[0].strip(),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dirs", nargs="+", required=True, help="one per drive")

    def source_args(p: argparse.ArgumentParser, required: bool) -> None:
        p.add_argument(
            "--source",
            required=required,
            help="checkpoint | synthetic | synthetic-components | module:callable",
        )
        p.add_argument(
            "--arg", action="append", default=[], help="key=value for the source"
        )
        g = p.add_argument_group("--source checkpoint")
        g.add_argument("--model", help="the checkpoint directory")
        g.add_argument(
            "--engine-quantization",
            help="the quantization the engine resolves (its log line "
            "'quantization=...'); part of the store's identity",
        )
        g.add_argument(
            "--engine-model",
            help="the model string the engine is served with (default --model)",
        )
        g.add_argument("--engine-dtype", help="the engine's dtype (default bfloat16)")
        g.add_argument("--engine-revision", help="the engine's revision (default none)")
        g.add_argument("--name-format", help="tensor name pattern ({layer} ...)")
        g.add_argument("--kernels-src", help="directory holding vllm_xpu_kernels")
        g.add_argument(
            "--strict-qzeros",
            action="store_true",
            help="refuse a checkpoint whose zero points are not symmetric int4 "
            "(compressed-tensors: whose weight_shape is not [out, in])",
        )

    def source_from(args: argparse.Namespace) -> UnitSource:
        extra = []
        if args.source == "checkpoint":
            for flag, field in (
                ("model", "model"),
                ("engine_quantization", "engine_quantization"),
                ("engine_model", "engine_model"),
                ("engine_dtype", "engine_dtype"),
                ("engine_revision", "engine_revision"),
                ("name_format", "name_format"),
                ("kernels_src", "kernels_src"),
            ):
                value = getattr(args, flag, None)
                if value is not None:
                    extra.append(f"{field}={json.dumps(value)}")
            if getattr(args, "strict_qzeros", False):
                extra.append("strict_qzeros=true")
        return load_source(args.source, [*extra, *args.arg])

    p = sub.add_parser("build", help="write (or resume) units from a source")
    common(p)
    source_args(p, True)
    p.add_argument("--part", default="0/1", help="i/n: this process's share")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--flush-every", type=int, default=64)
    p.add_argument("--rebuild", action="store_true", help="discard any store here")
    p.add_argument("--finalize", action="store_true", help="publish (single part)")
    p.add_argument("--no-sha256", action="store_true", help="skip the sha256 sidecar")
    p.add_argument(
        "--expect-identity",
        help="refuse unless the source's identity is this (e.g. an engine's)",
    )
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="torch intra-op threads per process (the workers are the parallelism)",
    )

    p = sub.add_parser("finalize", help="publish a store whose writers all finished")
    common(p)
    p.add_argument("--no-sha256", action="store_true")

    p = sub.add_parser(
        "verify", help="check sha256 sidecars and unit digests, optionally a source"
    )
    common(p)
    source_args(p, False)
    p.add_argument("--sample", type=int, default=64)
    p.add_argument("--no-sha256", action="store_true")
    p.add_argument("--no-units", action="store_true")

    p = sub.add_parser(
        "info",
        help="schema, identity, sizes, stripe layout of a store (--dirs) and/or "
        "what a source finds (--source)",
    )
    p.add_argument("--dirs", nargs="+", help="one per drive")
    source_args(p, False)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser(
        "checksums", help="add the unit checksum table to a complete older store"
    )
    common(p)
    p.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    from vllm.model_executor.offloader import card_guard

    card_guard.prepare_env()
    card_guard.install()
    try:
        if args.cmd == "build":
            import torch

            torch.set_num_threads(max(1, args.torch_threads))
            source = source_from(args)
            _log(card_guard.summary(card_guard.check()))  # imports done, nothing read
            result = build_source(
                args.dirs,
                source,
                part=parse_part(args.part),
                workers=args.workers,
                flush_every=args.flush_every,
                rebuild=args.rebuild,
                finalize=args.finalize,
                checksum=not args.no_sha256,
                log=_log,
                expect_identity=args.expect_identity,
            )
            _log(
                f"built: {result.written} written, {result.skipped} already done,"
                f" {result.seconds:.1f} s"
            )
        elif args.cmd == "finalize":
            finalize_store(args.dirs, checksum=not args.no_sha256)
            _log("finalized")
        elif args.cmd == "verify":
            checked_against = source_from(args) if args.source else None
            report = verify_store(
                args.dirs,
                sha=not args.no_sha256,
                units=not args.no_units,
                source=checked_against,
                sample=args.sample,
            )
            print(json.dumps(report, indent=1))
            _log(card_guard.summary(card_guard.check()))
            return 0 if report["ok"] else 1
        elif args.cmd == "info":
            if not args.dirs and not args.source:
                raise ValueError("info needs --dirs and/or --source")
            out: dict[str, Any] = {}
            if args.source:
                src = source_from(args)
                describe = getattr(src, "info", None)
                out["source"] = describe() if describe else {"identity": src.identity}
                out["source_identity"] = src.identity
            if args.dirs:
                out["store"] = info_store(args.dirs)
                if args.source:
                    out["identity_matches"] = (
                        out["store"]["identity"] == out["source_identity"]
                    )
            if args.json and not args.source:
                print(json.dumps(out["store"], indent=1))  # the shape it always had
            elif args.json:
                print(json.dumps(out, indent=1, default=str))
            else:
                if "source" in out and "name_format" in out["source"]:
                    from vllm.model_executor.offloader.expert_store_ckpt import (
                        format_info as format_source,
                    )

                    print(format_source(out["source"]))
                elif "source" in out:
                    print(f"source      identity {out['source_identity']}")
                if "store" in out:
                    print(format_info(out["store"]))
                if "identity_matches" in out:
                    print(
                        "identity    the store's identity "
                        + ("MATCHES" if out["identity_matches"] else "DIFFERS from")
                        + " the source's"
                    )
            _log(card_guard.summary(card_guard.check()))
        elif args.cmd == "checksums":
            count = add_checksums(args.dirs, force=args.force)
            _log(f"{count} unit checksums written")
        if args.cmd == "build":
            _log(card_guard.summary(card_guard.check()))
    except card_guard.CardTouchedError as exc:
        _log(f"error: the accelerator was touched: {exc}")
        return 3
    except (es.ExpertStoreError, ValueError, OSError) as exc:
        _log(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

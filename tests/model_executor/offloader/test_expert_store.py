# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the NVMe expert store and its pinned pool."""

import errno
import os
import pathlib
import threading
import time

import pytest
import torch

from vllm.model_executor.offloader.expert_store import (
    DATA_FILE,
    CountDecayPolicy,
    EntryEvictedError,
    ExpertStore,
    ExpertStoreError,
    ExpertStoreWriter,
    LayerGroup,
    StoreClosedError,
    StoreCorruptError,
    StoreIncompleteError,
    StoreMismatchError,
    StoreSchema,
    TensorSpec,
    _Layout,
)

IDENT = "ckpt-abc/kernel-v1"


def schema_a(layers: int = 3, experts: int = 5) -> StoreSchema:
    return StoreSchema.uniform(
        layers,
        experts,
        [
            TensorSpec("w13", "int8", (8, 16)),
            TensorSpec("scales", "bfloat16", (4, 8)),
            TensorSpec("odd", "uint8", (5,)),
        ],
    )


def schema_b() -> StoreSchema:
    """Two layer groups with different tensors, hence different unit sizes."""
    return StoreSchema(
        4,
        6,
        (
            LayerGroup(0, 1, (TensorSpec("a", "float32", (33, 7)),)),
            LayerGroup(
                1,
                3,
                (
                    TensorSpec("b", "int16", (100,)),
                    TensorSpec("c", "float16", (3, 3)),
                    TensorSpec("d", "float8_e4m3fn", (1000,)),
                ),
            ),
        ),
    )


def qwen_schema(layers: int, experts: int) -> StoreSchema:
    """2,534,400 B per expert, the shape of the current target model."""
    return StoreSchema.uniform(
        layers,
        experts,
        [
            TensorSpec("w13", "int8", (1280, 1280)),
            TensorSpec("w2", "int8", (2560, 320)),
            TensorSpec("s13", "bfloat16", (1280, 20)),
            TensorSpec("s2", "bfloat16", (2560, 5)),
        ],
    )


def make(spec: TensorSpec, layer: int, expert: int) -> torch.Tensor:
    seed = layer * 100003 + expert * 31 + sum(map(ord, spec.name))
    gen = torch.Generator().manual_seed(seed)
    dtype = getattr(torch, spec.dtype)
    if spec.dtype in ("int8", "int16"):
        return torch.randint(-100, 100, spec.shape, generator=gen).to(dtype)
    if spec.dtype == "uint8":
        return torch.randint(0, 255, spec.shape, generator=gen).to(dtype)
    return torch.randn(spec.shape, generator=gen).to(dtype)


def unit_tensors(schema: StoreSchema, layer: int, expert: int) -> dict:
    group = next(
        g
        for g in schema.groups
        if g.first_layer <= layer < g.first_layer + g.num_layers
    )
    return {t.name: make(t, layer, expert) for t in group.tensors}


def build(
    dirs: list[str],
    schema: StoreSchema,
    identity: str = IDENT,
    checksum: bool = True,
    skip: set | None = None,
) -> None:
    with ExpertStoreWriter(dirs, schema, identity) as writer:
        for layer in range(schema.layers):
            for expert in range(schema.experts_per_layer):
                if skip and (layer, expert) in skip:
                    continue
                writer.write(layer, expert, unit_tensors(schema, layer, expert))
        writer.finalize(checksum=checksum)


def make_dirs(tmp_path, n: int) -> list[str]:
    dirs = []
    for i in range(n):
        d = tmp_path / f"drive{i}"
        d.mkdir()
        dirs.append(str(d))
    return dirs


def check_unit(entry, schema: StoreSchema, layer: int, expert: int) -> None:
    want = unit_tensors(schema, layer, expert)
    got = entry.tensors()
    assert set(got) == set(want)
    for name, tensor in want.items():
        assert got[name].dtype == tensor.dtype
        assert got[name].shape == tensor.shape
        assert torch.equal(got[name].view(torch.uint8), tensor.view(torch.uint8))


def open_store(dirs, schema, slots: int, **kw) -> ExpertStore:
    layout = _Layout(schema, len(dirs), IDENT)
    kw.setdefault("pin", False)
    kw.setdefault("io_threads", 4)
    return ExpertStore.open(dirs, IDENT, slots * layout.slot_bytes, **kw)


# --------------------------------------------------------------------------
# Format and round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stripes", [1, 2, 4])
@pytest.mark.parametrize("which", ["a", "b"])
def test_round_trip_bit_exact(tmp_path, stripes, which):
    schema = schema_a() if which == "a" else schema_b()
    dirs = make_dirs(tmp_path, stripes)
    build(dirs, schema)
    layout = _Layout(schema, stripes, IDENT)
    for i, d in enumerate(dirs):
        assert os.path.getsize(os.path.join(d, DATA_FILE)) == layout.stripe_bytes[i]
        assert layout.stripe_bytes[i] % 4096 == 0
    with open_store(dirs, schema, slots=schema.layers * schema.experts_per_layer) as s:
        keys = [
            (layer, e)
            for layer in range(schema.layers)
            for e in range(schema.experts_per_layer)
        ]
        s.request(keys)
        for layer, e in keys:
            with s.get((layer, e)) as entry:
                check_unit(entry, schema, layer, e)
                assert entry.raw.nbytes == layout.group_of(layer).payload_bytes
        st = s.stats()
        assert st["reads"] == len(keys) and st["misses"] == len(keys)
        assert st["bytes_read"] == sum(layout.locate(*k)[2] for k in keys)


def test_locate_is_a_bijection(tmp_path):
    schema = schema_b()
    for stripes in (1, 3, 4):
        layout = _Layout(schema, stripes, IDENT)
        spans: dict[int, list[tuple[int, int]]] = {s: [] for s in range(stripes)}
        for layer in range(schema.layers):
            for e in range(schema.experts_per_layer):
                stripe, off, size = layout.locate(layer, e)
                assert stripe == (layer * schema.experts_per_layer + e) % stripes
                assert off % 4096 == 0 and size % 4096 == 0
                spans[stripe].append((off, off + size))
        for stripe, items in spans.items():
            items.sort()
            assert items[0][0] == 4096
            for (_a0, a1), (b0, _) in zip(items, items[1:]):
                assert a1 <= b0
            assert items[-1][1] == layout.stripe_bytes[stripe]


def test_manifest_and_sidecar(tmp_path):
    import json

    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema_a())
    for d in dirs:
        names = sorted(os.listdir(d))
        assert DATA_FILE in names and DATA_FILE + ".sha256" in names
        assert "manifest.json" in names and DATA_FILE + ".part" not in names
    first = pathlib.Path(dirs[0], "manifest.json")
    m = json.loads(first.read_bytes())
    assert m["complete"] and m["identity"] == IDENT and m["num_stripes"] == 2
    assert m["stripe_fn"] == {"kind": "modulo"}
    assert m["groups"][0]["unit_bytes"] % 4096 == 0
    assert first.read_bytes() == pathlib.Path(dirs[1], "manifest.json").read_bytes()


def test_dirs_in_any_order_on_open(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 3)
    build(dirs, schema)
    with open_store(dirs[::-1], schema, slots=4) as s, s.get((2, 4)) as entry:
        check_unit(entry, schema, 2, 4)


@pytest.mark.parametrize(
    "schema,unit_bytes",
    [
        (qwen_schema(1, 3), 2_535_424),
        (
            StoreSchema.uniform(1, 3, [TensorSpec("blob", "uint8", (25_171_968,))]),
            25_174_016,
        ),
    ],
)
def test_real_unit_sizes(tmp_path, schema, unit_bytes):
    dirs = make_dirs(tmp_path, 2)
    layout = _Layout(schema, 2, IDENT)
    assert layout.groups[0].unit_bytes == unit_bytes
    assert layout.groups[0].payload_bytes in (2_534_400, 25_171_968)
    build(dirs, schema)
    with open_store(dirs, schema, slots=2) as s:
        for e in (2, 0, 1):
            with s.get((0, e)) as entry:
                check_unit(entry, schema, 0, e)
        assert s.stats()["evictions"] >= 1


def test_writer_checks_its_input(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 1)
    with ExpertStoreWriter(dirs, schema, IDENT) as w:
        good = unit_tensors(schema, 0, 0)
        with pytest.raises(ValueError, match="needs tensors"):
            w.write(0, 0, {"w13": good["w13"]})
        bad = dict(good, w13=good["w13"].to(torch.int16))
        with pytest.raises(ValueError, match="dtype"):
            w.write(0, 0, bad)
        bad = dict(good, w13=torch.zeros(16, 8, dtype=torch.int8))
        with pytest.raises(ValueError, match="shape"):
            w.write(0, 0, bad)
        with pytest.raises(ValueError, match="outside"):
            w.write(9, 0, good)


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------


def test_concurrent_writers_disjoint_units(tmp_path):
    schema = schema_b()
    dirs = make_dirs(tmp_path, 2)
    n_writers = 4
    writers = [
        ExpertStoreWriter(dirs, schema, IDENT, writer_id=f"rank{i}")
        for i in range(n_writers)
    ]

    def run(i: int) -> None:
        for layer in range(schema.layers):
            for e in range(schema.experts_per_layer):
                if e % n_writers == i:
                    writers[i].write(layer, e, unit_tensors(schema, layer, e))
        writers[i].close()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(w.units_written for w in writers) == 24
    writers[2].finalize()
    with open_store(dirs, schema, slots=24, verify=True) as s:
        for layer in range(schema.layers):
            for e in range(schema.experts_per_layer):
                with s.get((layer, e)) as entry:
                    check_unit(entry, schema, layer, e)


def test_finalize_refuses_while_a_writer_is_missing(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 2)
    w0 = ExpertStoreWriter(dirs, schema, IDENT, writer_id="r0")
    w1 = ExpertStoreWriter(dirs, schema, IDENT, writer_id="r1")
    for layer in range(schema.layers):
        for e in range(schema.experts_per_layer):
            if e % 2 == 0:
                w0.write(layer, e, unit_tensors(schema, layer, e))
    w0.close()
    with pytest.raises(StoreIncompleteError, match="not recorded"):
        w0.finalize()
    with pytest.raises(StoreIncompleteError, match="not complete"):
        ExpertStore.open(dirs, IDENT, 1 << 20, pin=False)
    # The missing writer arrives; the store completes.
    for layer in range(schema.layers):
        for e in range(schema.experts_per_layer):
            if e % 2 == 1:
                w1.write(layer, e, unit_tensors(schema, layer, e))
    w1.close()
    w1.finalize()
    with open_store(dirs, schema, slots=2) as s, s.get((1, 3)) as entry:
        check_unit(entry, schema, 1, 3)


def test_overlapping_writers_are_refused(tmp_path):
    schema = schema_a(layers=1, experts=2)
    dirs = make_dirs(tmp_path, 1)
    w0 = ExpertStoreWriter(dirs, schema, IDENT, writer_id="r0")
    w1 = ExpertStoreWriter(dirs, schema, IDENT, writer_id="r1")
    for w in (w0, w1):
        for e in range(2):
            w.write(0, e, unit_tensors(schema, 0, e))
        w.close()
    with pytest.raises(ExpertStoreError, match="more than one writer"):
        w0.finalize()


def test_resume_and_rebuild(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 2)
    w = ExpertStoreWriter(dirs, schema, IDENT, writer_id="r0")
    for e in range(schema.experts_per_layer):
        w.write(0, e, unit_tensors(schema, 0, e))
    w.close()
    # A new process with the same id resumes; the rest is added.
    again = ExpertStoreWriter(dirs, schema, IDENT, writer_id="r0")
    assert again.units_written == schema.experts_per_layer
    for layer in range(1, schema.layers):
        for e in range(schema.experts_per_layer):
            again.write(layer, e, unit_tensors(schema, layer, e))
    again.close()
    again.finalize()
    with open_store(dirs, schema, slots=3) as s, s.get((0, 4)) as entry:
        check_unit(entry, schema, 0, 4)
    # A complete store is not silently overwritten; rebuild replaces it.
    with pytest.raises(StoreMismatchError, match="complete store"):
        ExpertStoreWriter(dirs, schema, IDENT)
    with pytest.raises(StoreMismatchError, match="another layout"):
        ExpertStoreWriter(dirs, schema, "other-identity")
    other = schema_a(layers=2)
    w2 = ExpertStoreWriter(dirs, other, "other-identity", rebuild=True)
    assert w2.units_written == 0
    w2.close()
    with pytest.raises(StoreIncompleteError):
        ExpertStore.open(dirs, "other-identity", 1 << 20, pin=False)


# --------------------------------------------------------------------------
# Refusals at open
# --------------------------------------------------------------------------


def test_refuses_wrong_identity(tmp_path):
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema_a())
    with pytest.raises(StoreMismatchError, match="identity"):
        ExpertStore.open(dirs, "ckpt-abc/kernel-v2", 1 << 22, pin=False)


def test_refuses_missing_stripe_and_wrong_count(tmp_path):
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema_a())
    with pytest.raises(StoreMismatchError, match="2 stripes"):
        ExpertStore.open(dirs[:1], IDENT, 1 << 22, pin=False)
    os.unlink(os.path.join(dirs[1], DATA_FILE))
    with pytest.raises(StoreIncompleteError, match="missing"):
        ExpertStore.open(dirs, IDENT, 1 << 22, pin=False)


def test_refuses_truncated_stripe(tmp_path):
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema_a())
    path = os.path.join(dirs[1], DATA_FILE)
    os.truncate(path, os.path.getsize(path) - 4096)
    with pytest.raises(StoreCorruptError, match="truncated"):
        ExpertStore.open(dirs, IDENT, 1 << 22, pin=False)


def test_stripes_are_identified_by_header_not_directory(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema)
    a, b = (os.path.join(d, DATA_FILE) for d in dirs)
    os.rename(a, a + ".x")
    os.rename(b, a)
    os.rename(a + ".x", b)
    with open_store(dirs, schema, slots=4) as s:
        for e in range(5):
            with s.get((1, e)) as entry:
                check_unit(entry, schema, 1, e)
    # The same stripe twice is refused.
    os.unlink(b)
    os.link(a, b)
    with pytest.raises(StoreMismatchError, match="given twice"):
        ExpertStore.open(dirs, IDENT, 1 << 22, pin=False)


def test_flipped_byte_is_caught_only_when_verifying(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema)
    path = os.path.join(dirs[0], DATA_FILE)
    with open(path, "r+b") as f:
        f.seek(4096 + 7)
        byte = f.read(1)
        f.seek(4096 + 7)
        f.write(bytes([byte[0] ^ 0x01]))
    ExpertStore.open(dirs, IDENT, 1 << 22, pin=False).close()
    with pytest.raises(StoreCorruptError, match="sha256"):
        ExpertStore.open(dirs, IDENT, 1 << 22, pin=False, verify=True)


def test_verify_needs_checksums(tmp_path):
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema_a(), checksum=False)
    ExpertStore.open(dirs, IDENT, 1 << 22, pin=False).close()
    with pytest.raises(StoreIncompleteError, match="without checksums"):
        ExpertStore.open(dirs, IDENT, 1 << 22, pin=False, verify=True)


def test_pool_smaller_than_one_slot_is_refused(tmp_path):
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema_a())
    with pytest.raises(ValueError, match="smaller than one slot"):
        ExpertStore.open(dirs, IDENT, 1000, pin=False)


# --------------------------------------------------------------------------
# Pool: leases, eviction, re-read
# --------------------------------------------------------------------------


def test_leases_gate_the_queue(tmp_path):
    schema = schema_a(layers=1, experts=8)
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema)
    with open_store(dirs, schema, slots=3) as s:
        futures = s.request([(0, e) for e in range(8)], lease=True)
        # Wait for the first three to land, then check nothing else can.
        deadline = time.time() + 10
        while time.time() < deadline and sum(f.done() for f in futures) < 3:
            time.sleep(0.01)
        time.sleep(0.2)
        assert sum(f.done() for f in futures) == 3
        st = s.stats()
        assert st["slots_leased"] == 3 and st["slots_queued_keys"] == 5
        assert st["evictions"] == 0
        # Release in order: each release lets one more key in.
        seen = 0
        for i in range(8):
            entry = futures[i].result(timeout=10)
            check_unit(entry, schema, 0, i)
            entry.release()
            seen += 1
        assert seen == 8
        st = s.stats()
        assert st["slots_leased"] == 0 and st["reads"] == 8
        assert st["evictions"] == 5


def test_leased_slot_is_never_evicted(tmp_path):
    schema = schema_a(layers=1, experts=8)
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    with open_store(dirs, schema, slots=3) as s:
        held = s.get((0, 0))
        for _ in range(3):
            for e in range(1, 8):
                with s.get((0, e)) as entry:
                    check_unit(entry, schema, 0, e)
        assert s.resident((0, 0))
        check_unit(held, schema, 0, 0)
        held.release()


def test_get_on_evicted_unit_rereads(tmp_path):
    schema = schema_a(layers=1, experts=4)
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema)
    with open_store(dirs, schema, slots=2) as s:
        first = s.get((0, 0), lease=False)
        for e in (1, 2, 3):
            s.get((0, e)).release()
        assert not s.resident((0, 0))
        assert not first.valid()
        with pytest.raises(EntryEvictedError):
            first.acquire()
        reads = s.stats()["reads"]
        with s.get((0, 0)) as entry:
            check_unit(entry, schema, 0, 0)
        assert s.stats()["reads"] == reads + 1


def test_eviction_order_by_count(tmp_path):
    schema = schema_a(layers=1, experts=8)
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    with open_store(dirs, schema, slots=3, decay_interval=10**9) as s:
        for e, uses in ((0, 5), (1, 3), (2, 1)):
            for _ in range(uses):
                s.request([(0, e)])[0].result(10)
        s.request([(0, 3)])[0].result(10)  # evicts the lowest count: unit 2
        assert not s.resident((0, 2))
        assert s.resident((0, 0)) and s.resident((0, 1)) and s.resident((0, 3))
        s.request([(0, 4)])[0].result(10)  # now unit 3 (count 1) goes, not 1 or 0
        assert not s.resident((0, 3))
        assert s.resident((0, 0)) and s.resident((0, 1))
        # Equal counts: the least recently used goes.
        s.request([(0, 5)])[0].result(10)  # evicts unit 4 (count 1)
        assert not s.resident((0, 4))
        assert s.stats()["evictions"] == 3


def test_count_decay_policy_halves_and_breaks_ties_by_recency():
    p = CountDecayPolicy(decay_interval=4)
    a, b, c = (0, 0), (0, 1), (0, 2)
    # a: 4 uses at the start; b: 1 use, 8 requests later (two half-lives on).
    for _ in range(4):
        p.tick()
        p.record_use(a)
    for _ in range(8):
        p.tick()
    p.tick()
    p.record_use(b)
    expect = 0.0
    for _ in range(4):
        expect = expect * 2 ** (-1 / 4) + 1.0
    assert p.count(a) == pytest.approx(expect * 2 ** (-9 / 4), rel=1e-9)
    assert p.count(b) == pytest.approx(1.0)
    for key in (a, b):
        p.make_evictable(key)
    assert p.pop_victim() == a  # below 1.0: the old, once-popular key is colder
    assert p.pop_victim() == b
    assert p.pop_victim() is None
    # Ties: same count, the older use goes first.
    q = CountDecayPolicy(decay_interval=10**9)
    for key in (a, b, c):
        q.tick()
        q.record_use(key)
        q.make_evictable(key)
    q.record_use(a)
    q.record_use(a)  # a is popular; b is older than c
    assert [q.pop_victim() for _ in range(3)] == [b, c, a]
    # A pinned key is never chosen; unpinning brings it back.
    q.make_evictable(a)
    q.make_pinned(a)
    assert q.pop_victim() is None
    q.make_evictable(a)
    assert q.pop_victim() == a


def test_many_threads_overlapping_keys_one_read_each(tmp_path):
    schema = schema_a(layers=2, experts=8)
    dirs = make_dirs(tmp_path, 4)
    build(dirs, schema)
    keys = [(layer, e) for layer in range(2) for e in range(8)]
    errors = []
    with open_store(dirs, schema, slots=16, io_threads=8) as s:
        start = threading.Barrier(16)

        def run(i: int) -> None:
            try:
                start.wait()
                mine = keys[i % 4 :] + keys[: i % 4]
                futures = s.request(mine, lease=True)
                for key, f in zip(mine, futures):
                    entry = f.result(timeout=30)
                    check_unit(entry, schema, *key)
                    entry.release()
                    with s.get(key) as again:
                        check_unit(again, schema, *key)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        st = s.stats()
        assert st["reads"] == 16  # one per unit, whatever the overlap
        assert st["misses"] == 16
        assert st["hits"] + st["joins"] + st["misses"] == 16 * 16
        assert st["evictions"] == 0 and st["slots_leased"] == 0


def test_get_timeout_does_not_leak_a_lease(tmp_path):
    schema = schema_a(layers=1, experts=3)
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    with open_store(dirs, schema, slots=1) as s:
        held = s.get((0, 0))
        with pytest.raises(TimeoutError):
            s.get((0, 1), timeout=0.1)
        held.release()
        time.sleep(0.2)  # the abandoned read may still land
        st = s.stats()
        assert st["slots_leased"] == 0
        with s.get((0, 2), timeout=10) as entry:
            check_unit(entry, schema, 0, 2)


def test_read_failure_reaches_the_caller_and_frees_the_slot(tmp_path):
    schema = schema_a(layers=1, experts=2)
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    calls = []

    def broken(fd, bufs, offset):
        calls.append(offset)
        raise OSError(errno.EIO, "injected")

    with open_store(dirs, schema, slots=1, preadv=broken) as s:
        with pytest.raises(ExpertStoreError, match="read failed"):
            s.get((0, 0), timeout=10)
        st = s.stats()
        assert st["read_errors"] == 1 and st["slots_used"] == 0
        assert len(calls) == 4  # first try and three retries


def test_direct_io_falls_back_to_buffered(tmp_path):
    schema = schema_a(layers=1, experts=2)
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    first = []

    def refusing(fd, bufs, offset):
        if not first:
            first.append(1)
            raise OSError(errno.EINVAL, "no direct I/O here")
        return os.preadv(fd, bufs, offset)

    with open_store(dirs, schema, slots=2, preadv=refusing) as s:
        s._stripes[0].direct = True  # pretend the open took O_DIRECT
        with s.get((0, 0), timeout=10) as entry:
            check_unit(entry, schema, 0, 0)
        st = s.stats()
        assert st["direct"] == [False]
        assert st["reads_buffered"] == 1 and st["reads_direct"] == 0


def test_stats_counters(tmp_path):
    schema = schema_a(layers=1, experts=4)
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema)
    layout = _Layout(schema, 2, IDENT)
    with open_store(dirs, schema, slots=2) as s:
        st = s.stats()
        assert st["slots"] == 2 and st["occupancy"] == 0 and st["reads"] == 0
        s.request([(0, 0), (0, 1)])[1].result(10)
        s.request([(0, 0)])[0].result(10)  # hit
        entry = s.get((0, 2))  # miss; evicts one of the two
        st = s.stats()
        assert st["misses"] == 3 and st["hits"] == 1
        assert st["evictions"] == 1 and st["reads"] == 3
        assert st["bytes_read"] == 3 * layout.groups[0].unit_bytes
        assert st["reads_direct"] + st["reads_buffered"] == 3
        assert st["occupancy"] == 1.0 and st["slots_leased"] == 1
        assert 0 < st["read_ms"]["p50"] <= st["read_ms"]["p95"] <= st["read_ms"]["max"]
        entry.release()
        assert s.stats()["slots_leased"] == 0


def test_waits_are_counted(tmp_path):
    schema = schema_a(layers=1, experts=2)
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    gate = threading.Event()

    def slow(fd, bufs, offset):
        gate.wait(5)
        return os.preadv(fd, bufs, offset)

    with open_store(dirs, schema, slots=2, preadv=slow) as s:
        s.request([(0, 0)])
        timer = threading.Timer(0.2, gate.set)
        timer.start()
        with s.get((0, 0), timeout=10) as entry:
            check_unit(entry, schema, 0, 0)
        st = s.stats()
        assert st["waits"] == 1 and st["wait_seconds"] >= 0.1
        assert st["joins"] == 0  # get after request is one use


def test_close_joins_and_fails_pending(tmp_path):
    schema = schema_a(layers=1, experts=4)
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    s = open_store(dirs, schema, slots=1)
    held = s.get((0, 0))
    pending = s.request([(0, 1)])[0]
    s.close()
    with pytest.raises(StoreClosedError):
        pending.result(timeout=5)
    with pytest.raises(StoreClosedError):
        s.request([(0, 2)])
    with pytest.raises(StoreClosedError):
        s.get((0, 2))
    s.close()  # idempotent
    del held


def test_pinned_pool_when_available(tmp_path):
    schema = schema_a(layers=1, experts=2)
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    try:
        s = open_store(dirs, schema, slots=2, pin=True)
    except RuntimeError as exc:
        pytest.skip(f"no pinned host memory on this machine: {exc}")
    with s, s.get((0, 1)) as entry:
        check_unit(entry, schema, 0, 1)
        assert entry.slot_tensor().is_pinned()

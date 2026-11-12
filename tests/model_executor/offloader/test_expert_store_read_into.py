# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the expert store's reads into caller memory (direct, buffered
and through a bounce buffer) and of its status check."""

import errno
import mmap
import os

import numpy as np
import pytest
import torch

from vllm.model_executor.offloader.expert_store import (
    DIRECT_ALIGN,
    ExpertStore,
    ExpertStoreError,
    ExpertStoreWriter,
    StoreCorruptError,
    StoreSchema,
    TensorSpec,
    _Layout,
)

IDENT = "ckpt/kernel-v1"
SIZES = {"w13": 4096, "w2": 2048, "s13": 512, "s2": 1024}  # all 512-multiples
PATHS = {"direct", "buffered", "bounce"}


def schema(sizes=None, layers=2, experts=6) -> StoreSchema:
    sizes = sizes or SIZES
    return StoreSchema.uniform(
        layers,
        experts,
        [TensorSpec(name, "uint8", (n,)) for name, n in sizes.items()],
    )


def unit(sch: StoreSchema, layer: int, expert: int) -> dict[str, np.ndarray]:
    out = {}
    for t in sch.groups[0].tensors:
        rng = np.random.default_rng(layer * 1000 + expert * 7 + len(t.name))
        out[t.name] = rng.integers(0, 256, t.nbytes, dtype=np.uint8)
    return out


def build(tmp_path, sch: StoreSchema, stripes: int = 1) -> list[str]:
    dirs = []
    for i in range(stripes):
        d = tmp_path / f"drive{i}"
        d.mkdir()
        dirs.append(str(d))
    with ExpertStoreWriter(dirs, sch, IDENT) as w:
        for layer in range(sch.layers):
            for e in range(sch.experts_per_layer):
                w.write(layer, e, unit(sch, layer, e))
        w.finalize()
    return dirs


def open_store(dirs, sch, slots=2, **kw) -> ExpertStore:
    kw.setdefault("pin", False)
    kw.setdefault("io_threads", 4)
    layout = _Layout(sch, len(dirs), IDENT)
    return ExpertStore.open(dirs, IDENT, slots * layout.slot_bytes, **kw)


def aligned(nbytes: int, align: int = 4096) -> np.ndarray:
    """Zeroed, writable, ``align``-aligned bytes (an anonymous mapping)."""
    buf = mmap.mmap(-1, nbytes + align)
    arr = np.frombuffer(buf, dtype=np.uint8)
    start = -arr.ctypes.data % align
    return arr[start : start + nbytes]  # a view: keeps the mapping alive


def dests(sch: StoreSchema, offset: int = 0) -> dict[str, np.ndarray]:
    out = {}
    for t in sch.groups[0].tensors:
        # every destination is 4096-aligned, or shifted by ``offset`` bytes
        out[t.name] = aligned(t.nbytes + offset)[offset:]
    return out


def _addr(buf) -> int:
    return np.frombuffer(buf, dtype=np.uint8).ctypes.data


class Direct:
    """A ``preadv`` that behaves like O_DIRECT on a 512-byte drive: it
    refuses a call whose offset, iovec bases or lengths are not multiples of
    512, and records the calls."""

    def __init__(self, partial: bool = False):
        self.calls: list[tuple[int, int]] = []  # (number of iovecs, bytes)
        self.partial = partial

    def __call__(self, fd, bufs, offset):
        total = sum(len(b) for b in bufs)
        self.calls.append((len(bufs), total))
        for b in bufs:
            if _addr(b) % DIRECT_ALIGN or len(b) % DIRECT_ALIGN or offset % DIRECT_ALIGN:
                raise OSError(errno.EINVAL, "misaligned direct I/O")
        if self.partial and total > 2 * DIRECT_ALIGN:
            # a short read that ends mid-tensor, on a 512 boundary
            half = (total // 2) // DIRECT_ALIGN * DIRECT_ALIGN
            data = os.pread(fd, half, offset)
            done = 0
            for b in bufs:
                take = min(len(b), half - done)
                if take <= 0:
                    break
                np.frombuffer(b, dtype=np.uint8)[:take] = np.frombuffer(
                    data[done : done + take], dtype=np.uint8
                )
                done += take
            return half
        return os.preadv(fd, bufs, offset)


class Refusing:
    """A ``preadv`` like the kernel's for memory that cannot take some reads:
    EFAULT when an iovec lies in ``rows`` (the caller's "pinned" buffers) and
    the descriptor is the stripe's direct one, or always with ``buffered``
    too. Memory outside ``rows`` (a bounce buffer) is never refused."""

    def __init__(self, store_ref: list, rows: dict[str, np.ndarray], buffered: bool):
        self.store_ref = store_ref
        self.ranges = [(v.ctypes.data, v.ctypes.data + v.nbytes) for v in rows.values()]
        self.buffered = buffered
        self.calls: list[tuple[int, bool]] = []  # (iovecs, on the direct fd)

    def __call__(self, fd, bufs, offset):
        store = self.store_ref[0]
        direct = any(st.fd == fd and st.direct for st in store._stripes)
        self.calls.append((len(bufs), direct))
        inside = any(lo <= _addr(b) < hi for b in bufs for lo, hi in self.ranges)
        if inside and (direct or self.buffered):
            raise OSError(errno.EFAULT, "Bad address")
        return os.preadv(fd, bufs, offset)


def make_direct(store: ExpertStore) -> None:
    for stripe in store._stripes:
        stripe.direct = True  # whatever the open took here; behave as direct


def check(out, sch, layer, e) -> None:
    for name, want in unit(sch, layer, e).items():
        assert np.array_equal(out[name], want), (layer, e, name)


@pytest.mark.parametrize("stripes", [1, 3])
def test_direct_read_lands_every_tensor_with_one_call(tmp_path, stripes):
    sch = schema()
    dirs = build(tmp_path, sch, stripes)
    direct = Direct()
    with open_store(dirs, sch, preadv=direct) as s:
        make_direct(s)
        for layer, e in [(0, 0), (1, 5), (0, 3), (1, 1)]:
            out = dests(sch)
            assert s.read_into((layer, e), out) == "direct"
            check(out, sch, layer, e)
        st = s.stats()
        assert st["into_reads"] == {"direct": 4, "buffered": 0, "bounce": 0}
        assert st["reads"] == 4 and st["slots_used"] == 0  # nothing in the pool
        assert st["into_bytes"] == 4 * sum(SIZES.values())
        assert all(n == 4 for n, _ in direct.calls)  # one preadv, four iovecs


def test_unaligned_destinations_take_the_bounce_buffer_for_that_read(tmp_path):
    sch = schema()
    dirs = build(tmp_path, sch)
    direct = Direct()
    with open_store(dirs, sch, preadv=direct) as s:
        make_direct(s)
        out = dests(sch, offset=64)  # 64-aligned only
        assert s.read_into((1, 2), out) == "bounce"
        check(out, sch, 1, 2)
        st = s.stats()
        assert st["into_reads"]["bounce"] == 1 and st["read_fallbacks"] == {}
        assert st["into_read_mode"] == "direct" and st["slots_used"] == 0
        # a destination of the right alignment is still read directly
        assert s.read_into((1, 2), dests(sch)) == "direct"


def test_a_drive_that_refuses_the_alignment_moves_reads_to_bounce(tmp_path):
    sch = schema()
    dirs = build(tmp_path, sch)
    calls = []

    def refuse_scatter(fd, bufs, offset):
        calls.append(len(bufs))
        if len(bufs) > 1:
            raise OSError(errno.EINVAL, "no scatter here")
        return os.preadv(fd, bufs, offset)

    with open_store(dirs, sch, preadv=refuse_scatter) as s:
        make_direct(s)
        out = dests(sch)
        assert s.read_into((0, 1), out) == "bounce"
        check(out, sch, 0, 1)
        st = s.stats()
        assert st["into_read_mode"] == "bounce"
        assert st["read_fallbacks"] == {"EINVAL_direct_into": 1}
        assert st["read_mode"] == "direct"  # the pool's own mode did not move
        before = len(calls)
        assert s.read_into((0, 2), dests(sch)) == "bounce"
        assert all(n == 1 for n in calls[before:])  # no more scatter attempts
        assert s.stats()["read_fallbacks"] == {"EINVAL_direct_into": 1}


def test_auto_moves_to_buffered_when_direct_reads_into_the_rows_are_refused(
    tmp_path,
):
    """Pinned accelerator memory: O_DIRECT into it is EFAULT, a buffered read
    is taken."""
    sch = schema()
    dirs = build(tmp_path, sch)
    rows = dests(sch)
    ref: list = []
    preadv = Refusing(ref, rows, buffered=False)
    with open_store(dirs, sch, preadv=preadv) as s:
        ref.append(s)
        make_direct(s)
        assert s.read_into((0, 1), rows) == "buffered"
        check(rows, sch, 0, 1)
        st = s.stats()
        assert st["into_read_mode"] == "buffered"
        assert st["read_fallbacks"] == {"EFAULT_direct_into": 1}
        assert st["read_errors"] == 0
        assert len(preadv.calls) == 2  # one refused, one taken: no retry ladder
        # what worked is kept: no further direct attempt
        assert s.read_into((0, 2), rows) == "buffered"
        check(rows, sch, 0, 2)
        assert [d for _, d in preadv.calls[2:]] == [False]


def test_auto_ends_in_bounce_when_the_rows_take_no_read_at_all(tmp_path):
    sch = schema()
    dirs = build(tmp_path, sch)
    rows = dests(sch)
    ref: list = []
    preadv = Refusing(ref, rows, buffered=True)
    with open_store(dirs, sch, preadv=preadv) as s:
        ref.append(s)
        make_direct(s)
        assert s.read_into((1, 4), rows) == "bounce"
        check(rows, sch, 1, 4)
        st = s.stats()
        assert st["into_read_mode"] == "bounce"
        assert st["read_fallbacks"] == {
            "EFAULT_direct_into": 1,
            "EFAULT_buffered_into": 1,
        }
        assert s.read_into((1, 5), rows) == "bounce"
        check(rows, sch, 1, 5)
        assert st["slots_used"] == 0  # the pool is not involved


def test_a_forced_mode_does_not_fall_back(tmp_path):
    sch = schema()
    dirs = build(tmp_path, sch)
    rows = dests(sch)
    ref: list = []
    preadv = Refusing(ref, rows, buffered=False)
    with open_store(dirs, sch, preadv=preadv, read_mode="direct") as s:
        ref.append(s)
        make_direct(s)
        with pytest.raises(ExpertStoreError, match="errno 14"):
            s.read_into((0, 1), rows)
        assert len(preadv.calls) == 4  # one try and the three retries
        assert s.stats()["into_read_mode"] == "direct"
    with open_store(dirs, sch, read_mode="bounce") as s:
        assert s.read_into((0, 0), dests(sch)) == "bounce"
        assert s.stats()["into_read_mode"] == "bounce"
    with open_store(dirs, sch, read_mode="buffered") as s:
        assert s.read_into((0, 0), dests(sch)) == "buffered"
    with pytest.raises(ValueError, match="read mode"):
        open_store(dirs, sch, read_mode="fast")


def test_unit_with_padding_between_tensors_is_read_through_bounce(tmp_path):
    sizes = {"a": 100, "b": 200}  # 100 is not a multiple of 64: a gap before b
    sch = schema(sizes)
    dirs = build(tmp_path, sch)
    with open_store(dirs, sch) as s:
        out = {n: np.zeros(k, dtype=np.uint8) for n, k in sizes.items()}
        assert s.read_into((1, 4), out) == "bounce"
        check(out, sch, 1, 4)


def test_short_direct_reads_are_continued(tmp_path):
    sch = schema()
    dirs = build(tmp_path, sch)
    with open_store(dirs, sch, preadv=Direct(partial=True)) as s:
        make_direct(s)
        out = dests(sch)
        assert s.read_into((1, 3), out) == "direct"
        check(out, sch, 1, 3)


def test_torch_destinations_any_dtype_and_async_reads(tmp_path):
    sch = schema()
    dirs = build(tmp_path, sch, stripes=2)
    with open_store(dirs, sch, slots=3) as s:
        keys = [(layer, e) for layer in range(2) for e in range(6)]
        rows = {
            k: {
                "w13": torch.zeros(4096, dtype=torch.int8),
                "w2": torch.zeros((32, 64), dtype=torch.uint8),
                "s13": torch.zeros(256, dtype=torch.bfloat16),
                "s2": torch.zeros(1024, dtype=torch.uint8),
            }
            for k in keys
        }
        futures = [s.read_into_async(k, rows[k]) for k in keys]
        assert {f.result(timeout=30) for f in futures} <= PATHS
        for k in keys:
            want = unit(sch, *k)
            for name, tensor in rows[k].items():
                got = tensor.view(torch.uint8).reshape(-1).numpy()
                assert np.array_equal(got, want[name]), (k, name)


def test_bad_destinations_are_refused(tmp_path):
    sch = schema()
    dirs = build(tmp_path, sch)
    with open_store(dirs, sch) as s:
        out = dests(sch)
        out["w2"] = out["w2"][:-1]
        with pytest.raises(ValueError, match="B, the tensor"):
            s.read_into((0, 0), out)
        out = dests(sch)
        del out["s2"]
        with pytest.raises(ValueError, match="no destination"):
            s.read_into((0, 0), out)
        out = dests(sch)
        out["extra"] = np.zeros(4, dtype=np.uint8)
        with pytest.raises(ValueError, match="unknown"):
            s.read_into((0, 0), out)
        ro = dests(sch)
        ro["w13"].flags.writeable = False
        with pytest.raises(ValueError, match="writable"):
            s.read_into((0, 0), ro)


def _flip(dirs, sch, layer, e, at: int) -> None:
    stripe, offset, _ = _Layout(sch, len(dirs), IDENT).locate(layer, e)
    path = os.path.join(sorted(dirs)[stripe], "experts.bin")
    with open(path, "r+b") as handle:
        handle.seek(offset + at)
        byte = handle.read(1)
        handle.seek(offset + at)
        handle.write(bytes([byte[0] ^ 0x40]))


@pytest.mark.parametrize("mode", ["auto", "buffered", "bounce"])
def test_a_read_into_rows_is_checked_against_the_unit_digest(tmp_path, mode):
    sch = schema()
    dirs = build(tmp_path, sch)
    _flip(dirs, sch, 1, 2, at=5000)  # inside the second tensor
    with open_store(dirs, sch, read_mode=mode, check_every=1) as s:
        out = dests(sch)
        s.read_into((1, 1), out)  # a clean unit passes
        check(out, sch, 1, 1)
        with pytest.raises(StoreCorruptError, match=r"unit \(1, 2\)"):
            s.read_into((1, 2), dests(sch))
        st = s.stats()
        assert st["checked"] == 2 and st["check_failures"] == 1
    with open_store(dirs, sch, read_mode=mode, check_every=0) as s:
        s.read_into((1, 2), dests(sch))  # off: the bad unit is not noticed
        assert s.stats()["checked"] == 0


def test_status_reads_the_manifests_only(tmp_path):
    sch = schema()
    d = tmp_path / "drive"
    d.mkdir()
    dirs = [str(d)]
    assert ExpertStore.status(dirs, IDENT) == "absent"
    w = ExpertStoreWriter(dirs, sch, IDENT)
    assert ExpertStore.status(dirs, IDENT) == "incomplete"
    for layer in range(2):
        for e in range(6):
            w.write(layer, e, unit(sch, layer, e))
    w.close()
    w.finalize()
    assert ExpertStore.status(dirs, IDENT) == "complete"
    assert ExpertStore.status(dirs, "other") == "mismatch"


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg, *args, **kwargs) -> None:
        self.lines.append(msg % args if args else msg)

    warning = info


def test_each_read_path_says_so_once(tmp_path, monkeypatch):
    from vllm.model_executor.offloader import expert_store

    log = _Log()
    monkeypatch.setattr(expert_store, "logger", log)
    sch = schema()
    dirs = build(tmp_path, sch)
    with open_store(dirs, sch, preadv=Direct()) as s:
        make_direct(s)
        for e in range(3):
            s.read_into((0, e), dests(sch))
        assert len([x for x in log.lines if "straight into" in x]) == 1
        assert not [x for x in log.lines if "(bounce)" in x]
        s.read_into((0, 4), dests(sch, offset=64))
        s.read_into((0, 5), dests(sch, offset=64))
        assert len([x for x in log.lines if "(bounce)" in x]) == 1

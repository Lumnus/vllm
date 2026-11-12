# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the expert store's read modes (direct, buffered, bounce)."""

import ctypes
import errno
import os

import numpy as np
import pytest

from vllm.model_executor.offloader.expert_store import (
    ExpertStore,
    ExpertStoreError,
    ExpertStoreWriter,
    StoreSchema,
    TensorSpec,
)

IDENT = "modes/v1"
SCHEMA = StoreSchema.uniform(2, 6, [TensorSpec("w", "uint8", (10000,))])


def payload(layer: int, expert: int) -> bytes:
    rng = np.random.default_rng([layer, expert])
    return rng.integers(0, 256, 10000, dtype=np.uint8).tobytes()


@pytest.fixture
def dirs(tmp_path):
    d = tmp_path / "drive"
    d.mkdir()
    with ExpertStoreWriter([str(d)], SCHEMA, IDENT) as writer:
        for layer in range(2):
            for expert in range(6):
                writer.write(layer, expert, {"w": payload(layer, expert)})
        writer.finalize()
    return [str(d)]


class PinnedRefuser:
    """A ``preadv`` that fails with EFAULT for buffers inside the pool.

    ``refuse`` says which reads: ``"direct"`` (only on O_DIRECT descriptors) or
    ``"all"`` (buffered too), as the kernel does for memory it cannot take.
    """

    def __init__(self, refuse: str):
        self.refuse = refuse
        self.lo = self.hi = 0
        self.refused = 0

    def set_range(self, store: ExpertStore) -> None:
        arrays = {id(s.chunk): s.chunk.array for s in store._slots}.values()
        self.lo = min(a.ctypes.data for a in arrays)
        self.hi = max(a.ctypes.data + a.size for a in arrays)

    def __call__(self, fd, buffers, offset):
        view = buffers[0]
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
        direct = self._is_direct(fd)
        if self.lo <= address < self.hi and (
            self.refuse == "all" or (self.refuse == "direct" and direct)
        ):
            self.refused += 1
            raise OSError(errno.EFAULT, os.strerror(errno.EFAULT))
        return os.preadv(fd, buffers, offset)

    @staticmethod
    def _is_direct(fd: int) -> bool:
        import fcntl

        return bool(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT)


def open_store(dirs, refuser, mode="auto"):
    store = ExpertStore.open(
        dirs,
        IDENT,
        6 * 12288,
        io_threads=2,
        pin=False,
        read_mode=mode,
        preadv=refuser,
    )
    refuser.set_range(store)
    return store


def require_direct(store):
    if not store.stats()["direct"][0]:
        pytest.skip("this filesystem takes no O_DIRECT")


def read_all(store):
    for layer in range(2):
        for expert in range(6):
            with store.get((layer, expert)) as entry:
                assert bytes(entry.raw[:10000]) == payload(layer, expert)


def no_slot_leaked(store):
    stats = store.stats()
    assert stats["slots_leased"] == 0
    assert stats["slots_used"] == stats["slots_ready"]


def test_auto_ends_in_bounce_when_direct_and_buffered_are_refused(dirs):
    refuser = PinnedRefuser("all")
    with open_store(dirs, refuser) as store:
        read_all(store)
        stats = store.stats()
        assert stats["read_mode"] == "bounce"
        assert stats["reads"] == 12 and stats["reads_bounce"] == 12
        assert stats["read_errors"] == 0
        assert stats["read_fallbacks"] == {"EFAULT_direct": 1, "EFAULT_buffered": 1}
        assert refuser.refused >= 2
        no_slot_leaked(store)


def test_auto_settles_on_buffered_when_only_direct_is_refused(dirs):
    refuser = PinnedRefuser("direct")
    with open_store(dirs, refuser) as store:
        require_direct(store)
        read_all(store)
        stats = store.stats()
        assert stats["read_mode"] == "buffered"
        assert stats["reads_bounce"] == 0
        assert stats["read_fallbacks"] == {"EFAULT_direct": 1}
        assert stats["reads_direct"] == 0 and stats["reads_buffered"] == 12
        no_slot_leaked(store)


def test_auto_without_a_refusal_stays_direct(dirs):
    refuser = PinnedRefuser("none")
    with open_store(dirs, refuser) as store:
        read_all(store)
        stats = store.stats()
        assert stats["read_mode"] == "direct" and stats["read_fallbacks"] == {}


def test_forced_direct_surfaces_the_error_after_the_retries(dirs):
    refuser = PinnedRefuser("direct")
    with open_store(dirs, refuser, "direct") as store:
        require_direct(store)
        with pytest.raises(ExpertStoreError, match="errno 14"):
            store.get((0, 0))
        stats = store.stats()
        assert refuser.refused == 4  # one try and three retries
        assert stats["read_errors"] == 1 and stats["read_mode"] == "direct"
        assert stats["read_fallbacks"] == {}
        no_slot_leaked(store)
        assert stats["slots_used"] == 0  # the slot went back


def test_forced_buffered_and_bounce(dirs):
    refuser = PinnedRefuser("all")
    with open_store(dirs, refuser, "buffered") as store:
        with pytest.raises(ExpertStoreError, match="errno 14"):
            store.get((1, 1))
        no_slot_leaked(store)
    with open_store(dirs, refuser, "bounce") as store:  # works whatever is refused
        read_all(store)
        stats = store.stats()
        assert stats["reads_bounce"] == 12 and stats["read_fallbacks"] == {}
    clean = PinnedRefuser("none")
    with open_store(dirs, clean, "buffered") as store:
        read_all(store)
        assert store.stats()["reads_buffered"] == 12


def test_unknown_mode_is_refused(dirs):
    with pytest.raises(ValueError, match="read mode"):
        ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, read_mode="fast")

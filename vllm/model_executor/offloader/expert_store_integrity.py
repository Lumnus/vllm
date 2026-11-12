# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-unit checksums for the expert store.

Every unit gets a 64-bit keyed BLAKE3 digest when it is written. The digests of
a store are one flat table (``unit_checksums.bin``, 8 bytes per unit, indexed by
the unit's global index), the same bytes in every drive directory, with its
size and sha256 recorded in the manifest. The table is not part of the layout:
the stripe files, their offsets and the 4096-byte unit padding are the same
with and without it, so a store built before the table existed keeps its
stripes and can be given one later (:func:`expert_store_build.add_checksums`).

The reader checks a unit right after it was read into its pool slot and before
the slot is handed out. The digest is keyed with the unit index, so a unit read
from the wrong place fails as well.

This module imports nothing from ``expert_store`` (the store imports it).
"""

import hashlib
import itertools
import json
import os
import secrets
import threading
from typing import Any

import numpy as np

CHECKSUM_FILE = "unit_checksums.bin"
CHECKSUM_ALGO = "blake3-64-keyed"
_DTYPE = np.dtype("<u8")


def _blake3() -> Any:
    try:
        import blake3
    except ImportError as exc:  # pragma: no cover - blake3 is a common requirement
        raise ImportError(
            "per-unit checksums need the 'blake3' package (a vLLM requirement)"
        ) from exc
    return blake3.blake3


def unit_digest(data: Any, index: int, threads: int = 1) -> int:
    """64-bit digest of ``data`` keyed by the unit's global ``index``.

    ``threads`` is BLAKE3's own worker count for this one call (1 = the calling
    thread only, 0 = as many as it likes); the call releases the GIL.
    """
    hasher = _blake3()
    key = int(index).to_bytes(32, "little")
    max_threads = hasher.AUTO if threads == 0 else int(threads)
    return int.from_bytes(
        hasher(data, key=key, max_threads=max_threads).digest(8), "little"
    )


def unit_digest_parts(parts: Any, index: int, threads: int = 1) -> int:
    """:func:`unit_digest` of the concatenation of ``parts`` (bytes-likes),
    without joining them: a unit read as separate tensors plus its padding."""
    hasher = _blake3()
    key = int(index).to_bytes(32, "little")
    max_threads = hasher.AUTO if threads == 0 else int(threads)
    state = hasher(key=key, max_threads=max_threads)
    for part in parts:
        state.update(part)
    return int.from_bytes(state.digest(8), "little")


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class ChecksumLog:
    """What one writer knows: the digest of every unit it wrote.

    Saved next to the writer's record (``writers/<id>.crc``) before the record
    that claims those units, so a record never names a unit without a digest.
    """

    def __init__(self, total_units: int):
        self.values = np.zeros(total_units, dtype=_DTYPE)

    def set(self, index: int, value: int) -> None:
        self.values[index] = value

    def get(self, index: int) -> int:
        return int(self.values[index])

    def save(self, path: str) -> None:
        _write_atomic(path, self.values.tobytes())

    def load(self, path: str) -> bool:
        """Load a saved log; False if there is none."""
        try:
            values = np.fromfile(path, dtype=_DTYPE)
        except FileNotFoundError:
            return False
        if values.size != self.values.size:
            raise ValueError(
                f"{path}: {values.size} digests, expected {self.values.size}"
            )
        self.values = values
        return True


def merge_logs(folder: str, total_units: int) -> np.ndarray:
    """One table from every writer record in ``folder`` (disjoint units)."""
    table = np.zeros(total_units, dtype=_DTYPE)
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(folder, name), "rb") as handle:
            runs = json.loads(handle.read())["runs"]
        if not runs:
            continue
        log = ChecksumLog(total_units)
        if not log.load(os.path.join(folder, name[: -len(".json")] + ".crc")):
            raise ValueError(f"writer record {name} has no checksum log")
        for start, end in runs:
            table[start:end] = log.values[start:end]
    return table


def publish_table(dirs: list[str], table: np.ndarray) -> dict[str, Any]:
    """Write the table into every directory; the manifest entry describing it."""
    payload = np.ascontiguousarray(table, dtype=_DTYPE).tobytes()
    for directory in dirs:
        _write_atomic(os.path.join(directory, CHECKSUM_FILE), payload)
    return {
        "algo": CHECKSUM_ALGO,
        "file": CHECKSUM_FILE,
        "count": int(table.size),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_table(directory: str, info: dict[str, Any]) -> np.ndarray:
    """The table of a store, checked against the manifest entry ``info``.

    Raises ``ValueError`` (message says what) if it is missing, of another
    algorithm, or does not match its recorded size or hash.
    """
    if info.get("algo") != CHECKSUM_ALGO:
        raise ValueError(f"checksum algorithm {info.get('algo')!r} is not supported")
    path = os.path.join(directory, str(info.get("file", CHECKSUM_FILE)))
    try:
        with open(path, "rb") as handle:
            payload = handle.read()
    except FileNotFoundError:
        raise ValueError(f"{path} is missing") from None
    if len(payload) != 8 * int(info["count"]):
        raise ValueError(f"{path}: {len(payload)} B, manifest says {8 * info['count']}")
    if hashlib.sha256(payload).hexdigest() != info["sha256"]:
        raise ValueError(f"{path}: sha256 does not match the manifest")
    return np.frombuffer(payload, dtype=_DTYPE)


class UnitChecker:
    """Checks units against the table, every read or every Nth read.

    ``every=1`` checks every read, ``every=N`` the 1st, (N+1)th, ... read.
    Thread-safe.
    """

    def __init__(self, table: np.ndarray, every: int, threads: int = 4):
        if every < 1:
            raise ValueError("every must be >= 1")
        self.table = table
        self.every = every
        self.threads = threads
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self.checked = 0
        self.failed = 0

    def due(self) -> bool:
        return next(self._counter) % self.every == 0

    def check(self, index: int, data: Any) -> int | None:
        """None if ``data`` is the unit ``index`` as written, else its digest."""
        got = unit_digest(data, index, self.threads)
        ok = got == int(self.table[index])
        with self._lock:
            self.checked += 1
            if not ok:
                self.failed += 1
        return None if ok else got

    def check_parts(self, index: int, parts: Any) -> int | None:
        """:meth:`check` for a unit held as consecutive pieces."""
        got = unit_digest_parts(parts, index, self.threads)
        ok = got == int(self.table[index])
        with self._lock:
            self.checked += 1
            if not ok:
                self.failed += 1
        return None if ok else got

    def expected(self, index: int) -> int:
        return int(self.table[index])

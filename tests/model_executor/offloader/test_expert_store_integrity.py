# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the expert store's per-unit checksums."""

import json
import os
import pathlib

import pytest
import torch

from vllm.model_executor.offloader import expert_store_integrity as integrity
from vllm.model_executor.offloader.expert_store import (
    DATA_FILE,
    MANIFEST_FILE,
    ExpertStore,
    ExpertStoreWriter,
    LayerGroup,
    StoreCorruptError,
    StoreMismatchError,
    StoreSchema,
    TensorSpec,
    _Layout,
)
from vllm.model_executor.offloader.expert_store_build import add_checksums

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
    return StoreSchema(
        4,
        6,
        (
            LayerGroup(0, 1, (TensorSpec("a", "float32", (33, 7)),)),
            LayerGroup(1, 3, (TensorSpec("b", "int16", (100,)),)),
        ),
    )


def make(spec: TensorSpec, layer: int, expert: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(layer * 100003 + expert * 31 + len(spec.name))
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


def build(dirs: list[str], schema: StoreSchema) -> None:
    with ExpertStoreWriter(dirs, schema, IDENT) as writer:
        for layer in range(schema.layers):
            for expert in range(schema.experts_per_layer):
                writer.write(layer, expert, unit_tensors(schema, layer, expert))
        writer.finalize()


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
        assert torch.equal(got[name].view(torch.uint8), tensor.view(torch.uint8))


def flip(dirs: list[str], schema, key, where: str = "payload") -> None:
    """Flip one bit of a unit on the drive (``where``: payload or padding)."""
    layout = _Layout(schema, len(dirs), IDENT)
    stripe, offset, unit_bytes = layout.locate(*key)
    group = layout.group_of(key[0])
    pos = offset + (3 if where == "payload" else group.payload_bytes + 1)
    if where == "padding":
        assert unit_bytes > group.payload_bytes, "schema has no padding to flip"
    with open(os.path.join(dirs[stripe], DATA_FILE), "r+b") as handle:
        handle.seek(pos)
        byte = handle.read(1)[0]
        handle.seek(pos)
        handle.write(bytes([byte ^ 0x10]))


def read_manifest(directory: str) -> dict:
    return json.loads(pathlib.Path(directory, MANIFEST_FILE).read_text())


def test_table_is_published_and_clean_store_passes(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema)
    for d in dirs:
        manifest = read_manifest(d)
        info = manifest["unit_checksums"]
        assert manifest["format"] == "expert-store/v1"
        assert info["algo"] == integrity.CHECKSUM_ALGO
        assert info["count"] == 15
        assert os.path.getsize(os.path.join(d, integrity.CHECKSUM_FILE)) == 15 * 8
    with ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, check_every=1) as store:
        for layer in range(3):
            for expert in range(5):
                with store.get((layer, expert)) as entry:
                    check_unit(entry, schema, layer, expert)
        stats = store.stats()
    assert stats["checked"] == stats["reads"] == 15
    assert stats["check_failures"] == 0


def test_check_off_does_no_work(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    with ExpertStore.open(dirs, IDENT, 1 << 20, pin=False) as store:
        assert store._integrity is None
        with store.get((1, 1)) as entry:
            check_unit(entry, schema, 1, 1)
        stats = store.stats()
    assert stats["checked"] == 0


@pytest.mark.parametrize("stripes", [1, 2])
def test_every_read_catches_a_bit_flip_and_hands_out_no_slot(tmp_path, stripes):
    schema = schema_a()
    dirs = make_dirs(tmp_path, stripes)
    build(dirs, schema)
    flip(dirs, schema, (1, 2))
    with ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, check_every=1) as store:
        with store.get((1, 1)) as entry:  # a clean neighbour is unaffected
            check_unit(entry, schema, 1, 1)
        with pytest.raises(StoreCorruptError, match=r"\(1, 2\)"):
            store.get((1, 2))
        stats = store.stats()
        assert stats["read_errors"] == 1
        assert stats["check_failures"] == 1
        assert not store.resident((1, 2))
        # the slot went back: every slot is free or holds a clean unit
        assert stats["slots_used"] == stats["slots_ready"] == 1
        assert stats["slots_leased"] == 0
        # asking again reads and fails again, it is not cached as good
        with pytest.raises(StoreCorruptError):
            store.get((1, 2))


def test_padding_corruption_is_caught_too(tmp_path):
    schema = schema_b()  # a 1-layer group of float32 33x7 pads to the next page
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    flip(dirs, schema, (0, 3), where="padding")
    with (
        ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, check_every=1) as store,
        pytest.raises(StoreCorruptError, match=r"\(0, 3\)"),
    ):
        store.get((0, 3))


def test_sampled_mode_checks_every_nth_read(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    flip(dirs, schema, (0, 1))  # read number 1: not sampled with N=3
    flip(dirs, schema, (0, 3))  # read number 3: sampled
    with ExpertStore.open(
        dirs, IDENT, 1 << 20, pin=False, io_threads=1, check_every=3
    ) as store:
        with store.get((0, 0)) as entry:  # read 0: checked, clean
            check_unit(entry, schema, 0, 0)
        with store.get((0, 1)) as entry:  # read 1: not checked, silently wrong
            want = unit_tensors(schema, 0, 1)
            assert not torch.equal(entry.tensors()["w13"], want["w13"])
        with store.get((0, 2)) as entry:  # read 2: not checked, clean
            check_unit(entry, schema, 0, 2)
        with pytest.raises(StoreCorruptError, match=r"\(0, 3\)"):  # read 3
            store.get((0, 3))
        stats = store.stats()
    assert stats["reads"] == 3 and stats["read_errors"] == 1  # the failed read
    assert stats["checked"] == 2
    assert stats["check_failures"] == 1


def test_off_mode_does_not_notice_but_verify_does(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    flip(dirs, schema, (2, 4))
    with ExpertStore.open(dirs, IDENT, 1 << 20, pin=False) as store:
        store.get((2, 4)).release()  # no error: the read path does no extra work
    with pytest.raises(StoreCorruptError):
        ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, verify=True)


def test_digest_is_keyed_by_the_unit_index():
    data = bytes(range(256)) * 16
    assert integrity.unit_digest(data, 3) == integrity.unit_digest(data, 3)
    assert integrity.unit_digest(data, 3) != integrity.unit_digest(data, 4)
    assert integrity.unit_digest(data, 3, threads=1) == integrity.unit_digest(
        data, 3, threads=4
    )


def test_checker_sampling_counts():
    import numpy as np

    table = np.zeros(4, dtype="<u8")
    checker = integrity.UnitChecker(table, every=4)
    assert [checker.due() for _ in range(9)] == [
        True, False, False, False, True, False, False, False, True,
    ]  # fmt: skip
    with pytest.raises(ValueError):
        integrity.UnitChecker(table, every=0)


def test_tampered_table_is_refused(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema)
    path = pathlib.Path(dirs[0], integrity.CHECKSUM_FILE)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(bytes(raw))
    # off: the table is not read at all
    ExpertStore.open(dirs, IDENT, 1 << 20, pin=False).close()
    with pytest.raises(StoreCorruptError, match="sha256"):
        ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, check_every=1)


def strip_checksums(dirs: list[str]) -> None:
    """Make a store look as one written before unit checksums existed: the
    stripes and the layout are the same, there is no table and no manifest
    entry for it."""
    for d in dirs:
        os.unlink(os.path.join(d, integrity.CHECKSUM_FILE))
        doc = read_manifest(d)
        del doc["unit_checksums"]
        pathlib.Path(d, MANIFEST_FILE).write_text(json.dumps(doc, indent=1))


def test_store_without_a_table_still_opens_and_says_what_to_do(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema)
    strip_checksums(dirs)
    with (
        ExpertStore.open(dirs, IDENT, 1 << 20, pin=False) as store,
        store.get((1, 3)) as entry,
    ):
        check_unit(entry, schema, 1, 3)
    with pytest.raises(StoreMismatchError, match="checksums"):
        ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, check_every=1)
    with pytest.raises(StoreMismatchError, match="rebuild"):
        ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, check_every=1)


def test_checksums_can_be_added_to_an_older_store(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 2)
    build(dirs, schema)
    strip_checksums(dirs)
    stripe_before = [pathlib.Path(d, DATA_FILE).read_bytes() for d in dirs]
    assert add_checksums(dirs) == 15
    # the stripes did not change, the layout hash is the same
    assert [pathlib.Path(d, DATA_FILE).read_bytes() for d in dirs] == stripe_before
    flip(dirs, schema, (0, 0))
    with ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, check_every=1) as store:
        with store.get((0, 1)) as entry:
            check_unit(entry, schema, 0, 1)
        with pytest.raises(StoreCorruptError):
            store.get((0, 0))


def test_resume_keeps_the_digests_of_the_first_run(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 2)
    first = ExpertStoreWriter(dirs, schema, IDENT, writer_id="w")
    for expert in range(5):
        first.write(0, expert, unit_tensors(schema, 0, expert))
    first.flush()
    del first  # as if the process died after the flush
    with ExpertStoreWriter(dirs, schema, IDENT, writer_id="w") as writer:
        assert writer.has_unit(0, 4) and not writer.has_unit(1, 0)
        for layer in (1, 2):
            for expert in range(5):
                writer.write(layer, expert, unit_tensors(schema, layer, expert))
        writer.finalize()
    with ExpertStore.open(dirs, IDENT, 1 << 20, pin=False, check_every=1) as store:
        for layer in range(3):
            for expert in range(5):
                with store.get((layer, expert)) as entry:
                    check_unit(entry, schema, layer, expert)
        assert store.stats()["check_failures"] == 0


def test_resume_of_a_record_without_digests_is_refused(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 1)
    writer = ExpertStoreWriter(dirs, schema, IDENT, writer_id="old")
    writer.write(0, 0, unit_tensors(schema, 0, 0))
    writer.flush()
    del writer
    os.unlink(os.path.join(dirs[0], "writers", "old.crc"))  # a pre-digest record
    with pytest.raises(StoreMismatchError, match="rebuild"):
        ExpertStoreWriter(dirs, schema, IDENT, writer_id="old")
    ExpertStoreWriter(dirs, schema, IDENT, writer_id="old", rebuild=True).close()

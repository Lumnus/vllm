# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert store's settings changed while it serves: unit checks and the
stats log interval."""

import os

import pytest
import torch

from vllm.config.expert_store import ExpertStoreConfig
from vllm.model_executor.offloader.expert_store import (
    DATA_FILE,
    ExpertStore,
    ExpertStoreWriter,
    StoreCorruptError,
    StoreSchema,
    TensorSpec,
    _Layout,
)
from vllm.model_executor.offloader.expert_tier_store import StoreSession

IDENT = "ckpt-abc/kernel-v1"


def schema_a() -> StoreSchema:
    return StoreSchema.uniform(
        3, 5, [TensorSpec("w13", "int8", (8, 16)), TensorSpec("odd", "uint8", (5,))]
    )


def build(dirs: list[str], schema: StoreSchema) -> None:
    with ExpertStoreWriter(dirs, schema, IDENT) as writer:
        for layer in range(schema.layers):
            for expert in range(schema.experts_per_layer):
                gen = torch.Generator().manual_seed(layer * 101 + expert)
                writer.write(
                    layer,
                    expert,
                    {
                        "w13": torch.randint(-100, 100, (8, 16), generator=gen).to(
                            torch.int8
                        ),
                        "odd": torch.randint(0, 255, (5,), generator=gen).to(
                            torch.uint8
                        ),
                    },
                )
        writer.finalize()


def make_dirs(tmp_path, n: int) -> list[str]:
    dirs = []
    for i in range(n):
        d = tmp_path / f"drive{i}"
        d.mkdir()
        dirs.append(str(d))
    return dirs


def flip(dirs: list[str], schema: StoreSchema, key) -> None:
    """Flip one bit of a unit's payload on the drive."""
    layout = _Layout(schema, len(dirs), IDENT)
    stripe, offset, _ = layout.locate(*key)
    with open(os.path.join(dirs[stripe], DATA_FILE), "r+b") as handle:
        handle.seek(offset + 3)
        byte = handle.read(1)[0]
        handle.seek(offset + 3)
        handle.write(bytes([byte ^ 0x10]))


def test_unit_checks_turn_on_and_off_on_an_open_store(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    flip(dirs, schema, (1, 2))
    with ExpertStore.open(dirs, IDENT, 1 << 20, pin=False) as store:
        assert store.check_every == 0 and store.can_check_units()
        with store.get((1, 2)):  # unchecked: handed out
            pass
        store.set_check_every(1)
        assert store.check_every == 1
        with store.get((0, 0)):
            pass
        assert store.stats()["checked"] == 1
        store.set_check_every(5)
        assert store.check_every == 5
        store.set_check_every(0)
        assert store.check_every == 0
        with pytest.raises(ValueError):
            store.set_check_every(-1)
    with ExpertStore.open(dirs, IDENT, 1 << 20, pin=False) as store:
        store.set_check_every(1)
        with pytest.raises(StoreCorruptError, match=r"\(1, 2\)"):
            store.get((1, 2))


def test_the_stats_line_interval_changes_while_open(tmp_path):
    schema = schema_a()
    dirs = make_dirs(tmp_path, 1)
    build(dirs, schema)
    config = ExpertStoreConfig(dirs=dirs, pool_gib=0.001)
    session = StoreSession(
        config, schema, IDENT, rank=0, num_ranks=1, barrier=lambda: None, pin=False
    )
    try:
        session.set_stats_interval(5.0)  # not open yet: only the setting
        assert session._stats_stop is None and config.stats_interval_s == 5.0
        session.open_reader()
        first = session._stats_stop
        assert first is not None
        session.set_stats_interval(10.0)
        assert first.is_set() and session._stats_stop is not None
        assert session._stats_stop is not first
        session.set_stats_interval(0)
        assert session._stats_stop is None and config.stats_interval_s == 0
    finally:
        session.close()

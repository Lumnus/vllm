# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 PLE table file: builder, reader and load-time checks (CPU)."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm.models.qwen4_exp.common import ple_int8_table
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLENvmeEmbedding
from vllm.models.qwen4_exp.common.ple_int8 import packed_row_bytes, quantize_rows
from vllm.models.qwen4_exp.common.ple_nvme import (
    PleNvmeRowStore,
    safetensors_tensor_location,
)

PREFIX = "model.language_model.layers.1.ple.ple_embedding"
DIM = 16
ROWS = 1234
SHARDS = 5


def _table() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    table = torch.randn(ROWS, DIM, generator=generator) * 0.02
    table[0] = 0.0
    return table.to(torch.bfloat16)


def _layout() -> dict[str, torch.Tensor]:
    return {
        "ngram_heads_offsets": torch.tensor([0, 600], dtype=torch.int64),
        "ngram_heads_vocab_sizes": torch.tensor([600, 634], dtype=torch.int64),
        "layer_multipliers": torch.tensor([3, 5, 7], dtype=torch.int64),
    }


def _write_checkpoint(model_dir, table: torch.Tensor, prefix: str = PREFIX) -> None:
    """A two-file sharded checkpoint with the table split into SHARDS parts."""
    os.makedirs(model_dir, exist_ok=True)
    shard_rows = (ROWS + SHARDS - 1) // SHARDS
    tensors = {
        f"{prefix}.ngram_embedding.shard_{i}.weight": table[
            i * shard_rows : (i + 1) * shard_rows
        ].contiguous()
        for i in range(SHARDS)
    }
    tensors.update({f"{prefix}.{k}": v for k, v in _layout().items()})
    names = sorted(tensors)
    files = {"a.safetensors": names[::2], "b.safetensors": names[1::2]}
    weight_map = {}
    for file, keys in files.items():
        save_file({k: tensors[k] for k in keys}, os.path.join(model_dir, file))
        weight_map.update({k: file for k in keys})
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def test_build_writes_quantized_rows_aligned_and_verifies(tmp_path) -> None:
    table = _table()
    _write_checkpoint(tmp_path / "ckpt", table)
    out = str(tmp_path / "ple.safetensors")
    written = ple_int8_table.build(str(tmp_path / "ckpt"), out, chunk_rows=100)
    assert written == [out]

    info = ple_int8_table.table_info(out, DIM)
    assert info.data_start % ple_int8_table.DATA_ALIGN == 0
    assert (info.num_rows, info.row_bytes) == (ROWS, packed_row_bytes(DIM))
    assert info.metadata["format"] == ple_int8_table.FORMAT
    for key, value in _layout().items():
        assert np.array_equal(info.layout[key], value.numpy())
    rows = np.asarray(ple_int8_table.map_rows(info))
    assert np.array_equal(rows, quantize_rows(table).numpy())
    # A standard safetensors reader agrees on the layout.
    from safetensors import safe_open

    with safe_open(out, framework="pt") as handle:
        assert torch.equal(handle.get_tensor("table"), quantize_rows(table))

    result = ple_int8_table.verify(str(tmp_path / "ckpt"), out, sample=300)
    assert result["rows_mismatched"] == 0 and result["rows_checked"] > 0
    with pytest.raises(FileExistsError):
        ple_int8_table.build(str(tmp_path / "ckpt"), out)


def test_verify_detects_a_foreign_table(tmp_path) -> None:
    _write_checkpoint(tmp_path / "ckpt", _table())
    _write_checkpoint(tmp_path / "other", (_table().float() * 2).to(torch.bfloat16))
    out = str(tmp_path / "other.safetensors")
    ple_int8_table.build(str(tmp_path / "other"), out)
    with pytest.raises(ValueError, match="differ"):
        ple_int8_table.verify(str(tmp_path / "ckpt"), out, sample=64)


def test_table_info_rejects_unknown_format_and_width(tmp_path) -> None:
    path = str(tmp_path / "x.safetensors")
    save_file(
        {"table": torch.zeros(4, DIM + 4, dtype=torch.uint8)},
        path,
        metadata={"format": "something-else"},
    )
    with pytest.raises(ValueError, match="format"):
        ple_int8_table.table_info(path)
    legacy = str(tmp_path / "legacy.safetensors")
    save_file(
        {"table": torch.zeros(4, DIM + 4, dtype=torch.uint8)},
        legacy,
        metadata={"format": ple_int8_table.LEGACY_FORMATS[0]},
    )
    assert ple_int8_table.table_info(legacy, DIM).num_rows == 4
    with pytest.raises(ValueError, match="rows are"):
        ple_int8_table.table_info(legacy, DIM + 1)


def test_layer_placeholder_names_one_file_per_layer(tmp_path) -> None:
    table = _table()
    _write_checkpoint(tmp_path / "ckpt", table)
    _write_checkpoint(
        tmp_path / "ckpt2", table, prefix="model.language_model.layers.7.ple.ple_emb"
    )
    # Merge the second checkpoint's files into the first.
    with open(tmp_path / "ckpt" / "model.safetensors.index.json") as f:
        index = json.load(f)
    with open(tmp_path / "ckpt2" / "model.safetensors.index.json") as f:
        other = json.load(f)["weight_map"]
    for file in set(other.values()):
        os.replace(tmp_path / "ckpt2" / file, tmp_path / "ckpt" / f"l7-{file}")
    index["weight_map"].update({k: f"l7-{v}" for k, v in other.items()})
    with open(tmp_path / "ckpt" / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    with pytest.raises(ValueError, match="layer"):
        ple_int8_table.build(str(tmp_path / "ckpt"), str(tmp_path / "one.safetensors"))
    written = ple_int8_table.build(
        str(tmp_path / "ckpt"), str(tmp_path / "ple-{layer}.safetensors")
    )
    assert sorted(os.path.basename(p) for p in written) == [
        "ple-1.safetensors",
        "ple-7.safetensors",
    ]


def test_nvme_reader_rows_equal_the_file(tmp_path) -> None:
    table = _table()
    _write_checkpoint(tmp_path / "ckpt", table)
    out = str(tmp_path / "ple.safetensors")
    ple_int8_table.build(str(tmp_path / "ckpt"), out)
    data_start, shape, dtype = safetensors_tensor_location(out, "table")
    assert dtype == "U8" and shape == [ROWS, packed_row_bytes(DIM)]
    try:
        store = PleNvmeRowStore(out, data_start, shape[1], shape[0], io_threads=4)
    except OSError as exc:  # e.g. tmpfs without O_DIRECT
        pytest.skip(f"O_DIRECT unavailable here: {exc}")
    try:
        rows = np.array([0, 1, 24, 25, 500, ROWS - 1])  # incl. page-crossing rows
        assert np.array_equal(store.read_rows(rows), quantize_rows(table)[rows].numpy())
    finally:
        store.close()


def _fake_nvme_embedding(path: str, tp_start: int, tp_end: int, verify_rows: int):
    fake = SimpleNamespace(
        embedding_dim=DIM,
        org_vocab_size=ROWS,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=tp_start, org_vocab_end_index=tp_end
        ),
        _verify_rows=verify_rows,
        verified_rows=0,
        _table_rows=None,
        table=None,
    )
    Qwen4ExpPLENvmeEmbedding.open_table(fake, path)
    return fake


def test_nvme_embedding_checks_checkpoint_shards_against_the_file(tmp_path) -> None:
    table = _table()
    _write_checkpoint(tmp_path / "ckpt", table)
    out = str(tmp_path / "ple.safetensors")
    ple_int8_table.build(str(tmp_path / "ckpt"), out)
    fake = _fake_nvme_embedding(out, 300, 900, verify_rows=16)
    check = Qwen4ExpPLENvmeEmbedding._check_checkpoint_rows
    for start in range(0, ROWS, 250):
        check(fake, None, table[start : start + 250], checkpoint_start=start)
    assert fake.verified_rows > 0

    wrong = table.clone()
    wrong[250:500] = wrong[250:500] * 3
    with pytest.raises(ValueError, match="does not match"):
        check(fake, None, wrong[250:500], checkpoint_start=250)
    # A shard outside this rank's rows is not checked.
    check(fake, None, wrong[1000:1234], checkpoint_start=1000)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-disk INT8 PLE table: format, reader helpers and the offline builder.

The NVMe-backed PLE table (``ple_nvme.py``) reads rows straight from a file.
The file is a regular ``.safetensors`` file with a documented layout, so it
can be inspected with standard tools:

* ``__metadata__["format"]`` is ``FORMAT`` (``LEGACY_FORMATS`` are accepted
  when reading);
* tensor ``table`` is ``U8 [rows, dim + 4]``: packed rows exactly as produced
  by ``ple_int8.quantize_rows`` (``dim`` int8 values, then the float32 scale,
  little-endian). Row ``r`` starts at ``data_start + r * (dim + 4)``;
* the header is padded with spaces so that ``data_start`` (= 8 + header
  length) is a multiple of 4096. The reader does not need this (it computes
  page-aligned reads for any offset) but it keeps every row inside at most
  two pages;
* optional I64 tensors ``ngram_heads_offsets``, ``ngram_heads_vocab_sizes``
  and ``layer_multipliers`` copied from the checkpoint, so a server can
  check the file belongs to the model it serves.

Rows are quantized with the same function the in-memory INT8 loader uses, so
the bytes on disk equal the bytes an in-memory INT8 table would hold.

Build (streams the checkpoint shards; memory stays bounded):

    python -m vllm.models.qwen4_exp.common.ple_int8_table build \\
        --model /path/to/checkpoint --out /nvme/ple_int8.safetensors
    python -m vllm.models.qwen4_exp.common.ple_int8_table verify \\
        --model /path/to/checkpoint --table /nvme/ple_int8.safetensors

Checkpoints with several PLE layers produce one file per layer; pass
``--out`` with a ``{layer}`` placeholder (the layer's module prefix index).

This module depends on torch, numpy and safetensors only.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import torch

from .ple_int8 import SCALE_BYTES, packed_row_bytes, quantize_rows

FORMAT = "qwen4exp-ple-int8-rowscale/v1"
LEGACY_FORMATS = ("lumnus-ple-int8-rowscale/v1",)
ACCEPTED_FORMATS = (FORMAT, *LEGACY_FORMATS)
ROW_LAYOUT = (
    "dim x int8 q | float32 LE s; s = absmax(row)/127, "
    "q = round_half_even(x/s) clamped to +-127 (0 when s == 0); x ~= q*s"
)
LAYOUT_KEYS = ("ngram_heads_offsets", "ngram_heads_vocab_sizes", "layer_multipliers")
DATA_ALIGN = 4096

_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)\.ngram_embedding\.shard_(?P<index>\d+)\.weight$"
)
_WHOLE_RE = re.compile(r"^(?P<prefix>.+)\.ngram_embedding\.weight$")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


@dataclass(frozen=True)
class TableInfo:
    """Where the packed rows of a table file are, and what came with them."""

    path: str
    data_start: int
    num_rows: int
    row_bytes: int
    metadata: dict[str, str]
    layout: dict[str, np.ndarray]


def read_header(path: str) -> tuple[dict, int]:
    """(safetensors header, absolute byte offset of the data section)."""
    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


def table_info(path: str, dim: int | None = None) -> TableInfo:
    """Validate a table file and locate its rows (and layout tensors)."""
    header, data_section = read_header(path)
    metadata = {str(k): str(v) for k, v in (header.get("__metadata__") or {}).items()}
    fmt = metadata.get("format")
    if fmt not in ACCEPTED_FORMATS:
        raise ValueError(
            f"{path}: format {fmt!r} is not an INT8 PLE table "
            f"(expected one of {ACCEPTED_FORMATS})"
        )
    entry = header.get("table")
    if entry is None or entry.get("dtype") != "U8" or len(entry["shape"]) != 2:
        raise ValueError(f"{path}: needs a 2D U8 'table' tensor, got {entry}")
    num_rows, row_bytes = (int(v) for v in entry["shape"])
    if dim is not None and row_bytes != packed_row_bytes(dim):
        raise ValueError(
            f"{path}: rows are {row_bytes} B, expected {packed_row_bytes(dim)} "
            f"({dim} int8 values + {SCALE_BYTES} scale bytes)"
        )
    data_start = data_section + int(entry["data_offsets"][0])
    layout: dict[str, np.ndarray] = {}
    with open(path, "rb") as handle:
        for key in LAYOUT_KEYS:
            info = header.get(key)
            if info is None:
                continue
            if info.get("dtype") != "I64":
                raise ValueError(f"{path}: {key} must be I64, got {info.get('dtype')}")
            begin, end = (int(v) for v in info["data_offsets"])
            handle.seek(data_section + begin)
            values = np.frombuffer(handle.read(end - begin), dtype="<i8")
            layout[key] = values.reshape([int(v) for v in info["shape"]]).copy()
    return TableInfo(path, data_start, num_rows, row_bytes, metadata, layout)


def map_rows(info: TableInfo) -> np.ndarray:
    """Read-only memory map of the packed rows, uint8 [num_rows, row_bytes]."""
    return np.memmap(
        info.path,
        dtype=np.uint8,
        mode="r",
        offset=info.data_start,
        shape=(info.num_rows, info.row_bytes),
    )


# ---------------------------------------------------------------------------
# Offline builder
# ---------------------------------------------------------------------------


def _weight_map(model_dir: str) -> dict[str, str]:
    """Tensor name -> file, for a (possibly sharded) safetensors checkpoint."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as handle:
            mapping = json.load(handle)["weight_map"]
        return {name: os.path.join(model_dir, file) for name, file in mapping.items()}
    from safetensors import safe_open

    mapping = {}
    for file in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(file, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                mapping[name] = file
    return mapping


def find_ple_tables(model_dir: str) -> dict[str, list[tuple[str, str]]]:
    """PLE n-gram module prefix -> its table tensors [(name, file)], in row order."""
    tables: dict[str, list[tuple[int, str, str]]] = {}
    for name, file in _weight_map(model_dir).items():
        match = _SHARD_RE.match(name)
        if match:
            tables.setdefault(match["prefix"], []).append(
                (int(match["index"]), name, file)
            )
            continue
        match = _WHOLE_RE.match(name)
        if match:
            tables.setdefault(match["prefix"], []).append((-1, name, file))
    result = {}
    for prefix, parts in tables.items():
        parts.sort()
        indices = [index for index, _, _ in parts]
        if indices != [-1] and indices != list(range(len(indices))):
            raise ValueError(f"{prefix}: table shards are not contiguous: {indices}")
        result[prefix] = [(name, file) for _, name, file in parts]
    return result


def _layout_tensors(model_dir: str, prefix: str) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    mapping = _weight_map(model_dir)
    layout = {}
    for key in LAYOUT_KEYS:
        name = f"{prefix}.{key}"
        if name not in mapping:
            continue
        with safe_open(mapping[name], framework="pt") as handle:
            layout[key] = handle.get_tensor(name).to(torch.int64).numpy()
    return layout


def _iter_table_chunks(parts: list[tuple[str, str]], chunk_rows: int):
    """Yield the checkpoint table in row order, ``chunk_rows`` rows at a time."""
    from safetensors import safe_open

    for name, file in parts:
        with safe_open(file, framework="pt") as handle:
            view = handle.get_slice(name)
            rows = int(view.get_shape()[0])
            for start in range(0, rows, chunk_rows):
                yield view[start : min(rows, start + chunk_rows)]


def _table_shape(parts: list[tuple[str, str]]) -> tuple[int, int]:
    from safetensors import safe_open

    rows, dim = 0, None
    for name, file in parts:
        with safe_open(file, framework="pt") as handle:
            shape = handle.get_slice(name).get_shape()
        if len(shape) != 2 or (dim is not None and shape[1] != dim):
            raise ValueError(f"{name}: unexpected table shard shape {shape}")
        rows += int(shape[0])
        dim = int(shape[1])
    assert dim is not None
    return rows, dim


def _header_bytes(num_rows: int, dim: int, layout, metadata) -> bytes:
    row_bytes = packed_row_bytes(dim)
    header: dict = {
        "__metadata__": metadata,
        "table": {
            "dtype": "U8",
            "shape": [num_rows, row_bytes],
            "data_offsets": [0, num_rows * row_bytes],
        },
    }
    position = num_rows * row_bytes
    for key in LAYOUT_KEYS:
        if key in layout:
            size = layout[key].size * 8
            header[key] = {
                "dtype": "I64",
                "shape": list(layout[key].shape),
                "data_offsets": [position, position + size],
            }
            position += size
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * ((-(8 + len(text))) % DATA_ALIGN)
    return len(text).to_bytes(8, "little") + text


def write_table(
    out_path: str,
    chunks,
    num_rows: int,
    dim: int,
    layout: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> str:
    """Quantize ``chunks`` (float [n, dim] tensors, row order) into ``out_path``.

    Writes ``out_path + '.part'``, fsyncs and renames. Returns the sha256 of
    the file. Refuses to overwrite an existing file.
    """
    if os.path.exists(out_path):
        raise FileExistsError(out_path)
    metadata = {"format": FORMAT, "row_layout": ROW_LAYOUT, **metadata}
    header = _header_bytes(num_rows, dim, layout, metadata)
    digest = hashlib.sha256(header)
    part = out_path + ".part"
    written_rows = 0
    with open(part, "wb") as handle:
        handle.write(header)
        for chunk in chunks:
            packed = quantize_rows(chunk).numpy().tobytes()
            handle.write(packed)
            digest.update(packed)
            written_rows += int(chunk.shape[0])
        if written_rows != num_rows:
            raise ValueError(f"wrote {written_rows} rows, expected {num_rows}")
        for key in LAYOUT_KEYS:
            if key in layout:
                data = np.ascontiguousarray(layout[key], dtype="<i8").tobytes()
                handle.write(data)
                digest.update(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, out_path)
    return digest.hexdigest()


def _out_path(template: str, prefix: str, multiple: bool) -> str:
    if "{layer}" in template:
        match = _LAYER_RE.search(prefix + ".")
        if match is None:
            raise ValueError(f"cannot find a layer index in {prefix!r}")
        return template.format(layer=match.group(1))
    if multiple:
        raise ValueError("several PLE tables found; put {layer} in --out")
    return template


def build(model_dir: str, out: str, *, chunk_rows: int = 1 << 20) -> list[str]:
    tables = find_ple_tables(model_dir)
    if not tables:
        raise ValueError(f"no PLE n-gram tables found in {model_dir}")
    written = []
    for prefix, parts in sorted(tables.items()):
        path = _out_path(out, prefix, len(tables) > 1)
        num_rows, dim = _table_shape(parts)
        layout = _layout_tensors(model_dir, prefix)
        sha = write_table(
            path,
            _iter_table_chunks(parts, chunk_rows),
            num_rows,
            dim,
            layout,
            {"source_prefix": prefix, "source_tensors": str(len(parts))},
        )
        print(f"{path}: {num_rows} rows x {packed_row_bytes(dim)} B, sha256 {sha}")
        written.append(path)
    return written


def verify(model_dir: str, table: str, *, sample: int = 65536, seed: int = 0) -> dict:
    """Re-quantize random checkpoint rows and compare them with the file."""
    tables = find_ple_tables(model_dir)
    info = table_info(table)
    prefix = info.metadata.get("source_prefix")
    if prefix not in tables:
        if len(tables) != 1:
            raise ValueError(f"{table}: cannot tell which PLE table it was built from")
        prefix = next(iter(tables))
    parts = tables[prefix]
    num_rows, dim = _table_shape(parts)
    if info.num_rows != num_rows or info.row_bytes != packed_row_bytes(dim):
        raise ValueError(
            f"{table}: table {info.num_rows} x {info.row_bytes}, checkpoint "
            f"{num_rows} x {dim}"
        )
    layout = _layout_tensors(model_dir, prefix)
    for key, values in layout.items():
        if key in info.layout and not np.array_equal(info.layout[key], values):
            raise ValueError(f"{table}: {key} differs from the checkpoint")
    rows = np.unique(np.random.default_rng(seed).integers(0, num_rows, sample))
    mapped = map_rows(info)
    from safetensors import safe_open

    mismatched = checked = 0
    start = 0
    for name, file in parts:
        with safe_open(file, framework="pt") as handle:
            view = handle.get_slice(name)
            count = int(view.get_shape()[0])
            local = rows[(rows >= start) & (rows < start + count)]
            for row in local.tolist():
                want = quantize_rows(view[row - start : row - start + 1]).numpy()
                mismatched += int(not np.array_equal(mapped[row], want[0]))
                checked += 1
        start += count
    result = {"table": table, "rows_checked": checked, "rows_mismatched": mismatched}
    print(json.dumps(result))
    if mismatched:
        raise ValueError(f"{table}: {mismatched} of {checked} rows differ")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="write the INT8 table file(s)")
    build_parser.add_argument("--model", required=True, help="checkpoint directory")
    build_parser.add_argument("--out", required=True, help="output file ({layer} ok)")
    build_parser.add_argument("--chunk-rows", type=int, default=1 << 20)
    verify_parser = sub.add_parser("verify", help="check a file against a checkpoint")
    verify_parser.add_argument("--model", required=True)
    verify_parser.add_argument("--table", required=True)
    verify_parser.add_argument("--sample", type=int, default=65536)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.model, args.out, chunk_rows=args.chunk_rows)
    else:
        verify(args.model, args.table, sample=args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the host side of the XPU Qwen4Exp PLE tables."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding
from vllm.models.qwen4_exp.xpu import ple_int8, ple_rows
from vllm.models.qwen4_exp.xpu.ple_rows import (
    PAGE,
    PinnedRowSlabs,
    RowCache,
    RowExtent,
    RowLookahead,
    RowReader,
    RowServer,
    RowStateSaver,
    RowStats,
    current_chunk_keys,
    host_ngram_ids,
    load_row_state,
    merge_extents,
    owned_heads,
    plan_next_chunks,
    plan_slabs,
    read_file_range,
    row_state_file,
    row_state_meta,
    save_row_state,
    scan_rows,
)

ROW_BYTES = 164


def _row_bytes(rows: np.ndarray, row_bytes: int = ROW_BYTES) -> np.ndarray:
    """Deterministic bytes of a row, different in every column and row."""
    rows = np.asarray(rows, dtype=np.int64)[:, None]
    return ((rows * 31 + np.arange(row_bytes)[None, :] * 7) % 251).astype(np.uint8)


@pytest.fixture
def table_file(tmp_path: Path) -> tuple[Path, int, int]:
    """A file of 2,000 rows, starting at an offset that is not a page."""
    rows, offset = 2000, 512
    path = tmp_path / "table.bin"
    path.write_bytes(bytes(offset) + _row_bytes(np.arange(rows)).tobytes())
    return path, offset, rows


@pytest.mark.parametrize(
    ("row_bytes", "num_slabs"),
    [
        pytest.param(164, 6, id="int8"),
        pytest.param(320, 8, id="bf16"),
        pytest.param(160, 8, id="fp8"),
    ],
)
def test_slabs_keep_the_pinned_footprint_at_the_table_size(row_bytes, num_slabs):
    rows = 80_000_384
    slabs = plan_slabs(rows, row_bytes)
    assert sum(slabs) == rows
    assert len(slabs) == num_slabs
    assert max(slabs) * row_bytes <= 1 << 33
    # The allocator rounds each request up to a power of two.
    pinned = sum(1 << ((n * row_bytes - 1).bit_length()) for n in slabs)
    assert pinned - rows * row_bytes < 64 << 20


def test_int8_share_of_a_rank_matches_the_table_size():
    assert plan_slabs(80_000_384, 164)[0] * 164 <= 1 << 33
    assert pytest.approx(12.22, abs=0.01) == 80_000_384 * 164 / 2**30


def test_slabs_load_across_extents_and_find_rows(table_file):
    path, offset, _ = table_file
    extents = [
        RowExtent(str(path), offset, 700),
        RowExtent(str(path), offset + 700 * ROW_BYTES, 1300),
    ]
    assert merge_extents(extents, ROW_BYTES) == [RowExtent(str(path), offset, 2000)]
    slabs = PinnedRowSlabs(100, 1900, ROW_BYTES, max_bytes=1 << 16, tail_bytes=1 << 12)
    assert len(slabs.slabs) > 2
    slabs.load(extents, direct=False, threads=2, chunk_bytes=PAGE * 4)
    rows = np.array([100, 101, 1899, 950, 399, 400])
    assert np.array_equal(slabs.rows(rows), _row_bytes(rows))
    with pytest.raises(IndexError):
        slabs.rows(np.array([1900]))


def test_slab_scales_are_checked():
    slabs = PinnedRowSlabs(10, 14, 8)
    for slab in slabs.slabs:
        slab[:, 4:] = np.frombuffer(np.float32(1.0).tobytes(), dtype=np.uint8)
    slabs.check_scales(4)
    slabs.slabs[-1][-1, 4:] = np.frombuffer(np.float32(-1.0).tobytes(), dtype=np.uint8)
    with pytest.raises(ValueError, match="row 13"):
        slabs.check_scales(4)


def test_file_range_read_retries_a_short_read_then_raises(table_file):
    path, offset, _ = table_file
    calls = []

    def flaky(fd, buffers, position):
        calls.append(position)
        if len(calls) == 1:
            return 0
        return os.preadv(fd, buffers, position)

    out = np.empty((10, ROW_BYTES), dtype=np.uint8)
    read_file_range(str(path), offset, out, direct=False, threads=1, preadv=flaky)
    assert len(calls) > 1
    assert np.array_equal(out, _row_bytes(np.arange(10)))

    def broken(fd, buffers, position):
        raise OSError(5, "Input/output error")

    with pytest.raises(RuntimeError, match="errno 5"):
        read_file_range(str(path), offset, out, direct=False, threads=1, preadv=broken)


def test_row_reader_reads_rows_that_cross_pages(table_file):
    path, offset, num_rows = table_file
    # PAGE // ROW_BYTES rows in, some row crosses the first page boundary.
    reader = RowReader([RowExtent(str(path), offset, num_rows)], ROW_BYTES, threads=4)
    try:
        rows = np.concatenate([np.arange(0, 60), np.array([1999, 1000, 1000])])
        latencies: list = []
        assert np.array_equal(
            reader.read_rows(rows, latencies=latencies), _row_bytes(rows)
        )
        assert latencies[0].shape[0] == rows.shape[0]
        with pytest.raises(IndexError):
            reader.read_rows(np.array([num_rows]))
    finally:
        reader.close()


def test_row_reader_finds_rows_in_several_files(tmp_path):
    rows = np.arange(40)
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.write_bytes(_row_bytes(rows[:15]).tobytes())
    second.write_bytes(bytes(100) + _row_bytes(rows[15:]).tobytes())
    extents = [RowExtent(str(first), 0, 15), RowExtent(str(second), 100, 25)]
    reader = RowReader(extents, ROW_BYTES, threads=2)
    try:
        assert np.array_equal(reader.read_rows(rows[::-1]), _row_bytes(rows[::-1]))
    finally:
        reader.close()


def test_row_reader_retries_failed_reads_and_never_zero_fills(table_file, monkeypatch):
    path, offset, num_rows = table_file
    reader = RowReader([RowExtent(str(path), offset, num_rows)], ROW_BYTES, threads=1)
    try:
        # Reads through a closed descriptor fail; the retry path reads properly.
        good_fds = reader._extent_fds.copy()
        reader._extent_fds[:] = 1_000_000
        retry = os.preadv

        def from_good_fd(fd, buffers, position):
            return retry(int(good_fds[0]), buffers, position)

        monkeypatch.setattr(reader, "_preadv", from_good_fd)
        rows = np.array([5, 7])
        assert np.array_equal(reader.read_rows(rows), _row_bytes(rows))
        assert reader.retries == 2

        def broken(fd, buffers, position):
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(reader, "_preadv", broken)
        with pytest.raises(RuntimeError, match="errno 5"):
            reader.read_rows(rows)
    finally:
        reader._extent_fds[:] = good_fds
        reader.close()


def test_truncated_file_is_refused(table_file):
    path, offset, num_rows = table_file
    with pytest.raises(ValueError, match="past the file"):
        RowReader([RowExtent(str(path), offset, num_rows + 1)], ROW_BYTES)


def _cache(num_rows: int, capacity: int) -> RowCache:
    return RowCache(num_rows, np.zeros((capacity, ROW_BYTES), dtype=np.uint8))


def test_row_cache_protects_this_step_and_the_last():
    cache = _cache(100, 6)
    cache.begin_step()
    first = np.array([1, 2, 3])
    cache.install(first, _row_bytes(first))
    cache.begin_step()
    second = np.array([4, 5, 6])
    cache.install(second, _row_bytes(second))
    cache.begin_step()
    # Rows of the first step are the only evictable ones.
    third = np.array([7, 8, 9])
    cache.lookup(second)
    cache.install(third, _row_bytes(third))
    assert (cache.row2slot[first] == -1).all()
    assert (cache.row2slot[second] >= 0).all()
    slots = cache.lookup(third)
    assert np.array_equal(cache.slab[slots], _row_bytes(third))


def test_row_cache_raises_rather_than_evict_a_protected_row():
    cache = _cache(100, 4)
    cache.begin_step()
    rows = np.array([1, 2, 3])
    cache.install(rows, _row_bytes(rows))
    with pytest.raises(RuntimeError, match="disk_row_cache_gib"):
        more = np.array([4, 5])
        cache.install(more, _row_bytes(more))
    assert (cache.row2slot[rows] >= 0).all()
    assert cache.row2slot[4] == -1


def _server(path: Path, offset: int, start=0, end=2000, capacity=64) -> RowServer:
    reader = RowReader([RowExtent(str(path), offset, 2000)], ROW_BYTES, threads=2)
    return RowServer(
        start=start,
        end=end,
        multipliers=np.array([3, 5, 7], dtype=np.int64),
        sizes=np.array([1000, 1000], dtype=np.int64),
        offsets=np.array([0, 1000], dtype=np.int64),
        eos_token_id=0,
        heads_per_ngram=1,
        cache=_cache(end - start, capacity),
        reader=reader,
        stats=RowStats(0, 0.0),
    )


def test_server_reads_misses_and_keeps_hits_in_memory(table_file, monkeypatch):
    path, offset, _ = table_file
    server = _server(path, offset, start=500, end=1500)
    own = np.array([[510, 1600], [510, 1400], [100, 1400]])
    slots, mask = server.resolve(own)
    assert mask.tolist() == [[True, False], [True, True], [False, True]]
    assert np.array_equal(server.cache.slab[slots[mask]], _row_bytes(own[mask]))
    monkeypatch.setattr(server.reader, "read_rows", None)  # No second read.
    slots, mask = server.resolve(own)
    assert np.array_equal(server.cache.slab[slots[mask]], _row_bytes(own[mask]))


def test_server_uses_prefetched_rows_before_reading(table_file, monkeypatch):
    path, offset, _ = table_file
    server = _server(path, offset)
    pre_rows = np.array([3, 8, 20])
    reads: list = []
    read_rows = server.reader.read_rows

    def counted(rows, **kwargs):
        reads.append(rows)
        return read_rows(rows, **kwargs)

    monkeypatch.setattr(server.reader, "read_rows", counted)
    own = np.array([[3], [8], [9]])
    slots, mask = server.resolve(own, prefetched=(pre_rows, _row_bytes(pre_rows)))
    assert np.array_equal(server.cache.slab[slots[mask]], _row_bytes(own[mask]))
    assert [r.tolist() for r in reads] == [[9]]
    # The unused prefetched row went in as well.
    assert server.cache.row2slot[20] >= 0


def test_wrong_lookahead_rows_give_way_to_the_next_step(table_file):
    """At the minimum capacity, two steps of two rows, rows from a wrong
    lookahead guess must not take the next step's room."""
    path, offset, _ = table_file
    server = _server(path, offset, capacity=4)
    wrong = np.array([2, 3])
    server.resolve(np.array([[0], [1]]), prefetched=(wrong, _row_bytes(wrong)))
    assert (server.cache.row2slot[wrong] >= 0).all()
    own = np.array([[4], [5]])
    slots, mask = server.resolve(own)
    assert np.array_equal(server.cache.slab[slots[mask]], _row_bytes(own[mask]))
    # The previous step's rows stay protected; the guesses gave way.
    assert (server.cache.row2slot[[0, 1]] >= 0).all()
    assert (server.cache.row2slot[wrong] == -1).all()


def test_a_lookahead_row_is_protected_once_a_step_asks_for_it(table_file):
    path, offset, _ = table_file
    server = _server(path, offset, capacity=4)
    guess = np.array([2, 3])
    server.resolve(np.array([[0], [1]]), prefetched=(guess, _row_bytes(guess)))
    server.resolve(np.array([[2], [3]]))  # right guess: hits, now protected
    server.resolve(np.array([[4], [5]]))
    assert (server.cache.row2slot[guess] >= 0).all()
    assert (server.cache.row2slot[[0, 1]] == -1).all()


def test_lookahead_plan_continues_each_prefill_within_the_budget():
    idx = np.array([7, 9, 4])
    computed = np.array([0, 100, 50])
    scheduled = np.array([10, 20, 1])
    prefill = np.array([100, 120, 50])
    # Request 7 has 90 left, 9 is done, 4 decodes, so the budget of 40 loses two.
    assert plan_next_chunks(idx, computed, scheduled, prefill, 40) == [(7, 10, 48)]
    assert current_chunk_keys(idx, computed, scheduled, prefill) == {(7, 0), (9, 100)}


def test_lookahead_delivers_the_rows_of_a_chunk(table_file):
    path, offset, _ = table_file
    server = _server(path, offset)
    lookahead = RowLookahead(
        server, RowReader([RowExtent(str(path), offset, 2000)], ROW_BYTES)
    )
    try:
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        keys = frozenset({(0, 8)})
        lookahead.submit(keys, tokens, np.array([0, 4]), np.array([[0, 0]]))
        prefetched, _ = lookahead.take(keys)
        assert prefetched is not None
        rows, data = prefetched
        assert (np.diff(rows) > 0).all()
        assert np.array_equal(data, _row_bytes(rows))
        own = server.hash_own(tokens, np.array([0, 4]), np.array([[0, 0]]))
        assert set(own.reshape(-1).tolist()) == set(rows.tolist())
        assert lookahead.take(keys) == (None, 0.0)
    finally:
        lookahead.close()
        server.reader.close()


def test_lookahead_skips_cached_rows_and_survives_a_failure(
    table_file, caplog, monkeypatch
):
    path, offset, _ = table_file
    server = _server(path, offset)
    lookahead = RowLookahead(
        server, RowReader([RowExtent(str(path), offset, 2000)], ROW_BYTES)
    )
    context = np.array([[0, 0]])
    try:
        tokens = np.array([1, 2], dtype=np.int64)
        own = server.hash_own(tokens, np.array([0, 2]), context)
        server.resolve(own)
        lookahead.submit(frozenset({(0, 0)}), tokens, np.array([0, 2]), context)
        assert lookahead.take(frozenset({(0, 0)}))[0] is None

        def broken(rows, **_):
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(lookahead.reader, "read_rows", broken)
        tokens = np.array([9], dtype=np.int64)
        lookahead.submit(frozenset({(0, 0)}), tokens, np.array([0, 1]), context)
        with caplog.at_level(logging.WARNING):
            assert lookahead.take(frozenset({(0, 0)}))[0] is None
        assert lookahead.errors == 1
    finally:
        lookahead.close()
        server.reader.close()


def test_stats_line_counts_the_step():
    stats = RowStats(2, 1000.0)
    stats.read_latencies.append(np.array([1e-3, 2e-3]))
    stats.record(lookups=10, misses=2, reads=1, step_ms=3.0, from_lookahead=1)
    line = stats.line()
    assert line.startswith("PLE row cache rank 2: 1 steps")
    assert "hit rate 0.800 (8/10)" in line
    assert "from lookahead 1" in line
    off = RowStats(0, 0.0)
    off.record(lookups=10, misses=2, reads=1, step_ms=3.0)
    assert off.steps == 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_host_hash_matches_the_torch_hash(seed):
    generator = torch.Generator().manual_seed(seed)
    eos, ngram_size, per_gram = 7, 3, 2
    head_count = (ngram_size - 1) * per_gram
    embedding = SimpleNamespace(
        ngram_size=ngram_size,
        heads_per_ngram=per_gram,
        eos_token_id=eos,
        layer_multipliers=torch.tensor([1_000_003, 998_244_353, 2_147_483_647]),
        ngram_heads_vocab_sizes=torch.tensor([101, 103, 107, 109]),
        ngram_heads_offsets=torch.tensor([0, 101, 204, 311]),
        _shift_precompute=Qwen4ExpNGramEmbedding._shift_precompute,
        _shift_apply=Qwen4ExpNGramEmbedding._shift_apply,
    )
    lengths = torch.randint(1, 30, (5,), generator=generator)
    qsl = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    # A narrow range makes EOS frequent.
    tokens = torch.randint(0, 20, (int(qsl[-1]),), generator=generator)
    context = torch.randint(0, 20, (5, ngram_size - 1), generator=generator)
    want = Qwen4ExpNGramEmbedding.compute_ngram_ids(
        cast(Qwen4ExpNGramEmbedding, embedding), tokens, qsl, context
    )
    heads = np.arange(head_count)
    got = host_ngram_ids(
        tokens.numpy(),
        qsl.numpy(),
        context.numpy(),
        multipliers=embedding.layer_multipliers.numpy(),
        sizes=embedding.ngram_heads_vocab_sizes.numpy(),
        offsets=embedding.ngram_heads_offsets.numpy(),
        eos_token_id=eos,
        heads_per_ngram=per_gram,
        heads=heads,
    )
    assert np.array_equal(got, want.numpy())


def test_owned_heads_are_those_that_reach_the_range():
    sizes, offsets = np.array([10, 10, 10]), np.array([0, 10, 20])
    assert owned_heads(sizes, offsets, 5, 12).tolist() == [0, 1]
    assert owned_heads(sizes, offsets, 25, 40).tolist() == [2]


@pytest.fixture
def source_checkpoint(tmp_path: Path) -> Path:
    prefix = "model.layers.1.ple.ple_embedding."
    table = torch.randn(40, 16, generator=torch.Generator().manual_seed(4))
    tensors = {
        f"{prefix}ngram_embedding.shard_{i}.weight": shard.to(torch.bfloat16).clone()
        for i, shard in enumerate(table.split(10))
    }
    tensors[f"{prefix}ngram_heads_offsets"] = torch.tensor([0, 20])
    tensors[f"{prefix}ngram_heads_vocab_sizes"] = torch.tensor([20, 20])
    tensors[f"{prefix}layer_multipliers"] = torch.tensor([3, 5])
    folder = tmp_path / "ckpt"
    folder.mkdir()
    save_file(tensors, folder / "model.safetensors")
    return folder


def test_quantize_matches_the_format():
    rows = torch.tensor([[1.0, -0.5, 0.25], [0.0, 0.0, 0.0]]).to(torch.bfloat16)
    packed = ple_int8.quantize_rows(rows)
    assert packed[0, :3].view(torch.int8).tolist() == [127, -64, 32]
    assert packed[1, :3].tolist() == [0, 0, 0]
    assert packed[0, 3:].clone().view(torch.float32).item() == pytest.approx(1 / 127)
    decoded = ple_int8.decode_rows(packed)
    assert decoded[1].tolist() == [0.0, 0.0, 0.0]
    assert ple_int8.relative_l2(decoded[0], rows[0].float()) < 0.02


def test_build_is_deterministic_and_verifies(source_checkpoint, tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    sha = ple_int8.build(source_checkpoint, first, chunk_rows=7)
    assert ple_int8.build(source_checkpoint, second, chunk_rows=64) == sha
    assert first.read_bytes() == second.read_bytes()
    assert Path(f"{first}.sha256").read_text() == f"{sha}  {first.name}\n"
    start, rows, row_bytes, layout = ple_int8.read_table_header(str(first))
    assert start % PAGE == 0
    assert (rows, row_bytes) == (40, 20)
    assert layout["layer_multipliers"].tolist() == [3, 5]
    result = ple_int8.verify(source_checkpoint, first)
    assert result["sha256"] == sha
    assert result["rel_l2"] < ple_int8.MAX_REL_ERROR
    with pytest.raises(FileExistsError):
        ple_int8.build(source_checkpoint, first)


def test_verify_catches_a_changed_row(source_checkpoint, tmp_path):
    table = tmp_path / "table.safetensors"
    ple_int8.build(source_checkpoint, table)
    start = ple_int8.read_table_header(str(table))[0]
    with open(table, "r+b") as handle:
        handle.seek(start + 20)
        handle.write(bytes(20))
    with pytest.raises(ValueError, match="does not match"):
        ple_int8.verify(source_checkpoint, table)
    with pytest.raises(ValueError, match="fresh quantization"):
        ple_int8.verify(source_checkpoint, table, check_sha=False)


def test_build_refuses_an_fp8_source(tmp_path):
    folder = tmp_path / "ckpt"
    folder.mkdir()
    prefix = "model.layers.1.ple.ple_embedding."
    save_file(
        {
            f"{prefix}ngram_embedding.shard_0.weight": torch.zeros(
                4, 8, dtype=torch.float8_e4m3fn
            ),
        },
        folder / "model.safetensors",
    )
    with pytest.raises(ValueError, match="built from BF16"):
        ple_int8.build(folder, tmp_path / "out.safetensors")


def test_table_with_another_format_is_refused(tmp_path):
    path = tmp_path / "bad.safetensors"
    save_file({"table": torch.zeros(2, 2, dtype=torch.uint8)}, path, {"format": "x"})
    with pytest.raises(ValueError, match="expected 'ple-int8-rowscale/v1'"):
        ple_int8.read_table_header(str(path))


# --------------------------------------------------------------------------
# Warm start: the saved hot rows and the page cache
# --------------------------------------------------------------------------


def _meta(table_file, start=500, end=1500) -> dict:
    path, offset, rows = table_file
    return row_state_meta([RowExtent(str(path), offset, rows)], ROW_BYTES, start, end)


def test_row_state_round_trips_in_order(table_file, tmp_path, caplog):
    meta = _meta(table_file)
    rows = np.array([7, 999, 0, 42, 500], dtype=np.int64)
    path = str(tmp_path / "state" / "ple.rows.tp0of2")
    save_row_state(path, rows, meta)
    assert not [p for p in os.listdir(tmp_path / "state") if ".part" in p]
    with caplog.at_level(logging.WARNING):
        assert load_row_state(path, meta).tolist() == rows.tolist()
    assert not caplog.records
    assert row_state_file("/s/ple.rows", 1, 2) == "/s/ple.rows.tp1of2"


def test_a_missing_row_state_is_quiet(table_file, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert load_row_state(str(tmp_path / "none"), _meta(table_file)) is None
    assert not caplog.records


def _damage(path: Path, how: str) -> None:
    data = bytearray(path.read_bytes())
    if how == "magic":
        data[0:8] = b"NOTROWS!"
    elif how == "truncated":
        data = data[:-3]
    elif how == "flipped":
        data[-1] ^= 0xFF
    elif how == "header":
        data[16:20] = b"\xff\xfe\xfd\xfc"
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("how", ["magic", "truncated", "flipped", "header"])
def test_a_damaged_row_state_is_ignored_with_one_warning(
    table_file, tmp_path, caplog, how
):
    meta = _meta(table_file)
    path = tmp_path / "rows"
    save_row_state(str(path), np.arange(10), meta)
    _damage(path, how)
    with caplog.at_level(logging.WARNING):
        assert load_row_state(str(path), meta) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "not usable" in warnings[0].getMessage()


@pytest.mark.parametrize(
    "change",
    [
        {"start": 0},
        {"end": 1400},
        {"row_bytes": 320},
        {"extents": [["/elsewhere/table", 512, 2000, 328512]]},
    ],
)
def test_a_row_state_of_another_table_or_rank_is_ignored_with_one_warning(
    table_file, tmp_path, caplog, change
):
    meta = _meta(table_file)
    path = str(tmp_path / "rows")
    save_row_state(path, np.arange(10), meta)
    with caplog.at_level(logging.WARNING):
        assert load_row_state(path, dict(meta, **change)) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "another table or rank" in warnings[0].getMessage()


def test_a_row_state_with_rows_out_of_range_or_twice_is_ignored(
    table_file, tmp_path, caplog
):
    meta = _meta(table_file)
    path = str(tmp_path / "rows")
    for rows in ([1, 2, 1000], [3, 4, 3]):  # the range holds 1,000 rows
        save_row_state(path, np.array(rows), meta)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert load_row_state(path, meta) is None
        assert len(caplog.records) == 1


def test_preload_fills_exactly_the_named_rows(table_file):
    path, offset, rows = table_file
    reader = RowReader([RowExtent(str(path), offset, rows)], ROW_BYTES, threads=2)
    try:
        cache = _cache(1000, 16)
        local = np.array([3, 8, 400, 999])
        referenced = np.array([True, False, True, False])
        cache.preload(
            local, referenced, lambda out: reader.read_rows(local + 500, out=out)
        )
        held = np.nonzero(cache.row2slot >= 0)[0]
        assert held.tolist() == local.tolist()
        assert np.array_equal(
            cache.slab[cache.row2slot[local]], _row_bytes(local + 500)
        )
        assert cache.ref[:4].tolist() == [1, 0, 1, 0]
        assert (cache.slot2row[4:] == -1).all()
        with pytest.raises(RuntimeError, match="unused cache"):
            cache.preload(local, referenced, lambda out: None)
    finally:
        reader.close()


def test_preloaded_rows_never_block_a_step():
    """A cache preloaded to the brim still serves a full step: preloaded rows are
    evictable until a step asks for them."""
    cache = _cache(100, 6)
    local = np.arange(6)
    cache.preload(local, np.ones(6, dtype=bool), lambda out: out.fill(1))
    cache.begin_step()
    hits = cache.lookup(np.array([0, 1]))
    assert (hits >= 0).all()
    new = np.array([50, 51, 52, 53])
    cache.install(new, _row_bytes(new))
    # The two rows the step asked for stay; four untouched ones gave way.
    assert (cache.row2slot[[0, 1]] >= 0).all()
    assert (cache.row2slot[new] >= 0).all()
    assert (cache.row2slot[2:6] == -1).all()


def test_hot_rows_are_most_recently_used_first():
    cache = _cache(100, 8)
    cache.begin_step()
    cache.install(np.array([10, 11]), _row_bytes(np.array([10, 11])))
    cache.begin_step()
    cache.install(np.array([20]), _row_bytes(np.array([20])))
    cache.install(
        np.array([30]), _row_bytes(np.array([30])), speculative=True
    )  # never asked for
    cache.begin_step()
    cache.lookup(np.array([11]))
    assert cache.hot_rows().tolist() == [11, 20, 10, 30]


def test_scan_reads_only_chunks_with_picked_rows(table_file, monkeypatch):
    path, offset, rows = table_file
    extents = [
        RowExtent(str(path), offset, 700),
        RowExtent(str(path), offset + 700 * ROW_BYTES, rows - 700),
    ]
    reads: list[int] = []
    real = ple_rows.read_file_range

    def counted(path, start, out, **kwargs):
        reads.append(out.shape[0])
        return real(path, start, out, **kwargs)

    monkeypatch.setattr(ple_rows, "read_file_range", counted)
    pick = np.array([500, 501, 699, 700, 1499])
    out = np.empty((pick.shape[0], ROW_BYTES), np.uint8)
    chunk = 100 * ROW_BYTES  # 100 rows
    scan_rows(
        extents,
        ROW_BYTES,
        500,
        1500,
        pick=pick,
        out=out,
        direct=False,
        chunk_bytes=chunk,
    )
    assert np.array_equal(out, _row_bytes(pick))
    # Chunks [500, 600), [600, 700), [700, 800), [1400, 1500) of 10.
    assert sum(reads) == 400
    reads.clear()
    scan_rows(extents, ROW_BYTES, 500, 1500, direct=False, chunk_bytes=chunk)
    assert sum(reads) == 1000  # a warm reads every row of the range


def test_page_cache_reader_reads_the_same_bytes(table_file):
    path, offset, rows = table_file
    extents = [RowExtent(str(path), offset, rows)]
    direct = RowReader(extents, ROW_BYTES, threads=2)
    cached = RowReader(extents, ROW_BYTES, threads=2, direct=False)
    try:
        assert not cached.direct
        picked = np.array([0, 24, 25, 1999, 1000])
        assert np.array_equal(direct.read_rows(picked), cached.read_rows(picked))
    finally:
        direct.close()
        cached.close()


def test_the_saver_writes_on_its_interval_and_at_close(table_file, tmp_path):
    meta = _meta(table_file)
    cache = _cache(1000, 8)
    cache.begin_step()
    cache.install(np.array([5, 6]), _row_bytes(np.array([5, 6])))
    path = str(tmp_path / "rows")
    saver = RowStateSaver(cache, path, meta, interval_s=3600.0, rank=0)
    saver.maybe_save()
    assert saver._thread is None  # not yet due
    saver.set_interval(1e-9)
    saver.maybe_save()
    assert saver._thread is not None
    saver._thread.join()
    assert saver.saves == 1 and sorted(load_row_state(path, meta).tolist()) == [5, 6]
    saver.set_interval(0.0)  # only at close from now on
    cache.begin_step()
    cache.install(np.array([7]), _row_bytes(np.array([7])))
    saver.maybe_save()
    assert saver.saves == 1
    saver.close()
    assert saver.saves == 2 and load_row_state(path, meta).tolist()[0] == 7
    assert not saver.save()  # closed


def test_a_failed_save_is_counted_and_logged(table_file, tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    saver = RowStateSaver(
        _cache(1000, 4), str(blocker / "rows"), _meta(table_file), 0.0, rank=0
    )
    with caplog.at_level(logging.WARNING):
        saver.close()
    assert saver.errors == 1 and saver.saves == 0
    assert "not written" in caplog.text


def _decode(server: RowServer, tokens: list[int], drafts: int = 3) -> int:
    """Serve a decode sequence step by step: each step verifies one token and
    ``drafts`` draft tokens after it; returns the rows read from the file."""
    reads = 0
    for step in range(2, len(tokens) - drafts):
        window = np.array(tokens[step : step + 1 + drafts], dtype=np.int64)
        context = np.array([tokens[step - 2 : step]], dtype=np.int64)
        own = server.hash_own(window, np.array([0, window.shape[0]]), context)
        reads_before = server.reader.reads_total
        server.resolve(own)
        reads += server.reader.reads_total - reads_before
    return reads


@pytest.fixture
def counted_reader(monkeypatch):
    real = RowReader.read_rows

    def read_rows(self, rows, out=None, latencies=None):
        self.reads_total = getattr(self, "reads_total", 0) + np.asarray(rows).size
        return real(self, rows, out=out, latencies=latencies)

    monkeypatch.setattr(RowReader, "read_rows", read_rows)
    monkeypatch.setattr(RowReader, "reads_total", 0, raising=False)


def test_a_warm_start_covers_a_repeated_decode_and_reads_only_new_rows(
    table_file, tmp_path, counted_reader
):
    path, offset, rows = table_file
    meta = row_state_meta([RowExtent(str(path), offset, rows)], ROW_BYTES, 0, 2000)
    generator = np.random.default_rng(7)
    text = generator.integers(1, 200, 120).tolist()
    # A first run, cold: every distinct n-gram is read once.
    cold = _server(path, offset, capacity=4096)
    cold_reads = _decode(cold, text)
    assert cold_reads > 0
    state = str(tmp_path / "rows")
    RowStateSaver(cold.cache, state, meta, 0.0, rank=0).close()
    cold.reader.close()
    # The next boot preloads what the first run held: the same text reads nothing.
    warm = _server(path, offset, capacity=4096)
    saved = load_row_state(state, meta)
    assert saved is not None
    local = np.sort(saved)
    warm.cache.preload(
        local,
        np.ones(local.shape[0], dtype=bool),
        lambda out: warm.reader.read_rows(local, out=out),
    )
    warm.reader.reads_total = 0
    assert _decode(warm, text) == 0
    # New text reads its new n-grams only, each once.
    changed = text[:60] + generator.integers(200, 400, 60).tolist()
    new_rows = set()
    for step in range(2, len(changed) - 3):
        window = np.array(changed[step : step + 4], dtype=np.int64)
        context = np.array([changed[step - 2 : step]], dtype=np.int64)
        own = warm.hash_own(window, np.array([0, 4]), context).reshape(-1).tolist()
        new_rows.update(r for r in own if warm.cache.row2slot[r] < 0)
    assert new_rows
    assert _decode(warm, changed) == len(new_rows)
    warm.reader.close()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 row-scale PLE table: format, loader and lookup paths (CPU)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.common.ngram_embedding as ngram_embedding_module
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEInt8RowwiseEmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from vllm.models.qwen4_exp.common.ple_int8 import (
    SCALE_BYTES,
    copy_quantized_shard_,
    dequantize_rows,
    packed_row_bytes,
    quantize_rows,
    row_scales,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer

PREFIX = "model.layers.1.ple.ple_embedding.ngram_embedding"


def _reference_quantize(rows_bf16: torch.Tensor) -> torch.Tensor:
    """The offline table builder's recipe, written independently (numpy)."""
    import numpy as np

    x = rows_bf16.to(torch.float32).numpy()
    s = (np.abs(x).max(axis=1, keepdims=True) / np.float32(127.0)).astype(np.float32)
    q = np.clip(np.rint(x / np.maximum(s, np.float32(1e-30))), -127, 127)
    out = np.empty((x.shape[0], x.shape[1] + 4), dtype=np.uint8)
    out[:, : x.shape[1]] = q.astype(np.int8).view(np.uint8)
    out[:, x.shape[1] :] = s.astype("<f4").view(np.uint8).reshape(-1, 4)
    return torch.from_numpy(out)


def _rows(num_rows: int, dim: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(num_rows, dim, generator=generator) * 0.05
    rows[0] = 0.0  # zero row: scale 0, values 0
    rows[1, 3] = 1e4  # one outlier
    rows[2] = -rows[2].abs()  # all negative
    return rows.to(torch.bfloat16)


def test_quantize_matches_offline_recipe_byte_for_byte() -> None:
    rows = _rows(4096, 160)
    packed = quantize_rows(rows)
    assert packed.dtype == torch.uint8
    assert packed.shape == (4096, packed_row_bytes(160))
    assert torch.equal(packed, _reference_quantize(rows))


def test_dequantize_error_and_special_rows() -> None:
    rows = _rows(2048, 160, seed=1)
    packed = quantize_rows(rows)
    restored = dequantize_rows(packed, 160, torch.float32)
    reference = rows.float()
    assert torch.count_nonzero(restored[0]) == 0
    assert float(row_scales(packed, 160)[0]) == 0.0
    body = slice(3, None)
    rel = (restored[body] - reference[body]).norm(dim=1) / reference[body].norm(dim=1)
    assert float(rel.mean()) < 0.01
    scales = row_scales(packed, 160)
    assert bool(torch.isfinite(scales).all()) and bool((scales >= 0).all())
    # The largest value of every row is reproduced exactly (q = +-127).
    peak = reference.abs().argmax(dim=1, keepdim=True)
    torch.testing.assert_close(
        restored.gather(1, peak)[body].abs(),
        reference.gather(1, peak)[body].abs(),
        rtol=1e-6,
        atol=0,
    )


def test_dequantize_heads_side_by_side() -> None:
    rows = _rows(12, 8, seed=2)
    packed = quantize_rows(rows)  # [12, 12]
    tokens = packed.reshape(3, 4 * packed_row_bytes(8))  # 3 tokens x 4 heads
    got = dequantize_rows(tokens, 8, torch.bfloat16)
    want = dequantize_rows(packed, 8, torch.bfloat16).reshape(3, 32)
    assert got.shape == (3, 32)
    assert torch.equal(got, want)
    with pytest.raises(ValueError, match="uint8"):
        dequantize_rows(tokens[:, :-1], 8, torch.bfloat16)


@pytest.mark.parametrize("tp_size", [1, 3, 4])
def test_sharded_quantized_load_equals_whole_table(tp_size: int) -> None:
    dim, vocab = 16, 1000
    rows = _rows(vocab, dim, seed=3)
    whole = quantize_rows(rows)
    shard_rows = 137  # checkpoint split does not align with the TP split
    per_rank = (vocab + tp_size - 1) // tp_size
    for rank in range(tp_size):
        tp_start, tp_end = rank * per_rank, min(vocab, (rank + 1) * per_rank)
        table = torch.full((per_rank, packed_row_bytes(dim)), 0xAB, dtype=torch.uint8)
        stored = 0
        for start in range(0, vocab, shard_rows):
            stored += copy_quantized_shard_(
                table,
                rows[start : start + shard_rows],
                checkpoint_start=start,
                tp_start=tp_start,
                tp_end=tp_end,
                chunk_rows=50,
            )
        assert stored == tp_end - tp_start
        assert torch.equal(table[:stored], whole[tp_start:tp_end])


def test_packed_shard_is_copied_verbatim_and_widths_are_checked() -> None:
    rows = _rows(10, 8, seed=4)
    packed = quantize_rows(rows)
    table = torch.zeros(10, packed_row_bytes(8), dtype=torch.uint8)
    assert (
        copy_quantized_shard_(table, packed, checkpoint_start=0, tp_start=0, tp_end=10)
        == 10
    )
    assert torch.equal(table, packed)
    with pytest.raises(ValueError, match="width"):
        copy_quantized_shard_(
            table, rows[:, :7], checkpoint_start=0, tp_start=0, tp_end=10
        )


def _mock_groups(monkeypatch: pytest.MonkeyPatch, world_size: int, all_reduce):
    group = SimpleNamespace(
        rank_in_group=0, world_size=world_size, all_reduce=all_reduce
    )
    monkeypatch.setattr(ngram_embedding_module, "get_etp_group", lambda: group)
    monkeypatch.setattr(
        ngram_embedding_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=world_size),
    )
    for module in (embedding_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: world_size
        )


def test_from_quant_config_selects_int8_only_when_requested(monkeypatch) -> None:
    monkeypatch.delenv("B70_PLE_INT8_QUANTIZE_AT_LOAD", raising=False)
    assert isinstance(
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, PREFIX),
        Qwen4ExpPLEUnquantizedEmbeddingMethod,
    )
    monkeypatch.setenv("B70_PLE_INT8_QUANTIZE_AT_LOAD", "1")
    assert isinstance(
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, PREFIX),
        Qwen4ExpPLEInt8RowwiseEmbeddingMethod,
    )
    with pytest.raises(NotImplementedError, match="INT8 PLE storage"):
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, PREFIX, "float8_e4m3fn")
    monkeypatch.setenv("B70_PLE_INT8_QUANTIZE_AT_LOAD", "0")
    assert isinstance(
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(None, PREFIX, "float8_e4m3fn"),
        Qwen4ExpPLEFp8EmbeddingMethod,
    )


def _int8_device_layer(monkeypatch, vocab: int, dim: int, world_size: int, reduce):
    _mock_groups(monkeypatch, world_size, reduce)
    return Qwen4ExpPLEDeviceEmbedding(
        vocab,
        dim,
        params_dtype=torch.bfloat16,
        padding_size=1,
        prefix=PREFIX,
        embedding_method=Qwen4ExpPLEInt8RowwiseEmbeddingMethod(),
    )


def test_int8_device_embedding_loads_shards_and_dequantizes(monkeypatch) -> None:
    dim, vocab = 8, 12
    layer = _int8_device_layer(monkeypatch, vocab, dim, 1, lambda t: t)
    assert layer.weight.dtype == torch.uint8
    assert layer.storage_dim == dim + SCALE_BYTES
    assert layer.embedding_dim == dim
    rows = _rows(vocab, dim, seed=5)
    for start in range(0, vocab, 5):
        layer.weight.weight_loader(
            layer.weight, rows[start : start + 5], checkpoint_start=start
        )
    assert torch.equal(layer.weight.data[:vocab], quantize_rows(rows))

    ids = torch.tensor([[3, 7], [11, 0]])  # 2 tokens x 2 heads
    looked_up = layer(ids)
    assert looked_up.dtype == torch.uint8
    assert looked_up.shape == (2, 2, dim + SCALE_BYTES)

    ple_layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(ple_layer)
    ple_layer.ple_embedding = nn.Module()
    ple_layer.ple_embedding.ngram_embedding = layer
    output = ple_layer._dequantize_embeddings(looked_up.flatten(-2), torch.bfloat16)
    want = dequantize_rows(quantize_rows(rows), dim, torch.bfloat16)[ids].flatten(-2)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, want)


def test_int8_device_embedding_loads_unsharded_table(monkeypatch) -> None:
    layer = _int8_device_layer(monkeypatch, 6, 4, 1, lambda t: t)
    rows = _rows(6, 4, seed=6)
    layer.weight.weight_loader(layer.weight, rows)
    assert torch.equal(layer.weight.data[:6], quantize_rows(rows))
    with pytest.raises(ValueError, match="rows"):
        layer.weight.weight_loader(layer.weight, rows[:5])


def test_int8_device_embedding_reduces_bytes_as_int8(monkeypatch) -> None:
    reduced = []

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        reduced.append(tensor.dtype)
        return tensor.clone()

    monkeypatch.setattr(
        embedding_module,
        "get_masked_input_and_mask",
        lambda *args: (torch.tensor([1, 0]), torch.tensor([False, True])),
    )
    layer = _int8_device_layer(monkeypatch, 4, 4, 2, all_reduce)
    rows = _rows(3, 4, seed=7)[1:]
    layer.weight.data[:2].copy_(quantize_rows(rows))

    output = layer(torch.tensor([1, 3]))

    assert reduced == [torch.int8]
    assert output.dtype == torch.uint8
    assert torch.equal(output[0], layer.weight.data[1])
    assert torch.count_nonzero(output[1]) == 0


def test_pinned_int8_reduce_keeps_packed_bytes() -> None:
    embedding = Qwen4ExpPLEPinnedHostEmbedding.__new__(Qwen4ExpPLEPinnedHostEmbedding)
    nn.Module.__init__(embedding)
    embedding.tp_size = 2
    reduced = []

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        reduced.append(tensor.dtype)
        return tensor.clone()

    embedding.parallel_group = SimpleNamespace(all_reduce=all_reduce)
    packed = quantize_rows(_rows(3, 8, seed=8)).reshape(1, 3, 12)
    output = embedding._reduce_etp_embeddings(packed)
    assert reduced == [torch.int8]
    assert output.dtype == torch.uint8
    assert torch.equal(output, packed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_int8_pinned_embedding_looks_up_packed_rows_through_uva(monkeypatch) -> None:
    _mock_groups(monkeypatch, 1, lambda t: t)
    dim, vocab = 8, 6
    with torch.device("cuda:0"):
        embedding = Qwen4ExpPLEPinnedHostEmbedding(
            vocab,
            dim,
            params_dtype=torch.bfloat16,
            padding_size=1,
            prefix=PREFIX,
            embedding_method=Qwen4ExpPLEInt8RowwiseEmbeddingMethod(),
        )
    rows = _rows(vocab, dim, seed=9)
    embedding.weight.weight_loader(embedding.weight, rows, checkpoint_start=0)
    assert embedding.weight.device.type == "cpu"
    assert embedding.weight.is_pinned()
    assert embedding.weight.dtype == torch.uint8

    input_ids = torch.tensor([[5, 0], [1, 2]], device="cuda:0")
    output = embedding._lookup(input_ids)
    packed = quantize_rows(rows)
    assert output.shape == (2, 2, dim + SCALE_BYTES)
    assert torch.equal(output.cpu(), packed[input_ids.cpu()])
    restored = embedding.dequantize(output.flatten(-2), torch.bfloat16)
    want = dequantize_rows(packed, dim, torch.bfloat16)[input_ids.cpu()].flatten(-2)
    assert torch.equal(restored.cpu(), want)

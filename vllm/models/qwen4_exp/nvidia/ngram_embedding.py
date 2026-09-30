# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp n-gram embeddings with device and pinned-host storage."""

import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)

from ..common.ngram_embedding import (
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
    _is_xpu,
    _host_memory_note,
    _ple_direct_pinned_enabled,
    _b70_mmap_safetensors,
    _b70_ple_fp8_crosscheck,
    _b70_ple_fp8_layout_check,
    _b70_ple_fp8_segments,
    B70PLEFp8PinnedEmbeddingMethod,
    _ple_fp8_enabled,
    # B70 PLE helpers (patches 0002/0006-0008 live in common/)
    _b70_ple_int8_crosscheck,
    _B70_PLE_INT8_FORMATS,
    _B70_PLE_INT8_SCALE_BYTES,
    _b70_ple_int8_table,
    _b70_safetensors_metadata,
    B70PLEInt8RowPinnedEmbeddingMethod,
    _ple_int8_enabled,
)
from .ops.ple import ple_ngram_ids

logger = init_logger(__name__)

__all__ = [
    "Qwen4ExpPLEDeviceEmbedding",
    "Qwen4ExpPLEEmbedding",
    "Qwen4ExpPLEEmbeddingMethod",
    "Qwen4ExpPLEFp8EmbeddingMethod",
    "Qwen4ExpPLEPinnedHostEmbedding",
    "Qwen4ExpPLEUnquantizedEmbeddingMethod",
    "Qwen4ExpNGramEmbedding",
]


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    def _load_ple_table_from_path(self) -> bool:
        """Fill the pinned-host PLE table from the PLE_TABLE_PATH mmap file.

        XPU bring-up: this checkpoint carries no embedding rows; the full
        [org_vocab_size, head_dim] bf16 table lives in one external file
        (``{"table": Tensor}`` mmap) whose rows already follow the exact
        layout the shard branch of ``load_weights`` consumes. Slice it into
        ``split_ngram_parts`` virtual shards and run each through the same
        ``weight_loader(checkpoint_start=...)`` path so only TP-owned rows
        are copied into this rank's pinned storage.
        """
        if _is_xpu() and _ple_int8_enabled():
            return self._load_ple_int8_table()
        if _is_xpu() and _ple_fp8_enabled():
            return self._load_ple_fp8_table()
        table_path = os.environ.get("PLE_TABLE_PATH")
        if not table_path:
            return False
        embedding = self.ngram_embedding
        if not isinstance(embedding, Qwen4ExpPLEPinnedHostEmbedding):
            raise RuntimeError(
                "PLE_TABLE_PATH is set but the PLE table is device-resident; "
                "XPU requires pinned-host table storage"
            )
        table_file = torch.load(table_path, mmap=True, weights_only=True)
        table = table_file["table"] if isinstance(table_file, dict) else table_file
        org_vocab_size = embedding.org_vocab_size
        if (
            table.shape[0] < org_vocab_size
            or table.shape[1] != embedding.embedding_dim
        ):
            raise ValueError(
                f"PLE table at {table_path} has shape {tuple(table.shape)}, "
                f"expected >= [{org_vocab_size}, {embedding.embedding_dim}]"
            )
        if (
            _is_xpu()
            and _ple_direct_pinned_enabled()
            and isinstance(
                embedding.embedding_method, Qwen4ExpPLEUnquantizedEmbeddingMethod
            )
        ):
            # B70 0006: the weight_loader loop below would first fill the
            # 23.8 GiB/rank pageable shard and only then copy it into the
            # pinned slabs (47.7 GiB/rank transient, ~191 GiB over 4 ranks
            # at once). Copy the TP-owned rows from the mmap into the slabs
            # directly instead. Same rows, same dtype cast (copy_), same slab
            # layout; the pageable shard is released untouched.
            logger.info(
                "PLE direct-pinned load (B70_PLE_DIRECT_PINNED=1) starting: %s",
                _host_memory_note(),
            )
            embedding._materialize_pinned_xpu_slabs(source=table)
            logger.info(
                "Loaded PLE table from %s: direct-pinned, tp rows [%d, %d) of "
                "%d, dtype=%s, storage=pinned-host slabs; %s",
                table_path,
                embedding.shard_indices.org_vocab_start_index,
                embedding.shard_indices.org_vocab_end_index,
                org_vocab_size,
                table.dtype,
                _host_memory_note(),
            )
            return True
        shard_size = (
            org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        rows_loaded = 0
        for shard_index in range(self.split_ngram_parts):
            checkpoint_start = shard_index * shard_size
            expected_rows = max(
                0, min(shard_size, org_vocab_size - checkpoint_start)
            )
            if expected_rows == 0:
                break
            shard = table.narrow(0, checkpoint_start, expected_rows)
            embedding.weight.weight_loader(
                embedding.weight,
                shard,
                checkpoint_start=checkpoint_start,
            )
            rows_loaded += expected_rows
        if _is_xpu():
            embedding._materialize_pinned_xpu_slabs()
        logger.info(
            "Loaded PLE table from %s: %d shards, %d/%d rows, dtype=%s, "
            "storage=pinned-host slabs",
            table_path,
            self.split_ngram_parts,
            rows_loaded,
            org_vocab_size,
            table.dtype,
        )
        return True

    def _load_ple_fp8_table(self) -> bool:
        """B70 0007: fill the pinned slabs with the FP8 table, raw bytes.

        Reads B70_PLE_FP8_PATH (mmap, zero-copy), refuses on any shape,
        dtype or scale mismatch, copies only this rank's TP rows into the
        FP8 pinned slabs (no pageable copy, as 0006) and sets the global
        scale. When PLE_TABLE_PATH (the BF16 table) is also set, dequantised
        FP8 rows are compared with it on a random sample and the load refuses
        above B70_PLE_FP8_MAX_REL_ERR (default 0.25).
        """
        path = os.environ["B70_PLE_FP8_PATH"]
        embedding = self.ngram_embedding
        if not isinstance(embedding, Qwen4ExpPLEPinnedHostEmbedding):
            raise RuntimeError("B70_PLE_FP8=1 requires pinned-host PLE storage")
        if not isinstance(embedding.embedding_method, Qwen4ExpPLEFp8EmbeddingMethod):
            raise RuntimeError(
                "B70_PLE_FP8=1 but the PLE embedding method is "
                f"{type(embedding.embedding_method).__name__}"
            )
        if embedding.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"B70_PLE_FP8=1 but PLE storage dtype is {embedding.weight.dtype}"
            )
        tensors = _b70_mmap_safetensors(path)
        segments, scale, layout = _b70_ple_fp8_segments(
            tensors,
            embedding.org_vocab_size,
            embedding.embedding_dim,
            self.split_ngram_parts,
        )
        layout_checked = _b70_ple_fp8_layout_check(
            tensors,
            {
                "ngram_heads_offsets": self.ngram_heads_offsets,
                "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
                "layer_multipliers": self.layer_multipliers,
            },
        )
        if layout_checked:
            layout += f"; layout matches model: {', '.join(layout_checked)}"
        reference_path = os.environ.get("PLE_TABLE_PATH")
        if reference_path and os.environ.get("B70_PLE_FP8_CROSSCHECK", "1") == "1":
            reference_file = torch.load(reference_path, mmap=True, weights_only=True)
            reference = (
                reference_file["table"]
                if isinstance(reference_file, dict)
                else reference_file
            )
            if tuple(reference.shape[1:]) != (embedding.embedding_dim,):
                raise ValueError(
                    f"BF16 PLE table {reference_path} has shape "
                    f"{tuple(reference.shape)}, cannot cross-check"
                )
            limit = min(reference.shape[0], embedding.org_vocab_size)
            generator = torch.Generator().manual_seed(20260928)
            rows = torch.randint(0, limit, (4096,), generator=generator)
            rel_err = _b70_ple_fp8_crosscheck(segments, scale, reference, rows)
            max_rel_err = float(os.environ.get("B70_PLE_FP8_MAX_REL_ERR", "0.25"))
            logger.info(
                "FP8 PLE cross-check vs %s: relative L2 error %.4f on 4096 "
                "rows (limit %.2f)",
                reference_path,
                rel_err,
                max_rel_err,
            )
            if not rel_err <= max_rel_err:
                raise ValueError(
                    f"FP8 PLE table {path} disagrees with the BF16 table "
                    f"{reference_path}: relative error {rel_err:.4f} > "
                    f"{max_rel_err} (wrong file, row order or scale?)"
                )
            del reference, reference_file
        logger.info(
            "PLE FP8 direct-pinned load (B70_PLE_FP8=1) starting: %s", _host_memory_note()
        )
        embedding._materialize_pinned_xpu_slabs(source=segments)
        with torch.no_grad():
            embedding.weight_scale.data.copy_(
                scale.to(
                    device=embedding.weight_scale.device,
                    dtype=embedding.weight_scale.dtype,
                ).reshape(embedding.weight_scale.shape)
            )
        logger.info(
            "Loaded FP8 PLE table from %s (%s): tp rows [%d, %d) of %d, "
            "scale=%g, storage=pinned-host FP8 slabs; %s",
            path,
            layout,
            embedding.shard_indices.org_vocab_start_index,
            embedding.shard_indices.org_vocab_end_index,
            embedding.org_vocab_size,
            float(scale[0]),
            _host_memory_note(),
        )
        return True

    def _load_ple_int8_table(self) -> bool:
        """B70 0008: fill the pinned slabs with the INT8 row-scale table.

        Reads B70_PLE_INT8_PATH (mmap, zero-copy), refuses on a wrong format
        tag, dtype, width, row count or n-gram layout, copies only this
        rank's TP rows (164-byte rows, verbatim) into uint8 pinned slabs (no
        pageable copy, as 0006/0007), then refuses if any owned row scale is
        non-finite or negative. When PLE_TABLE_PATH (the BF16 table) is also
        set, dequantised rows are compared with it on a random sample and the
        load refuses above B70_PLE_INT8_MAX_REL_ERR (default 0.02).
        """
        path = os.environ["B70_PLE_INT8_PATH"]
        embedding = self.ngram_embedding
        dim = embedding.embedding_dim
        if not isinstance(embedding, Qwen4ExpPLEPinnedHostEmbedding):
            raise RuntimeError("B70_PLE_INT8=1 requires pinned-host PLE storage")
        if not isinstance(
            embedding.embedding_method, B70PLEInt8RowPinnedEmbeddingMethod
        ):
            raise RuntimeError(
                "B70_PLE_INT8=1 but the PLE embedding method is "
                f"{type(embedding.embedding_method).__name__}"
            )
        if embedding.weight.dtype != torch.uint8 or (
            embedding._b70_storage_dim != dim + _B70_PLE_INT8_SCALE_BYTES
        ):
            raise RuntimeError(
                f"B70_PLE_INT8=1 but PLE storage is {embedding.weight.dtype} "
                f"width {embedding._b70_storage_dim}"
            )
        fmt = _b70_safetensors_metadata(path).get("format")
        if fmt not in _B70_PLE_INT8_FORMATS:
            raise ValueError(
                f"INT8 PLE file {path} has format {fmt!r}, expected one of "
                f"{_B70_PLE_INT8_FORMATS!r}"
            )
        tensors = _b70_mmap_safetensors(path)
        table = _b70_ple_int8_table(tensors, embedding.org_vocab_size, dim)
        layout = f"table {tuple(table.shape)}"
        layout_checked = _b70_ple_fp8_layout_check(
            tensors,
            {
                "ngram_heads_offsets": self.ngram_heads_offsets,
                "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
                "layer_multipliers": self.layer_multipliers,
            },
        )
        if layout_checked:
            layout += f"; layout matches model: {', '.join(layout_checked)}"
        reference_path = os.environ.get("PLE_TABLE_PATH")
        if reference_path and os.environ.get("B70_PLE_INT8_CROSSCHECK", "1") == "1":
            reference_file = torch.load(reference_path, mmap=True, weights_only=True)
            reference = (
                reference_file["table"]
                if isinstance(reference_file, dict)
                else reference_file
            )
            if tuple(reference.shape[1:]) != (dim,):
                raise ValueError(
                    f"BF16 PLE table {reference_path} has shape "
                    f"{tuple(reference.shape)}, cannot cross-check"
                )
            limit = min(reference.shape[0], embedding.org_vocab_size)
            generator = torch.Generator().manual_seed(20260928)
            rows = torch.randint(0, limit, (4096,), generator=generator)
            rel_err = _b70_ple_int8_crosscheck(table, dim, reference, rows)
            max_rel_err = float(os.environ.get("B70_PLE_INT8_MAX_REL_ERR", "0.02"))
            logger.info(
                "INT8 PLE cross-check vs %s: relative L2 error %.4f on 4096 "
                "rows (limit %.3f)",
                reference_path,
                rel_err,
                max_rel_err,
            )
            if not rel_err <= max_rel_err:
                raise ValueError(
                    f"INT8 PLE table {path} disagrees with the BF16 table "
                    f"{reference_path}: relative error {rel_err:.4f} > "
                    f"{max_rel_err} (wrong file, row order or packing?)"
                )
            del reference, reference_file
        logger.info(
            "PLE INT8 direct-pinned load (B70_PLE_INT8=1) starting: %s",
            _host_memory_note(),
        )
        embedding._materialize_pinned_xpu_slabs(source=[(0, table)])
        # Every owned row's scale, straight from the pinned slabs.
        step = 1 << 22
        for slab in embedding._xpu_slabs or []:
            for start in range(0, slab.shape[0], step):
                part = slab.narrow(0, start, min(step, slab.shape[0] - start))
                scales = part[:, dim:].contiguous().view(torch.float32)
                if not bool(torch.isfinite(scales).all()) or bool(
                    (scales < 0).any()
                ):
                    raise ValueError(
                        f"INT8 PLE table {path}: non-finite or negative row "
                        "scale in this rank's rows"
                    )
        logger.info(
            "Loaded INT8 PLE table from %s (%s): tp rows [%d, %d) of %d, "
            "storage=pinned-host uint8 slabs, %d B/row; %s",
            path,
            layout,
            embedding.shard_indices.org_vocab_start_index,
            embedding.shard_indices.org_vocab_end_index,
            embedding.org_vocab_size,
            embedding._b70_storage_dim,
            _host_memory_note(),
        )
        return True

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        *,
        data_parallel_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        embedding_prefix = f"{prefix}.ngram_embedding"
        ple_embedding_dtype = getattr(config, "ple_embedding_dtype", None)
        b70_ple_fp8 = _is_xpu() and _ple_fp8_enabled()
        if b70_ple_fp8:
            # B70 0007: the served checkpoint (W4A16, PLE excluded from
            # quantization) carries no PLE rows; the FP8 table and its global
            # scale come from B70_PLE_FP8_PATH instead of PLE_TABLE_PATH.
            if not os.environ.get("B70_PLE_FP8_PATH"):
                raise RuntimeError(
                    "B70_PLE_FP8=1 requires B70_PLE_FP8_PATH (the FP8 PLE "
                    ".safetensors file)"
                )
            ple_embedding_dtype = "float8_e4m3fn"
        b70_ple_int8 = _is_xpu() and _ple_int8_enabled()
        if b70_ple_int8:
            # B70 0008: INT8 row-scale table from B70_PLE_INT8_PATH.
            if b70_ple_fp8:
                raise RuntimeError("B70_PLE_INT8=1 and B70_PLE_FP8=1 are exclusive")
            if not os.environ.get("B70_PLE_INT8_PATH"):
                raise RuntimeError(
                    "B70_PLE_INT8=1 requires B70_PLE_INT8_PATH (the INT8 PLE "
                    ".safetensors file)"
                )
        embedding_quant_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            quant_config,
            embedding_prefix,
            ple_embedding_dtype,
        )
        if b70_ple_fp8:
            embedding_quant_method = B70PLEFp8PinnedEmbeddingMethod()
        if b70_ple_int8:
            embedding_quant_method = B70PLEInt8RowPinnedEmbeddingMethod()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        engram_config = get_current_vllm_config().engram_config
        if engram_config is not None and engram_config.cpu_offload:
            embedding_cls = Qwen4ExpPLEPinnedHostEmbedding
        elif _is_xpu() and (
            os.environ.get("PLE_TABLE_PATH") or b70_ple_fp8 or b70_ple_int8
        ):
            # B70 XPU bring-up: pinned-host PLE table selected by
            # PLE_TABLE_PATH because the Engram route is CUDA-only
            # (config/engram.py rejects non-CUDA platforms) while the stock
            # XPU default would allocate the table device-resident (~26
            # GB/rank at TP4) and OOM the 32 GB cards.
            embedding_cls = Qwen4ExpPLEPinnedHostEmbedding
        else:
            embedding_cls = Qwen4ExpPLEDeviceEmbedding
        self.ngram_embedding = embedding_cls(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=embedding_prefix,
            embedding_method=embedding_quant_method,
            num_ngram_heads=self.ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        if self.ngram_embedding.supports_prefetch:
            # The side-stream lookup outlives eager-break args, whose
            # graph-pool storage later segments may reuse.
            self._prefetch_ids = torch.empty(
                max_total_tokens, self.ngram_heads, dtype=torch.long
            )
        weight = self.ngram_embedding.weight
        logger.info(
            "Initialized PLE embedding %s: quantization_method=%s, "
            "weight_dtype=%s, weight_device=%s, pinned=%s",
            embedding_prefix,
            type(embedding_quant_method).__name__,
            weight.dtype,
            weight.device,
            weight.is_pinned(),
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute n-gram embedding indices for the current request layout."""
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        if input_ids.is_cuda:
            return ple_ngram_ids(
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                layer_multipliers=self.layer_multipliers,
                ngram_heads_vocab_sizes=self.ngram_heads_vocab_sizes,
                ngram_heads_offsets=self.ngram_heads_offsets,
                eos_token_id=self.eos_token_id,
                heads_per_ngram=self.heads_per_ngram,
                output=output,
            )
        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()
        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, num_tokens),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.ngram_embedding
        if embedding.supports_prefetch:
            return embedding(hidden_states)
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Start the pinned lookup while the preceding decoder layer runs."""
        embedding = self.ngram_embedding
        if not embedding.supports_prefetch:
            return
        ngram_ids = self.compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
            output=self._prefetch_ids[: input_ids.numel()],
        )
        embedding.start_prefetch(hidden_states, ngram_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""
        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                embedding.weight.weight_loader(
                    embedding.weight,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        if "ngram_embedding.weight" not in loaded:
            # XPU bring-up: the checkpoint has no embedding rows, so fill
            # the pinned-host table from the external PLE_TABLE_PATH file.
            if self._load_ple_table_from_path():
                loaded.add("ngram_embedding.weight")
                if _is_xpu() and _ple_fp8_enabled():
                    loaded.add("ngram_embedding.weight_scale")
        return loaded


__all__ = [
    "Qwen4ExpNGramEmbedding",
]

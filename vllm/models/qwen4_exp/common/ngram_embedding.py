# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Qwen4Exp n-gram embedding storage with device and pinned-host backends.

Both the NVIDIA and AMD Qwen4Exp implementations use these classes so the (large)
n-gram embedding table can be kept in pinned host memory and looked up through
Unified Virtual Addressing on any CUDA-alike platform.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_dp_group, get_etp_group, get_tp_group
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .ple import PLEVocabParallelEmbedding

logger = init_logger(__name__)


def _is_xpu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_xpu()


def _ple_direct_pinned_enabled() -> bool:
    """B70 0006 gate: fill the XPU pinned PLE slabs straight from the mmap.

    Default OFF (unset or anything but "1"): the default load path runs
    unchanged. See _materialize_pinned_xpu_slabs(source=...).
    """
    return os.environ.get("B70_PLE_DIRECT_PINNED", "0") == "1"


def _host_memory_note() -> str:
    """VmRSS of this rank and node MemAvailable, for the load log line."""
    fields = {}
    for path, keys in (
        ("/proc/self/status", ("VmRSS",)),
        ("/proc/meminfo", ("MemAvailable",)),
    ):
        try:
            with open(path) as handle:
                for line in handle:
                    name, _, value = line.partition(":")
                    if name in keys:
                        fields[name] = int(value.split()[0]) / 2**20
        except OSError:
            pass
    return ", ".join(f"{k}={v:.1f} GiB" for k, v in fields.items()) or "n/a"


def _ple_fp8_enabled() -> bool:
    """B70 0007 gate: keep the XPU pinned PLE table in FP8 (E4M3).

    Default OFF (unset or anything but "1"): the table loads as today. On:
    the table comes from B70_PLE_FP8_PATH (a .safetensors file), the pinned
    slabs hold its raw FP8 bytes, and each looked-up row is dequantised
    after the gather (Qwen4ExpPLELayer._dequantize_embeddings).
    """
    return os.environ.get("B70_PLE_FP8", "0") == "1"


def _ple_int8_enabled() -> bool:
    """B70 0008 gate: keep the XPU pinned PLE table in INT8, one scale per row.

    Default OFF (unset or anything but "1"). On: the table comes from
    B70_PLE_INT8_PATH (a .safetensors file whose ``table`` is U8
    [rows, 164]: 160 int8 values then the row's float32 scale), the pinned
    slabs hold those 164-byte rows verbatim, and each looked-up row is
    dequantised after the gather as q * s. Mutually exclusive with
    B70_PLE_FP8.
    """
    return os.environ.get("B70_PLE_INT8", "0") == "1"


# B70 0008: bytes of the per-row float32 scale packed after the int8 values.
_B70_PLE_INT8_SCALE_BYTES = 4
# Accepted table format tags: the in-tree builder's tag first, then the
# legacy tag of tables built before it (same layout, same bytes).
_B70_PLE_INT8_FORMAT = "qwen4exp-ple-int8-rowscale/v1"
_B70_PLE_INT8_FORMATS = (_B70_PLE_INT8_FORMAT, "lumnus-ple-int8-rowscale/v1")


_B70_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _b70_mmap_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Map every tensor of a .safetensors file without reading it (B70 0007).

    Zero-copy views over one private (copy-on-write, never written) mmap, so
    rows only become resident as clean, evictable page cache when copied.
    Tensors whose dtype is outside _B70_SAFETENSORS_DTYPES are skipped (not
    mapped); callers refuse later only if a tensor they need is absent.
    Malformed offsets refuse for every tensor.
    """
    import json
    import mmap

    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        size = os.fstat(handle.fileno()).st_size
        if header_len <= 0 or 8 + header_len > size:
            raise ValueError(f"{path}: not a safetensors file (header {header_len})")
        header = json.loads(handle.read(header_len))
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
    data_start = 8 + header_len
    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = _B70_SAFETENSORS_DTYPES.get(info["dtype"])
        begin, end = (int(v) for v in info["data_offsets"])
        if dtype is None:
            if begin < 0 or end < begin or data_start + end > size:
                raise ValueError(f"{path}: tensor {name} offsets [{begin}, {end}) outside the file")
            continue
        shape = [int(v) for v in info["shape"]]
        numel = 1
        for dim in shape:
            numel *= dim
        itemsize = torch.empty((), dtype=dtype).element_size()
        if begin < 0 or end < begin or data_start + end > size:
            raise ValueError(f"{path}: tensor {name} offsets [{begin}, {end}) outside the file")
        if end - begin != numel * itemsize:
            raise ValueError(
                f"{path}: tensor {name} is {end - begin} bytes, shape {shape} x "
                f"{itemsize} needs {numel * itemsize}"
            )
        raw = torch.frombuffer(
            mapped, dtype=torch.uint8, count=end - begin, offset=data_start + begin
        ) if end > begin else torch.empty(0, dtype=torch.uint8)
        tensors[name] = raw.view(dtype).reshape(shape)
    return tensors


def _b70_copy_rows(
    destination: torch.Tensor,
    segments: list[tuple[int, torch.Tensor]],
    first_row: int,
) -> int:
    """Fill ``destination`` with table rows [first_row, first_row + len) (0007).

    ``segments`` are (global start row, rows) pieces of one logical table;
    same-dtype pieces are copied as raw bytes (no numeric conversion, also
    for FP8). Rows no segment covers are zeroed (0x00 = +0.0 in E4M3).
    Returns the number of rows copied from the segments.
    """
    count = destination.shape[0]
    covered = torch.zeros(count, dtype=torch.bool)
    copied = 0
    for seg_start, seg in segments:
        lo = max(first_row, seg_start)
        hi = min(first_row + count, seg_start + seg.shape[0])
        if lo >= hi:
            continue
        src = seg.narrow(0, lo - seg_start, hi - lo)
        dst = destination.narrow(0, lo - first_row, hi - lo)
        if src.dtype == dst.dtype and src.element_size() == 1:
            dst.view(torch.uint8).copy_(src.view(torch.uint8))
        else:
            dst.copy_(src)
        covered[lo - first_row : hi - first_row] = True
        copied += hi - lo
    if not bool(covered.all()):
        destination.view(torch.uint8)[~covered] = 0
    return copied


def _b70_fp8_lut(scale: torch.Tensor) -> torch.Tensor:
    """256-entry float32 table: E4M3 code -> value * scale (0007).

    Every E4M3 value (3 mantissa bits) times a BF16/F32 scale (<= 24 bits)
    is exact in float32, so rounding lut[code] to BF16 once equals the stock
    ``embeddings.to(bf16) * scale.to(bf16)`` bit for bit.
    """
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    values = codes.view(torch.float8_e4m3fn).to(torch.float32)
    return values * scale.detach().to(device="cpu", dtype=torch.float32).reshape(())


def _b70_fp8_dequantize_lut(
    lut: torch.Tensor, embeddings: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    """Dequantise gathered FP8 PLE rows with the code LUT (0007)."""
    codes = embeddings.view(torch.uint8)
    values = torch.index_select(lut, 0, codes.reshape(-1).to(torch.int64))
    return values.reshape(codes.shape).to(output_dtype)


def _b70_ple_fp8_segments(
    tensors: dict[str, torch.Tensor],
    org_vocab_size: int,
    embedding_dim: int,
    split_ngram_parts: int,
) -> tuple[list[tuple[int, torch.Tensor]], torch.Tensor, str]:
    """Resolve the FP8 PLE rows and global scale in a mapped file (0007).

    Accepted layouts (the file written by the table-extraction step):
      * ``table`` [>= org_vocab_size, dim] F8_E4M3 + ``weight_scale``;
      * the checkpoint's own keys, any prefix: ``...shard_<i>.weight``
        (i < split_ngram_parts, ceil(org_vocab_size / parts) rows each, the
        last one shorter) + ``...weight_scale``.
    The scale must be one element (the FP8 checkpoint's single global
    scale). Anything else refuses to start.
    """
    scale_keys = [k for k in tensors if k == "weight_scale" or k.endswith(".weight_scale")]
    if len(scale_keys) != 1:
        raise ValueError(
            f"FP8 PLE file must hold exactly one weight_scale, found {scale_keys}"
        )
    scale = tensors[scale_keys[0]]
    if scale.numel() != 1:
        raise ValueError(
            f"FP8 PLE scale {scale_keys[0]} has shape {tuple(scale.shape)}; "
            "only one global scale is supported"
        )
    scale = scale.reshape(1).to(torch.float32)
    if not bool(torch.isfinite(scale).all()) or float(scale[0]) <= 0.0:
        raise ValueError(f"FP8 PLE scale must be finite and > 0, got {float(scale[0])}")

    def check(name: str, tensor: torch.Tensor, rows: int | None) -> None:
        if tensor.dtype != torch.float8_e4m3fn:
            raise ValueError(f"FP8 PLE tensor {name} is {tensor.dtype}, expected float8_e4m3fn")
        if tensor.ndim != 2 or tensor.shape[1] != embedding_dim:
            raise ValueError(
                f"FP8 PLE tensor {name} has shape {tuple(tensor.shape)}, "
                f"expected [*, {embedding_dim}]"
            )
        if rows is not None and tensor.shape[0] != rows:
            raise ValueError(
                f"FP8 PLE tensor {name} has {tensor.shape[0]} rows, expected {rows}"
            )

    if "table" in tensors:
        table = tensors["table"]
        check("table", table, None)
        if table.shape[0] < org_vocab_size:
            raise ValueError(
                f"FP8 PLE table has {table.shape[0]} rows, expected >= {org_vocab_size}"
            )
        return [(0, table)], scale, f"table {tuple(table.shape)}"

    import re

    shard_re = re.compile(r"(?:^|\.)shard_(\d+)\.weight$")
    shards: dict[int, str] = {}
    for name in tensors:
        match = shard_re.search(name)
        if match:
            index = int(match.group(1))
            if index in shards:
                raise ValueError(f"FP8 PLE shard {index} appears twice ({shards[index]}, {name})")
            shards[index] = name
    shard_size = (org_vocab_size + split_ngram_parts - 1) // split_ngram_parts
    expected = [
        i for i in range(split_ngram_parts)
        if min(shard_size, org_vocab_size - i * shard_size) > 0
    ]
    if sorted(shards) != expected:
        raise ValueError(
            f"FP8 PLE file needs 'table' or shards {expected[0]}..{expected[-1]} "
            f"(split_ngram_parts={split_ngram_parts}); found {len(shards)} shard keys"
        )
    segments = []
    for index in expected:
        start = index * shard_size
        rows = min(shard_size, org_vocab_size - start)
        check(shards[index], tensors[shards[index]], rows)
        segments.append((start, tensors[shards[index]]))
    return segments, scale, f"{len(segments)} shards x {shard_size} rows"


def _b70_ple_fp8_layout_check(
    tensors: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor]
) -> list[str]:
    """Compare the file's optional n-gram layout tensors with the model (0007).

    For each of ngram_heads_offsets / ngram_heads_vocab_sizes /
    layer_multipliers present in the file (any prefix), the values must equal
    the model's loaded buffer, else the table rows index a different hash
    layout. Absent ones are skipped. Returns the names that were compared.
    """
    compared = []
    for leaf, buffer in buffers.items():
        keys = [k for k in tensors if k == leaf or k.endswith("." + leaf)]
        if not keys:
            continue
        if len(keys) != 1:
            raise ValueError(f"FP8 PLE file has {len(keys)} '{leaf}' tensors: {keys}")
        got = tensors[keys[0]]
        want = buffer.detach().to("cpu")
        if tuple(got.shape) != tuple(want.shape) or not torch.equal(
            got.to(torch.int64), want.to(torch.int64)
        ):
            raise ValueError(
                f"FP8 PLE file {leaf} {got.tolist()} does not match the model's "
                f"{want.tolist()}: the table was built for a different n-gram layout"
            )
        compared.append(leaf)
    return compared


def _b70_ple_fp8_crosscheck(
    segments: list[tuple[int, torch.Tensor]],
    scale: torch.Tensor,
    reference: torch.Tensor,
    rows: torch.Tensor,
) -> float:
    """Relative L2 error of dequantised FP8 rows against the BF16 table (0007).

    A table from the same parent weights lands near the FP8 rounding error
    (a few percent); a misaligned, wrong-scale or foreign table lands near
    or above 1.0.
    """
    picked = torch.empty(rows.numel(), reference.shape[1], dtype=torch.float8_e4m3fn)
    for out_row, row in enumerate(rows.tolist()):
        _b70_copy_rows(picked.narrow(0, out_row, 1), segments, int(row))
    lut = _b70_fp8_lut(scale)
    got = _b70_fp8_dequantize_lut(lut, picked, torch.float32)
    want = reference.index_select(0, rows).to(torch.float32)
    denom = float(torch.linalg.vector_norm(want))
    return float(torch.linalg.vector_norm(got - want)) / max(denom, 1e-30)


def _b70_safetensors_metadata(path: str) -> dict[str, str]:
    """The ``__metadata__`` block of a .safetensors header (B70 0008)."""
    import json

    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    meta = header.get("__metadata__") or {}
    return {str(k): str(v) for k, v in meta.items()}


def _b70_int8_dequantize(
    packed: torch.Tensor, logical_dim: int, output_dtype: torch.dtype
) -> torch.Tensor:
    """Dequantise gathered INT8 row-scale PLE rows (B70 0008).

    ``packed`` is uint8 [..., k * (logical_dim + 4)]: per row ``logical_dim``
    int8 values then a little-endian float32 scale. Returns
    [..., k * logical_dim] in ``output_dtype``: q * s in float32, one
    rounding. Standard elementwise ops only (view, slice, cast, multiply).
    """
    storage_dim = logical_dim + _B70_PLE_INT8_SCALE_BYTES
    lead = packed.shape[:-1]
    if packed.dtype != torch.uint8 or packed.shape[-1] % storage_dim:
        raise ValueError(
            f"INT8 PLE rows must be uint8 [..., k*{storage_dim}], got "
            f"{packed.dtype} {tuple(packed.shape)}"
        )
    rows = packed.reshape(*lead, packed.shape[-1] // storage_dim, storage_dim)
    values = rows[..., :logical_dim].view(torch.int8).to(torch.float32)
    scales = rows[..., logical_dim:].contiguous().view(torch.float32)
    return (values * scales).to(output_dtype).reshape(*lead, -1)


def _b70_ple_int8_table(
    tensors: dict[str, torch.Tensor],
    org_vocab_size: int,
    embedding_dim: int,
) -> torch.Tensor:
    """Resolve the packed INT8 PLE table in a mapped file (B70 0008)."""
    if "table" not in tensors:
        raise ValueError(
            f"INT8 PLE file needs a 'table' tensor, found {sorted(tensors)[:8]}"
        )
    table = tensors["table"]
    storage_dim = embedding_dim + _B70_PLE_INT8_SCALE_BYTES
    if table.dtype != torch.uint8:
        raise ValueError(f"INT8 PLE table is {table.dtype}, expected uint8 (packed rows)")
    if table.ndim != 2 or table.shape[1] != storage_dim:
        raise ValueError(
            f"INT8 PLE table has shape {tuple(table.shape)}, expected "
            f"[*, {storage_dim}] ({embedding_dim} int8 + float32 scale)"
        )
    if table.shape[0] < org_vocab_size:
        raise ValueError(
            f"INT8 PLE table has {table.shape[0]} rows, expected >= {org_vocab_size}"
        )
    return table


def _b70_ple_int8_crosscheck(
    table: torch.Tensor,
    embedding_dim: int,
    reference: torch.Tensor,
    rows: torch.Tensor,
) -> float:
    """Relative L2 error of dequantised INT8 rows against the BF16 table (0008).

    The sampled scales must be finite and >= 0, else refuse. A table cut
    from the same BF16 rows lands near 0.0066; a shifted or foreign table
    lands near or above 1.0.
    """
    picked = table.index_select(0, rows).contiguous()
    scales = picked[:, embedding_dim:].contiguous().view(torch.float32)
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise ValueError("INT8 PLE table has non-finite or negative row scales")
    got = _b70_int8_dequantize(picked, embedding_dim, torch.float32)
    want = reference.index_select(0, rows).to(torch.float32)
    denom = float(torch.linalg.vector_norm(want))
    return float(torch.linalg.vector_norm(got - want)) / max(denom, 1e-30)


class Qwen4ExpPLEEmbedding(PLEVocabParallelEmbedding, ABC):
    """ETP-sharded PLE table shared by device and pinned-host backends."""

    supports_prefetch: ClassVar[bool] = False

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: "Qwen4ExpPLEEmbeddingMethod",
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        del num_ngram_heads, max_total_tokens
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            quant_method=embedding_method,
            parallel_group=get_etp_group(),
        )
        self.embedding_method = embedding_method
        self.data_parallel_rank = data_parallel_rank
        tp_size = get_tp_group().world_size
        if self.tp_size % tp_size:
            raise ValueError(
                "ETP size must be divisible by TP size, but got "
                f"ETP={self.tp_size} and TP={tp_size}"
            )
        self.etp_data_parallel_size = self.tp_size // tp_size

    @abstractmethod
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate storage for the complete embedding weight."""
        raise NotImplementedError

    def dequantize(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Delegate storage-format conversion to the embedding method."""
        return self.embedding_method.dequantize(self, embeddings, output_dtype)

    def _get_dp_gather_slot(self, local_num_tokens: int) -> tuple[int, int]:
        """Return the per-DP slot size and this rank's slot offset."""
        if self.etp_data_parallel_size == 1:
            return local_num_tokens, 0
        dp_metadata: DPMetadata | None = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError("ETP spanning DP requires DP token metadata")
        group_start = (self.data_parallel_rank // self.etp_data_parallel_size) * (
            self.etp_data_parallel_size
        )
        group_end = group_start + self.etp_data_parallel_size
        token_counts = dp_metadata.num_tokens_across_dp_cpu.tolist()
        group_counts = token_counts[group_start:group_end]
        slot_size = max(group_counts)
        dp_rank = get_dp_group().rank_in_group
        return slot_size, dp_rank * slot_size

    def _gather_dp_ids(
        self,
        ngram_ids: torch.Tensor,
        slot_size: int,
    ) -> torch.Tensor:
        """Gather DP-local IDs that share one ETP-sharded PLE table."""
        if self.etp_data_parallel_size == 1:
            return ngram_ids
        if ngram_ids.shape[0] < slot_size:
            padding = ngram_ids.new_zeros(
                slot_size - ngram_ids.shape[0], ngram_ids.shape[1]
            )
            ngram_ids = torch.cat((ngram_ids, padding), dim=0)
        return get_dp_group().all_gather(ngram_ids, dim=0)

    def _select_embeddings(
        self,
        embeddings: torch.Tensor,
        local_num_tokens: int,
        slot_offset: int,
    ) -> torch.Tensor:
        """Select this DP rank's rows from the ETP-reduced embeddings."""
        if self.etp_data_parallel_size == 1:
            return embeddings
        return embeddings.narrow(0, slot_offset, local_num_tokens)

    @abstractmethod
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError


class Qwen4ExpPLEEmbeddingMethod(QuantizeMethodBase):
    """Quantization interface shared by resident and pinned PLE tables."""

    # PLE post-load processing only validates scales in their current storage.
    requires_device_loading: bool = False

    @staticmethod
    def from_quant_config(
        quant_config: QuantizationConfig | None,
        prefix: str,
        embedding_dtype: str | None = None,
    ) -> "Qwen4ExpPLEEmbeddingMethod":
        """Select the concrete PLE embedding format for a layer."""
        if embedding_dtype == "float8_e4m3fn":
            return Qwen4ExpPLEFp8EmbeddingMethod()
        if quant_config is None:
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            if quant_config._resolve_quant_algo(prefix) == "FP8":
                return Qwen4ExpPLEFp8EmbeddingMethod()
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(
            quant_config, ModelOptQuantConfigBase
        ) and quant_config.is_layer_excluded(prefix):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if isinstance(quant_config, CompressedTensorsConfig):
            # W4A16 pack-quantized checkpoints exclude the PLE embedding
            # through the compressed-tensors ignore list (e.g.
            # "re:.*\.ple\..*"); those tables are bf16 and load
            # unquantized. A table that is NOT excluded has no supported
            # compressed-tensors serialization here.
            if should_ignore_layer(
                prefix,
                ignore=quant_config.ignore,
                fused_mapping=quant_config.packed_modules_mapping,
            ):
                return Qwen4ExpPLEUnquantizedEmbeddingMethod()
            raise NotImplementedError(
                "Qwen4Exp PLE embedding is not in the compressed-tensors "
                "ignore list; compressed-tensors PLE quantization is not "
                "supported"
            )
        if not isinstance(quant_config, Fp8Config):
            raise NotImplementedError(
                "Qwen4Exp PLE embedding does not support quantization config "
                f"{type(quant_config).__name__}"
            )

        ignored_layers = quant_config.ignored_layers
        if is_layer_skipped(
            prefix,
            ignored_layers,
            quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        # PLE checkpoint shards form one runtime embedding parameter.
        shard_prefix = f"{prefix}.shard_"
        if any(name.startswith(shard_prefix) for name in ignored_layers):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not quant_config.is_checkpoint_fp8_serialized:
            raise NotImplementedError(
                "Qwen4Exp PLE embedding only supports serialized FP8 checkpoints"
            )
        return Qwen4ExpPLEFp8EmbeddingMethod()

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)

    @abstractmethod
    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert looked-up PLE rows to the activation dtype."""
        raise NotImplementedError


class Qwen4ExpPLEUnquantizedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Unquantized PLE embedding storage and lookup semantics."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight = nn.Parameter(
            layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer, output_dtype
        return embeddings


class Qwen4ExpPLEFp8EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition,
                torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            scale_dtype=torch.float32,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Reject FP8 PLE checkpoints without a global scale."""
        sentinel = torch.finfo(torch.float32).min
        if torch.any(layer.weight_scale == sentinel):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight_scale = getattr(layer, "weight_scale", None)
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)


class B70PLEFp8PinnedEmbeddingMethod(Qwen4ExpPLEFp8EmbeddingMethod):
    """B70 0007: FP8 PLE held in XPU pinned slabs, LUT dequant after gather.

    Same storage and scale contract as the stock FP8 method. Dequantisation
    goes through a 256-entry code table (value x global scale, float32) so
    the XPU never has to cast float8 tensors; the result is bitwise equal to
    the stock cast-and-multiply (see _b70_fp8_lut). B70_PLE_FP8_DEQUANT=cast
    selects the stock path instead.
    """

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        scale = layer.weight_scale
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError(
                f"FP8 PLE global scale must be finite and > 0, got {scale.tolist()}"
            )
        # Read once here, not per forward (keeps env reads out of the
        # compiled graph).
        use_lut = os.environ.get("B70_PLE_FP8_DEQUANT", "lut") != "cast"
        layer._b70_ple_fp8_lut = (
            _b70_fp8_lut(scale).to(scale.device) if use_lut else None
        )

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        lut = getattr(layer, "_b70_ple_fp8_lut", None)
        if lut is None:
            return super().dequantize(layer, embeddings, output_dtype)
        if lut.device != embeddings.device:
            raise RuntimeError("FP8 PLE dequant table must be on the output device")
        return _b70_fp8_dequantize_lut(lut, embeddings, output_dtype)


class B70PLEInt8RowPinnedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """B70 0008: INT8 PLE with one float32 scale per row, XPU pinned slabs.

    Storage is uint8 [rows, embedding_dim + 4]: each row's int8 values then
    its scale, so the scale travels with the row through the byte gather,
    the int8 byte-sum all-reduce (one owner per row) and the copy into the
    graph-owned output. ``layer.embedding_dim`` stays the logical width
    (160); the pinned embedding reads the storage width from the weight.
    """

    def create_weights(
        self,
        layer: Qwen4ExpPLEEmbedding,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight = nn.Parameter(
            layer.allocate_embedding_weight(
                sum(output_partition_sizes),
                input_size_per_partition + _B70_PLE_INT8_SCALE_BYTES,
                torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        del layer

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return _b70_int8_dequantize(embeddings, layer.embedding_dim, output_dtype)


class Qwen4ExpPLEDeviceEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table allocated on the active model device."""

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight on the active device."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Resident embedding prefetch is a no-op."""
        return None

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP inputs, look up embeddings, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = super().forward(gathered_ids)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )


@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    slab_start,
    slab_end,
    BLOCK_D: tl.constexpr,
):
    """Look up TP-owned PLE rows through an accelerator view of pinned memory.

    One launch covers one pinned slab whose rows are global ids in
    [slab_start, slab_end); rows outside the slab are not stored so a
    caller looping over slabs writes each row exactly once.
    """
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    in_slab = (global_idx >= slab_start) & (global_idx < slab_end)
    offsets = tl.arange(0, BLOCK_D)
    store_mask = (offsets < embedding_dim) & in_slab
    load_mask = (offsets < embedding_dim) & in_range
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=load_mask,
        other=0.0,
    )
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        values,
        mask=store_mask,
    )


class Qwen4ExpPLEPinnedHostEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table loaded into pinned CPU memory and looked up through UVA."""

    supports_prefetch: ClassVar[bool] = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        # B70 0008: the row width in storage can exceed the logical
        # embedding_dim (INT8 rows carry their float32 scale: 160 + 4 bytes).
        # Every storage-side size below uses it; for BF16/FP8 it is equal.
        self._b70_storage_dim = int(self.weight.shape[1])
        self._block_d = triton.next_power_of_2(self._b70_storage_dim)
        # XPU mirrors the CUDA stream API (Stream/current_stream/stream and
        # Tensor.record_stream all exist on torch 2.13.0+xpu). XPU pins at
        # most 16 GiB (2^34 B) per allocation, so a TP shard larger than
        # that cannot be one pinned tensor: the weight stays pageable here
        # and _materialize_pinned_xpu_slabs splits it after loading.
        self._xpu_slabs: list[torch.Tensor] | None = None
        self._xpu_slab_views: list[torch.Tensor] | None = None
        self._xpu_slab_rows = 0
        self._xpu_shard_rows = 0
        if _is_xpu():
            self._uva_weight = None
            self._stream_mod = torch.xpu
            self._prefetch_stream = torch.xpu.Stream()
        else:
            self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
            self._stream_mod = getattr(torch, self._uva_weight.device.type)
            self._prefetch_stream = self._stream_mod.Stream(
                device=self._uva_weight.device
            )
        self._prefetch_buffer = torch.empty(
            max_total_tokens * self.etp_data_parallel_size,
            num_ngram_heads,
            self._b70_storage_dim,
            dtype=self.weight.dtype,
            device=("xpu" if _is_xpu() else self._uva_weight.device),
        )
        self._output_dim = num_ngram_heads * self._b70_storage_dim

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate the complete PLE weight in host memory.

        CUDA keeps the stock single pinned allocation. On XPU, torch
        (2.13.0+xpu) silently returns *non-pinned* memory for pinned
        requests above 2^34 bytes and the UVA-view op then faults, so the
        TP shard is allocated pageable here and re-materialized into
        <=2^34-byte pinned slabs after the rows are loaded.
        """
        if _is_xpu():
            return torch.empty(
                num_embeddings, embedding_dim, dtype=dtype, device="cpu"
            )
        return torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def _materialize_pinned_xpu_slabs(
        self,
        source: torch.Tensor | list[tuple[int, torch.Tensor]] | None = None,
    ) -> None:
        """Copy the loaded shard into pinned slabs with UVA views (XPU).

        With ``source`` (the full mmap'd PLE table, B70 0006, gated by
        B70_PLE_DIRECT_PINNED=1) each slab is filled straight from the
        table's TP-owned rows, so the pageable ``self.weight`` shard is never
        written and never becomes resident: the per-rank host peak drops from
        pageable shard + pinned slabs to pinned slabs alone. Rows past the
        TP-owned range (vocab padding) are zeroed, which is what the stock
        path reads from the never-written pageable pages.

        B70 0007: ``source`` may also be a list of (global start row,
        rows) segments (a sharded FP8 file). FP8 slabs are filled as raw
        bytes and exposed to the lookup kernel as uint8 views, so no float8
        tensor reaches the XPU UVA op or Triton.
        """
        if self._xpu_slabs is not None:
            return
        shard = self.weight
        shard_rows = shard.shape[0]
        itemsize = shard.element_size()
        dim = shard.shape[1]
        # Keep every slab safely under the 2^34-byte pinned ceiling.
        slab_rows_limit = ((1 << 34) - (1 << 26)) // (dim * itemsize)
        num_slabs = max(1, -(-shard_rows // slab_rows_limit))
        slab_rows = -(-shard_rows // num_slabs)
        tp_start = self.shard_indices.org_vocab_start_index
        slabs: list[torch.Tensor] = []
        views: list[torch.Tensor] = []
        self._xpu_slab_rows = slab_rows
        self._xpu_shard_rows = shard_rows
        for index in range(num_slabs):
            start = index * slab_rows
            rows = min(slab_rows, shard_rows - start)
            if rows <= 0:
                break
            slab = torch.empty(rows, dim, dtype=shard.dtype, pin_memory=True)
            if not slab.is_pinned():
                raise RuntimeError(
                    f"PLE pinned slab {index} ({rows * dim * itemsize / 2**30:.1f}"
                    " GiB) silently failed to pin on XPU"
                )
            if source is None:
                slab.copy_(shard.narrow(0, start, rows))
            elif isinstance(source, list):
                owned = self.shard_indices.org_vocab_end_index - tp_start
                valid = max(0, min(rows, owned - start))
                if valid:
                    _b70_copy_rows(
                        slab.narrow(0, 0, valid), source, tp_start + start
                    )
                if valid < rows:
                    slab.narrow(0, valid, rows - valid).view(torch.uint8).zero_()
            else:
                owned = self.shard_indices.org_vocab_end_index - tp_start
                valid = max(0, min(rows, owned - start))
                if valid:
                    slab.narrow(0, 0, valid).copy_(
                        source.narrow(0, tp_start + start, valid)
                    )
                if valid < rows:
                    slab.narrow(0, valid, rows - valid).zero_()
            slabs.append(slab)
            if slab.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                views.append(
                    get_accelerator_view_from_cpu_tensor(slab.view(torch.uint8))
                )
            else:
                views.append(get_accelerator_view_from_cpu_tensor(slab))
        self._xpu_slabs = slabs
        self._xpu_slab_views = views
        logger.info(
            "Materialized PLE pinned-host slabs: %d slabs x %d rows "
            "(shard %d rows, tp range [%d, %d))",
            len(slabs),
            slab_rows,
            shard_rows,
            tp_start,
            self.shard_indices.org_vocab_end_index,
        )
        # Free the transient pageable copy now that every row lives in a
        # pinned slab; the forward path only reads the slab views.
        shard.data = torch.empty(0, dim, dtype=shard.dtype)

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Look up local ETP rows while preserving the weight storage dtype."""
        expected_shape = (*input_ids.shape, self._b70_storage_dim)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=self.weight.dtype,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != self.weight.dtype
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape, weight dtype, "
                "and input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            tp_start = self.shard_indices.org_vocab_start_index
            tp_end = self.shard_indices.org_vocab_end_index
            if self._xpu_slab_views is not None:
                # Rows owned by other ETP ranks are zero-contributions in
                # the all-reduce; no slab stores them, so zero first. Each
                # slab launch indexes rows against the slab's own bounds:
                # local_idx = id - slab_start stays inside the slab view.
                # B70 0007: FP8 slabs are exposed as uint8 views; the
                # kernel copies bytes, so the output is viewed the same way.
                kernel_output = output
                if output.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    kernel_output = output.view(torch.uint8)
                kernel_output.zero_()
                for index, view in enumerate(self._xpu_slab_views):
                    slab_start = tp_start + index * self._xpu_slab_rows
                    slab_end = tp_start + min(
                        (index + 1) * self._xpu_slab_rows, self._xpu_shard_rows
                    )
                    _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                        view,
                        flat_ids,
                        kernel_output,
                        self._b70_storage_dim,
                        slab_start,
                        slab_end,
                        slab_start,
                        slab_end,
                        BLOCK_D=self._block_d,
                    )
            elif self._uva_weight is not None:
                _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                    self._uva_weight,
                    flat_ids,
                    output,
                    self._b70_storage_dim,
                    tp_start,
                    tp_end,
                    tp_start,
                    tp_end,
                    BLOCK_D=self._block_d,
                )
            else:
                raise RuntimeError(
                    "XPU PLE lookup called before the pinned slabs were "
                    "materialized (PLE table not loaded?)"
                )
        return output

    def sync_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Synchronous UVA lookup for platforms without prefetch wiring."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        embeddings = self._lookup(gathered_ids)
        embeddings = self._reduce_etp_embeddings(embeddings)
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )

    def _reduce_etp_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Combine pinned lookup results owned by different ETP ranks."""
        if self.tp_size == 1:
            return embeddings
        assert self.parallel_group is not None
        if embeddings.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Each vocabulary row has one owner, so reduce the raw FP8 bytes.
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(embeddings.dtype)
        if embeddings.dtype == torch.uint8:
            # B70 0008: packed INT8 rows (values + scale bytes); one owner
            # per row, so the int8 byte sum is exact, as for FP8 (0007).
            reduced = self.parallel_group.all_reduce(embeddings.view(torch.int8))
            return reduced.view(torch.uint8)
        return self.parallel_group.all_reduce(embeddings)

    def _in_stream_capture(self) -> bool:
        """True while an XPU command-graph capture owns the current stream.

        L0 command-graph builds reject cross-stream event joins recorded
        into the graph ("Event dependency from handler::depends_on does not
        correspond to a node within the graph"), which kills FULL-graph
        capture at the pinned-lookup join (Stream.wait_stream on the
        prefetch stream). During capture the lookup must run directly on
        the capture stream; the pinned slabs are immutable, so replay
        re-reads them bitwise-exact through the captured kernel. Every
        non-capture path (eager prefill, PIECEWISE eager segments) keeps
        the side-stream overlap, and CUDA is unaffected.
        """
        return _is_xpu() and self._stream_mod.is_current_stream_capturing()

    @eager_break_during_capture
    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        ngram_ids: torch.Tensor,
    ) -> None:
        """Gather ETP IDs and launch their UVA lookup on the side stream."""
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        active_output = self._prefetch_buffer[: gathered_ids.shape[0]]
        if self._in_stream_capture():
            self._lookup(gathered_ids, output=active_output)
            return
        prefetch_stream = self._prefetch_stream
        prefetch_stream.wait_stream(self._stream_mod.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        with self._stream_mod.stream(prefetch_stream):
            self._lookup(gathered_ids, output=active_output)

    @eager_break_during_capture
    def _finalize_prefetch(
        self,
        prefetch_output: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Join the side stream, reduce ETP shards, and select local rows."""
        if not self._in_stream_capture():
            self._stream_mod.current_stream().wait_stream(self._prefetch_stream)
        slot_size, slot_offset = self._get_dp_gather_slot(output.shape[0])
        active_output = prefetch_output[: slot_size * self.etp_data_parallel_size]
        embeddings = self._reduce_etp_embeddings(active_output)
        embeddings = self._select_embeddings(
            embeddings,
            output.shape[0],
            slot_offset,
        )
        if output.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # B70 0007: move FP8 rows as bytes (no float8 copy kernel).
            output.view(torch.uint8).copy_(
                embeddings.flatten(-2).view(torch.uint8)
            )
            return
        output.copy_(embeddings.flatten(-2))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Finish the pinned lookup into graph-owned output storage."""
        output = self._prefetch_buffer.new_empty(
            (hidden_states.shape[0], self._output_dim)
        )
        self._finalize_prefetch(self._prefetch_buffer, output)
        return output

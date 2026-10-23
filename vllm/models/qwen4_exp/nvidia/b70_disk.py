# SPDX-License-Identifier: Apache-2.0
"""Bounded BF16 Disk PLE ported from the corrected 1.0.4 release.

The mapping, selected-row gather, TP ownership and DMA/graph lifetime fences
retain the qualified semantics. v0.32 uses the split start/finalize API.
"""

import ctypes
import os
import threading
from typing import ClassVar

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..common.ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)
_disk_bf16_quarantine: list[object] = []


def _validate_disk_bf16_config(embedding_method, params_dtype, max_total_tokens):
    config = get_current_vllm_config()
    p, s = config.parallel_config, config.scheduler_config
    if (
        not current_platform.is_xpu()
        or params_dtype != torch.bfloat16
        or not isinstance(embedding_method, Qwen4ExpPLEUnquantizedEmbeddingMethod)
        or not os.environ.get("PLE_TABLE_PATH")
        or p.tensor_parallel_size != 4
        or not p.enable_expert_parallel
        or p.pipeline_parallel_size != 1
        or p.data_parallel_size != 1
        or p.use_ubatching
        or p.use_sequence_parallel_moe
        or p.enable_hc_sp
        or p.decode_context_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or s.max_num_seqs != 4
    ):
        raise ValueError(
            "Disk PLE requires BF16 TP4+EP/PP1/DP1, four slots"
        )
    spec = config.speculative_config
    if spec is not None:
        draft = spec.draft_model_config
        if (
            spec.method != "dflash"
            or draft.architectures != ["DFlash2DraftModel"]
            or draft.dtype != torch.bfloat16
            or draft.quantization is not None
            or spec.draft_parallel_config.tensor_parallel_size != 4
            or not 1 <= spec.num_speculative_tokens
            < spec.draft_model_config.hf_config.dflash_config.get("block_size", 0)
            or spec.draft_sample_method != "probabilistic"
            or spec.rejection_sample_method != "standard"
            or spec.uses_dynamic_speculative_decoding()
            or spec.enable_adaptive_verification
        ):
            raise ValueError(
                "Disk PLE speculation requires BF16 TP4 DFlash2, trained-width "
                "proposals, conditional probabilistic drafting and standard rejection"
            )
    if any(
        os.environ.get(n) == "1"
        for n in (
            "B70_PLE_DIRECT_PINNED",
            "B70_PLE_FP8",
            "B70_PLE_INT8",
            "B70_PLE_INT8_NVME",
            "B70_PLE_INT8_NVME_LOOKAHEAD",
            "VLLM_PLE_CPU_OFFLOAD",
        )
    ):
        raise ValueError("Conflicting PLE storage mode")
    captures = config.compilation_config.cudagraph_capture_sizes or []
    if (
        max_total_tokens != s.max_num_batched_tokens
        or max_total_tokens != 256
        or any(n <= 0 or n > max_total_tokens for n in captures)
    ):
        raise ValueError(
            "Phase 1 Disk PLE requires bounded 256-token dispatch/capture capacity"
        )
    return max_total_tokens


def _protect_disk_bf16_mapping(table: torch.Tensor, path: str) -> None:
    """Make the torch-owned file mapping read-only, without touching its pages."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    start = table.data_ptr() // page_size * page_size
    end = (
        (table.data_ptr() + table.numel() * table.element_size() + page_size - 1)
        // page_size
        * page_size
    )
    stat = os.stat(path)
    cursor = start
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            left, right = (int(value, 16) for value in fields[0].split("-"))
            if right <= cursor or left >= end:
                continue
            major, minor = (int(value, 16) for value in fields[3].split(":"))
            if (
                left > cursor
                or "p" not in fields[1]
                or int(fields[4]) != stat.st_ino
                or os.makedev(major, minor) != stat.st_dev
            ):
                raise ValueError(
                    "PLE tensor is not a private mmap of the selected BF16 file"
                )
            cursor = min(right, end)
    if cursor != end:
        raise ValueError("PLE tensor mapping does not cover the selected BF16 table")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int
    if libc.mprotect(start, end - start, 1) != 0:  # PROT_READ
        raise OSError(ctypes.get_errno(), "Cannot protect BF16 PLE mmap")


class Qwen4ExpPLEDiskBF16Embedding(Qwen4ExpPLEPinnedHostEmbedding):
    """Immutable torch mmap plus one serialized, bounded XPU DMA stage.

    Only the producer changes. The inherited forward still reduces owned
    contributions and copies into graph-owned output before GPU PLE math.
    """

    disk_bf16: ClassVar[bool] = True

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
        reduce_results: bool = True,
    ) -> None:
        capacity = _validate_disk_bf16_config(
            embedding_method,
            params_dtype,
            max_total_tokens,
        )
        if (num_embeddings, embedding_dim, num_ngram_heads) != (320001536, 160, 16):
            raise ValueError(
                "Disk BF16 PLE supports only the original B70 table layout"
            )
        Qwen4ExpPLEEmbedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        if self.etp_data_parallel_size != 1 or self.tp_size != 4:
            raise ValueError("Disk BF16 PLE requires ETP=TP=4")
        self._stream_mod = torch.xpu
        self._capacity = capacity
        self._heads = num_ngram_heads
        self._output_dim = num_ngram_heads * embedding_dim
        self._prefetch_buffer = torch.empty(
            capacity,
            num_ngram_heads,
            embedding_dim,
            dtype=params_dtype,
            device="xpu",
        )
        self._stage = torch.empty(
            capacity,
            num_ngram_heads,
            embedding_dim,
            dtype=params_dtype,
            device="cpu",
            pin_memory=True,
        )
        if not self._stage.is_pinned():
            raise RuntimeError("Disk BF16 PLE stage failed to pin on XPU")
        self._mmap_owner = None
        self._table = None
        self._pending = False
        self._copy_event = torch.xpu.Event()
        self._owner_stream = None
        self._owner_thread = None
        self._lock = threading.Lock()
        self._failed = False
        self._closed = False
        self._generation = 0
        self._prepared_tokens = 0

    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        # Metadata/loaded-name compatibility only: never reserve a TP shard.
        return torch.empty(0, embedding_dim, dtype=dtype, device="cpu")

    def weight_loader(self, *args, **kwargs) -> None:
        raise RuntimeError("Disk BF16 PLE rejects checkpoint row-shards and reloads")

    def load_mmap(self, path: str) -> None:
        if self._mmap_owner is not None or self._closed or self._failed:
            raise RuntimeError("Disk BF16 PLE rejects mmap reload")
        from ..intel_checkpoint import map_ple_table

        owner = map_ple_table(path)
        table = owner["table"] if isinstance(owner, dict) else owner
        if (
            not isinstance(table, torch.Tensor)
            or table.device.type != "cpu"
            or table.dtype != torch.bfloat16
            or tuple(table.shape) != (self.org_vocab_size, self.embedding_dim)
            or not table.is_contiguous()
            or table.requires_grad
            or table.is_pinned()
        ):
            raise ValueError(
                "Disk BF16 PLE requires an exact contiguous CPU BF16 table"
            )
        _protect_disk_bf16_mapping(table, path)
        self._mmap_owner = owner
        self._table = table
        logger.info(
            "Disk BF16 PLE mmap ready: shape=%s, capacity=%d, pinned_stage_bytes=%d, "
            "weight_elements=%d, tp_range=[%d,%d)",
            tuple(table.shape),
            self._capacity,
            self._stage.numel() * self._stage.element_size(),
            self.weight.numel(),
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
        )

    def _check_capacity(self, num_tokens: int) -> None:
        if not 0 <= num_tokens <= self._capacity:
            raise ValueError(
                f"Disk BF16 PLE stage overflow: {num_tokens} > {self._capacity}"
            )
        if self._closed or self._failed:
            raise RuntimeError("Disk BF16 PLE stage is closed or quarantined")
        if self._stream_mod.is_current_stream_capturing():
            raise RuntimeError(
                "Disk BF16 PLE host preparation cannot run inside capture"
            )

    def _wait_for_stage(self) -> None:
        if self._pending:
            try:
                self._copy_event.synchronize()
            except BaseException:
                self._failed = True
                _disk_bf16_quarantine.append(self)
                raise
            self._pending = False

    def _fill_stage(self, ids_cpu: torch.Tensor, num_tokens: int) -> None:
        """CPU-only selected-row gather; pads/non-owned contributions are zero."""
        self._stage.zero_()
        ids = ids_cpu[:num_tokens].reshape(-1)
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.org_vocab_size):
            raise ValueError("Disk BF16 PLE received an out-of-table row ID")
        owned = (ids >= self.shard_indices.org_vocab_start_index) & (
            ids < self.shard_indices.org_vocab_end_index
        )
        # Advanced indexing allocates at most dispatch capacity, never a shard.
        self._stage[:num_tokens].view(-1, self.embedding_dim)[owned] = self._table[
            ids[owned]
        ]

    def prepare(
        self, ngram_ids: torch.Tensor | None, num_tokens: int, num_tokens_padded: int
    ) -> None:
        """Refresh the fixed GPU producer outside forward, including FULL replay."""
        self._check_capacity(num_tokens_padded)
        if not 0 <= num_tokens <= num_tokens_padded:
            raise ValueError("Disk BF16 PLE invalid actual/padded token counts")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent Disk BF16 PLE stage preparation")
        try:
            stream = self._stream_mod.current_stream()
            thread = threading.get_ident()
            # The worker RPC loop serializes prepare -> forward/replay. Host
            # ownership may change only between these submitted dispatches.
            # Recheck after acquiring the lock: close/failure may have raced us.
            self._check_capacity(num_tokens_padded)
            self._wait_for_stage()  # CPU DMA source must be safe before refill.
            if self._owner_stream is not None and (
                stream != self._owner_stream or thread != self._owner_thread
            ):
                # The copy event precedes the model's GPU reads. Drain the OLD
                # stream as well, including FULL replay which bypasses Python.
                # This occurs only on ownership changes, outside capture; the
                # ordinary same-owner path retains current-stream ordering.
                try:
                    self._owner_stream.synchronize()
                except BaseException:
                    self._failed = True
                    _disk_bf16_quarantine.append(self)
                    raise
            # Publish the new owner only after both fences have succeeded.
            self._owner_stream, self._owner_thread = stream, thread
            if ngram_ids is None:
                # Deterministic dummy contents before warmup/capture; no table IO.
                self._stage.zero_()
            else:
                if (
                    self._table is None
                    or ngram_ids.dtype != torch.int64
                    or tuple(ngram_ids.shape) != (num_tokens_padded, self._heads)
                    or ngram_ids.device != self._prefetch_buffer.device
                ):
                    raise ValueError(
                        "Disk BF16 PLE missing mmap or invalid dispatch IDs"
                    )
                ids_cpu = ngram_ids.to(device="cpu", non_blocking=False)
                self._fill_stage(ids_cpu, num_tokens)
            if num_tokens_padded:
                # Same-stream ordering prevents overwrite while a previous graph
                # reads the GPU buffer. The event protects the CPU source.
                self._pending = True
                try:
                    self._prefetch_buffer[:num_tokens_padded].copy_(
                        self._stage[:num_tokens_padded],
                        non_blocking=True,
                    )
                    self._copy_event.record(stream)
                except BaseException:
                    self._failed = True
                    _disk_bf16_quarantine.append(self)
                    raise
            self._generation += 1
            self._prepared_tokens = num_tokens_padded
        finally:
            self._lock.release()

    def start_prefetch(self, ngram_ids) -> None:
        return None

    @eager_break_during_capture
    def _finalize_prefetch(self, prefetch_output, output) -> None:
        if self._closed or self._failed or self._prepared_tokens != output.shape[0]:
            raise RuntimeError(
                "Disk BF16 PLE consumer has no matching prepared dispatch"
            )
        embeddings = self._reduce_etp_embeddings(prefetch_output[: output.shape[0]])
        output.copy_(embeddings.flatten(-2))

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        output = self._prefetch_buffer.new_empty((total_tokens, self._output_dim))
        self._finalize_prefetch(self._prefetch_buffer, output)
        return output

    def close(self) -> None:
        # Worker lifetime owns all buffers. Explicit teardown fences before free.
        if getattr(self, "_closed", True):
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close a busy Disk BF16 PLE stage")
        try:
            if self._failed:
                raise RuntimeError("Cannot release a quarantined Disk BF16 PLE stage")
            self._wait_for_stage()
            # Also drain graph readers of the GPU destination.
            if self._owner_stream is not None:
                try:
                    self._owner_stream.synchronize()
                except BaseException:
                    self._failed = True
                    _disk_bf16_quarantine.append(self)
                    raise
            self._closed = True
        finally:
            self._lock.release()

    def __del__(self):
        try:
            self.close()
        except BaseException:
            _disk_bf16_quarantine.append(self)

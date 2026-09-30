# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs
        # The NVMe PLE lookahead reads the next prefill chunk's tokens from
        # here in pre_forward (same step, same stream).
        self._req_states = req_states

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        return model_inputs

    # ------------------------------------------------------------------
    # NVMe-backed PLE table (see common/ple_nvme.py)
    # ------------------------------------------------------------------

    def _nvme_modules(self) -> list[nn.Module]:
        """PLE n-gram modules served from NVMe (resolved once, then cached)."""
        modules = getattr(self, "_nvme_modules_cached", None)
        if modules is not None:
            return modules
        from .ngram_embedding import Qwen4ExpNGramEmbedding

        modules = [
            module
            for module in self.model.modules()
            if isinstance(module, Qwen4ExpNGramEmbedding) and module._nvme_active
        ]
        self._nvme_modules_cached = modules
        if not modules:
            return modules
        self._nvme_stage = torch.empty(
            self.max_num_tokens + self.max_num_reqs * self.ngram_context_len,
            dtype=torch.int32,
            pin_memory=True,
        )
        self._nvme_lookahead_on = any(m.nvme_lookahead_on for m in modules)
        if self._nvme_lookahead_on:
            self._nvme_lookahead_tokens = int(
                os.environ.get("B70_PLE_INT8_NVME_LOOKAHEAD_TOKENS", "0")
                or self.max_num_tokens
            )
            # Next-chunk tokens plus the ngram context before each chunk.
            self._nvme_lookahead_stage = torch.empty(
                self._nvme_lookahead_tokens
                + self.max_num_reqs * self.ngram_context_len,
                dtype=torch.int32,
                pin_memory=True,
            )
        return modules

    def _nvme_lookahead_plan(self, input_batch: InputBatch):
        """Predict the next step's prefill chunks and queue the D2H copy of
        their tokens (and the ngram context before each) before the sync."""
        from ..common.ple_nvme import current_chunk_keys, plan_next_chunks

        num_reqs = input_batch.num_reqs
        idx = input_batch.idx_mapping_np[:num_reqs]
        computed = input_batch.num_computed_tokens_np[:num_reqs]
        scheduled = input_batch.num_scheduled_tokens[:num_reqs]
        prefill_len = input_batch.prefill_len_np[:num_reqs]
        keys = current_chunk_keys(idx, computed, scheduled, prefill_len)
        if not input_batch.has_prefill:
            return keys, []
        plan = plan_next_chunks(
            idx, computed, scheduled, prefill_len, self._nvme_lookahead_tokens
        )
        all_tokens = self._req_states.all_token_ids.gpu
        stage = self._nvme_lookahead_stage
        ctx_len = self.ngram_context_len
        spans = []
        offset = 0
        for req_idx, start, end in plan:
            low = max(0, start - ctx_len)
            count = end - low
            stage[offset : offset + count].copy_(
                all_tokens[req_idx, low:end], non_blocking=True
            )
            spans.append((req_idx, start, end, offset, start - low))
            offset += count
        return keys, spans

    def _nvme_lookahead_submit(self, modules, spans) -> None:
        """Build the predicted chunks' tokens, query_start_loc and context on
        the host (after the sync) and hand them to each module's lookahead."""
        if not spans:
            return
        staged = self._nvme_lookahead_stage.numpy()
        ctx_len = self.ngram_context_len
        context = np.full((len(spans), ctx_len), self.ngram_eos_token_id, np.int64)
        query_start_loc = np.zeros(len(spans) + 1, dtype=np.int64)
        tokens = []
        keys = set()
        for i, (req_idx, start, end, offset, before) in enumerate(spans):
            # Same rule as _prepare_ngram_context: positions < 0 are EOS.
            if before:
                context[i, ctx_len - before :] = staged[offset : offset + before]
            tokens.append(staged[offset + before : offset + before + end - start])
            query_start_loc[i + 1] = query_start_loc[i] + (end - start)
            keys.add((req_idx, start))
        flat = np.concatenate(tokens).astype(np.int64)
        for module in modules:
            module.nvme_lookahead_submit(
                frozenset(keys), flat, query_start_loc, context
            )

    def pre_forward(
        self,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
    ) -> None:
        """Resolve NVMe-backed PLE rows on the host before the forward.

        One D2H copy of the real tokens and the n-gram context (int32, a few
        KB) and one sync, which also waits for the previous step's sampling
        (the decode token exists only on the device). The n-gram ids are then
        computed on the host (bit-exact with ``compute_ngram_ids``), resolved
        against the row cache, the misses read from the table file, and the
        gather launched into the static prefetch buffer. No-op unless the
        NVMe-backed table is active.
        """
        if not self.uses_ngram_embedding:
            return
        modules = self._nvme_modules()
        if not modules:
            return
        num_tokens = input_batch.num_tokens
        num_reqs = input_batch.num_reqs
        ctx_len = self.ngram_context_len
        stage = self._nvme_stage
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            input_ids = input_batch.input_ids
        stage[:num_tokens].copy_(input_ids[:num_tokens], non_blocking=True)
        stage[num_tokens : num_tokens + num_reqs * ctx_len].copy_(
            model_inputs["ngram_context"][:num_reqs].reshape(-1), non_blocking=True
        )
        lookahead_keys = None
        spans: list = []
        if self._nvme_lookahead_on:
            lookahead_keys, spans = self._nvme_lookahead_plan(input_batch)
        torch.accelerator.current_stream(input_ids.device).synchronize()
        # The device is idle from here until the gather and the forward are
        # launched: this is the per-step host bubble the stats report.
        t_start = time.perf_counter()
        staged = stage.numpy()
        tokens = staged[:num_tokens]
        context = staged[num_tokens : num_tokens + num_reqs * ctx_len].reshape(
            num_reqs, ctx_len
        )
        query_start_loc = input_batch.query_start_loc_np[: num_reqs + 1]
        for module in modules:
            module.nvme_pre_forward(
                tokens,
                query_start_loc,
                context,
                input_batch.num_tokens_after_padding,
                t_start,
                lookahead_keys=lookahead_keys,
            )
        if spans:
            # After the gather launch: the lookahead reads while this step's
            # forward runs.
            self._nvme_lookahead_submit(modules, spans)

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]

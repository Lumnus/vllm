# SPDX-License-Identifier: Apache-2.0
"""B70-Q27-0006: a reduced draft vocabulary for the DFlash2 drafter on Qwen3_5 targets.

B70_DFLASH_DRAFT_VOCAB=<file of token ids, whitespace separated> gives the DFlash2
drafter its own small head: the target lm_head rows of a token shortlist (e.g. 67,263 of
248,320), used only for the drafter's candidate top-k (compute_candidates). The target
keeps its full-vocab head, B70_HEAD_FP8 included, and verifies every draft against the
full distribution. Idea and list: steveseguin/b70-optimization-lab R294 (Unlicense).

Exactness: DFlash2's proposal distribution is already supported on its top-k candidates
only (draft_logits holds the realized scores at the K candidate ids and -inf elsewhere,
CandidateSampler). The shortlist only changes which K ids are candidates; the walk and
the rejection sampler read the same cached scores, so q stays one consistent
distribution and standard rejection with residual max(p - q, 0) over the full vocab
keeps the output exact.

Rows are spread evenly over the TP ranks (shortlist position i -> rank i % tp), not by
vocab shard: the frequency list sits mostly in the low ids, so by-shard rank 0 would
read 58K rows and rank 1 9K. Rows are exchanged once at load over the TP CPU (gloo)
group, from the BF16 shards (before B70_HEAD_FP8 releases them at the first forward).
Under B70_HEAD_FP8=1 on XPU the copy is FP8 W8A16 with the target's own per-row recipe
(b70_fp8_head_of), so a shortlisted row scores exactly as the target's FP8 row would.

Env unset -> nothing is built; compute_candidates takes the stock path. Only for a
DFlash2 drafter whose target language model is a Qwen3_5ForCausalLMBase (Flash-Next's
Qwen4Exp targets are left alone)."""

import os

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "B70_DFLASH_DRAFT_VOCAB"
_CHUNK_ROWS = 4096


def read_draft_vocab(path: str, vocab_size: int) -> list[int]:
    """Sorted unique token ids from a whitespace-separated file; each in [0, vocab)."""
    with open(path) as f:
        ids = sorted({int(t) for t in f.read().split()})
    if not ids:
        raise ValueError(f"{ENV}: {path} holds no token ids")
    if ids[0] < 0 or ids[-1] >= vocab_size:
        raise ValueError(
            f"{ENV}: ids out of range [0, {vocab_size}): {ids[0]}..{ids[-1]}"
        )
    return ids


def rank_assignment(ids: list[int], tp_size: int, tp_rank: int) -> list[int]:
    """Shortlist positions i with i % tp_size == tp_rank: disjoint, balanced (+-1)."""
    return ids[tp_rank::tp_size]


def target_is_qwen3_5(target_model: nn.Module) -> bool:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    lm = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    return isinstance(lm, Qwen3_5ForCausalLMBase)


class B70DraftVocabHead(nn.Module):
    """This rank's shortlist rows of the target lm_head and their global ids."""

    def __init__(self, weight: torch.Tensor, ids: torch.Tensor, n_total: int):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("ids", ids, persistent=False)
        self.n_total = n_total
        self.fp8 = None

    def maybe_fp8(self) -> None:
        if os.environ.get("B70_HEAD_FP8", "0") != "1":
            return
        from vllm.platforms import current_platform

        if not current_platform.is_xpu():
            return
        from vllm.models.qwen4_exp.nvidia.b70_fp8_head import b70_fp8_head_of

        self.fp8 = b70_fp8_head_of(self, release_bf16=True)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp8 is not None:
            return self.fp8.apply(self, x)
        return F.linear(x, self.weight)

    def get_top_k_tokens(
        self, processor, hidden_states: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LogitsProcessor.get_top_k_tokens over the shortlist: local top-k, global ids,
        TP all-gather of k values + ids, global top-k, then scale / soft cap on the
        k."""
        from vllm.distributed import tensor_model_parallel_all_gather
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

        logits = self.project(hidden_states)
        values, local = torch.topk(logits, k, dim=-1)
        ids = self.ids[local]
        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = torch.topk(values, k, dim=-1)
            ids = ids.gather(-1, selected)
        values = values.float()
        if processor.scale != 1.0:
            values = values * processor.scale
        if processor.soft_cap is not None:
            values = torch.tanh(values / processor.soft_cap) * processor.soft_cap
        return ids, values


def _gather_rows(
    lm_head: nn.Module, mine: list[int], all_ids: list[int], tp_size: int, tp_rank: int
) -> torch.Tensor:
    """BF16 rows for `mine` from the vocab-sharded lm_head: each chunk of shortlist rows
    is
    filled by its owning rank (zeros elsewhere) and summed over the TP CPU group. x + 0
    is
    exact, so every row arrives bit-identical."""
    weight = lm_head.weight
    hidden = weight.shape[1]
    si = lm_head.shard_indices
    start, end = si.org_vocab_start_index, si.org_vocab_end_index
    if tp_size == 1:
        idx = torch.tensor([t - start for t in mine], device=weight.device)
        return weight.index_select(0, idx).contiguous()
    from vllm.distributed.parallel_state import get_tp_group

    cpu_group = get_tp_group().cpu_group
    out = torch.empty((len(mine), hidden), dtype=weight.dtype, device=weight.device)
    filled = 0
    for c in range(0, len(all_ids), _CHUNK_ROWS):
        chunk = all_ids[c : c + _CHUNK_ROWS]
        buf = torch.zeros((len(chunk), hidden), dtype=torch.float32)
        own = [j for j, t in enumerate(chunk) if start <= t < end]
        if own:
            rows = torch.tensor([chunk[j] - start for j in own], device=weight.device)
            buf[torch.tensor(own)] = weight.index_select(0, rows).float().cpu()
        torch.distributed.all_reduce(buf, group=cpu_group)
        keep = [j for j in range(len(chunk)) if (c + j) % tp_size == tp_rank]
        out[filled : filled + len(keep)] = buf[torch.tensor(keep)].to(
            device=weight.device, dtype=weight.dtype
        )
        filled += len(keep)
    assert filled == len(mine)
    return out


def maybe_build_draft_vocab_head(
    draft_model: nn.Module, target_model: nn.Module, top_k: int
) -> B70DraftVocabHead | None:
    """Called once from DFlash2Speculator.load_draft_model, on every TP rank."""
    path = os.environ.get(ENV, "").strip()
    if not path:
        return None
    if not target_is_qwen3_5(target_model):
        logger.warning_once(
            "B70 DFlash draft vocab: target %s is not a Qwen3_5 model; off",
            type(target_model).__name__,
        )
        return None
    lm_head = getattr(draft_model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dim() != 2
        or weight.numel() == 0
        or weight.dtype != torch.bfloat16
        or getattr(lm_head, "shard_indices", None) is None
    ):
        logger.warning_once(
            "B70 DFlash draft vocab: the shared lm_head is not a plain BF16 "
            "vocab-parallel weight (already FP8, or quantized); off"
        )
        return None
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    if getattr(lm_head, "tp_size", tp_size) != tp_size:
        raise ValueError(f"{ENV}: lm_head TP {lm_head.tp_size} != TP group {tp_size}")
    vocab = lm_head.org_vocab_size
    ids = read_draft_vocab(path, vocab)
    mine = rank_assignment(ids, tp_size, tp_rank)
    if len(mine) < top_k:
        raise ValueError(
            f"{ENV}: {len(mine)} rows on TP rank {tp_rank} < selector_top_k {top_k}"
        )
    with torch.no_grad():
        rows = _gather_rows(lm_head, mine, ids, tp_size, tp_rank)
    head = B70DraftVocabHead(
        rows, torch.tensor(mine, dtype=torch.int64, device=weight.device), len(ids)
    )
    head.maybe_fp8()
    logger.info_once("B70 DFlash draft vocab: %d tokens", len(ids))
    logger.info(
        "B70 DFlash draft vocab: TP rank %d/%d scores %d rows (%s, %.0f MB) of %d; "
        "target lm_head stays full vocab (%d)",
        tp_rank,
        tp_size,
        len(mine),
        "fp8 w8a16" if head.fp8 is not None else str(rows.dtype),
        (head.fp8.w8.numel() if head.fp8 is not None else rows.numel() * 2) / 1e6,
        len(ids),
        vocab,
    )
    return head

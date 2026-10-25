# SPDX-License-Identifier: Apache-2.0
"""B70-QSA qualification hooks. Every hook is off unless its env var is set.

The flags are read once at import. With all of them unset nothing here runs on
the serving path: the call sites test a module-level constant first.

  B70_QSA_DEBUG_RING=raise|warn   host mirror of each raw-key ring's write
                                  frontier, checked at every metadata build.
                                  Catches a step that needs committed keys of
                                  the open compression group which the ring
                                  does not hold: an unaligned prefix/offload
                                  hit, a ring too narrow for the speculative
                                  lookahead, or a ring block shared by two
                                  live requests. Forces a device->host copy of
                                  three small tensors per build.
  B70_QSA_DEBUG_DUMP=<dir>        after each step that carries prefill rows,
                                  rank 0 writes every prefill request's
                                  compressed-key rows (logical order) and the
                                  selected block ids of its last row, for the
                                  layers in B70_QSA_DEBUG_DUMP_LAYERS (default
                                  "0"). Decode-only steps (the graph-captured
                                  ones) are never dumped.
  B70_QSA_DEBUG_TIES=1            count top-k rows whose selection boundary is
                                  an exact score tie (the rows where native and
                                  torch top-k may legitimately pick different
                                  blocks). Eager steps only; logs a running
                                  total every B70_QSA_DEBUG_TIES_PERIOD rows.
  B70_QSA_GRAPH_CHECK=1           at C1 target-graph capture only, compare the
                                  QSA side-cache rows the step writes after the
                                  eager pass and after the replay, and the null
                                  block before and after; a mismatch disables
                                  that graph key.
"""

import os
import threading
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.b70_qsa_debug")

TAG = "B70-QSA-DEBUG"

RING_MODE = os.environ.get("B70_QSA_DEBUG_RING", "").strip().lower()
if RING_MODE in ("1", "true", "yes", "on"):
    RING_MODE = "raise"
if RING_MODE not in ("", "raise", "warn"):
    raise ValueError(f"B70_QSA_DEBUG_RING must be raise|warn, got {RING_MODE!r}")

DUMP_DIR = os.environ.get("B70_QSA_DEBUG_DUMP", "").strip()
DUMP_LAYERS = frozenset(
    int(x) for x in os.environ.get("B70_QSA_DEBUG_DUMP_LAYERS", "0").split(",") if x
)
TIES = os.environ.get("B70_QSA_DEBUG_TIES", "").strip() in ("1", "true", "on")
TIES_PERIOD = int(os.environ.get("B70_QSA_DEBUG_TIES_PERIOD", "100000"))
GRAPH_CHECK = os.environ.get("B70_QSA_GRAPH_CHECK", "").strip() in ("1", "true", "on")


class QSARingContractError(RuntimeError):
    pass


class QSARingContract:
    """Host mirror of each ring block's last written logical position.

    A step for one request computes rows ``[s, s + q)`` (``s`` = seq_len - q).
    Compressing the group that contains ``s`` reads the committed keys of
    positions ``[s - s % ratio, s)`` from the ring. They are there iff the ring
    was last written through position ``>= s - 1`` and the newest write
    ``last`` has not wrapped onto them: ``last - capacity + 1 <= s - s % ratio``.
    Rows written past ``s - 1`` are rejected drafts the step rewrites.
    """

    def __init__(self, capacity: int, compress_ratio: int, mode: str) -> None:
        self.capacity = capacity
        self.compress_ratio = compress_ratio
        self.mode = mode
        self.last: dict[int, int] = {}
        self.counts = {
            "steps": 0,
            "checked": 0,
            "first_sight_unaligned": 0,
            "missing": 0,
            "overwritten": 0,
        }
        self._lock = threading.Lock()

    def _violation(self, kind: str, message: str) -> None:
        self.counts[kind] += 1
        n = self.counts[kind]
        if self.mode == "raise" and kind != "first_sight_unaligned":
            raise QSARingContractError(f"{TAG} ring {kind}: {message}")
        if n <= 50 or n % 1000 == 0:
            logger.warning(
                "%s ring %s #%d: %s counts=%s", TAG, kind, n, message, self.counts
            )

    def observe(
        self,
        ring_blocks: list[int],
        query_start_loc: list[int],
        seq_lens: list[int],
    ) -> None:
        ratio, cap = self.compress_ratio, self.capacity
        with self._lock:
            self.counts["steps"] += 1
            for r, block in enumerate(ring_blocks):
                q = query_start_loc[r + 1] - query_start_loc[r]
                if q <= 0 or block <= 0:
                    continue  # empty row or a dummy/padding request (null block)
                s = seq_lens[r] - q
                if s % ratio:
                    self.counts["checked"] += 1
                    open_start = s - s % ratio
                    last = self.last.get(block)
                    where = f"block={block} start={s} rows={q} last_write={last}"
                    if last is None:
                        # Either warmup or a fresh request whose prefix hit is
                        # not a multiple of the ratio. Hard in real traffic.
                        self._violation("first_sight_unaligned", where)
                    elif last < s - 1:
                        self._violation(
                            "missing",
                            f"{where}: ring never received positions "
                            f"[{last + 1}, {s}) of the open group",
                        )
                    elif last - cap + 1 > open_start:
                        self._violation(
                            "overwritten",
                            f"{where}: capacity {cap} lets position {last} "
                            f"overwrite committed position {open_start}",
                        )
                self.last[block] = s + q - 1


_ring_contracts: dict[tuple[int, int], QSARingContract] = {}


def ring_contract(capacity: int, compress_ratio: int) -> QSARingContract:
    key = (capacity, compress_ratio)
    contract = _ring_contracts.get(key)
    if contract is None:
        contract = _ring_contracts[key] = QSARingContract(
            capacity, compress_ratio, RING_MODE
        )
    return contract


def observe_ring_metadata(common_attn_metadata, capacity: int, ratio: int) -> None:
    """Called from QSAMetadataBuilder.build for the ring group (debug only)."""
    if _is_capturing():
        return
    num_reqs = common_attn_metadata.query_start_loc.shape[0] - 1
    if num_reqs <= 0 or not all(common_attn_metadata.block_table_tensor.shape):
        return
    # Device values: under async scheduling the CPU copies can be stale.
    qsl = common_attn_metadata.query_start_loc[: num_reqs + 1].cpu().tolist()
    seq = common_attn_metadata.seq_lens[:num_reqs].cpu().tolist()
    blocks = common_attn_metadata.block_table_tensor[:num_reqs, 0].cpu().tolist()
    ring_contract(capacity, ratio).observe(blocks, qsl, seq)


def _is_capturing() -> bool:
    for mod in (getattr(torch, "xpu", None), getattr(torch, "cuda", None)):
        fn = getattr(mod, "is_current_stream_capturing", None)
        try:
            if fn is not None and fn():
                return True
        except Exception:
            pass
    return False


is_capturing = _is_capturing


def layer_index(prefix: str) -> int | None:
    parts = prefix.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


_dump_step = [0]


def dump_prefill_state(
    prefix: str,
    compressed_metadata,
    compressed_cache: torch.Tensor,
    selected: torch.Tensor,
) -> None:
    """Write prefill requests' compressed rows and last-row selections."""
    if not DUMP_DIR or _is_capturing():
        return
    layer = layer_index(prefix)
    if layer is None or layer not in DUMP_LAYERS:
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    m = compressed_metadata
    if m.num_decode_tokens >= m.num_actual_tokens:
        return
    qsl = m.query_start_loc.cpu().tolist()
    seq = m.seq_lens.cpu().tolist()
    page = compressed_cache.shape[1]
    requests = []
    for r in range(m.num_decodes, len(qsl) - 1):
        q = qsl[r + 1] - qsl[r]
        if q <= 0:
            continue
        groups = seq[r] // m.compress_ratio
        logical = torch.arange(groups, device=compressed_cache.device)
        pages = m.block_table[r, logical // page].long()
        rows = compressed_cache[pages, logical % page, 0].float().cpu()
        last = qsl[r + 1] - 1
        requests.append(
            {
                "seq_len": seq[r],
                "query_len": q,
                "compressed": rows,
                "last_row_selected": selected[last].cpu(),
            }
        )
    if not requests:
        return
    _dump_step[0] += 1
    path = Path(DUMP_DIR)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"layer": layer, "step": _dump_step[0], "requests": requests},
        path / f"qsa-L{layer:02d}-{os.getpid()}-{_dump_step[0]:07d}.pt",
    )


_tie_counts = {"rows": 0, "tied": 0}


def count_boundary_ties(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_indices: torch.Tensor,
    block_topk: int,
) -> None:
    """Count rows whose k-th score is shared by a column outside the selection."""
    if _is_capturing():
        return
    tied = boundary_tie_rows(logits, visible_blocks, block_indices, block_topk)
    _tie_counts["rows"] += int(logits.shape[0])
    before = _tie_counts["tied"]
    _tie_counts["tied"] += int(tied.sum().item())
    if _tie_counts["rows"] // TIES_PERIOD != (
        _tie_counts["rows"] - logits.shape[0]
    ) // TIES_PERIOD or (before == 0 and _tie_counts["tied"]):
        logger.info(
            "%s top-k boundary ties: %d of %d rows",
            TAG,
            _tie_counts["tied"],
            _tie_counts["rows"],
        )


def boundary_tie_rows(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_indices: torch.Tensor,
    block_topk: int,
) -> torch.Tensor:
    """Rows where the selection is not unique: more than k visible columns
    score at least the smallest selected score. Device ops only."""
    rows, width = logits.shape
    k = min(block_topk, width)
    sel = block_indices[:rows, :k].long()
    valid = sel >= 0
    picked = logits.gather(1, sel.clamp_min(0))
    inf = torch.tensor(float("inf"), device=logits.device, dtype=logits.dtype)
    threshold = torch.where(valid, picked, inf).min(dim=1).values
    visible = visible_blocks[:rows].to(torch.int64).clamp(0, width)
    live = torch.arange(width, device=logits.device).unsqueeze(0) < visible.unsqueeze(1)
    at_least = ((logits >= threshold.unsqueeze(1)) & live).sum(dim=1)
    return (visible > k) & (at_least > k)

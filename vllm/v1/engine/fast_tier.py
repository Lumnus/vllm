# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine core's side of a card tier resize (runtime control
``tiers.fast``), exchanging device memory with the KV cache.

The workers move their card rows (``vllm/model_executor/layers/fused_moe/
xpu_fast_tier.py``); this controller makes the memory for a grow and gives
back the memory of a shrink, the way an encoder tower's residency does
(``tower_residency.py``):

- grow: every rank reports the bytes the new rows map; the KV cache gives
  back that much on every rank by a shrink that drains (the fewest tail
  blocks that release it, no preemption unless asked, a deadline); once the
  shrink has released, between two steps, every rank maps the rows and moves
  the hottest slots in (two phases: every rank checks, then applies; a rank
  that fails is undone on the others), and the KV cache grows back to its
  size before if the rows could not be taken. A shrink that is refused or
  aborted leaves the card tier as it was.
- shrink: every rank moves the coldest slots out and releases the rows'
  memory within the call; then the KV cache grows by what was released
  (the grow's own vote decides what it gets).

One resize at a time, and none while another KV cache resize runs. Advanced
in the busy loop right after the KV resize, so the rows are mapped in the
same iteration as the KV memory is released, before any step can take it.
"""

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class FastTierState(StrEnum):
    IDLE = "idle"
    # a grow that waits for EPLB before it takes KV memory
    WAITING = "waiting"
    DRAINING = "draining"
    COMMITTING = "committing"
    FINISHED = "finished"
    UNCHANGED = "unchanged"
    FAILED = "failed"

    @property
    def active(self) -> bool:
        return self in (
            FastTierState.WAITING,
            FastTierState.DRAINING,
            FastTierState.COMMITTING,
        )


@dataclass
class FastTierResize:
    direction: str | None
    from_rows: int
    to_rows: int
    state: FastTierState
    bytes_per_rank: int = 0
    kv_blocks_before: int = 0
    kv_blocks_after: int | None = None
    kv: dict[str, Any] | None = None
    from_kv: bool = True
    message: str = ""
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    rolled_back: bool | None = None
    # the KV cache resize this one waits for
    kv_resize: Any = None
    # how long a commit waits for EPLB's rearrangement or tier pass in flight
    deadline_s: float = 60.0
    # when the commit first found EPLB moving experts (None: it did not)
    eplb_wait_since: float | None = None
    # a grow's KV shrink may preempt at its deadline
    preempt: bool = False
    # the ranks refused at their check (nothing was changed on any rank)
    refused: bool = False
    # torch's room on each rank after the operation (the rows' readout)
    torch_after: list[Any] | None = None

    def record(self, rows_now: int | None) -> dict[str, Any]:
        end = self.ended if self.ended is not None else time.monotonic()
        out = {
            "state": self.state.value,
            "direction": self.direction,
            "from_rows": self.from_rows,
            "to_rows": self.to_rows,
            "rows_now": rows_now,
            "bytes_per_rank": self.bytes_per_rank,
            "kv_blocks_before": self.kv_blocks_before,
            "kv_blocks_after": self.kv_blocks_after,
            "kv": self.kv,
            "from_kv": self.from_kv,
            "seconds": round(end - self.started, 3),
            "message": self.message,
        }
        if self.rolled_back is not None:
            out["rolled_back"] = self.rolled_back
        if self.torch_after is not None:
            out["torch_after"] = self.torch_after
        return out


class FastTierController:
    """Held by the runtime control surface (made on first use); the busy
    loop advances it (``EngineCore._fast_tier``)."""

    DEFAULT_DEADLINE_S = 60.0

    def __init__(self, engine: Any, control: Any) -> None:
        self.engine = engine
        self.control = control
        self.resize: FastTierResize | None = None

    # -- readout ----------------------------------------------------------

    def _ranks(self, targets: list[int] | None = None) -> list[dict]:
        args: dict[str, Any] = {"only": ["tiers", "kv"]}
        if targets:
            args["fast_rows"] = targets
        return self.control._rpc("state", "", args)

    @staticmethod
    def _fast(ranks: list[dict]) -> list[dict | None]:
        return [(r.get("tiers") or {}).get("fast") for r in ranks]

    # -- the verb ---------------------------------------------------------

    def start(self, args: dict[str, Any]) -> dict[str, Any]:
        from vllm.v1.engine.runtime_control import ControlRefused

        if self.resize is not None and self.resize.state.active:
            raise ControlRefused(
                f"a fast tier resize to {self.resize.to_rows} rows is "
                f"{self.resize.state.value}; one at a time"
            )
        unknown = sorted(
            set(args) - {"rows", "delta", "deadline_s", "preempt", "from_kv"}
        )
        if unknown:
            raise ControlRefused(
                f"unknown arguments {unknown}; tiers.fast takes rows or delta, "
                "deadline_s, preempt, from_kv"
            )
        if ("rows" in args) == ("delta" in args):
            raise ControlRefused("give exactly one of rows and delta")
        ranks = self._ranks()
        fast = self._fast(ranks)
        if not fast or any(f is None for f in fast):
            raise ControlRefused(
                "the card tier is fixed in this engine: start with "
                "--expert-store-config fast_rows_max (an expert store and "
                "--cpu-offload-per-expert)"
            )
        rows_now = fast[0]["rows"]
        if any(f["rows"] != rows_now for f in fast):
            raise ControlRefused(
                "the ranks hold different card rows per layer "
                f"({[f['rows'] for f in fast]}); repair with rows set to one of them"
            )
        value = args.get("rows", args.get("delta"))
        if not isinstance(value, int) or isinstance(value, bool):
            raise ControlRefused(f"rows and delta are integers; got {value!r}")
        target = value if "rows" in args else rows_now + value
        lo = max(f["min"] for f in fast)
        hi = min(f["max"] for f in fast)
        if not lo <= target <= hi:
            raise ControlRefused(
                f"{target} card rows per layer is outside {lo}..{hi} (fast_rows_min "
                "and fast_rows_max, or the pinned rows a layer without CPU experts "
                "needs)"
            )
        deadline = args.get("deadline_s", self.DEFAULT_DEADLINE_S)
        if not isinstance(deadline, int | float) or isinstance(deadline, bool):
            raise ControlRefused(f"deadline_s is a number; got {deadline!r}")
        from_kv = bool(args.get("from_kv", True))
        preempt = bool(args.get("preempt", False))
        cache_config = self.engine.vllm_config.cache_config
        blocks = int(cache_config.num_gpu_blocks or 0)
        if target == rows_now:
            self.resize = FastTierResize(
                None, rows_now, target, FastTierState.UNCHANGED, kv_blocks_before=blocks
            )
            self.resize.ended = self.resize.started
            return self._reply()
        kv_active = (
            self.engine._kv_resize is not None and self.engine._kv_resize.state.active
        )
        if from_kv and kv_active:
            raise ControlRefused("a KV cache resize is in progress; ask after it ends")
        if target < rows_now:
            return self._shrink(rows_now, target, blocks, from_kv, float(deadline))
        return self._grow(rows_now, target, blocks, from_kv, float(deadline), preempt)

    def _reply(self) -> dict[str, Any]:
        assert self.resize is not None
        record = self.resize.record(
            self.resize.to_rows
            if self.resize.state == FastTierState.FINISHED
            else self.resize.from_rows
        )
        applied = (
            "all"
            if self.resize.state != FastTierState.FAILED
            else ("none" if self.resize.rolled_back is not False else "partial")
        )
        return {"op": "tiers.fast", "applied": applied, "fast": record}

    # -- grow -------------------------------------------------------------

    def _grow(
        self,
        rows_now: int,
        target: int,
        blocks: int,
        from_kv: bool,
        deadline: float,
        preempt: bool,
    ) -> dict[str, Any]:
        from vllm.v1.engine.runtime_control import ControlRefused

        ranks = self._ranks([target])
        need = max(f["bytes_to"][str(target)] for f in self._fast(ranks))  # type: ignore[index]
        resize = FastTierResize(
            "grow",
            rows_now,
            target,
            FastTierState.COMMITTING,
            bytes_per_rank=need,
            kv_blocks_before=blocks,
            from_kv=from_kv,
            deadline_s=deadline,
            preempt=preempt,
        )
        # Ask the ranks before the KV cache gives anything: a rearrangement
        # of EPLB in flight is waited for (then the KV shrink starts), any
        # other refusal is the answer and nothing changes.
        why = self._precheck(target)
        if why is not None and not self._eplb_busy(why):
            raise ControlRefused(why)
        if why is not None:
            self.resize = resize
            resize.state = FastTierState.WAITING
            self._waits_for_eplb(resize, {"error": why})
            return self._reply()
        if from_kv and need > 0:
            self._release(resize)
        self.resize = resize
        self.advance()
        if (
            resize.state == FastTierState.FAILED
            and resize.refused
            and resize.kv is None
        ):
            # refused by the ranks with nothing taken from the KV cache
            raise ControlRefused(resize.message)
        return self._reply()

    def _precheck(self, rows: int) -> str | None:
        """Why the ranks would refuse ``rows`` now, memory aside (EPLB, the
        range, a prepared move), or None."""
        checks = self.control._rpc(
            "check", "tiers.fast", {"rows": rows, "precheck": True}
        )
        refusals = [c for c in checks if c.get("refusal")]
        if not refusals:
            return None
        return "; ".join(f"rank {c['rank']}: {c['refusal']}" for c in refusals)

    def _eplb_busy(self, why: str) -> bool:
        return any(b in why for b in self._EPLB_BUSY)

    def _release(self, resize: FastTierResize) -> None:
        """A grow's KV shrink of the bytes the rows map; raises
        ControlRefused when the KV cache cannot give them."""
        from vllm.v1.engine.runtime_control import ControlRefused

        need = resize.bytes_per_rank
        try:
            kv = self.engine.request_kv_cache_release(
                need, urgent=resize.preempt, deadline_s=resize.deadline_s
            )
        except (ValueError, RuntimeError, NotImplementedError) as e:
            raise ControlRefused(
                f"the KV cache cannot give back {need:,} bytes per rank for "
                f"{resize.to_rows - resize.from_rows} more card rows per layer: {e}"
            ) from e
        resize.kv = kv
        resize.kv_resize = self.engine._kv_resize
        resize.state = FastTierState.DRAINING
        logger.info(
            "Fast tier: card rows per layer %d -> %d asked; the KV cache gives "
            "back %.2f GiB per rank first (shrink %d -> %d blocks, %s).",
            resize.from_rows,
            resize.to_rows,
            need / 2**30,
            resize.kv_blocks_before,
            kv.get("to_blocks", kv.get("num_blocks", -1)),
            kv.get("state"),
        )

    def advance(self) -> None:
        """Between two steps (busy loop): start the KV shrink of a grow that
        waited for EPLB, commit a grow whose KV shrink has released, or close
        one whose shrink ended otherwise."""
        resize = self.resize
        if resize is None or not resize.state.active:
            return
        if resize.state == FastTierState.WAITING:
            from vllm.v1.engine.runtime_control import ControlRefused

            why = self._precheck(resize.to_rows)
            if why is not None:
                if self._eplb_busy(why) and self._waits_for_eplb(
                    resize, {"error": why}
                ):
                    return
                self._fail(resize, why)
                return
            resize.message = ""
            if resize.from_kv and resize.bytes_per_rank > 0:
                # the KV cache's size when it starts giving (a grow of the KV
                # cache may have run while this waited)
                resize.kv_blocks_before = (
                    self.engine.vllm_config.cache_config.num_gpu_blocks
                )
                try:
                    self._release(resize)
                except ControlRefused as e:
                    self._fail(resize, str(e))
                    return
            else:
                resize.state = FastTierState.COMMITTING
        if resize.state == FastTierState.DRAINING:
            from vllm.v1.engine.kv_cache_resize import KVCacheResizeState

            kv = resize.kv_resize
            if kv is None or self.engine._kv_resize is not kv:
                self._fail(resize, "the KV cache shrink was replaced by another resize")
                return
            resize.kv = kv.record(self.engine.vllm_config.cache_config.num_gpu_blocks)
            if kv.state.active:
                return
            if kv.state != KVCacheResizeState.FINISHED:
                self._fail(
                    resize,
                    f"the KV cache shrink ended {kv.state.value}"
                    + (f" ({kv.message})" if kv.message else ""),
                )
                return
            resize.state = FastTierState.COMMITTING
        if resize.direction == "shrink":
            self._shrink_commit(resize)
        else:
            self._commit(resize)

    # EPLB owns every expert move: a resize does not start while one of its
    # rearrangements or tier passes is in flight, it waits for it.
    _EPLB_BUSY = ("is in flight", "an EPLB move is prepared")

    def _waits_for_eplb(self, resize: FastTierResize, reply: dict) -> bool:
        """Whether a refused commit is EPLB moving experts on some rank and
        the resize still has time to wait for it (then it stays committing
        and the busy loop asks again between the next steps)."""
        error = str(reply.get("error") or "")
        if reply.get("rolled_back") is False or not any(
            b in error for b in self._EPLB_BUSY
        ):
            return False
        now = time.monotonic()
        if resize.eplb_wait_since is None:
            resize.eplb_wait_since = now
            logger.info(
                "Fast tier: card rows per layer %d -> %d waits for EPLB (%s); "
                "at most %.0f s.",
                resize.from_rows,
                resize.to_rows,
                error,
                resize.deadline_s,
            )
        if resize.deadline_s > 0 and now - resize.eplb_wait_since > resize.deadline_s:
            return False
        resize.message = f"waiting for EPLB: {error}"
        return True

    def _op(self, args: dict[str, Any]) -> dict[str, Any]:
        """The worker op; a refusal at the ranks' check (nothing changed on
        any rank) as a reply, so that the caller can wait for EPLB or fail
        the resize with it."""
        from vllm.v1.engine.runtime_control import ControlRefused

        try:
            return self.control._worker_op("tiers.fast", args)
        except ControlRefused as e:
            return {"op": "tiers.fast", "applied": "none", "error": str(e),
                    "refused": True, "ranks": []}  # fmt: skip

    def _commit(self, resize: FastTierResize) -> None:
        args: dict[str, Any] = {"rows": resize.to_rows}
        if resize.from_kv and resize.kv is not None and resize.bytes_per_rank > 0:
            # the KV cache released these bytes for the rows, in this resize
            args["kv_released"] = resize.bytes_per_rank
        reply = self._op(args)
        if reply["applied"] != "all" and self._waits_for_eplb(resize, reply):
            return
        if reply["applied"] == "all":
            resize.state = FastTierState.FINISHED
            resize.ended = time.monotonic()
            blocks = self.engine.vllm_config.cache_config.num_gpu_blocks
            resize.kv_blocks_after = blocks
            logger.info(
                "Fast tier: card rows per layer %d -> %d in %.2f s on %d ranks; "
                "%.2f GiB per rank %s (KV cache %d -> %d blocks).",
                resize.from_rows,
                resize.to_rows,
                resize.ended - resize.started,
                len(reply["ranks"]),
                resize.bytes_per_rank / 2**30,
                "taken from the KV cache"
                if resize.from_kv and resize.kv is not None
                else "mapped from free device memory",
                resize.kv_blocks_before,
                blocks,
            )
            self._note_torch_room(resize)
            self.engine._publish_scheduler_stats()
            return
        resize.refused = bool(reply.get("refused"))
        resize.rolled_back = reply.get("rolled_back", reply["applied"] == "none")
        self._fail(resize, reply.get("error") or "not applied on every rank")

    def _fail(self, resize: FastTierResize, why: str) -> None:
        """The rows stay where they were; the KV cache gets back what it
        gave: its size before, restored as it was (the bytes it released
        were never mapped by the rows), not a new measured grow."""
        resize.state = FastTierState.FAILED
        resize.message = why
        resize.ended = time.monotonic()
        cache_config = self.engine.vllm_config.cache_config
        blocks = cache_config.num_gpu_blocks
        if resize.from_kv and blocks < resize.kv_blocks_before:
            try:
                kv = self.engine.resize_kv_cache(resize.kv_blocks_before, restore=True)
                resize.kv = kv
                if cache_config.num_gpu_blocks < resize.kv_blocks_before:
                    resize.message += (
                        f"; the KV cache came back to {cache_config.num_gpu_blocks}"
                        f" of {resize.kv_blocks_before} blocks"
                        + (f" ({kv.get('message')})" if kv.get("message") else "")
                    )
            except Exception as e:  # noqa: BLE001 - reported in the record
                resize.message += f"; growing the KV cache back failed: {e}"
        resize.kv_blocks_after = cache_config.num_gpu_blocks
        logger.warning(
            "Fast tier: card rows per layer %d -> %d failed after %.2f s: %s. The "
            "card tier stays at %d rows%s; KV cache %d -> %d blocks.",
            resize.from_rows,
            resize.to_rows,
            resize.ended - resize.started,
            why,
            resize.from_rows,
            "" if resize.rolled_back is not False else " on the ranks that undid it",
            resize.kv_blocks_before,
            resize.kv_blocks_after,
        )
        self._note_torch_room(resize)
        self.engine._publish_scheduler_stats()

    # -- shrink -----------------------------------------------------------

    def _shrink(
        self,
        rows_now: int,
        target: int,
        blocks: int,
        from_kv: bool,
        deadline: float = DEFAULT_DEADLINE_S,
    ) -> dict[str, Any]:
        resize = FastTierResize(
            "shrink",
            rows_now,
            target,
            FastTierState.COMMITTING,
            kv_blocks_before=blocks,
            from_kv=from_kv,
            deadline_s=deadline,
        )
        self.resize = resize
        self._shrink_commit(resize)
        if resize.state == FastTierState.FAILED and resize.refused:
            from vllm.v1.engine.runtime_control import ControlRefused

            raise ControlRefused(resize.message)
        return self._reply()

    def _shrink_commit(self, resize: FastTierResize) -> None:
        rows_now, target = resize.from_rows, resize.to_rows
        blocks = resize.kv_blocks_before
        from_kv = resize.from_kv
        reply = self._op({"rows": target})
        if reply["applied"] != "all" and self._waits_for_eplb(resize, reply):
            return
        if reply["applied"] != "all":
            resize.refused = bool(reply.get("refused"))
            resize.rolled_back = reply.get("rolled_back", reply["applied"] == "none")
            resize.state = FastTierState.FAILED
            resize.message = reply.get("error") or "not applied on every rank"
            resize.ended = time.monotonic()
            resize.kv_blocks_after = blocks
            logger.warning(
                "Fast tier: card rows per layer %d -> %d failed: %s.",
                rows_now,
                target,
                resize.message,
            )
            return
        released = [
            int((r.get("result") or {}).get("fast", {}).get("released_bytes", 0))
            for r in reply["ranks"]
        ]
        resize.bytes_per_rank = min(released) if released else 0
        if from_kv and resize.bytes_per_rank > 0:
            ranks = self._ranks()
            per_block = [(r.get("kv") or {}).get("bytes_per_block") for r in ranks]
            per_block = [b for b in per_block if b]
            if per_block:
                more = resize.bytes_per_rank // max(per_block)
                try:
                    # The rows' released bytes go back as a restore: a grow
                    # of N rows took N rows' blocks unmeasured (the KV shrink),
                    # so the shrink returns them the same way; the measured
                    # vote and the torch floor would keep part of them.
                    resize.kv = self.engine.resize_kv_cache(blocks + more, restore=True)
                except Exception as e:  # noqa: BLE001 - reported in the record
                    resize.message = f"the KV cache did not grow: {e}"
        resize.state = FastTierState.FINISHED
        resize.ended = time.monotonic()
        resize.kv_blocks_after = self.engine.vllm_config.cache_config.num_gpu_blocks
        logger.info(
            "Fast tier: card rows per layer %d -> %d in %.2f s on %d ranks; %.2f "
            "GiB per rank given back to the KV cache (KV cache %d -> %d blocks).",
            rows_now,
            target,
            resize.ended - resize.started,
            len(reply["ranks"]),
            resize.bytes_per_rank / 2**30,
            blocks,
            resize.kv_blocks_after,
        )
        self._note_torch_room(resize)
        self.engine._publish_scheduler_stats()

    def _note_torch_room(self, resize: FastTierResize) -> None:
        """Each rank's torch room after the operation into the record, and
        one line with the least of them: what the next peak has."""
        try:
            ranks = self._ranks()
        except Exception:  # noqa: BLE001 - the record goes without it
            return
        rooms = [((r.get("tiers") or {}).get("fast") or {}).get("torch") for r in ranks]
        resize.torch_after = rooms
        known = [t for t in rooms if t]
        dev = [t["device_room"] for t in known if t.get("device_room") is not None]
        cap = [t["cap_room"] for t in known if t.get("cap_room") is not None]
        if dev or cap:
            logger.info(
                "Fast tier: torch's room after the resize (least rank): %s on "
                "the device, %s under its cap.",
                f"{min(dev) / 2**30:.2f} GiB" if dev else "?",
                f"{min(cap) / 2**30:.2f} GiB" if cap else "no cap",
            )


def register_runtime_params(register_verb, register_param, Verb, Param) -> None:
    """This verb and the card tier's start settings, for the runtime control
    surface's schema (``vllm/v1/engine/runtime_params.py``)."""
    register_verb(
        Verb(
            "tiers.fast",
            "Card rows per MoE layer per rank: grow (the KV cache gives the "
            "memory) or shrink (the KV cache gets it); EPLB's load chooses the "
            "experts that enter or leave.",
            args={
                "rows": "int, the target (or delta)",
                "delta": "int, the change (or rows)",
                "deadline_s": "float, the KV shrink's drain deadline and the "
                "wait for EPLB (default 60)",
                "preempt": "bool, preempt at the deadline instead of aborting",
                "from_kv": "bool, take from / give to the KV cache (default true)",
            },
        )
    )
    start_only = "the device range is reserved at load"
    for name, type_, doc, reason in (
        (
            "store.fast_rows_max",
            "int",
            (
                "Card rows per layer the reserved range holds "
                "(0 = the card tier is fixed)."
            ),
            start_only,
        ),
        (
            "store.fast_rows_min",
            "int",
            "The fewest card rows per layer a resize may go to.",
            "pinned rows for it are sized at load",
        ),
        (
            "store.fast_chunk_mib",
            "int",
            "Granule of the card rows' mapped memory.",
            start_only,
        ),
        (
            "store.pinned_release",
            "bool",
            (
                "Pinned rows follow the pinned bound (a lower bound gives "
                "host memory back after EPLB's tier pass)."
            ),
            "the pinned arena's layout is chosen at load",
        ),
    ):
        register_param(
            Param(
                name,
                type_,
                doc,
                restart_reason=reason,
                start="--expert-store-config " + name.split(".", 1)[1],
            )
        )

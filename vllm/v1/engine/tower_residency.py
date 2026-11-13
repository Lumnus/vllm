# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where an encoder tower with residency "dynamic" lives while the engine runs.

A tower named in `--mm-encoder-host-towers` with `"residency": "dynamic"`
starts in host memory (or on disk) and moves to the accelerators while its
modality is in use:

* disk -> host: a request for its modalities loads the host copy (the host
  tower does that by itself).
* host -> card: `promote_after` requests within `promote_window_s`. Every
  rank stages a copy in host memory, the KV cache gives up the copy's bytes
  plus `card_reserve_mib` (a shrink through the engine core's resize state
  machine), every rank moves its copy to its accelerator, and only then the
  scheduler sends new items to the workers. The host copy serves meanwhile.
* card -> host (or disk): `demote_idle_s` without such a request. The
  scheduler sends new items to the host copy again, every rank drops its
  copy, and the KV cache grows back to its size before the promotion.
* host -> disk: the same idle time, when `idle_location` is `"disk"`.

A promotion that cannot finish (the KV cache refuses or the drain runs out of
time, a rank fails to stage or commit) leaves the tower where it was, drops
whatever the ranks staged, gives the KV cache its blocks back and holds off
for `FAILURE_HOLD_S`. Each transition is one INFO line, and the counters are
in the scheduler's stats (`host_tower_stats`).

Everything here runs on the engine core's thread, between two steps, like the
KV cache resize it relies on.
"""

import enum
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1 import runtime_debug

if TYPE_CHECKING:
    from vllm.config.multimodal import HostTowerOptions
    from vllm.distributed.ec_transfer.ec_connector.host_tower.scheduler import (
        ECHostTowerScheduler,
    )

logger = init_logger(__name__)


class TowerPlace(enum.StrEnum):
    DISK = "disk"
    HOST = "host"
    CARD = "card"


class TowerPhase(enum.StrEnum):
    STEADY = "steady"
    STAGING = "staging"
    """Every rank builds its copy in host memory and reads the weights."""
    SHRINKING = "shrinking"
    """The KV cache drains and releases the blocks the copy needs."""


PLACE_CODE = {TowerPlace.DISK: 0.0, TowerPlace.HOST: 1.0, TowerPlace.CARD: 2.0}
_RESIZE_ACTIVE = ("draining", "releasing", "unfinished")


def mem_available_bytes() -> int | None:
    """The host's MemAvailable (Linux), or None where it cannot be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class TowerResidency:
    """One dynamic tower's place and the moves between places."""

    # Seconds between two reads of the ranks' staging progress.
    POLL_INTERVAL_S = 0.5
    # Seconds a failed promotion holds off the next one.
    FAILURE_HOLD_S = 60.0
    # Longest wait for a rank's weight warm up at commit time.
    COMMIT_WAIT_S = 120.0
    # Longest wait for the ranks' reply to a residency call (the commit gets
    # COMMIT_WAIT_S on top); a rank that does not answer fails the executor.
    RPC_TIMEOUT_S = 120.0

    def __init__(
        self,
        name: str,
        options: "HostTowerOptions",
        engine: Any,
        delegate: "ECHostTowerScheduler",
        clock: Callable[[], float],
    ) -> None:
        self.name = name
        self.options = options
        self._engine = engine
        self._delegate = delegate
        self._clock = clock

        now = clock()
        self.place = TowerPlace.HOST
        self.phase = TowerPhase.STEADY
        self.arrivals: deque[float] = deque()
        self.last_use = now
        self.hold_until = 0.0
        self.promotions = 0
        self.demotions = 0
        self.failed_promotions = 0

        # The promotion in progress, or the last one.
        self._from = TowerPlace.HOST
        self._started = 0.0
        self._reason = ""
        self._blocks_before: int | None = None
        self._card_bytes = 0
        self._last_poll = 0.0
        # A KV cache size to grow back to once no resize is in the way.
        self._grow_to: int | None = None
        # The placement policy, set while serving (tower.set, a serving
        # profile): "dynamic" moves by requests and idle time as above;
        # "host" keeps it off the accelerators (demoted at once if there);
        # "card" brings it there and keeps it. Its only source of accelerator
        # memory is the KV cache (request_kv_cache_release), never the experts.
        self.policy = "dynamic"
        # Seconds idle in host memory after which the host copy is released
        # too (the drive tier: the next request reloads it). None: the start
        # option `idle_location` decides, as before.
        self.disk_idle_s: float | None = None

        if options.idle_location == "disk":
            delegate.host_tower(name).release(wait=False)
            self.place = TowerPlace.DISK
        self._publish()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def on_arrival(self, now: float) -> None:
        """A request with an item of this tower's modalities."""
        self.arrivals.append(now)
        self.last_use = now
        if runtime_debug.on("tower"):
            runtime_debug.emit(
                "tower",
                logger,
                "%s: a request for its modalities (%d in the window, in %s, policy %s)",
                self.name,
                len(self.arrivals),
                self.place.value,
                self.policy,
            )

    def seconds_to_next_event(self, now: float) -> float | None:
        if self.phase != TowerPhase.STEADY or self._grow_to is not None:
            return self.POLL_INTERVAL_S
        if (self.policy == "host" and self.place == TowerPlace.CARD) or (
            self.policy == "card" and self.place != TowerPlace.CARD
        ):
            return self.POLL_INTERVAL_S
        deadlines: list[float] = []
        if self.place == TowerPlace.CARD and self.policy == "dynamic":
            deadlines.append(self.last_use + self.options.demote_idle_s)
        disk_after = self._disk_after()
        if self.place == TowerPlace.HOST and disk_after is not None:
            deadlines.append(self.last_use + disk_after)
        if (
            self.policy == "dynamic"
            and self.place != TowerPlace.CARD
            and len(self.arrivals) >= self.options.promote_after
        ):
            # Held off after a failure, or waiting for another resize.
            deadlines.append(max(self.hold_until, now + self.POLL_INTERVAL_S))
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - now) + 0.01

    # ------------------------------------------------------------------
    # The state machine
    # ------------------------------------------------------------------

    def advance(self, now: float) -> None:
        if self.phase == TowerPhase.STAGING:
            self._advance_staging(now)
            return
        if self.phase == TowerPhase.SHRINKING:
            self._advance_shrinking(now)
            return
        self._retry_grow()

        host = self._delegate.host_tower(self.name)
        if self.place == TowerPlace.DISK and host.location.value == "host":
            self._log_transition(
                TowerPlace.DISK,
                TowerPlace.HOST,
                None,
                host.nbytes,
                "host memory",
                self._num_blocks(),
                self._num_blocks(),
                "a request for its modalities loaded it",
            )
            self.place = TowerPlace.HOST
            self._publish()

        window = self.options.promote_window_s
        while self.arrivals and self.arrivals[0] < now - window:
            self.arrivals.popleft()

        idle = now - self.last_use
        disk_after = self._disk_after()
        if self.place in (TowerPlace.HOST, TowerPlace.DISK):
            earned = (
                self.policy == "card"
                or self.policy == "dynamic"
                and len(self.arrivals) >= self.options.promote_after
            )
            if earned and now >= self.hold_until and not self._resize_active():
                self._start_promotion(
                    now,
                    "policy card"
                    if self.policy == "card"
                    else f"{len(self.arrivals)} requests for its modalities in "
                    f"{window:.0f} s",
                )
            elif (
                self.place == TowerPlace.HOST
                and disk_after is not None
                and idle >= disk_after
            ):
                self._host_to_disk(now, idle)
        elif self.place == TowerPlace.CARD:
            if self.policy == "host":
                self._demote(now, idle, "policy host")
            elif self.policy == "dynamic" and idle >= self.options.demote_idle_s:
                self._demote(now, idle)

    def _disk_after(self) -> float | None:
        """Idle seconds in host memory before the host copy is released."""
        if self.disk_idle_s is not None:
            return self.disk_idle_s
        if self.options.idle_location == "disk":
            return self.options.demote_idle_s
        return None

    def _start_promotion(self, now: float, reason: str) -> None:
        if runtime_debug.on("tower"):
            runtime_debug.emit(
                "tower",
                logger,
                "%s: promotion starts (%s); KV cache %d blocks",
                self.name,
                reason,
                self._num_blocks(),
            )
        self._from = self.place
        self._started = now
        self._reason = reason
        self._blocks_before = self._num_blocks()
        logger.info(
            "Tower %s: moving to the accelerators (%s); every rank measures its "
            "copy and warms its weights first.",
            self.name,
            reason,
        )
        try:
            results = self._rpc("tower_residency_stage")
        except Exception as e:
            self._fail(now, f"a rank could not stage its copy: {e}")
            return
        self._card_bytes = max(int(r["bytes"]) for r in results)
        refusal = self._host_memory_refusal(
            sum(int(r.get("host_bytes", 0)) for r in results)
        )
        if refusal is not None:
            self._fail(now, refusal)
            return
        self.phase = TowerPhase.STAGING
        self._last_poll = float("-inf")
        self._publish()
        self._advance_staging(now)

    def _advance_staging(self, now: float) -> None:
        if now - self._last_poll < self.POLL_INTERVAL_S:
            return
        self._last_poll = now
        try:
            results = self._rpc("tower_residency_poll")
        except Exception as e:
            self._fail(now, f"a rank could not report its staging: {e}")
            return
        errors = [r["error"] for r in results if r.get("error")]
        if errors:
            self._fail(now, f"a rank could not read the weights: {errors[0]}")
            return
        if not all(r["staged"] for r in results):
            return
        need = self._card_bytes + self.options.card_reserve_mib * (1 << 20)
        try:
            record = self._engine.request_kv_cache_release(
                need, urgent=False, deadline_s=self.options.shrink_deadline_s
            )
        except (ValueError, RuntimeError, NotImplementedError) as e:
            self._fail(now, f"the KV cache cannot give up {need:,} bytes per rank: {e}")
            return
        self.phase = TowerPhase.SHRINKING
        self._publish()
        self._advance_shrinking(now, record)

    def _advance_shrinking(
        self, now: float, record: dict[str, Any] | None = None
    ) -> None:
        if record is None:
            record = self._engine.get_kv_cache_resize_state()
        state = record.get("state")
        if state in _RESIZE_ACTIVE:
            return
        if state not in ("finished", "unchanged"):
            self._fail(
                now,
                f"the KV cache shrink ended {state}"
                + (f" ({record['message']})" if record.get("message") else ""),
            )
            return
        self._commit(now)

    def _commit(self, now: float) -> None:
        try:
            results = self._rpc(
                "tower_residency_commit",
                self.COMMIT_WAIT_S,
                timeout=self.COMMIT_WAIT_S + self.RPC_TIMEOUT_S,
            )
        except Exception as e:
            self._release_everywhere()
            self._fail(now, f"a rank could not put its copy on the accelerator: {e}")
            return
        self._delegate.set_tower_on_card(self.name, True)
        end = self._clock()
        self.place = TowerPlace.CARD
        self.phase = TowerPhase.STEADY
        self.promotions += 1
        self.arrivals.clear()
        self._log_transition(
            self._from,
            TowerPlace.CARD,
            end - self._started,
            max(int(r["bytes"]) for r in results),
            "each accelerator",
            self._blocks_before,
            self._num_blocks(),
            self._reason,
        )
        self._publish()

    def _fail(self, now: float, why: str) -> None:
        try:
            self._rpc("tower_residency_discard")
        except Exception:
            logger.exception("Tower %s: dropping the staged copies failed.", self.name)
        self.phase = TowerPhase.STEADY
        self.failed_promotions += 1
        self.hold_until = now + self.FAILURE_HOLD_S
        before = self._blocks_before
        if before is not None and self._num_blocks() < before:
            self._grow_to = before
            self._retry_grow()
        logger.warning(
            "Tower %s: the move to the accelerators failed after %.2f s: %s. It "
            "stays in %s; KV cache %s -> %d blocks%s; next try not before %.0f s "
            "from now.",
            self.name,
            self._clock() - self._started,
            why,
            self.place.value,
            before,
            self._num_blocks(),
            " (growing back when no resize is in the way)"
            if self._grow_to is not None
            else "",
            self.FAILURE_HOLD_S,
        )
        self._publish()

    def _release_everywhere(self) -> int:
        try:
            results = self._rpc("tower_residency_release")
        except Exception:
            logger.exception(
                "Tower %s: a rank did not confirm that its accelerator copy is "
                "gone. Requests are served from host memory.",
                self.name,
            )
            return 0
        return max((int(r["bytes"]) for r in results), default=0)

    def _demote(self, now: float, idle: float, reason: str | None = None) -> None:
        start = self._clock()
        blocks = self._num_blocks()
        # New items go to the host copy from the next scheduling pass; the
        # release below reaches the workers after every batch already
        # launched.
        self._delegate.set_tower_on_card(self.name, False)
        nbytes = self._release_everywhere()
        host = self._delegate.host_tower(self.name)
        if self.options.idle_location == "disk" and self.disk_idle_s is None:
            host.release(wait=False)
            target = TowerPlace.DISK
        else:
            target = (
                TowerPlace.HOST if host.location.value == "host" else TowerPlace.DISK
            )
        if self._blocks_before is not None:
            self._grow_to = self._blocks_before
            self._retry_grow()
        self.place = target
        self.demotions += 1
        self.arrivals.clear()
        self._log_transition(
            TowerPlace.CARD,
            target,
            self._clock() - start,
            nbytes,
            "each accelerator, freed",
            blocks,
            self._num_blocks(),
            reason or f"no request for its modalities in {idle:.0f} s",
        )
        self._publish()

    def _host_to_disk(self, now: float, idle: float) -> None:
        host = self._delegate.host_tower(self.name)
        nbytes = host.nbytes
        host.release(wait=False)
        self.place = TowerPlace.DISK
        self._log_transition(
            TowerPlace.HOST,
            TowerPlace.DISK,
            None,
            nbytes,
            "host memory, freed",
            self._num_blocks(),
            self._num_blocks(),
            f"no request for its modalities in {idle:.0f} s",
        )
        self._publish()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rpc(self, method: str, *args: Any, timeout: float | None = None) -> list:
        return self._engine.collective_rpc(
            method,
            timeout=self.RPC_TIMEOUT_S if timeout is None else timeout,
            args=(self.name, *args),
        )

    def _host_memory_refusal(self, host_bytes: int) -> str | None:
        """Why the host cannot take the staging now, or None. The ranks hold
        at most host_bytes together while they load (one tensor each)."""
        floor = int(self.options.host_available_floor_gib * (1 << 30))
        if floor <= 0:
            return None
        available = mem_available_bytes()
        if available is None:
            return None
        if available - host_bytes >= floor:
            return None
        return (
            f"host memory: {available / (1 << 30):.2f} GiB available, the "
            f"copies need {host_bytes / (1 << 30):.2f} GiB while they load, and "
            f"{self.options.host_available_floor_gib:.2f} GiB must stay available"
        )

    def _num_blocks(self) -> int:
        return int(self._engine.vllm_config.cache_config.num_gpu_blocks)

    def _resize_active(self) -> bool:
        return self._engine.get_kv_cache_resize_state().get("state") in _RESIZE_ACTIVE

    def _retry_grow(self) -> None:
        target = self._grow_to
        if target is None or self._resize_active():
            return
        self._grow_to = None
        if self._num_blocks() >= target:
            return
        try:
            self._engine.resize_kv_cache(target)
        except Exception as e:
            logger.warning(
                "Tower %s: the KV cache could not grow back to %d blocks: %s",
                self.name,
                target,
                e,
            )

    def _log_transition(
        self,
        src: TowerPlace,
        dst: TowerPlace,
        seconds: float | None,
        nbytes: int,
        where: str,
        blocks_before: int | None,
        blocks_after: int,
        reason: str,
    ) -> None:
        logger.info(
            "Tower %s: %s -> %s%s (%s); %s bytes per rank in %s; KV cache %s -> "
            "%d blocks.",
            self.name,
            src.value,
            dst.value,
            "" if seconds is None else f" in {seconds:.2f} s",
            reason,
            f"{nbytes:,}",
            where,
            blocks_before,
            blocks_after,
        )

    def _publish(self) -> None:
        self._delegate.update_tower_stats(
            self.name,
            location=PLACE_CODE[self.place],
            in_transition=float(self.phase != TowerPhase.STEADY),
            promotions=float(self.promotions),
            demotions=float(self.demotions),
            failed_promotions=float(self.failed_promotions),
        )
        # The gauges move now, not with the next step's output.
        publish = getattr(self._engine, "_publish_scheduler_stats", None)
        if publish is not None:
            publish()

    def record(self) -> dict[str, Any]:
        return {
            "place": self.place.value,
            "phase": self.phase.value,
            "arrivals_in_window": len(self.arrivals),
            "promotions": self.promotions,
            "demotions": self.demotions,
            "failed_promotions": self.failed_promotions,
        }


class TowerResidencyController:
    """The dynamic towers of one engine core."""

    def __init__(
        self,
        engine: Any,
        delegate: "ECHostTowerScheduler",
        options: dict[str, "HostTowerOptions"],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self.towers = {
            name: TowerResidency(name, opts, engine, delegate, clock)
            for name, opts in options.items()
        }
        delegate.set_arrival_listener(self._on_arrival)
        logger.info(
            "Encoder towers that move between host and accelerators: %s.",
            ", ".join(
                f"{name} (after {o.promote_after} requests in "
                f"{o.promote_window_s:.0f} s; back to {o.idle_location} after "
                f"{o.demote_idle_s:.0f} s idle; {o.card_layout})"
                for name, o in options.items()
            ),
        )

    @classmethod
    def maybe_create(cls, engine: Any) -> "TowerResidencyController | None":
        """The controller for `engine`, or None without a dynamic tower."""
        model_config = engine.vllm_config.model_config
        mm_config = getattr(model_config, "multimodal_config", None)
        towers = getattr(mm_config, "mm_encoder_host_towers", None) or {}
        options = {
            name: opts for name, opts in towers.items() if opts.residency == "dynamic"
        }
        if not options:
            return None
        connector = engine.scheduler.get_ec_connector()
        delegate = getattr(connector, "connector_scheduler", None)
        if delegate is None or not hasattr(delegate, "set_tower_on_card"):
            raise ValueError(
                "Encoder towers with residency 'dynamic' need the host tower "
                "connector in the engine."
            )
        try:
            engine._check_kv_cache_resizable()
        except (ValueError, NotImplementedError) as e:
            raise ValueError(
                f"Encoder towers with residency 'dynamic' ({', '.join(options)}) "
                f"need a KV cache that can be resized while serving: {e}"
            ) from e
        return cls(engine, delegate, options)

    def _on_arrival(self, name: str) -> None:
        tower = self.towers.get(name)
        if tower is not None:
            tower.on_arrival(self._clock())

    def advance(self) -> None:
        now = self._clock()
        for tower in self.towers.values():
            try:
                tower.advance(now)
            except Exception:
                logger.exception(
                    "Tower %s: the residency step failed; serving goes on.",
                    tower.name,
                )

    def seconds_to_next_event(self) -> float | None:
        now = self._clock()
        waits = [
            wait
            for tower in self.towers.values()
            if (wait := tower.seconds_to_next_event(now)) is not None
        ]
        return min(waits) if waits else None

    def record(self) -> dict[str, Any]:
        return {name: tower.record() for name, tower in self.towers.items()}

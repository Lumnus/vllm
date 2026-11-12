# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECHostTowerScheduler: scheduler delegate of the ECHostTowerConnector.

Runs in the engine process, next to the scheduler. It owns the host towers
and the shared region their embeddings are handed over in, and it is what
makes a request wait for an embedding without making anything else wait:

* `ensure_cache_available` submits the encodes a request needs to the towers'
  own threads and answers False until all of them are in the region. The
  scheduler skips the request meanwhile and schedules the others.
* A finished embedding is written into the region and pinned for the requests
  that wait for it. From then on the scheduler sees it as an encoder output
  it can load (`has_cache_item`) instead of one to compute on the workers,
  and `update_state_after_alloc` dispatches the load.

All state here is touched by the scheduler thread only. A tower thread
computes and nothing else; its result is picked up by the next call.
"""

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECCPUWorkerMetadata,
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal.host_tower import HostTower, TowerEncodeResult
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ECConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# How long a finished embedding may wait for room in the region before the
# requests waiting for it are failed (retryable).
_SPACE_WAIT_TIMEOUT_S = 60.0


@dataclass
class _Encode:
    """An item on its way through a tower."""

    tower: str
    future: "Future[TowerEncodeResult]"
    # Requests that wait for this item.
    waiters: set[str] = field(default_factory=set)
    # When the finished embedding first found the region full.
    space_wait_since: float | None = None


def _new_tower_stats() -> dict[str, float]:
    return {
        # 0 disk, 1 host, 2 card (vllm/v1/engine/tower_residency.py).
        "location": 1.0,
        "in_transition": 0.0,
        "promotions": 0.0,
        "demotions": 0.0,
        "failed_promotions": 0.0,
        "image_requests": 0.0,
        "items_encoded_host": 0.0,
        "items_encoded_card": 0.0,
        "encode_seconds_host": 0.0,
    }


def _block_runs(block_ids: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Split ascending block ids into `(first, count)` runs of neighbours."""
    start = 0
    for i in range(1, len(block_ids) + 1):
        if i == len(block_ids) or block_ids[i] != block_ids[i - 1] + 1:
            yield block_ids[start], i - start
            start = i


def write_blocks(
    blocks: torch.Tensor, block_ids: Sequence[int], embedding: torch.Tensor
) -> None:
    """Copy `embedding`'s bytes into the region blocks `block_ids`, in order."""
    src = embedding.reshape(-1).view(torch.int8)
    dst = blocks.view(-1)
    block_size = blocks.shape[1]
    offset = 0
    for first, count in _block_runs(block_ids):
        nbytes = min(count * block_size, src.numel() - offset)
        dst[first * block_size : first * block_size + nbytes] = src[
            offset : offset + nbytes
        ]
        offset += nbytes
    assert offset == src.numel(), "embedding does not fit its blocks"


class ECHostTowerScheduler:
    """Scheduler delegate for the ECHostTowerConnector."""

    def __init__(
        self, vllm_config: "VllmConfig", towers: Mapping[str, "HostTower"]
    ) -> None:
        self._region = create_ec_shared_region(vllm_config)
        # Block allocator and eviction order of the region.
        self._cache = EmbeddingCache(self._region.num_blocks)

        self._towers = dict(towers)
        # modality -> the host tower that encodes it.
        self._tower_of: dict[str, HostTower] = {
            modality: tower
            for tower in towers.values()
            for modality in tower.modalities
        }

        # identifier -> encode in flight, or finished and waiting for room.
        self._encodes: dict[str, _Encode] = {}
        # request id -> identifiers pinned in the region for that request.
        self._interest: dict[str, set[str]] = {}
        # Requests whose embeddings cannot be produced; the scheduler fails them.
        self._unavailable: set[str] = set()

        # identifier -> (transfer id, block ids, shape) to load this step.
        self._pending_loads: dict[str, tuple[int, list[int], tuple[int, int]]] = {}
        # transfer id -> (identifier, worker reports still outstanding).
        self._load_acks: dict[int, tuple[str, int]] = {}
        self._next_transfer_id = 0

        # Residency "dynamic" (vllm/v1/engine/tower_residency.py): towers whose
        # accelerator copy encodes now, the engine core's listener for image
        # requests, the requests it has heard of, and what was encoded where.
        # All of it stays empty and unused without a dynamic tower.
        self._on_card: set[str] = set()
        self._arrival_listener: Callable[[str], None] | None = None
        self._seen_requests: set[str] = set()
        self._tower_stats: dict[str, dict[str, float]] = {}

        pc = vllm_config.parallel_config
        # Only pipeline stage 0 runs the connector, on each of its ranks.
        self._expected_load_reports = (
            pc.tensor_parallel_size * pc.prefill_context_parallel_size
            if pc.distributed_executor_backend == "mp"
            else 1
        )

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        entry = self._cache.get(identifier)
        return entry is not None and entry.ready

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        self._collect_encodes()

        request_id = request.request_id
        if request_id in self._unavailable:
            return False
        if self._arrival_listener is not None and request_id not in self._seen_requests:
            self._note_arrival(request)
        held = self._interest.setdefault(request_id, set())

        ready = True
        for feature in request.mm_features:
            tower = self._tower_of.get(feature.modality)
            if tower is None:
                # Encoded by the workers, on the accelerator.
                continue
            position = feature.mm_position
            if position.offset + position.length <= num_computed_tokens:
                # Already in the KV cache: no embedding needed.
                continue
            identifier = feature.identifier
            if identifier in held:
                continue
            entry = self._cache.pin_if_ready(identifier)
            if entry is not None and entry.ready:
                held.add(identifier)
                continue
            if tower.name in self._on_card and identifier not in self._encodes:
                # The tower's accelerator copy encodes it on the workers. An
                # item already on its way through the host copy waits for it.
                continue

            ready = False
            encode = self._encodes.get(identifier)
            if encode is None:
                encode = self._submit(request, feature, tower)
                if encode is None:
                    return False
            encode.waiters.add(request_id)
        return ready

    # ------------------------------------------------------------------
    # Residency "dynamic": the engine core's controller drives these
    # ------------------------------------------------------------------

    def set_arrival_listener(self, listener: Callable[[str], None]) -> None:
        """Have `listener(tower_name)` called once per request that has an
        item of that tower's modalities, when the scheduler first looks at
        it."""
        self._arrival_listener = listener
        for name in self._towers:
            self._tower_stats.setdefault(name, _new_tower_stats())

    def set_tower_on_card(self, name: str, on_card: bool) -> None:
        """From the next scheduling pass, items of tower `name` go to the
        workers' copy (True) or to the host copy (False)."""
        if on_card:
            self._on_card.add(name)
        else:
            self._on_card.discard(name)

    def tower_on_card(self, name: str) -> bool:
        return name in self._on_card

    def host_tower(self, name: str) -> "HostTower":
        return self._towers[name]

    def update_tower_stats(self, name: str, **values: float) -> None:
        self._tower_stats.setdefault(name, _new_tower_stats()).update(values)

    def host_tower_stats(self) -> dict[str, dict[str, float]]:
        """Per tower: where it is and what it did (empty without a dynamic
        tower)."""
        return {name: dict(stats) for name, stats in self._tower_stats.items()}

    def _note_arrival(self, request: "Request") -> None:
        self._seen_requests.add(request.request_id)
        assert self._arrival_listener is not None
        names = {
            tower.name
            for feature in request.mm_features
            if (tower := self._tower_of.get(feature.modality)) is not None
        }
        for name in sorted(names):
            self._count(name, "image_requests")
            self._arrival_listener(name)

    def _count(self, name: str, key: str, value: float = 1.0) -> None:
        stats = self._tower_stats.get(name)
        if stats is not None:
            stats[key] += value

    def _submit(
        self,
        request: "Request",
        feature: "MultiModalFeatureSpec",
        tower: "HostTower",
    ) -> _Encode | None:
        data = feature.data
        if data is None or "address" in data:
            logger.error(
                "Host tower %s: request %s carries no input for item %s in "
                "the engine process; failing the request.",
                tower.name,
                request.request_id,
                feature.identifier,
            )
            self._unavailable.add(request.request_id)
            return None

        encode = _Encode(
            tower=tower.name, future=tower.submit([(feature.modality, data)])
        )
        self._encodes[feature.identifier] = encode
        logger.info(
            "Host tower %s: queued %s item %s for request %s (%d in flight).",
            tower.name,
            feature.modality,
            feature.identifier,
            request.request_id,
            len(self._encodes),
        )
        return encode

    def _collect_encodes(self) -> None:
        """Move finished embeddings into the region."""
        if not self._encodes:
            return
        finished = [
            identifier
            for identifier, encode in self._encodes.items()
            if encode.future.done()
        ]
        for identifier in finished:
            encode = self._encodes[identifier]
            if not encode.waiters:
                # Every request that wanted it has left.
                del self._encodes[identifier]
                continue
            error = encode.future.exception()
            if error is not None:
                logger.error(
                    "Host tower %s: encoding item %s failed; failing %d request(s).",
                    encode.tower,
                    identifier,
                    len(encode.waiters),
                    exc_info=error,
                )
                self._unavailable.update(encode.waiters)
                del self._encodes[identifier]
                continue
            if self._store(identifier, encode, encode.future.result()):
                del self._encodes[identifier]

    def _store(
        self, identifier: str, encode: _Encode, result: "TowerEncodeResult"
    ) -> bool:
        """Write a finished embedding into the region and pin it for its
        waiters. Returns False while the region has no room for it."""
        (embedding,) = result.embeddings
        nbytes = embedding.numel() * embedding.element_size()
        block_size = self._region.block_size_bytes
        num_blocks = max(1, -(-nbytes // block_size))

        fits = embedding.ndim == 2 and num_blocks <= self._region.num_blocks
        entry = (
            self._cache.alloc(
                identifier, num_blocks, (embedding.shape[0], embedding.shape[1])
            )
            if fits
            else None
        )
        if entry is None:
            now = time.monotonic()
            if encode.space_wait_since is None:
                encode.space_wait_since = now
            if fits and now - encode.space_wait_since <= _SPACE_WAIT_TIMEOUT_S:
                # Room appears when a request that holds embeddings finishes.
                return False
            logger.error(
                "Host tower %s: no room for the embedding of item %s (shape %s, "
                "%d bytes) in the %d byte hand-over region; failing %d "
                "request(s). Raise cache_gb in --mm-encoder-host-towers.",
                encode.tower,
                identifier,
                tuple(embedding.shape),
                nbytes,
                self._region.num_blocks * block_size,
                len(encode.waiters),
            )
            self._unavailable.update(encode.waiters)
            return True

        write_blocks(self._region.blocks, entry.block_ids, embedding)
        self._cache.mark_ready(identifier)
        if self._tower_stats:
            self._count(encode.tower, "items_encoded_host")
            self._count(encode.tower, "encode_seconds_host", result.encode_seconds)
        for request_id in encode.waiters:
            self._cache.pin(identifier)
            self._interest.setdefault(request_id, set()).add(identifier)
        logger.info(
            "Host tower %s: item %s encoded in %.0f ms after %.0f ms in queue; "
            "%d x %d %s (%d bytes) ready for the workers, %d request(s) waiting.",
            encode.tower,
            identifier,
            result.encode_seconds * 1e3,
            result.queue_seconds * 1e3,
            embedding.shape[0],
            embedding.shape[1],
            str(embedding.dtype).removeprefix("torch."),
            nbytes,
            len(encode.waiters),
        )
        return True

    def take_unavailable_requests(self) -> set[str]:
        if not self._unavailable:
            return set()
        failed = self._unavailable
        self._unavailable = set()
        return failed

    # ------------------------------------------------------------------
    # Hand-over to the workers
    # ------------------------------------------------------------------

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        identifier = request.mm_features[index].identifier
        if identifier in self._pending_loads:
            return
        entry = self._cache.pin_if_ready(identifier)
        if entry is None or not entry.ready:
            # Not a host tower's item: the workers encode it.
            if self._on_card:
                tower = self._tower_of.get(request.mm_features[index].modality)
                if tower is not None and tower.name in self._on_card:
                    self._count(tower.name, "items_encoded_card")
            return
        assert entry.shape is not None
        transfer_id = self._next_transfer_id
        self._next_transfer_id += 1
        self._pending_loads[identifier] = (
            transfer_id,
            list(entry.block_ids),
            entry.shape,
        )
        self._load_acks[transfer_id] = (identifier, self._expected_load_reports)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        # Once per step even when no request asked: results nobody waits for
        # any more are dropped here.
        self._collect_encodes()
        meta = ECCPUConnectorMetadata()
        meta.loads = self._pending_loads
        self._pending_loads = {}
        return meta

    def update_connector_output(self, connector_output: "ECConnectorOutput") -> None:
        """Release a load's pin once every rank has copied it."""
        meta = connector_output.ec_connector_worker_meta
        if not isinstance(meta, ECCPUWorkerMetadata):
            return
        for transfer_id in meta.completed_loads:
            pending = self._load_acks.get(transfer_id)
            if pending is None:
                continue
            identifier, outstanding = pending
            if outstanding > 1:
                self._load_acks[transfer_id] = (identifier, outstanding - 1)
                continue
            del self._load_acks[transfer_id]
            self._cache.unpin(identifier)

    def has_pending_push_work(self) -> bool:
        """Keep the engine stepping while a load awaits the workers' reports."""
        return bool(self._load_acks)

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def update_state_after_free(self, request: "Request", index: int) -> None:
        """The request has the item in its KV cache: stop holding it."""
        identifier = request.mm_features[index].identifier
        held = self._interest.get(request.request_id)
        if held is not None and identifier in held:
            held.discard(identifier)
            self._cache.unpin(identifier)

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """Drop everything held or computed for a finished or aborted request.

        An encode nobody waits for any more is cancelled if it has not started;
        one that is running finishes and its result is dropped.
        """
        request_id = request.request_id
        for identifier in self._interest.pop(request_id, ()):
            self._cache.unpin(identifier)
        for identifier, encode in list(self._encodes.items()):
            encode.waiters.discard(request_id)
            if not encode.waiters and encode.future.cancel():
                del self._encodes[identifier]
        self._unavailable.discard(request_id)
        self._seen_requests.discard(request_id)
        return False, None

    def shutdown(self) -> None:
        for encode in self._encodes.values():
            encode.future.cancel()
        self._encodes.clear()
        self._interest.clear()
        self._pending_loads.clear()
        self._load_acks.clear()
        for tower in self._towers.values():
            tower.shutdown()
        try:
            self._region.cleanup()
        except Exception:
            logger.debug("host tower: region cleanup failed", exc_info=True)

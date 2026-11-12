# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two delegates of the ECHostTowerConnector against a toy tower: what a
request waits for, what reaches the workers, and what is left behind."""

import threading
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.ec_transfer.ec_connector.host_tower.scheduler as sched_mod
import vllm.distributed.ec_transfer.ec_connector.host_tower.worker as worker_mod
from tests.v1.ec_connector.unit.host_tower.utils import (
    OUT_DIM,
    FakeRegion,
    direct_embedding,
    make_item,
    make_tower,
)
from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECCPUWorkerMetadata
from vllm.distributed.ec_transfer.ec_connector.host_tower.scheduler import (
    ECHostTowerScheduler,
    write_blocks,
)
from vllm.distributed.ec_transfer.ec_connector.host_tower.worker import (
    ECHostTowerWorker,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

# float16 embeddings of width OUT_DIM: one row per block.
BLOCK_BYTES = OUT_DIM * 2


class _Env:
    def __init__(
        self, monkeypatch, *, num_blocks=64, tensor_parallel_size=1, gate=None
    ):
        self.region = FakeRegion(num_blocks, BLOCK_BYTES)
        monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda _: self.region)
        monkeypatch.setattr(
            worker_mod, "create_ec_shared_region", lambda _: self.region
        )
        config = create_ec_vllm_config(
            ec_role="ec_consumer", tensor_parallel_size=tensor_parallel_size
        )
        self.tower = make_tower(gate)
        self.tower.to_host()
        self.sched = ECHostTowerScheduler(config, {self.tower.name: self.tower})
        self.worker = ECHostTowerWorker(config)

    def wait(self, identifier: str) -> None:
        """Until the tower has finished `identifier` (or failed on it)."""
        future = self.sched._encodes[identifier].future
        future.exception(timeout=60)

    def admit(self, request) -> bool:
        return self.sched.ensure_cache_available(request, 0)

    def load(self, request, cache: dict[str, torch.Tensor]):
        """What the scheduler and one worker do once a request is admitted."""
        for index in range(len(request.mm_features)):
            self.sched.update_state_after_alloc(request, index)
        meta = self.sched.build_connector_meta(scheduler_output=None)
        self.worker.start_load_caches(cache, connector_metadata=meta)
        return meta, self.worker.build_connector_worker_meta()


@pytest.fixture
def make_env(monkeypatch):
    envs: list[_Env] = []

    def make(**kwargs) -> _Env:
        env = _Env(monkeypatch, **kwargs)
        envs.append(env)
        return env

    yield make
    for env in envs:
        gate = env.tower._module.gate
        if gate is not None:
            gate.set()
        env.sched.shutdown()


def _request(request_id: str, *images: tuple[str, int, int], modality: str = "image"):
    """A request with `(identifier, offset, rows)` items."""
    return SimpleNamespace(
        request_id=request_id,
        mm_features=[
            MultiModalFeatureSpec(
                data=make_item(identifier, rows),
                modality=modality,
                identifier=identifier,
                mm_position=PlaceholderRange(offset=offset, length=rows),
            )
            for identifier, offset, rows in images
        ],
    )


def _report(*transfer_ids: int):
    return SimpleNamespace(
        ec_connector_worker_meta=ECCPUWorkerMetadata(completed_loads=list(transfer_ids))
    )


def test_request_waits_for_the_tower_and_gets_its_exact_output(make_env):
    gate = threading.Event()
    env = make_env(gate=gate)
    request = _request("r1", ("a", 3, 5))

    # Held while the tower works, and nothing is offered to the workers.
    assert env.admit(request) is False
    assert env.admit(request) is False
    assert env.sched.has_cache_item("a") is False
    assert env.sched.build_connector_meta(scheduler_output=None).loads == {}

    gate.set()
    env.wait("a")
    assert env.admit(request) is True
    assert env.sched.has_cache_item("a") is True

    cache: dict[str, torch.Tensor] = {}
    meta, report = env.load(request, cache)
    assert set(meta.loads) == {"a"}

    expected = direct_embedding(env.tower, request.mm_features[0].data)
    assert cache["a"].dtype == torch.float16
    assert cache["a"].shape == (5, OUT_DIM)
    assert torch.equal(cache["a"], expected)
    assert report is not None
    assert report.completed_loads == [meta.loads["a"][0]]


def test_request_without_a_host_tower_item_never_waits(make_env):
    gate = threading.Event()
    env = make_env(gate=gate)

    # An item of a modality the workers encode themselves.
    audio = _request("r1", ("x", 0, 4), modality="audio")
    assert env.admit(audio) is True
    env.sched.update_state_after_alloc(audio, 0)
    assert env.sched.build_connector_meta(scheduler_output=None).loads == {}
    assert env.sched._encodes == {}
    assert env.tower.num_encodes == 0


def test_item_already_in_the_kv_cache_is_not_encoded(make_env):
    env = make_env()
    request = _request("r1", ("a", 0, 4), ("b", 10, 4))

    # The prefix cache covers the first item entirely.
    assert env.sched.ensure_cache_available(request, 4) is False
    assert set(env.sched._encodes) == {"b"}


def test_two_images_in_one_prompt(make_env):
    env = make_env()
    request = _request("r1", ("a", 0, 4), ("b", 10, 6))

    assert env.admit(request) is False
    env.wait("a")
    env.wait("b")
    assert env.admit(request) is True
    assert env.tower.num_encodes == 2

    cache: dict[str, torch.Tensor] = {}
    meta, _ = env.load(request, cache)
    assert set(meta.loads) == {"a", "b"}
    for feature in request.mm_features:
        assert torch.equal(
            cache[feature.identifier], direct_embedding(env.tower, feature.data)
        )


def test_same_image_twice_is_encoded_once(make_env):
    env = make_env()
    # Twice in one prompt ...
    first = _request("r1", ("a", 0, 4), ("a", 10, 4))
    assert env.admit(first) is False
    env.wait("a")
    assert env.admit(first) is True

    cache: dict[str, torch.Tensor] = {}
    meta, report = env.load(first, cache)
    assert set(meta.loads) == {"a"}
    env.sched.update_connector_output(_report(*report.completed_loads))

    # ... and again in a later request: served from the region.
    second = _request("r2", ("a", 5, 4))
    assert env.admit(second) is True
    assert env.tower.num_encodes == 1


def test_abort_of_a_waiting_request_frees_everything(make_env):
    gate = threading.Event()
    env = make_env(gate=gate)
    running = _request("r1", ("a", 0, 4))
    queued = _request("r2", ("b", 0, 4))

    assert env.admit(running) is False
    assert env.tower._module.started.wait(timeout=60)
    assert env.admit(queued) is False

    # An encode that has not started is cancelled on the spot.
    env.sched.request_finished(queued)
    assert set(env.sched._encodes) == {"a"}

    # One that is running finishes, and its result is dropped.
    env.sched.request_finished(running)
    gate.set()
    env.wait("a")
    assert env.admit(_request("r3")) is True

    assert env.sched._encodes == {}
    assert env.sched._interest == {"r3": set()}
    assert env.sched.has_cache_item("a") is False
    assert not env.sched._cache.has_held_entries()
    assert len(env.sched._cache._free_blocks) == env.region.num_blocks
    assert env.sched.has_pending_push_work() is False
    assert env.tower.num_encodes == 1


def test_abort_keeps_what_another_request_still_waits_for(make_env):
    gate = threading.Event()
    env = make_env(gate=gate)
    first = _request("r1", ("a", 0, 4))
    second = _request("r2", ("a", 0, 4))
    assert env.admit(first) is False
    assert env.admit(second) is False

    env.sched.request_finished(first)
    gate.set()
    env.wait("a")
    assert env.admit(second) is True
    assert env.tower.num_encodes == 1


def test_finished_request_lets_its_embeddings_go(make_env):
    env = make_env()
    request = _request("r1", ("a", 0, 4))
    assert env.admit(request) is False
    env.wait("a")
    assert env.admit(request) is True
    _, report = env.load(request, {})
    env.sched.update_connector_output(_report(*report.completed_loads))

    # Held for the request until it is done with the item ...
    assert env.sched._cache.has_held_entries()
    env.sched.request_finished(request)
    # ... then only cached: still there for a repeat, and evictable.
    assert not env.sched._cache.has_held_entries()
    assert env.sched.has_cache_item("a") is True


def test_item_past_its_placeholders_is_released(make_env):
    env = make_env()
    request = _request("r1", ("a", 0, 4))
    assert env.admit(request) is False
    env.wait("a")
    assert env.admit(request) is True
    _, report = env.load(request, {})
    env.sched.update_connector_output(_report(*report.completed_loads))

    env.sched.update_state_after_free(request, 0)
    assert not env.sched._cache.has_held_entries()
    # Idempotent: the scheduler frees per input, the request finishes later.
    env.sched.update_state_after_free(request, 0)
    env.sched.request_finished(request)


def test_load_is_held_until_every_rank_reports(make_env):
    env = make_env(tensor_parallel_size=2)
    request = _request("r1", ("a", 0, 4))
    assert env.admit(request) is False
    env.wait("a")
    assert env.admit(request) is True
    meta, _ = env.load(request, {})
    (transfer_id, _, _) = meta.loads["a"]
    env.sched.request_finished(request)

    assert env.sched.has_pending_push_work() is True
    env.sched.update_connector_output(_report(transfer_id))
    assert env.sched._cache.has_held_entries()
    # An unknown or repeated report changes nothing.
    env.sched.update_connector_output(_report(transfer_id + 100))
    env.sched.update_connector_output(_report(transfer_id))
    assert not env.sched._cache.has_held_entries()
    assert env.sched.has_pending_push_work() is False
    env.sched.update_connector_output(_report(transfer_id))


def test_tower_failure_fails_the_waiting_requests(make_env):
    env = make_env()
    env.tower._module.fail = True
    request = _request("r1", ("a", 0, 4))

    assert env.admit(request) is False
    env.wait("a")
    assert env.admit(request) is False
    assert env.sched.take_unavailable_requests() == {"r1"}
    assert env.sched.take_unavailable_requests() == set()
    assert env.sched._encodes == {}


def test_item_without_input_in_the_engine_fails_the_request(make_env):
    env = make_env()
    request = _request("r1", ("a", 0, 4))
    request.mm_features[0].data = None

    assert env.admit(request) is False
    assert env.sched.take_unavailable_requests() == {"r1"}
    assert env.tower.num_encodes == 0


def test_full_region_makes_the_embedding_wait_for_room(make_env):
    # Room for two 4-row items.
    env = make_env(num_blocks=8)
    holders = [_request("r1", ("a", 0, 4)), _request("r2", ("b", 0, 4))]
    for request, identifier in zip(holders, "ab"):
        assert env.admit(request) is False
        env.wait(identifier)
        assert env.admit(request) is True

    late = _request("r3", ("c", 0, 4))
    assert env.admit(late) is False
    env.wait("c")
    # Encoded, but both residents are held by requests that still need them.
    assert env.admit(late) is False
    assert env.sched.take_unavailable_requests() == set()

    env.sched.request_finished(holders[0])
    assert env.admit(late) is True
    assert env.sched.has_cache_item("a") is False
    assert env.sched.has_cache_item("b") is True
    assert env.tower.num_encodes == 3


def test_embedding_that_never_finds_room_fails_the_request(make_env, monkeypatch):
    env = make_env(num_blocks=8)
    holder = _request("r1", ("a", 0, 8))
    assert env.admit(holder) is False
    env.wait("a")
    assert env.admit(holder) is True

    monkeypatch.setattr(sched_mod, "_SPACE_WAIT_TIMEOUT_S", -1.0)
    late = _request("r2", ("b", 0, 4))
    assert env.admit(late) is False
    env.wait("b")
    assert env.admit(late) is False
    assert env.sched.take_unavailable_requests() == {"r2"}

    # Larger than the whole region: no point in waiting at all.
    monkeypatch.setattr(sched_mod, "_SPACE_WAIT_TIMEOUT_S", 60.0)
    huge = _request("r3", ("c", 0, 9))
    assert env.admit(huge) is False
    env.wait("c")
    assert env.admit(huge) is False
    assert env.sched.take_unavailable_requests() == {"r3"}


@pytest.mark.parametrize("block_ids", [(0, 1, 2), (5, 1, 2, 7), (3,)])
def test_write_blocks_fills_the_given_blocks_in_order(block_ids):
    blocks = torch.zeros(8, BLOCK_BYTES, dtype=torch.int8)
    # Ends partway into its last block.
    rows = len(block_ids) * OUT_DIM - 3
    embedding = torch.randn(rows, 1).to(torch.float16)
    ordered = tuple(sorted(block_ids))

    write_blocks(blocks, ordered, embedding)

    nbytes = rows * 2
    written = blocks[list(ordered)].reshape(-1)[:nbytes].view(torch.float16)
    assert torch.equal(written, embedding.reshape(-1))
    untouched = [i for i in range(8) if i not in block_ids]
    assert not blocks[untouched].any()


def test_shutdown_stops_the_towers_and_releases_the_region(make_env):
    gate = threading.Event()
    env = make_env(gate=gate)
    assert env.admit(_request("r1", ("a", 0, 4))) is False
    assert env.admit(_request("r2", ("b", 0, 4))) is False

    gate.set()
    env.sched.shutdown()
    assert env.sched._encodes == {}
    assert env.region.cleaned_up
    with pytest.raises(RuntimeError):
        env.tower.submit([])

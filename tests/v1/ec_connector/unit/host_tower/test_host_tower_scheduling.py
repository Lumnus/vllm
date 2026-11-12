# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The scheduler with a host tower behind it: an image request waits for its
embeddings while text requests are scheduled, and joins them afterwards."""

import threading

import pytest
import torch

import vllm.distributed.ec_transfer.ec_connector.host_tower.scheduler as sched_mod
import vllm.distributed.ec_transfer.ec_connector.host_tower.worker as worker_mod
from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.ec_connector.unit.host_tower.utils import (
    FakeRegion,
    direct_embedding,
    make_item,
    make_tower,
)
from vllm.config.ec_transfer import ECTransferConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECCPUWorkerMetadata
from vllm.distributed.ec_transfer.ec_connector.host_tower.connector import (
    ECHostTowerConnector,
)
from vllm.distributed.ec_transfer.ec_connector.host_tower.scheduler import (
    ECHostTowerScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.host_tower.worker import (
    ECHostTowerWorker,
)
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.outputs import ECConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test

MODEL = "llava-hf/llava-1.5-7b-hf"


class _Engine:
    """A scheduler with the host tower connector, and a stand-in for the
    executor: one worker that loads what it is told to and samples a token
    for every request that reached the end of its prompt."""

    def __init__(self, monkeypatch, gate: threading.Event | None) -> None:
        self.scheduler = create_scheduler(model=MODEL, max_num_batched_tokens=1024)
        config = self.scheduler.vllm_config
        config.ec_transfer_config = ECTransferConfig(
            ec_connector="ECHostTowerConnector", ec_role="ec_consumer"
        )

        embed_bytes = config.model_config.get_inputs_embeds_size() * 2
        region = FakeRegion(64, embed_bytes)
        monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda _: region)
        monkeypatch.setattr(worker_mod, "create_ec_shared_region", lambda _: region)

        self.tower = make_tower(gate)
        self.tower.to_host()
        monkeypatch.setattr(
            ECHostTowerConnector,
            "_make_scheduler",
            lambda _, cfg: ECHostTowerScheduler(cfg, {self.tower.name: self.tower}),
        )
        self.connector = ECHostTowerConnector(config, ECConnectorRole.SCHEDULER)
        self.scheduler.ec_connector = self.connector
        self.delegate = self.connector.connector_scheduler

        self.worker = ECHostTowerWorker(config)
        self.encoder_cache: dict[str, torch.Tensor] = {}
        self.requests = {}

    def add(self, request) -> None:
        self.requests[request.request_id] = request
        self.scheduler.add_request(request)

    def step(self):
        """One engine step. Returns the scheduler output."""
        output = self.scheduler.schedule()
        # No host tower item is ever handed to the workers to encode.
        assert not output.scheduled_encoder_inputs

        self.worker.start_load_caches(
            self.encoder_cache, connector_metadata=output.ec_connector_metadata
        )
        req_ids, sampled = [], []
        for req_id in output.num_scheduled_tokens:
            request = self.requests[req_id]
            done = request.num_computed_tokens >= request.num_tokens
            req_ids.append(req_id)
            sampled.append([1000] if done else [])
        self.scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                sampled_token_ids=sampled,
                ec_connector_output=ECConnectorOutput(
                    ec_connector_worker_meta=self.worker.build_connector_worker_meta()
                ),
            ),
        )
        return output

    def wait(self, identifier: str) -> None:
        self.delegate._encodes[identifier].future.result(timeout=60)

    def shutdown(self) -> None:
        self.connector.shutdown()


@pytest.fixture
def make_engine(monkeypatch):
    engines: list[_Engine] = []

    def make(gate: threading.Event | None = None) -> _Engine:
        engine = _Engine(monkeypatch, gate)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        gate = engine.tower._module.gate
        if gate is not None:
            gate.set()
        engine.shutdown()


def _text(req_id: str, num_tokens: int = 20):
    (request,) = create_requests(
        1, num_tokens=num_tokens, req_ids=[req_id], max_tokens=64
    )
    return request


def _image(req_id: str, *images: tuple[str, int, int], num_tokens: int = 100):
    """A request with `(identifier, offset, rows)` images in its prompt."""
    (request,) = create_requests(
        1,
        num_tokens=num_tokens,
        req_ids=[req_id],
        max_tokens=64,
        mm_hashes_list=[[identifier for identifier, _, _ in images]],
        mm_positions=[
            [PlaceholderRange(offset=offset, length=rows) for _, offset, rows in images]
        ],
    )
    for feature, (identifier, _, rows) in zip(request.mm_features, images):
        feature.data = make_item(identifier, rows)
    return request


def test_image_request_waits_while_text_requests_are_scheduled(make_engine):
    gate = threading.Event()
    engine = make_engine(gate)
    image = _image("image", ("a", 10, 16))
    engine.add(image)
    engine.add(_text("text-0"))

    # The image arrived first and is skipped; the text request runs.
    output = engine.step()
    assert set(output.num_scheduled_tokens) == {"text-0"}
    assert output.ec_connector_metadata.loads == {}
    assert image.status == RequestStatus.WAITING

    # It keeps decoding, and a text request that arrives later is admitted,
    # all while the tower has not answered.
    engine.add(_text("text-1"))
    for _ in range(3):
        output = engine.step()
        assert set(output.num_scheduled_tokens) == {"text-0", "text-1"}
    assert engine.tower.num_encodes == 0
    assert len(engine.requests["text-0"].output_token_ids) == 4

    gate.set()
    engine.wait("a")

    # Admitted with its embeddings on the way to the workers, not to be
    # computed there.
    output = engine.step()
    assert set(output.num_scheduled_tokens) == {"image", "text-0", "text-1"}
    assert set(output.ec_connector_metadata.loads) == {"a"}
    assert image.status == RequestStatus.RUNNING
    assert torch.equal(
        engine.encoder_cache["a"],
        direct_embedding(engine.tower, image.mm_features[0].data),
    )

    # The worker's report came back with that step: the load is settled and
    # the image request decodes like any other.
    assert engine.delegate.has_pending_push_work() is False
    output = engine.step()
    assert "image" in output.num_scheduled_tokens
    assert len(image.output_token_ids) == 2
    assert engine.tower.num_encodes == 1


def test_abort_of_a_waiting_image_request_frees_everything(make_engine):
    gate = threading.Event()
    engine = make_engine(gate)
    engine.add(_image("image-0", ("a", 0, 8)))
    engine.add(_image("image-1", ("b", 0, 8)))
    engine.add(_text("text-0"))

    output = engine.step()
    assert set(output.num_scheduled_tokens) == {"text-0"}
    assert engine.tower._module.started.wait(timeout=60)

    engine.scheduler.finish_requests(
        ["image-0", "image-1"], RequestStatus.FINISHED_ABORTED
    )
    # The encode that had not started is gone at once.
    assert set(engine.delegate._encodes) == {"a"}

    # The one that was running finishes and its result is dropped.
    gate.set()
    engine.wait("a")
    output = engine.step()
    assert set(output.num_scheduled_tokens) == {"text-0"}

    delegate = engine.delegate
    assert delegate._encodes == {}
    assert delegate._interest == {}
    assert delegate.has_cache_item("a") is False
    assert delegate.has_cache_item("b") is False
    assert not delegate._cache.has_held_entries()
    assert len(delegate._cache._free_blocks) == delegate._region.num_blocks
    assert engine.scheduler.get_num_unfinished_requests() == 1
    assert engine.tower.num_encodes == 1


def test_two_images_in_one_prompt_and_a_repeat(make_engine):
    engine = make_engine()
    first = _image("first", ("a", 5, 8), ("b", 40, 12))
    engine.add(first)
    engine.step()
    engine.wait("a")
    engine.wait("b")

    output = engine.step()
    assert set(output.ec_connector_metadata.loads) == {"a", "b"}
    for feature in first.mm_features:
        assert torch.equal(
            engine.encoder_cache[feature.identifier],
            direct_embedding(engine.tower, feature.data),
        )
    assert engine.tower.num_encodes == 2

    # The same image in a later prompt costs no second encode.
    repeat = _image("repeat", ("a", 20, 8))
    engine.add(repeat)
    output = engine.step()
    assert "repeat" in output.num_scheduled_tokens
    assert engine.tower.num_encodes == 2


def test_failed_encode_finishes_the_request_with_an_error(make_engine):
    engine = make_engine()
    engine.tower._module.fail = True
    image = _image("image", ("a", 0, 8))
    engine.add(image)
    engine.add(_text("text-0"))

    engine.step()
    engine.delegate._encodes["a"].future.exception(timeout=60)
    engine.step()
    engine.step()

    assert image.status == RequestStatus.FINISHED_ERROR
    assert engine.delegate._interest.get("image") is None
    assert engine.requests["text-0"].status == RequestStatus.RUNNING


def test_worker_report_type_matches_the_scheduler(make_engine):
    """The reports the workers aggregate are the ones the scheduler reads."""
    first = ECCPUWorkerMetadata(completed_loads=[1])
    merged = first.aggregate(ECCPUWorkerMetadata(completed_loads=[1]))
    assert isinstance(merged, ECCPUWorkerMetadata)
    assert merged.completed_loads == [1, 1]

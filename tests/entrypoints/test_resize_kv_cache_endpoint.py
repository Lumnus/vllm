# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dev endpoint that grows the KV cache of a serving engine."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.dev.cache import api_router


class FakeEngineClient:
    def __init__(self, result: int = 0, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.requested: list[int] = []

    async def resize_kv_cache(self, target_num_blocks: int) -> int:
        self.requested.append(target_num_blocks)
        if self.error is not None:
            raise self.error
        return self.result


def _client(engine: FakeEngineClient) -> TestClient:
    app = FastAPI()
    app.state.engine_client = engine
    api_router.attach_router(app)
    return TestClient(app)


def test_resize_returns_the_block_count_in_force():
    engine = FakeEngineClient(result=9000)
    response = _client(engine).post("/resize_kv_cache", params={"num_blocks": 12000})
    assert response.status_code == 200
    assert response.json() == {"num_blocks": 9000}
    assert engine.requested == [12000]


@pytest.mark.parametrize("num_blocks", ["0", "-5", "many"])
def test_resize_rejects_a_bad_block_count(num_blocks):
    engine = FakeEngineClient()
    response = _client(engine).post(
        "/resize_kv_cache", params={"num_blocks": num_blocks}
    )
    assert response.status_code == 422
    assert engine.requested == []


def test_resize_reports_what_the_engine_refused():
    engine = FakeEngineClient(error=Exception("Call to resize_kv_cache failed: asleep"))
    response = _client(engine).post("/resize_kv_cache", params={"num_blocks": 100})
    assert response.status_code == 400
    assert "asleep" in response.json()["error"]

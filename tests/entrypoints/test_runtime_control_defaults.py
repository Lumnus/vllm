# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request defaults changed in a running server (penalties, sampling, the
thinking budget, the Han mask default), answered by the API server process;
and the restart_required flag on refusals."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.dev.control import api_router
from vllm.v1.sample import b70_cjk

from .test_runtime_control_endpoint import FakeEngineClient


def _app(api_server_count: int = 1):
    app = FastAPI()
    app.state.engine_client = FakeEngineClient()
    app.state.args = Namespace(api_server_count=api_server_count)
    app.state.openai_serving_chat = SimpleNamespace(
        default_sampling_params={"temperature": 0.6, "top_p": 0.95}
    )
    app.state.openai_serving_completion = SimpleNamespace(
        default_sampling_params={"temperature": 0.6, "top_p": 0.95}
    )
    api_router.attach_router(app)
    return app, TestClient(app)


@pytest.fixture(autouse=True)
def _cjk_default_back():
    yield
    b70_cjk.set_default_mode("auto")


def test_sampling_defaults_change_and_go_back():
    app, client = _app()
    r = client.post(
        "/runtime_control/defaults.set",
        json={
            "temperature": 0.7,
            "presence_penalty": 1.5,
            "thinking_token_budget": 512,
        },
    )
    assert r.status_code == 200, r.json()
    assert r.json()["changes"] == {
        "temperature": [0.6, 0.7],
        "presence_penalty": [None, 1.5],
        "thinking_token_budget": [None, 512],
    }
    for serving in (app.state.openai_serving_chat, app.state.openai_serving_completion):
        assert serving.default_sampling_params == {
            "temperature": 0.7,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "thinking_token_budget": 512,
        }
    r = client.post(
        "/runtime_control/defaults.set",
        json={"temperature": None, "presence_penalty": None},
    )
    assert r.json()["changes"] == {
        "temperature": [0.7, 0.6],
        "presence_penalty": [1.5, None],
    }
    state = client.post("/runtime_control/defaults.state").json()
    assert state["defaults"]["temperature"] == 0.6
    assert state["start"]["temperature"] == 0.6
    assert state["defaults"]["thinking_token_budget"] == 512
    assert app.state.engine_client.calls == []  # never reached the engine


def test_bad_values_and_unknown_names_change_nothing():
    app, client = _app()
    for body in ({"top_p": 0}, {"top_k": 1.5}, {"temperature": -1}, {"seed": 1}):
        r = client.post("/runtime_control/defaults.set", json=body)
        assert r.status_code == 400 and "restart_required" not in r.json()
    assert app.state.openai_serving_chat.default_sampling_params == {
        "temperature": 0.6,
        "top_p": 0.95,
    }


def test_several_api_servers_need_a_restart():
    _, client = _app(api_server_count=2)
    r = client.post("/runtime_control/defaults.set", json={"temperature": 0.7})
    assert r.status_code == 400
    assert r.json()["restart_required"] is True and "2" in r.json()["error"]


def test_the_han_mask_default(monkeypatch):
    _, client = _app()
    monkeypatch.delenv(b70_cjk.ENV, raising=False)
    r = client.post("/runtime_control/defaults.set", json={"cjk_mask_default": "off"})
    assert r.status_code == 400 and r.json()["restart_required"] is True
    monkeypatch.setenv(b70_cjk.ENV, "1")
    msgs = [{"role": "user", "content": "hello"}]
    assert b70_cjk.chat_extra_args({}, msgs, None)["cjk_mask"] is True
    r = client.post("/runtime_control/defaults.set", json={"cjk_mask_default": "off"})
    assert r.status_code == 200 and r.json()["changes"] == {
        "cjk_mask_default": ["auto", "off"]
    }
    assert b70_cjk.chat_extra_args({}, msgs, None)["cjk_mask"] is False
    # an explicit request value still wins
    assert b70_cjk.chat_extra_args({"cjk_mask": "on"}, msgs, None)["cjk_mask"] is True
    client.post("/runtime_control/defaults.set", json={"cjk_mask_default": None})
    assert b70_cjk.default_mode() == "auto"
    bad = client.post("/runtime_control/defaults.set", json={"cjk_mask_default": "x"})
    assert bad.status_code == 400


def test_an_engine_refusal_that_needs_a_restart_says_so():
    app = FastAPI()
    app.state.engine_client = FakeEngineClient(
        error=Exception("restart_required: num_speculative_tokens 4 is above 3")
    )
    api_router.attach_router(app)
    r = TestClient(app).post(
        "/runtime_control/spec.set", json={"num_speculative_tokens": 4}
    )
    assert r.status_code == 400
    assert r.json() == {
        "error": "num_speculative_tokens 4 is above 3",
        "restart_required": True,
    }


def test_the_thinking_budget_default_is_read_per_request():
    def params(**kw):
        request = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], **kw
        )
        return request.to_sampling_params(16, {"thinking_token_budget": 64})

    assert params().thinking_token_budget == 64
    assert params(thinking_token_budget=8).thinking_token_budget == 8
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )
    assert request.to_sampling_params(16, {}).thinking_token_budget is None

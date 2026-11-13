# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The runtime control routes, and how the server registers them."""

from argparse import Namespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.launchers.api_server import routers
from vllm.entrypoints.serve.dev.control import api_router


class FakeEngineClient:
    def __init__(self, reply=None, error: Exception | None = None) -> None:
        self.reply = reply or {"op": "eplb.pause", "applied": "all", "ranks": []}
        self.error = error
        self.calls: list[tuple] = []

    async def runtime_control(self, op, args):
        self.calls.append((op, args))
        if self.error is not None:
            raise self.error
        return self.reply

    async def get_runtime_control_state(self, only=None):
        self.calls.append(("state", only))
        if self.error is not None:
            raise self.error
        return {"ranks": [], "kv": {}}


def _client(engine, keys=None) -> TestClient:
    app = FastAPI()
    app.state.engine_client = engine
    api_router.attach_router(app, keys)
    return TestClient(app)


def test_state_and_a_part_of_it():
    engine = FakeEngineClient()
    client = _client(engine)
    assert client.get("/runtime_control").json() == {"ranks": [], "kv": {}}
    client.get("/runtime_control", params={"only": "eplb,kv"})
    assert engine.calls == [("state", None), ("state", ["eplb", "kv"])]


def test_an_operation_with_its_body():
    engine = FakeEngineClient()
    client = _client(engine)
    response = client.post("/runtime_control/eplb.set", json={"step_interval": 300})
    assert response.status_code == 200
    assert engine.calls == [("eplb.set", {"step_interval": 300})]
    assert client.post("/runtime_control/eplb.pause").status_code == 200
    assert engine.calls[-1] == ("eplb.pause", {})


def test_refused_partial_and_bad_bodies():
    client = _client(FakeEngineClient(error=Exception("rank 0: paused")))
    response = client.post("/runtime_control/eplb.rearrange")
    assert response.status_code == 400
    assert response.json() == {"error": "rank 0: paused"}
    partial = {"op": "store.scrub", "applied": "partial", "ranks": []}
    assert (
        _client(FakeEngineClient(partial))
        .post("/runtime_control/store.scrub")
        .status_code
        == 409
    )
    client = _client(FakeEngineClient())
    assert client.post("/runtime_control/eplb.set", content=b"[1]").status_code == 400
    assert (
        client.post("/runtime_control/eplb.set", content=b"{not json").status_code
        == 400
    )


def test_the_api_key_guards_these_routes():
    client = _client(FakeEngineClient(), keys=["sekrit"])
    assert client.get("/runtime_control").status_code == 401
    assert client.post("/runtime_control/eplb.pause").status_code == 401
    ok = client.get("/runtime_control", headers={"Authorization": "Bearer sekrit"})
    assert ok.status_code == 200
    bad = client.get("/runtime_control", headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401


def test_the_flag_parses_and_is_off_by_default():
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser())
    assert parser.parse_args([]).enable_runtime_control is False
    assert parser.parse_args(["--enable-runtime-control"]).enable_runtime_control


def _paths(monkeypatch, dev_mode: bool, **args) -> set[str]:
    monkeypatch.setattr(routers.envs, "VLLM_SERVER_DEV_MODE", dev_mode)
    monkeypatch.setattr(routers.envs, "VLLM_API_KEY", None)
    app = FastAPI()
    routers.register_runtime_control_router(Namespace(**args), app)
    return {getattr(r, "path", "") for r in app.routes}


def test_registered_with_the_flag_or_in_development_mode(monkeypatch):
    assert "/runtime_control" not in _paths(monkeypatch, False)
    assert "/runtime_control" in _paths(monkeypatch, False, enable_runtime_control=True)
    assert "/runtime_control" in _paths(monkeypatch, True)
    # Only these routes come with the flag; the server's key guards them.
    monkeypatch.setattr(routers.envs, "VLLM_SERVER_DEV_MODE", False)
    app = FastAPI()
    app.state.engine_client = FakeEngineClient()
    routers.register_runtime_control_router(
        Namespace(enable_runtime_control=True, api_key=["k"]), app
    )
    assert {getattr(r, "path", "") for r in app.routes} & {
        "/collective_rpc",
        "/resize_kv_cache",
    } == set()
    assert TestClient(app).get("/runtime_control").status_code == 401

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The runtime control surface: read and change the dynamic subsystems of a
running engine (EPLB, the expert tiers and store, the KV pool, the encoder
towers, the routing trace, the PLE row cache).

Registered with the development routes (`VLLM_SERVER_DEV_MODE=1`), or alone
with `--enable-runtime-control`, which exposes these routes without the rest
of the development family (`/collective_rpc` calls any worker method). When
the server has an API key, these routes ask for it too (the key middleware
guards only `/v1`, `/v2`, `/inference` and `/cohere`).

    GET  /runtime_control                 the state, engine core and every rank
    GET  /runtime_control?only=eplb,kv    a part of each rank's state
    POST /runtime_control/{op}            apply one operation; JSON body = its args

A refused operation answers 400 with `{"error": ...}`; nothing was changed.
An operation that some rank could not apply answers 409 with the per-rank
replies and whether the others were undone (`applied`: `partial` or `none`).
"""

import hashlib
import json
import secrets
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _check_key(request: Request) -> None:
    hashes: list[bytes] = getattr(request.app.state, "runtime_control_keys", [])
    if not hashes:
        return
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() == "bearer":
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        if any(secrets.compare_digest(digest, h) for h in hashes):
            return
    raise HTTPException(
        status_code=HTTPStatus.UNAUTHORIZED.value, detail="Unauthorized"
    )


@router.get("/runtime_control", dependencies=[Depends(_check_key)])
async def runtime_control_state(raw_request: Request, only: str | None = None):
    """The state of every dynamic subsystem: the engine core's (KV resize,
    towers) and each rank's (EPLB, tiers, store, KV pool, trace, PLE row
    cache, armed faults), and whether the ranks agree."""
    parts = [p for p in only.split(",") if p] if only else None
    try:
        state = await engine_client(raw_request).get_runtime_control_state(parts)
    except Exception as e:  # noqa: BLE001 - reported as a plain Exception
        return JSONResponse(content={"error": str(e)}, status_code=500)
    return JSONResponse(content=state)


@router.get("/runtime_control/schema", dependencies=[Depends(_check_key)])
async def runtime_control_schema(raw_request: Request):
    """Every operation and parameter of the surface, with type, unit, range
    or choices, default, current value, whether it is live (and through
    which operation) or needs a restart (and why). The request defaults'
    values are this API server process's."""
    from vllm.entrypoints.serve.dev.control import server_defaults

    try:
        reply = await engine_client(raw_request).runtime_control("schema", {})
    except Exception as e:  # noqa: BLE001 - reported as a plain Exception
        return JSONResponse(content={"error": str(e)}, status_code=500)
    now = server_defaults.state(raw_request.app.state)
    for param in reply.get("params", []):
        if param.get("side") != "server" or not param["name"].startswith("defaults."):
            continue
        key = param["name"][len("defaults.") :]
        param["value"] = (
            now.get(key) if key == server_defaults.CJK_KEY else now["defaults"].get(key)
        )
    return JSONResponse(content=reply)


@router.post("/runtime_control/{op}", dependencies=[Depends(_check_key)])
async def runtime_control(op: str, raw_request: Request):
    """Apply `op` with the JSON body as its arguments (an empty body is
    `{}`). See `vllm/v1/engine/runtime_control.py` for the operations."""
    raw = await raw_request.body()
    try:
        args = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        return JSONResponse(
            content={"error": f"JSON decode error: {e}"}, status_code=400
        )
    if not isinstance(args, dict):
        return JSONResponse(
            content={"error": "the body must be a JSON object"}, status_code=400
        )
    logger.info("Runtime control: %s %s", op, args)
    if op.startswith("defaults."):
        return _server_defaults(op, args, raw_request)
    if op == "profile.define" and isinstance(args.get("defaults"), dict):
        # The request defaults of a profile are this process's; check them
        # now so that applying the profile cannot fail on them later.
        from vllm.entrypoints.serve.dev.control import server_defaults

        try:
            server_defaults.check(raw_request.app.state, args["defaults"])
        except server_defaults.DefaultsRefused as e:
            content: dict = {"error": f"defaults: {e}"}
            if e.restart_required:
                content["restart_required"] = True
            return JSONResponse(content=content, status_code=400)
    try:
        reply = await engine_client(raw_request).runtime_control(op, args)
    except Exception as e:  # noqa: BLE001 - the engine core reports a refusal so
        logger.warning("Runtime control: %s refused: %s", op, e)
        return JSONResponse(content=_refusal(str(e)), status_code=400)
    if op in ("profile.apply", "profile.state"):
        _profile_defaults(op, reply, raw_request)
    status = {"all": 200, "applying": 202}.get(reply.get("applied"), 409)
    return JSONResponse(content=reply, status_code=status)


def _profile_defaults(op: str, reply: dict, raw_request: Request) -> None:
    """A profile's request defaults are applied here, in the process that
    reads them, once the engine has accepted the profile (they hold no
    accelerator memory, so they need not wait for its memory steps); the
    state adds their target against their value now."""
    from vllm.entrypoints.serve.dev.control import server_defaults

    wanted = reply.pop("server_defaults", None)
    if not wanted:
        return
    app_state = raw_request.app.state
    if op == "profile.apply" and reply.get("applied") in ("all", "applying"):
        try:
            reply["defaults"] = server_defaults.apply(app_state, wanted)["changes"]
        except server_defaults.DefaultsRefused as e:
            reply["defaults_error"] = str(e)
            reply["applied"] = "partial"
    now = server_defaults.state(app_state)
    targets = reply.setdefault("targets", {}) if op == "profile.state" else {}
    for key, value in wanted.items():
        actual = (
            now.get(server_defaults.CJK_KEY)
            if key == server_defaults.CJK_KEY
            else now["defaults"].get(key)
        )
        targets[f"defaults.{key}"] = {
            "target": value,
            "actual": actual,
            "match": value == actual,
        }


# A refusal whose reason a restart (with other flags) removes starts with it.
RESTART_REQUIRED = "restart_required: "


def _refusal(text: str) -> dict:
    """The refusal reply; `restart_required` true when the setting cannot
    change in a running engine (the reason says why and what to restart
    with)."""
    at = text.find(RESTART_REQUIRED)
    if at < 0:
        return {"error": text}
    return {"error": text[at + len(RESTART_REQUIRED) :], "restart_required": True}


def _server_defaults(op: str, args: dict, raw_request: Request) -> JSONResponse:
    """`defaults.state` / `defaults.set`: request defaults of this API server
    process (see server_defaults.py), answered here, not by the engine."""
    from vllm.entrypoints.serve.dev.control import server_defaults

    app_state = raw_request.app.state
    if op == "defaults.state":
        return JSONResponse(content={"op": op, **server_defaults.state(app_state)})
    if op != "defaults.set":
        return JSONResponse(
            content={
                "error": f"unknown operation {op!r}; defaults.set, defaults.state"
            },
            status_code=400,
        )
    try:
        reply = server_defaults.apply(app_state, args)
    except server_defaults.DefaultsRefused as e:
        logger.warning("Runtime control: %s refused: %s", op, e)
        content: dict = {"error": str(e)}
        if e.restart_required:
            content["restart_required"] = True
        return JSONResponse(content=content, status_code=400)
    return JSONResponse(content=reply)


def attach_router(app: FastAPI, api_keys: list[str] | None = None) -> None:
    app.state.runtime_control_keys = [
        hashlib.sha256(k.encode("utf-8")).digest() for k in (api_keys or []) if k
    ]
    app.include_router(router)

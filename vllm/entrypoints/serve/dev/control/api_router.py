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
    try:
        reply = await engine_client(raw_request).runtime_control(op, args)
    except Exception as e:  # noqa: BLE001 - the engine core reports a refusal so
        logger.warning("Runtime control: %s refused: %s", op, e)
        return JSONResponse(content={"error": str(e)}, status_code=400)
    status = 200 if reply.get("applied") == "all" else 409
    return JSONResponse(content=reply, status_code=status)


def attach_router(app: FastAPI, api_keys: list[str] | None = None) -> None:
    app.state.runtime_control_keys = [
        hashlib.sha256(k.encode("utf-8")).digest() for k in (api_keys or []) if k
    ]
    app.include_router(router)

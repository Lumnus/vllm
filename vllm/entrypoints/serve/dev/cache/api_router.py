# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/reset_prefix_cache")
async def reset_prefix_cache(
    raw_request: Request,
    reset_running_requests: bool = Query(default=False),
    reset_external: bool = Query(default=False),
):
    """Reset the local prefix cache.

    Optionally, if the query parameter `reset_external=true`
    also resets the external (connector-managed) prefix cache.

    Returns `{"success": bool}`. The reset fails (`success=false`) while
    blocks are still held, e.g. by running requests or in-flight async KV
    offload transfers; callers may retry.

    Example:
       POST /reset_prefix_cache?reset_external=true

    """
    logger.info("Resetting prefix cache...")

    success = await engine_client(raw_request).reset_prefix_cache(
        reset_running_requests, reset_external
    )
    return JSONResponse(content={"success": bool(success)})


@router.post("/resize_kv_cache")
async def resize_kv_cache(
    raw_request: Request,
    num_blocks: int = Query(gt=0),
    deadline_s: float | None = Query(default=None, ge=0),
    preempt: bool = Query(default=False),
):
    """Resize the extensible KV cache toward `num_blocks` blocks while serving.

    Above the current size the engine commits what the memory on every card
    fits, up to `num_blocks` and the reserved capacity, and returns when done
    (`state` "finished" or "unchanged"). Below it the engine starts a shrink
    and returns at once, usually with `state` "draining": the blocks from
    `num_blocks` on are handed out no more, and are given back once the
    requests holding them finish; read the progress with
    `GET /kv_cache_resize`. After `deadline_s` (engine default if unset, 0 for
    none) a drain still waiting is aborted, or with `preempt=true` the
    requests holding those blocks are preempted. The record carries
    `num_blocks` (in force now), `from_blocks`, `to_blocks`, `fenced_blocks`,
    `held_blocks`, `released_bytes_per_rank` and a `message`.

    Example:
       POST /resize_kv_cache?num_blocks=9000
       POST /resize_kv_cache?num_blocks=6000&deadline_s=30&preempt=true

    """
    logger.info("Resizing the KV cache toward %d blocks...", num_blocks)
    try:
        record = await engine_client(raw_request).resize_kv_cache(
            num_blocks, deadline_s, preempt
        )
    except Exception as e:  # noqa: BLE001
        # The engine core reports a failed call as a plain Exception.
        logger.warning("Resizing the KV cache failed: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=400)
    return JSONResponse(content=record)


@router.get("/kv_cache_resize")
async def kv_cache_resize_state(raw_request: Request):
    """The record of the current or last KV cache resize (see
    `POST /resize_kv_cache`), or `{"state": "idle", "num_blocks": int}`."""
    try:
        record = await engine_client(raw_request).get_kv_cache_resize_state()
    except Exception as e:  # noqa: BLE001
        return JSONResponse(content={"error": str(e)}, status_code=500)
    return JSONResponse(content=record)


@router.post("/reset_mm_cache")
async def reset_mm_cache(raw_request: Request):
    """Reset the multi-modal cache. Note that we currently do not check if the
    multi-modal cache is successfully reset in the API server.
    """
    logger.info("Resetting multi-modal cache...")
    await engine_client(raw_request).reset_mm_cache()
    return Response(status_code=200)


@router.post("/reset_encoder_cache")
async def reset_encoder_cache(raw_request: Request):
    """Reset the encoder cache. Note that we currently do not check if the
    encoder cache is successfully reset in the API server.
    """
    logger.info("Resetting encoder cache...")
    await engine_client(raw_request).reset_encoder_cache()
    return Response(status_code=200)


def attach_router(app: FastAPI):
    app.include_router(router)

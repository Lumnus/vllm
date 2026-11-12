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
):
    """Grow the extensible KV cache toward `num_blocks` blocks while serving.

    The engine commits what the memory on every card fits, up to `num_blocks`
    and the reserved capacity, and never fewer blocks than it has. Returns
    `{"num_blocks": int}`, the count in force afterwards; a request at or below
    the current size changes nothing.

    Example:
       POST /resize_kv_cache?num_blocks=9000

    """
    logger.info("Resizing the KV cache toward %d blocks...", num_blocks)
    try:
        committed = await engine_client(raw_request).resize_kv_cache(num_blocks)
    except Exception as e:  # noqa: BLE001
        # The engine core reports a failed call as a plain Exception.
        logger.warning("Resizing the KV cache failed: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=400)
    return JSONResponse(content={"num_blocks": committed})


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

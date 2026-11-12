# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECHostTowerWorker: worker delegate of the ECHostTowerConnector.

Runs in every worker. It never encodes: it copies the embeddings a host tower
left in the shared region into this rank's encoder cache, where the model
finds them as if its own tower had produced them.
"""

from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECCPUWorkerMetadata,
    create_ec_shared_region,
)
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ECHostTowerWorker:
    """Worker delegate for the ECHostTowerConnector."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        from vllm.platforms import current_platform

        self._region = create_ec_shared_region(vllm_config)
        dtype = vllm_config.model_config.dtype
        assert isinstance(dtype, torch.dtype)
        self._dtype = dtype

        # A bare device type resolves to this worker's own device, not to
        # device 0, on every platform that selects the device per process.
        device = current_platform.device_type
        if current_platform.is_cuda_alike():
            device = f"{device}:{current_platform.current_device()}"
        self._device = device

        # Transfer ids copied this step, reported back so the scheduler can
        # let the region blocks go.
        self._completed_loads: list[int] = []
        self._logged_first_load = False

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        for mm_hash, (
            transfer_id,
            block_ids,
            shape,
        ) in connector_metadata.loads.items():
            nbytes = shape[0] * shape[1] * self._dtype.itemsize
            embedding = (
                self._region.blocks[block_ids]
                .reshape(-1)[:nbytes]
                .view(self._dtype)
                .view(shape)
            )
            # A blocking copy of one item's embeddings, once per item: the
            # request has waited for its encode already.
            with gpu_sync_allowed():
                encoder_cache[mm_hash] = embedding.to(self._device)
            self._completed_loads.append(transfer_id)
            # The first copy is logged as proof of the path, the rest at debug.
            log = logger.debug if self._logged_first_load else logger.info
            self._logged_first_load = True
            log(
                "Host tower embeddings for item %s: %d x %d %s (%d bytes) "
                "copied to this worker's encoder cache on %s.",
                mm_hash,
                shape[0],
                shape[1],
                str(self._dtype).removeprefix("torch."),
                nbytes,
                self._device,
            )

    def build_connector_worker_meta(self) -> ECCPUWorkerMetadata | None:
        if not self._completed_loads:
            return None
        completed, self._completed_loads = self._completed_loads, []
        return ECCPUWorkerMetadata(completed_loads=completed)

    def shutdown(self) -> None:
        self._completed_loads.clear()
        try:
            self._region.cleanup()
        except Exception:
            logger.debug("host tower: worker region cleanup failed", exc_info=True)

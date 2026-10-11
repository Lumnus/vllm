# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_processing
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def get_external_weight_memory(self, vllm_config: VllmConfig) -> int:
        """Get weights memory from external process;
        0 when the weights are not external."""
        return 0

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def create_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Create a model with the given configurations."""
        model = initialize_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )
        log_online_quantization(vllm_config)
        log_model_inspection(model)
        return model

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                time_before_load = time.perf_counter()
                model = self.create_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
                time_after_load = time.perf_counter()
                logger.info_once(
                    "Initializing model took %.6f seconds",
                    time_after_load - time_before_load,
                )

            logger.debug("Loading weights on %s ...", load_device)
            self.load_weights(model, model_config)

            # Release transient tensors held by PyTorch's caching allocator
            # before weight post-processing and raw CUDA driver allocations.
            gc.collect()
            torch.accelerator.empty_cache()

            # Log peak GPU memory after loading weights. This is needed
            # to have test coverage on peak memory for online quantization.
            if current_platform.is_cuda_alike() or current_platform.is_xpu():
                peak_memory = torch.accelerator.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                )

            # Process weights into kernel format. Note that when using online
            # quantization, weights are (typically) quantized as they are loaded.
            if _has_online_quant(model):
                finalize_layerwise_processing(model, model_config)

            process_weights_after_loading(model, model_config, target_device)
            log_online_quantization_time(vllm_config)
            _b70_maybe_embed_uva(model)

            gc.collect()
            torch.accelerator.empty_cache()

        return model.eval()


def _b70_maybe_embed_uva(model: nn.Module) -> None:
    """B70 0044: keep the input embedding table in pinned host RAM (B70_EMBED_UVA=1).

    The lookup reads a few rows per token through a UVA view (a few KB over
    PCIe), so the table's VRAM can hold more hot experts instead. Only the
    target's embed_tokens moves; lm_head (a full GEMM over every row) and the
    MTP draft's own copy (replaced by the target's when shared) stay put."""
    import os

    if os.environ.get("B70_EMBED_UVA", "0") != "1" or not current_platform.is_xpu():
        return
    if "MTP" in type(model).__name__:
        return
    from vllm.model_executor.layers.fused_moe.b70_hotcold import _ARENA
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    for name, module in model.named_modules():
        if (not isinstance(module, VocabParallelEmbedding) or isinstance(module, ParallelLMHead)
                or not name.endswith("embed_tokens")):
            continue
        weight = module.weight
        pinned = _ARENA.empty(weight.shape, weight.dtype)
        pinned.copy_(weight.data)
        weight.data = get_accelerator_view_from_cpu_tensor(pinned)
        weight._vllm_is_uva_offloaded = True
        torch.accelerator.empty_cache()
        logger.info("B70 embed UVA: %s %s %s (%.2f GiB) now in pinned host RAM", name,
                    tuple(weight.shape), weight.dtype, pinned.numel() * pinned.element_size() / 2**30)


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))


def _get_online_quantization_config(vllm_config: VllmConfig):
    """Return the online quantization config when one is configured."""
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    quant_config = vllm_config.quant_config
    online_quantization_config = getattr(
        quant_config, "online_quantization_config", None
    )
    if isinstance(online_quantization_config, OnlineQuantizationConfig):
        quant_config = online_quantization_config
    if not isinstance(quant_config, OnlineQuantizationConfig):
        return None
    return quant_config


def log_online_quantization(vllm_config: VllmConfig) -> None:
    """Log the online-quantized layer count and types, when applicable."""
    quant_config = _get_online_quantization_config(vllm_config)
    if quant_config is None:
        return

    logger.info(
        "Quantizing %d layers of types: %s",
        len(quant_config.quantized_layers),
        "; ".join(quant_config.quantized_layer_summaries),
    )


def log_online_quantization_time(vllm_config: VllmConfig) -> None:
    """Log online quantization processing time, when applicable."""
    quant_config = _get_online_quantization_config(vllm_config)
    if quant_config is None:
        return

    logger.info(
        "Online quantization as part of model loading took %.2f seconds",
        quant_config.online_quantization_time,
    )


def _has_online_quant(model: nn.Module):
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if getattr(quant_method, "uses_meta_device", False):
            return True

    return False

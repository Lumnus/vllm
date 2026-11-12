# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder towers kept in host memory.

A host tower is one encoder tower of the model (vision, audio, ...) held in
host memory and computed on CPU threads, in a flow of its own. The accelerator
then holds neither the tower's weights nor its activations, and a slow encode
occupies CPU threads only. See `MultiModalConfig.mm_encoder_host_towers`.

The pieces, from the outside in:

* `HostTower` is one tower: where it is now (`TowerLocation`), the calls that
  move it (`to_host`, `to_card`, `release`), and `submit`, which encodes items
  on the tower's own thread.
* `TowerWeightsSource` is where a tower's weights come from when it is
  brought to host memory. Today that is the model checkpoint
  (`CheckpointTowerWeights`); a store of named weight units fits the same
  interface.
* `build_host_tower_module` builds the module a host tower computes with: the
  model's own class, constructed with that one tower and no language model, so
  `embed_multimodal` is the code path the workers would have run.
"""

import copy
import enum
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import available_cpu_count, set_default_torch_dtype

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.multimodal import HostTowerOptions
    from vllm.multimodal.inputs import MultiModalKwargsItem

logger = init_logger(__name__)


class TowerLocation(enum.Enum):
    """Where a tower's weights are."""

    CARD = "card"
    """In accelerator memory, computed there by the workers."""

    HOST = "host"
    """In host memory, computed on CPU threads."""

    DISK = "disk"
    """Only in their source; the tower takes no memory and cannot encode
    until it is brought back."""


class TowerWeightsSource(ABC):
    """Where the weights of one tower come from, and go back to.

    The source fills the parameters of a tower module that was built without
    weights. The parameter names are the model's own (`visual.blocks.0...`),
    so a source is keyed by them whatever it reads: a checkpoint, or a store
    that keeps the tower as named units on disk.
    """

    @abstractmethod
    def load_into(self, module: nn.Module) -> None:
        """Fill `module`'s parameters in host memory and leave it ready to
        run. May take long; called off the scheduling path."""
        raise NotImplementedError

    def release(self, module: nn.Module) -> None:
        """Give the host memory of `module`'s parameters back. The weights
        stay recoverable: a later `load_into` on a new module restores them."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot release a tower's host memory."
        )


@dataclass
class TowerEncodeResult:
    """What one `HostTower.submit` produced."""

    embeddings: list[torch.Tensor]
    queue_seconds: float
    """Time the encode waited for the tower's thread."""
    encode_seconds: float
    """Time the tower computed."""


class HostTower:
    """One encoder tower and where it lives.

    The residency contract, for a policy that moves towers while requests are
    in flight:

    * `location` never blocks and may be read from any thread.
    * `to_host()` leaves the tower able to encode in host memory. It is
      idempotent, safe to call concurrently, and never drops a submitted
      encode: encodes submitted while it runs wait for it.
    * `to_card()` must hand the tower to the workers at a step boundary on
      every rank, and only after every encode submitted before it has
      delivered its embeddings; requests admitted on those embeddings keep
      them. Not implemented: `NotImplementedError`, tower unchanged.
    * `release()` must wait for the encodes in flight, then drop the host
      weights; embeddings already produced stay valid, and a later `submit`
      brings the tower back by itself. Not implemented: `NotImplementedError`,
      tower unchanged.

    `submit` is the only way work reaches the tower: one encode at a time, on
    the tower's own thread, with `threads` CPU threads inside it.
    """

    def __init__(
        self,
        name: str,
        *,
        build: Callable[[], nn.Module],
        weights: TowerWeightsSource,
        threads: int,
        output_dtype: torch.dtype,
    ) -> None:
        self.name = name
        self.threads = threads
        self.output_dtype = output_dtype
        self._build = build
        self._weights = weights

        self._module: nn.Module | None = None
        self._modalities: frozenset[str] = frozenset()
        self._location = TowerLocation.DISK
        self._move_lock = threading.Lock()
        self._num_encodes = 0

        # One thread: a second image waits for the first rather than halving
        # its threads.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"host-tower-{name}",
            initializer=torch.set_num_threads,
            initargs=(threads,),
        )

    @property
    def location(self) -> TowerLocation:
        return self._location

    @property
    def modalities(self) -> frozenset[str]:
        """The modalities this tower encodes. Known once it has been built."""
        return self._modalities

    @property
    def num_encodes(self) -> int:
        """Encodes run so far."""
        return self._num_encodes

    @property
    def nbytes(self) -> int:
        """Host memory of the tower's weights, 0 while it is not on the host."""
        module = self._module
        if module is None:
            return 0
        return sum(p.numel() * p.element_size() for p in module.parameters())

    def to_host(self) -> None:
        if self._module is not None:
            return
        with self._move_lock:
            if self._module is not None:
                return
            start = time.perf_counter()
            module = self._build()
            self._weights.load_into(module)
            module.eval()
            self._modalities = frozenset(
                getattr(module, "_tower_modalities", {}).get(self.name, ())
            )
            self._module = module
            self._location = TowerLocation.HOST
            param = next(module.parameters(), None)
            logger.info(
                "Host tower %s: placed on host (%s GiB, %s, %d threads) in %.1f s.",
                self.name,
                format_gib(self.nbytes),
                str(param.dtype).removeprefix("torch.") if param is not None else "-",
                self.threads,
                time.perf_counter() - start,
            )

    def to_card(self) -> None:
        raise NotImplementedError(
            f"Host tower {self.name!r} cannot be moved to the accelerator "
            "while the engine runs."
        )

    def release(self) -> None:
        raise NotImplementedError(
            f"Host tower {self.name!r} cannot give up its host memory while "
            "the engine runs."
        )

    def submit(
        self, items: Sequence[tuple[str, "MultiModalKwargsItem"]]
    ) -> "Future[TowerEncodeResult]":
        """Encode `(modality, item)` pairs on the tower's thread.

        The future resolves to one embedding per item, in order, in host
        memory and in `output_dtype`. Cancelling it before the encode starts
        costs nothing.
        """
        return self._executor.submit(self._timed_encode, list(items), time.monotonic())

    def _timed_encode(
        self,
        items: list[tuple[str, "MultiModalKwargsItem"]],
        submitted: float,
    ) -> "TowerEncodeResult":
        started = time.monotonic()
        embeddings = self.encode(items)
        return TowerEncodeResult(
            embeddings=embeddings,
            queue_seconds=started - submitted,
            encode_seconds=time.monotonic() - started,
        )

    def encode(
        self, items: Sequence[tuple[str, "MultiModalKwargsItem"]]
    ) -> list[torch.Tensor]:
        """Encode on the calling thread. `submit` is the serving entry point."""
        from vllm.multimodal.utils import group_and_batch_mm_kwargs
        from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs

        self.to_host()
        module = self._module
        assert module is not None

        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for _, num_items, mm_kwargs in group_and_batch_mm_kwargs(list(items)):
                embeddings = module.embed_multimodal(**mm_kwargs)  # type: ignore[operator]
                sanity_check_mm_encoder_outputs(
                    embeddings, expected_num_items=num_items
                )
                outputs.extend(
                    embedding.to(self.output_dtype).contiguous()
                    for embedding in embeddings
                )
        self._num_encodes += 1
        return outputs

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def resolve_host_tower_threads(options: "HostTowerOptions") -> int:
    return options.threads or max(1, available_cpu_count() // 2)


def make_host_tower_config(
    vllm_config: "VllmConfig", tower_name: str, options: "HostTowerOptions"
) -> "VllmConfig":
    """The config a host tower's module is built from.

    A private copy of the engine's config that describes the tower alone, in
    host memory, in one piece: no language model, no tensor parallelism, no
    compilation, nothing offloaded. The engine's config is not changed.
    """
    from vllm.config import CompilationConfig, DeviceConfig, ParallelConfig
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.config.offload import OffloadConfig

    config = copy.copy(vllm_config)

    model_config = copy.copy(vllm_config.model_config)
    mm_config = copy.copy(model_config.get_multimodal_config())
    mm_config.mm_encoder_host_twin = tower_name
    # The loader then skips the checkpoint shards that hold only language
    # model weights.
    mm_config.mm_encoder_only = True
    mm_config.mm_encoder_tp_mode = "weights"
    model_config.multimodal_config = mm_config
    if options.dtype == "float32":
        model_config.dtype = torch.float32
    config.model_config = model_config

    config.device_config = DeviceConfig(device="cpu")
    config.parallel_config = ParallelConfig()
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        custom_ops=["all"],
    )
    config.offload_config = OffloadConfig()
    config.ec_transfer_config = None
    config.lora_config = None
    return config


def _use_host_forward(module: nn.Module) -> None:
    """Point the custom ops of `module` at their CPU implementation.

    A custom op picks its kernel for the engine's accelerator when it is
    constructed. This module computes in host memory whatever the engine
    runs on.
    """
    from vllm.model_executor.custom_op import CustomOp

    for submodule in module.modules():
        if isinstance(submodule, CustomOp):
            submodule._forward_method = submodule.forward_cpu


def build_host_tower_module(twin_config: "VllmConfig") -> nn.Module:
    """Build the module of a host tower, without weights.

    The model's own class is constructed from `twin_config` (see
    `make_host_tower_config`): it builds the named tower whole, in host
    memory, and nothing else, and its `embed_multimodal` is what encodes.

    Raises:
        ValueError: The model cannot keep this tower on the host.

    """
    from vllm.distributed.parallel_state import single_rank_model_parallel
    from vllm.model_executor.model_loader.utils import (
        get_model_architecture,
        initialize_model,
    )

    model_config = twin_config.model_config
    tower_name = model_config.get_multimodal_config().mm_encoder_host_twin
    assert tower_name is not None

    model_cls, arch = get_model_architecture(model_config)
    # The opt-in is a property of the class's own constructor, so a subclass
    # with another constructor does not inherit it.
    if not vars(model_cls).get("supports_host_tower", False):
        raise ValueError(
            f"{arch} does not support --mm-encoder-host-towers: its towers "
            "can only run on the accelerator."
        )

    with (
        set_default_torch_dtype(model_config.dtype),
        torch.device("cpu"),
        single_rank_model_parallel(),
    ):
        module = initialize_model(vllm_config=twin_config, model_config=model_config)

    towers: dict[str, frozenset[str]] = getattr(module, "_tower_modalities", {})
    if tower_name not in towers:
        raise ValueError(
            f"{arch} has no tower named {tower_name!r}; "
            f"its towers are: {sorted(towers) or 'none'}."
        )
    if next(module.parameters(), None) is None:
        raise ValueError(
            f"Tower {tower_name!r} of {arch} was not built: every modality it "
            f"serves ({sorted(towers[tower_name])}) is disabled."
        )

    _use_host_forward(module)
    return module


class CheckpointTowerWeights(TowerWeightsSource):
    """A tower's weights read from the model checkpoint, the way a worker
    loads them."""

    def __init__(self, twin_config: "VllmConfig") -> None:
        self._config = twin_config

    def load_into(self, module: nn.Module) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader import get_model_loader
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        model_config = self._config.model_config
        loader = get_model_loader(self._config.load_config)
        with (
            set_current_vllm_config(self._config),
            set_default_torch_dtype(model_config.dtype),
        ):
            loader.load_weights(module, model_config)
            process_weights_after_loading(module, model_config, torch.device("cpu"))


def build_host_towers(vllm_config: "VllmConfig") -> dict[str, HostTower]:
    """Build the towers named in `mm_encoder_host_towers` and bring them to
    host memory."""
    mm_config = vllm_config.model_config.get_multimodal_config()
    model_dtype = vllm_config.model_config.dtype
    assert isinstance(model_dtype, torch.dtype)

    towers: dict[str, HostTower] = {}
    for name, options in mm_config.mm_encoder_host_towers.items():
        twin_config = make_host_tower_config(vllm_config, name, options)
        tower = HostTower(
            name,
            build=lambda twin_config=twin_config: build_host_tower_module(twin_config),  # type: ignore[misc]
            weights=CheckpointTowerWeights(twin_config),
            threads=resolve_host_tower_threads(options),
            output_dtype=model_dtype,
        )
        tower.to_host()
        towers[name] = tower
    return towers

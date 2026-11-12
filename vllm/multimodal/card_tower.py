# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The accelerator copy of an encoder tower with residency "dynamic".

A worker whose model was built with a tower kept in host memory holds a
`StageMissingLayer` where the tower would be. When the engine core decides to
bring the tower to the accelerators (`vllm/v1/engine/tower_residency.py`),
every worker builds its own copy and puts it in that place; when the engine
core sends the tower back, every worker takes it out again. `CardTower` is
that copy on one worker, in four steps, each an RPC from the engine core:

* `stage()` builds the tower module in host memory and starts reading its
  weights on a side thread. The worker keeps serving meanwhile.
* `commit()` loads the weights into the module, moves it to the accelerator
  and puts it in the model. The engine core calls it after the KV cache gave
  up the memory, and routes requests to the workers' tower only after every
  rank's commit succeeded.
* `release()` takes the tower out of the model and drops it.
* `discard()` drops a staged copy that will not be committed.

None of the steps is a collective, so a failure on one rank never leaves
another waiting; each step can be repeated.

Building and loading run on the worker's main thread: they swap process
globals (the current config, and for a whole replica the model-parallel
groups) that a forward on another thread would see. Only the reading of the
checkpoint's bytes runs on the side thread.
"""

import gc
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

TensorReader = Callable[[], dict[str, torch.Tensor]]


def module_bytes(module: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters()) + sum(
        b.numel() * b.element_size() for b in module.buffers()
    )


def tower_paths(model: nn.Module, tower_name: str) -> list[str]:
    """Where `model` holds a placeholder for the tower `tower_name`."""
    from vllm.model_executor.models.utils import StageMissingLayer

    return [
        path
        for path, module in model.named_modules()
        if isinstance(module, StageMissingLayer) and module.stage_name == tower_name
    ]


def _set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, name, module)


class CardTower:
    """One worker's accelerator copy of one tower.

    Args:
        name: The tower's name (`vision_tower`, ...).
        model: The worker's model, with placeholders where the tower goes.
        device: The worker's accelerator.
        build: Builds the tower module in host memory, without weights. It
            holds the tower under the same attribute paths as `model`.
        read: Reads the tower's checkpoint tensors; called on a side thread.
        load: Loads `read`'s tensors into the built module and finishes it
            (post-processing); called on the main thread.
        build_context: Entered around `build` and `load` (the model-parallel
            groups a whole replica is built with).

    """

    def __init__(
        self,
        name: str,
        *,
        model: nn.Module,
        device: torch.device,
        build: Callable[[], nn.Module],
        read: TensorReader,
        load: Callable[[nn.Module, dict[str, torch.Tensor]], None],
        build_context: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> None:
        self.name = name
        self._model = model
        self._device = device
        self._build = build
        self._read = read
        self._load = load
        self._build_context = build_context

        self._module: nn.Module | None = None
        self._reader: threading.Thread | None = None
        self._tensors: dict[str, torch.Tensor] | None = None
        self._read_error: BaseException | None = None
        self._token: object | None = None
        self._staged_at = 0.0
        # Placeholders taken out of the model, by path, while the copy is in.
        self._placeholders: dict[str, nn.Module] = {}

    @property
    def on_card(self) -> bool:
        return bool(self._placeholders)

    @property
    def staged(self) -> bool:
        return self._module is not None and not self.on_card

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def stage(self) -> dict[str, Any]:
        """Build the module in host memory and start reading its weights.
        Returns the bytes the copy will take on the accelerator."""
        if self.on_card:
            assert self._module is not None
            return {"bytes": module_bytes(self._module), "on_card": True}
        if self._module is None:
            paths = tower_paths(self._model, self.name)
            if not paths:
                raise ValueError(
                    f"The model holds no place for tower {self.name!r}: it was "
                    "built with the tower, or without its modalities."
                )
            start = time.perf_counter()
            with self._build_context():
                module = self._build()
            missing = [p for p in paths if _get(module, p) is None]
            if missing:
                raise ValueError(
                    f"The built tower {self.name!r} has nothing at {missing}."
                )
            self._module = module
            self._tensors = None
            self._read_error = None
            self._staged_at = time.perf_counter()
            # A read left over from a discarded staging cannot write into
            # this one.
            token = object()
            self._token = token
            self._reader = threading.Thread(
                target=self._read_tensors,
                args=(token,),
                name=f"card-tower-{self.name}-read",
                daemon=True,
            )
            self._reader.start()
            logger.info(
                "Tower %s: staging a copy for the accelerator (%s GiB, built in "
                "%.2f s); reading its weights.",
                self.name,
                format_gib(module_bytes(module)),
                self._staged_at - start,
            )
        return {"bytes": module_bytes(self._module), "on_card": False}

    def _read_tensors(self, token: object) -> None:
        try:
            tensors = self._read()
        except BaseException as e:  # noqa: BLE001 - reported by poll/commit
            if token is self._token:
                self._read_error = e
            return
        if token is self._token:
            self._tensors = tensors

    def poll(self) -> dict[str, Any]:
        """Whether the staged weights are read (or failed to be)."""
        if self.on_card:
            return {"staged": True, "error": None}
        if self._module is None:
            return {"staged": False, "error": "nothing is staged"}
        error = None if self._read_error is None else repr(self._read_error)
        return {"staged": self._tensors is not None and error is None, "error": error}

    def commit(self, wait_s: float | None = None) -> dict[str, Any]:
        """Load, move to the accelerator and put the copy in the model.
        Waits for the read (at most wait_s). On any failure the model is left
        as it was and the staged copy is dropped."""
        if self.on_card:
            assert self._module is not None
            return {"bytes": module_bytes(self._module), "seconds": 0.0}
        module = self._module
        if module is None:
            raise RuntimeError(f"Tower {self.name!r}: nothing is staged.")
        start = time.perf_counter()
        try:
            if self._reader is not None:
                self._reader.join(wait_s)
                if self._reader.is_alive():
                    raise TimeoutError(
                        f"Tower {self.name!r}: its weights were not read in {wait_s} s."
                    )
            if self._read_error is not None:
                raise RuntimeError(
                    f"Tower {self.name!r}: reading its weights failed: "
                    f"{self._read_error!r}"
                ) from self._read_error
            tensors = self._tensors
            assert tensors is not None
            with self._build_context():
                self._load(module, tensors)
            self._tensors = None
            module.to(self._device)
            paths = tower_paths(self._model, self.name)
            for path in paths:
                placeholder = self._model.get_submodule(path)
                copy = _get(module, path)
                assert copy is not None
                _set_submodule(self._model, path, copy)
                self._placeholders[path] = placeholder
        except BaseException:
            self._put_placeholders_back()
            self.discard()
            raise
        nbytes = module_bytes(module)
        seconds = time.perf_counter() - start
        logger.info(
            "Tower %s: copy on %s in place (%s GiB, committed in %.2f s, "
            "%.2f s after staging began).",
            self.name,
            self._device,
            format_gib(nbytes),
            seconds,
            time.perf_counter() - self._staged_at,
        )
        return {"bytes": nbytes, "seconds": seconds}

    def release(self) -> dict[str, Any]:
        """Take the copy out of the model and drop it. Repeatable."""
        was_on_card = self.on_card
        nbytes = module_bytes(self._module) if self._module is not None else 0
        self._put_placeholders_back()
        self.discard()
        if was_on_card:
            logger.info(
                "Tower %s: copy on %s released (%s GiB).",
                self.name,
                self._device,
                format_gib(nbytes),
            )
        return {"bytes": nbytes if was_on_card else 0}

    def discard(self) -> None:
        """Drop a staged copy (a read in flight finishes and is dropped)."""
        if self.on_card:
            return
        self._module = None
        self._tensors = None
        self._reader = None
        self._read_error = None
        self._token = None
        gc.collect()

    def _put_placeholders_back(self) -> None:
        for path, placeholder in list(self._placeholders.items()):
            _set_submodule(self._model, path, placeholder)
            del self._placeholders[path]


def _get(module: nn.Module, path: str) -> nn.Module | None:
    try:
        return module.get_submodule(path)
    except AttributeError:
        return None


# ----------------------------------------------------------------------
# The real pieces: the model's own class, the checkpoint's safetensors.
# ----------------------------------------------------------------------


def checkpoint_tower_reader(
    twin_config: "VllmConfig", model: nn.Module, paths: list[str]
) -> TensorReader:
    """A reader of exactly the checkpoint tensors that load into the tower at
    `paths` of `model` (after the model's own name mapping), from the
    safetensors shards. Nothing else is read.

    The shard list is resolved here, on the caller's thread; the returned
    callable only reads files."""
    from safetensors import safe_open

    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    model_config = twin_config.model_config
    loader = get_model_loader(twin_config.load_config)
    if not isinstance(loader, DefaultModelLoader):
        raise ValueError(
            f"A tower can only be brought to the accelerator from a checkpoint "
            f"read by the default loader, not {type(loader).__name__}."
        )
    _, files, use_safetensors, _ = loader._prepare_weights(
        model_config.model,
        None,
        model_config.revision,
        fall_back_to_pt=False,
        allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
    )
    if not use_safetensors:
        raise ValueError("A tower is brought to the accelerator from safetensors only.")
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    prefixes = tuple(f"{path}." for path in paths)

    def wanted(name: str) -> bool:
        mapped = mapper.map_name(name) if mapper is not None else name
        return mapped is not None and mapped.startswith(prefixes)

    def read() -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for file in files:
            with safe_open(file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if wanted(name):
                        tensors[name] = f.get_tensor(name)
        if not tensors:
            raise ValueError(f"No checkpoint tensor loads into {list(paths)}.")
        return tensors

    return read


def load_tower_tensors(
    module: nn.Module, tensors: dict[str, torch.Tensor], twin_config: "VllmConfig"
) -> None:
    """Load checkpoint tensors into a built tower module the way the model
    loader would, and refuse a module left with parameters it did not get."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.utils.torch_utils import set_default_torch_dtype

    model_config = twin_config.model_config
    with (
        set_current_vllm_config(twin_config),
        set_default_torch_dtype(model_config.dtype),
    ):
        loaded = module.load_weights(iter(tensors.items()))  # type: ignore[operator]
        if loaded is not None:
            missing = {n for n, _ in module.named_parameters()} - set(loaded)
            if missing:
                raise ValueError(
                    f"{len(missing)} tower parameters got no checkpoint tensor, "
                    f"e.g. {sorted(missing)[:3]}."
                )
        process_weights_after_loading(module, model_config, torch.device("cpu"))
    module.eval()


def make_card_tower(
    vllm_config: "VllmConfig",
    name: str,
    model: nn.Module,
    device: torch.device,
) -> CardTower:
    """The accelerator copy of tower `name` for a worker serving `model`."""
    from vllm.distributed.parallel_state import single_rank_model_parallel
    from vllm.multimodal.host_tower import (
        build_host_tower_module,
        make_host_tower_config,
    )

    mm_config = vllm_config.model_config.get_multimodal_config()
    options = mm_config.mm_encoder_host_towers[name]
    whole = options.card_layout == "replicated"
    twin_config = make_host_tower_config(vllm_config, name, options, for_card=True)
    paths = tower_paths(model, name)
    reader_holder: list[TensorReader] = []

    def build() -> nn.Module:
        module = build_host_tower_module(twin_config, host_forward=False, whole=False)
        reader_holder[:] = [checkpoint_tower_reader(twin_config, module, paths)]
        return module

    def read() -> dict[str, torch.Tensor]:
        return reader_holder[0]()

    return CardTower(
        name,
        model=model,
        device=device,
        build=build,
        read=read,
        load=lambda module, tensors: load_tower_tensors(module, tensors, twin_config),
        # A whole replica is built and loaded with one-rank groups; shards are
        # cut by the worker's own groups.
        build_context=single_rank_model_parallel if whole else nullcontext,
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The accelerator copy of an encoder tower with residency "dynamic".

A worker whose model was built with a tower kept in host memory holds a
`StageMissingLayer` where the tower would be. When the engine core decides to
bring the tower to the accelerators (`vllm/v1/engine/tower_residency.py`),
every worker builds its own copy and puts it in that place; when the engine
core sends the tower back, every worker takes it out again. `CardTower` is
that copy on one worker, in four steps, each an RPC from the engine core:

* `stage()` measures the copy (a build on the meta device: no memory),
  lists the checkpoint tensors it loads, and warms them into the page cache
  on a side thread. It holds no copy of the weights: host memory stays as it
  was, apart from the page cache, which the kernel gives back on demand.
* `commit()` builds the tower on the accelerator, streams its tensors from
  the checkpoint one at a time into it, and puts it in the model. The engine
  core calls it after the KV cache gave up the memory, and routes requests to
  the workers' tower only after every rank's commit succeeded.
* `release()` takes the tower out of the model and drops it.
* `discard()` forgets a staging that will not be committed.

None of the steps is a collective, so a failure on one rank never leaves
another waiting; each can be repeated, and `release` and `discard` never
raise on a half-built copy.

Building and loading run on the worker's main thread: they swap process
globals (the current config, and for a whole replica the model-parallel
groups) that a forward on another thread would see. Only the page-cache warm
up runs on the side thread.
"""

import gc
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

TensorStream = Callable[[], Iterator[tuple[str, torch.Tensor]]]


def module_bytes(module: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters()) + sum(
        b.numel() * b.element_size() for b in module.buffers()
    )


def meta_tensors(module: nn.Module) -> list[str]:
    """Parameters and buffers of `module` without data (on the meta device)."""
    found = [
        f"parameter {name} {tuple(p.shape)}"
        for name, p in module.named_parameters()
        if p.is_meta
    ]
    found += [
        f"buffer {name} {tuple(b.shape)}"
        for name, b in module.named_buffers()
        if b.is_meta
    ]
    return found


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


def _get(module: nn.Module, path: str) -> nn.Module | None:
    try:
        return module.get_submodule(path)
    except AttributeError:
        return None


class CardTower:
    """One worker's accelerator copy of one tower.

    Args:
        name: The tower's name (`vision_tower`, ...).
        model: The worker's model, with placeholders where the tower goes.
        device: The worker's accelerator.
        build: `build(device)` builds the tower module on `device` (the
            meta device to measure it), without weights. It holds the tower
            under the same attribute paths as `model`.
        tensors: Lists the checkpoint tensors of the tower and how many bytes
            each takes in host memory while it is loaded; called once, on the
            main thread, with the module built on the meta device.
        stream: `stream()` yields the tower's checkpoint tensors one at a time.
        load: `load(module, tensor_iterator)` loads them into the module and
            finishes it (post-processing); called on the main thread.
        build_context: Entered around `build` and `load` (the model-parallel
            groups a whole replica is built with).

    """

    def __init__(
        self,
        name: str,
        *,
        model: nn.Module,
        device: torch.device,
        build: Callable[[torch.device], nn.Module],
        tensors: Callable[[nn.Module], dict[str, int]],
        stream: Callable[[], Iterator[tuple[str, torch.Tensor]]],
        load: Callable[[nn.Module, Iterator[tuple[str, torch.Tensor]]], None],
        build_context: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> None:
        self.name = name
        self._model = model
        self._device = device
        self._build = build
        self._list_tensors = tensors
        self._stream = stream
        self._load = load
        self._build_context = build_context

        # The staging: what the copy needs, and the page-cache warm up.
        self._staged: dict[str, int] | None = None
        self._warmer: threading.Thread | None = None
        self._warm_error: BaseException | None = None
        self._warm_done = False
        self._token: object | None = None
        self._staged_at = 0.0

        # The copy in the model, and the placeholders it took the place of.
        self._module: nn.Module | None = None
        self._placeholders: dict[str, nn.Module] = {}

    @property
    def on_card(self) -> bool:
        return bool(self._placeholders)

    @property
    def staged(self) -> bool:
        return self._staged is not None and not self.on_card

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def stage(self) -> dict[str, Any]:
        """Measure the copy and start warming its tensors. Returns the bytes
        the copy takes on the accelerator, and the most host memory the
        commit holds at once (one tensor)."""
        if self.on_card:
            assert self._module is not None
            nbytes = module_bytes(self._module)
            return {"bytes": nbytes, "host_bytes": 0, "on_card": True}
        if self._staged is None:
            if not tower_paths(self._model, self.name):
                raise ValueError(
                    f"The model holds no place for tower {self.name!r}: it was "
                    "built with the tower, or without its modalities."
                )
            start = time.perf_counter()
            with self._build_context():
                shape = self._build(torch.device("meta"))
            tensors = self._list_tensors(shape)
            self._staged = {
                "bytes": module_bytes(shape),
                "host_bytes": max(tensors.values(), default=0),
                "num_tensors": len(tensors),
            }
            del shape
            self._warm_error = None
            self._warm_done = False
            self._staged_at = time.perf_counter()
            token = object()
            self._token = token
            self._warmer = threading.Thread(
                target=self._warm,
                args=(token,),
                name=f"card-tower-{self.name}-warm",
                daemon=True,
            )
            self._warmer.start()
            logger.info(
                "Tower %s: staging a copy for the accelerator (%s GiB in %d "
                "checkpoint tensors, measured in %.2f s); warming its tensors.",
                self.name,
                format_gib(self._staged["bytes"]),
                len(tensors),
                self._staged_at - start,
            )
        return {
            "bytes": self._staged["bytes"],
            "host_bytes": self._staged["host_bytes"],
            "on_card": False,
        }

    def _warm(self, token: object) -> None:
        try:
            for _, tensor in self._stream():
                del tensor
                if token is not self._token:
                    return
        except BaseException as e:  # noqa: BLE001 - reported by poll/commit
            if token is self._token:
                self._warm_error = e
            return
        if token is self._token:
            self._warm_done = True

    def poll(self) -> dict[str, Any]:
        """Whether the staged tensors are warm (or failed to be read)."""
        if self.on_card:
            return {"staged": True, "error": None}
        if self._staged is None:
            return {"staged": False, "error": "nothing is staged"}
        error = None if self._warm_error is None else repr(self._warm_error)
        return {"staged": self._warm_done and error is None, "error": error}

    def commit(self, wait_s: float | None = None) -> dict[str, Any]:
        """Build the copy on the accelerator, stream its weights into it and
        put it in the model. Waits for the warm up (at most wait_s; a cold
        read is slower, not wrong). On any failure the model is left as it was
        and the copy is dropped."""
        if self.on_card:
            assert self._module is not None
            return {"bytes": module_bytes(self._module), "seconds": 0.0}
        if self._staged is None:
            raise RuntimeError(f"Tower {self.name!r}: nothing is staged.")
        if self._warm_error is not None:
            error = self._warm_error
            self.discard()
            raise RuntimeError(
                f"Tower {self.name!r}: reading its weights failed: {error!r}"
            ) from error
        if self._warmer is not None:
            self._warmer.join(wait_s)
        start = time.perf_counter()
        try:
            with self._build_context():
                module = self._build(self._device)
                self._module = module
                self._load(module, self._stream())
            paths = tower_paths(self._model, self.name)
            copies = {path: _get(module, path) for path in paths}
            missing = [path for path, copy in copies.items() if copy is None]
            if missing:
                raise ValueError(f"The built tower has nothing at {missing}.")
            empty = [
                f"{path}: {what}"
                for path, copy in copies.items()
                for what in meta_tensors(assert_module(copy))
            ]
            if empty:
                raise ValueError(
                    f"Tower {self.name!r}: {len(empty)} tensors of the copy hold "
                    f"no data, e.g. {empty[:4]}."
                )
            for path, copy in copies.items():
                placeholder = self._model.get_submodule(path)
                _set_submodule(self._model, path, assert_module(copy))
                self._placeholders[path] = placeholder
        except BaseException:
            self._put_placeholders_back()
            self._module = None
            self.discard()
            raise
        self._staged = None
        self._token = None  # a warm up still running stops
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
        """Take the copy out of the model and drop it. Repeatable; never
        raises on a half-built copy."""
        was_on_card = self.on_card
        nbytes = 0
        if was_on_card and self._module is not None:
            nbytes = module_bytes(self._module)
        self._put_placeholders_back()
        self._module = None
        self.discard()
        if was_on_card:
            logger.info(
                "Tower %s: copy on %s released (%s GiB).",
                self.name,
                self._device,
                format_gib(nbytes),
            )
        return {"bytes": nbytes}

    def discard(self) -> None:
        """Forget a staging (a warm up in flight stops at its next tensor).
        Never raises."""
        if self.on_card:
            return
        self._staged = None
        self._warmer = None
        self._warm_error = None
        self._warm_done = False
        self._token = None
        self._module = None
        gc.collect()

    def _put_placeholders_back(self) -> None:
        for path, placeholder in list(self._placeholders.items()):
            _set_submodule(self._model, path, placeholder)
            del self._placeholders[path]


def assert_module(module: nn.Module | None) -> nn.Module:
    assert module is not None
    return module


# ----------------------------------------------------------------------
# The real pieces: the model's own class, the checkpoint's safetensors.
# ----------------------------------------------------------------------


class CheckpointTowerTensors:
    """The checkpoint tensors that load into the tower at `paths` (after the
    model's own name mapping), read from the safetensors shards one at a
    time. Nothing else is read.

    `index(module)` resolves the shards and the tensor names on the main
    thread; `stream()` only reads files and may run on any thread."""

    def __init__(self, twin_config: "VllmConfig", paths: list[str]) -> None:
        self._config = twin_config
        self._prefixes = tuple(f"{path}." for path in paths)
        self._paths = list(paths)
        # file -> names, in file order.
        self._files: dict[str, list[str]] = {}

    def index(self, module: nn.Module) -> dict[str, int]:
        from safetensors import safe_open

        from vllm.model_executor.model_loader import get_model_loader
        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )

        model_config = self._config.model_config
        loader = get_model_loader(self._config.load_config)
        if not isinstance(loader, DefaultModelLoader):
            raise ValueError(
                "A tower can only be brought to the accelerator from a "
                f"checkpoint read by the default loader, not "
                f"{type(loader).__name__}."
            )
        _, files, use_safetensors, _ = loader._prepare_weights(
            model_config.model,
            None,
            model_config.revision,
            fall_back_to_pt=False,
            allow_patterns_overrides=getattr(module, "allow_patterns_overrides", None),
        )
        if not use_safetensors:
            raise ValueError(
                "A tower is brought to the accelerator from safetensors only."
            )
        mapper = getattr(module, "hf_to_vllm_mapper", None)

        sizes: dict[str, int] = {}
        self._files = {}
        for file in files:
            with safe_open(file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    mapped = mapper.map_name(name) if mapper is not None else name
                    if mapped is None or not mapped.startswith(self._prefixes):
                        continue
                    piece = f.get_slice(name)
                    numel = 1
                    for dim in piece.get_shape():
                        numel *= dim
                    sizes[name] = numel * _dtype_bytes(piece.get_dtype())
                    self._files.setdefault(file, []).append(name)
        if not sizes:
            raise ValueError(f"No checkpoint tensor loads into {self._paths}.")
        return sizes

    def stream(self) -> Iterator[tuple[str, torch.Tensor]]:
        from safetensors import safe_open

        for file, names in self._files.items():
            with safe_open(file, framework="pt") as f:
                for name in names:
                    yield name, f.get_tensor(name)


_SAFETENSORS_DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _dtype_bytes(dtype: str) -> int:
    return _SAFETENSORS_DTYPE_BYTES.get(dtype, 4)


def load_tower_tensors(
    module: nn.Module,
    tensors: Iterator[tuple[str, torch.Tensor]],
    twin_config: "VllmConfig",
    device: torch.device,
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
        loaded = module.load_weights(tensors)  # type: ignore[operator]
        if loaded is not None:
            missing = {n for n, _ in module.named_parameters()} - set(loaded)
            if missing:
                raise ValueError(
                    f"{len(missing)} tower parameters got no checkpoint tensor, "
                    f"e.g. {sorted(missing)[:3]}."
                )
        process_weights_after_loading(module, model_config, device)
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
    tensors = CheckpointTowerTensors(twin_config, tower_paths(model, name))

    return CardTower(
        name,
        model=model,
        device=device,
        # The build context supplies the one-rank groups of a whole replica.
        build=lambda where: build_host_tower_module(
            twin_config, host_forward=False, whole=False, device=where
        ),
        tensors=tensors.index,
        stream=tensors.stream,
        load=lambda module, it: load_tower_tensors(module, it, twin_config, device),
        # A whole replica is built and loaded with one-rank groups; shards are
        # cut by the worker's own groups.
        build_context=single_rank_model_parallel if whole else nullcontext,
    )

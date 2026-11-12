# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stand-ins for host tower tests: a tower that computes something small and
can be held back, and a hand-over region that lives in process memory."""

import threading
import zlib

import torch
from torch import nn

from vllm.multimodal.host_tower import HostTower, TowerWeightsSource
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
)

TOWER_NAME = "vision_tower"
IN_DIM = 8
OUT_DIM = 32


class FakeRegion:
    """The shared hand-over region, without the shared memory file."""

    def __init__(self, num_blocks: int, block_size_bytes: int) -> None:
        self.num_blocks = num_blocks
        self.block_size_bytes = block_size_bytes
        self.blocks = torch.zeros(num_blocks, block_size_bytes, dtype=torch.int8)
        self.cleaned_up = False

    def cleanup(self) -> None:
        self.cleaned_up = True


class ToyTowerModule(nn.Module):
    """A tower: one projection per item, as slow as its gate makes it."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.proj = nn.Linear(IN_DIM, OUT_DIM)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(OUT_DIM, IN_DIM, generator=generator))
            self.proj.bias.copy_(torch.randn(OUT_DIM, generator=generator))
        self._tower_modalities = {TOWER_NAME: frozenset({"image"})}
        self.gate = gate
        self.started = threading.Event()
        self.fail = False
        # The torch thread count each encode ran with.
        self.threads_seen: list[int] = []

    def embed_multimodal(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.started.set()
        self.threads_seen.append(torch.get_num_threads())
        if self.gate is not None:
            assert self.gate.wait(timeout=60), "the test never released the tower"
        if self.fail:
            raise RuntimeError("tower failure")
        return tuple(self.proj(item) for item in pixel_values)


class NoWeights(TowerWeightsSource):
    """The toy tower comes with its weights."""

    def __init__(self) -> None:
        self.loads = 0

    def load_into(self, module: nn.Module) -> None:
        self.loads += 1


def make_tower(
    gate: threading.Event | None = None,
    *,
    threads: int = 2,
    output_dtype: torch.dtype = torch.float16,
) -> HostTower:
    return HostTower(
        TOWER_NAME,
        build=lambda: ToyTowerModule(gate),
        weights=NoWeights(),
        threads=threads,
        output_dtype=output_dtype,
    )


def make_item(identifier: str, rows: int) -> MultiModalKwargsItem:
    """The processed input of one image; the same identifier gives the same
    pixels."""
    generator = torch.Generator().manual_seed(zlib.crc32(identifier.encode()))
    pixels = torch.randn(rows, IN_DIM, generator=generator)
    return MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(
                data=pixels, field=MultiModalBatchedField()
            )
        }
    )


def direct_embedding(
    tower: HostTower, item: MultiModalKwargsItem, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """What the tower's module computes for `item` when called directly."""
    module = tower._module
    assert isinstance(module, ToyTowerModule)
    with torch.inference_mode():
        return module.proj(item["pixel_values"].data).to(dtype)

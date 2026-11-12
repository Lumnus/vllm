# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration of the expert store behind per-expert CPU offload tiers."""

from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from vllm.config.utils import CompileFactors, config, get_compile_factors

if TYPE_CHECKING:
    from vllm.config.offload import OffloadConfig
    from vllm.config.parallel import ParallelConfig


@config
class ExpertStoreConfig:
    """Every routed expert on NVMe, behind a pinned host pool.

    With ``dirs`` set and ``--cpu-offload-per-expert`` on, the experts of the
    offloaded layers have a durable copy in kernel layout on the drives. The
    host tier of a layer is then a pool of rows filled from that copy, EPLB's
    rearrangements read the experts a rank receives from the drives instead of
    sending them between ranks, and tier changes need no card-to-host copy.
    The first boot builds the store from the checkpoint; later boots fill the
    host tier from it.
    """

    dirs: list[str] = Field(default_factory=list)
    """One directory per drive, 1 to N. Non-empty turns the store on. The same
    list in the same launch for every rank."""

    pool_gib: float = Field(default=4.0, gt=0)
    """GiB of pinned host memory, over all ranks, for the store's staging slots
    (units read ahead of use, and the copy source of reads that cannot land in
    a row directly)."""

    io_threads: int = Field(default=16, ge=1)
    """Reader threads per rank."""

    direct: bool = True
    """Read with O_DIRECT where the filesystem takes it."""

    read_mode: Literal["auto", "direct", "buffered", "bounce"] = "auto"
    """How a unit is read, into a host row and into a staging slot alike:
    ``direct`` is one O_DIRECT read straight into the memory (an iovec per
    tensor for a row); ``buffered`` the same read through the page cache;
    ``bounce`` a direct read into an ordinary page-aligned buffer and a host
    copy. ``auto`` starts direct and moves to buffered, then to bounce, when
    the kernel refuses the memory (as it does for direct reads into some
    accelerators' pinned memory); rows and staging slots move separately."""

    decay_interval: int = Field(default=256, ge=1)
    """Requests per half-life of the staging pool's use counts."""

    build: Literal["auto", "never", "rebuild"] = "auto"
    """``auto``: build the store at load when it is absent or was built for
    another checkpoint, layout or schema; ``never``: refuse to load without a
    matching store; ``rebuild``: build it anew at every load."""

    verify: bool = False
    """Hash every stripe against its checksum at open (a full read)."""

    spare_rows: int | None = Field(default=None, ge=0)
    """Host rows per layer beyond the host tier's slots, for experts being
    read while the old ones still serve. None: EPLB's ``max_moves_per_rank``,
    the most slots one rearrangement changes on a rank."""

    stats_interval_s: float = Field(default=0.0, ge=0)
    """Seconds between store log lines per rank; 0 for none."""

    @model_validator(mode="after")
    def _check_dirs(self) -> "ExpertStoreConfig":
        if len(set(self.dirs)) != len(self.dirs):
            raise ValueError("expert store dirs must be distinct")
        return self

    @property
    def enabled(self) -> bool:
        return bool(self.dirs)

    def verify_parallel_config(
        self, parallel_config: "ParallelConfig", offload_config: "OffloadConfig"
    ) -> None:
        """Reject settings the store cannot serve."""
        if not self.enabled:
            return
        if not offload_config.uva.cpu_offload_per_expert:
            raise ValueError(
                "--expert-store-config backs the expert tiers of "
                "--cpu-offload-per-expert: add it (with --cpu-offload-gb)."
            )
        if parallel_config.enable_elastic_ep:
            raise ValueError("the expert store is not supported with elastic EP")
        eplb = parallel_config.eplb_config
        if eplb.policy != "incremental":
            raise ValueError(
                "the expert store moves a bounded number of experts per "
                "rearrangement: set --eplb-config policy to 'incremental'"
            )
        if not eplb.use_async:
            raise ValueError(
                "the expert store reads experts while serving: set --eplb-config "
                "use_async to true"
            )
        if self.spare_rows is not None and self.spare_rows < eplb.max_moves_per_rank:
            raise ValueError(
                f"spare_rows={self.spare_rows} is below max_moves_per_rank="
                f"{eplb.max_moves_per_rank}: one rearrangement could not land"
            )

    def spare_rows_for(self, max_moves_per_rank: int) -> int:
        return max_moves_per_rank if self.spare_rows is None else self.spare_rows

    def compile_factors(self) -> CompileFactors:
        """Only what changes the shapes the kernels are launched with."""
        return get_compile_factors(
            self,
            {
                "dirs",
                "pool_gib",
                "io_threads",
                "direct",
                "read_mode",
                "decay_interval",
                "build",
                "verify",
                "stats_interval_s",
            },
        )

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

    pool_pinned: bool = False
    """Pin the staging slots. Off by default: direct reads then land in
    ordinary page-aligned memory, which every drive and accelerator takes,
    and a row is filled from a slot by a host copy. On: a slot can be the
    source of an asynchronous copy to the device, and reads into it follow
    ``read_mode`` like reads into rows."""

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

    check_every: int = Field(default=8, ge=0)
    """Check units against their recorded digests as they are read: 1 every
    read, N every Nth read (the default samples one read in eight), 0 never.
    A unit that fails is never mapped: the read raises and the move that
    asked for it is abandoned."""

    check_threads: int = Field(default=2, ge=0)
    """Hashing threads of one unit check (0: as many as the library likes)."""

    read_timeout_s: float = Field(default=30.0, gt=0)
    """Seconds a move waits for the units of one layer. A read that fails or
    does not return in time does not stop serving: the rearrangement is
    abandoned from that layer on, on every rank, with an error line on the
    rank that saw it."""

    spare_rows: int | None = Field(default=None, ge=0)
    """Host rows per layer beyond the host tier's slots, for experts being
    read while the old ones still serve. None: EPLB's ``max_moves_per_rank``,
    the most slots one rearrangement changes on a rank."""

    stats_interval_s: float = Field(default=0.0, ge=0)
    """Seconds between store log lines per rank (and, with
    ``--cpu-offload-pinned-gb``, CPU expert lines: routed calls per tier,
    host cache hits and drive reads, time per layer); 0 for none."""

    scrub_every: int = Field(default=0, ge=0)
    """Debugging aid, off at 0. N: once after load (every slot) and after
    every Nth rearrangement (the slots moved since the last scrub), each rank
    reads the rows its kernels read back from the device and through the
    device views and compares them with the store, and the tier maps on the
    device with the placement; one log line with the counts, an error line
    naming the first mismatch. The step that scrubs waits for it."""

    host_cache_gib: float = Field(default=4.0, gt=0)
    """With ``--cpu-offload-pinned-gb``: GiB of ordinary (pageable) host
    memory per rank for the experts computed on the CPU, a cache in front of
    the drive. An expert a step routes to that is not in it is read from the
    drive while the step waits (one unit, about a millisecond on NVMe)."""

    cpu_threads: int = Field(default=8, ge=1, le=1024)
    """With ``--cpu-offload-pinned-gb``: host threads per rank that compute
    the experts on the CPU."""

    cpu_isa: Literal["auto", "avx2", "scalar"] = "auto"
    """The CPU experts' code: ``auto`` takes AVX2 with FMA and refuses to
    start without it; ``avx2`` the same; ``scalar`` the portable code (the
    same results, hundreds of times slower; for tests)."""

    host_promote_misses: int = Field(default=0, ge=0)
    """With ``--cpu-offload-pinned-gb``: promote a host slot to a pinned row
    between two steps once the CPU has served it in this many layer calls
    (counts halve every ``host_promote_window`` steps); 0 leaves the pinned
    set to EPLB's rearrangements alone."""

    host_promote_window: int = Field(default=64, ge=1)
    """Steps per halving of the counts that ``host_promote_misses`` reads."""

    host_promote_rows: int = Field(default=4, ge=0)
    """Pinned rows per layer beyond the spare rows, for those promotions;
    EPLB's next rearrangement chooses the pinned set again and frees them."""

    tier_seed: str | None = None
    """Loads to place the experts by at start when EPLB has none of its own
    (no ``state_path``, or no file there yet): the hottest slots of each rank
    go to the card, the next ones to the pinned rows, the rest to the CPU,
    and EPLB's policy balances the ranks by them. The file is an EPLB state
    file (layer name -> load per logical expert; also what
    ``tools/analyze_expert_trace.py --write-loads`` writes from a routing
    trace), a rankings file (decoder layer number -> logical experts hottest
    first) or a learned-state file (``{"layers": {layer: {expert:
    score}}}``)."""

    working_set: bool = False
    """With ``--cpu-offload-pinned-gb``: choose which slots of a layer are on
    the card, in pinned rows and on the CPU from each slot's temperature
    (decayed call rate) while serving, between two steps, in budgeted
    batches read in the background; EPLB's rearrangements then move experts
    between ranks only. Runtime setting (``tiers.set working_set``)."""

    ws_halflife_steps: float = Field(default=64.0, gt=0)
    """Steps per half-life of a slot's temperature."""

    ws_hot: float = Field(default=1.5, gt=0)
    """Temperature (calls per token over the mean expert's) at or above which
    a slot is hot."""

    ws_warm: float = Field(default=0.5, gt=0)
    """Temperature at or above which a slot is warm; below it is cold. Only
    warm and hot slots are moved into a faster tier."""

    ws_hysteresis: float = Field(default=1.25, ge=1.0)
    """A slot keeps its class until its temperature is below the class's
    threshold divided by this; a swap within a class needs this ratio."""

    ws_batch: int = Field(default=4, ge=1)
    """Experts read per layer batch (each swap reads one)."""

    ws_units_per_step: int = Field(default=16, ge=1)
    """Experts read per step over all layers of a rank."""

    ws_cpu_call_us: float = Field(default=110.0, gt=0)
    """Cost model: one expert computed on the CPU at a few tokens."""

    ws_pinned_call_us: float = Field(default=28.0, ge=0)
    """Cost model: one expert read by the card from a pinned row, over a card
    row."""

    ws_move_us: float = Field(default=250.0, ge=0)
    """Cost model: one expert copied into a row (a swap pays when its gain
    over one half-life is at least this)."""

    stage_min_calls: int = Field(default=0, ge=0)
    """With ``--cpu-offload-pinned-gb``: an expert of the CPU tier that one
    layer call routes this many tokens to is copied into a staging row of
    pinned memory and computed by the card (as the pinned rows are) instead
    of on the CPU; 0 keeps every CPU-tier expert on the CPU. Runtime
    setting."""

    stage_rows: int = Field(default=0, ge=0)
    """Staging rows (pinned, shared by the layers of a rank); 0: as many as
    a layer has CPU slots."""

    @model_validator(mode="after")
    def _check_dirs(self) -> "ExpertStoreConfig":
        if len(set(self.dirs)) != len(self.dirs):
            raise ValueError("expert store dirs must be distinct")
        if self.ws_warm > self.ws_hot:
            raise ValueError("ws_warm must not be above ws_hot")
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
        if getattr(parallel_config, "pipeline_parallel_size", 1) > 1:
            # Each stage is its own group of ranks with half of the layers:
            # the build's writers and its finalize are agreed within one.
            raise ValueError(
                "the expert store is not supported with pipeline parallelism "
                "(--pipeline-parallel-size > 1)"
            )
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
                "pool_pinned",
                "io_threads",
                "direct",
                "read_mode",
                "decay_interval",
                "build",
                "verify",
                "scrub_every",
                "check_every",
                "check_threads",
                "read_timeout_s",
                "stats_interval_s",
                # the CPU experts act inside the opaque moe_forward op
                "host_cache_gib",
                "cpu_threads",
                "cpu_isa",
                "host_promote_misses",
                "host_promote_window",
                "host_promote_rows",
                # the working set and the seed only choose which slot is in
                # which tier; staging adds a call outside the captured graphs
                "tier_seed",
                "working_set",
                "ws_halflife_steps",
                "ws_hot",
                "ws_warm",
                "ws_hysteresis",
                "ws_batch",
                "ws_units_per_step",
                "ws_cpu_call_us",
                "ws_pinned_call_us",
                "ws_move_us",
                "stage_min_calls",
                "stage_rows",
            },
        )

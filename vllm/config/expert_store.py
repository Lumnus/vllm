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

    cpu_isa: Literal[
        "auto",
        "avx2",
        "avx-vnni",
        "avx512",
        "avx512-vnni",
        "avx512-bf16",
        "amx",
        "amx-card-class",
        "scalar",
    ] = "auto"
    """The CPU experts' code: ``auto`` takes the widest the host has (AVX2
    with FMA, AVX-VNNI, AVX-512, AVX-512 VNNI or BF16, AMX) and refuses to
    start without AVX2 and FMA; a named one is that code or a refusal when
    the host lacks it. All give the scalar code's results bit for bit except
    ``amx-card-class`` (AMX-BF16 for prefill calls in the grouped GEMM's
    arithmetic class, never automatic); ``scalar`` is the portable code
    (hundreds of times slower; for tests)."""

    cpu_arith: Literal["card", "gemv"] = "card"
    """The arithmetic of the CPU experts' dot products: ``card`` (default)
    takes, per GEMM and layer call, the card kernel's that the card would run
    for the same call (its grouped GEMM, weights rounded to bf16(q * s), for
    every call of a model with top-k above 8; its int4 GEMV for a call of at
    most 8 rows), so an expert gives the card's values wherever it is
    computed, to the order of fp32 sums; ``gemv`` takes the GEMV's (scale per
    32-element chunk sum) for every call: up to a third faster at one token
    per expert, no farther from the unrounded weights, but most of an
    expert's values then differ from the card's in the last bit, and a tier
    move changes outputs. Needs kernels whose host op has ``arith``; with an
    older op the GEMV's arithmetic is all there is."""

    host_promote_misses: int = Field(default=0, ge=0)
    """Removed (0 only): a CPU slot enters a pinned or a card row in EPLB's
    tier pass (``--eplb-config tier_interval``), planned from its load window,
    not by a count of its own."""

    host_promote_window: int = Field(default=64, ge=1)
    """Removed with ``host_promote_misses``; accepted and unused."""

    host_promote_rows: int = Field(default=4, ge=0)
    """Pinned rows per layer beyond the bound at load and the spare rows,
    which a larger pinned bound (``tiers.set host_tier_bound``) may take
    while serving."""

    stage_min_calls: int = Field(default=0, ge=0)
    """With ``--cpu-offload-pinned-gb``: an expert of the CPU tier that one
    layer call routes at least this many tokens to is copied into a staging
    row of pinned memory and computed by the card from it (as the pinned rows
    are) instead of on the CPU; 0 (default) keeps every CPU-tier expert on
    the CPU. A read path for one layer call: it moves no expert and changes
    no tier. Runtime setting (``cpu.set stage_min_calls``)."""

    stage_rows: int = Field(default=0, ge=0)
    """Staging rows (pinned, shared by the layers of a rank); 0: as many as
    a layer has CPU slots."""

    stage_min_tokens: int = Field(default=0, ge=0)
    """With ``--cpu-offload-pinned-gb``: a layer call of at least this many
    tokens chooses the CPU-tier experts it stages by a cost model instead of
    ``stage_min_calls``: the busiest ones, as many as make the layer's
    host-side time (copies into the staging rows, then the CPU's experts
    while the card reads the staged ones) smallest, with the CPU's cost per
    expert and per call fitted to its own measured calls. 0 (default): off.
    Runtime setting (``cpu.set stage_min_tokens``)."""

    stage_copy_threads: int = Field(default=1, ge=1, le=256)
    """Threads that copy units from the host cache into the staging rows
    (1, the default: one, on the forward's thread). Runtime setting."""

    stage_card_us: float = Field(default=100.0, gt=0)
    """The card's time per staged expert the staging cost model assumes
    (µs; about what reading one unit through UVA costs a card beside its
    resident experts). Runtime setting."""

    fast_rows_max: int = Field(default=0, ge=0)
    """0 (default): the card tier of each layer is fixed at load, as many rows
    as ``--cpu-offload-gb`` leaves on the card. N: the card tier can grow and
    shrink while serving (runtime control ``tiers.fast``), up to N rows per
    layer per rank (at most every local slot): its rows sit in a device
    address range reserved for N rows, physical memory is mapped for the
    rows in use only, and the kernels and the captured graphs see the N-row
    range, so a resize recaptures nothing. A grow takes its device memory
    from the KV cache (a shrink of the pool that drains), a shrink gives it
    back."""

    fast_rows_min: int | None = Field(default=None, ge=1)
    """With ``fast_rows_max``: the fewest card rows per layer a shrink may
    leave. None: one with experts computed on the CPU
    (``--cpu-offload-pinned-gb``), where a slot leaving the card can go to
    the CPU; otherwise the rows at load, since every slot off the card then
    needs a pinned row: a lower floor allocates that many more pinned rows per
    layer at load."""

    fast_chunk_mib: int = Field(default=2, ge=1)
    """With ``fast_rows_max``: MiB per physical object of the card tier's
    memory (rounded to the device's granularity). A shrink gives back the
    objects wholly above the rows that stay; smaller objects free memory row
    by row, larger ones need fewer driver allocations."""

    pinned_release: bool = False
    """With ``--cpu-offload-pinned-gb``: lowering the pinned bound at run
    time (``tiers.set host_tier_bound``) also gives the pinned host memory of
    the rows no longer needed back to the system, layer by layer (each
    layer's pinned rows are laid out again, smaller, and a chunk of pinned
    memory is freed once no layer holds rows in it), and raising it lays the
    rows out larger, up to every slot off the card. Off: the rows stay
    allocated and the bound stays within the rows sized at load."""

    @model_validator(mode="after")
    def _check_dirs(self) -> "ExpertStoreConfig":
        if len(set(self.dirs)) != len(self.dirs):
            raise ValueError("expert store dirs must be distinct")
        if self.host_promote_misses:
            raise ValueError(
                "host_promote_misses is removed: CPU slots enter pinned and "
                "card rows in EPLB's tier pass (--eplb-config tier_interval)"
            )
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
        if (
            self.fast_rows_min is not None
            and self.fast_rows_max
            and self.fast_rows_min > self.fast_rows_max
        ):
            raise ValueError(
                f"fast_rows_min={self.fast_rows_min} is above fast_rows_max="
                f"{self.fast_rows_max}"
            )

    def spare_rows_for(self, max_moves_per_rank: int) -> int:
        return max_moves_per_rank if self.spare_rows is None else self.spare_rows

    @property
    def fast_tier_resizable(self) -> bool:
        return self.enabled and self.fast_rows_max > 0

    def compile_factors(self) -> CompileFactors:
        """Only what changes the shapes the kernels are launched with: of the
        card tier's settings, its capacity, and only when it is set (the
        hash of an engine without it is the one before the setting
        existed)."""
        factors = self._base_compile_factors()
        if self.fast_rows_max:
            factors["fast_rows_max"] = self.fast_rows_max
        return factors

    def _base_compile_factors(self) -> CompileFactors:
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
                "cpu_arith",
                "host_promote_misses",
                "host_promote_window",
                "host_promote_rows",
                # staging adds a call outside the captured graphs
                "stage_min_calls",
                "stage_rows",
                "stage_min_tokens",
                "stage_copy_threads",
                "stage_card_us",
                # the card tier's run-time settings (fast_rows_max is added
                # by compile_factors when set)
                "fast_rows_max",
                "fast_rows_min",
                "fast_chunk_mib",
                "pinned_release",
            },
        )

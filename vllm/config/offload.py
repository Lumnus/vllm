# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from vllm.config.utils import CompileFactors, config, get_compile_factors

OffloadBackend = Literal["auto", "uva", "prefetch"]


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
    """

    cpu_offload_per_expert: bool = False
    """Spend `cpu_offload_gb` on routed expert slots instead of whole
    parameters (XPU, int4 GPTQ/AutoRound experts; needs --enable-eplb). Every
    MoE decoder layer keeps the same number of its local expert slots in
    pinned host memory, read by the kernels through UVA, and the rest in
    device memory. EPLB places the experts: its load statistics decide which
    experts take the device slots, its rearrangement moves them, and with
    `--eplb-config` `state_path` the placement survives restarts. Works on a
    single rank too. The budget goes to routed experts only:
    `cpu_offload_params` is ignored. Refused when UVA is disabled
    (`VLLM_WEIGHT_OFFLOADING_DISABLE_UVA`)."""
    cpu_offload_pinned_gb: float | None = Field(default=None, gt=0)
    """With `cpu_offload_per_expert` and an expert store
    (`--expert-store-config`): GiB per rank of pinned host memory for expert
    slots, at most `cpu_offload_gb`. The slots off the device that it pays for
    hold pinned rows the kernels read through UVA; the others are in no
    memory the device reads: their experts stay in the expert store (a cache
    of `host_cache_gib` of ordinary memory in front of the drive) and are
    computed on the CPU when a step routes to them, with the same arithmetic
    as the device's, and added to the layer's output. The MoE layers then run
    outside the captured CUDA/XPU graphs (the graphs are split at the MoE op;
    full graphs are turned off). Per layer, the most-loaded experts of EPLB's
    window take the device slots, the next ones the pinned rows. None (the
    default): every slot off the device is pinned."""
    cpu_offload_stage_min_tokens: int = Field(default=0, ge=-1)
    """Steps with at least this many tokens copy each layer's offloaded
    fused-MoE expert weights into a device buffer before the expert kernel,
    which then reads the copy instead of host memory. The kernel reads an
    expert once per tile of its tokens, so in large steps a single copy moves
    fewer bytes than reading in place; small steps (decode) read only the
    routed experts and are better off in place. One buffer per distinct
    offloaded expert-weight shape is shared by all layers and counted by memory
    profiling, so it comes out of the KV cache.
        - 0 (default): never stage.
        - "auto" (or -1): stage every step that runs without CUDA graphs, i.e.
          more than `compilation_config.max_cudagraph_capture_size` tokens,
          and at least 512.
        - N > 0: stage steps with at least N tokens.
    Steps inside CUDA graphs never stage. CUDA and XPU.
    """

    cpu_offload_stage_prefetch: bool = False
    """With staging on, keep two buffers per staged weight shape and copy
    the next layer's weights on a side stream while the current layer's
    experts kernel runs."""

    @field_validator("cpu_offload_stage_min_tokens", mode="before")
    @classmethod
    def _parse_auto(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() == "auto":
            return -1
        return value

    def resolve_stage_min_tokens(self, max_cudagraph_capture_size: int) -> int:
        """`cpu_offload_stage_min_tokens` with "auto" (-1) resolved."""
        if self.cpu_offload_stage_min_tokens >= 0:
            return self.cpu_offload_stage_min_tokens
        return max((max_cudagraph_capture_size or 0) + 1, 512)

    def compile_factors(self) -> CompileFactors:
        """Provide the factors used for hashing UVAOffloadConfig. The staging
        threshold and prefetch are left out: they only act inside the opaque
        moe_forward op."""
        return get_compile_factors(
            self,
            ignored_factors={
                "cpu_offload_stage_min_tokens",
                "cpu_offload_stage_prefetch",
                # acts inside the opaque moe_forward op; the graph split it
                # asks for is part of the compilation config's own factors
                "cpu_offload_pinned_gb",
            },
        )

    def host_experts_on(self) -> bool:
        """Whether some expert slots off the device are computed on the CPU
        (the pinned budget is below the host budget)."""
        return (
            self.cpu_offload_per_expert
            and self.cpu_offload_pinned_gb is not None
            and self.cpu_offload_pinned_gb < self.cpu_offload_gb
        )


@config
class PrefetchOffloadConfig:
    """Configuration for prefetch-based CPU offloading.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    """

    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for prefetch offloading.
    Unmatched parameters are not offloaded. If this set is empty, ALL
    parameters of each offloaded layer are offloaded.
    Uses segment matching: "w13_weight" matches "mlp.experts.w13_weight"
    but not "mlp.experts.w13_weight_scale".
    """


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Async prefetch with group-based layer offloading.
    """

    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        if self.offload_backend == "prefetch" or self.prefetch.offload_group_size > 0:
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        # Warn if both backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        if self.offload_backend == "uva" and prefetch_active:
            warnings.warn(
                "Prefetch offload fields are set but offload_backend='uva'. "
                "Prefetch settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "prefetch" and uva_active:
            warnings.warn(
                "UVA offload fields are set but offload_backend='prefetch'. "
                "UVA settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "auto" and uva_active and prefetch_active:
            warnings.warn(
                "Both UVA and prefetch offload fields are set with "
                "offload_backend='auto'. Prefetch backend will be selected. "
                "Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )
        return self

    def compile_factors(self) -> CompileFactors:
        """Provide a hash that uniquely identifies all the offload configs.

        All fields but the UVA staging threshold are included because
        PrefetchOffloader patches module forwards and inserts custom ops
        (wait_prefetch, start_prefetch) into the computation graph. Changing
        any offload setting can alter which layers are hooked and how prefetch
        indices are computed, so the compilation cache must distinguish them.
        """
        return get_compile_factors(self, ignored_factors=set())

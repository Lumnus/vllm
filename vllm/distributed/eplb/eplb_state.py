# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert parallelism load balancer (EPLB) metrics and states.

# Glossary

- **Logical Expert**: An expert that is part of the model's logical structure.
  It holds a set of weights and is replicated across multiple physical
  experts.
- **Redundant Expert**: To achieve load balancing, for some popular logical
  experts, we create additional copies of the expert weights. During inference,
  each of these copies can be routed to by the same set of tokens.
- **Physical Expert**: An expert that is instantiated on a specific device.
  It is a replica of a logical expert and can be rearranged across devices.
  I.e., one logical expert may have multiple sets of weights initialized on
  different devices, and each of these sets is a physical expert.
- **Local Physical Expert**: A physical expert that is instantiated on the
  current device.

For example: DeepSeek-R1 has 256 logical experts, so each MoE layer
has 256 sets of linear layer weights in the model parameters. If we add 32
redundant experts, DeepSeek-R1 will have 256 + 32 = 288 physical experts in
total. And when deploying, we'll have 288 sets of linear layer weights for each
MoE layer. If we have 32 EP ranks, then each GPU will hold 288 / 32 = 9 local
physical experts.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.distributed import ProcessGroup, all_reduce

from vllm.config import ModelConfig, ParallelConfig
from vllm.config.utils import hash_config_cached
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_ep_group,
    get_eplb_group,
    get_node_count,
    in_the_same_node_as,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1 import runtime_debug

from . import fault_injection
from .async_worker import start_async_worker
from .eplb_communicator import EplbCommunicator, create_eplb_communicator
from .eplb_state_file import read_expert_loads, read_seed_loads, write_expert_loads
from .eplb_trace import ExpertLoadTracer
from .eplb_utils import CpuGpuEvent
from .expert_weights import (
    ExpertMover,
    expert_mover,
    fast_slot_count,
    new_expert_buffer,
    order_slots_by_tier,
)
from .policy import (
    EPLB_POLICIES,
    AbstractEplbPolicy,
    DefaultEplbPolicy,
    IncrementalEplbPolicy,
)
from .rebalance_execute import (
    AsyncEplbLayerResult,
    copy_rows,
    move_from_buffer,
    rearrange_expert_weights_inplace,
)
from .residency_mask import ResidencyMask, logical_mask_from_residency

logger = init_logger(__name__)


def _compute_eplb_load_stats(
    num_tokens_per_rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    avg_tokens = num_tokens_per_rank.mean(dim=1).sum()
    max_tokens = num_tokens_per_rank.max(dim=1).values.sum()
    return avg_tokens, max_tokens


@dataclass
class EplbStats:
    """Model stats used in EPLB rebalancing algorithm."""

    global_expert_load_window: torch.Tensor
    """
    Experts load window.
    Shape: (window_size, num_moe_layers, num_physical_experts)
    """
    num_replicas: int
    """
    Number of physical experts.
    """
    num_groups: int
    """
    Number of expert groups.
    """
    num_nodes: int
    """
    Number of nodes.
    """
    num_gpus: int
    """
    Number of GPUs.
    """
    tier_only: bool = False
    """A tier pass: the placement of experts on ranks stays, the layers
    choose the tiers of their slots from the window
    (`EPLBConfig.tier_interval`)."""


@dataclass
class EplbModelState:
    """EPLB metrics."""

    physical_to_logical_map: torch.Tensor
    """
    Mapping from physical experts to logical experts.

    Shape: (num_moe_layers, num_physical_experts)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[0, 1, 2, 3, 0, 1],
     [0, 2, 0, 1, 0, 3]]
    ```
    """
    physical_to_logical_map_buffer: torch.Tensor
    """Maximum-capacity buffer backing `physical_to_logical_map`."""
    logical_to_physical_map: torch.Tensor
    """
    Mapping from logical experts to physical experts.

    This is a sparse matrix, where -1 indicates no mapping.

    Shape: (num_moe_layers, num_logical_experts, num_redundant_experts + 1)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[[0, 4, -1],
      [1, 5, -1],
      [2, -1, -1],
      [3, -1, -1]],
     [[0, 2, 4],
      [3, -1, -1],
      [1, -1, -1],
      [5, -1, -1]]]
    ```
    """
    logical_replica_count: torch.Tensor
    """
    Number of replicas for each logical expert.
    This is exactly the non-`-1` count in the `logical_to_physical_map`.

    Shape: (num_moe_layers, num_logical_experts)

    # Example
    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the count could look like this:

    ```
    [[2, 2, 1, 1],
     [3, 1, 1, 1]]
    """

    expert_load_pass: torch.Tensor
    """
    Expert load during this forward pass. 
    We use the token count each expert processes as the load.

    Shape: (num_moe_layers, num_physical_experts)
    """
    expert_load_pass_buffer: torch.Tensor
    """Maximum-capacity buffer backing `expert_load_pass`."""
    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_physical_experts)

    NOTE: The expert_load_view now records load for all physical experts
    rather than just local experts. This ensures consistent load statistics
    across different dispatch methods (naive all-to-all, DeepEP).
    The recorded load will be multiplied by dp_size when using naive all-to-all
    due to each DP rank contributing the same token set to the calculation.
    See:
    https://github.com/vllm-project/vllm/pull/22167#pullrequestreview-3086143856
    """
    last_expert_load: torch.Tensor
    """Global logical-expert totals from the last reshuffle; equal demand initially."""
    model_name: str
    model: MixtureOfExperts
    expert_buffer: list[torch.Tensor]
    """
    The buffer to store the expert weights during transfer.
    """
    rebalanced: bool
    """
    This flag is only used when running Async EPLB. It is set to True by the main thread
    after the new expert maps have been computed. This indicates that the async worker
    should start transferring weights. move_to_workspace sets this flag to False when
    all weights have been transferred and the new map has been successfully committed.

    rebalanced relies on the GIL to synchronize access between the main thread and
    the async worker.
    """
    eplb_stats: EplbStats | None
    """
    EPLB stats for the model.
    """
    device_index: int | None
    """
    Device index for the async EPLB worker thread.
    """
    communicator: EplbCommunicator
    """
    The communicator for expert weight transfers.
    """
    pending_result: AsyncEplbLayerResult | None = None
    """
    Set by the async worker after all writes to expert_buffer are done. Consumed
    and reset to None by the main thread in move_to_workspace() after the contents of
    expert_buffer have been transferred out. At most one result is pending at a time.

    pending_result relies on the GIL to synchronize access between the main thread and
    the async worker.
    """
    num_unpadded_tokens_tensors: list[torch.Tensor] | None = None
    """
    Per-ubatch scalar int32 tensors holding the number of real (non-padding)
    tokens.  Allocated once in `EplbState.add_model` so that device
    pointers remain stable across CUDA-graph replays.  The router kernel
    indexes this list with `dbo_current_ubatch_id()`.
    """
    fast_slots: list[int | None] | None = None
    """Per MoE layer, the fast-tier slots of each rank when the layer keeps
    its expert slots in tiers (`TieredExpertWeight`), else None; None when
    no layer does."""
    has_saved_load: bool = False
    """`last_expert_load` holds loads read from `EPLBConfig.state_path` that
    `restore_saved_placement` has not placed the experts by yet."""
    movers: list[ExpertMover | None] | None = None
    """Per MoE layer, the mover of a layer that places its own experts (from the
    expert store) instead of having EPLB send them; set when every layer of
    the model does. Such a model has no expert buffer and no communicator
    transfer."""
    plan_load: torch.Tensor | None = None
    """The load window (CPU, logical experts) the async worker's current plan was
    made from."""
    rearrangement_stats: dict[str, float] = field(default_factory=dict)
    """Units and bytes read and milliseconds spent by the movers since the
    last log line of a store-backed rearrangement."""
    abandon_votes: int = 0
    """Ranks whose mover could not prepare the pending layer, as of the last
    `_all_ranks_result_ready` that answered true (the same on every rank)."""
    abandoned_rearrangements: int = 0
    """Rearrangements the ranks agreed to stop because a mover could not
    prepare a layer."""
    completed_rearrangements: int = 0
    """Async rearrangements whose last layer was committed."""
    rearrangement_started: float = 0.0
    """`time.perf_counter()` when the async rearrangement in flight started."""
    last_rearrangement_seconds: float | None = None
    """Seconds from start to the last layer's commit of the last completed
    async rearrangement on this rank."""
    load_scrubbed: bool = False
    """The movers' scrub after load has been offered (it runs only with the
    layers' debug setting on)."""
    residency: ResidencyMask | None = None
    """The model's residency mask (`EPLBConfig.residency_mask`), if on."""
    map_change_hook: Callable[["EplbModelState"], None] | None = None
    """Called before a placement is committed to this model (the expert load
    trace maps the counts so far with the placement still in force)."""
    move_cost: dict[str, float] = field(default_factory=dict)
    """Where the async rearrangement in flight spends its time on this rank,
    by phase (`MOVE_COST_FIELDS`), written by the async worker and the
    serving thread; one `[debug eplb] move` line at its end, then cleared."""


MOVE_COST_FIELDS = (
    "plan_ms",
    "flag_ms",
    "prepare_ms",
    "device_wait_ms",
    "ready_checks",
    "ready_wait_ms",
    "commit_ms",
    "commit_steps",
    "steps",
    "step_ms",
    "units",
    "bytes",
    "card_bytes",
    "changed",
    "verify_ms",
)
"""The per-move phase counters; see `EplbState._emit_move_cost`."""


class EplbState:
    """EplbState of each expert parallel model. Key is the model config hash."""

    def __init__(self, parallel_config: ParallelConfig, device: torch.device):
        self.parallel_config = parallel_config
        self.device = device
        self.model_states: dict[str, EplbModelState] = {}
        self.policy: type[AbstractEplbPolicy] = DefaultEplbPolicy
        """
        Selected EPLB algorithm class
        """
        self.expert_load_window_step: int = 0
        """
        Current step in the sliding window.

        Different from `expert_rearrangement_step`, 
        each EP rank may have its own `expert_load_window_step`.
        """
        self.expert_load_window_size: int = 0
        """
        Size of the expert load sliding window.
        This is a constant and is taken from the config.
        """
        self.expert_rearrangement_step: int = 0
        """
        Steps after last rearrangement.
        Will trigger a rearrangement if it exceeds the threshold.

        NOTE: Keep in mind that all EP ranks need to have the same
        `expert_rearrangement_step` value to ensure synchronization.
        Otherwise, the rearrangement will hang at collective
        communication calls.
        """
        self.expert_rearrangement_step_interval: int = 0
        """
        Interval for expert rearrangement steps.
        This is a constant and is taken from the config.
        0: no periodic rearrangement (`EPLBConfig.step_interval`).
        """
        self.rearrange_requested: bool = False
        """A rearrangement asked for (runtime control) starts at the next
        step whatever the interval; set on every rank between the same two
        steps."""
        self.low_imbalance_skips: int = 0
        """Periodic rearrangements not started because the window's imbalance
        was below `EPLBConfig.min_imbalance`."""
        self.last_imbalance: float | None = None
        """Busiest rank's load over the mean rank load at the last periodic
        rearrangement's test (`min_imbalance`), or None."""
        self._last_step_time: float | None = None
        self._step_ms_idle: float | None = None
        """Moving mean of the serving step period outside rearrangements, for
        the `[debug eplb] move` line."""
        self.should_record_tensor: torch.Tensor | None = None
        """
        Shared scalar bool tensor for all layers.  Every
        `EplbLayerState` holds a reference to the **same** object so
        a single `.fill_()` updates all layers at once.  Allocated on the
        first call to `_propagate_shared_tensors`.
        """
        self._should_record_value: bool | None = None
        self._step_token_weight: float = 1.0
        """Weight of the coming step's load, from its decode/prefill mix
        (`EPLBConfig.decode_token_weight` / `prefill_token_weight`)."""
        self._window_token_weights: list[float] = []
        """The weight of each step in the load window."""
        """Last value written to `should_record_tensor`."""
        self._pending_balancedness: deque[
            tuple[int, int, torch.Event, list[tuple[str, torch.Tensor]]]
        ] = deque()
        """Balancedness stats whose copies to the host are in flight, oldest
        first."""
        self._trace: ExpertLoadTracer | None = None
        """Writer of the routed expert load over time, when
        `EPLBConfig.trace_path` is set."""
        eplb_config = parallel_config.eplb_config
        if eplb_config.trace_path is not None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            self._trace = ExpertLoadTracer(
                eplb_config.trace_path,
                eplb_config.trace_interval,
                rank,
                sample=eplb_config.trace_sample,
                rotate_bytes=eplb_config.trace_rotate_mb << 20,
                keep=eplb_config.trace_keep,
            )
        self.is_async: bool = False
        """
        The flag indicates whether the EPLB is running in async mode.
        """
        self.rearrange_event: CpuGpuEvent = CpuGpuEvent()
        """
        Event to signal when a new rearrangement is needed for the async thread.
        """
        self.ft_reset_epoch: int = 0
        """
        Incremented by FT recovery (reset_eplb_async_state). The async worker
        snapshots it when a transfer cycle starts; a change in the value means
        a recovery happened, so the cycle is abandoned.
        """
        self.async_worker: threading.Thread | None = None
        """
        Background thread handling async transfers.
        """
        self.async_worker_error: Exception | None = None
        self.last_rearrange_changed: bool = False
        """Whether the last synchronous rearrangement changed any expert slot.
        The same on every EP rank (the placement follows all-reduced loads)."""
        """
        Exception that stopped the async worker thread, if any.
        """
        self.device_index: int | None = None
        """
        Device index for the async EPLB worker thread.
        """
        self.skipped_rearrangements: int = 0
        """Rearrangements every rank skipped because one rank could not
        allocate memory for them (`_skip_rearrangement`); the same on every EP
        rank."""
        # Set from outside a step by the runtime control surface
        # (`vllm/distributed/eplb/runtime_control.py`), between two steps on
        # every EP rank alike; read by `step`.
        self.rearrangements_paused: bool = False
        """While set, an interval that ends starts no rearrangement (one in
        flight finishes); the step counter starts the next interval."""
        self.paused_intervals: int = 0
        """Intervals that ended while rearrangements were paused."""
        self.next_uses_saved_load: bool = False
        """The next rearrangement places by `last_expert_load` (loads read
        from a state file on demand) instead of the load window."""
        self.window_fill: int = 0
        """Steps the load window has recorded since the last rearrangement,
        at most its size."""
        self.rearrangements: int = 0
        """Rearrangements started (past both agreements; profile runs and
        skipped ones not counted)."""
        self.tier_passes: int = 0
        """Tier passes started (`EPLBConfig.tier_interval`, or asked for by a
        layer whose tier capacity changed)."""
        self.tier_step: int = 0
        """Steps since the last tier pass."""
        self.last_rearrange_seconds: float | None = None
        """Wall time of the last synchronous rearrangement on this rank."""
        if self.device.type in ("cuda", "xpu"):
            self.device_index = self.device.index
            if self.device_index is None and torch.accelerator.is_available():
                self.device_index = torch.accelerator.current_device_index()

    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int,
        num_redundant_experts: int,
    ) -> Sequence[int]:
        """Build an initial expert arrangement using the following structure:
        [original routed experts, redundant experts]

        Returns:
            physical_to_logical_map (Sequence[int]): A list of integers,
                where each integer is the index of the logical expert
                that the corresponding physical expert maps to.

        """
        global_physical_to_logical_map = list(range(num_routed_experts))
        global_physical_to_logical_map += [
            i % num_routed_experts for i in range(num_redundant_experts)
        ]
        return global_physical_to_logical_map

    def validate_ep_configuration(self, new_model: MixtureOfExperts):
        """Validate that the expert parallel configuration of
        the new model is the same as the existing models.
        """
        if len(self.model_states) > 0:
            model = next(iter(self.model_states.values())).model
            self.assert_confs_equal(model, new_model)

    @staticmethod
    def assert_confs_equal(
        model: MixtureOfExperts, new_model: MixtureOfExperts
    ) -> None:
        """Raise if the models have different expert parallel configurations."""
        if (
            model.num_routed_experts != new_model.num_routed_experts
            or model.num_redundant_experts != new_model.num_redundant_experts
            or model.num_physical_experts != new_model.num_physical_experts
            or model.num_logical_experts != new_model.num_logical_experts
            or model.num_expert_groups != new_model.num_expert_groups
        ):
            raise RuntimeError(
                f"Model: {type(model)} "
                f"with config {model.num_routed_experts} "
                f"{model.num_redundant_experts} {model.num_physical_experts} "
                f"{model.num_logical_experts} {model.num_expert_groups} "
                f"mismatch with new model {type(new_model)} "
                f"with config {new_model.num_routed_experts} "
                f"{new_model.num_redundant_experts} {new_model.num_physical_experts} "
                f"{new_model.num_logical_experts} {new_model.num_expert_groups}"
            )

    def add_model(
        self,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        model_name: str | None = None,
    ):
        """Build the initial EPLB state."""
        self.validate_ep_configuration(model)
        self.is_async = self.parallel_config.eplb_config.use_async

        physical_to_logical_map_list = (
            EplbState.build_initial_global_physical_to_logical_map(
                model.num_routed_experts,
                model.num_redundant_experts,
            )
        )
        physical_to_logical_map = torch.tensor(
            physical_to_logical_map_list,
            device=self.device,
        )
        physical_expert_capacity = model.num_physical_experts
        if self.parallel_config.enable_elastic_ep:
            physical_expert_capacity = (
                model.num_local_physical_experts
                * self.parallel_config.elastic_ep_max_dp_size
                * get_ep_group().world_size
                // self.parallel_config.data_parallel_size
            )
            physical_to_logical_map = torch.nn.functional.pad(
                physical_to_logical_map,
                (0, physical_expert_capacity - model.num_physical_experts),
                value=-1,
            )
        # Assuming 8 GPUs per node, this supports up to
        # (1023 + 1) / 8 = 128 nodes for now.
        # TODO(rui): make this configurable
        MAX_EXPERT_REDUNDANCY = 1023
        assert model.num_redundant_experts <= MAX_EXPERT_REDUNDANCY, (
            f"num_redundant_experts {model.num_redundant_experts} "
            f"must be less than or equal to {MAX_EXPERT_REDUNDANCY}"
        )
        # Only elastic EP changes the physical expert count at runtime.
        max_slots_per_logical_expert = (
            MAX_EXPERT_REDUNDANCY + 1
            if self.parallel_config.enable_elastic_ep
            else model.num_redundant_experts + 1
        )
        logical_to_physical_map = torch.full(
            (model.num_logical_experts, max_slots_per_logical_expert),
            -1,
            device=self.device,
        )
        logical_replica_count = torch.zeros(
            (model.num_logical_experts,),
            device=self.device,
            dtype=torch.long,
        )

        for i in range(model.num_physical_experts):
            logical_idx = physical_to_logical_map[i]
            logical_to_physical_map[logical_idx, logical_replica_count[logical_idx]] = i
            logical_replica_count[logical_idx] += 1

        # Duplicate initial mapping for all layers
        physical_to_logical_map = (
            physical_to_logical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )
        physical_to_logical_map_buffer = physical_to_logical_map
        physical_to_logical_map = physical_to_logical_map_buffer[
            :, : model.num_physical_experts
        ]
        logical_to_physical_map = (
            logical_to_physical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
                -1,
            )
            .contiguous()
        )
        logical_replica_count = (
            logical_replica_count.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )

        expert_load_pass_buffer = torch.zeros(
            (model.num_moe_layers, physical_expert_capacity),
            dtype=torch.int32,
            device=self.device,
        )
        expert_load_pass = expert_load_pass_buffer[:, : model.num_physical_experts]
        self.expert_load_window_size = self.parallel_config.eplb_config.window_size
        self._window_token_weights = [1.0] * self.expert_load_window_size
        expert_load_window = torch.zeros(
            (
                self.expert_load_window_size,
                model.num_moe_layers,
                model.num_physical_experts,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = self.parallel_config.eplb_config.step_interval
        self.expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4
        )
        self.expert_rearrangement_step_interval = eplb_step_interval

        eplb_config = self.parallel_config.eplb_config
        policy_type = eplb_config.policy
        self.policy = EPLB_POLICIES[policy_type]
        if issubclass(self.policy, IncrementalEplbPolicy):
            self.policy = self.policy.configured(
                eplb_config.max_moves_per_rank, eplb_config.move_min_gain
            )
        logger.debug("Selected EPLB policy: %s", policy_type)

        # num_ubatches is 0 when DBO is disabled.
        num_ubatches = max(1, self.parallel_config.num_ubatches)
        num_unpadded_tokens_tensors = [
            torch.tensor(0, dtype=torch.int32, device=self.device)
            for _ in range(num_ubatches)
        ]

        model.set_eplb_state(
            expert_load_pass_buffer,
            logical_to_physical_map,
            logical_replica_count,
        )
        self._propagate_shared_tensors(model, num_unpadded_tokens_tensors)

        backend = self.parallel_config.eplb_config.communicator
        assert backend is not None, (
            "EPLB communicator backend must be set by ParallelConfig"
        )
        num_local_experts = model.expert_weights[0][0].shape[0]
        movers = [expert_mover(ws) for ws in model.expert_weights]
        store_backed = any(m is not None for m in movers)
        if store_backed:
            if not all(m is not None for m in movers):
                raise NotImplementedError(
                    "EPLB: some MoE layers place their own experts (expert store) "
                    "and some do not; the store needs every layer of the model"
                )
            if not self.is_async:
                raise ValueError(
                    "EPLB with the expert store reads experts while serving: "
                    "set use_async to true"
                )
        # Tiered slots keep a full buffer: the tier step after the policy may
        # reorder any of a rank's slots.
        tiered = any(fast_slot_count(ws) is not None for ws in model.expert_weights)
        num_buffer_rows = (
            num_local_experts
            if tiered
            else self._transfer_buffer_rows(num_local_experts)
        )
        expert_buffer = (
            None
            if backend == "symm_mem"
            else []  # the experts are read from the store: nothing is staged
            if store_backed
            else [
                new_expert_buffer(w)
                if num_buffer_rows == num_local_experts
                else w.new_empty((num_buffer_rows, *w.shape[1:]))
                for w in model.expert_weights[0]
            ]
        )
        communicator = create_eplb_communicator(
            group_coordinator=get_eplb_group(),
            backend=backend,
            expert_weights=model.expert_weights,
            expert_buffer=expert_buffer,
        )
        expert_buffer = communicator.owned_expert_buffer or expert_buffer
        assert expert_buffer is not None
        # Compile the copy-back kernels now rather than mid-serving (no-op copy).
        if not store_backed:
            copy_rows(model.expert_weights[0], [0], model.expert_weights[0], [0])

        model_state = EplbModelState(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_buffer=physical_to_logical_map_buffer,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_pass=expert_load_pass,
            expert_load_pass_buffer=expert_load_pass_buffer,
            expert_load_window=expert_load_window,
            last_expert_load=torch.ones(
                (model.num_moe_layers, model.num_logical_experts),
                dtype=torch.int64,
                device=self.device,
            ),
            model_name=model_name or model_config.model,
            model=model,
            expert_buffer=expert_buffer,
            rebalanced=False,
            eplb_stats=None,
            device_index=self.device_index,
            communicator=communicator,
            num_unpadded_tokens_tensors=num_unpadded_tokens_tensors,
        )
        fast_slots = [fast_slot_count(ws) for ws in model.expert_weights]
        if store_backed:
            # The layers choose their fast set themselves, without permuting
            # slots: EPLB's tier step has nothing to order.
            model_state.movers = movers
            logger.info(
                "EPLB: the expert store mover places the experts of %d layers: "
                "no expert buffer, no profile-time reservation, no rank-to-rank "
                "transfer.",
                len(movers),
            )
        elif any(f is not None for f in fast_slots):
            model_state.fast_slots = fast_slots
        if self._trace is not None:
            model_state.map_change_hook = self._trace.before_map_change
        self.model_states[hash_config_cached(model_config)] = model_state
        self._add_residency_mask(model_state)
        if self.parallel_config.eplb_config.state_path is not None:
            self._read_saved_load(model_state)
        if not model_state.has_saved_load and self.has_tier_seed():
            self._read_seed_load(model_state)

    def has_tier_seed(self) -> bool:
        """Whether `EPLBConfig.seed_path` is set (configuration: the same on
        every rank)."""
        return self.parallel_config.eplb_config.seed_path is not None

    def _read_seed_load(self, model_state: EplbModelState) -> None:
        """Fill `last_expert_load` from `EPLBConfig.seed_path` when
        `state_path` gave none: the restore then places the experts, and the
        layers' tiers, by it."""
        path = self.parallel_config.eplb_config.seed_path
        assert path is not None
        model = model_state.model
        names = [layer.layer_name for layer in model.moe_layers]
        seed = read_seed_loads(path, names, model.num_logical_experts)
        found = 0
        for layer_idx, name in enumerate(names):
            loads = seed.get(name)
            if loads is None:
                continue
            model_state.last_expert_load[layer_idx].copy_(
                torch.tensor(loads, dtype=torch.int64)
            )
            found += 1
        model_state.has_saved_load = found > 0
        logger.info(
            "EPLB: no saved expert loads; the seed %s gives loads for %d of "
            "%d MoE layers of %s: the experts and their tiers start from them.",
            path,
            found,
            model.num_moe_layers,
            model_state.model_name,
        )

    def _read_saved_load(self, model_state: EplbModelState) -> None:
        """Fill `last_expert_load` from `EPLBConfig.state_path`."""
        path = self.parallel_config.eplb_config.state_path
        assert path is not None
        saved = read_expert_loads(path)
        model = model_state.model
        found = 0
        for layer_idx, layer in enumerate(model.moe_layers):
            loads = saved.get(layer.layer_name)
            if loads is None:
                continue
            if len(loads) != model.num_logical_experts:
                logger.warning(
                    "EPLB: %s has %d saved loads for %d logical experts in %s; "
                    "not used",
                    layer.layer_name,
                    len(loads),
                    model.num_logical_experts,
                    path,
                )
                continue
            model_state.last_expert_load[layer_idx].copy_(
                torch.tensor(loads, dtype=torch.int64)
            )
            found += 1
        model_state.has_saved_load = found > 0
        logger.info(
            "EPLB: saved expert loads for %d of %d MoE layers of %s in %s",
            found,
            model.num_moe_layers,
            model_state.model_name,
            path,
        )

    def order_tiers(
        self,
        model_state: EplbModelState,
        physical_to_logical_map: torch.Tensor,
        load: torch.Tensor,
        num_ranks: int,
    ) -> torch.Tensor:
        """The policy's placement with each rank's most-loaded experts in its
        fast slots, for layers whose slots are in tiers (CPU tensors)."""
        if model_state.fast_slots is None:
            return physical_to_logical_map
        ordered = order_slots_by_tier(
            physical_to_logical_map.numpy(),
            load.float().numpy(),
            model_state.fast_slots,
            num_ranks,
            self.parallel_config.eplb_config.tier_hysteresis,
        )
        return torch.from_numpy(ordered)

    def _transfer_buffer_rows(self, num_local_experts: int) -> int:
        """Rows of the transfer buffer per expert tensor: one per local expert,
        or, when the policy bounds how many slots of a rank a rearrangement
        changes (`incremental`), that bound. The bounded size needs a
        communicator that is handed buffer rows (not one that addresses the
        buffer by expert row itself) and a fixed set of ranks."""
        config = self.parallel_config.eplb_config
        bounded = (
            issubclass(self.policy, IncrementalEplbPolicy)
            and config.communicator
            in ("torch_nccl", "torch_gloo", "torch_xccl", "pynccl")
            and not config.migration_batching_enabled
            and not self.parallel_config.enable_elastic_ep
        )
        if not bounded:
            return num_local_experts
        return min(num_local_experts, config.max_moves_per_rank)

    def _restore_rounds(self) -> int:
        """Blocking rearrangements the restore may take: a bounded policy
        reaches the saved placement over several."""
        if not issubclass(self.policy, IncrementalEplbPolicy):
            return 1
        num_local = max(
            s.model.num_physical_experts // get_ep_group().device_group.size()
            for s in self.model_states.values()
        )
        return 2 * -(-num_local // self.policy.max_moves_per_rank)

    def restore_saved_placement(self) -> None:
        """Place the experts by the loads read from `EPLBConfig.state_path`,
        once, before serving. A blocking rearrangement on every EP rank; the
        decision follows EP rank 0, which also supplies the loads."""
        ep_group = get_ep_group().device_group
        pending = torch.tensor(
            [any(s.has_saved_load for s in self.model_states.values())],
            dtype=torch.int32,
            device=self.device,
        )
        if ep_group.rank() != 0:
            pending.zero_()
        all_reduce(pending, group=ep_group)
        with gpu_sync_allowed():
            restore = bool(pending.item())
        if restore and any(s.movers is not None for s in self.model_states.values()):
            self._restore_by_fill()
        if restore and any(s.movers is None for s in self.model_states.values()):
            # Every rank computes the same placement from the same loads, so
            # every rank stops after the same round.
            for _ in range(self._restore_rounds()):
                self.rearrange(use_last_expert_load=True, blocking=True)
                if not self.last_rearrange_changed:
                    break
        for model_state in self.model_states.values():
            model_state.has_saved_load = False

    def _restore_by_fill(self) -> None:
        """Place the experts by the saved loads, once, before serving, for models
        whose layers fill their slots from the expert store. The policy's rounds
        run on the host over the placement alone; the layers are then filled
        straight into the final placement, so the restore moves no expert
        between ranks and reads each slot once."""
        ep_group = get_ep_group().device_group
        is_main_rank = ep_group.rank() == 0
        states = [s for s in self.model_states.values() if s.movers is not None]
        windows = []
        for state in states:
            saved = state.last_expert_load
            windows.append(saved.clone() if is_main_rank else torch.zeros_like(saved))
        windows = self._allreduce_list(windows)
        num_nodes = get_node_count()
        num_gpus = ep_group.size()
        if num_gpus % num_nodes != 0:
            num_nodes = 1
        hysteresis = self.parallel_config.eplb_config.tier_hysteresis
        start = time.perf_counter()
        self.last_rearrange_changed = False
        for state, window in zip(states, windows):
            assert state.movers is not None
            state.last_expert_load = window
            model = state.model
            with gpu_sync_allowed():
                load = window.cpu()
                current = state.physical_to_logical_map.cpu()
            placement = current
            for _ in range(self._restore_rounds()):
                new = self.policy.rebalance_experts(
                    load,
                    model.num_physical_experts,
                    model.num_expert_groups,
                    num_nodes,
                    num_gpus,
                    placement,
                )
                if torch.equal(new, placement):
                    break
                placement = new
            changed = not torch.equal(placement, current)
            self.last_rearrange_changed |= changed
            # Even with the placement unchanged the saved loads decide which
            # slots are on the card: the layers fill (or keep) their rows.
            for layer, mover in enumerate(state.movers):
                assert mover is not None
                mover.fill_to(
                    placement[layer].numpy(), load[layer].float().numpy(), hysteresis
                )
            if changed:
                _commit_eplb_maps(state, new_physical_to_logical_map=placement)
        logger.info(
            "EPLB: restored the saved placement by filling the slots from the "
            "expert store in %.2f s.",
            time.perf_counter() - start,
        )

    def _save_expert_load(self, global_expert_load_windows: list[torch.Tensor]) -> None:
        """Write the loads a rearrangement uses to `EPLBConfig.state_path`."""
        path = self.parallel_config.eplb_config.state_path
        assert path is not None
        layers: dict[str, list[int]] = {}
        for model_state, window in zip(
            self.model_states.values(), global_expert_load_windows
        ):
            with gpu_sync_allowed():
                loads = window.cpu().tolist()
            for layer, layer_loads in zip(model_state.model.moe_layers, loads):
                if sum(layer_loads) > 0:  # no traffic must not erase a state
                    layers[layer.layer_name] = [round(x) for x in layer_loads]
        if not layers:
            return
        try:
            write_expert_loads(path, layers)
        except OSError as err:
            logger.warning("EPLB: cannot write the expert loads to %s: %s", path, err)

    def prepare_forward(
        self,
        model_config: ModelConfig,
        num_unpadded_tokens: int,
        ubatch_slices: list | None = None,
        num_prefill_tokens: int | None = None,
    ) -> None:
        """Fill the per-[u]batch `num_unpadded_tokens` tensors before a
        forward pass, and weigh the step's load by its token mix.

        Args:
            model_config: Identifies which `EplbModelState` to update.
            num_unpadded_tokens: Total number of real (non-padding) tokens
                in the batch.
            ubatch_slices: When DBO is active, a list of
                `UBatchSlice` objects describing each micro-batch's
                token range.  When `None`, only `tensors[0]` is filled.
            num_prefill_tokens: Tokens of requests still in prefill; the
                rest decode. Only read when the load is weighted by phase.

        """
        self._step_token_weight = self.step_token_weight(
            num_unpadded_tokens, num_prefill_tokens
        )
        model_state = self.model_states.get(hash_config_cached(model_config))
        if model_state is None or model_state.num_unpadded_tokens_tensors is None:
            return
        tensors = model_state.num_unpadded_tokens_tensors
        if ubatch_slices is None:
            tensors[0].fill_(num_unpadded_tokens)
        else:
            for i, ubatch_slice in enumerate(ubatch_slices):
                ts = ubatch_slice.token_slice
                # Real tokens in this ubatch: clamp the global count into
                # the slice range so partially-filled ubatches get the
                # correct count.
                val = max(0, min(num_unpadded_tokens, ts.stop) - ts.start)
                tensors[i].fill_(val)

    def step_token_weight(
        self, num_tokens: int, num_prefill_tokens: int | None
    ) -> float:
        """The weight of a step of `num_tokens` tokens, `num_prefill_tokens`
        of them prefill: the token-weighted mean of the two phase weights."""
        config = self.parallel_config.eplb_config
        if not config.weights_tokens or num_prefill_tokens is None:
            return 1.0
        if num_tokens <= 0:
            return 0.0
        prefill = min(max(num_prefill_tokens, 0), num_tokens)
        decode = num_tokens - prefill
        return (
            config.decode_token_weight * decode + config.prefill_token_weight * prefill
        ) / num_tokens

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Step the EPLB state.

        Args:
            is_dummy (bool): If `True`, this is a dummy step and the load
                metrics recorded in this forward pass will not count.
                Defaults to `False`.
            is_profile (bool): If `True`, perform a dummy rearrangement
                with maximum communication cost. This is used in
                `profile_run` to reserve enough memory
                for the communication buffer.
            log_stats (bool): If `True`, log the expert load metrics.

        # Stats
            The metrics are all summed up across layers.
            - `avg_tokens`: The average load across ranks.
            - `max_tokens`: The maximum load across ranks.
            - `balancedness`: The ratio of average load to maximum load.

        """
        ep_group = get_ep_group().device_group
        for eplb_model_state in self.model_states.values():
            if not eplb_model_state.load_scrubbed:
                # The first step after load (a dummy run before serving): with
                # the layers' debug setting on, check every slot once.
                eplb_model_state.load_scrubbed = True
                _scrub_movers(eplb_model_state, ep_group.rank(), full=True, when="load")
        if is_dummy:
            # Do not record load metrics for dummy steps
            for eplb_model_state in self.model_states.values():
                eplb_model_state.expert_load_pass.zero_()

        if is_profile:
            self.rearrange(is_profile=True)
            return
        if log_stats:
            self._maybe_log_balancedness()
        if (
            log_stats
            and self.expert_rearrangement_step
            % self.parallel_config.eplb_config.log_balancedness_interval
            == 0
        ):
            # Sync the expert load pass for each model (main and drafter).
            # expert_load_pass: (num_moe_layers, num_physical_experts)
            expert_load_pass_list = self._sync_load_pass()
            ep_group = get_ep_group().device_group
            if ep_group.rank() == 0:
                stats: list[tuple[str, torch.Tensor]] = []
                for expert_load_pass, eplb_model_state in zip(
                    expert_load_pass_list, self.model_states.values()
                ):
                    # num_tokens_per_rank: (num_moe_layers, num_ranks)
                    num_tokens_per_rank = (
                        expert_load_pass.reshape(
                            expert_load_pass.shape[0], ep_group.size(), -1
                        )
                        .sum(dim=-1)
                        .float()
                    )
                    # Compute balancedness ratio:
                    # for each layer:
                    #   (mean load across ranks) / (max load across ranks)
                    avg_tokens_tensor, max_tokens_tensor = _compute_eplb_load_stats(
                        num_tokens_per_rank
                    )
                    # A blocking read here would wait for the queued forward.
                    tokens = torch.stack([avg_tokens_tensor, max_tokens_tensor])
                    stats.append(
                        (
                            eplb_model_state.model_name,
                            tokens.to("cpu", non_blocking=True),
                        )
                    )
                copied = torch.Event()
                copied.record()
                self._pending_balancedness.append(
                    (
                        self.expert_rearrangement_step,
                        self.expert_rearrangement_step_interval
                        - self.expert_rearrangement_step,
                        copied,
                        stats,
                    )
                )

        # Update the expert load sliding window
        if not is_dummy:
            trace = self._trace
            if trace is not None and trace.failed:
                self._drop_trace()
                trace = None
            should_record = self._window_records_current_step(log_stats=log_stats)
            if trace is not None:
                trace.accumulate(list(self.model_states.values()), should_record)
            for eplb_model_state in self.model_states.values():
                if should_record:
                    eplb_model_state.expert_load_window[
                        self.expert_load_window_step
                    ].copy_(eplb_model_state.expert_load_pass)
                    eplb_model_state.expert_load_pass.zero_()

            if should_record:
                self._window_token_weights[self.expert_load_window_step] = (
                    self._step_token_weight
                )
                self.window_fill = min(
                    self.window_fill + 1, self.expert_load_window_size
                )
                self.expert_load_window_step += 1
                if self.expert_load_window_step >= self.expert_load_window_size:
                    self.expert_load_window_step = 0

            if trace is not None:
                trace.end_step(list(self.model_states.values()))

        # Step the expert rearrangement step
        # Note that even if this is a dummy step, we still increment the
        # rearrangement step and perform rearrangement to ensure all ranks are
        # performing collective communication.
        self.expert_rearrangement_step += 1
        self._time_step()

        # Models whose pending layer was committed (or given up) at this step:
        # the same on every rank, since every rank takes the same branch.
        settled: set[int] = set()
        if self.is_async:
            # Run _move_to_workspace if all ranks have finished transferring the
            # new weights to the intermediate buffer
            for eplb_model_state in self.model_states.values():
                pending = eplb_model_state.pending_result
                self._consume_async_result(eplb_model_state, ep_group.rank())
                if pending is not None and eplb_model_state.pending_result is None:
                    settled.add(id(eplb_model_state))
                    if not eplb_model_state.rebalanced:
                        self._after_async_cycle(eplb_model_state, ep_group.rank())
        self._step_residency_masks(settled)

        interval = self.expert_rearrangement_step_interval
        if self.rearrange_requested or (
            interval > 0 and self.expert_rearrangement_step >= interval
        ):
            if self.is_async and any(
                eplb_model_state.rebalanced
                for eplb_model_state in self.model_states.values()
            ):
                # Still performing asynchronous rearrangement; update
                # should_record (step > step_interval, so always True) and
                # bail out before the step counter is reset.
                self._update_layer_should_record(log_stats=log_stats)
                return
            self.expert_rearrangement_step = 0
            requested, self.rearrange_requested = self.rearrange_requested, False
            if self.rearrangements_paused:
                self.paused_intervals += 1
            else:
                use_saved = self.next_uses_saved_load
                self.next_uses_saved_load = False
                self.window_fill = 0
                self.rearrange(
                    use_last_expert_load=use_saved,
                    periodic=not requested and not use_saved,
                )
        elif self._tier_pass_due():
            self.rearrange(tier_only=True)

        self._update_layer_should_record(log_stats=log_stats)

    def _time_step(self) -> None:
        """The serving step's period (time between two EPLB steps on this
        thread), into the move in flight or the moving mean outside moves."""
        now = time.perf_counter()
        last, self._last_step_time = self._last_step_time, now
        if last is None:
            return
        period = (now - last) * 1e3
        moving = [ms for ms in self.model_states.values() if ms.rebalanced]
        for ms in moving:
            cost = ms.move_cost
            cost["steps"] = cost.get("steps", 0.0) + 1
            cost["step_ms"] = cost.get("step_ms", 0.0) + period
        if not moving:
            idle = self._step_ms_idle
            self._step_ms_idle = period if idle is None else 0.9 * idle + 0.1 * period

    def _after_async_cycle(self, model_state: EplbModelState, ep_rank: int) -> None:
        """An async rearrangement or tier pass of `model_state` ended at this
        step (last layer committed, or abandoned): the same step on every
        rank."""
        stats = model_state.eplb_stats
        full = stats is not None and not stats.tier_only
        if full and self.parallel_config.eplb_config.interval_from_end:
            self.expert_rearrangement_step = 0
        if runtime_debug.on("eplb"):
            self._emit_move_cost(model_state, ep_rank, full)
        model_state.move_cost = {}

    def _emit_move_cost(
        self, model_state: EplbModelState, ep_rank: int, full: bool
    ) -> None:
        """One `[debug eplb] move` line per rank: where the cycle's time went.
        Worker: plan, the per-layer flag exchange, prepare (tier plan and store
        reads, or the transfer), and its waits on the device for the serving
        thread's commits (the last layer's wait may land after this line).
        Serving thread: the ready checks it ran at every step of the cycle
        (a host barrier between the ranks), the commits, the step period
        inside the cycle against outside."""
        c = model_state.move_cost
        steps = int(c.get("steps", 0))
        seconds = (
            time.perf_counter() - model_state.rearrangement_started
            if model_state.rearrangement_started
            else 0.0
        )
        idle = self._step_ms_idle
        runtime_debug.emit(
            "eplb",
            logger,
            "move (%s, EP rank %d): %s in %.2f s over %d steps; step period "
            "%.1f ms inside, %s outside. Worker: plan %.0f ms, flag exchange "
            "%.0f ms, prepare %.0f ms (%d units read, %.1f MiB), device wait "
            "%.0f ms . Serving thread: %d ready checks %.1f ms, commit "
            "%.1f ms at %d steps. %d experts changed, %.1f MiB copied to card "
            "rows; verify %.0f ms.",
            model_state.model_name,
            ep_rank,
            "rearrangement" if full else "tier pass",
            seconds,
            steps,
            c.get("step_ms", 0.0) / max(steps, 1),
            "n/a" if idle is None else f"{idle:.1f} ms",
            c.get("plan_ms", 0.0),
            c.get("flag_ms", 0.0),
            c.get("prepare_ms", 0.0),
            int(c.get("units", 0)),
            c.get("bytes", 0.0) / 2**20,
            c.get("device_wait_ms", 0.0),
            int(c.get("ready_checks", 0)),
            c.get("ready_wait_ms", 0.0),
            c.get("commit_ms", 0.0),
            int(c.get("commit_steps", 0)),
            int(c.get("changed", 0)),
            c.get("card_bytes", 0.0) / 2**20,
            c.get("verify_ms", 0.0),
        )

    def _window_imbalance(
        self, global_expert_load_windows: list[torch.Tensor]
    ) -> float:
        """The busiest EP rank's load over the mean rank load, by the current
        placement, from the all-reduced logical windows (the largest over the
        models). Identical on every rank: its inputs are."""
        num_ranks = get_ep_group().device_group.size()
        worst = 1.0
        with gpu_sync_allowed():
            for model_state, window in zip(
                self.model_states.values(), global_expert_load_windows
            ):
                load = window.float().cpu().numpy()
                phys2log = model_state.physical_to_logical_map.cpu().numpy()
                layers, num_physical = phys2log.shape
                if num_physical % num_ranks:
                    continue
                per_rank = np.zeros(num_ranks, dtype=np.float64)
                rank_of = np.repeat(np.arange(num_ranks), num_physical // num_ranks)
                for layer in range(layers):
                    slots = phys2log[layer]
                    held = slots >= 0
                    count = np.bincount(slots[held], minlength=load.shape[1])
                    share = np.zeros(num_physical, dtype=np.float64)
                    share[held] = load[layer][slots[held]] / np.maximum(
                        count[slots[held]], 1
                    )
                    per_rank += np.bincount(rank_of, weights=share, minlength=num_ranks)
                mean = float(per_rank.mean())
                if mean > 0:
                    worst = max(worst, float(per_rank.max()) / mean)
        return worst

    def _tier_pass_due(self) -> bool:
        """Whether this step starts a tier pass: every `tier_interval` steps
        (not while paused), or at once when a layer asks for one because a
        tier's capacity changed. Never while a rearrangement or a pass is in
        flight. The same on every rank: the counters and the settings are,
        and a capacity changes on every rank between the same two steps."""
        if not self.is_async or not any(
            s.movers is not None for s in self.model_states.values()
        ):
            return False
        interval = self.parallel_config.eplb_config.tier_interval
        wanted = any(
            getattr(m, "tier_pass_wanted", False)
            for s in self.model_states.values()
            if s.movers is not None
            for m in s.movers
            if m is not None
        )
        if interval <= 0 and not wanted:
            return False
        self.tier_step += 1
        if any(s.rebalanced for s in self.model_states.values()):
            return False
        due = wanted or (
            interval > 0
            and self.tier_step >= interval
            and not self.rearrangements_paused
        )
        if due:
            self.tier_step = 0
        return due

    def _add_residency_mask(self, model_state: EplbModelState) -> None:
        """Create the model's residency mask and hand each MoE layer's router
        its rows (`EPLBConfig.residency_mask`)."""
        config = self.parallel_config.eplb_config
        if config.residency_mask == "off":
            return
        model = model_state.model
        top_k = _router_top_k(model)
        mask = ResidencyMask(
            mode=config.residency_mask,
            num_layers=model.num_moe_layers,
            num_logical=model.num_logical_experts,
            top_k=top_k,
            share=config.residency_mask_share,
            interval=config.residency_mask_interval,
            window_steps=config.window_size,
            count_displaced=config.residency_mask_stats,
            device=self.device,
        )
        model_state.residency = mask
        handed = 0
        for idx, layer in enumerate(model.moe_layers):
            layer_state = getattr(layer, "eplb_state", None)
            if isinstance(layer_state, EplbLayerState):
                (
                    layer_state.residency_mask,
                    layer_state.residency_stats,
                    layer_state.residency_shadow_load,
                ) = mask.layer_tensors(idx)
                handed += 1
        logger.info(
            "EPLB residency mask (%s%s) on %d of %d MoE layers of %s: refreshed "
            "every %d steps over %d experts per layer, top_k %d, displaced "
            "calls %s.",
            config.residency_mask,
            f", share {config.residency_mask_share}"
            if config.residency_mask == "coldest"
            else "",
            handed,
            model.num_moe_layers,
            model_state.model_name,
            config.residency_mask_interval,
            model.num_logical_experts,
            top_k,
            "counted" if config.residency_mask_stats else "not counted",
        )

    def _step_residency_masks(self, settled: set[int]) -> None:
        """The step hook's part of the residency masks: refresh a model's mask
        every `residency_mask_interval` steps, at the first step, and, for
        "nonresident", at every step that committed an async layer (a commit
        may change what the tiers hold). Runs at the same step on every rank,
        between steps, on the stream the next forward is enqueued on."""
        for model_state in self.model_states.values():
            mask = model_state.residency
            if mask is None:
                continue
            mask.steps += 1
            due = mask.steps % mask.interval == 0
            if mask.mode == "nonresident":
                due = due or mask.steps == 1 or id(model_state) in settled
            if due:
                self._refresh_residency_mask(model_state, mask)

    def _refresh_residency_mask(
        self, model_state: EplbModelState, mask: ResidencyMask
    ) -> None:
        ep_group = get_ep_group().device_group
        if mask.mode == "coldest":
            load = self._allreduce_list([mask.take_interval_load()])[0]
            with gpu_sync_allowed():
                load_cpu = load.cpu().numpy()
            new = mask.choose_coldest(mask.add_interval_load(load_cpu))
        else:
            if mask.count_displaced:
                mask.take_interval_load()  # counted for the log only
            counts = self._allreduce_list([self._nonresident_counts(model_state)])[0]
            with gpu_sync_allowed():
                counts_cpu = counts.cpu().numpy()
                phys2log = model_state.physical_to_logical_map.cpu().numpy()
            new = logical_mask_from_residency(
                counts_cpu, phys2log, mask.num_logical, mask.top_k
            )
        changed = mask.write(new)
        if not mask.count_displaced:
            return
        with gpu_sync_allowed():
            # a copy: on a CPU device .cpu() would alias the zeroed counters
            stats = mask.stats.cpu().numpy().copy()
        mask.stats.zero_()
        line = mask.take_stats(stats)
        if line is not None and ep_group.rank() == 0:
            logger.info(
                "EPLB residency mask (%s, %s) at step %d: %s%s.",
                mask.mode,
                model_state.model_name,
                mask.steps,
                line,
                "; mask changed" if changed else "",
            )

    def _nonresident_counts(self, model_state: EplbModelState) -> torch.Tensor:
        """`[layers, physical]` int32 on the device: 1 where this rank owns a
        slot that its layer's mover reports held by no tier."""
        model = model_state.model
        counts = torch.zeros(
            (model.num_moe_layers, model.num_physical_experts), dtype=torch.int32
        )
        movers = model_state.movers or [None] * model.num_moe_layers
        for layer, mover in enumerate(movers):
            report = getattr(mover, "nonresident_physical", None)
            if report is None:
                continue
            ids = list(report())
            if ids:
                counts[layer, ids] = 1
        return counts.to(self.device)

    def _maybe_log_balancedness(self) -> None:
        pending = self._pending_balancedness
        while pending:
            step, steps_left, copied, stats = pending[0]
            if not copied.query():
                break
            pending.popleft()
            for model_name, tokens in stats:
                avg_tokens, max_tokens = tokens.tolist()
                balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0
                logger.info(
                    "EPLB step: %d for model %s: avg_tokens=%.2f, "
                    "max_tokens=%d, balancedness=%.4f, "
                    "steps until the next rearrangement: %d",
                    step,
                    model_name,
                    avg_tokens,
                    max_tokens,
                    balancedness,
                    steps_left,
                )

    def _should_record_current_step(self, log_stats: bool = False) -> bool:
        """Return whether the router kernel should count this step's load: the
        window needs it (`_window_records_current_step`), or the trace samples
        the step."""
        if self._window_records_current_step(log_stats):
            return True
        return self._trace is not None and self._trace.records_next_step()

    def _drop_trace(self) -> None:
        """The trace cannot be written: stop it, and leave the load pass as it
        is without a trace (zero between window steps)."""
        logger.warning("EPLB: the expert load trace is off from here.")
        self._trace = None
        for model_state in self.model_states.values():
            model_state.map_change_hook = None
            model_state.expert_load_pass.zero_()
        self._should_record_value = None

    def _window_records_current_step(self, log_stats: bool = False) -> bool:
        """Return whether the sliding window takes this step's load.

        Recording is enabled when we are close to either:
        1) The next rearrangement step, so the sliding window is ready.
        2) The next balancedness logging step, when log_stats is enabled.
        """
        steps_remaining = (
            self.expert_rearrangement_step_interval - self.expert_rearrangement_step
        )
        should_record_for_rearrange = (
            steps_remaining <= self.expert_load_window_size
            or self.parallel_config.eplb_config.tier_interval > 0
        )

        if not log_stats:
            return should_record_for_rearrange

        log_interval = self.parallel_config.eplb_config.log_balancedness_interval
        steps_until_next_log = (
            log_interval - (self.expert_rearrangement_step % log_interval)
        ) % log_interval
        should_record_for_log = steps_until_next_log <= self.expert_load_window_size
        return should_record_for_rearrange or should_record_for_log

    def _update_layer_should_record(self, log_stats: bool = False) -> None:
        """Update the shared `should_record_tensor` for all layers."""
        if self._trace is not None and self._window_records_current_step(log_stats):
            # The window starts to copy the pass next step: it must not hold
            # steps the trace let accumulate.
            self._trace.fold_pending(list(self.model_states.values()))
        if self.should_record_tensor is None:
            return
        should_record = self._should_record_current_step(log_stats=log_stats)
        # The value only flips at window edges; skip the per-step fill kernel.
        if should_record != self._should_record_value:
            self.should_record_tensor.fill_(should_record)
            self._should_record_value = should_record

    def _propagate_shared_tensors(
        self,
        model: "MixtureOfExperts",  # type: ignore[name-defined]
        num_unpadded_tokens_tensors: list[torch.Tensor],
    ) -> None:
        """Propagate shared tensors to every `EplbLayerState`.

        Allocates `should_record_tensor` on the first call and then
        assigns both it and `num_unpadded_tokens_tensors` to every
        MoE layer's `EplbLayerState`.  All layers reference the
        **same** objects so a single update is visible everywhere.

        Must be called after `model.set_eplb_state` so that each
        layer's `eplb_state` is already populated.
        """
        layer_states = [
            layer.eplb_state
            for layer in model.moe_layers
            if hasattr(layer, "eplb_state")
            and isinstance(layer.eplb_state, EplbLayerState)
        ]

        if self.should_record_tensor is None and layer_states:
            self.should_record_tensor = torch.ones(
                (), dtype=torch.bool, device=self.device
            )

        for ls in layer_states:
            if ls is not None:
                ls.should_record_tensor = self.should_record_tensor
                ls.num_unpadded_tokens_tensors = num_unpadded_tokens_tensors

    def rearrange(
        self,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
        use_last_expert_load: bool = False,
        blocking: bool = False,
        tier_only: bool = False,
        periodic: bool = False,
    ) -> torch.Tensor | None:
        """Rearrange the experts according to the current load.

        Args:
            is_profile (bool): If `True`, perform a dummy rearrangement.
                This is used in `profile_run` to reserve enough memory,
                no memory movement will be performed. Default is False.
            rank_mapping (dict[int, int] | None): The rank mapping
                when scaling is done in EEP.
            use_last_expert_load: Use the last reshuffle's logical-expert totals.
            blocking: Move the weights before returning, also in async mode.
            tier_only: A tier pass (`EPLBConfig.tier_interval`): no expert
                changes rank; the layers that place their own experts cut
                their tiers from the window. Async, those models only.
            periodic: Started by the interval (not asked for): it is skipped
                when the window's imbalance is below
                `EPLBConfig.min_imbalance`.

        """
        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()

        start_event = None
        end_event = None
        is_main_rank = ep_rank == 0
        sync = not self.is_async or is_profile or blocking
        if tier_only and sync:
            return None  # the layers that cut tiers move asynchronously only
        if is_main_rank and tier_only:
            logger.debug("EPLB tier pass (async mode)...")
        elif is_main_rank:
            if sync:
                start_event = torch.Event(enable_timing=True)
                end_event = torch.Event(enable_timing=True)
                start_event.record()
            logger.info(
                "Rearranging experts %s %s...",
                "sync mode" if sync else "(async mode)",
                "(profile)" if is_profile else "",
            )

        # Every allocation a rearrangement makes before it moves a weight is
        # made in two phases, each closed by an agreement of the EP ranks on
        # the host (`_skip_rearrangement`): if any rank could not allocate,
        # every rank skips this rearrangement, so that no rank enters a
        # collective the others left and the placement stays the same on
        # every rank. The device may be full while serving (a KV cache grow or
        # another holder can take it to the last byte), and a rearrangement
        # is optional work. A profile run keeps failing loudly.
        guard = not is_profile
        self.last_rearrange_changed = False

        # Phase 1, on this rank only: the logical load windows and what the
        # all-reduce sends.
        error: torch.OutOfMemoryError | None = None
        global_expert_load_windows: list[torch.Tensor] = []
        reduce_input: torch.Tensor | None = None
        started = time.perf_counter()
        try:
            if guard:
                fault_injection.maybe_fail("load")
            global_expert_load_windows = self._logical_load_windows(
                use_last_expert_load, is_main_rank
            )
            if len(global_expert_load_windows) > 1:
                reduce_input = self._concat_for_allreduce(global_expert_load_windows)
        except torch.OutOfMemoryError as e:
            if not guard:
                raise
            error = e
        if guard and self._skip_rearrangement("computing the expert load", error):
            return None
        # Perform all-reduce to get the expert load across all ranks for each model
        global_expert_load_windows = self._allreduce_prepared(
            global_expert_load_windows, reduce_input
        )
        if (
            self.parallel_config.eplb_config.state_path is not None
            and is_main_rank
            and not is_profile
            and not use_last_expert_load
            and not tier_only
        ):
            self._save_expert_load(global_expert_load_windows)
        min_imbalance = self.parallel_config.eplb_config.min_imbalance
        if periodic and min_imbalance > 0 and not (tier_only or is_profile):
            imbalance = self._window_imbalance(global_expert_load_windows)
            self.last_imbalance = imbalance
            if imbalance < 1.0 + min_imbalance:
                self.low_imbalance_skips += 1
                if is_main_rank and runtime_debug.on("eplb"):
                    runtime_debug.emit(
                        "eplb",
                        logger,
                        "no rearrangement: busiest rank %.4f x the mean rank "
                        "load, below 1 + min_imbalance %g (%d skipped so far)",
                        imbalance,
                        min_imbalance,
                        self.low_imbalance_skips,
                    )
                return None

        # TODO(bowen): Treat differently for prefill and decode nodes
        eplb_model_state = next(iter(self.model_states.values()))
        model = eplb_model_state.model
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups

        if rank_mapping is not None and len(rank_mapping) == ep_group.size():
            # NOTE(yongji): scale down, we need to rebalance the experts on
            # remaining GPUs, transfer the experts while we haven't shutdown
            # the GPUs to be released.
            coordinator = get_ep_group()
            assert isinstance(coordinator, StatelessGroupCoordinator)
            tcp_store_group = coordinator.tcp_store_group
            num_nodes = _node_count_with_rank_mapping(tcp_store_group, rank_mapping)
            num_gpus = sum(new_rank != -1 for new_rank in rank_mapping.values())
            num_replicas = (
                num_replicas // ep_group.size() * num_gpus
            )  # handle num replicas change
        else:
            num_nodes = get_node_count()
            num_gpus = ep_group.size()

        if num_gpus % num_nodes != 0:
            num_nodes = 1
            logger.warning_once(
                f"num_gpus % num_nodes != 0, "
                "not using hierarchical rearrangement algorithm.\n"
                f"{num_gpus=}, {num_nodes=}"
            )

        # Phase 2, on this rank only: every model's plan (sync: the new maps,
        # computed on the host; async: the snapshot the worker reads), before
        # any weight moves or any state changes.
        # (model state, its load window, sync: (old map, new map, skip) or
        # None; async: the window's snapshot)
        plans: list[tuple[EplbModelState, torch.Tensor, Any]] = []
        error = None
        try:
            if guard:
                fault_injection.maybe_fail("plan")
            for eplb_model_state, global_expert_load_window in zip(
                self.model_states.values(), global_expert_load_windows
            ):
                if tier_only and eplb_model_state.movers is None:
                    continue  # its experts sit in one tier
                if sync and eplb_model_state.movers is not None:
                    if is_profile or use_last_expert_load:
                        # nothing to reserve; the restore fills these models itself
                        plans.append(
                            (eplb_model_state, global_expert_load_window, None)
                        )
                        continue
                    raise NotImplementedError(
                        "EPLB: a model whose layers fill their slots from the expert "
                        "store is rearranged asynchronously only (the boot restore "
                        "fills the slots directly)"
                    )
                plans.append(
                    (
                        eplb_model_state,
                        global_expert_load_window,
                        self._plan_sync(
                            eplb_model_state,
                            global_expert_load_window,
                            num_replicas,
                            num_groups,
                            num_nodes,
                            num_gpus,
                            is_profile,
                            rank_mapping,
                            is_main_rank,
                        )
                        if sync
                        # We copy the tensor to snapshot the
                        # global_expert_load_window on the main thread so that
                        # async worker can access it safely while the main
                        # thread is running.
                        else global_expert_load_window.clone(),
                    )
                )
        except torch.OutOfMemoryError as e:
            if not guard:
                raise
            error = e
        if guard and self._skip_rearrangement("planning the moves", error):
            return None

        # Phase 3: the moves. Collectives with allocations of their own; a
        # failure here is not caught (see `_skip_rearrangement`).
        if tier_only:
            self.tier_passes += 1
        elif not is_profile:
            self.rearrangements += 1
        for eplb_model_state, global_expert_load_window, plan in plans:
            if not is_profile and not tier_only:
                eplb_model_state.last_expert_load = global_expert_load_window
            if sync:
                if plan is None:
                    continue
                old_physical_to_logical_map, new_physical_to_logical_map = (
                    plan[0],
                    plan[1],
                )
                skip_rearrange = bool(plan[2])
                if not skip_rearrange and not is_profile:
                    self.last_rearrange_changed |= bool(
                        (old_physical_to_logical_map != new_physical_to_logical_map)
                        .any()
                        .item()
                    )
                if not skip_rearrange:
                    # Update expert weights
                    rearrange_expert_weights_inplace(
                        eplb_model_state.physical_to_logical_map,
                        new_physical_to_logical_map,
                        eplb_model_state.model.expert_weights,
                        eplb_model_state.expert_buffer,
                        ep_group,
                        eplb_model_state.communicator,
                        is_profile,
                        rank_mapping,
                    )

                    if not is_profile:
                        _commit_eplb_maps(
                            eplb_model_state,
                            new_physical_to_logical_map=new_physical_to_logical_map,
                        )

                if is_main_rank:
                    assert start_event is not None
                    assert end_event is not None
                    end_event.record()
                    end_event.synchronize()
                    gpu_elapsed = start_event.elapsed_time(end_event) / 1000.0
                    logger.info(
                        "Rearranged experts %s in %.2f s.",
                        " (profile) " if is_profile else " ",
                        gpu_elapsed,
                    )
            else:
                eplb_model_state.eplb_stats = EplbStats(
                    global_expert_load_window=plan,
                    num_replicas=num_replicas,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                    num_gpus=num_gpus,
                    tier_only=tier_only,
                )
                eplb_model_state.move_cost = {}
                eplb_model_state.rebalanced = True
                eplb_model_state.rearrangement_started = started
        # Signal async thread to start transferring layers
        if not sync:
            self.rearrange_event.record()
        elif not is_profile:
            self.last_rearrange_seconds = time.perf_counter() - started
        return None

    def _logical_load_windows(
        self, use_last_expert_load: bool, is_main_rank: bool
    ) -> list[torch.Tensor]:
        """Each model's load window over its logical experts, summed over the
        window (this rank's counts; rank 0's saved ones with
        use_last_expert_load)."""
        # Map the physical expert load to global logical experts
        global_expert_load_windows = []
        for eplb_model_state in self.model_states.values():
            if use_last_expert_load:
                # Distribute rank 0's saved counts through the existing all-reduce.
                saved_load = eplb_model_state.last_expert_load
                global_expert_load_windows.append(
                    saved_load.clone() if is_main_rank else torch.zeros_like(saved_load)
                )
                continue
            expert_load_window = eplb_model_state.expert_load_window
            physical_to_logical = eplb_model_state.physical_to_logical_map
            invalid_idx = eplb_model_state.model.num_logical_experts
            logical_expert_load_window = torch.zeros(
                self.expert_load_window_size,
                eplb_model_state.model.num_moe_layers,
                invalid_idx + 1,
                dtype=eplb_model_state.expert_load_window.dtype,
                device=eplb_model_state.expert_load_window.device,
            )
            logical_expert_load_window.scatter_add_(
                dim=-1,
                index=physical_to_logical.masked_fill(
                    physical_to_logical < 0, invalid_idx
                )
                .unsqueeze(0)
                .expand_as(expert_load_window)
                .long(),
                src=expert_load_window,
            )

            if self.parallel_config.eplb_config.weights_tokens:
                # Each window step weighed by its decode/prefill mix.
                step_weights = torch.tensor(
                    self._window_token_weights,
                    dtype=torch.float32,
                    device=logical_expert_load_window.device,
                )
                global_expert_load_window = (
                    logical_expert_load_window[..., :-1].float()
                    * step_weights[:, None, None]
                ).sum(dim=0)
            else:
                global_expert_load_window = logical_expert_load_window[..., :-1].sum(
                    dim=0
                )
            global_expert_load_windows.append(global_expert_load_window)
        return global_expert_load_windows

    def _plan_sync(
        self,
        eplb_model_state: EplbModelState,
        global_expert_load_window: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_gpus: int,
        is_profile: bool,
        rank_mapping: dict[int, int] | None,
        is_main_rank: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """A synchronous rearrangement's plan for one model: the current and
        the new physical-to-logical map (host) and whether to skip the move."""
        ep_group = get_ep_group().device_group
        # Get new expert mappings for the model. The policy runs on the
        # host, so the load window and current map have to come back.
        with gpu_sync_allowed():
            load_window_cpu = global_expert_load_window.cpu()
            old_physical_to_logical_map = eplb_model_state.physical_to_logical_map.cpu()
        new_physical_to_logical_map = self.policy.rebalance_experts(
            load_window_cpu,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
            old_physical_to_logical_map,
        )
        new_physical_to_logical_map = self.order_tiers(
            eplb_model_state,
            new_physical_to_logical_map,
            load_window_cpu,
            num_gpus,
        )
        if runtime_debug.on("eplb"):
            changed = old_physical_to_logical_map != new_physical_to_logical_map
            runtime_debug.emit(
                "eplb",
                logger,
                "plan (%s): %d of %d slots change, in %d of %d layers; load "
                "window total %d tokens",
                eplb_model_state.model_name,
                int(changed.sum()),
                changed.numel(),
                int(changed.any(dim=1).sum()),
                changed.shape[0],
                int(load_window_cpu.sum()),
            )

        skip_rearrange = False
        if (
            current_platform.is_rocm()
            and not is_profile
            and rank_mapping is None
            and bool((old_physical_to_logical_map >= 0).all())
        ):
            logical_loads = load_window_cpu.float()
            ep_size = ep_group.size()

            def rank_load_imbalance(
                mapping: torch.Tensor,
                logical_loads: torch.Tensor = logical_loads,
                ep_size: int = ep_size,
            ) -> float:
                mapping = mapping.to(
                    device=logical_loads.device,
                    dtype=torch.long,
                )
                replica_counts = torch.zeros_like(logical_loads)
                replica_counts.scatter_add_(
                    dim=1,
                    index=mapping,
                    src=torch.ones_like(mapping, dtype=logical_loads.dtype),
                )
                loads_per_replica = torch.gather(
                    logical_loads / replica_counts.clamp_min(1),
                    dim=1,
                    index=mapping,
                )
                loads_per_rank = loads_per_replica.reshape(
                    logical_loads.shape[0], ep_size, -1
                ).sum(dim=(0, 2))
                mean_load = loads_per_rank.mean()
                if mean_load == 0:
                    return 1.0
                return (loads_per_rank.max() / mean_load).item()

            current_imbalance = rank_load_imbalance(old_physical_to_logical_map)
            proposed_imbalance = rank_load_imbalance(new_physical_to_logical_map)
            relative_improvement = (
                current_imbalance - proposed_imbalance
            ) / current_imbalance
            skip_rearrange = relative_improvement < 0.05
            if skip_rearrange and is_main_rank:
                logger.info(
                    "[EPLB] Skip rearrange: imbalance %.4f -> %.4f (no material gain)",
                    current_imbalance,
                    proposed_imbalance,
                )
        return old_physical_to_logical_map, new_physical_to_logical_map, skip_rearrange

    def _skip_rearrangement(
        self, stage: str, error: "torch.OutOfMemoryError | None"
    ) -> bool:
        """The EP ranks agree, on the host (no device memory needed), whether
        any of them failed to allocate while `stage`; if so every rank skips
        this rearrangement and the placement stays as it is on every rank.
        One WARNING on EP rank 0 with the counts; a failing rank other than 0
        logs its own error at INFO."""
        ep_group = get_ep_group()
        num_failed = int(error is not None)
        if ep_group.device_group.size() > 1:
            flags = torch.tensor([num_failed], dtype=torch.int32, device="cpu")
            ep_group.cpu_all_reduce_(flags)
            num_failed = int(flags.item())
        if num_failed == 0:
            return False
        self.skipped_rearrangements += 1
        rank = ep_group.device_group.rank()
        if rank == 0:
            logger.warning(
                "EPLB: skipping this rearrangement: %d of %d EP ranks could not "
                "allocate memory while %s%s. Placement unchanged on every rank; "
                "the next rearrangement in %d steps (%d skipped so far).",
                num_failed,
                ep_group.device_group.size(),
                stage,
                "" if error is None else f" (rank 0: {str(error).splitlines()[0]})",
                self.expert_rearrangement_step_interval,
                self.skipped_rearrangements,
            )
        elif error is not None:
            logger.info(
                "EPLB (EP rank %d): could not allocate memory while %s: %s",
                rank,
                stage,
                str(error).splitlines()[0],
            )
        return True

    @staticmethod
    def _concat_for_allreduce(tensor_list: list[torch.Tensor]) -> torch.Tensor:
        """What `_allreduce_list` sends for several models, in one tensor."""
        assert all(t.dim() == 2 for t in tensor_list), "All tensors must be 2D."
        assert all(t.shape[1] == tensor_list[0].shape[1] for t in tensor_list), (
            "All tensors must have the same shape[1]."
        )
        return torch.cat(tensor_list, dim=0)

    def _allreduce_prepared(
        self, tensor_list: list[torch.Tensor], concat: torch.Tensor | None
    ) -> list[torch.Tensor]:
        """`_allreduce_list` with the concatenation already made."""
        ep_group = get_ep_group().device_group
        if concat is None:
            assert len(tensor_list) == 1
            all_reduce(tensor_list[0], group=ep_group)
            return tensor_list
        all_reduce(concat, group=ep_group)
        all_reduce_list = []
        offset = 0
        for tensor in tensor_list:
            all_reduce_list.append(concat[offset : offset + tensor.shape[0], :])
            offset += tensor.shape[0]
        return all_reduce_list

    def start_async_loop(
        self,
        rank_mapping: dict[int, int] | None = None,
        is_profile: bool = False,
    ):
        if not self.is_async:
            return
        if self.async_worker is None:
            self.async_worker = start_async_worker(
                self,
                is_profile=is_profile,
            )

    def drain_async(self) -> None:
        """Drain in-flight async EPLB by consuming all remaining layer results.

        Each pending result is acknowledged (consumed_event recorded) so the
        async worker can proceed, but the transferred weights are intentionally
        NOT applied — a full rearrange is expected to follow.

        Ranks are kept in lockstep via _all_ranks_result_ready (all_reduce
        on the EP CPU group).  The async worker's coordinated-stop collectives
        use the separate EPLB group, so the two sets of collectives do not
        interfere.

        No-op when no async cycle is in progress (rebalanced=False).
        """
        if not self.is_async:
            return
        for model_key, ms in self.model_states.items():
            needs_drain = ms.rebalanced
            if needs_drain:
                logger.info(
                    "Draining async EPLB worker for model %s",
                    model_key,
                )
            while ms.rebalanced:
                if self._all_ranks_result_ready(ms):
                    result = ms.pending_result
                    assert result is not None
                    if result.layer_idx == ms.model.num_moe_layers - 1:
                        ms.rebalanced = False
                    ms.pending_result = None
                    if result.prepared is not None and ms.movers is not None:
                        mover = ms.movers[result.layer_idx]
                        assert mover is not None
                        mover.discard(result.prepared)
                    result.consumed_event.record()
                else:
                    time.sleep(0.001)
            if needs_drain:
                logger.info(
                    "Async EPLB worker drained for model %s",
                    model_key,
                )

    def scrub_expert_store(self, full: bool = True) -> list[dict]:
        """Run the movers' scrub now, whatever their debug setting says: for
        each model whose layers place their own experts, compare what this
        rank's kernels read with the experts' home, the device's tier maps
        with the placement and the placement with EPLB's map. One INFO line
        per model with the counts, an ERROR line naming the first mismatch.

        Call on the thread that enqueues the forward, between two steps (it
        blocks on device reads and waits for the commits queued before it).
        It reads mapped rows only, so a rearrangement being prepared in the
        background does not disturb it. Returns one summary per such model.
        """
        ep_rank = get_ep_group().device_group.rank()
        summaries = []
        for model_state in self.model_states.values():
            summary = _scrub_movers(
                model_state, ep_rank, full=full, when="on request", force=True
            )
            if summary is not None:
                summaries.append(summary)
        return summaries

    def _consume_async_result(self, model_state: EplbModelState, ep_rank: int) -> None:
        """The step hook's part of an async rearrangement, for one model: once
        every rank has the pending layer's result, commit it, or, if any
        rank's mover could not prepare it, leave the layer and the rest of
        the cycle as they are. The same branch on every rank at the same
        step."""
        # rebalanced must remain consistent amongst all ranks otherwise the
        # all_reduce in _all_ranks_result_ready will hang
        if not model_state.rebalanced:
            return
        started = time.perf_counter()
        ready = self._all_ranks_result_ready(model_state)
        cost = model_state.move_cost
        cost["ready_checks"] = cost.get("ready_checks", 0.0) + 1
        cost["ready_wait_ms"] = (
            cost.get("ready_wait_ms", 0.0) + (time.perf_counter() - started) * 1e3
        )
        if not ready:
            return
        if model_state.abandon_votes:
            _abandon_layer(model_state=model_state, ep_rank=ep_rank)
        else:
            _move_to_workspace(model_state=model_state, ep_rank=ep_rank)

    def _all_ranks_result_ready(self, model_state: EplbModelState) -> bool:
        result = model_state.pending_result
        flags = torch.tensor(
            (
                int(self.async_worker_error is not None),
                int(result is None),
                int(result is not None and result.abandoned),
            ),
            dtype=torch.int32,
            device="cpu",
        )
        # cpu_all_reduce_ only sums, so count failed, pending and abandoning
        # ranks.
        get_ep_group().cpu_all_reduce_(flags)
        num_failed, num_pending, num_abandoning = flags.tolist()
        if num_failed:
            raise RuntimeError(
                "Async EPLB worker failed on at least one rank; see the worker logs."
            ) from self.async_worker_error
        if num_pending == 0:
            # Only meaningful once every rank has its result.
            model_state.abandon_votes = num_abandoning
        return num_pending == 0

    def _allreduce_list(self, tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """All-reduce a list of tensors."""
        ep_group = get_ep_group().device_group
        if len(tensor_list) == 1:
            all_reduce(tensor_list[0], group=ep_group)
            return tensor_list
        assert all(t.dim() == 2 for t in tensor_list), "All tensors must be 2D."
        assert all(t.shape[1] == tensor_list[0].shape[1] for t in tensor_list), (
            "All tensors must have the same shape[1]."
        )
        # Concatenate, all_reduce, then unpack to original shapes.
        # We assume all tensors are 2D and shape[1] (num_physical_experts)
        # is the same across all models.
        shapes = [t.shape for t in tensor_list]
        concat_tensor = torch.cat(tensor_list, dim=0)

        all_reduce(concat_tensor, group=ep_group)

        all_reduce_list = []
        offset = 0
        for shape in shapes:
            all_reduce_list.append(concat_tensor[offset : offset + shape[0], :])
            offset += shape[0]
        return all_reduce_list

    def _sync_load_pass(self) -> list[torch.Tensor]:
        """Sync the expert load pass across all ranks for log stats.
        Doesn't update the expert load pass in eplb_model_state.
        """
        load_pass_list = []
        for eplb_model_state in self.model_states.values():
            load_pass_list.append(eplb_model_state.expert_load_pass.clone())
        return self._allreduce_list(load_pass_list)

    @classmethod
    def from_mapping(
        cls,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        device: torch.device,
        parallel_config: ParallelConfig,
        expanded_physical_to_logical: torch.Tensor,
    ) -> "EplbState":
        eplb_state = cls(
            parallel_config=parallel_config,
            device=device,
        )
        eplb_state.add_model(
            model=model,
            model_config=model_config,
        )
        eplb_state.update_mapping(
            model_config,
            expanded_physical_to_logical,
        )

        return eplb_state

    def update_mapping(
        self,
        model_config: ModelConfig,
        expanded_physical_to_logical: torch.Tensor,
    ) -> None:
        eplb_model_state = self.model_states[hash_config_cached(model_config)]
        if eplb_model_state.map_change_hook is not None:
            eplb_model_state.map_change_hook(eplb_model_state)
        eplb_model_state.physical_to_logical_map_buffer.copy_(
            expanded_physical_to_logical
        )
        eplb_model_state.expert_load_pass.zero_()
        physical_to_logical_map = eplb_model_state.physical_to_logical_map

        (logical_to_physical_map_cpu, logical_replica_count_cpu) = compute_logical_maps(
            physical_to_logical_map.cpu(),
            eplb_model_state.model.num_logical_experts,
        )

        max_num_replicas = eplb_model_state.logical_to_physical_map.shape[-1]
        num_replicas = logical_to_physical_map_cpu.shape[-1]
        logical_to_physical_map = torch.nn.functional.pad(
            logical_to_physical_map_cpu,
            (
                0,
                max_num_replicas - num_replicas,
            ),
            value=-1,
        ).to(self.device)
        logical_replica_count = logical_replica_count_cpu.to(self.device)

        eplb_model_state.logical_to_physical_map.copy_(logical_to_physical_map)
        eplb_model_state.logical_replica_count.copy_(logical_replica_count)

    def reconfigure_physical_expert_slots(
        self,
        model_config: ModelConfig,
        num_physical_experts: int,
    ) -> None:
        """Replace physical_to_logical_map and expert_load_pass with views
        covering the new active size.
        """
        model_state = self.model_states[hash_config_cached(model_config)]
        old_num_physical_experts = model_state.model.num_physical_experts
        first_slot, last_slot = sorted((old_num_physical_experts, num_physical_experts))
        physical_to_logical_map_buffer = model_state.physical_to_logical_map_buffer
        expert_load_pass_buffer = model_state.expert_load_pass_buffer
        assert last_slot <= physical_to_logical_map_buffer.shape[1]
        expert_slots = slice(first_slot, last_slot)
        physical_to_logical_map_buffer[:, expert_slots].fill_(-1)
        expert_load_pass_buffer[:, expert_slots].zero_()
        model_state.physical_to_logical_map = physical_to_logical_map_buffer[
            :, :num_physical_experts
        ]
        model_state.expert_load_pass = expert_load_pass_buffer[:, :num_physical_experts]

        pad_size = num_physical_experts - model_state.expert_load_window.shape[-1]
        model_state.expert_load_window = torch.nn.functional.pad(
            model_state.expert_load_window, (0, pad_size)
        )

    def create_communicator(
        self, model_config: ModelConfig, group_coordinator: GroupCoordinator
    ) -> EplbCommunicator:
        model_state = self.model_states[hash_config_cached(model_config)]
        backend = self.parallel_config.eplb_config.communicator
        assert backend is not None
        return create_eplb_communicator(
            group_coordinator,
            backend,
            model_state.model.expert_weights,
            model_state.expert_buffer,
        )

    def update_communicator(
        self,
        model_config: ModelConfig,
        communicator: EplbCommunicator,
    ) -> None:
        self.model_states[hash_config_cached(model_config)].communicator = communicator


@dataclass
class EplbLayerState:
    """Runtime EPLB data stored in the MoE layer."""

    expert_load_view: torch.Tensor | None = None
    logical_to_physical_map: torch.Tensor | None = None
    logical_replica_count: torch.Tensor | None = None
    should_record_tensor: torch.Tensor | None = None
    """
    Shared scalar bool tensor controlling whether to accumulate expert load
    metrics during this forward pass.  All layers reference the **same**
    tensor object, which is owned and updated by `EplbState`.

    Set to `False` for the first `step_interval - window_size` steps of
    each rearrangement period: those steps would be overwritten in the
    sliding window before the next rearrangement, so recording them wastes
    GPU work.
    """
    num_unpadded_tokens_tensors: list[torch.Tensor] | None = None
    """
    Reference to the parent `EplbModelState`'s tensor list so the
    router can read the correct per-[u]batch unpadded token count.
    """
    residency_mask: torch.Tensor | None = None
    """`[logical experts]` bool, this layer's row of the model's residency
    mask (`EPLBConfig.residency_mask`); None when the mask is off."""
    residency_stats: torch.Tensor | None = None
    """`[3]` float32: displaced calls, routed calls, displaced weight, added
    to by the router; None when not counted."""
    residency_shadow_load: torch.Tensor | None = None
    """`[logical experts]` float32: top-k calls of the unmasked logits per
    expert, added to by the router; None when not counted."""

    def set_layer_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]


def _router_top_k(model: MixtureOfExperts) -> int:
    """The router's top_k of the model's MoE layers."""
    for layer in model.moe_layers:
        top_k = getattr(getattr(layer, "router", None), "top_k", None)
        if top_k:
            return int(top_k)
    raise ValueError(
        "EPLB residency mask: no MoE layer of the model exposes its router's top_k"
    )


def _node_count_with_rank_mapping(
    pg: ProcessGroup | StatelessProcessGroup,
    rank_mapping: dict[int, int],
) -> int:
    if isinstance(pg, ProcessGroup):
        world_size = torch.distributed.get_world_size(group=pg)
    else:
        world_size = pg.world_size

    if world_size == 1:
        return 1

    # Build node assignment map
    node_assignment = [0] * world_size  # rank -> node_id
    next_node_id = 0

    for current_rank in range(world_size):
        if node_assignment[current_rank] != 0:
            continue  # Already assigned to a node

        assert current_rank in rank_mapping
        if rank_mapping[current_rank] == -1:
            continue  # Pending shutdown

        # Assign current rank to a new node
        next_node_id += 1
        node_assignment[current_rank] = next_node_id

        # Find all ranks on the same node as current_rank
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id

    return next_node_id


def compute_logical_maps(
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive logical_to_physical_map and logical_replica_count from
    physical_to_logical_map.

    Args:
        physical_to_logical_map: [num_layers, num_physical_experts], logical
            expert index for each physical expert slot
        num_logical_experts: total number of logical experts

    Returns:
        logical_to_physical_map: [num_layers, num_logical_experts, max_replicas],
            physical slots per logical expert; -1 where unused
        logical_replica_count: [num_layers, num_logical_experts], number of
            physical replicas per logical expert

    """
    device = physical_to_logical_map.device
    assert physical_to_logical_map.device.type == "cpu"

    dtype = physical_to_logical_map.dtype

    # If computing maps for a single layer, unsqueeze a single element layer dimension
    per_layer = physical_to_logical_map.dim() == 1
    physical_to_logical_map_view = physical_to_logical_map
    if per_layer:
        physical_to_logical_map_view = physical_to_logical_map.unsqueeze(0)
    assert len(physical_to_logical_map_view.shape) == 2
    num_layers, num_physical = physical_to_logical_map_view.shape

    valid_mask = physical_to_logical_map_view >= 0
    logical_replica_count = torch.zeros(
        num_layers,
        num_logical_experts,
        dtype=dtype,
        device=device,
    )
    logical_replica_count.scatter_add_(
        1,
        physical_to_logical_map_view.clamp(min=0),
        valid_mask.to(dtype),
    )

    max_replicas = int(logical_replica_count.max().item())
    logical_to_physical_map_out = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=dtype,
        device=device,
    )

    # Stable sorting preserves physical-slot order within each logical expert.
    logical_ids, physical_slots = torch.sort(
        physical_to_logical_map_view, dim=1, stable=True
    )
    valid_layers, sorted_positions = (logical_ids >= 0).nonzero(as_tuple=True)
    valid_experts = logical_ids[valid_layers, sorted_positions]
    expert_offsets = logical_replica_count.cumsum(dim=1) - logical_replica_count
    num_inactive = (~valid_mask).sum(dim=1)
    replica_indices = (
        sorted_positions
        - num_inactive[valid_layers]
        - expert_offsets[valid_layers, valid_experts]
    )
    logical_to_physical_map_out[valid_layers, valid_experts, replica_indices] = (
        physical_slots[valid_layers, sorted_positions].to(dtype)
    )

    # If computing maps for a single layer, squeeze out the extra layer dimension
    # before returning
    if per_layer:
        return logical_to_physical_map_out.squeeze(0), logical_replica_count.squeeze(0)
    return logical_to_physical_map_out, logical_replica_count


def _pad_out_tensor(src: torch.Tensor, dst: torch.Tensor) -> None:
    padded = torch.full(dst.shape, -1, dtype=dst.dtype, pin_memory=PIN_MEMORY)
    padded[..., : src.shape[-1]] = src
    dst.copy_(padded, non_blocking=True)


def _commit_eplb_maps_for_layer(
    model_state: EplbModelState,
    new_physical_to_logical_map: torch.Tensor,
    layer: int,
) -> None:
    """Per-layer version of _commit_eplb_maps that's used by the sync portion of EPLB
    when running async EPLB. Copies all of the new_* maps into model_state. After this
    function completes, the new mappings will become the current mappings and will be
    visible to the model.
    """
    if model_state.map_change_hook is not None:
        model_state.map_change_hook(model_state)
    if runtime_debug.on("eplb"):
        runtime_debug.emit(
            "eplb", logger, "commit (%s): layer %d", model_state.model_name, layer
        )
    # Commit physical_to_logical_map
    src = new_physical_to_logical_map
    dst = model_state.physical_to_logical_map[layer]
    assert src.shape == dst.shape, (
        "The number of physical experts must stay the same while running Async EPLB. "
        f"Current number of physical experts: {dst.shape[0]}. New number of physical "
        f"experts {src.shape[0]}."
    )
    if PIN_MEMORY and src.is_cpu:
        src = src.new_empty(src.shape, pin_memory=True).copy_(src)
    dst.copy_(src, non_blocking=True)

    num_logical_experts = model_state.logical_to_physical_map.shape[1]
    new_logical, new_replica_count = compute_logical_maps(src, num_logical_experts)
    # Commit logical_to_physical_map
    _pad_out_tensor(src=new_logical, dst=model_state.logical_to_physical_map[layer])

    # Commit logical_replica_count
    src = new_replica_count
    dst = model_state.logical_replica_count[layer]
    assert src.shape == dst.shape
    if PIN_MEMORY:
        src = src.pin_memory()
    dst.copy_(src, non_blocking=True)


def _commit_eplb_maps(
    model_state: EplbModelState,
    new_physical_to_logical_map: torch.Tensor,
) -> None:
    """Copies all of the new_* maps into model_state. After this function completes,
    the new mappings will become the current mappings and will be visible to the
    model.
    """
    if model_state.map_change_hook is not None:
        model_state.map_change_hook(model_state)
    if runtime_debug.on("eplb"):
        runtime_debug.emit(
            "eplb",
            logger,
            "commit (%s): new physical-to-logical map %s",
            model_state.model_name,
            tuple(new_physical_to_logical_map.shape),
        )
    # Commit physical_to_logical_map
    src = new_physical_to_logical_map
    dst = model_state.physical_to_logical_map

    if PIN_MEMORY and src.is_cpu:
        src = src.new_empty(src.shape, pin_memory=True).copy_(src)
    # When the number of physical experts changes, refresh the active map view to
    # cover the new size before copying it.
    if src.shape[1] != dst.shape[1]:
        dst = model_state.physical_to_logical_map_buffer[:, : src.shape[1]]
        assert dst.shape == src.shape
        model_state.physical_to_logical_map = dst
    dst.copy_(src, non_blocking=True)

    num_logical_experts = model_state.logical_to_physical_map.shape[1]
    new_logical, new_replica_count = compute_logical_maps(src, num_logical_experts)
    # Commit logical_to_physical_map
    _pad_out_tensor(
        src=new_logical,
        dst=model_state.logical_to_physical_map,
    )

    # Commit logical_replica_count
    src = new_replica_count
    dst = model_state.logical_replica_count
    if PIN_MEMORY:
        src = src.pin_memory()
    dst.copy_(src, non_blocking=True)


def _scrub_movers(
    model_state: EplbModelState,
    ep_rank: int,
    *,
    full: bool,
    when: str,
    force: bool = False,
) -> dict | None:
    """The movers' scrub over every layer of a model whose layers place their
    own experts (None for other models, and unless `force`, when the layers'
    debug setting is off). Blocking; the thread that enqueues the forward."""
    movers = model_state.movers
    if not movers or movers[0] is None:
        return None
    if not force and not movers[0].scrub_every:
        return None
    started = time.perf_counter()
    with gpu_sync_allowed():
        phys2log = model_state.physical_to_logical_map.cpu().numpy()
        reports = []
        for layer, mover in enumerate(movers):
            assert mover is not None
            reports.append(mover.scrub(phys2log[layer], full))
    summary = {
        "model": model_state.model_name,
        "layers": len(reports),
        "slots": sum(r["slots"] for r in reports),
        "mismatches": sum(r["mismatches"] for r in reports),
        "maps_differ": sum(not r["maps_equal"] for r in reports),
        "placement_differs": sum(not r["placement_equal"] for r in reports),
        "first": next((r["first"] for r in reports if r["first"]), None),
        "seconds": time.perf_counter() - started,
    }
    logger.info(
        "Expert store scrub (EP rank %d, %s, %s): %d slots of %d layers read "
        "back and compared with the store, %d mismatches; tier maps differ "
        "from the placement on %d layers, the placement from EPLB's map on "
        "%d; %.2f s.",
        ep_rank,
        when,
        "every slot" if full else "slots moved since the last scrub",
        summary["slots"],
        summary["layers"],
        summary["mismatches"],
        summary["maps_differ"],
        summary["placement_differs"],
        summary["seconds"],
    )
    if summary["first"]:
        logger.error(
            "Expert store scrub (EP rank %d): first mismatch: %s",
            ep_rank,
            summary["first"],
        )
    return summary


def _abandon_layer(
    model_state: EplbModelState,
    ep_rank: int,
) -> None:
    """Every rank has the pending layer's result and at least one rank's mover
    could not prepare it: nobody commits the layer. This rank gives back what
    it prepared, and the cycle ends here on every rank (the worker's next
    coordinated flag sees `rebalanced` false everywhere). Layers already
    committed keep the new placement, this layer and the later ones keep the
    old one, and EPLB's maps agree with both; the next rearrangement plans
    from there."""
    result = model_state.pending_result
    assert result is not None
    if result.prepared is not None and model_state.movers is not None:
        mover = model_state.movers[result.layer_idx]
        assert mover is not None
        mover.discard(result.prepared)
    model_state.abandoned_rearrangements += 1
    logger.warning(
        "EPLB (EP rank %d): rearrangement of %s abandoned at layer %d of %d: "
        "%d rank(s) could not prepare the layer (%s). Nothing of it was "
        "mapped; serving continues on the current placement.",
        ep_rank,
        model_state.model_name,
        result.layer_idx,
        model_state.model.num_moe_layers,
        model_state.abandon_votes,
        "this rank among them, see the error above"
        if result.abandoned
        else "see their logs",
    )
    model_state.rebalanced = False
    model_state.abandon_votes = 0
    model_state.rearrangement_stats.clear()
    # Reset pending_result before unblocking the async worker
    model_state.pending_result = None
    result.consumed_event.record()


def _move_to_workspace(
    model_state: EplbModelState,
    ep_rank: int,
) -> None:
    result = model_state.pending_result
    assert result is not None
    cost = model_state.move_cost
    commit_started = time.perf_counter()
    if result.prepared is not None:
        # The layer places its own experts: its prepared rows and tier maps
        # switch in this window, next to EPLB's maps below.
        assert model_state.movers is not None
        mover = model_state.movers[result.layer_idx]
        assert mover is not None
        started = time.perf_counter()
        mover.commit(result.prepared)
        stats = model_state.rearrangement_stats
        for key, value in mover.prepared_stats(result.prepared).items():
            if key in ("units", "bytes", "card_bytes", "changed"):
                cost[key] = cost.get(key, 0.0) + value
            if key in ("card_bytes", "changed"):
                continue  # the move line's; not the INFO line's
            stats[key] = stats.get(key, 0.0) + value
        stats["commit_ms"] = (
            stats.get("commit_ms", 0.0) + (time.perf_counter() - started) * 1e3
        )
    else:
        assert result.transfer_metadata is not None
        move_from_buffer(
            expert_weights=model_state.model.expert_weights[result.layer_idx],
            expert_weights_buffers=model_state.expert_buffer,
            transfer_metadata=result.transfer_metadata,
            new_indices=result.new_physical_to_logical_map.numpy(),
            ep_rank=ep_rank,
        )

    _commit_eplb_maps_for_layer(
        model_state,
        new_physical_to_logical_map=result.new_physical_to_logical_map,
        layer=result.layer_idx,
    )
    cost["commit_ms"] = (
        cost.get("commit_ms", 0.0) + (time.perf_counter() - commit_started) * 1e3
    )
    cost["commit_steps"] = cost.get("commit_steps", 0.0) + 1

    if result.layer_idx == model_state.model.num_moe_layers - 1:
        model_state.rebalanced = False
        if model_state.rearrangement_stats:
            stats = model_state.rearrangement_stats
            logger.info(
                "Expert store rearrangement (EP rank %d): %d units read "
                "(%.1f MiB), prepare %.0f ms, commit %.1f ms over %d layers.",
                ep_rank,
                int(stats.get("units", 0)),
                stats.get("bytes", 0.0) / 2**20,
                stats.get("prepare_ms", 0.0),
                stats.get("commit_ms", 0.0),
                model_state.model.num_moe_layers,
            )
            stats.clear()
        model_state.completed_rearrangements += 1
        if model_state.rearrangement_started:
            model_state.last_rearrangement_seconds = (
                time.perf_counter() - model_state.rearrangement_started
            )
        if model_state.movers is not None and model_state.movers[0] is not None:
            every = model_state.movers[0].scrub_every
            if every and model_state.completed_rearrangements % every == 0:
                started = time.perf_counter()
                _scrub_movers(model_state, ep_rank, full=False, when="rearrangement")
                cost["verify_ms"] = (time.perf_counter() - started) * 1e3

    # Reset pending_result before unblocking the async worker
    model_state.pending_result = None
    result.consumed_event.record()

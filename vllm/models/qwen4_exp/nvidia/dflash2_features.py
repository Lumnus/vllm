# SPDX-License-Identifier: Apache-2.0
"""DFlash2 features from completed Qwen4Exp hyper-connection layers."""

import torch

from vllm.model_executor.models.interfaces import EagleModelMixin

TARGET_LAYER_IDS = (5, 15, 25, 35, 45)
# vLLM converts DFlash's completed zero-based layer IDs to these boundaries.
AUX_LAYER_BOUNDARIES = tuple(i + 1 for i in TARGET_LAYER_IDS)


class DFlash2FeatureCapture(EagleModelMixin):
    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers and tuple(layers) != AUX_LAYER_BOUNDARIES:
            raise ValueError(
                "Flash-Next DFlash2 requires completed layers 5/15/25/35/45 "
                "as vLLM boundaries 6/16/26/36/46"
            )
        if layers and (
            self.config.hc_count != 4
            or self.config.hidden_size != 2560
            or self.start_layer != 0
            or self.end_layer != 48
            or self.use_hc_sequence_parallel
        ):
            raise ValueError("DFlash2 capture requires unsharded HC4/H2560, PP1")
        super()._set_aux_hidden_state_layers(layers)

    def capture_completed_layer(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor | None,
        injection: torch.Tensor | None,
    ) -> None:
        if layer_idx + 1 not in self.aux_hidden_state_layers:
            return
        # A decoder returns a delayed MLP injection. Materialize it for the
        # feature without consuming or modifying the tuple passed downstream.
        completed = hidden_states
        if block_output is not None:
            completed = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
        # HC is the outer axis; TP4 owns replicated full-width token rows.
        # mean allocates independent storage, so later workspace reuse cannot
        # overwrite these snapshots. Preserve packed token order verbatim.
        feature = completed.unflatten(-1, (4, 2560)).mean(dim=-2)
        aux_hidden_states.append(feature)

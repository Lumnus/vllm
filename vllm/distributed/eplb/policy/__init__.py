# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import get_args

from vllm.config.parallel import EPLBPolicyOption

from .abstract import AbstractEplbPolicy
from .default import DefaultEplbPolicy
from .incremental import IncrementalEplbPolicy

EPLB_POLICIES: dict[str, type[DefaultEplbPolicy] | type[IncrementalEplbPolicy]] = {
    "default": DefaultEplbPolicy,
    "incremental": IncrementalEplbPolicy,
}

# Ensure that the EPLB_POLICIES keys match the EPLBPolicyOption values
assert set(EPLB_POLICIES.keys()) == set(get_args(EPLBPolicyOption))

__all__ = [
    "AbstractEplbPolicy",
    "DefaultEplbPolicy",
    "IncrementalEplbPolicy",
    "EPLB_POLICIES",
]

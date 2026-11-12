# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register vLLM's custom ops the way an accelerator engine does.

`direct_register_custom_op` registers an op's kernel under the dispatch key
of the current platform, so in an XPU or CUDA engine a tensor in host memory
finds no kernel behind the op. A CPU test run registers under `CPU` and
cannot see that. Loaded early (`pytest -p`, or imported first in a script),
this module sets the key the ops are registered under from the environment
variable `VLLM_TEST_OPS_DISPATCH_KEY`; it must run before the first op is
registered, i.e. before the model code is imported.
"""

import os

ENV = "VLLM_TEST_OPS_DISPATCH_KEY"

_key = os.environ.get(ENV)
if _key:
    from vllm.platforms.cpu import CpuPlatform
    from vllm.platforms.interface import Platform

    Platform.dispatch_key = _key
    CpuPlatform.dispatch_key = _key

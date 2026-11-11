# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import multiprocessing
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures.process import ProcessPoolExecutor
from functools import cache
from typing import Any

import regex as re
import torch


def cuda_is_initialized() -> bool:
    """Check if CUDA is initialized."""
    if not torch.cuda._is_compiled():
        return False
    return torch.cuda.is_initialized()


def xpu_is_initialized() -> bool:
    """Check if XPU is initialized."""
    if not torch.xpu._is_compiled():
        return False
    return torch.xpu.is_initialized()


def cuda_get_device_properties(
    device, names: Sequence[str], init_cuda=False
) -> tuple[Any, ...]:
    """Get specified CUDA device property values without initializing CUDA in
    the current process."""
    if init_cuda or cuda_is_initialized():
        props = torch.cuda.get_device_properties(device)
        return tuple(getattr(props, name) for name in names)

    # Run in subprocess to avoid initializing CUDA as a side effect.
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as executor:
        return executor.submit(cuda_get_device_properties, device, names, True).result()


_XPU_PROPERTIES_TIMEOUT_S = 120
# Run with `python -c`: the child imports torch and nothing else.
_XPU_PROPERTIES_SCRIPT = """\
import json, sys, torch
props = torch.xpu.get_device_properties(int(sys.argv[1]))
print(json.dumps([getattr(props, name) for name in sys.argv[2:]]))
"""
# How much of the child's stderr an error carries.
_XPU_PROPERTIES_STDERR_TAIL = 2000


class XPUPropertiesChildError(RuntimeError):
    """The child process that reads XPU device properties was started and
    did not deliver: it timed out or exited with an error.

    Deliberately not an OSError (the built-in TimeoutError is one), so that
    "the child could not be started" stays a separate case.
    """


def _stderr_tail(stderr: str | bytes | None) -> str:
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return (stderr or "").strip()[-_XPU_PROPERTIES_STDERR_TAIL:]


def xpu_get_device_properties(
    device: int, names: Sequence[str], init_xpu=False
) -> tuple[Any, ...]:
    """Get specified XPU device property values without initializing XPU in
    the current process.

    An initialized XPU runtime stays for the lifetime of the process and
    holds device memory on every visible device. When the runtime is not
    initialized here, a child process reads the values: it imports torch,
    initializes XPU, prints the values and exits, so the memory is held for
    the few seconds the child lives.

    The child has `_XPU_PROPERTIES_TIMEOUT_S` seconds. Raises
    XPUPropertiesChildError when it does not answer in time or exits with an
    error (with the tail of its stderr), and OSError when it could not be
    started.
    """
    if init_xpu or xpu_is_initialized():
        props = torch.xpu.get_device_properties(device)
        return tuple(getattr(props, name) for name in names)

    try:
        result = subprocess.run(
            [sys.executable, "-c", _XPU_PROPERTIES_SCRIPT, str(device), *names],
            capture_output=True,
            text=True,
            timeout=_XPU_PROPERTIES_TIMEOUT_S,
            check=True,
        )
    except subprocess.TimeoutExpired as e:
        raise XPUPropertiesChildError(
            f"Reading the properties of XPU device {device} in a child process "
            f"took more than {_XPU_PROPERTIES_TIMEOUT_S} s: {_stderr_tail(e.stderr)}"
        ) from e
    except subprocess.CalledProcessError as e:
        raise XPUPropertiesChildError(
            f"The child process reading the properties of XPU device {device} "
            f"exited with status {e.returncode}: {_stderr_tail(e.stderr)}"
        ) from e
    # The values are the last line; anything a library prints comes before.
    return tuple(json.loads(result.stdout.strip().splitlines()[-1]))


# Per device: the facts, or the error of the read that failed.
_xpu_device_facts: dict[int, tuple[str, int, int] | Exception] = {}


def xpu_device_facts(device: int) -> tuple[str, int, int]:
    """Name, total memory and shared local memory size of an XPU device.

    Read together and remembered, so that a process which has not
    initialized XPU starts one child process at most, whoever asks first. A
    failed read is remembered as well: later callers get the same error at
    once instead of waiting for another child. Once XPU is initialized in
    the process the read is direct, and a remembered failure is retried.
    """
    known = _xpu_device_facts.get(device)
    if known is None or (isinstance(known, Exception) and xpu_is_initialized()):
        try:
            name, total_memory, local_mem_size = xpu_get_device_properties(
                device, ("name", "total_memory", "local_mem_size")
            )
            known = (name, total_memory, local_mem_size)
        except Exception as e:
            known = e
        _xpu_device_facts[device] = known
    if isinstance(known, Exception):
        raise known
    return known


@cache
def is_pin_memory_available() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_pin_memory_available()


@cache
def is_uva_available() -> bool:
    """Check if Unified Virtual Addressing (UVA) is available."""
    from vllm.platforms import current_platform

    if current_platform.is_cpu():
        return True
    # UVA requires pinned memory. Under NVIDIA Confidential Computing the
    # device-side view of pinned host memory is not coherent (the GPU reads a
    # stale snapshot), so treat UVA as unavailable and fall back to explicit
    # copies (see vllm.v1.conf_compute_utils).
    return is_pin_memory_available() and not current_platform.is_confidential_compute()


@cache
def num_compute_units(device_id: int = 0) -> int:
    """Get the number of compute units of the current device."""
    from vllm.platforms import current_platform

    return current_platform.num_compute_units(device_id)


@cache
def get_device_name_as_file_name(device_id: int = 0) -> str:
    from vllm.platforms import current_platform

    name = current_platform.get_device_name(device_id)
    name = re.sub(r"[\s/]+", "_", name)
    return name

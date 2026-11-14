# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The device waits for a word the host writes (the CPU experts' handshake).

A host thread computes a layer's CPU experts while the main thread goes on
enqueuing the step; the device must not read the CPU's result rows before
the host has written them. :func:`wait_host_flag` enqueues, in stream order,
a one-program kernel that reads a pinned host word through its device view
(UVA) until it holds ``seq``: the host's ``done`` word (word 0) or its
``failed`` word (word 1). After ``max_spins`` reads it gives up and writes
``seq`` into word 2 (``timed out``), which the host checks; the step is then
failed by the host, never used.

The words, ``flag`` (int32, pinned, at least 3):

* 0: the last sequence number the host finished (written by the host);
* 1: the last sequence number the host failed (written by the host);
* 2: the last sequence number the device stopped waiting for (written here).

Sequence numbers run 1 .. 2**31 - 1 and wrap; the host finishes them in
order, and the device waits for one only after every earlier wait ended, so
a word holds either the number waited for or the one before it.

Reads (``VLLM_CPU_EXPERTS_ASYNC_READ``): "volatile" (default) a volatile
load, "atomic" an acquire load at system scope (an atomic add of 0, which
needs atomics on host memory over PCIe). Which of them sees a host write
without a stale cached copy on a given card and driver, and at what cost,
is what ``benchmarks/kernels/benchmark_xpu_host_flag_wait.py`` measures; a
read that never sees the write ends in the bounded give-up, which fails the
step. On the CPU device (tests) the wait is a host loop.
"""

import os
import time

import torch

from vllm.triton_utils import tl, triton

DONE, FAILED, TIMED_OUT = 0, 1, 2
FLAG_WORDS = 4
_READ_ENV = "VLLM_CPU_EXPERTS_ASYNC_READ"


def read_mode() -> str:
    mode = os.environ.get(_READ_ENV, "").strip().lower() or "volatile"
    if mode not in ("atomic", "volatile"):
        raise ValueError(f"{_READ_ENV} is 'atomic' or 'volatile', not {mode!r}")
    return mode


@triton.jit(do_not_specialize=["seq", "max_spins"])
def _wait_host_flag_kernel(flag_ptr, seq, max_spins, ATOMIC: tl.constexpr):
    state = 0
    spins = 0
    while state == 0:
        if ATOMIC:
            done = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys")
            failed = tl.atomic_add(flag_ptr + 1, 0, sem="acquire", scope="sys")
        else:
            done = tl.load(flag_ptr, volatile=True)
            failed = tl.load(flag_ptr + 1, volatile=True)
        spins += 1
        state = tl.where(done == seq, 1, 0)
        state = tl.where((state == 0) & (failed == seq), 2, state)
        state = tl.where((state == 0) & (spins >= max_spins), 3, state)
    tl.store(flag_ptr + 2, seq, mask=state == 3)


def flag_words() -> torch.Tensor:
    """The host words of one handshake: int32, zeroed, pinned where the
    platform pins."""
    from vllm.utils.platform_utils import is_pin_memory_available

    return torch.zeros(
        FLAG_WORDS, dtype=torch.int32, pin_memory=is_pin_memory_available()
    )


def device_view(flag: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``flag`` as the device reads it (the host tensor itself on the CPU)."""
    if device.type == "cpu":
        return flag
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    return get_accelerator_view_from_cpu_tensor(flag)


def wait_host_flag(
    flag: torch.Tensor,
    view: torch.Tensor,
    seq: int,
    max_spins: int,
    *,
    atomic: bool = False,
    host_spin_s: float = 1e-6,
) -> None:
    """Enqueue on the current stream: wait until word 0 or word 1 of
    ``flag`` holds ``seq``, at most ``max_spins`` reads (then word 2 :=
    ``seq``). ``view`` is :func:`device_view` of ``flag``. On the CPU device
    the caller's thread waits here (``max_spins`` reads ``host_spin_s``
    apart)."""
    if view.device.type == "cpu":
        words = flag.numpy()
        for _ in range(max_spins):
            if words[DONE] == seq or words[FAILED] == seq:
                return
            time.sleep(host_spin_s)
        words[TIMED_OUT] = seq
        return
    _wait_host_flag_kernel[(1,)](view, seq, max_spins, ATOMIC=atomic)

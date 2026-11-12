# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep an offline tool off the accelerator, and prove it did.

The store's build tool turns a checkpoint into units on the CPU. On a host
whose torch build sees the cards, an innocent call (``torch.xpu.is_available``
during platform detection, a pinned allocation, a kernels library import that
loads its native extensions) is enough to initialise the runtime and map device
memory, which a serving engine on the same card then cannot have. Three layers:

1. :func:`prepare_env` pins the platform to CPU *before* vLLM is imported
   (``VLLM_TARGET_DEVICE=cpu``, no CUDA devices).
2. :func:`install` wraps the entry points of ``torch.xpu`` that initialise or
   enumerate the runtime: enumeration reports no device, anything that would
   initialise raises :class:`CardTouchedError`. Every call is recorded.
3. :func:`check` looks at the process itself: mappings of device nodes
   (``/proc/self/maps``, where device memory shows) and open device
   descriptors, and whether ``torch.xpu`` reports itself initialised.

Run as a script this file is a launcher that does 1 and 2 before the first
vLLM import, then runs the build tool::

    python <tree>/vllm/model_executor/offloader/card_guard.py build --dirs ...
"""

import os
import sys
import threading
from typing import Any

DEVICE_PREFIXES = ("/dev/dri/", "/dev/accel/", "/dev/nvidia", "/dev/kfd")

# torch.xpu entry points: (name, what the wrapper does). "hide" answers as a
# host without the device; "raise" refuses.
_HIDE = {"is_available": False, "device_count": 0}
_RAISE = (
    "init",
    "_lazy_init",
    "current_device",
    "set_device",
    "synchronize",
    "empty_cache",
    "mem_get_info",
    "memory_allocated",
    "get_device_properties",
    "current_stream",
)

_lock = threading.Lock()
_calls: list[str] = []
_installed = False


class CardTouchedError(RuntimeError):
    """The process initialised, or tried to initialise, the accelerator."""


def prepare_env() -> None:
    """Pin the platform to CPU. Call before the first vLLM import."""
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _caller() -> str:
    frame = sys._getframe(2)  # _caller <- the wrapper <- whoever called it
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def install() -> bool:
    """Wrap ``torch.xpu``. Returns False if torch has no ``xpu`` module."""
    global _installed
    import torch

    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    with _lock:
        if _installed:
            return True
        for name, answer in _HIDE.items():
            if hasattr(xpu, name):
                setattr(xpu, name, _hider(name, answer))
        for name in _RAISE:
            if hasattr(xpu, name):
                setattr(xpu, name, _refuser(name))
        _installed = True
    return True


def _hider(name: str, answer: Any) -> Any:
    def hidden(*args: Any, **kwargs: Any) -> Any:
        with _lock:
            _calls.append(f"{name} (answered {answer}) from {_caller()}")
        return answer

    hidden.__name__ = name
    return hidden


def _refuser(name: str) -> Any:
    def refused(*args: Any, **kwargs: Any) -> Any:
        where = _caller()
        with _lock:
            _calls.append(f"{name} (refused) from {where}")
        raise CardTouchedError(
            f"torch.xpu.{name} called from {where}: the build tool must not "
            "initialise the accelerator"
        )

    refused.__name__ = name
    return refused


def calls() -> list[str]:
    """What touched ``torch.xpu`` since :func:`install`."""
    with _lock:
        return list(_calls)


def _device_mappings() -> list[str]:
    found: set[str] = set()
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].strip().startswith(DEVICE_PREFIXES):
                    found.add(parts[5].strip())
    except OSError:
        pass
    return sorted(found)


def _device_fds() -> list[str]:
    found: list[str] = []
    try:
        for fd in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                continue
            if target.startswith(DEVICE_PREFIXES):
                found.append(target)
    except OSError:
        pass
    return sorted(found)


def _xpu_initialized() -> bool:
    # Read the real state, not the wrapped entry points.
    try:
        import torch

        return bool(torch.xpu.is_initialized())
    except Exception:
        return False


def report() -> dict[str, Any]:
    return {
        "xpu_initialized": _xpu_initialized(),
        "xpu_calls": calls(),
        "device_mappings": _device_mappings(),
        "device_fds": _device_fds(),
    }


def check(*, strict_fds: bool = False) -> dict[str, Any]:
    """Raise :class:`CardTouchedError` if the process holds card state.

    Fails on: ``torch.xpu`` initialised, a refused ``torch.xpu`` call, a
    mapping of a device node (device memory is mapped through it). Open device
    descriptors fail only with ``strict_fds`` (a driver may keep a node open
    after enumerating, without holding memory); they are always reported.
    """
    state = report()
    problems = []
    if state["xpu_initialized"]:
        problems.append("torch.xpu reports itself initialised")
    problems += [c for c in state["xpu_calls"] if "(refused)" in c]
    if state["device_mappings"]:
        problems.append(f"device node mapped: {state['device_mappings']}")
    if strict_fds and state["device_fds"]:
        problems.append(f"device node open: {state['device_fds']}")
    if problems:
        raise CardTouchedError("; ".join(problems))
    return state


def summary(state: dict[str, Any]) -> str:
    text = (
        f"card guard: xpu initialised={state['xpu_initialized']}, "
        f"torch.xpu calls={len(state['xpu_calls'])}, "
        f"device mappings={len(state['device_mappings'])}, "
        f"device fds={len(state['device_fds'])}"
    )
    if state["xpu_calls"]:
        text += " [" + "; ".join(state["xpu_calls"][:3]) + "]"
    return text


def main() -> int:
    """Launcher: guard first, then the build tool (its own command line)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == here:
        sys.path.pop(0)  # the tool's directory is not a place to import from
    prepare_env()
    install()
    import runpy

    module = "vllm.model_executor.offloader.expert_store_build"
    sys.argv[0] = module
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

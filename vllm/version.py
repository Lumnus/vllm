# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def _b70_source_version() -> str:
    """A source tree run without a build: the engine line and, in a git
    checkout, the commit it runs from (dev+b70.l34.g<sha10>)."""
    import os
    import subprocess

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        sha = ""
    return f"dev+b70.l34.g{sha}" if sha else "dev+b70.l34"


try:
    from ._version import __version__, __version_tuple__
except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash:\n{e}", RuntimeWarning, stacklevel=2)

    # B70: a source tree run without a build still names its engine line
    # and commit.
    __version__ = _b70_source_version()
    __version_tuple__ = (0, 0, __version__)


def _prev_minor_version_was(version_str):
    """Check whether a given version matches the previous minor version.

    Return True if version_str matches the previous minor version.

    For example - return True if the current version if 0.7.4 and the
    supplied version_str is '0.6'.

    Used for --show-hidden-metrics-for-version.
    """
    # Match anything if this is a dev tree
    if __version_tuple__[0:2] == (0, 0):
        return True

    assert isinstance(__version_tuple__[1], int)
    return version_str == f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"


def _prev_minor_version():
    """For the purpose of testing, return a previous minor version number."""
    # In dev tree, this will return "0.-1", but that will work fine"
    assert isinstance(__version_tuple__[1], int)
    return f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"

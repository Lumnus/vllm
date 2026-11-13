# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server defaults that every request reads, changed in a running server.

The OpenAI-compatible handlers (chat, completions, responses, and the
handlers built on them) fill what a request leaves out from their
``default_sampling_params`` (the model's generation config and
``--override-generation-config``), read per request; the chat handler
resolves the Han mask flag per request. Both live in the API server
process, not in the engine, so ``defaults.set`` and ``defaults.state`` are
answered here: the change applies to the next request this process parses.
A value of ``null`` puts the start value back.
"""

import copy
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Handlers on app.state that read default_sampling_params per request.
SERVING_ATTRS = (
    "openai_serving_chat",
    "openai_serving_chat_batch",
    "openai_serving_completion",
    "openai_serving_responses",
    "anthropic_serving_messages",
)


def _number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _integer(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# name -> (check, text): check(value) is True when the value is valid.
SAMPLING_DEFAULTS: dict[str, tuple[Any, str]] = {
    "temperature": (lambda v: _number(v) and v >= 0, "a number >= 0"),
    "top_p": (lambda v: _number(v) and 0 < v <= 1, "a number in (0, 1]"),
    "top_k": (lambda v: _integer(v) and v >= -1, "an integer >= -1 (0 or -1: off)"),
    "min_p": (lambda v: _number(v) and 0 <= v <= 1, "a number in [0, 1]"),
    "repetition_penalty": (lambda v: _number(v) and v > 0, "a number > 0"),
    "presence_penalty": (lambda v: _number(v) and -2 <= v <= 2, "a number in [-2, 2]"),
    "frequency_penalty": (
        lambda v: _number(v) and -2 <= v <= 2,
        "a number in [-2, 2]",
    ),
    "thinking_token_budget": (
        lambda v: _integer(v) and v >= 0,
        "an integer >= 0 (tokens of thinking before the end-of-thinking token)",
    ),
}
CJK_KEY = "cjk_mask_default"


class DefaultsRefused(Exception):
    def __init__(self, message: str, restart_required: bool = False) -> None:
        super().__init__(message)
        self.restart_required = restart_required


def _servings(app_state: Any) -> list[tuple[str, Any]]:
    out = []
    for attr in SERVING_ATTRS:
        serving = getattr(app_state, attr, None)
        params = getattr(serving, "default_sampling_params", None)
        if isinstance(params, dict):
            out.append((attr, serving))
    return out


def _base(app_state: Any) -> dict[str, dict]:
    """The start values, per handler, kept on first use."""
    base = getattr(app_state, "server_defaults_base", None)
    if base is None:
        base = {
            attr: copy.deepcopy(s.default_sampling_params)
            for attr, s in _servings(app_state)
        }
        app_state.server_defaults_base = base
    return base


def _api_servers(app_state: Any) -> int:
    args = getattr(app_state, "args", None)
    return int(getattr(args, "api_server_count", None) or 1)


def state(app_state: Any) -> dict[str, Any]:
    from vllm.v1.sample import b70_cjk

    servings = _servings(app_state)
    base = _base(app_state)
    now: dict[str, Any] = {}
    start: dict[str, Any] = {}
    if servings:
        attr, serving = servings[0]
        for key in SAMPLING_DEFAULTS:
            now[key] = serving.default_sampling_params.get(key)
            start[key] = base.get(attr, {}).get(key)
    return {
        "defaults": now,
        "start": start,
        "handlers": [attr for attr, _ in servings],
        CJK_KEY: b70_cjk.default_mode() if b70_cjk.enabled() else None,
        "cjk_mask_available": b70_cjk.enabled(),
        "api_servers": _api_servers(app_state),
    }


def check(app_state: Any, args: dict[str, Any]) -> None:
    """Raise DefaultsRefused (nothing changed) when `args` cannot apply."""
    from vllm.v1.sample import b70_cjk

    if not args:
        raise DefaultsRefused("name at least one default to change")
    n = _api_servers(app_state)
    if n > 1:
        raise DefaultsRefused(
            f"these defaults live in each API server process and this server "
            f"runs {n} (--api-server-count): a change here would reach only "
            f"the requests this process parses; start with one API server to "
            f"change them live",
            restart_required=True,
        )
    known = sorted([*SAMPLING_DEFAULTS, CJK_KEY])
    unknown = sorted(set(args) - set(known))
    if unknown:
        raise DefaultsRefused(f"unknown defaults {unknown}; known: {known}")
    for key, value in args.items():
        if key == CJK_KEY:
            if not b70_cjk.enabled():
                raise DefaultsRefused(
                    "the Han mask is not in this server: its vocabulary mask is "
                    f"built on every worker at start ({b70_cjk.ENV}=1)",
                    restart_required=True,
                )
            if value is not None and value not in b70_cjk.DEFAULT_MODES:
                raise DefaultsRefused(
                    f"{CJK_KEY} must be one of {list(b70_cjk.DEFAULT_MODES)} or "
                    f"null, got {value!r}"
                )
            continue
        ok, text = SAMPLING_DEFAULTS[key]
        if value is not None and not ok(value):
            raise DefaultsRefused(f"{key} must be {text} or null, got {value!r}")
    if any(k in SAMPLING_DEFAULTS for k in args) and not _servings(app_state):
        raise DefaultsRefused("no request handler with defaults in this server")


def apply(app_state: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Check, then change every handler's defaults; {key: [old, new]}."""
    from vllm.v1.sample import b70_cjk

    check(app_state, args)
    base = _base(app_state)
    changes: dict[str, list] = {}
    servings = _servings(app_state)
    for key, value in args.items():
        if key == CJK_KEY:
            old = b70_cjk.default_mode()
            new = "auto" if value is None else value
            b70_cjk.set_default_mode(new)
            if old != new:
                changes[key] = [old, new]
            continue
        old = servings[0][1].default_sampling_params.get(key)
        for attr, serving in servings:
            params = serving.default_sampling_params
            start = base.get(attr, {}).get(key)
            target = start if value is None else value
            if target is None:
                params.pop(key, None)
            else:
                params[key] = target
        new = servings[0][1].default_sampling_params.get(key)
        if old != new:
            changes[key] = [old, new]
    for key, (old, new) in changes.items():
        logger.info(
            "Runtime control: defaults %s %s -> %s (%s).",
            key,
            old,
            new,
            ", ".join(attr for attr, _ in servings) if key != CJK_KEY else "chat",
        )
    if not changes:
        logger.info("Runtime control: defaults unchanged.")
    return {
        "op": "defaults.set",
        "applied": "all",
        "changes": changes,
        **state(app_state),
    }

# SPDX-License-Identifier: Apache-2.0
"""B70-CJK: CPU tests for the startup default (B70_CJK_MASK_DEFAULT), the
per-request effect of the server default on a Han-carrying chat turn, and the
off-decision log (no tokenizer, no device)."""

import importlib
import json

import pytest

from vllm.v1.sample import b70_cjk

HAN_TURN = "Добре, напомни ми: 服务/于 беше в отговор 1"
USER_BG = [{"role": "user", "content": "Здравей"}]
USER_HAN = [{"role": "user", "content": HAN_TURN}]


@pytest.fixture(autouse=True)
def _restore_mode():
    mode = b70_cjk.default_mode()
    yield
    b70_cjk.set_default_mode(mode)
    b70_cjk._OFF_LOG = b70_cjk.OffLog()


@pytest.fixture
def lines(monkeypatch):
    out: list[str] = []
    monkeypatch.setattr(
        b70_cjk.logger, "info", lambda fmt, *a, **k: out.append(fmt % a)
    )
    return out


# ------------------------------------------------------- startup default (env)


@pytest.mark.parametrize("mode", ["auto", "on", "off"])
def test_default_from_env_modes(mode):
    assert b70_cjk.default_from_env({b70_cjk.ENV_DEFAULT: mode}) == mode
    assert b70_cjk.default_from_env({b70_cjk.ENV_DEFAULT: f" {mode.upper()} "}) == mode


def test_default_from_env_unset_is_auto():
    assert b70_cjk.default_from_env({}) == "auto"
    assert b70_cjk.default_from_env({b70_cjk.ENV_DEFAULT: ""}) == "auto"


def test_default_from_env_invalid_warns_and_falls_back(monkeypatch):
    warned: list[str] = []
    monkeypatch.setattr(
        b70_cjk.logger, "warning", lambda fmt, *a, **k: warned.append(fmt % a)
    )
    assert b70_cjk.default_from_env({b70_cjk.ENV_DEFAULT: "sometimes"}) == "auto"
    assert len(warned) == 1 and b70_cjk.ENV_DEFAULT in warned[0]


@pytest.mark.parametrize("mode", ["on", "off", "auto", "bogus"])
def test_process_start_reads_env(monkeypatch, mode):
    """The module-level default (set at import, i.e. at server start) follows
    the variable; the runtime surface changes it afterwards and ``null``
    returns to the start value."""
    monkeypatch.setenv(b70_cjk.ENV_DEFAULT, mode)
    try:
        importlib.reload(b70_cjk)
        want = mode if mode in b70_cjk.DEFAULT_MODES else "auto"
        assert b70_cjk.default_mode() == want
        assert b70_cjk.start_default_mode() == want
        b70_cjk.set_default_mode("off" if want != "off" else "on")
        assert b70_cjk.default_mode() != want
        assert b70_cjk.start_default_mode() == want
    finally:
        monkeypatch.delenv(b70_cjk.ENV_DEFAULT, raising=False)
        importlib.reload(b70_cjk)
    assert b70_cjk.default_mode() == "auto"


# ------------------------------------------------ a chat turn that carries Han


def _req(**kw):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    body = {"model": "m", "messages": USER_HAN, **kw}
    return ChatCompletionRequest.model_validate_json(json.dumps(body))


def test_han_user_turn_auto_off_on_on(monkeypatch):
    monkeypatch.setenv(b70_cjk.ENV, "1")
    b70_cjk.set_default_mode("auto")
    assert _req().to_sampling_params(64, {}).extra_args == {"cjk_mask": False}
    b70_cjk.set_default_mode("on")
    assert _req().to_sampling_params(64, {}).extra_args == {"cjk_mask": True}
    b70_cjk.set_default_mode("off")
    assert _req(messages=USER_BG).to_sampling_params(64, {}).extra_args == {
        "cjk_mask": False
    }


def test_explicit_flag_beats_every_default(monkeypatch):
    for mode in b70_cjk.DEFAULT_MODES:
        b70_cjk.set_default_mode(mode)
        assert b70_cjk.chat_extra_args({"cjk_mask": True}, USER_HAN, None) == {
            "cjk_mask": True
        }
        assert b70_cjk.chat_extra_args({"cjk_mask": False}, USER_BG, None) == {
            "cjk_mask": False
        }


def test_unset_variable_keeps_stock_behaviour():
    b70_cjk.set_default_mode(b70_cjk.default_from_env({}))
    assert b70_cjk.chat_extra_args({}, USER_BG, None) == {"cjk_mask": True}
    assert b70_cjk.chat_extra_args({}, USER_HAN, None) == {"cjk_mask": False}


def test_han_roles():
    msgs = [
        {"role": "system", "content": "Отговаряй на български."},
        {"role": "assistant", "content": "例如"},
        {"role": "user", "content": "看"},
        {"role": "developer", "content": "答"},
        {"role": "user", "content": "again 中"},
    ]
    assert b70_cjk.han_roles(msgs) == ("user", "developer")
    assert b70_cjk.han_roles(USER_BG) == ()


# --------------------------------------------------------------- decision log


def test_auto_off_logs_id_rule_role_no_text(lines):
    b70_cjk.set_default_mode("auto")
    b70_cjk.chat_extra_args({}, USER_HAN, None, "req-1")
    assert len(lines) == 1
    assert (
        "req-1" in lines[0]
        and "rule=auto-latest-user-han" in lines[0]
        and "role=user" in lines[0]
    )
    assert "服" not in lines[0] and "напомни" not in lines[0]


def test_explicit_off_logs_rule(lines):
    b70_cjk.set_default_mode("auto")
    b70_cjk.chat_extra_args({"cjk_mask": False}, USER_BG, None, "req-2")
    assert len(lines) == 1
    assert "req-2" in lines[0] and "rule=explicit" in lines[0] and "role=-" in lines[0]


def test_default_off_logs_rule(lines):
    b70_cjk.set_default_mode("off")
    b70_cjk.chat_extra_args({}, USER_BG, None, "req-3")
    assert len(lines) == 1 and "rule=default-off" in lines[0]


def test_mask_on_logs_nothing(lines):
    b70_cjk.set_default_mode("on")
    b70_cjk.chat_extra_args({}, USER_HAN, None, "req-4")
    b70_cjk.set_default_mode("auto")
    b70_cjk.chat_extra_args({}, USER_BG, None, "req-5")
    assert lines == []


def test_protocol_passes_request_id(monkeypatch, lines):
    monkeypatch.setenv(b70_cjk.ENV, "1")
    b70_cjk.set_default_mode("auto")
    _req(request_id="chat-abc").to_sampling_params(64, {})
    assert len(lines) == 1 and "chat-abc" in lines[0]
    assert "服" not in lines[0]


def test_request_id_is_bounded_and_quoted(lines):
    b70_cjk.set_default_mode("auto")
    b70_cjk.chat_extra_args({}, USER_HAN, None, "x\nfake line " + "y" * 200)
    assert len(lines) == 1 and "\n" not in lines[0] and len(lines[0]) < 200


def test_rate_limit_and_summary(lines):
    t = [0.0]
    log = b70_cjk.OffLog(burst=3, window=10.0, clock=lambda: t[0])
    for i in range(10):
        log.record(f"r{i}", "auto-han", "user")
    assert len(lines) == 3  # the burst; the other 7 are counted
    assert log.suppressed == {"auto-han": 7}
    t[0] = 11.0  # next window: summary first, then this request's own line
    log.record("r10", "explicit", None)
    assert len(lines) == 5
    assert "auto-han=7" in lines[3] and "auto-han=10" in lines[3]
    assert "r10" in lines[4]
    assert log.suppressed == {}

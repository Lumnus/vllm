# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-CJK: serving-side Han mask, host-side helpers (env ``B70_CJK_MASK=1``).

Import-light (no torch): the API server uses it to decide per request whether
the mask applies, the worker uses it to build the vocabulary mask.

Per-request gate. The mask is ON for a request unless
  1. the request sets ``cjk_mask`` explicitly (``vllm_xargs`` or a top-level
     extra body field; false / 0 / "off" turn it off, true forces it on), or
  2. the server default is "auto" and the latest user message asks for
     Chinese in the user's own words: its text carries Han, or asks in words
     for Chinese output (a short Bulgarian and English pattern list,
     ``asked_for_chinese``). Blocks an agent client injects into the turn
     (Hermes recalled memory, OpenClaw inbound context, reminder wrappers;
     ``user_own_text``) are cut before the scan. Earlier user turns and the
     system and developer turns do not count: recalled memories, pasted logs
     and old questions carry Han the user is not asking for now, and a system
     prompt that wants Chinese sends ``cjk_mask: false``.
  For other endpoints (no chat turns) the prompt tokens are scanned.
The decision lands in ``SamplingParams.extra_args["cjk_mask"]`` as a bool.

Server default. A chat request without an explicit flag follows the server
default: "auto" (the rule above), "on" (always masked) or "off". It starts at
``B70_CJK_MASK_DEFAULT`` (auto when unset or invalid) and the runtime control
surface changes it live.

Decision log. A request whose mask is off logs one INFO line: request id,
rule (``explicit``, ``default-off``, ``auto-latest-user-han``,
``auto-asked-chinese``) and role. Never the text. The lines are rate-limited;
the excess is counted and the next window opens with a summary line.

Vocabulary mask. A token is masked when
  1. its byte-level text contains a Han code point (U+3400-4DBF, U+4E00-9FFF,
     U+F900-FAFF), or
  2. it ends in an incomplete UTF-8 sequence with a lead byte 0xE3-0xE9 or
     0xEF whose code-point range reaches Han and does not reach kana
     (U+3040-30FF, U+31F0-31FF, halfwidth U+FF65-FF9F). That keeps the lone
     0xE3 byte (it leads kana, CJK punctuation and Han Ext-A alike), every
     0xE3 0x80-0x8F pair and the lone 0xEF byte; 15 kana (e.g. U+3062, U+30C5)
     have no single token and are spelled through those byte tokens, so kana
     stays emittable. The price: Ext-A (U+3400-3FFF) and the compatibility
     ideographs (U+F900-FAFF) that have no full token stay reachable through
     the lone 0xE3 / 0xEF byte tokens; both blocks are rare.
Special and added tokens are never masked. Continuation-only byte tokens are
never masked (they also spell Cyrillic and every other multi-byte script).
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "B70_CJK_MASK"
ENV_DEFAULT = "B70_CJK_MASK_DEFAULT"
XARG = "cjk_mask"

HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))
KANA_RANGES = ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF65, 0xFF9F))
_HAN_RE = re.compile("[㐀-䶿一-鿿豈-﫿]")
_LEADS = frozenset(range(0xE3, 0xEA)) | {0xEF}  # 3-byte leads reaching Han
_PROMPT_ROLES = ("system", "developer", "user")

_TRUE = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no"}


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


# What a chat request without an explicit flag gets: "auto" (on unless the
# prompt has Han), "on" or "off". This process's value; the runtime control
# surface changes it (`defaults.set {"cjk_mask_default": ...}`).
DEFAULT_MODES = ("auto", "on", "off")


def default_from_env(environ: Mapping[str, str] | None = None) -> str:
    """Startup default from ``B70_CJK_MASK_DEFAULT``; "auto" when unset, and
    (with a warning) when not one of DEFAULT_MODES."""
    raw = (os.environ if environ is None else environ).get(ENV_DEFAULT)
    if raw is None or not raw.strip():
        return "auto"
    mode = raw.strip().lower()
    if mode in DEFAULT_MODES:
        return mode
    logger.warning(
        "%s=%r is not one of %s; using 'auto'", ENV_DEFAULT, raw[:32], DEFAULT_MODES
    )
    return "auto"


_START_MODE = default_from_env()
_DEFAULT_MODE = _START_MODE


def start_default_mode() -> str:
    """The default this process started with (what ``null`` resets to)."""
    return _START_MODE


def default_mode() -> str:
    return _DEFAULT_MODE


def set_default_mode(mode: str) -> None:
    global _DEFAULT_MODE
    if mode not in DEFAULT_MODES:
        raise ValueError(f"{XARG} default must be one of {DEFAULT_MODES}, got {mode!r}")
    _DEFAULT_MODE = mode


def has_han(text: str) -> bool:
    return _HAN_RE.search(text) is not None


def parse_flag(value: Any) -> bool | None:
    """Explicit per-request value -> bool; None when absent.

    Raises ValueError on a value that is neither."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
        return value.strip().lower() in _TRUE
    raise ValueError(
        f"{XARG} must be a boolean (true/false, 1/0, on/off); got {value!r}"
    )


def resolve(explicit: Any, prompt_has_han: bool) -> bool:
    """Mask on unless the prompt has Han; an explicit value wins."""
    flag = parse_flag(explicit)
    return (not prompt_has_han) if flag is None else flag


def _content_texts(content: Any) -> Iterable[str]:
    if isinstance(content, str):
        yield content
    elif isinstance(content, Iterable) and not isinstance(content, (bytes, Mapping)):
        for part in content:
            if isinstance(part, str):
                yield part
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    yield text
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    yield text


def han_roles(messages: Iterable[Any], roles=_PROMPT_ROLES) -> tuple[str, ...]:
    """Roles (system / developer / user, in order of first sight) whose text
    carries Han."""
    found: list[str] = []
    for msg in messages:
        role, content = _role_content(msg)
        if role not in roles or role in found:
            continue
        if any(has_han(t) for t in _content_texts(content)):
            found.append(role)
    return tuple(found)


def messages_have_han(messages: Iterable[Any], roles=_PROMPT_ROLES) -> bool:
    """Han in the text of the system / developer / user turns."""
    return bool(han_roles(messages, roles))


# ------------------------------------------- the auto rule: the user's own words

# Blocks an agent client puts into a user turn that the user did not type,
# with the delimiters the clients write:
#   Hermes agent (agent/memory_context.py, build_memory_context_block):
#     recalled memory appended to the turn as "<memory-context>", the note
#     "[System note: The following is recalled memory context, NOT new user
#     input. Treat as ...]" (its text is configurable), the memories, then
#     "</memory-context>".
#   OpenClaw (auto-reply/reply/strip-inbound-meta, inbound-context-marker):
#     a header line ending in the marker U+27E6 "openclaw:ctx" U+27E7 opens a
#     block (a ```json fence, else the lines up to the first blank line);
#     "Context: <marker>" opens the trailing context, to the end of the turn;
#     "Context:" over an <active_memory_plugin> block is recalled memory.
#     Older transcripts write "<label> (untrusted metadata):", "<label>
#     (untrusted, ...):" and "Untrusted context (metadata, ...):".
#   Reminder wrappers: <system-reminder> ... </system-reminder>.
# An unterminated tag block runs to the end of the turn, as Hermes's own
# scrubber reads it. Every cut errs towards the mask staying on.
_TAG_BLOCK_RE = re.compile(
    r"<\s*(memory-context|system-reminder|active_memory_plugin)\s*>"
    r"[\s\S]*?(?:<\s*/\s*\1\s*>|\Z)",
    re.IGNORECASE,
)
_HERMES_NOTE_RE = re.compile(
    r"\[System note:\s*The following is recalled memory context,[^\]]*\]",
    re.IGNORECASE,
)
OPENCLAW_MARKER = "⟦openclaw:ctx⟧"
_OC_TRAILING = f"Context: {OPENCLAW_MARKER}"
_OC_LEGACY_BLOCK_RE = re.compile(
    r"\((?:untrusted metadata|untrusted, [^)\n]*)\):$", re.IGNORECASE
)
_OC_LEGACY_TRAILING_RE = re.compile(
    r"^Untrusted context \(metadata[^)\n]*\):$", re.IGNORECASE
)
_ACTIVE_MEMORY_OPEN = "<active_memory_plugin>"


def _openclaw_block_end(lines: list[str], i: int) -> int:
    """Index after the block whose header is lines[i]."""
    i += 1
    if i < len(lines) and lines[i].strip() == "```json":
        i += 1
        while i < len(lines) and lines[i].strip() != "```":
            i += 1
        return i + 1
    while i < len(lines) and lines[i].strip():
        i += 1
    return i


def _strip_openclaw(text: str) -> str:
    low = text.lower()
    if OPENCLAW_MARKER not in text and "untrusted" not in low and "context:" not in low:
        return text
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        nxt = lines[i + 1].strip().lower() if i + 1 < len(lines) else ""
        legacy_trailing = _OC_LEGACY_TRAILING_RE.match(s) is not None
        if (s == "Context:" or legacy_trailing) and nxt == _ACTIVE_MEMORY_OPEN:
            i += 1  # header of an active-memory block; the tag pass cuts the block
            continue
        if s == _OC_TRAILING or legacy_trailing:
            break  # trailing inbound context, to the end of the turn
        if (len(s) > 14 and s.endswith(OPENCLAW_MARKER)) or (
            _OC_LEGACY_BLOCK_RE.search(s) is not None
        ):
            i = _openclaw_block_end(lines, i)
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def user_own_text(content: Any) -> str:
    """The text of a user turn with the client-injected blocks cut out."""
    text = "\n".join(_content_texts(content))
    text = _TAG_BLOCK_RE.sub("", _strip_openclaw(text))
    return _HERMES_NOTE_RE.sub("", text)


# A request in words for Chinese output, Bulgarian and English. Conservative:
# an imperative (or "can you ...", "how do you say ...") that opens a clause,
# then within the same clause "на китайски" / "in Chinese" and their kin,
# where the target ends the clause or is followed by a word that keeps it
# about the language ("език", "characters", "please", ...); or a line that is
# only the target ("А на китайски?", "In Chinese, please."). A negation right
# before the target ("не на китайски", "not in Chinese", "rather than in
# Chinese") does not count. Talk about Chinese ("защо пишеш на китайски",
# "китайски философи", "in Chinese philosophy") does not match; neither does
# a target that opens the clause before its verb ("на китайски напиши").
_BG_IMPERATIVE = (
    r"(?:на)?пиши(?:те)?|изпиши(?:те)?|запиши(?:те)?|преведи(?:те)?"
    r"|превеждай(?:те)?|отговори(?:те)?|отговаряй(?:те)?|кажи(?:те)?"
    r"|говори(?:те)?|продължи(?:те)?|продължавай(?:те)?|обясни(?:те)?"
    r"|разкажи(?:те)?|покажи(?:те)?|дай(?:те)?"
)
_BG_2SG = (
    r"(?:на)?пишеш|изпишеш|запишеш|преведеш|отговориш|кажеш|говориш"
    r"|продължиш|обясниш|разкажеш|покажеш|дадеш"
)
_EN_VERB = (
    r"write|translate|answer|reply|respond|say|speak|continue|explain"
    r"|rewrite|tell\s+me|give\s+me|show\s+me|put\s+it"
)
_OPENERS = (
    rf"(?:{_BG_IMPERATIVE}"
    rf"|(?:можеш|можете)\s+ли\s+да\s+(?:(?:ми|ни|го|я|ги)\s+)*(?:{_BG_2SG})"
    r"|как\s+се\s+(?:казва|пише|изписва|превежда|произнася)"
    rf"|(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?(?:{_EN_VERB})"
    r"|how\s+(?:do|would|can|should)\s+(?:you|i|we|one)\s+"
    r"(?:say|write|spell|translate))\b"
)
_FILLERS = (
    r"(?:(?:моля(?:\s+(?:те|ви))?|добре|ок|ok|okay|хайде|сега|и|а|ами|also"
    r"|please|pls|kindly|now|and|then|so)[\s,]+)*"
)
_NOT_BEFORE = (
    r"(?<!\bне\s)(?<!\bnot\s)(?<!\bno\s)(?<!\bnever\s)(?<!\bникога\s)"
    r"(?<!\bthan\s)(?<!\bвместо\s)"
)
_BG_TARGET = (
    r"(?:на\s+(?:китайски(?:я)?|мандарин(?:ски)?)"
    r"(?=\s*(?:$|[^\w\s-]|(?:език|езика|йероглифи|символи|моля|ли|също"
    r"|сега|отново|пак)\b))"
    r"|по-китайски\b"
    r"|с(?:ъс)?\s+(?:китайски\s+)?йероглифи\b"
    r"|с(?:ъс)?\s+китайски\s+(?:символи|знаци)\b)"
)
_EN_TARGET = (
    r"(?:(?:in|into|to)\s+(?:simplified\s+|traditional\s+)?"
    r"(?:chinese|mandarin|cantonese)"
    r"(?=\s*(?:$|[^\w\s]|(?:please|language|script|characters|too|only"
    r"|instead|again|now|and|then|as|for|with)\b))"
    r"|(?:with|using|in)\s+(?:chinese|han)\s+characters\b"
    r"|in\s+hanzi\b)"
)
_TARGET = rf"\b{_NOT_BEFORE}(?:{_BG_TARGET}|{_EN_TARGET})"
_ASK_RE = re.compile(
    rf"(?:^|[.!?;:,])\s*{_FILLERS}{_OPENERS}[^.!?;:,\n]{{0,80}}?{_TARGET}"
    rf"|^\s*{_FILLERS}{_TARGET}\s*(?:,?\s*(?:моля|please))?\s*[.!?]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def asked_for_chinese(text: str) -> bool:
    """The text asks in words for an answer in Chinese (see ``_ASK_RE``)."""
    return _ASK_RE.search(text) is not None


def _role_content(msg: Any) -> tuple[Any, Any]:
    if isinstance(msg, Mapping):
        return msg.get("role"), msg.get("content")
    return getattr(msg, "role", None), getattr(msg, "content", None)


def latest_user_text(messages: Iterable[Any]) -> str:
    """The user's own text in the latest user message ("" when none)."""
    last = None
    for msg in messages:
        role, content = _role_content(msg)
        if role == "user":
            last = content
    return "" if last is None else user_own_text(last)


def auto_rule(messages: Iterable[Any]) -> str | None:
    """Why "auto" turns the mask off for this chat, or None (mask on)."""
    text = latest_user_text(messages)
    if has_han(text):
        return "auto-latest-user-han"
    if asked_for_chinese(text):
        return "auto-asked-chinese"
    return None


class OffLog:
    """INFO lines for requests whose mask is off, rate-limited.

    At most ``burst`` per-request lines per ``window`` seconds; the excess is
    counted by rule, and the first off-request of the next window logs one
    summary line (this window's excess plus the totals since start). Nothing
    but the request id, the rule and the role is ever logged: no prompt text.
    """

    def __init__(
        self,
        burst: int = 5,
        window: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.burst, self.window, self.clock = burst, window, clock
        self._lock = threading.Lock()
        self._start: float | None = None
        self._emitted = 0
        self.suppressed: dict[str, int] = {}
        self.totals: dict[str, int] = {}

    def record(self, request_id: str | None, rule: str, role: str | None) -> None:
        now = self.clock()
        with self._lock:
            self.totals[rule] = self.totals.get(rule, 0) + 1
            summary = None
            if self._start is None or now - self._start >= self.window:
                if self.suppressed:
                    summary = (dict(self.suppressed), dict(self.totals))
                self._start, self._emitted, self.suppressed = now, 0, {}
            if self._emitted >= self.burst:
                self.suppressed[rule] = self.suppressed.get(rule, 0) + 1
                return
            self._emitted += 1
        if summary is not None:
            more, totals = summary
            logger.info(
                "B70-CJK mask off, not listed in the last window: %s; "
                "totals since start: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(more.items())),
                ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            )
        rid = (request_id or "-")[:64]
        logger.info(
            "B70-CJK mask off for request %r: rule=%s role=%s", rid, rule, role or "-"
        )


_OFF_LOG = OffLog()


def chat_extra_args(
    extra_args: dict[str, Any],
    messages: Iterable[Any],
    model_extra: Any,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Resolve the chat request's flag into ``extra_args`` (a new dict).

    Rules, first match wins: "explicit" (the request's own flag), the server
    default "on" / "off" ("default-off" when off), and in "auto" the latest
    user message in the user's own words: "auto-latest-user-han" (it carries
    Han) or "auto-asked-chinese" (it asks for Chinese in words)."""
    explicit = extra_args.get(XARG)
    if explicit is None and isinstance(model_extra, Mapping):
        explicit = model_extra.get(XARG)
    rule, role = None, None
    if explicit is not None:
        flag = resolve(explicit, False)
        rule = "explicit"
    elif _DEFAULT_MODE != "auto":
        flag = _DEFAULT_MODE == "on"
        rule = "default-off"
    else:
        rule = auto_rule(messages)
        flag, role = rule is None, "user"
    if not flag:
        _OFF_LOG.record(request_id, rule, role)
    out = dict(extra_args)
    out[XARG] = flag
    return out


# ---------------------------------------------------------------- vocabulary


def _bytes_to_unicode() -> dict[str, int]:
    """GPT-2 byte-level alphabet, inverted: printable char -> byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_U2B = _bytes_to_unicode()
_SP_BYTE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def token_bytes(token: str) -> bytes:
    """Raw bytes of a vocabulary string (byte-level BPE or SentencePiece)."""
    if all(c in _U2B for c in token):
        return bytes(_U2B[c] for c in token)
    m = _SP_BYTE.match(token)
    if m:
        return bytes([int(m.group(1), 16)])
    return token.replace("▁", " ").encode("utf-8")


def _overlaps(lo: int, hi: int, ranges) -> bool:
    return any(lo <= r_hi and r_lo <= hi for r_lo, r_hi in ranges)


def _trailing_lead_masked(b: bytes) -> bool:
    """Incomplete trailing 3-byte sequence that can only finish as Han."""
    n = len(b)
    for k in (1, 2):  # lead at n-1 (lone) or n-2 (lead + one continuation)
        i = n - k
        if i < 0:
            break
        lead = b[i]
        if lead in _LEADS and all(0x80 <= x <= 0xBF for x in b[i + 1 :]):
            base = (lead & 0x0F) << 12
            if k == 1:
                lo, hi = base, base + 0xFFF
            else:
                lo = base | ((b[i + 1] & 0x3F) << 6)
                hi = lo + 0x3F
            return _overlaps(lo, hi, HAN_RANGES) and not _overlaps(lo, hi, KANA_RANGES)
    return False


def classify(b: bytes) -> str | None:
    """'han', 'partial' or None (allowed)."""
    try:
        text = b.decode("utf-8")
        return "han" if has_han(text) else None
    except UnicodeDecodeError:
        if has_han(b.decode("utf-8", "ignore")):
            return "han"
        return "partial" if _trailing_lead_masked(b) else None


def build_mask(tokenizer: Any, vocab_size: int):
    """Bool mask [vocab_size] (numpy) and counts, from the tokenizer vocabulary."""
    import numpy as np

    mask = np.zeros(vocab_size, dtype=bool)
    vocab = tokenizer.get_vocab()
    skip = set(getattr(tokenizer, "all_special_ids", None) or ())
    with contextlib.suppress(Exception):
        skip |= set(tokenizer.get_added_vocab().values())
    counts = {"han": 0, "partial": 0, "vocab": len(vocab), "kana_only": 0}
    for tok, idx in vocab.items():
        if idx in skip or not 0 <= idx < vocab_size:
            continue
        b = token_bytes(tok)
        kind = classify(b)
        if kind is not None:
            mask[idx] = True
            counts[kind] += 1
    counts["masked"] = int(mask.sum())
    return mask, counts


_PROMPT_MASK: Any = None


def prompt_has_han_tokens(tokenizer: Any, token_ids: Any) -> bool:
    """Fallback scan for non-chat prompts (the whole prompt, history included):
    a token whose text has Han, or a byte token that is not valid UTF-8 alone
    and holds a 0xE3-0xE9 byte (a character split across tokens may be Han)."""
    global _PROMPT_MASK
    import numpy as np

    if token_ids is None or tokenizer is None:
        return False
    if _PROMPT_MASK is None:
        vocab = tokenizer.get_vocab()
        size = max(vocab.values()) + 1 if vocab else 0
        m = np.zeros(size, dtype=bool)
        for tok, idx in vocab.items():
            b = token_bytes(tok)
            try:
                m[idx] = has_han(b.decode("utf-8"))
            except UnicodeDecodeError:
                m[idx] = classify(b) is not None or any(x in _LEADS for x in b)
        _PROMPT_MASK = m
    ids = np.asarray(token_ids, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < _PROMPT_MASK.shape[0])]
    return bool(_PROMPT_MASK[ids].any())

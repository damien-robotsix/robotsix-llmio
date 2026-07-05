"""Claude Agent SDK transient signatures, layered on the core set.

The SDK drives the ``claude`` CLI as a subprocess; a flaky spawn, a dropped
control-protocol connection, or a malformed JSON frame from the CLI is an
infrastructure hiccup that a re-run usually clears — treat those as transient.
"""

from __future__ import annotations

from ..core.retry import is_transient as _core_is_transient

# Subprocess/transport failures raised by claude_agent_sdk. Matched by type
# name so importing this module never requires the SDK to be installed.
_SDK_TRANSIENT_NAMES = {
    "CLIConnectionError",  # lost the control-protocol connection to the CLI
    "CLIJSONDecodeError",  # CLI emitted a malformed JSON frame
    "ProcessError",  # the CLI process exited non-zero
    "ProcessLookupError",  # the subprocess vanished mid-stream
    "ClaudeSDKQueryTimeout",  # our per-call wall-clock cap tripped (stalled run)
}

# The SDK's wording when its agent loop exhausts ``max_turns`` without producing
# a final answer ("Reached maximum number of turns (N)").
_TURN_LIMIT_SIGNATURE = "maximum number of turns"

# claude_agent_sdk collapses a self-contradictory frame
# (is_error=True, errors=[], subtype="success") into the message
# "; ".join(errors) or str(subtype) -> "success". A re-run clears it.
_DEGENERATE_SUCCESS_SIGNATURE = "returned an error result: success"

# The Claude CLI's own wording when a tier's usage credits are exhausted. This
# arrives as ordinary assistant TEXT inside an is_error=True ResultMessage, not
# as a raised exception — best-effort string matching, same tradeoff as the
# other signatures here, since the CLI's exact phrasing isn't a documented
# contract. Unlike a degenerate success, a re-run at the SAME tier cannot
# help — the credits stay exhausted until they reset — so this is excluded
# from "transient" (see is_claude_sdk_transient) and must be handled by
# falling back to a different tier instead.
_USAGE_EXHAUSTED_SIGNATURE = "out of usage credits"


def is_usage_exhausted_text(text: str) -> bool:
    """True if *text* (assistant-visible turn text) reports exhausted usage
    credits for the current tier, per the Claude CLI's own wording."""
    return _USAGE_EXHAUSTED_SIGNATURE in text.lower()


def is_claude_sdk_degenerate_success(exc: BaseException) -> bool:
    """True if *exc* (or anything in its cause/context chain) is the upstream
    ``claude_agent_sdk`` degenerate-success frame — a self-contradictory
    ``is_error=True``/``errors=[]``/``subtype="success"`` result that collapses
    into the bare message ``"Claude Code returned an error result: success"``.
    A re-run clears it, so it should be treated as transient. Matched
    case-insensitively, walking the bounded cause/context chain like the other
    helpers."""
    cur: BaseException | None = exc
    seen = 0
    while cur is not None and seen < 10:
        if _DEGENERATE_SUCCESS_SIGNATURE in str(cur).lower():
            return True
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False


def is_claude_sdk_usage_exhausted(exc: BaseException) -> bool:
    """True if *exc* (or anything in its cause/context chain) is the
    dedicated ``ClaudeSDKUsageExhaustedError`` — a tier's usage credits are
    exhausted. Matched by name so this stays free of an import cycle with
    ``model.py``, walking the bounded cause/context chain like the other
    helpers."""
    cur: BaseException | None = exc
    seen = 0
    while cur is not None and seen < 10:
        if type(cur).__name__ == "ClaudeSDKUsageExhaustedError":
            return True
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False


def is_claude_sdk_turn_limit(exc: BaseException) -> bool:
    """True if *exc* (or anything in its cause/context chain) is the Claude Agent
    SDK turn-cap failure — either the dedicated ``ClaudeSDKTurnLimitError`` or the
    raw SDK message. Matched by name/string so this stays free of the SDK and the
    model module (no import cycle)."""
    cur: BaseException | None = exc
    seen = 0
    while cur is not None and seen < 10:
        if type(cur).__name__ == "ClaudeSDKTurnLimitError":
            return True
        if _TURN_LIMIT_SIGNATURE in str(cur).lower():
            return True
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False


def is_claude_sdk_transient(exc: BaseException) -> bool:
    """Core transient set OR a Claude Agent SDK subprocess/transport failure,
    walking the cause/context chain for the latter.

    The turn-cap failure and usage-exhaustion are explicitly excluded — and
    checked FIRST, so they win even when the CLI surfaces them as a
    (normally-transient) ``ProcessError``. Retrying either at the same tier
    would just repeat the identical failure, so both must fail loudly rather
    than burn retries and end in an opaque error."""
    if is_claude_sdk_turn_limit(exc):
        return False
    if is_claude_sdk_usage_exhausted(exc):
        return False
    if is_claude_sdk_degenerate_success(exc):
        return True
    if _core_is_transient(exc):
        return True
    cur: BaseException | None = exc
    seen = 0
    while cur is not None and seen < 10:
        if type(cur).__name__ in _SDK_TRANSIENT_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False

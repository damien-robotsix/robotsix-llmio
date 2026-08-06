"""Claude Agent SDK transient signatures, layered on the core set.

The SDK drives the ``claude`` CLI as a subprocess; a flaky spawn, a dropped
control-protocol connection, or a malformed JSON frame from the CLI is an
infrastructure hiccup that a re-run usually clears — treat those as transient.
"""

from __future__ import annotations

from ..core.retry import _walk_cause_chain
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


# The Claude CLI's wording when the Anthropic API rejects the request itself
# ("API Error: 400 `task_budget.total` must be at least 20,000 tokens..."). Like
# usage exhaustion this arrives as ordinary assistant TEXT inside an
# is_error=True ResultMessage, not as a raised exception — same best-effort
# string-matching tradeoff, since the CLI's phrasing isn't a documented
# contract. Scoped to 400 deliberately: a 400 is request validation and is
# perfectly reproducible, so retrying is pure waste, whereas 429 and 5xx are
# genuinely retryable and must stay transient.
_PERMANENT_API_ERROR_SIGNATURE = "api error: 400"


# The one 400 above that is NOT the caller's fault and IS worth re-running.
# ``task_budget`` is accepted only by a subset of models; every other model
# rejects the request outright with this wording. Because the parameter is one
# *this transport* adds (from the tier's ``max_tokens``), the fix is to drop it
# and retry rather than to fail the caller — see
# ``_task_budget.mark_task_budget_unsupported``. Matched on the distinctive
# phrase rather than the full sentence so minor rewording upstream still hits.
_TASK_BUDGET_UNSUPPORTED_SIGNATURE = "does not support user-configurable task budgets"


# The Claude CLI's wording when the stored OAuth credential is rejected
# ("Failed to authenticate. API Error: 401 OAuth access token has expired.
# Re-authenticate to continue."). Same delivery shape as the two signatures
# above — assistant-visible TEXT inside an is_error=True ResultMessage, not a
# raised exception — so the same best-effort string-matching tradeoff applies.
#
# Scoped to 401 and the CLI's own auth wording, deliberately: a 401 means the
# credential itself is dead, so every retry at this tier re-sends against the
# same dead credential. 403 is left out because it is authorisation (a scope or
# entitlement problem) rather than a bad credential, and 429/5xx stay transient.
_AUTH_ERROR_SIGNATURES = (
    "api error: 401",
    "oauth access token has expired",
    "failed to authenticate",
)


def is_auth_error_text(text: str) -> bool:
    """True if *text* (assistant-visible turn text) reports an authentication
    failure — an expired or rejected Claude OAuth credential. No re-run at this
    tier can clear it; a human must re-authenticate."""
    lowered = text.lower()
    return any(sig in lowered for sig in _AUTH_ERROR_SIGNATURES)


def is_claude_sdk_auth_error(exc: BaseException) -> bool:
    """True if *exc* (or anything in its cause/context chain) is the dedicated
    ``ClaudeSDKAuthError``. Matched by name so this stays free of an import
    cycle with the error module, walking the bounded cause/context chain like
    the other helpers."""
    return any(
        type(cur).__name__ == "ClaudeSDKAuthError" for cur in _walk_cause_chain(exc)
    )


def is_permanent_api_error_text(text: str) -> bool:
    """True if *text* (assistant-visible turn text) reports an API ``400`` —
    a request-validation rejection that a re-run cannot clear."""
    return _PERMANENT_API_ERROR_SIGNATURE in text.lower()


def is_task_budget_unsupported_text(text: str) -> bool:
    """True if *text* reports the one ``400`` that IS clearable by re-running.

    ``task_budget`` is only accepted by a subset of models. Sending it to any
    other one is rejected with this message, and unlike every other 400 here
    the offending parameter is one this transport added on the caller's behalf
    — so dropping it and retrying is both possible and correct, where retrying
    a genuinely malformed request would be waste. Callers pair this with
    :func:`~robotsix_llmio.claude_sdk._task_budget.mark_task_budget_unsupported`
    so the retry happens once per model rather than once per call.
    """
    return _TASK_BUDGET_UNSUPPORTED_SIGNATURE in text.lower()


def is_task_budget_unsupported_error(exc: BaseException) -> bool:
    """True if *exc* (or anything in its cause/context chain) carries the
    task-budget-unsupported rejection. The raise sites fold the offending turn
    text into the exception message, so matching the message is what's
    available — same best-effort string tradeoff as the signatures above."""
    return any(
        is_task_budget_unsupported_text(str(cur)) for cur in _walk_cause_chain(exc)
    )


def is_claude_sdk_permanent_api_error(exc: BaseException) -> bool:
    """True if *exc* (or anything in its cause/context chain) is the dedicated
    ``ClaudeSDKPermanentAPIError``. Matched by name so this stays free of an
    import cycle with the error module, walking the bounded cause/context chain
    like the other helpers."""
    return any(
        type(cur).__name__ == "ClaudeSDKPermanentAPIError"
        for cur in _walk_cause_chain(exc)
    )


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
    return any(
        _DEGENERATE_SUCCESS_SIGNATURE in str(cur).lower()
        for cur in _walk_cause_chain(exc)
    )


def is_claude_sdk_usage_exhausted(exc: BaseException) -> bool:
    """True if *exc* (or anything in its cause/context chain) is the
    dedicated ``ClaudeSDKUsageExhaustedError`` — a tier's usage credits are
    exhausted. Matched by name so this stays free of an import cycle with
    ``model.py``, walking the bounded cause/context chain like the other
    helpers."""
    return any(
        type(cur).__name__ == "ClaudeSDKUsageExhaustedError"
        for cur in _walk_cause_chain(exc)
    )


def is_claude_sdk_turn_limit(exc: BaseException) -> bool:
    """True if *exc* (or anything in its cause/context chain) is the Claude Agent
    SDK turn-cap failure — either the dedicated ``ClaudeSDKTurnLimitError`` or the
    raw SDK message. Matched by name/string so this stays free of the SDK and the
    model module (no import cycle)."""
    return any(
        type(cur).__name__ == "ClaudeSDKTurnLimitError"
        or _TURN_LIMIT_SIGNATURE in str(cur).lower()
        for cur in _walk_cause_chain(exc)
    )


def is_claude_sdk_transient(exc: BaseException) -> bool:
    """Core transient set OR a Claude Agent SDK subprocess/transport failure,
    walking the cause/context chain for the latter.

    The turn-cap failure, usage-exhaustion, API request-validation (400)
    rejections, and authentication (401) failures are explicitly excluded — and
    checked FIRST, so they win even when the CLI surfaces them as a
    (normally-transient) ``ProcessError`` or as the degenerate-success frame.
    Retrying any of them would just repeat the identical failure, so all four
    must fail loudly rather than burn retries and end in an opaque error."""
    if is_claude_sdk_turn_limit(exc):
        return False
    if is_claude_sdk_usage_exhausted(exc):
        return False
    if is_claude_sdk_permanent_api_error(exc):
        return False
    if is_claude_sdk_auth_error(exc):
        return False
    if is_claude_sdk_degenerate_success(exc):
        return True
    if _core_is_transient(exc):
        return True
    return any(
        type(cur).__name__ in _SDK_TRANSIENT_NAMES for cur in _walk_cause_chain(exc)
    )

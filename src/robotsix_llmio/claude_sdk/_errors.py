"""Claude Agent SDK error classes — shared by transport, stream, and transient
layers.

These are the library-facing wrappers for Claude Agent SDK failures. All
inherit from :class:`~robotsix_llmio.exceptions.RobotsixLLMIOError` so
consumers catch a single base without importing SDK internals.
"""

from __future__ import annotations

from ..exceptions import RobotsixLLMIOError


class ClaudeSDKTurnLimitError(RobotsixLLMIOError):
    """The Claude Agent SDK loop hit its turn cap (``_MAX_TURNS``) without
    returning a final answer.

    A hard failure surfaced loudly: the agent loop did not converge, and the
    identical request would just loop to the cap again — so it is never treated
    as transient (see
    :func:`~robotsix_llmio.claude_sdk.transient.is_claude_sdk_transient`)."""


class ClaudeSDKQueryTimeout(RobotsixLLMIOError):
    """A single Claude Agent SDK ``query()`` exceeded the per-call wall-clock cap
    (:data:`~robotsix_llmio.core.constants.SDK_QUERY_TIMEOUT`).

    Unlike the turn-limit failure, a timeout is a *stall* (the subprocess made no
    progress — often startup contention), not a non-converging loop. Re-running
    usually clears it, so it is treated as **transient** and retried by the
    bounded retry (matched by name in
    :data:`~robotsix_llmio.claude_sdk.transient._SDK_TRANSIENT_NAMES`)."""


class ClaudeSDKUsageExhaustedError(RobotsixLLMIOError):
    """The Claude subscription has exhausted its usage credits for the
    ``ClaudeAgentOptions.model`` tier this call used.

    The SDK reports this as a normal-looking ``ResultMessage`` (``is_error=True``,
    often ``subtype="success"``) carrying the assistant-visible text "You're out
    of usage credits" rather than raising — so left unhandled, that text would be
    returned as if it were a genuine reply. A re-run at the *same* tier cannot
    help (the credits are exhausted until they reset), so this is never treated
    as transient (see
    :func:`~robotsix_llmio.claude_sdk.transient.is_claude_sdk_transient`) —
    callers should catch it and fall back to a different capability tier
    instead (e.g. via
    :func:`~robotsix_llmio.core.tier_fallback.acall_with_tier_fallback`)."""


class ClaudeSDKAPIError(RobotsixLLMIOError):
    """A terminal Claude Agent SDK transport or process failure that survived
    the transient classification and retry loop.

    Wraps a raw ``claude_agent_sdk`` exception (e.g. ``CLIConnectionError``,
    ``ProcessError``) whose transient retries were exhausted — the original
    is preserved as ``__cause__`` so the transient classifier can inspect it.
    All library consumers can catch this (or its base
    :class:`RobotsixLLMIOError`) without importing ``claude_agent_sdk``
    internals."""

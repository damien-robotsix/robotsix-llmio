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


class ClaudeSDKPermanentAPIError(RobotsixLLMIOError):
    """The Anthropic API rejected the request itself with a ``400`` — a
    malformed or out-of-range parameter, not an infrastructure hiccup.

    Like usage exhaustion, the CLI surfaces this as assistant-visible text
    ("API Error: 400 ...") inside an ``is_error=True`` ``ResultMessage`` rather
    than raising, and ``claude_agent_sdk`` then collapses that frame into its
    generic degenerate-success message. Left unclassified it therefore looks
    *transient*: the bounded retry burns every attempt re-sending the identical
    (still-invalid) request, and the exhausted retry surfaces as an opaque
    transport failure that callers may charitably read as "ran, changed
    nothing". A 400 is deterministic — the same request always reproduces it —
    so this is never treated as transient (see
    :func:`~robotsix_llmio.claude_sdk.transient.is_claude_sdk_transient`) and
    must fail loudly so the offending parameter gets fixed."""


class ClaudeSDKAuthError(RobotsixLLMIOError):
    """The ``claude`` CLI could not authenticate — the stored OAuth credential
    is expired, revoked, or otherwise rejected with a ``401``.

    Surfaced exactly like usage exhaustion and the ``400``: the CLI streams the
    failure as assistant-visible text ("Failed to authenticate. API Error: 401
    OAuth access token has expired.") inside an ``is_error=True``
    ``ResultMessage``, which ``claude_agent_sdk`` then collapses into its
    generic degenerate-success message. Left unclassified it looks *transient*,
    so the bounded retry re-sends the identical request against the same dead
    credential and the exhausted retry surfaces as an opaque transport failure —
    the exact shape that made a plain expired token read as an SDK bug.

    Re-running at the *same* tier can never help: the credential stays invalid
    until a human re-authenticates. It is therefore never treated as transient
    (see :func:`~robotsix_llmio.claude_sdk.transient.is_claude_sdk_transient`).
    Because the credential is per-provider rather than per-request, callers
    should treat it like usage exhaustion and fall back to a *different*
    capability tier (e.g. via
    :func:`~robotsix_llmio.core.tier_fallback.acall_with_tier_fallback`), which
    keeps a keyed provider serving while the Claude credential is dead."""


class ClaudeSDKAPIError(RobotsixLLMIOError):
    """A terminal Claude Agent SDK transport or process failure that survived
    the transient classification and retry loop.

    Wraps a raw ``claude_agent_sdk`` exception (e.g. ``CLIConnectionError``,
    ``ProcessError``) whose transient retries were exhausted — the original
    is preserved as ``__cause__`` so the transient classifier can inspect it.
    All library consumers can catch this (or its base
    :class:`RobotsixLLMIOError`) without importing ``claude_agent_sdk``
    internals."""

"""Base exception hierarchy for robotsix-llmio."""


class RobotsixLLMIOError(Exception):
    """Base exception for all robotsix-llmio errors.

    All custom exceptions raised by this library (transport failures,
    configuration errors, provider-specific errors) inherit from this
    class so callers can catch them with a single ``except`` clause.
    """


class ProviderExhaustedError(RobotsixLLMIOError):
    """Marker base for a *provider-wide* exhaustion — a backend is out of
    capacity until a quota resets, and **every** tier level backed by that
    same provider shares the exhausted capacity.

    The canonical case is a Claude subscription out of usage credits
    (:class:`~robotsix_llmio.claude_sdk._errors.ClaudeSDKUsageExhaustedError`):
    the sibling Claude tiers all draw on the one subscription, so once one is
    exhausted the rest are too. The tier-fallback loop
    (:func:`~robotsix_llmio.core.failover.acall_with_failover`)
    treats this specially: on such a failure it skips *all* remaining levels
    on the exhausted provider in one step, instead of wasting fallback hops
    walking sibling tiers that are already spent.

    This is deliberately distinct from a per-run rate-limit
    (``pydantic_ai.UsageLimitExceeded``), which is a single-run budget cap
    rather than a provider-wide outage — rate-limited runs still fall back
    level-by-level as normal.
    """

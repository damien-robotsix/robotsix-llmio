"""Consumer-facing factory — create a provider from a capability *level*.

This is the single entry-point consumers use to obtain a working model/agent.
Callers can supply just ``level`` (and optionally ``transport`` to override the
level-based provider) — no direct provider-class or ``claude_agent_sdk`` import
is ever needed.

Built on top of :func:`~robotsix_llmio.core.factory.get_provider` and the
:class:`~robotsix_llmio.config.tier.TierConfig` schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.factory import get_provider

if TYPE_CHECKING:
    from ..core.provider import LLMProvider

from .transport import TRANSPORT_ALIASES


def create_model(
    *,
    level: int = 1,
    transport: str | None = None,
    tier_config: "TierConfig | None" = None,
    **provider_kwargs: Any,
) -> LLMProvider:
    """Create a provider instance for the given capability *level*.

    When *transport* is ``None`` (the default), the provider is resolved
    from ``tier_config.for_level(level).provider``.  When *transport* is
    supplied it overrides the level-based choice — useful for pinning a
    specific provider regardless of tier (e.g. forcing OpenRouter even at
    level 3 where ``"claude-sdk"`` is the default).

    Parameters
    ----------
    level:
        Capability level (1, 2, or 3).  Level 1 selects the cheap/fast model;
        levels 2 and 3 select progressively more capable defaults.
    transport:
        Optional consumer-facing transport alias — one of ``"claude-sdk"`` or
        ``"openrouter[deepseek]"``.  Mapped to a provider registry name via
        :data:`~.transport.TRANSPORT_ALIASES`.  When ``None``, the provider
        is resolved from *tier_config*.
    tier_config:
        Optional :class:`~.tier.TierConfig` to resolve the provider + model.
        When ``None``, a default is built from baked module-level defaults
        (:data:`~.tier.LEVEL1_DEFAULT`, :data:`~.tier.LEVEL2_DEFAULT`,
        :data:`~.tier.LEVEL3_DEFAULT`).
    **provider_kwargs:
        Forwarded to the provider constructor (e.g. ``api_key=...`` for the
        OpenRouter provider).  These override any ``provider_kwargs`` from
        the tier config.

    Returns
    -------
    LLMProvider
        A fully-instantiated provider ready for :meth:`~LLMProvider.build_agent`
        or :meth:`~LLMProvider.new_model` calls.

    Raises
    ------
    ValueError
        If *level* is not 1, 2, or 3, or if *transport* (when supplied) is
        not a recognised alias.
    ImportError
        If the provider's optional extra is not installed.

    Example
    -------
    .. code-block:: python

        from robotsix_llmio.config import create_model

        # Level 3 → Claude SDK by default.
        provider = create_model(level=3)
        agent = provider.build_agent(
            level=3,
            system_prompt="You are a helpful assistant.",
        )

        # Force OpenRouter at level 3 with an explicit transport.
        provider = create_model(level=3, transport="openrouter[deepseek]")
    """
    if tier_config is None:
        from .tier import LEVEL1_DEFAULT, LEVEL2_DEFAULT, LEVEL3_DEFAULT, TierConfig

        tier_config = TierConfig(
            level1=LEVEL1_DEFAULT,
            level2=LEVEL2_DEFAULT,
            level3=LEVEL3_DEFAULT,
        )

    tlc = tier_config.for_level(level)

    # Merge: tier-config provider_kwargs as base, explicit kwargs override.
    merged_kwargs: dict[str, Any] = {**tlc.provider_kwargs, **provider_kwargs}

    if transport is not None:
        # Explicit transport overrides the level-based provider.
        resolved_provider = TRANSPORT_ALIASES.get(transport)
        if resolved_provider is None:
            known = ", ".join(sorted(TRANSPORT_ALIASES))
            raise ValueError(
                f"Unknown transport {transport!r}. Known transports: {known}."
            )
        return get_provider(provider=resolved_provider, **merged_kwargs)

    # No transport → resolve provider from tier_config.
    return get_provider(provider=tlc.provider, **merged_kwargs)

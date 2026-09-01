"""Consumer-facing factory — create a provider from a capability *level*.

This is the single entry-point consumers use to obtain a working model/agent.
Callers supply just ``level`` — no direct provider-class or
``claude_agent_sdk`` import is ever needed.  The provider is derived from the
combined ``provider-model`` identifier on the tier config.

Built on top of :func:`~robotsix_llmio.core.factory.get_provider_for_identifier`
and the :class:`~robotsix_llmio.config.tier.TierConfig` schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.factory import get_provider_for_identifier

if TYPE_CHECKING:
    from ..core.provider import LLMProvider
    from .tier import ProviderSlotName, TierConfig


def create_model(
    *,
    level: int = 1,
    tier_config: TierConfig | None = None,
    slot: ProviderSlotName | None = None,
    **provider_kwargs: Any,
) -> LLMProvider:
    """Create a provider instance for the given capability *level*.

    The provider is derived from the combined ``provider-model`` identifier
    on the tier config — the identifier's prefix drives lazy backend import
    via :func:`~robotsix_llmio.core.factory.get_provider_for_identifier`.

    Args:
        level: Capability level (1, 2, or 3).  Level 1 is the cheap
            frequent tier, level 2 the workhorse, level 3 the frontier
            (``claudeSDK-claude-fable-5`` by default).  Resolution honours
            the provider slot the failover tracker currently designates as
            active (see :mod:`robotsix_llmio.core.failover`).
        tier_config: Optional :class:`~.tier.TierConfig` to resolve the
            provider + model.  When ``None``, a default ``TierConfig()``
            is built; its ``default_factory`` lambdas produce independent
            deep copies of the baked defaults.
        slot: Explicit provider slot (``"default"`` or ``"fallback"``) to
            resolve against; ``None`` follows the failover tracker's
            active slot.
        **provider_kwargs: Forwarded to the provider constructor (e.g.
            ``api_key=...`` for the OpenRouter provider).  These override
            any ``provider_kwargs`` from the tier config.

    Returns:
        LLMProvider: A fully-instantiated provider ready for
            :meth:`~LLMProvider.build_agent` or
            :meth:`~LLMProvider.new_model` calls.

    Raises:
        ValueError: If *level* is not 1, 2, or 3.
        ImportError: If the provider's optional extra is not installed.

    Example:

        .. code-block:: python

        from robotsix_llmio.config import create_model

        # Level 2 → Claude SDK by default (identifier "claudeSDK-opus").
        provider = create_model(level=2)
        agent = provider.build_agent(
            level=2,
            system_prompt="You are a helpful assistant.",
        )

    """
    if tier_config is None:
        from .tier import TierConfig

        tier_config = TierConfig()

    tlc = tier_config.for_level(level, slot=slot)

    # Merge: tier-config provider_kwargs as base, explicit kwargs override.
    merged_kwargs: dict[str, Any] = {**tlc.provider_kwargs, **provider_kwargs}

    if tlc.max_tokens is not None:
        merged_kwargs.setdefault("max_tokens", tlc.max_tokens)

    # Derive provider from the tier config's combined provider-model
    # identifier.  The identifier's prefix drives the lazy backend import;
    # the model name is the backend's concern.
    return get_provider_for_identifier(tlc.model, **merged_kwargs)

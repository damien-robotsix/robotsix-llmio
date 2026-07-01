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
    from .tier import TierConfig


def create_model(
    *,
    level: int = 1,
    tier_config: TierConfig | None = None,
    **provider_kwargs: Any,
) -> LLMProvider:
    """Create a provider instance for the given capability *level*.

    The provider is derived from the combined ``provider-model`` identifier
    on the tier config — the identifier's prefix drives lazy backend import
    via :func:`~robotsix_llmio.core.factory.get_provider_for_identifier`.

    Parameters
    ----------
    level:
        Capability level (1, 2, 3, or 4).  Level 1 selects the cheap/fast
        model; levels 2-4 select progressively more capable defaults (level
        4 is the frontier tier, ``claudeSDK-claude-fable-5`` by default).
    tier_config:
        Optional :class:`~.tier.TierConfig` to resolve the provider + model.
        When ``None``, a default is built from baked module-level defaults
        (:data:`~.tier.LEVEL1_DEFAULT`, :data:`~.tier.LEVEL2_DEFAULT`,
        :data:`~.tier.LEVEL3_DEFAULT`, :data:`~.tier.LEVEL4_DEFAULT`).
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
        If *level* is not 1, 2, 3, or 4.
    ImportError
        If the provider's optional extra is not installed.

    Example
    -------
    .. code-block:: python

        from robotsix_llmio.config import create_model

        # Level 3 → Claude SDK by default (identifier "claudeSDK-opus").
        provider = create_model(level=3)
        agent = provider.build_agent(
            level=3,
            system_prompt="You are a helpful assistant.",
        )
    """
    if tier_config is None:
        from .tier import (
            LEVEL1_DEFAULT,
            LEVEL2_DEFAULT,
            LEVEL3_DEFAULT,
            LEVEL4_DEFAULT,
            TierConfig,
        )

        tier_config = TierConfig(
            level1=LEVEL1_DEFAULT,
            level2=LEVEL2_DEFAULT,
            level3=LEVEL3_DEFAULT,
            level4=LEVEL4_DEFAULT,
        )

    tlc = tier_config.for_level(level)

    # Merge: tier-config provider_kwargs as base, explicit kwargs override.
    merged_kwargs: dict[str, Any] = {**tlc.provider_kwargs, **provider_kwargs}

    # Derive provider from the tier config's combined provider-model
    # identifier.  The identifier's prefix drives the lazy backend import;
    # the model name is the backend's concern.
    return get_provider_for_identifier(tlc.model, **merged_kwargs)

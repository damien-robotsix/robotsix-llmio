"""Tier configuration schema — three configurable provider+model bindings.

This module defines the *data model only*.  Wiring it into the provider
factory, ``build_agent``, and ``new_model`` is a follow-up concern.

The existing two-value :class:`~robotsix_llmio.core.Tier` enum is **not**
modified — callers that pass ``Tier.DEFAULT`` / ``Tier.CHEAP`` continue
to work unchanged.  The :data:`LEGACY_TIER_MAP` dictionary provides an
explicit, documented deprecation path.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from robotsix_llmio.core.provider import Tier as LegacyTier

# --------------------------------------------------------------------------- #
#  TierLevel — three configuration tiers                                      #
# --------------------------------------------------------------------------- #


class TierLevel(StrEnum):
    """Three-tier configuration selector.

    | Member   | Value     | Purpose                                   |
    |----------|-----------|-------------------------------------------|
    | LEVEL1   | ``level1`` | Cheap, obvious, repetitive tasks          |
    | LEVEL2   | ``level2`` | Intermediate (e.g. implementing code)     |
    | LEVEL3   | ``level3`` | High-level organisation and planning      |
    """

    LEVEL1 = "level1"
    LEVEL2 = "level2"
    LEVEL3 = "level3"


# --------------------------------------------------------------------------- #
#  TierLevelConfig — a single tier's provider+model binding                   #
# --------------------------------------------------------------------------- #


class TierLevelConfig(BaseModel):
    """A single tier's provider and model binding.

    Describes which provider backend to use and which model name/alias
    that provider resolves for a given :class:`TierLevel`.
    """

    provider: str = Field(
        description=(
            "Registry name passed to ``get_provider()`` — e.g. "
            "``'openrouter-deepseek'`` or ``'claude-sdk'``."
        ),
    )
    model: str = Field(
        description=(
            "Model name or alias the provider resolves — e.g. "
            "``'deepseek/deepseek-v4-flash'`` for OpenRouter or "
            "``'opus'`` for the Claude SDK."
        ),
    )
    provider_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments forwarded to the provider constructor "
            "(e.g. ``base_url`` for the OpenRouter provider). Defaults to ``{}``."
        ),
    )


# --------------------------------------------------------------------------- #
#  Baked defaults                                                             #
# --------------------------------------------------------------------------- #

LEVEL1_DEFAULT = TierLevelConfig(
    provider="openrouter-deepseek",
    model="deepseek/deepseek-v4-flash",
)

LEVEL2_DEFAULT = TierLevelConfig(
    provider="openrouter-deepseek",
    model="deepseek/deepseek-v4-pro",
)

LEVEL3_DEFAULT = TierLevelConfig(
    provider="claude-sdk",
    model="opus",
)


# --------------------------------------------------------------------------- #
#  TierConfig — aggregates three TierLevelConfig slots                        #
# --------------------------------------------------------------------------- #


class TierConfig(BaseModel):
    """Three-tier provider+model configuration.

    ``level1`` is required; ``level2`` and ``level3`` fall back to
    module-level baked defaults when omitted.

    Example YAML/JSON::

        {"level1": {"provider": "openrouter-deepseek",
                     "model": "deepseek/deepseek-v4-flash"}}
    """

    level1: TierLevelConfig = Field(
        description="Level 1 — cheap, obvious, repetitive tasks.",
    )
    level2: TierLevelConfig = Field(
        default_factory=lambda: LEVEL2_DEFAULT,
        description="Level 2 — intermediate tasks (e.g. implementing code).",
    )
    level3: TierLevelConfig = Field(
        default_factory=lambda: LEVEL3_DEFAULT,
        description="Level 3 — high-level organisation and planning.",
    )


# --------------------------------------------------------------------------- #
#  Legacy mapping                                                             #
# --------------------------------------------------------------------------- #

LEGACY_TIER_MAP: dict[LegacyTier, TierLevel] = {
    LegacyTier.CHEAP: TierLevel.LEVEL1,
    LegacyTier.DEFAULT: TierLevel.LEVEL2,
}

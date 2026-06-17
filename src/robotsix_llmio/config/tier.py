"""Tier configuration schema — three configurable provider+model bindings.

This module defines the *data model only*.  Wiring it into the provider
factory, ``build_agent``, and ``new_model`` is a follow-up concern.

The existing two-value :class:`~robotsix_llmio.core.Tier` enum is **not**
modified — callers that pass ``Tier.DEFAULT`` / ``Tier.CHEAP`` continue
to work unchanged.  The :data:`LEGACY_TIER_MAP` dictionary (deprecated,
emits :exc:`DeprecationWarning` on access) provided an explicit path, now
superseded by :meth:`TierConfig.for_level`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from robotsix_llmio.core.tier_enum import Tier as LegacyTier

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
    """A single tier's transport and model binding.

    Describes which consumer-facing *transport* to use and which model
    name/alias that transport resolves for a given :class:`TierLevel`.

    A :func:`~pydantic.model_validator` validates the *transport* against
    :data:`~.transport.TRANSPORT_ALIASES` and cross-checks the *model* field
    against :data:`~.model_registry.PROVIDER_MODELS` at construction time,
    raising :class:`~.transport.UnknownTransportError` for unknown transports
    and :class:`~.model_registry.UnknownModelError` for known providers with
    unknown model names.

    For backward compatibility a ``provider`` key in the input mapping (the
    old registry-name shape) is accepted and converted to the equivalent
    *transport* alias, and a read-only :attr:`provider` property resolves the
    transport back to its provider registry name.
    """

    transport: str = Field(
        description=(
            "Consumer-facing transport alias — one of ``'claude-sdk'`` or "
            "``'openrouter[deepseek]'``.  Resolved to a provider registry "
            "name via ``config.transport.TRANSPORT_ALIASES``."
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

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_provider(cls, data: Any) -> Any:
        """Accept the old ``provider`` shape and convert it to ``transport``.

        When the incoming mapping carries a ``provider`` key and no
        ``transport`` key, pop ``provider`` and set ``transport`` to the
        converted alias.  When both keys are present, ``transport`` wins and
        ``provider`` is dropped.  Non-mapping input passes through untouched.
        """
        if not isinstance(data, dict):
            return data
        if "provider" not in data:
            return data
        from .transport import provider_to_transport

        data = dict(data)
        provider = data.pop("provider")
        if "transport" not in data:
            data["transport"] = provider_to_transport(provider)
        return data

    @property
    def provider(self) -> str:
        """Resolved provider registry name for this tier's transport."""
        from .transport import TRANSPORT_ALIASES

        return TRANSPORT_ALIASES[self.transport]

    @model_validator(mode="after")
    def _validate_model_names(self) -> TierLevelConfig:
        """Validate *transport* and cross-check *model* against the registry."""
        from .model_registry import validate_model
        from .transport import TRANSPORT_ALIASES, validate_transport

        validate_transport(self.transport)
        provider = TRANSPORT_ALIASES[self.transport]
        validate_model(provider, self.model)
        return self


# --------------------------------------------------------------------------- #
#  Baked defaults                                                             #
# --------------------------------------------------------------------------- #

LEVEL1_DEFAULT = TierLevelConfig(
    transport="openrouter[deepseek]",
    model="deepseek/deepseek-v4-flash",
)

LEVEL2_DEFAULT = TierLevelConfig(
    transport="openrouter[deepseek]",
    model="deepseek/deepseek-v4-pro",
)

LEVEL3_DEFAULT = TierLevelConfig(
    transport="claude-sdk",
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

        {"level1": {"transport": "openrouter[deepseek]",
                     "model": "deepseek/deepseek-v4-flash"}}

    Use :meth:`for_level` to resolve an integer level to the corresponding
    :class:`TierLevelConfig` — this is the canonical level→(transport,model)
    resolution point replacing the legacy ``_level_to_tier()`` + tier map.
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

    def for_level(self, level: int) -> TierLevelConfig:
        """Return the :class:`TierLevelConfig` for the given integer *level*.

        | ``level`` | Attribute |
        |-----------|-----------|
        | 1         | ``self.level1`` |
        | 2         | ``self.level2`` |
        | 3         | ``self.level3`` |

        Raises:
            ValueError: If *level* is not 1, 2, or 3.
        """
        if level == 1:
            return self.level1
        if level == 2:
            return self.level2
        if level == 3:
            return self.level3
        raise ValueError(f"`level` must be 1, 2, or 3, got {level!r}")


# --------------------------------------------------------------------------- #
#  Legacy mapping                                                             #
# --------------------------------------------------------------------------- #

LEGACY_TIER_MAP: dict[LegacyTier, TierLevel] = {
    LegacyTier.CHEAP: TierLevel.LEVEL1,
    LegacyTier.DEFAULT: TierLevel.LEVEL2,
}
"""**Deprecated** — use :meth:`TierConfig.for_level` instead.

The public re-export paths (:mod:`~robotsix_llmio.core` and
:mod:`~robotsix_llmio.config`) emit a :exc:`DeprecationWarning` when
this mapping is accessed.
"""

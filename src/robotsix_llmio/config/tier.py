"""Tier configuration schema — four configurable provider+model bindings.

The canonical path for tier resolution is :meth:`TierConfig.for_level`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- #
#  TierLevel — four configuration tiers                                       #
# --------------------------------------------------------------------------- #


class TierLevel(StrEnum):
    """Four-tier configuration selector.

    | Member   | Value      | Purpose                                        |
    |----------|------------|------------------------------------------------|
    | LEVEL1   | ``level1`` | Cheap, obvious, repetitive tasks               |
    | LEVEL2   | ``level2`` | Intermediate (e.g. implementing code)          |
    | LEVEL3   | ``level3`` | High-level organisation and planning           |
    | LEVEL4   | ``level4`` | Frontier — hardest reasoning and long-horizon  |
    """

    LEVEL1 = "level1"
    LEVEL2 = "level2"
    LEVEL3 = "level3"
    LEVEL4 = "level4"


# --------------------------------------------------------------------------- #
#  TierLevelConfig — a single tier's provider+model binding                   #
# --------------------------------------------------------------------------- #


class TierLevelConfig(BaseModel):
    """A single tier's provider-model binding.

    Describes which combined *provider-model* identifier to use for a given
    :class:`TierLevel`.  The identifier is ``<provider>-<model-name>``: the
    provider prefix (before the first hyphen) and the concrete model name —
    e.g. ``"claudeSDK-opus"`` or
    ``"openrouter-deepseek/deepseek-v4-flash"``.

    A :func:`~pydantic.model_validator` parses the identifier and confirms
    the provider prefix is a known backend; the concrete model name is the
    backend's concern (no upfront registry check).
    """

    model: str = Field(
        description=(
            "Combined provider-model identifier — e.g. "
            "``'claudeSDK-opus'`` or "
            "``'openrouter-deepseek/deepseek-v4-flash'``. "
            "The provider prefix (before the first hyphen) "
            "drives lazy backend import; the remainder is the concrete "
            "model name fed to the backend."
        ),
    )
    provider_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments forwarded to the provider constructor "
            "(e.g. ``base_url`` for the OpenRouter provider). Defaults to ``{}``."
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        description=(
            "Optional output token cap forwarded to the provider as a "
            "``task_budget`` (Claude SDK) or ``max_tokens`` model setting "
            "(OpenRouter).  ``None`` means no cap (unbounded output)."
        ),
    )

    # -- parsed accessors ---------------------------------------------------

    @property
    def provider(self) -> str:
        """Provider prefix parsed from the identifier."""
        from robotsix_llmio.core.identifier import parse_model_identifier

        return parse_model_identifier(self.model).provider

    @property
    def model_name(self) -> str:
        """Concrete model name parsed from the identifier."""
        from robotsix_llmio.core.identifier import parse_model_identifier

        return parse_model_identifier(self.model).model_name

    # -- validator ----------------------------------------------------------

    @model_validator(mode="after")
    def _validate_identifier(self) -> TierLevelConfig:
        """Validate the identifier by parsing it and checking the provider
        prefix is a known backend."""
        from robotsix_llmio.core.factory import _PROVIDER_PREFIX_MAP
        from robotsix_llmio.core.identifier import parse_model_identifier

        parsed = parse_model_identifier(self.model)
        if parsed.provider not in _PROVIDER_PREFIX_MAP:
            known = ", ".join(sorted(_PROVIDER_PREFIX_MAP))
            raise ValueError(
                f"Unknown provider prefix {parsed.provider!r} in identifier "
                f"{self.model!r}. Known prefixes: {known}."
            )
        return self


# --------------------------------------------------------------------------- #
#  Baked defaults                                                             #
# --------------------------------------------------------------------------- #

LEVEL1_DEFAULT = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-flash",
    max_tokens=4096,
)

LEVEL2_DEFAULT = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-pro",
    max_tokens=8192,
)

LEVEL3_DEFAULT = TierLevelConfig(
    model="claudeSDK-opus",
    max_tokens=8192,
)

LEVEL4_DEFAULT = TierLevelConfig(
    model="claudeSDK-claude-fable-5",
    max_tokens=16384,
)


# --------------------------------------------------------------------------- #
#  TierConfig — aggregates four TierLevelConfig slots                         #
# --------------------------------------------------------------------------- #


class TierConfig(BaseModel):
    """Four-tier provider+model configuration.

    All four slots are optional: each falls back to its module-level baked
    default (:data:`LEVEL1_DEFAULT`, :data:`LEVEL2_DEFAULT`,
    :data:`LEVEL3_DEFAULT`, :data:`LEVEL4_DEFAULT`) when omitted, so
    ``TierConfig()`` yields the fully baked default configuration.

    Example YAML/JSON (override level 1 only; levels 2-4 stay default)::

        {"level1": {"model": "openrouter-deepseek/deepseek-v4-flash"}}

    Use :meth:`for_level` to resolve an integer level to the corresponding
    :class:`TierLevelConfig`.
    """

    level1: TierLevelConfig = Field(
        default_factory=lambda: LEVEL1_DEFAULT.model_copy(deep=True),
        description="Level 1 — cheap, obvious, repetitive tasks.",
    )
    level2: TierLevelConfig = Field(
        default_factory=lambda: LEVEL2_DEFAULT.model_copy(deep=True),
        description="Level 2 — intermediate tasks (e.g. implementing code).",
    )
    level3: TierLevelConfig = Field(
        default_factory=lambda: LEVEL3_DEFAULT.model_copy(deep=True),
        description="Level 3 — high-level organisation and planning.",
    )
    level4: TierLevelConfig = Field(
        default_factory=lambda: LEVEL4_DEFAULT.model_copy(deep=True),
        description="Level 4 — frontier: hardest reasoning and long-horizon work.",
    )

    def for_level(self, level: int) -> TierLevelConfig:
        """Return the :class:`TierLevelConfig` for the given integer *level*.

        | ``level`` | Attribute |
        |-----------|-----------|
        | 1         | ``self.level1`` |
        | 2         | ``self.level2`` |
        | 3         | ``self.level3`` |
        | 4         | ``self.level4`` |

        Raises:
            ValueError: If *level* is not 1, 2, 3, or 4.

        """
        if level == 1:
            return self.level1
        if level == 2:
            return self.level2
        if level == 3:
            return self.level3
        if level == 4:
            return self.level4
        raise ValueError(f"`level` must be 1, 2, 3, or 4, got {level!r}")

"""Tier configuration schema — five configurable provider+model bindings.

The canonical path for tier resolution is :meth:`TierConfig.for_level`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- #
#  TierLevel — five configuration tiers                                       #
# --------------------------------------------------------------------------- #


class TierLevel(StrEnum):
    """Five-tier configuration selector.

    | Member   | Value      | Purpose                                                  |
    |----------|------------|----------------------------------------------------------|
    | LEVEL1   | ``level1`` | Cheap, obvious, repetitive tasks (pay-per-token)         |
    | LEVEL2   | ``level2`` | Cheap flat-rate: monitors, retrospects, classification   |
    | LEVEL3   | ``level3`` | Intermediate (e.g. implementing code)                    |
    | LEVEL4   | ``level4`` | High-level organisation and planning                     |
    | LEVEL5   | ``level5`` | Frontier — hardest reasoning and long-horizon            |

    Level 2 was inserted on 2026-08-29 (Claude haiku on the flat-rate
    subscription); the former levels 2-4 shifted to 3-5.
    """

    LEVEL1 = "level1"
    LEVEL2 = "level2"
    LEVEL3 = "level3"
    LEVEL4 = "level4"
    LEVEL5 = "level5"


# --------------------------------------------------------------------------- #
#  TierLevelConfig — a single tier's provider+model binding                   #
# --------------------------------------------------------------------------- #


class TierLevelConfig(BaseModel):
    """A single tier's provider-model binding.

    Describes which combined *provider-model* identifier to use for a given
    :class:`TierLevel`.  The identifier is ``<provider>-<model-name>``: the
    provider prefix (before the first hyphen) and the concrete model name —
    e.g. ``"claudeSDK-opus"`` or
    ``"openrouter-deepseek/deepseek-v4-flash-latest"``.

    A :func:`~pydantic.model_validator` parses the identifier and confirms
    the provider prefix is a known backend; the concrete model name is the
    backend's concern (no upfront registry check).
    """

    model: str = Field(
        description=(
            "Combined provider-model identifier — e.g. "
            "``'claudeSDK-opus'`` or "
            "``'openrouter-deepseek/deepseek-v4-flash-latest'``. "
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

# ``max_tokens`` on the OpenRouter levels is a hard per-response output cap.
# Set it too low and a long generation does not degrade — it fails outright,
# raising "Model token limit (N) exceeded before any response was generated"
# and, once the retries and the fallback tier are spent, BLOCKing the caller.
# Observed 2026-08-20: implement agents asked to write a large file (splitting
# a ~1200-line module, cloning a workflow) blew the 8192 cap before emitting
# a single token, on models that serve 384k completion tokens. The values
# below leave generous headroom while still bounding a runaway response to a
# few cents at the capable tier's ~$2/M completion rate.
#
# The cap covers REASONING tokens too. Every tier above level 1 runs with
# ``reasoning_setting={"effort": "xhigh"}`` (see ``_deepseek_provider``), and
# those tokens are billed and counted against ``max_tokens`` before a single
# content token is emitted — so a cap sized for the visible answer alone is
# really a cap on the thinking. Observed 2026-08-27: mill's implement stage
# blew the 32768 cap on ``xiaomi/mimo-v2.5-pro`` (then level 2, now level 3)
# before generating any output, on every attempt, deterministically. Size
# these against the tier's
# *reasoning + answer*, not the answer.
#
# Keep the value at or below the smallest ``max_completion_tokens`` among the
# endpoints the tier's price ceiling admits, or OpenRouter rejects the request
# outright. For level 3 (mimo) today that floor is StreamLake's 128000.
# Level 1 pins the dated snapshot rather than OpenRouter's
# "~deepseek/deepseek-v4-flash-latest" alias: the un-prefixed "-latest" slug
# is not a routable model id, and a floating alias would drift out of the
# measured cheap-tier price ceiling anyway.
#
# ``preferred_provider`` is pinned to a stable *cheap* upstream (DeepInfra)
# rather than DeepSeek. On 2026-08-31 DeepSeek repriced its own flash serving
# to ~$0.44/$1.32 per 1M — 4.4x above the baked cheap-tier ceiling
# ($0.10/$0.20) — so DeepSeek no longer satisfies its own ceiling and the
# price-ceiling drift guard (scripts/check-price-ceilings.py, rule 1) failed.
# DeepInfra ($0.08/$0.18, cache-read <= $0.02/1M) sits under the ceiling and is
# one of several healthy endpoints serving the same snapshot, so preferring it
# keeps routing stable and cheap while ``allow_fallbacks`` still lets
# OpenRouter route past it under the same ceiling. The ceiling itself is
# unchanged (still ``DEFAULT_MAX_PRICE_CHEAP``): the guard did its job and the
# fix is the *preference*, not a relaxed cap. This mirrors how ``LEVEL3_DEFAULT``
# carries its routing (``preferred_provider``/``max_price``/``ignore``) in
# ``provider_kwargs``.
LEVEL1_DEFAULT = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-flash-20260731",
    max_tokens=16384,
    provider_kwargs={"preferred_provider": "DeepInfra"},
)

# Level 2 is the cheap FLAT-RATE tier (Claude haiku on the subscription):
# monitors, retrospects, classifiers — work that is too frequent for a
# pay-per-token model but does not need opus. Inserted 2026-08-29; the
# former levels 2-4 became 3-5.
LEVEL2_DEFAULT = TierLevelConfig(
    model="claudeSDK-haiku",
)

LEVEL3_DEFAULT = TierLevelConfig(
    model="openrouter-xiaomi/mimo-v2.5-pro",
    max_tokens=65536,
    provider_kwargs={
        "preferred_provider": "Xiaomi",
        "max_price_prompt": 0.55,
        "max_price_completion": 1.10,
        "ignore_providers": ["DigitalOcean", "DeepInfra"],
    },
)

# No ``max_tokens`` on the Claude SDK levels, deliberately. ``ClaudeAgentOptions``
# has no per-response cap at all, so the value can only be forwarded as a
# ``task_budget`` — an *advisory* whole-loop allowance the model is shown as a
# countdown, not a ceiling anything enforces. Both defaults sat below the API's
# 20,000 floor and were clamped UP, so they capped nothing and simply told the
# model it had a small allowance for the entire agentic task. Observed
# 2026-08-06: agents abandoning work before starting it ("I'm out of token
# budget for this task before I could load the required tools"), and, on models
# that reject the parameter outright, a hard 400 that killed the caller's stage.
# The OpenRouter levels above keep ``max_tokens`` — there it IS a real enforced
# per-response cap, which is what the field means.
LEVEL4_DEFAULT = TierLevelConfig(
    model="claudeSDK-opus",
)

LEVEL5_DEFAULT = TierLevelConfig(
    model="claudeSDK-claude-fable-5",
)


# --------------------------------------------------------------------------- #
#  TierConfig — aggregates four TierLevelConfig slots                         #
# --------------------------------------------------------------------------- #


class TierConfig(BaseModel):
    """Five-tier provider+model configuration.

    All five slots are optional: each falls back to its module-level baked
    default (:data:`LEVEL1_DEFAULT` … :data:`LEVEL5_DEFAULT`) when omitted,
    so ``TierConfig()`` yields the fully baked default configuration.

    Example YAML/JSON (override level 1 only; levels 2-5 stay default)::

        {"level1": {"model": "openrouter-deepseek/deepseek-v4-flash-latest"}}

    Use :meth:`for_level` to resolve an integer level to the corresponding
    :class:`TierLevelConfig`.
    """

    level1: TierLevelConfig = Field(
        default_factory=lambda: LEVEL1_DEFAULT.model_copy(deep=True),
        description="Level 1 — cheap, obvious, repetitive tasks.",
    )
    level2: TierLevelConfig = Field(
        default_factory=lambda: LEVEL2_DEFAULT.model_copy(deep=True),
        description="Level 2 — cheap flat-rate: monitors, retrospects, classification.",
    )
    level3: TierLevelConfig = Field(
        default_factory=lambda: LEVEL3_DEFAULT.model_copy(deep=True),
        description="Level 3 — intermediate tasks (e.g. implementing code).",
    )
    level4: TierLevelConfig = Field(
        default_factory=lambda: LEVEL4_DEFAULT.model_copy(deep=True),
        description="Level 4 — high-level organisation and planning.",
    )
    level5: TierLevelConfig = Field(
        default_factory=lambda: LEVEL5_DEFAULT.model_copy(deep=True),
        description="Level 5 — frontier: hardest reasoning and long-horizon work.",
    )

    def for_level(self, level: int) -> TierLevelConfig:
        """Return the :class:`TierLevelConfig` for the given integer *level*.

        | ``level`` | Attribute |
        |-----------|-----------|
        | 1         | ``self.level1`` |
        | 2         | ``self.level2`` |
        | 3         | ``self.level3`` |
        | 4         | ``self.level4`` |
        | 5         | ``self.level5`` |

        Raises:
            ValueError: If *level* is not 1, 2, 3, 4, or 5.

        """
        if level == 1:
            return self.level1
        if level == 2:
            return self.level2
        if level == 3:
            return self.level3
        if level == 4:
            return self.level4
        if level == 5:
            return self.level5
        raise ValueError(f"`level` must be 1, 2, 3, 4, or 5, got {level!r}")

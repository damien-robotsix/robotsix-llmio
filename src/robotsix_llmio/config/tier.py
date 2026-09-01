"""Tier configuration schema — three capability levels across two provider slots.

The canonical path for tier resolution is :meth:`TierConfig.for_level`.

Two *provider slots* each bind all three capability levels:

- ``default`` — the provider used in normal operation (baked: Anthropic via
  the Claude SDK subscription — haiku / opus / claude-fable-5).
- ``fallback`` — the provider used while automatic failover is active
  (baked: DeepSeek via OpenRouter — flash / pro / pro).

Capability levels never fall back to one another: a level-2 task is a
level-2 task on whichever provider slot is active. Failover switches the
*provider slot*, per :class:`FailoverConfig`, and is driven by
:mod:`robotsix_llmio.core.failover`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
#  TierLevel — three capability levels                                        #
# --------------------------------------------------------------------------- #


class TierLevel(StrEnum):
    """Three-level capability selector.

    | Member   | Value      | Purpose                                              |
    |----------|------------|------------------------------------------------------|
    | LEVEL1   | ``level1`` | Cheap, frequent: monitors, classifiers, summaries    |
    | LEVEL2   | ``level2`` | Workhorse: implementing code, main assistant turns   |
    | LEVEL3   | ``level3`` | Frontier: hardest reasoning and long-horizon work    |

    Collapsed from five levels on 2026-09-01: equivalent-capability models on
    other providers are no longer separate levels — they live in the
    ``fallback`` provider slot of :class:`TierConfig` at the *same* level.
    """

    LEVEL1 = "level1"
    LEVEL2 = "level2"
    LEVEL3 = "level3"


#: Slot names accepted by :meth:`TierConfig.for_level`.
ProviderSlotName = Literal["default", "fallback"]


# --------------------------------------------------------------------------- #
#  TierLevelConfig — a single level's provider+model binding                  #
# --------------------------------------------------------------------------- #


class TierLevelConfig(BaseModel):
    """A single level's provider-model binding.

    Describes which combined *provider-model* identifier to use for a given
    :class:`TierLevel`.  The identifier is ``<provider>-<model-name>``: the
    provider prefix (before the first hyphen) and the concrete model name —
    e.g. ``"claudeSDK-opus"`` or
    ``"openrouter-deepseek/deepseek-v4-flash-20260731"``.

    A :func:`~pydantic.model_validator` parses the identifier and confirms
    the provider prefix is a known backend; the concrete model name is the
    backend's concern (no upfront registry check).
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        description=(
            "Combined provider-model identifier — e.g. "
            "``'claudeSDK-opus'`` or "
            "``'openrouter-deepseek/deepseek-v4-flash-20260731'``. "
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
#  Baked defaults — default slot (Anthropic via the Claude SDK)               #
# --------------------------------------------------------------------------- #

# No ``max_tokens`` on the Claude SDK levels, deliberately. ``ClaudeAgentOptions``
# has no per-response cap at all, so the value can only be forwarded as a
# ``task_budget`` — an *advisory* whole-loop allowance the model is shown as a
# countdown, not a ceiling anything enforces. Values below the API's 20,000
# floor are clamped UP, so they cap nothing and simply tell the model it has a
# small allowance for the entire agentic task. Observed 2026-08-06: agents
# abandoning work before starting it ("I'm out of token budget for this task
# before I could load the required tools"), and, on models that reject the
# parameter outright, a hard 400 that killed the caller's stage. The OpenRouter
# levels below keep ``max_tokens`` — there it IS a real enforced per-response
# cap, which is what the field means.
DEFAULT_LEVEL1 = TierLevelConfig(model="claudeSDK-haiku")

DEFAULT_LEVEL2 = TierLevelConfig(model="claudeSDK-opus")

DEFAULT_LEVEL3 = TierLevelConfig(model="claudeSDK-claude-fable-5")


# --------------------------------------------------------------------------- #
#  Baked defaults — fallback slot (DeepSeek via OpenRouter)                   #
# --------------------------------------------------------------------------- #

# ``max_tokens`` on the OpenRouter levels is a hard per-response output cap,
# and it covers REASONING tokens too: at level >= 2 the DeepSeek layer runs
# with ``reasoning_setting={"effort": "xhigh"}`` and those tokens are billed
# and counted against ``max_tokens`` before a single content token is emitted.
# Size the cap against *reasoning + answer*, not the answer (observed
# 2026-08-27: a 32768 cap blew deterministically before any output on an
# agentic prompt). Set it too low and a long generation does not degrade — it
# fails outright ("Model token limit (N) exceeded before any response was
# generated"). Keep the value at or below the smallest
# ``max_completion_tokens`` among the endpoints the level's price ceiling
# admits, or OpenRouter rejects the request outright.
#
# Level 1 pins the dated flash snapshot rather than OpenRouter's
# "-latest" alias: the un-prefixed "-latest" slug is not a routable model id,
# and a floating alias would drift out of the measured cheap-tier price
# ceiling anyway. ``preferred_provider`` is pinned to a stable *cheap*
# upstream (DeepInfra, $0.08/$0.18, cache-read <= $0.02/1M) rather than
# DeepSeek: on 2026-08-31 DeepSeek repriced its own flash serving to
# ~$0.44/$1.32 per 1M — 4.4x above the baked cheap-tier ceiling — so DeepSeek
# no longer satisfies its own ceiling (scripts/check-price-ceilings.py,
# rule 1). ``allow_fallbacks`` still lets OpenRouter route past DeepInfra
# under the same ceiling.
FALLBACK_LEVEL1 = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-flash-20260731",
    max_tokens=16384,
    provider_kwargs={"preferred_provider": "DeepInfra"},
)

# Level 2 binds the PRO snapshot, same as level 3 — deliberately. The original
# rework bound flash here (cheap workhorse), but flash under xhigh reasoning
# degenerates into token loops on long agentic contexts (observed live
# 2026-09-01, an hour into the first failover window: a 90-turn chat session
# collapsed into word salad — the same failure the pre-rework fleet guarded
# against, and the reason the old workhorse fallback moved to pro that same
# week). A workhorse that cannot carry the fleet's real contexts is not a
# fallback. Flash remains the level-1 binding, where prompts are short and
# reasoning is off.
FALLBACK_LEVEL2 = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-pro-0813",
    max_tokens=131072,
    provider_kwargs={
        "preferred_provider": "StreamLake",
        "max_price_prompt": 1.16,
        "max_price_completion": 3.40,
    },
)

# Level 3 pins the DATED ``-0813`` pro snapshot — same lesson as level 1:
# dated snapshots route deterministically while undated slugs point at
# whatever pool OpenRouter aliases them to. Routing measured 2026-09-01 on
# ``deepseek-v4-pro-0813``: cheapest HEALTHY endpoints are StreamLake
# ($1.1154/$3.3462 per 1M, cache-read $0.0372), GMICloud ($1.122/$3.366) and
# Alibaba ($1.122/$3.366). The ceiling below is aligned tight to that band —
# it admits exactly those three (the price-ceiling guard requires >= 3
# healthy) and excludes the $1.30+/$3.96 tail, DeepSeek's own endpoint
# included ($1.32/$3.96). Admitted endpoints all serve
# max_completion >= 384000, so 131072 keeps ample reasoning + answer headroom
# while bounding a runaway response.
FALLBACK_LEVEL3 = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-pro-0813",
    max_tokens=131072,
    provider_kwargs={
        "preferred_provider": "StreamLake",
        "max_price_prompt": 1.16,
        "max_price_completion": 3.40,
    },
)


# --------------------------------------------------------------------------- #
#  ProviderSlotConfig — one provider's binding of all three levels            #
# --------------------------------------------------------------------------- #


class ProviderSlotConfig(BaseModel):
    """One provider slot's binding of all three capability levels.

    A slot is a *role* (``default`` or ``fallback`` in :class:`TierConfig`),
    not a provider name: each level entry carries its own combined
    provider-model identifier, so a slot can technically mix backends. The
    baked configuration keeps each slot on one provider — that is what makes
    provider-level failover meaningful.
    """

    model_config = ConfigDict(extra="forbid")

    level1: TierLevelConfig = Field(
        description="Level 1 — cheap, frequent: monitors, classifiers, summaries.",
    )
    level2: TierLevelConfig = Field(
        description="Level 2 — workhorse: implementing code, main assistant turns.",
    )
    level3: TierLevelConfig = Field(
        description="Level 3 — frontier: hardest reasoning and long-horizon work.",
    )

    def for_level(self, level: int) -> TierLevelConfig:
        """Return the :class:`TierLevelConfig` for integer *level* (1, 2, or 3).

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


def _default_slot() -> ProviderSlotConfig:
    """Fresh copy of the baked default (Anthropic / Claude SDK) slot."""
    return ProviderSlotConfig(
        level1=DEFAULT_LEVEL1.model_copy(deep=True),
        level2=DEFAULT_LEVEL2.model_copy(deep=True),
        level3=DEFAULT_LEVEL3.model_copy(deep=True),
    )


def _fallback_slot() -> ProviderSlotConfig:
    """Fresh copy of the baked fallback (DeepSeek / OpenRouter) slot."""
    return ProviderSlotConfig(
        level1=FALLBACK_LEVEL1.model_copy(deep=True),
        level2=FALLBACK_LEVEL2.model_copy(deep=True),
        level3=FALLBACK_LEVEL3.model_copy(deep=True),
    )


# --------------------------------------------------------------------------- #
#  FailoverConfig — when to switch slots, and for how long                    #
# --------------------------------------------------------------------------- #


class FailoverConfig(BaseModel):
    """Automatic provider-failover policy.

    After :attr:`failure_threshold` consecutive provider-shaped failures on
    the ``default`` slot (a provider-wide exhaustion arms failover
    immediately), calls route to the ``fallback`` slot for
    :attr:`window_seconds`, then automatically return to ``default``. See
    :mod:`robotsix_llmio.core.failover`.
    """

    model_config = ConfigDict(extra="forbid")

    failure_threshold: int = Field(
        default=3,
        ge=1,
        description=(
            "Consecutive provider-shaped failures on the default slot before "
            "failover arms. A provider-wide exhaustion (e.g. the Claude "
            "subscription out of usage credits) arms failover immediately, "
            "regardless of this threshold."
        ),
    )
    window_seconds: float = Field(
        default=900.0,
        gt=0,
        description=(
            "How long calls stay on the fallback slot once failover arms, in "
            "seconds (default 900 = 15 minutes). After the window expires the "
            "next call probes the default slot again; a still-broken default "
            "re-arms a fresh window."
        ),
    )


# --------------------------------------------------------------------------- #
#  TierConfig — two provider slots + failover policy                          #
# --------------------------------------------------------------------------- #


class TierConfig(BaseModel):
    """Two-slot, three-level provider+model configuration.

    All fields are optional: each slot falls back to its baked default
    (Anthropic via the Claude SDK for ``default``, DeepSeek via OpenRouter
    for ``fallback``), so ``TierConfig()`` yields the fully baked
    configuration.

    Example YAML/JSON (override the default slot's level 2 only; everything
    else stays baked)::

        {"default": {"level2": {"model": "claudeSDK-sonnet"}}}

    Use :meth:`for_level` to resolve an integer level to the corresponding
    :class:`TierLevelConfig`; by default it resolves against the slot the
    failover tracker currently designates as active.
    """

    model_config = ConfigDict(extra="forbid")

    default: ProviderSlotConfig = Field(
        default_factory=_default_slot,
        description=(
            "Provider slot used in normal operation. Baked: Anthropic via "
            "the Claude SDK — haiku / opus / claude-fable-5."
        ),
    )
    fallback: ProviderSlotConfig = Field(
        default_factory=_fallback_slot,
        description=(
            "Provider slot used while failover is active. Baked: DeepSeek "
            "via OpenRouter — flash / pro / pro."
        ),
    )
    failover: FailoverConfig = Field(
        default_factory=FailoverConfig,
        description="Automatic provider-failover policy (threshold + window).",
    )

    def slot(self, name: ProviderSlotName) -> ProviderSlotConfig:
        """Return the :class:`ProviderSlotConfig` named *name*."""
        if name == "default":
            return self.default
        if name == "fallback":
            return self.fallback
        raise ValueError(f"`slot` must be 'default' or 'fallback', got {name!r}")

    def for_level(
        self,
        level: int,
        slot: ProviderSlotName | None = None,
    ) -> TierLevelConfig:
        """Resolve integer *level* (1, 2, or 3) to a :class:`TierLevelConfig`.

        Args:
            level: Capability level — 1, 2, or 3.
            slot: Which provider slot to resolve against.  ``None`` (the
                default) resolves against the slot the process-wide failover
                tracker currently designates as active — ``default`` in
                normal operation, ``fallback`` while a failover window is
                open.  Pass an explicit slot name for pure config access
                (e.g. to display both bindings in a UI).

        Raises:
            ValueError: If *level* is not 1, 2, or 3, or *slot* is not a
                valid slot name.

        """
        if slot is None:
            from robotsix_llmio.core.failover import get_failover_tracker

            slot = get_failover_tracker().active_slot()
        return self.slot(slot).for_level(level)

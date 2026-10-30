"""Tier configuration schema — three capability levels across an ordered
chain of provider slots.

The canonical path for tier resolution is :meth:`TierConfig.for_level`.

:attr:`TierConfig.chain` is an ordered sequence of *provider slots*, each
binding all three capability levels. Position 0 is used in normal operation;
later positions are failover targets, in order. The baked chain is:

- position 0 — Anthropic via the direct Messages API (spends the Team
  subscription's API credit first) — claude-haiku-5-5 / claude-opus-5-5 /
  claude-fable-5, mirroring the Claude SDK bindings.
- position 1 — Anthropic via the Claude SDK subscription seat —
  haiku / opus / claude-fable-5.
- position 2 — DeepSeek via OpenRouter (last resort) — flash / flash / pro.

The legacy two-slot shape (``default`` / ``fallback`` keys) is migrated onto
a two-slot chain (Claude SDK, then OpenRouter, by default) by a validator, so
persisted configs keep loading.

Capability levels never fall back to one another: a level-2 task is a
level-2 task on whichever provider slot is active. Failover switches the
*provider slot*, per :class:`FailoverConfig`, and is driven by
:mod:`robotsix_llmio.core.failover`.
"""

from __future__ import annotations

import warnings
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


#: Legacy provider slot names, in chain order. ``"default"`` is chain
#: position 0 and ``"fallback"`` chain position 1. :class:`TierConfig` now
#: holds an ordered N-slot ``chain``; these names survive as aliases for the
#: first two positions (accepted by :meth:`TierConfig.slot` and migrated by
#: :class:`TierConfig`'s legacy-shape validator). The single source for the
#: legacy slot names used by :func:`robotsix_llmio.config.loader.load_tier_config`
#: (which iterates this tuple and builds its unknown-key error message from
#: it) — the loader no longer re-declares the slot names.
PROVIDER_SLOTS: tuple[str, ...] = ("default", "fallback")

#: Legacy slot-name aliases (``"default"`` = chain position 0,
#: ``"fallback"`` = chain position 1).
LegacyProviderSlotName = Literal["default", "fallback"]

#: Slot references accepted by :meth:`TierConfig.slot` /
#: :meth:`TierConfig.for_level`: a zero-based chain index, or one of the
#: legacy slot-name aliases.
ProviderSlotName = int | LegacyProviderSlotName


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
#  Baked defaults — anthropic slot (direct Anthropic Messages API)            #
# --------------------------------------------------------------------------- #

# Chain position 0: the direct Messages API spends the Team subscription's
# monthly API dollar credit before the subscription seat (Claude SDK, next
# slot) and OpenRouter (last resort). Model ids mirror the Claude SDK levels
# below: ``haiku`` / ``opus`` aliases pinned to their concrete ids, and the
# same ``claude-fable-5`` at level 3.
ANTHROPIC_LEVEL1 = TierLevelConfig(model="anthropic-claude-haiku-5-5")

ANTHROPIC_LEVEL2 = TierLevelConfig(model="anthropic-claude-opus-5-5")

ANTHROPIC_LEVEL3 = TierLevelConfig(model="anthropic-claude-fable-5")


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
#
# ``ignore_providers`` bars Wafer from cheap-tier fallback: it lists
# $0.130/$0.230 (inside the $0.13/$0.26 ceiling, so ``max_price`` admits it)
# but its cache-read runs $0.120/1M — ~8x DeepInfra's $0.015/1M, a cost
# ``max_price`` cannot see. The drift guard flagged it 2026-10-12 as >4x the
# preferred provider's cache-read rate; excluding it still leaves >= 3 healthy
# admitted endpoints (DeepInfra, DigitalOcean, BaseTen, Venice). This mirrors
# the capable tier's ``DEFAULT_IGNORE_CAPABLE`` list.
FALLBACK_LEVEL1 = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-flash-20260731",
    max_tokens=16384,
    provider_kwargs={
        "preferred_provider": "DeepInfra",
        "ignore_providers": ["Wafer"],
    },
)

# Level 2 binds the SAME flash snapshot as level 1 — the operator's explicit
# design decision (re-confirmed 2026-09-01 after a day on the fallback slot):
# L1/L2 flash, L3 pro. KNOWN RISK, accepted: flash under the level>=2 xhigh
# reasoning policy degenerated into token loops on one long (90-turn) chat
# session earlier the same day; the operator chose the ~14x cost saving over
# switching the workhorse to pro. If loops recur, the recorded alternatives
# are pro at L2 or a capped reasoning effort for flash — change only with an
# operator decision. The price ceiling is stated EXPLICITLY because the
# per-level defaults would otherwise apply the capable-tier ceiling
# ($1.16/$3.40) to a flash-priced model; the output cap exceeds level 1's
# because reasoning tokens bill against it.
FALLBACK_LEVEL2 = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-flash-20260731",
    max_tokens=65536,
    provider_kwargs={
        "preferred_provider": "DeepInfra",
        "max_price_prompt": 0.13,
        "max_price_completion": 0.26,
        # See FALLBACK_LEVEL1: Wafer's cache-read ($0.120/1M) is ~8x
        # DeepInfra's, invisible to max_price — barred via provider.ignore.
        "ignore_providers": ["Wafer"],
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
#  Baked default — vision binding (image questions for text-only models)     #
# --------------------------------------------------------------------------- #

# The model that answers ``ask_image`` tool calls (see
# ``core/image_tool.py``): the text-only OpenRouter models (DeepSeek) hand
# the agent a tool that interrogates attached images through this binding.
# Claude models read images natively and never use it.
# Always resolved directly — no slot membership, no failover indirection:
# it is the fleet's one image-reading model.
VISION_DEFAULT = TierLevelConfig(
    model="openrouter-deepseek/deepseek-v4-flash-vision-exp",
    max_tokens=8192,
)


# --------------------------------------------------------------------------- #
#  ProviderSlotConfig — one provider's binding of all three levels            #
# --------------------------------------------------------------------------- #


class ProviderSlotConfig(BaseModel):
    """One provider slot's binding of all three capability levels.

    A slot is a *role* (a position in :attr:`TierConfig.chain`),
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


def _anthropic_slot() -> ProviderSlotConfig:
    """Fresh copy of the baked direct-Anthropic (Messages API) slot."""
    return ProviderSlotConfig(
        level1=ANTHROPIC_LEVEL1.model_copy(deep=True),
        level2=ANTHROPIC_LEVEL2.model_copy(deep=True),
        level3=ANTHROPIC_LEVEL3.model_copy(deep=True),
    )


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
    reattempt_backoff_base: float = Field(
        default=2.0,
        gt=1.0,
        description=(
            "Exponential base for backing off repeated degraded-primary "
            "reattempts. The FIRST arming uses ``window_seconds`` unchanged; "
            "each subsequent re-arm (a still-degraded default that failed its "
            "probe after the window expired) grows the window by this base "
            "raised to the reattempt count, so a persistently broken primary "
            "is re-probed exponentially less often instead of churning a "
            "doomed attempt every window."
        ),
    )
    reattempt_backoff_cap_seconds: float = Field(
        default=14400.0,
        gt=0,
        description=(
            "Ceiling on the backed-off failover window in seconds (default "
            "14400 = 4 hours), so exponential growth of repeated degraded-"
            "primary reattempts can never park calls on the fallback slot "
            "indefinitely. Must be >= ``window_seconds``."
        ),
    )
    reattempt_backoff_jitter: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the backed-off window subtracted as random jitter: "
            "the effective window lands in "
            "``[window * (1 - jitter), window]``, spreading reattempts so "
            "fleet instances do not re-probe a recovering primary in "
            "lockstep. Only applies to backed-off re-arms, never to the first "
            "window or a quota-reset-hinted window."
        ),
    )

    max_request_cost_usd: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Per-request cost ceiling, in USD, for a failover slot switch. "
            "Before a request fails over to the candidate slot, the loop "
            "estimates that slot's worst-case per-request cost (its "
            "``max_price_completion`` ceiling times its ``max_tokens`` output "
            "cap — see "
            "``robotsix_llmio.core.failover.estimate_request_cost``); when the "
            "estimate strictly exceeds this ceiling the switch is refused and "
            "the request fails fast with a "
            "``FailoverCostGuardError`` instead of retrying on an expensive "
            "slot. ``None`` (the default) disables the ceiling check. Slots "
            "with no price ceiling or output cap cannot be estimated and are "
            "never blocked."
        ),
    )
    cost_ratio_threshold: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Refuse a failover slot switch when the candidate slot's "
            "estimated per-request cost strictly exceeds the active slot's "
            "estimate times this ratio — i.e. the candidate is *materially* "
            "more expensive than staying put. ``None`` (the default) disables "
            "the ratio check; it is also skipped when either slot's cost "
            "cannot be estimated."
        ),
    )

    @model_validator(mode="after")
    def _cap_not_below_window(self) -> FailoverConfig:
        """Reject a backoff cap that would shrink the base window."""
        if self.reattempt_backoff_cap_seconds < self.window_seconds:
            raise ValueError(
                "reattempt_backoff_cap_seconds "
                f"({self.reattempt_backoff_cap_seconds}) must be >= "
                f"window_seconds ({self.window_seconds})"
            )
        return self


# --------------------------------------------------------------------------- #
#  TierConfig — ordered provider-slot chain + failover policy                 #
# --------------------------------------------------------------------------- #


def _default_chain() -> list[ProviderSlotConfig]:
    """Fresh copy of the baked provider chain.

    Direct Anthropic API, then the Claude SDK slot, then OpenRouter.
    """
    return [_anthropic_slot(), _default_slot(), _fallback_slot()]


def _legacy_chain() -> list[ProviderSlotConfig]:
    """Baked slots backing the legacy ``default`` / ``fallback`` keys.

    Keeps the legacy shape's meaning stable: ``default`` is the Claude SDK
    slot and ``fallback`` the OpenRouter slot.
    """
    return [_default_slot(), _fallback_slot()]


class TierConfig(BaseModel):
    """Ordered N-slot, three-level provider+model configuration.

    :attr:`chain` is an ordered sequence of :class:`ProviderSlotConfig`
    entries: position 0 is the provider used in normal operation, and each
    later position is the next provider to cascade to while failover is
    active. Any number of slots (>= 1) is supported.

    All fields are optional: the baked chain is Anthropic via the direct
    Messages API (position 0), Anthropic via the Claude SDK (position 1),
    then DeepSeek via OpenRouter (position 2), so ``TierConfig()`` yields the
    fully baked configuration.

    The legacy two-slot shape (top-level ``default`` / ``fallback`` keys) is
    still accepted and migrated onto the chain — ``default`` becomes position
    0 and ``fallback`` position 1 of a two-slot chain; an omitted legacy
    key keeps its baked slot (Claude SDK / OpenRouter respectively).
    Example YAML/JSON (legacy shape)::

        {"default": {"level1": {...}, "level2": {...}, "level3": {...}}}

    or the chain shape::

        {"chain": [{"level1": {...}, ...}, {...}, {...}]}

    Use :meth:`for_level` to resolve an integer level to the corresponding
    :class:`TierLevelConfig`; by default it resolves against the slot the
    failover tracker currently designates as active.
    """

    model_config = ConfigDict(extra="forbid")

    chain: list[ProviderSlotConfig] = Field(
        default_factory=_default_chain,
        min_length=1,
        description=(
            "Ordered provider-slot chain. Position 0 is used in normal "
            "operation; later positions are cascaded to, in order, while "
            "failover is active. Baked: Anthropic via the direct Messages "
            "API (claude-haiku-5-5 / claude-opus-5-5 / claude-fable-5), then "
            "Anthropic via the Claude SDK (haiku / opus / claude-fable-5), "
            "then DeepSeek via OpenRouter (flash / flash / pro)."
        ),
    )
    failover: FailoverConfig = Field(
        default_factory=FailoverConfig,
        description="Automatic provider-failover policy (threshold + window).",
    )
    vision: TierLevelConfig = Field(
        default_factory=lambda: VISION_DEFAULT.model_copy(deep=True),
        description=(
            "Binding for the ask_image tool — the model that answers image "
            "questions on behalf of text-only transports. Resolved directly, "
            "never through the provider slots or failover."
        ),
    )

    # -- legacy two-slot migration ------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_slots(cls, data: Any) -> Any:
        """Map the legacy ``default`` / ``fallback`` keys onto :attr:`chain`.

        ``default`` becomes chain position 0 and ``fallback`` position 1; a
        legacy key that is absent keeps its baked slot. Mixing the legacy
        keys with an explicit ``chain`` is ambiguous and rejected.
        """
        if not isinstance(data, dict):
            return data
        legacy = [name for name in PROVIDER_SLOTS if name in data]
        if not legacy:
            return data
        if "chain" in data:
            raise ValueError(
                "Tier config mixes the legacy slot key(s) "
                f"{legacy!r} with an explicit 'chain'; use one shape only."
            )
        warnings.warn(
            f"Legacy tier config key(s) {legacy!r} are deprecated; use 'chain' "
            "instead. The legacy shape will be removed in the next major version.",
            DeprecationWarning,
            stacklevel=2,
        )
        migrated = {k: v for k, v in data.items() if k not in PROVIDER_SLOTS}
        baked = _legacy_chain()
        migrated["chain"] = [
            data.get(name, baked[index]) for index, name in enumerate(PROVIDER_SLOTS)
        ]
        return migrated

    # -- legacy accessors ---------------------------------------------------

    @property
    @warnings.deprecated(
        "Legacy tier config accessor 'TierConfig.default' is deprecated; "
        "use TierConfig.slot(0) (or chain[0]) instead."
    )
    def default(self) -> ProviderSlotConfig:
        """Legacy alias for chain position 0 (the normal-operation slot)."""
        return self.slot(0)

    @property
    @warnings.deprecated(
        "Legacy tier config accessor 'TierConfig.fallback' is deprecated; "
        "use TierConfig.slot(1) (or chain[1]) instead."
    )
    def fallback(self) -> ProviderSlotConfig:
        """Legacy alias for chain position 1 (the first failover slot).

        Raises:
            ValueError: If the chain holds a single slot.

        """
        return self.slot(1)

    # -- resolution ---------------------------------------------------------

    def slot_index(self, name: ProviderSlotName) -> int:
        """Return the zero-based chain position designated by *name*.

        Args:
            name: A chain index (``0 <= name < len(chain)``) or a legacy
                slot-name alias (``"default"`` = 0, ``"fallback"`` = 1).

        Raises:
            ValueError: If *name* is not a valid chain index or alias.

        """
        if isinstance(name, str) and name in PROVIDER_SLOTS:
            index = PROVIDER_SLOTS.index(name)
        elif isinstance(name, int) and not isinstance(name, bool):
            index = name
        else:
            index = -1
        if not 0 <= index < len(self.chain):
            raise ValueError(
                f"`slot` must be a chain index in [0, {len(self.chain)}) or a "
                f"legacy slot name 'default' or 'fallback' resolving inside "
                f"the chain, got {name!r}"
            )
        return index

    def slot(self, name: ProviderSlotName) -> ProviderSlotConfig:
        """Return the :class:`ProviderSlotConfig` at chain position *name*.

        Args:
            name: A chain index or a legacy slot-name alias — see
                :meth:`slot_index`.

        Raises:
            ValueError: If *name* is not a valid chain index or alias.

        """
        return self.chain[self.slot_index(name)]

    def for_level(
        self,
        level: int,
        slot: ProviderSlotName | None = None,
    ) -> TierLevelConfig:
        """Resolve integer *level* (1, 2, or 3) to a :class:`TierLevelConfig`.

        Args:
            level: Capability level — 1, 2, or 3.
            slot: Which provider slot to resolve against — a zero-based
                chain index or a legacy alias (``"default"`` / ``"fallback"``).
                ``None`` (the default) resolves against the slot the
                process-wide failover tracker currently designates as active.
                Pass an explicit slot for pure config access (e.g. to display
                every binding in a UI).

        Raises:
            ValueError: If *level* is not 1, 2, or 3, or *slot* is not a
                valid chain position.

        """
        if slot is None:
            from robotsix_llmio.core.failover import get_failover_tracker

            slot = get_failover_tracker().active_slot(chain_length=len(self.chain))
        return self.slot(slot).for_level(level)

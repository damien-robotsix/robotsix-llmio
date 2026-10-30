"""Failover policy — stateless rules deciding *whether* and *where* to cascade.

Extracted from :mod:`robotsix_llmio.core.failover`, which re-exports the
public name (:func:`is_provider_shaped`) so ``robotsix_llmio.core.failover``
stays the import surface. This module holds pure decision logic only:

- failure classification (:func:`is_provider_shaped`, :func:`_is_exhaustion`),
- the per-request slot-switch cap and attempt order
  (:func:`_max_slot_switches`, :func:`_attempt_order`),
- the per-request cost-guard wiring (:func:`_enforce_cost_guard`, which
  applies :func:`~robotsix_llmio.core._cost_guard._cost_guard_verdict`).

The mutable routing state lives in :mod:`robotsix_llmio.core._failover_state`
and the call loop in :mod:`robotsix_llmio.core.failover`.
"""

from __future__ import annotations

import logging
from typing import Any

from robotsix_llmio.config.tier import ProviderSlotName, TierConfig, TierLevelConfig
from robotsix_llmio.exceptions import FailoverCostGuardError, ProviderExhaustedError

from ._cost_guard import _cost_guard_verdict
from .retry import _walk_cause_chain, is_rate_limited, is_transient, is_usage_exhausted

log = logging.getLogger("robotsix_llmio.failover")

# Per-request cost-guard observability: set when a slot switch is refused
# because the candidate slot would cost too much.
_ATTR_COST_GUARD_BLOCKED = "llmio.failover.cost_guard_blocked"
_ATTR_COST_GUARD_REASON = "llmio.failover.cost_guard_reason"


# --------------------------------------------------------------------------- #
#  Failure classification                                                     #
# --------------------------------------------------------------------------- #


def is_provider_shaped(exc: BaseException) -> bool:
    """True when *exc* points at the provider rather than the task.

    A provider-shaped failure is one where running the same request on a
    *different provider* can plausibly succeed: transient transport errors
    that outlived their retries, rate limits, usage exhaustion, and
    provider-wide exhaustion (:class:`ProviderExhaustedError`). Task-shaped
    failures — a turn-limit blowout, a validation error in the caller's own
    output type — raise straight through: re-running a doomed task on the
    other provider would only spend money twice.
    """
    if any(isinstance(cur, ProviderExhaustedError) for cur in _walk_cause_chain(exc)):
        return True
    return is_transient(exc) or is_rate_limited(exc) or is_usage_exhausted(exc)


#: Exception class names that doom every future call on the provider until an
#: external fix, matched by name so the lightweight core does not import the
#: claude_sdk package. ``ClaudeSDKAuthError``: a dead OAuth credential fails
#: every call on the subscription exactly like exhaustion does — waiting for
#: the consecutive-failure threshold just burns doomed attempts.
_PROVIDER_DEAD_NAMES: frozenset[str] = frozenset({"ClaudeSDKAuthError"})


def _is_exhaustion(exc: BaseException) -> bool:
    """True when *exc* signals provider-wide exhaustion or a dead credential.

    Either way every future call on the same provider is also doomed until an
    external fix (quota reset, credential refresh), so failover arms
    immediately instead of waiting for the consecutive-failure threshold.
    """
    for cur in _walk_cause_chain(exc):
        if isinstance(cur, ProviderExhaustedError):
            return True
        if type(cur).__name__ in _PROVIDER_DEAD_NAMES:
            return True
    return is_usage_exhausted(exc)


# --------------------------------------------------------------------------- #
#  Slot-switch cap and attempt order                                          #
# --------------------------------------------------------------------------- #


def _max_slot_switches(chain_length: int) -> int:
    """Hard cap on provider-slot switches within a SINGLE request.

    A request cascades through the chain trying each slot AT MOST once, so it
    switches slots at most ``chain_length - 1`` times (exactly once for the
    legacy two-slot chain). This bounds per-request churn so a doomed request
    can never ping-pong between slots, multiplying spend on duplicate prompts.
    """
    return max(chain_length - 1, 0)


def _attempt_order(first: int, chain_length: int, failover_enabled: bool) -> list[int]:
    """Chain positions a single request tries, in order.

    Starts on *first* and, when failover is enabled, cascades through the
    remaining slots in chain order (wrapping past the end back to the earlier
    positions). The retry cap is enforced explicitly: whatever the order
    build produced, a single request attempts each slot at most once — it can
    never switch slots more than ``chain_length - 1`` times, so a doomed
    request cannot ping-pong between slots.
    """
    order: list[int] = [first]
    if failover_enabled:
        order.extend((first + step) % chain_length for step in range(1, chain_length))
    return order[: 1 + _max_slot_switches(chain_length)]


# --------------------------------------------------------------------------- #
#  Per-request cost guard                                                     #
# --------------------------------------------------------------------------- #


def _enforce_cost_guard(
    exc: BaseException,
    *,
    primary_tlc: TierLevelConfig,
    tier_config: TierConfig,
    level: int,
    slot_name: object,
    candidate_slot: ProviderSlotName,
    run_span: Any,
    what: str,
) -> None:
    """Refuse a slot switch the per-request cost guard rejects.

    Before spending money retrying on the candidate slot, refuse the switch
    when that slot would blow the per-request ceiling or be materially
    pricier than the slot the request started on. Raises
    :class:`FailoverCostGuardError` chained from *exc* on refusal.
    """
    candidate_tlc = tier_config.for_level(level, slot=candidate_slot)
    refusal = _cost_guard_verdict(primary_tlc, candidate_tlc, tier_config.failover)
    if refusal is None:
        return
    log.warning(
        "%s: refusing failover slot=%s -> slot=%s: %s",
        what,
        slot_name,
        candidate_slot,
        refusal,
    )
    if run_span is not None:
        run_span.set_attribute(_ATTR_COST_GUARD_BLOCKED, True)
        run_span.set_attribute(_ATTR_COST_GUARD_REASON, refusal)
    raise FailoverCostGuardError(
        f"{what}: failover to slot={candidate_slot} refused "
        f"by the per-request cost guard ({refusal})"
    ) from exc

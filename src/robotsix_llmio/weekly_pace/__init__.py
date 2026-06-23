"""Weekly Claude usage pace governor.

The :class:`PaceGovernor` compares model-weighted Claude consumption this
week (from the shared Langfuse project + in-process increments) against the
elapsed week fraction. When consumption is ahead of pace, it recommends
falling back to DeepSeek; when behind pace, it recommends Claude.

Usage::

    from robotsix_llmio.weekly_pace import PaceGovernor
    from robotsix_llmio.weekly_pace._config import WeeklyPaceConfig

    config = WeeklyPaceConfig(enabled=True, weekly_budget=10.0)
    governor = PaceGovernor(config, cost_source=langfuse_source)

    if governor.should_use_claude(agent_name="planner"):
        agent = claude_provider.build_agent(...)
    else:
        agent = deepseek_provider.build_agent(...)

    # After the run completes:
    governor.record_usage(cost_usd=result.total_cost_usd, model="opus")
"""

from __future__ import annotations

__all__ = ["PaceGovernor"]

import logging
import time as _time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from ..core.cost_log import CostWindow
from ._week_math import _current_week_window, week_fraction_elapsed

if TYPE_CHECKING:
    from ..core.langfuse_cost import LangfuseCostLogSource
    from ._config import WeeklyPaceConfig

logger = logging.getLogger(__name__)

_PROVIDER_FILTER = "claude-sdk"


class PaceGovernor:
    """Weekly Claude usage pace governor.

    Consults the shared Langfuse project for total Claude cost this week
    (cached with a configurable TTL), adds in-process increments for
    in-flight runs that haven't been exported yet, and compares the
    total against the elapsed week fraction with hysteresis margins.

    Construct with a :class:`~robotsix_llmio.weekly_pace._config.WeeklyPaceConfig`
    and an optional :class:`~robotsix_llmio.core.langfuse_cost.LangfuseCostLogSource`.
    When *cost_source* is None, only in-process increments are tracked
    (useful for testing or when Langfuse is unavailable).
    """

    def __init__(
        self,
        config: WeeklyPaceConfig,
        cost_source: LangfuseCostLogSource | None = None,
    ) -> None:
        self._config = config
        self._cost_source = cost_source
        # In-process accumulator: cost incurred since the last Langfuse query.
        # Reset on each cache refresh.
        self._in_process_cost: float = 0.0
        # Langfuse cache
        self._cached_weekly_cost: float = 0.0
        self._cache_timestamp: float | None = None
        # Current state for hysteresis (None = unknown, treat as under-pace)
        self._currently_over_pace: bool = False

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def should_use_claude(
        self,
        now: datetime | None = None,
        *,
        agent_name: str | None = None,
    ) -> bool:
        """Return True if Claude should be used for the next run.

        Always returns True when the governor is disabled.

        When *agent_name* is listed in ``always_claude_agents``, returns
        True regardless of pace.

        Otherwise, computes the current pace and applies hysteresis:
        returns False (→ DeepSeek) when ahead of pace, True when behind.
        """
        if not self._config.enabled:
            return True

        if agent_name and agent_name in self._config.always_claude_agents:
            return True

        resolved_now = now or datetime.now(UTC)
        week_start, week_end = _current_week_window(
            resolved_now,
            self._config.week_anchor_day,
            self._config.week_anchor_time,
        )
        fraction_elapsed = week_fraction_elapsed(resolved_now, week_start, week_end)

        budget_fraction = self._budget_fraction_used(week_start, week_end)

        over = budget_fraction > fraction_elapsed + self._config.hysteresis_over
        under = budget_fraction < fraction_elapsed - self._config.hysteresis_under

        if over:
            self._currently_over_pace = True
            return False
        elif under:
            self._currently_over_pace = False
            return True
        else:
            # In the hysteresis band: maintain current state.
            return not self._currently_over_pace

    def record_usage(
        self,
        cost_usd: float,
        model: str | None = None,
    ) -> None:
        """Record an in-flight run's cost so it counts toward the budget
        before the next Langfuse cache refresh.

        *cost_usd* is the raw USD cost (e.g. from the SDK's
        ``total_cost_usd``). *model* is an optional model alias (``"opus"``,
        ``"sonnet"``, ``"haiku"``) used to apply the per-model weight
        multiplier from ``model_weights``.
        """
        weight = self._model_weight(model)
        self._in_process_cost += cost_usd * weight

    # ------------------------------------------------------------------ #
    #  Internal                                                          #
    # ------------------------------------------------------------------ #

    def _fetch_weekly_cost(
        self, week_start: datetime, week_end: datetime
    ) -> float | None:
        """Query Langfuse for the total Claude cost this week.

        Returns None when the cost source is unavailable and
        ``fail_open`` is True.
        """
        if self._cost_source is None:
            return None

        try:
            result = self._cost_source.fetch_logged_cost_by_provider(
                CostWindow(start=week_start, end=week_end),
                _PROVIDER_FILTER,
            )
        except (RuntimeError, httpx.HTTPError):
            if self._config.fail_open:
                logger.warning(
                    "Pace governor: Langfuse query failed, failing open "
                    "(defaulting to Claude)"
                )
                return None
            raise

        return result.total_cost

    def _budget_fraction_used(self, week_start: datetime, week_end: datetime) -> float:
        """Return the fraction of the weekly budget consumed so far.

        Aggregates the cached Langfuse total (refreshed on TTL expiry)
        plus in-process increments.
        """
        if self._config.weekly_budget <= 0:
            return 0.0

        now_mono = _time.monotonic()
        cache_expired = self._cache_timestamp is None or (
            (now_mono - self._cache_timestamp) >= self._config.cache_ttl_seconds
        )

        if self._cost_source is not None and cache_expired:
            fetched = self._fetch_weekly_cost(week_start, week_end)
            self._cache_timestamp = now_mono
            if fetched is not None:
                self._cached_weekly_cost = fetched
                # Reset in-process accumulator on successful refresh — those
                # runs are now accounted for in the Langfuse total.
                self._in_process_cost = 0.0
            else:
                # fail_open: cost source configured but unreachable.
                # Seed cache with 0.0 so we don't re-query on every call
                # for the duration of the outage (the is-None check
                # overrides the TTL guard otherwise).
                self._cached_weekly_cost = 0.0
                self._in_process_cost = 0.0
                return 0.0
        elif cache_expired:
            # No cost source configured; seed cache with 0.0 and bump
            # timestamp so we don't re-enter this branch on every call.
            self._cached_weekly_cost = 0.0
            self._cache_timestamp = now_mono

        total = self._cached_weekly_cost + self._in_process_cost
        return total / self._config.weekly_budget

    def _model_weight(self, model: str | None) -> float:
        """Resolve the per-model weight multiplier from config."""
        if model is None:
            return 1.0
        weights = self._config.model_weights
        # Only resolve known model fields (avoids dynamic getattr on
        # arbitrary strings).
        if model == "opus":
            return weights.opus
        if model == "sonnet":
            return weights.sonnet
        if model == "haiku":
            return weights.haiku
        return 1.0

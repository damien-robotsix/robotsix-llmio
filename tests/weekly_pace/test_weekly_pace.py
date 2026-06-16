"""Unit tests for the weekly Claude usage pace governor.

Covers:
- Decision math (over/under pace, hysteresis no-flap, week-boundary reset)
- Model weighting
- fail_open path
- Caching and in-process increments
- Disabled governor (always True)
- always_claude_agents bypass
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from robotsix_llmio.config.weekly_pace import ModelWeightConfig, WeeklyPaceConfig
from robotsix_llmio.core.langfuse_cost import LangfuseCostLogSource
from robotsix_llmio.weekly_pace import PaceGovernor
from tests.core.conftest import install_transport

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

# Monday 2026-06-08 00:00 UTC is a known Monday (weekday 0).
_MONDAY = datetime(2026, 6, 8, 0, 0, 0, tzinfo=UTC)


def _config(**kwargs: Any) -> WeeklyPaceConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "weekly_budget": 10.0,
        "week_anchor_day": 0,  # Monday
        "week_anchor_time": "00:00",
        "hysteresis_over": 0.05,
        "hysteresis_under": 0.05,
        "cache_ttl_seconds": 120,
    }
    defaults.update(kwargs)
    return WeeklyPaceConfig(**defaults)


def _cost_source(
    monkeypatch: pytest.MonkeyPatch, cost: float = 0.0
) -> LangfuseCostLogSource:
    """Return a LangfuseCostLogSource whose observations endpoint returns
    *cost* total across all matched observations."""

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", 1))
        if page == 1:
            obs = {
                "id": "obs-1",
                "calculatedTotalCost": cost,
                "startTime": "2026-06-08T10:00:00Z",
                "metadata": {"provider": "claude-sdk"},
            }
            return httpx.Response(200, json={"data": [obs]})
        return httpx.Response(200, json={"data": []})

    install_transport(monkeypatch, handler)
    return LangfuseCostLogSource(
        public_key="pk", secret_key="sk", base_url="https://lf.example.com"
    )


# --------------------------------------------------------------------------- #
#  Disabled governor — always returns True                                     #
# --------------------------------------------------------------------------- #


def test_disabled_governor_always_true():
    """When enabled=False, should_use_claude always returns True."""
    config = _config(enabled=False)
    governor = PaceGovernor(config)
    assert governor.should_use_claude(_MONDAY) is True
    assert governor.should_use_claude(_MONDAY + timedelta(days=3)) is True


# --------------------------------------------------------------------------- #
#  always_claude_agents bypass                                                #
# --------------------------------------------------------------------------- #


def test_always_claude_agents_bypass(monkeypatch):
    """Agents in always_claude_agents always get True even when over pace."""
    config = _config(always_claude_agents=["planner"])
    # Set up high Langfuse cost so we're over pace.
    source = _cost_source(monkeypatch, cost=9.0)
    governor = PaceGovernor(config, source)
    # Record additional in-process cost to push over.
    governor.record_usage(1.5)

    # Late in the week (fraction ≈ 0.7), budget fraction ≈ 1.05 → over pace.
    now = _MONDAY + timedelta(days=5)  # Friday
    # Non-exempt agent: should fall back.
    assert governor.should_use_claude(now, agent_name="worker") is False
    # Exempt agent: always Claude.
    assert governor.should_use_claude(now, agent_name="planner") is True


# --------------------------------------------------------------------------- #
#  Decision math — under pace → Claude                                        #
# --------------------------------------------------------------------------- #


def test_under_pace_uses_claude(monkeypatch):
    """When budget consumption is well under the elapsed week fraction,
    should_use_claude returns True."""
    source = _cost_source(monkeypatch, cost=0.0)
    governor = PaceGovernor(_config(), source)
    # Mid-week (fraction ≈ 0.5), zero cost → well under pace.
    now = _MONDAY + timedelta(days=3, hours=12)
    assert governor.should_use_claude(now) is True


# --------------------------------------------------------------------------- #
#  Decision math — over pace → DeepSeek                                       #
# --------------------------------------------------------------------------- #


def test_over_pace_falls_back(monkeypatch):
    """When budget consumption exceeds the elapsed week fraction by more
    than the hysteresis margin, should_use_claude returns False."""
    # 9.0 cost out of 10.0 budget → fraction = 0.9
    source = _cost_source(monkeypatch, cost=9.0)
    governor = PaceGovernor(_config(hysteresis_over=0.05), source)
    # Early in the week (fraction ≈ 0.1), budget 0.9 → over pace by 0.8 > 0.05.
    now = _MONDAY + timedelta(hours=17)  # ~0.1 of week
    assert governor.should_use_claude(now) is False


# --------------------------------------------------------------------------- #
#  Hysteresis — no flap at boundary                                           #
# --------------------------------------------------------------------------- #


def test_hysteresis_band_maintains_state(monkeypatch):
    """When budget fraction is within the hysteresis band, the governor
    maintains its previous state rather than flapping."""
    config = _config(hysteresis_over=0.05, hysteresis_under=0.05)
    source = _cost_source(monkeypatch, cost=5.0)  # 0.5 budget
    governor = PaceGovernor(config, source)
    # Week fraction ≈ 0.5 → exactly on pace.
    now = _MONDAY + timedelta(days=3, hours=12)
    # First call: state is False (default), so returns True (under).
    assert governor.should_use_claude(now) is True
    # Second call at same time: still True (state maintained).
    assert governor.should_use_claude(now) is True


def test_hysteresis_transitions_on_clear_signal(monkeypatch):
    """Clear over-pace signal transitions state to over (→ False);
    clear under-pace signal transitions back."""
    config = _config(hysteresis_over=0.05, hysteresis_under=0.05)

    # Use a handler that returns different costs on successive calls.
    costs = [9.0, 1.0]
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", 1))
        if page == 1:
            cost = costs[min(call_count[0], len(costs) - 1)]
            obs = {
                "id": "obs-1",
                "calculatedTotalCost": cost,
                "startTime": "2026-06-08T10:00:00Z",
                "metadata": {"provider": "claude-sdk"},
            }
            return httpx.Response(200, json={"data": [obs]})
        return httpx.Response(200, json={"data": []})

    install_transport(monkeypatch, handler)
    source = LangfuseCostLogSource(
        public_key="pk", secret_key="sk", base_url="https://lf.example.com"
    )

    # First governor: early week, high cost → over pace.
    governor1 = PaceGovernor(config, source)
    now_early = _MONDAY + timedelta(hours=17)
    assert governor1.should_use_claude(now_early) is False
    call_count[0] += 1

    # Second governor (fresh): mid week, now low cost → under pace.
    governor2 = PaceGovernor(config, source)
    now_mid = _MONDAY + timedelta(days=3, hours=12)
    assert governor2.should_use_claude(now_mid) is True


# --------------------------------------------------------------------------- #
#  Week-boundary reset                                                        #
# --------------------------------------------------------------------------- #


def test_week_boundary_reset(monkeypatch):
    """At the start of a new week, fraction_elapsed ≈ 0, so a fresh
    query for the current week returns zero cost (last week's cost is
    outside the window)."""

    # Mock that is window-aware: only returns cost when the window
    # overlaps with the query range.
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", 1))
        from_start = request.url.params.get("fromStartTime", "")
        if page == 1 and "2026-06-08" in from_start:
            # This is the new week — return 0 cost (last week's
            # cost fell before this window).
            return httpx.Response(200, json={"data": []})
        if page == 1:
            obs = {
                "id": "obs-1",
                "calculatedTotalCost": 9.0,
                "startTime": "2026-06-01T10:00:00Z",
                "metadata": {"provider": "claude-sdk"},
            }
            return httpx.Response(200, json={"data": [obs]})
        return httpx.Response(200, json={"data": []})

    install_transport(monkeypatch, handler)
    source = LangfuseCostLogSource(
        public_key="pk", secret_key="sk", base_url="https://lf.example.com"
    )
    governor = PaceGovernor(_config(), source)
    # Just after the Monday anchor — fraction ≈ 0, fresh query returns
    # 0 cost for this week.
    now = _MONDAY + timedelta(minutes=1)
    assert governor.should_use_claude(now) is True


# --------------------------------------------------------------------------- #
#  Model weighting                                                            #
# --------------------------------------------------------------------------- #


def test_model_weighting_opus_vs_haiku():
    """The same USD cost on Opus advances budget_fraction_used more than
    on Haiku when per-model weights are configured."""
    config = _config(
        weekly_budget=10.0,
        model_weights=ModelWeightConfig(opus=5.0, sonnet=3.0, haiku=1.0),
    )

    # Record same USD cost on Opus vs Haiku.
    gov_opus = PaceGovernor(config)
    gov_opus.record_usage(1.0, model="opus")

    gov_haiku = PaceGovernor(config)
    gov_haiku.record_usage(1.0, model="haiku")

    # Opus weighted cost = 1.0 * 5.0 = 5.0 → fraction 0.5
    # Haiku weighted cost = 1.0 * 1.0 = 1.0 → fraction 0.1
    assert gov_opus._in_process_cost == pytest.approx(5.0)
    assert gov_haiku._in_process_cost == pytest.approx(1.0)


def test_model_weighting_default_unity():
    """With default weights (all 1.0), raw USD cost is used as-is."""
    config = _config(weekly_budget=10.0)
    governor = PaceGovernor(config)
    governor.record_usage(2.5, model="opus")
    governor.record_usage(1.5, model="haiku")
    assert governor._in_process_cost == pytest.approx(4.0)


def test_model_weighting_unknown_model_defaults_to_one():
    """An unrecognized model name gets weight 1.0."""
    config = _config(
        model_weights=ModelWeightConfig(opus=5.0, sonnet=3.0, haiku=1.0),
    )
    governor = PaceGovernor(config)
    governor.record_usage(2.0, model="unknown-model")
    assert governor._in_process_cost == pytest.approx(2.0)


def test_model_weighting_none_model():
    """A None model gets weight 1.0."""
    config = _config(
        model_weights=ModelWeightConfig(opus=5.0),
    )
    governor = PaceGovernor(config)
    governor.record_usage(3.0, model=None)
    assert governor._in_process_cost == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
#  In-process increment                                                       #
# --------------------------------------------------------------------------- #


def test_in_process_increment_counts_toward_budget(monkeypatch):
    """record_usage adds to the budget fraction alongside Langfuse cost."""
    source = _cost_source(monkeypatch, cost=3.0)  # 0.3 budget
    governor = PaceGovernor(_config(weekly_budget=10.0), source)
    governor.record_usage(2.0, model="haiku")  # +0.2 → total 0.5

    now = _MONDAY + timedelta(days=3, hours=12)  # 0.5 week
    # budget = 0.5, week = 0.5 → in hysteresis band, default state False → True
    assert governor.should_use_claude(now) is True


def test_in_process_accumulates_between_refreshes(monkeypatch):
    """Multiple record_usage calls accumulate until the next cache refresh."""
    source = _cost_source(monkeypatch, cost=0.0)
    governor = PaceGovernor(_config(weekly_budget=10.0, cache_ttl_seconds=999), source)
    governor.record_usage(1.0)
    governor.record_usage(2.0)
    governor.record_usage(0.5)
    assert governor._in_process_cost == pytest.approx(3.5)


# --------------------------------------------------------------------------- #
#  Cache TTL                                                                  #
# --------------------------------------------------------------------------- #


def test_cache_refresh_after_ttl(monkeypatch):
    """After the cache TTL expires, the Langfuse cost is re-fetched and
    the in-process accumulator is reset on successful fetch."""
    # Use a handler that tracks call count.
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        page = int(request.url.params.get("page", 1))
        if page == 1:
            obs = {
                "id": "obs-1",
                "calculatedTotalCost": 2.0,
                "startTime": "2026-06-08T10:00:00Z",
                "metadata": {"provider": "claude-sdk"},
            }
            return httpx.Response(200, json={"data": [obs]})
        return httpx.Response(200, json={"data": []})

    install_transport(monkeypatch, handler)
    source = LangfuseCostLogSource(
        public_key="pk", secret_key="sk", base_url="https://lf.example.com"
    )

    # Use a fake monotonic clock.
    fake_time = [0.0]

    def fake_monotonic() -> float:
        return fake_time[0]

    monkeypatch.setattr("time.monotonic", fake_monotonic)

    governor = PaceGovernor(_config(weekly_budget=10.0, cache_ttl_seconds=60), source)
    governor.record_usage(1.0)

    # First call: fetches 2.0 from Langfuse, adds 1.0 in-process = 3.0.
    now = _MONDAY + timedelta(days=3)
    governor.should_use_claude(now)
    # Two requests: page 1 (data) + page 2 (empty → break loop).
    assert call_count[0] == 2
    # After the call, cache was refreshed (first fetch ever).
    # In-process should be reset to 0, cached = 2.0.
    assert governor._in_process_cost == 0.0
    assert governor._cached_weekly_cost == pytest.approx(2.0)

    # Advance time past TTL and add more in-process cost.
    fake_time[0] = 120.0
    governor.record_usage(0.5)

    # Second call: TTL expired, re-fetches, resets in-process.
    governor.should_use_claude(now)
    assert call_count[0] == 4
    assert governor._in_process_cost == 0.0


# --------------------------------------------------------------------------- #
#  fail_open                                                                  #
# --------------------------------------------------------------------------- #


def test_fail_open_returns_true_on_langfuse_error(monkeypatch):
    """When Langfuse raises and fail_open=True, should_use_claude returns
    True (use Claude) and logs a warning."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    install_transport(monkeypatch, handler)
    source = LangfuseCostLogSource(
        public_key="pk", secret_key="sk", base_url="https://lf.example.com"
    )
    governor = PaceGovernor(_config(fail_open=True), source)
    # Should not raise; should return True.
    now = _MONDAY + timedelta(days=3, hours=12)
    assert governor.should_use_claude(now) is True


def test_fail_open_with_prior_successful_fetch(monkeypatch):
    """When fail_open=True and Langfuse becomes unreachable after a prior
    successful fetch recorded high cost, the governor stays on Claude
    rather than making decisions on stale data."""

    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        page = int(request.url.params.get("page", 1))
        if call_count[0] <= 2:  # first fetch (page 1 + page 2) succeeds
            if page == 1:
                obs = {
                    "id": "obs-1",
                    "calculatedTotalCost": 9.0,
                    "startTime": "2026-06-08T10:00:00Z",
                    "metadata": {"provider": "claude-sdk"},
                }
                return httpx.Response(200, json={"data": [obs]})
            return httpx.Response(200, json={"data": []})
        # Subsequent requests simulate a Langfuse outage.
        return httpx.Response(500, json={"error": "internal"})

    install_transport(monkeypatch, handler)
    source = LangfuseCostLogSource(
        public_key="pk", secret_key="sk", base_url="https://lf.example.com"
    )

    fake_time = [0.0]

    def fake_monotonic() -> float:
        return fake_time[0]

    monkeypatch.setattr("time.monotonic", fake_monotonic)

    governor = PaceGovernor(
        _config(weekly_budget=10.0, cache_ttl_seconds=60, fail_open=True),
        source,
    )

    # First call — early week, high cost → over pace → False.
    now_early = _MONDAY + timedelta(hours=17)
    assert governor.should_use_claude(now_early) is False
    assert governor._cached_weekly_cost == pytest.approx(9.0)

    # Advance past TTL and simulate Langfuse outage.
    fake_time[0] = 120.0

    # Second call — Langfuse is down but fail_open keeps us on Claude.
    assert governor.should_use_claude(now_early) is True
    # Cache is seeded with 0.0 so we don't re-query on every call
    # during the outage.
    assert governor._cached_weekly_cost == pytest.approx(0.0)
    assert governor._in_process_cost == pytest.approx(0.0)


def test_fail_closed_raises_on_langfuse_error(monkeypatch):
    """When Langfuse raises and fail_open=False, the error propagates."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    install_transport(monkeypatch, handler)
    source = LangfuseCostLogSource(
        public_key="pk", secret_key="sk", base_url="https://lf.example.com"
    )
    governor = PaceGovernor(_config(fail_open=False), source)
    now = _MONDAY + timedelta(days=3, hours=12)
    with pytest.raises(RuntimeError):
        governor.should_use_claude(now)


# --------------------------------------------------------------------------- #
#  No cost source (testing / offline)                                         #
# --------------------------------------------------------------------------- #


def test_no_cost_source_only_in_process():
    """Without a cost source, only in-process increments are counted."""
    governor = PaceGovernor(_config(weekly_budget=10.0))
    governor.record_usage(6.0)  # 0.6 budget
    now = _MONDAY + timedelta(days=3, hours=12)  # 0.5 week
    # 0.6 > 0.5 + 0.05 → over pace → False
    assert governor.should_use_claude(now) is False


def test_zero_budget_never_over_pace():
    """A zero or negative weekly budget always results in 0.0 fraction."""
    governor = PaceGovernor(_config(weekly_budget=0.0))
    governor.record_usage(1000.0)
    now = _MONDAY + timedelta(minutes=1)
    assert governor.should_use_claude(now) is True

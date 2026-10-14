"""Unit tests for the failover call loop — sync and async entry points."""

from __future__ import annotations

import asyncio

import httpx
import pytest

import robotsix_llmio.core.failover as failover_mod
from robotsix_llmio.config.tier import (
    FailoverConfig,
    ProviderSlotConfig,
    TierConfig,
    TierLevelConfig,
)
from robotsix_llmio.core.failover import (
    _ATTR_COST_GUARD_BLOCKED,
    _ATTR_COST_GUARD_REASON,
    _ATTR_MAX_SLOT_SWITCHES,
    _ATTR_SLOT,
    _ATTR_SLOT_SWITCHES,
    _MAX_SLOT_SWITCHES,
    _SPAN_FAILOVER_ATTEMPT,
    _SPAN_PRIMARY_ATTEMPT,
    acall_with_failover,
    call_with_failover,
    estimate_request_cost,
    get_failover_tracker,
)
from robotsix_llmio.exceptions import FailoverCostGuardError, ProviderExhaustedError

_CFG = TierConfig(
    default=ProviderSlotConfig(
        level1=TierLevelConfig(model="claudeSDK-haiku"),
        level2=TierLevelConfig(model="claudeSDK-opus"),
        level3=TierLevelConfig(model="claudeSDK-claude-fable-5"),
    ),
    fallback=ProviderSlotConfig(
        level1=TierLevelConfig(model="openrouter-deepseek/flash"),
        level2=TierLevelConfig(model="openrouter-deepseek/flash"),
        level3=TierLevelConfig(model="openrouter-deepseek/pro"),
    ),
    failover=FailoverConfig(failure_threshold=3, window_seconds=900.0),
)


class _Exhausted(ProviderExhaustedError):
    pass


def _factory(script: dict[str, list[Exception | str]], calls: list[str]):
    """Factory whose callable pops the next scripted outcome for its model.

    *script* maps model identifier → list of outcomes (an Exception instance
    to raise, or a value to return). *calls* records the models attempted.
    """

    def factory(tlc: TierLevelConfig):
        def fn():
            calls.append(tlc.model)
            outcome = script[tlc.model].pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return fn

    return factory


def _afactory(script: dict[str, list[Exception | str]], calls: list[str]):
    def factory(tlc: TierLevelConfig):
        async def fn():
            calls.append(tlc.model)
            outcome = script[tlc.model].pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return fn

    return factory


def test_success_on_default_touches_nothing_else():
    calls: list[str] = []
    result = call_with_failover(
        _factory({"claudeSDK-opus": ["ok"]}, calls),
        tier_config=_CFG,
        level=2,
    )
    assert result == "ok"
    assert calls == ["claudeSDK-opus"]


def test_provider_shaped_failure_retries_same_level_on_fallback():
    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                "claudeSDK-opus": [httpx.ReadTimeout("down")],
                "openrouter-deepseek/flash": ["rescued"],
            },
            calls,
        ),
        tier_config=_CFG,
        level=2,
    )
    assert result == "rescued"
    # Same LEVEL on both slots — never a different level.
    assert calls == ["claudeSDK-opus", "openrouter-deepseek/flash"]


def test_task_shaped_failure_raises_without_failover():
    calls: list[str] = []
    with pytest.raises(ValueError):
        call_with_failover(
            _factory({"claudeSDK-opus": [ValueError("caller bug")]}, calls),
            tier_config=_CFG,
            level=2,
        )
    assert calls == ["claudeSDK-opus"]
    # Task-shaped failures do not count toward the provider streak.
    assert get_failover_tracker().status().consecutive_failures == 0


def test_failover_disabled_raises_on_first_failure():
    calls: list[str] = []
    with pytest.raises(httpx.ReadTimeout):
        call_with_failover(
            _factory({"claudeSDK-opus": [httpx.ReadTimeout("down")]}, calls),
            tier_config=_CFG,
            level=2,
            failover_enabled=False,
        )
    assert calls == ["claudeSDK-opus"]


def test_both_slots_failing_raises_the_fallback_error():
    calls: list[str] = []
    with pytest.raises(httpx.ConnectError):
        call_with_failover(
            _factory(
                {
                    "claudeSDK-opus": [httpx.ReadTimeout("down")],
                    "openrouter-deepseek/flash": [httpx.ConnectError("also down")],
                },
                calls,
            ),
            tier_config=_CFG,
            level=2,
        )
    assert calls == ["claudeSDK-opus", "openrouter-deepseek/flash"]


def test_exhaustion_routes_subsequent_calls_straight_to_fallback():
    calls: list[str] = []
    call_with_failover(
        _factory(
            {
                "claudeSDK-opus": [_Exhausted("out of credits")],
                "openrouter-deepseek/flash": ["rescued"],
            },
            calls,
        ),
        tier_config=_CFG,
        level=2,
    )
    # Next call: window is armed, the doomed default attempt is skipped.
    calls.clear()
    result = call_with_failover(
        _factory({"openrouter-deepseek/flash": ["direct"]}, calls),
        tier_config=_CFG,
        level=2,
    )
    assert result == "direct"
    assert calls == ["openrouter-deepseek/flash"]


def test_during_window_fallback_failure_still_tries_default():
    """During the window the default slot is the *second* attempt — a
    last-ditch try beats failing the call outright."""
    tracker = get_failover_tracker()
    tracker.configure(_CFG.failover)
    tracker.record_failure("default", _Exhausted("out"))

    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                "openrouter-deepseek/flash": [httpx.ReadTimeout("down too")],
                "claudeSDK-opus": ["surprise recovery"],
            },
            calls,
        ),
        tier_config=_CFG,
        level=2,
    )
    assert result == "surprise recovery"
    assert calls == ["openrouter-deepseek/flash", "claudeSDK-opus"]


def test_threshold_arming_across_calls():
    calls: list[str] = []
    for _ in range(3):
        call_with_failover(
            _factory(
                {
                    "claudeSDK-haiku": [httpx.ReadTimeout("flaky")],
                    "openrouter-deepseek/flash": ["rescued"],
                },
                calls,
            ),
            tier_config=_CFG,
            level=1,
        )
    # Three consecutive default failures — the window is armed now.
    assert get_failover_tracker().active_slot() == "fallback"
    calls.clear()
    call_with_failover(
        _factory({"openrouter-deepseek/flash": ["direct"]}, calls),
        tier_config=_CFG,
        level=1,
    )
    assert calls == ["openrouter-deepseek/flash"]


def test_async_mirror_fails_over_and_records():
    calls: list[str] = []
    result = asyncio.run(
        acall_with_failover(
            _afactory(
                {
                    "claudeSDK-claude-fable-5": [httpx.ReadTimeout("down")],
                    "openrouter-deepseek/pro": ["rescued"],
                },
                calls,
            ),
            tier_config=_CFG,
            level=3,
        )
    )
    assert result == "rescued"
    assert calls == ["claudeSDK-claude-fable-5", "openrouter-deepseek/pro"]
    assert get_failover_tracker().status().consecutive_failures == 1


def test_async_success_resets_streak():
    calls: list[str] = []
    asyncio.run(
        acall_with_failover(
            _afactory({"claudeSDK-opus": ["ok"]}, calls),
            tier_config=_CFG,
            level=2,
        )
    )
    assert get_failover_tracker().status().consecutive_failures == 0


class _RecordingSpan:
    """Fake span accumulating ``set_attribute`` writes."""

    def __init__(self, name: str, attributes: dict[str, object]) -> None:
        self.name = name
        self.attributes = dict(attributes)

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


def _install_span_recorder(monkeypatch) -> list[_RecordingSpan]:
    """Patch ``start_span`` in the failover module to capture span name +
    attributes, and return the list spans are appended to."""
    recorded: list[_RecordingSpan] = []

    import contextlib

    @contextlib.contextmanager
    def _fake_start_span(tracer, name, attributes=None):
        span = _RecordingSpan(name, attributes or {})
        recorded.append(span)
        yield span

    monkeypatch.setattr(failover_mod, "start_span", _fake_start_span)
    return recorded


def test_default_slot_success_uses_non_failover_span_name(monkeypatch):
    recorded = _install_span_recorder(monkeypatch)
    calls: list[str] = []
    call_with_failover(
        _factory({"claudeSDK-opus": ["ok"]}, calls),
        tier_config=_CFG,
        level=2,
    )
    assert [s.name for s in recorded] == [_SPAN_PRIMARY_ATTEMPT]
    assert "failover" not in _SPAN_PRIMARY_ATTEMPT
    span = recorded[0]
    assert span.attributes[_ATTR_SLOT] == "default"


def test_fallback_slot_attempt_uses_failover_span_name(monkeypatch):
    recorded = _install_span_recorder(monkeypatch)
    calls: list[str] = []
    call_with_failover(
        _factory(
            {
                "claudeSDK-opus": [httpx.ReadTimeout("down")],
                "openrouter-deepseek/flash": ["rescued"],
            },
            calls,
        ),
        tier_config=_CFG,
        level=2,
    )
    # First attempt is the primary tier, the escalation is the failover span.
    assert [s.name for s in recorded] == [
        _SPAN_PRIMARY_ATTEMPT,
        _SPAN_FAILOVER_ATTEMPT,
    ]
    # The two attempts land under DIFFERENT span names.
    assert recorded[0].name != recorded[1].name
    assert recorded[1].attributes[_ATTR_SLOT] == "fallback"


# --------------------------------------------------------------------------- #
#  One-slot-switch retry cap + telemetry                                      #
# --------------------------------------------------------------------------- #


def test_slot_switch_cap_is_one():
    """A request attempts the active slot plus at most one fallback."""
    assert _MAX_SLOT_SWITCHES == 1


def test_request_switches_slots_at_most_once(monkeypatch):
    """Even with both slots failing, the loop makes exactly two attempts
    (one switch) and records the cap + switch count on the run span."""
    run_span = _RecordingSpan("run", {})
    monkeypatch.setattr(failover_mod, "get_recording_span", lambda: run_span)

    calls: list[str] = []
    with pytest.raises(httpx.ConnectError):
        call_with_failover(
            _factory(
                {
                    "claudeSDK-opus": [httpx.ReadTimeout("down")],
                    "openrouter-deepseek/flash": [httpx.ConnectError("also down")],
                },
                calls,
            ),
            tier_config=_CFG,
            level=2,
        )
    # Exactly one switch: active slot, then the other — never a third attempt.
    assert calls == ["claudeSDK-opus", "openrouter-deepseek/flash"]
    assert run_span.attributes[_ATTR_MAX_SLOT_SWITCHES] == 1
    assert run_span.attributes[_ATTR_SLOT_SWITCHES] == 1


def test_default_success_records_no_slot_switch(monkeypatch):
    run_span = _RecordingSpan("run", {})
    monkeypatch.setattr(failover_mod, "get_recording_span", lambda: run_span)

    calls: list[str] = []
    call_with_failover(
        _factory({"claudeSDK-opus": ["ok"]}, calls),
        tier_config=_CFG,
        level=2,
    )
    assert calls == ["claudeSDK-opus"]
    assert run_span.attributes[_ATTR_MAX_SLOT_SWITCHES] == 1
    # No switch happened, so the switch-count attribute is never written.
    assert _ATTR_SLOT_SWITCHES not in run_span.attributes


# --------------------------------------------------------------------------- #
#  Per-request cost guard                                                     #
# --------------------------------------------------------------------------- #


def _priced(model: str, *, price_completion: float, max_tokens: int):
    """An OpenRouter TierLevelConfig whose per-request cost is estimable."""
    return TierLevelConfig(
        model=model,
        max_tokens=max_tokens,
        provider_kwargs={"max_price_completion": price_completion},
    )


def _guard_cfg(default_l2, fallback_l2, failover: FailoverConfig) -> TierConfig:
    """A TierConfig whose level-2 bindings are the two slots under test.

    Levels 1 and 3 are irrelevant filler — every guard test calls level 2.
    """
    return TierConfig(
        default=ProviderSlotConfig(
            level1=TierLevelConfig(model="claudeSDK-haiku"),
            level2=default_l2,
            level3=TierLevelConfig(model="claudeSDK-claude-fable-5"),
        ),
        fallback=ProviderSlotConfig(
            level1=TierLevelConfig(model="openrouter-deepseek/l1-filler"),
            level2=fallback_l2,
            level3=TierLevelConfig(model="openrouter-deepseek/l3-filler"),
        ),
        failover=failover,
    )


def test_estimate_request_cost_uses_completion_ceiling_times_cap():
    tlc = _priced("openrouter-deepseek/flash", price_completion=5.0, max_tokens=100_000)
    # 5.0 USD / 1M tokens * 100_000 tokens = 0.5 USD.
    assert estimate_request_cost(tlc) == pytest.approx(0.5)


def test_estimate_request_cost_none_when_unpriced_or_uncapped():
    # Claude SDK slot: no price ceiling, no output cap.
    assert estimate_request_cost(TierLevelConfig(model="claudeSDK-opus")) is None
    # Priced but uncapped.
    assert (
        estimate_request_cost(
            TierLevelConfig(
                model="openrouter-deepseek/flash",
                provider_kwargs={"max_price_completion": 5.0},
            )
        )
        is None
    )
    # Capped but unpriced.
    assert (
        estimate_request_cost(
            TierLevelConfig(model="openrouter-deepseek/flash", max_tokens=100_000)
        )
        is None
    )


def test_ceiling_blocks_expensive_switch(monkeypatch):
    run_span = _RecordingSpan("run", {})
    monkeypatch.setattr(failover_mod, "get_recording_span", lambda: run_span)

    cfg = _guard_cfg(
        TierLevelConfig(model="claudeSDK-opus"),
        _priced("openrouter-deepseek/pricey", price_completion=5.0, max_tokens=100_000),
        FailoverConfig(max_request_cost_usd=0.1),
    )
    calls: list[str] = []
    with pytest.raises(FailoverCostGuardError) as excinfo:
        call_with_failover(
            _factory({"claudeSDK-opus": [httpx.ReadTimeout("down")]}, calls),
            tier_config=cfg,
            level=2,
        )
    # The expensive fallback slot was NEVER attempted — request failed fast.
    assert calls == ["claudeSDK-opus"]
    # The original provider failure is chained as the cause.
    assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)
    # The refusal is recorded on the run span.
    assert run_span.attributes[_ATTR_COST_GUARD_BLOCKED] is True
    assert "ceiling" in run_span.attributes[_ATTR_COST_GUARD_REASON]
    # The provider failure still counted toward the streak.
    assert get_failover_tracker().status().consecutive_failures == 1


def test_ceiling_boundary_equal_is_allowed():
    """An estimate exactly at the ceiling is permitted (strict >)."""
    cfg = _guard_cfg(
        TierLevelConfig(model="claudeSDK-opus"),
        _priced("openrouter-deepseek/edge", price_completion=5.0, max_tokens=100_000),
        FailoverConfig(max_request_cost_usd=0.5),  # exactly the estimate
    )
    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                "claudeSDK-opus": [httpx.ReadTimeout("down")],
                "openrouter-deepseek/edge": ["rescued"],
            },
            calls,
        ),
        tier_config=cfg,
        level=2,
    )
    assert result == "rescued"
    assert calls == ["claudeSDK-opus", "openrouter-deepseek/edge"]


def test_cheaper_candidate_still_fails_over():
    """Same-or-cheaper candidates are never blocked by the ratio check."""
    cfg = _guard_cfg(
        _priced("openrouter-deepseek/pricey", price_completion=5.0, max_tokens=100_000),
        _priced("openrouter-deepseek/cheap", price_completion=1.0, max_tokens=100_000),
        FailoverConfig(cost_ratio_threshold=2.0),
    )
    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                "openrouter-deepseek/pricey": [httpx.ReadTimeout("down")],
                "openrouter-deepseek/cheap": ["rescued"],
            },
            calls,
        ),
        tier_config=cfg,
        level=2,
    )
    assert result == "rescued"
    assert calls == ["openrouter-deepseek/pricey", "openrouter-deepseek/cheap"]


def test_ratio_blocks_materially_more_expensive():
    cfg = _guard_cfg(
        _priced("openrouter-deepseek/cheap", price_completion=1.0, max_tokens=100_000),
        _priced("openrouter-deepseek/pricey", price_completion=5.0, max_tokens=100_000),
        FailoverConfig(cost_ratio_threshold=2.0),  # 0.5 > 0.1 * 2 = 0.2 -> block
    )
    calls: list[str] = []
    with pytest.raises(FailoverCostGuardError) as excinfo:
        call_with_failover(
            _factory({"openrouter-deepseek/cheap": [httpx.ReadTimeout("down")]}, calls),
            tier_config=cfg,
            level=2,
        )
    assert calls == ["openrouter-deepseek/cheap"]
    assert "more than" in str(excinfo.value)


def test_ratio_boundary_equal_is_allowed():
    """Candidate at exactly primary * ratio is permitted (strict >)."""
    cfg = _guard_cfg(
        _priced("openrouter-deepseek/half", price_completion=2.5, max_tokens=100_000),
        _priced("openrouter-deepseek/full", price_completion=5.0, max_tokens=100_000),
        FailoverConfig(cost_ratio_threshold=2.0),  # 0.5 == 0.25 * 2 -> allowed
    )
    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                "openrouter-deepseek/half": [httpx.ReadTimeout("down")],
                "openrouter-deepseek/full": ["rescued"],
            },
            calls,
        ),
        tier_config=cfg,
        level=2,
    )
    assert result == "rescued"
    assert calls == ["openrouter-deepseek/half", "openrouter-deepseek/full"]


def test_missing_candidate_estimate_allows_switch():
    """An unpriceable candidate (Claude SDK default) is never blocked.

    During an armed window the active slot is ``fallback`` and the candidate
    is the unpriced ``default`` slot — the guard has nothing to price, so the
    switch proceeds even with a tiny ceiling configured.
    """
    cfg = _guard_cfg(
        TierLevelConfig(model="claudeSDK-opus"),
        _priced("openrouter-deepseek/flash", price_completion=5.0, max_tokens=100_000),
        FailoverConfig(max_request_cost_usd=0.001, cost_ratio_threshold=1.0),
    )
    tracker = get_failover_tracker()
    tracker.configure(cfg.failover)
    tracker.record_failure("default", _Exhausted("out"))  # arm -> active fallback

    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                "openrouter-deepseek/flash": [httpx.ReadTimeout("down too")],
                "claudeSDK-opus": ["surprise recovery"],
            },
            calls,
        ),
        tier_config=cfg,
        level=2,
    )
    assert result == "surprise recovery"
    assert calls == ["openrouter-deepseek/flash", "claudeSDK-opus"]


def test_missing_primary_estimate_skips_ratio_check():
    """With only the ratio configured, an unpriced primary disables it.

    The Claude SDK default slot cannot be priced, so there is no baseline to
    compare the fallback against — the switch proceeds.
    """
    cfg = _guard_cfg(
        TierLevelConfig(model="claudeSDK-opus"),
        _priced("openrouter-deepseek/flash", price_completion=5.0, max_tokens=100_000),
        FailoverConfig(cost_ratio_threshold=1.0),  # no ceiling, primary unpriced
    )
    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                "claudeSDK-opus": [httpx.ReadTimeout("down")],
                "openrouter-deepseek/flash": ["rescued"],
            },
            calls,
        ),
        tier_config=cfg,
        level=2,
    )
    assert result == "rescued"
    assert calls == ["claudeSDK-opus", "openrouter-deepseek/flash"]


def test_guard_disabled_by_default_does_not_block():
    """No cost knobs configured -> behaviour is unchanged (switch proceeds)."""
    cfg = _guard_cfg(
        TierLevelConfig(model="claudeSDK-opus"),
        _priced(
            "openrouter-deepseek/pricey",
            price_completion=99.0,
            max_tokens=999_999,
        ),
        FailoverConfig(),  # both knobs None
    )
    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                "claudeSDK-opus": [httpx.ReadTimeout("down")],
                "openrouter-deepseek/pricey": ["rescued"],
            },
            calls,
        ),
        tier_config=cfg,
        level=2,
    )
    assert result == "rescued"
    assert calls == ["claudeSDK-opus", "openrouter-deepseek/pricey"]

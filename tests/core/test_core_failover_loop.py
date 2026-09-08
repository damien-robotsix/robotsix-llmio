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
    _ATTR_SLOT,
    _SPAN_FAILOVER_ATTEMPT,
    _SPAN_PRIMARY_ATTEMPT,
    acall_with_failover,
    call_with_failover,
    get_failover_tracker,
)
from robotsix_llmio.exceptions import ProviderExhaustedError

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

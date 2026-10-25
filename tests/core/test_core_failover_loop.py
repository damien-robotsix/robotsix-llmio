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
    _SPAN_FAILOVER_ATTEMPT,
    _SPAN_PRIMARY_ATTEMPT,
    _max_slot_switches,
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


def test_slot_switch_cap_is_one_for_two_slot_chain():
    """A request attempts each chain slot at most once: one switch for the
    legacy two-slot chain, ``N - 1`` for an N-slot chain."""
    assert _max_slot_switches(2) == 1
    assert _max_slot_switches(3) == 2
    assert _max_slot_switches(1) == 0


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


# --------------------------------------------------------------------------- #
#  N-slot cascade (>= 3 slots)                                                #
# --------------------------------------------------------------------------- #


def _chain_slot(prefix: str) -> ProviderSlotConfig:
    return ProviderSlotConfig(
        level1=TierLevelConfig(model=f"{prefix}-l1"),
        level2=TierLevelConfig(model=f"{prefix}-l2"),
        level3=TierLevelConfig(model=f"{prefix}-l3"),
    )


def _chain_cfg(failover: FailoverConfig | None = None, **levels2) -> TierConfig:
    slots = [
        _chain_slot("deepseek-direct"),
        _chain_slot("claudeSDK-sdk"),
        _chain_slot("openrouter-deepseek/x"),
    ]
    for index, tlc in levels2.items():
        slot = slots[int(index[1:])]
        slots[int(index[1:])] = slot.model_copy(update={"level2": tlc})
    return TierConfig(
        chain=slots,
        failover=failover
        if failover is not None
        else FailoverConfig(failure_threshold=1, window_seconds=900.0),
    )


_S0, _S1, _S2 = "deepseek-direct-l2", "claudeSDK-sdk-l2", "openrouter-deepseek/x-l2"


def test_three_slot_cascade_reaches_slot_three(monkeypatch):
    recorded = _install_span_recorder(monkeypatch)
    run_span = _RecordingSpan("run", {})
    monkeypatch.setattr(failover_mod, "get_recording_span", lambda: run_span)

    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                _S0: [httpx.ReadTimeout("down")],
                _S1: [httpx.ConnectError("down too")],
                _S2: ["rescued"],
            },
            calls,
        ),
        tier_config=_chain_cfg(),
        level=2,
    )
    assert result == "rescued"
    assert calls == [_S0, _S1, _S2]
    assert [s.attributes[_ATTR_SLOT] for s in recorded] == ["default", "fallback", 2]
    assert [s.name for s in recorded] == [
        _SPAN_PRIMARY_ATTEMPT,
        _SPAN_FAILOVER_ATTEMPT,
        _SPAN_FAILOVER_ATTEMPT,
    ]
    assert run_span.attributes[_ATTR_MAX_SLOT_SWITCHES] == 2
    assert run_span.attributes[_ATTR_SLOT_SWITCHES] == 2


def test_three_slot_exhaustion_raises_last_slot_error():
    calls: list[str] = []
    with pytest.raises(httpx.RemoteProtocolError):
        call_with_failover(
            _factory(
                {
                    _S0: [httpx.ReadTimeout("a")],
                    _S1: [httpx.ConnectError("b")],
                    _S2: [httpx.RemoteProtocolError("c")],
                },
                calls,
            ),
            tier_config=_chain_cfg(),
            level=2,
        )
    # Each slot tried exactly once — no ping-pong.
    assert calls == [_S0, _S1, _S2]


def test_three_slot_windows_route_subsequent_calls_down_the_chain():
    cfg = _chain_cfg()
    tracker = get_failover_tracker()
    calls: list[str] = []
    call_with_failover(
        _factory(
            {_S0: [httpx.ReadTimeout("a")], _S1: [_Exhausted("b")], _S2: ["ok"]},
            calls,
        ),
        tier_config=cfg,
        level=2,
    )
    # Slots 0 and 1 are both armed; the last slot never arms.
    assert tracker.active_slot() == 2
    status = tracker.status()
    assert status.active_slot == 2
    assert status.failover_active

    calls.clear()
    result = call_with_failover(
        _factory({_S2: ["direct"]}, calls), tier_config=cfg, level=2
    )
    assert result == "direct"
    assert calls == [_S2]
    # Resolution through the config follows the cascade too.
    assert cfg.for_level(2).model == _S2


def test_three_slot_window_expiry_climbs_back_up(monkeypatch):
    cfg = _chain_cfg(FailoverConfig(failure_threshold=1, window_seconds=100.0))
    tracker = get_failover_tracker()
    tracker.configure(cfg.failover, chain_length=3)
    tracker.record_failure(0, httpx.ReadTimeout("a"), now=0.0)
    tracker.record_failure(1, httpx.ReadTimeout("b"), now=50.0)
    assert tracker.active_slot(now=60.0) == 2
    # Slot 0's window expires first: calls probe position 0 again.
    assert tracker.active_slot(now=101.0) == "default"
    tracker.record_failure(0, httpx.ReadTimeout("a again"), now=101.0)
    # Slot 0 re-armed (backed off); slot 1 still armed until 150.
    assert tracker.active_slot(now=120.0) == 2
    # Slot 1 expires before slot 0's backed-off window.
    assert tracker.active_slot(now=151.0) == "fallback"


def test_three_slot_backoff_is_per_slot(monkeypatch):
    monkeypatch.setattr(failover_mod.random, "random", lambda: 0.0)
    cfg = FailoverConfig(
        failure_threshold=1, window_seconds=100.0, reattempt_backoff_base=2.0
    )
    tracker = get_failover_tracker()
    tracker.configure(cfg, chain_length=3)
    tracker.record_failure(0, httpx.ReadTimeout("a"), now=0.0)
    tracker.record_failure(0, httpx.ReadTimeout("a"), now=100.0)  # re-arm: 200s
    # Slot 1's first arm uses the plain window — slot 0's escalation is its own.
    tracker.record_failure(1, httpx.ReadTimeout("b"), now=100.0)
    assert tracker.active_slot(now=199.0) == 2
    assert tracker.active_slot(now=201.0) == "fallback"
    assert tracker.active_slot(now=301.0) == "default"


def test_three_slot_cost_guard_applies_at_each_hop():
    """The ceiling is checked before every switch — a cheap hop to slot 2
    proceeds, an expensive hop to slot 3 is refused."""
    cfg = _chain_cfg(
        FailoverConfig(failure_threshold=3, max_request_cost_usd=1.0),
        s1=_priced(_S1, price_completion=1.0, max_tokens=1000),
        s2=_priced(_S2, price_completion=10.0, max_tokens=1_000_000),
    )
    calls: list[str] = []
    with pytest.raises(FailoverCostGuardError):
        call_with_failover(
            _factory(
                {_S0: [httpx.ReadTimeout("a")], _S1: [httpx.ReadTimeout("b")]},
                calls,
            ),
            tier_config=cfg,
            level=2,
        )
    assert calls == [_S0, _S1]


def test_three_slot_ratio_guard_compares_against_starting_slot():
    """The ratio is measured against the slot the request started on, so a
    cascade cannot ratchet the price up one tolerable hop at a time."""
    cfg = _chain_cfg(
        FailoverConfig(failure_threshold=3, cost_ratio_threshold=2.0),
        s0=_priced(_S0, price_completion=1.0, max_tokens=1000),
        s1=_priced(_S1, price_completion=2.0, max_tokens=1000),
        s2=_priced(_S2, price_completion=3.0, max_tokens=1000),
    )
    calls: list[str] = []
    with pytest.raises(FailoverCostGuardError):
        call_with_failover(
            _factory(
                {_S0: [httpx.ReadTimeout("a")], _S1: [httpx.ReadTimeout("b")]},
                calls,
            ),
            tier_config=cfg,
            level=2,
        )
    assert calls == [_S0, _S1]


def test_three_slot_failover_disabled_tries_only_active_slot():
    calls: list[str] = []
    with pytest.raises(httpx.ReadTimeout):
        call_with_failover(
            _factory({_S0: [httpx.ReadTimeout("a")]}, calls),
            tier_config=_chain_cfg(),
            level=2,
            failover_enabled=False,
        )
    assert calls == [_S0]


def test_three_slot_async_cascade_reaches_slot_three():
    calls: list[str] = []
    result = asyncio.run(
        acall_with_failover(
            _afactory(
                {
                    _S0: [httpx.ReadTimeout("down")],
                    _S1: [_Exhausted("out of credits")],
                    _S2: ["rescued"],
                },
                calls,
            ),
            tier_config=_chain_cfg(),
            level=2,
        )
    )
    assert result == "rescued"
    assert calls == [_S0, _S1, _S2]
    assert get_failover_tracker().active_slot() == 2


def test_three_slot_async_exhaustion_raises_last_slot_error():
    calls: list[str] = []
    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(
            acall_with_failover(
                _afactory(
                    {
                        _S0: [httpx.ReadTimeout("a")],
                        _S1: [httpx.ConnectError("b")],
                        _S2: [httpx.RemoteProtocolError("c")],
                    },
                    calls,
                ),
                tier_config=_chain_cfg(),
                level=2,
            )
        )
    assert calls == [_S0, _S1, _S2]


def test_three_slot_threshold_arms_each_slot_in_turn():
    """With a threshold of 3, slot 0 arms only after three consecutive
    failures; slot 1 then needs its own three before calls skip to slot 2."""
    cfg = _chain_cfg(FailoverConfig(failure_threshold=3, window_seconds=900.0))
    tracker = get_failover_tracker()
    calls: list[str] = []
    for expected_before in ("default", "default", "default"):
        assert tracker.active_slot(chain_length=3) == expected_before
        call_with_failover(
            _factory({_S0: [httpx.ReadTimeout("flaky")], _S1: ["ok"]}, calls),
            tier_config=cfg,
            level=2,
        )
    assert tracker.active_slot(chain_length=3) == "fallback"
    for _ in range(3):
        assert tracker.active_slot(chain_length=3) == "fallback"
        call_with_failover(
            _factory({_S1: [httpx.ReadTimeout("flaky")], _S2: ["ok"]}, calls),
            tier_config=cfg,
            level=2,
        )
    assert tracker.active_slot(chain_length=3) == 2
    calls.clear()
    assert (
        call_with_failover(_factory({_S2: ["direct"]}, calls), tier_config=cfg, level=2)
        == "direct"
    )
    assert calls == [_S2]


# --------------------------------------------------------------------------- #
#  Baked chain — anthropic → claudeSDK → openrouter                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("level", [1, 2, 3])
def test_baked_chain_cascades_anthropic_then_sdk_then_openrouter(level: int) -> None:
    cfg = TierConfig()
    models = [cfg.slot(i).for_level(level).model for i in range(3)]
    assert [m.split("-", 1)[0] for m in models] == [
        "anthropic",
        "claudeSDK",
        "openrouter",
    ]
    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                models[0]: [httpx.ReadTimeout("anthropic down")],
                models[1]: [_Exhausted("seat exhausted")],
                models[2]: ["rescued"],
            },
            calls,
        ),
        tier_config=cfg,
        level=level,
    )
    assert result == "rescued"
    # Same level on every slot, tried in chain order.
    assert calls == models


@pytest.mark.parametrize("level", [1, 2, 3])
def test_baked_chain_exhaustion_raises_openrouter_error(level: int) -> None:
    cfg = TierConfig()
    models = [cfg.slot(i).for_level(level).model for i in range(3)]
    calls: list[str] = []
    with pytest.raises(httpx.RemoteProtocolError):
        call_with_failover(
            _factory(
                {
                    models[0]: [httpx.ReadTimeout("a")],
                    models[1]: [httpx.ConnectError("b")],
                    models[2]: [httpx.RemoteProtocolError("c")],
                },
                calls,
            ),
            tier_config=cfg,
            level=level,
        )
    assert calls == models


# --------------------------------------------------------------------------- #
#  Legacy two-slot persisted config                                            #
# --------------------------------------------------------------------------- #


def test_legacy_two_slot_loaded_config_fails_over_default_to_fallback():
    from robotsix_llmio.config.loader import load_tier_config

    cfg = load_tier_config(
        {
            "default": {"level2": {"model": "claudeSDK-opus"}},
            "fallback": {"level2": {"model": "openrouter-deepseek/legacy"}},
        }
    )
    assert len(cfg.chain) == 2
    assert _max_slot_switches(len(cfg.chain)) == 1
    calls: list[str] = []
    result = call_with_failover(
        _factory(
            {
                "claudeSDK-opus": [httpx.ReadTimeout("down")],
                "openrouter-deepseek/legacy": ["rescued"],
            },
            calls,
        ),
        tier_config=cfg,
        level=2,
    )
    assert result == "rescued"
    assert calls == ["claudeSDK-opus", "openrouter-deepseek/legacy"]

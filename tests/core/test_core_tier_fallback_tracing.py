"""OTel span attribute tests for tier fallback.

Extracted from ``test_core_tier_fallback_sync.py``
(#20260801T130625Z-split-tests-core-test-core-tier-fallback-626a).
"""

from __future__ import annotations

import contextlib

from conftest import STD_TIER_CONFIG, _noop_sleep, tf_factory_that_succeeds

from robotsix_llmio.core import tier_fallback as tier_fallback_mod
from robotsix_llmio.core.tier_fallback import call_with_tier_fallback

# --------------------------------------------------------------------------- #
#  OTel span attribute tests                                                  #
# --------------------------------------------------------------------------- #


def test_tier_fallback_span_attributes_on_success(monkeypatch):
    """Verify llmio.tier.* attributes are set on the per-attempt child span
    when the call succeeds on the first try."""

    class _Span:
        def __init__(self):
            self.attrs: dict[str, object] = {}

        def set_attribute(self, key, value):
            self.attrs[key] = value

    captured: list[_Span] = []

    class _MockTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, name):
            span = _Span()
            captured.append(span)
            yield span

    monkeypatch.setattr(tier_fallback_mod, "get_tracer", lambda _name: _MockTracer())
    monkeypatch.setattr(tier_fallback_mod, "get_recording_span", lambda: None)

    call_with_tier_fallback(
        tf_factory_that_succeeds("ok"),
        tier_config=STD_TIER_CONFIG,
        what="span-test",
        sleep=_noop_sleep,
    )

    assert len(captured) == 1
    span = captured[0]
    assert span.attrs["llmio.tier.level"] == "level1"
    assert span.attrs["llmio.tier.provider"] == "claudeSDK"
    assert span.attrs["llmio.tier.model"] == "opus"
    assert span.attrs["llmio.tier.attempt_index"] == 1
    assert span.attrs["llmio.tier.succeeded"] is True


def test_tier_fallback_span_attributes_on_escalation(monkeypatch):
    """Verify per-attempt span attributes record failures and parent span
    records promotion counters when tier escalation occurs."""

    class _Span:
        def __init__(self):
            self.attrs: dict[str, object] = {}

        def set_attribute(self, key, value):
            self.attrs[key] = value

    captured: list[_Span] = []
    parent_span = _Span()

    class _MockTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, name):
            span = _Span()
            captured.append(span)
            yield span

    monkeypatch.setattr(tier_fallback_mod, "get_tracer", lambda _name: _MockTracer())
    monkeypatch.setattr(tier_fallback_mod, "get_recording_span", lambda: parent_span)

    counter = {"remaining": 1}

    def factory(tlc):
        def fn():
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise RuntimeError("test-fail")
            return "ok"

        return fn

    out = call_with_tier_fallback(
        factory,
        tier_config=STD_TIER_CONFIG,
        fallback_enabled=True,
        max_fallback_depth=2,
        what="span-escalation",
        sleep=_noop_sleep,
    )
    assert out == "ok"

    # Two attempts: first fails, second succeeds
    assert len(captured) == 2

    # First attempt span — failed
    s1 = captured[0]
    assert s1.attrs["llmio.tier.level"] == "level1"
    assert s1.attrs["llmio.tier.provider"] == "claudeSDK"
    assert s1.attrs["llmio.tier.model"] == "opus"
    assert s1.attrs["llmio.tier.attempt_index"] == 1
    assert s1.attrs["llmio.tier.succeeded"] is False

    # Second attempt span — succeeded
    s2 = captured[1]
    assert s2.attrs["llmio.tier.level"] == "level2"
    assert s2.attrs["llmio.tier.provider"] == "claudeSDK"
    assert s2.attrs["llmio.tier.model"] == "haiku"
    assert s2.attrs["llmio.tier.attempt_index"] == 2
    assert s2.attrs["llmio.tier.succeeded"] is True

    # Parent span records promotion
    assert parent_span.attrs["llmio.tier.promotions"] == 1
    assert parent_span.attrs["llmio.tier.fallback_activated"] is True


def test_tier_fallback_span_noop_without_otel(monkeypatch):
    """start_span yields None when get_tracer returns None; loop still works."""

    monkeypatch.setattr(tier_fallback_mod, "get_tracer", lambda _name: None)
    monkeypatch.setattr(tier_fallback_mod, "get_recording_span", lambda: None)

    out = call_with_tier_fallback(
        tf_factory_that_succeeds("ok"),
        tier_config=STD_TIER_CONFIG,
        sleep=_noop_sleep,
    )
    assert out == "ok"

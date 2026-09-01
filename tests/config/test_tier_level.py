"""Tests for the ``TierLevel`` enum — values, str behaviour, and membership."""

from __future__ import annotations

from robotsix_llmio.config.tier import TierLevel


def test_tier_level_values():
    """Each member carries the expected string value."""
    assert TierLevel.LEVEL1.value == "level1"
    assert TierLevel.LEVEL2.value == "level2"
    assert TierLevel.LEVEL3.value == "level3"


def test_tier_level_is_str_enum():
    """Members are instances of both ``str`` and ``StrEnum``."""
    for member in TierLevel:
        assert isinstance(member, str)
        assert isinstance(member, TierLevel)


def test_tier_level_members():
    """Only the three members exist — no extras (levels collapsed from five
    on 2026-09-01; equivalent-capability models live in the fallback slot)."""
    assert {m.name for m in TierLevel} == {"LEVEL1", "LEVEL2", "LEVEL3"}
    assert len({m.value for m in TierLevel}) == 3

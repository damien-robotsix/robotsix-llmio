"""Tests for the ``TierLevel`` enum — values, str behaviour, and membership."""

from __future__ import annotations

from robotsix_llmio.config.tier import TierLevel


def test_tier_level_values():
    """Each member carries the expected string value."""
    assert TierLevel.LEVEL1.value == "level1"
    assert TierLevel.LEVEL2.value == "level2"
    assert TierLevel.LEVEL3.value == "level3"
    assert TierLevel.LEVEL4.value == "level4"


def test_tier_level_is_str_enum():
    """Members are instances of both ``str`` and ``StrEnum``."""
    for member in TierLevel:
        assert isinstance(member, str)
        assert isinstance(member, TierLevel)


def test_tier_level_str_comparison():
    """Members compare equal to their string values."""
    assert TierLevel.LEVEL1.value == "level1"
    assert TierLevel.LEVEL2.value == "level2"
    assert TierLevel.LEVEL3.value == "level3"
    assert TierLevel.LEVEL4.value == "level4"


def test_tier_level_distinct_members():
    """Different members have different names and values."""
    members = list(TierLevel)
    assert len(members) == 5
    # All values are distinct.
    values = {m.value for m in members}
    assert len(values) == 5


def test_tier_level_members():
    """Only the five members exist — no extras."""
    assert {m.name for m in TierLevel} == {
        "LEVEL1",
        "LEVEL2",
        "LEVEL3",
        "LEVEL4",
        "LEVEL5",
    }
    assert len(list(TierLevel)) == 5

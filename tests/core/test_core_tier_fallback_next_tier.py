"""_next_unvisited_tier helper + _ALL_TIER_LEVELS guard tests.

Follows the existing ``test_core_retry.py`` conventions: no mocks,
standalone ``def test_*`` functions.
"""

from __future__ import annotations

import pytest

from robotsix_llmio.config.tier import TierLevel
from robotsix_llmio.core.tier_fallback import _ALL_TIER_LEVELS, _next_unvisited_tier

# --------------------------------------------------------------------------- #
#  _next_unvisited_tier                                                       #
# --------------------------------------------------------------------------- #


def test_next_unvisited_from_level1_no_visited():
    assert _next_unvisited_tier(TierLevel.LEVEL1, frozenset()) == TierLevel.LEVEL2


def test_next_unvisited_from_level1_level2_visited():
    assert (
        _next_unvisited_tier(TierLevel.LEVEL1, frozenset({TierLevel.LEVEL2}))
        == TierLevel.LEVEL3
    )


def test_next_unvisited_from_level1_all_higher_visited_returns_lower_none():
    # No lower tiers for LEVEL1, and all higher are visited → None
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL1,
            frozenset(
                {TierLevel.LEVEL2, TierLevel.LEVEL3, TierLevel.LEVEL4, TierLevel.LEVEL5}
            ),
        )
        is None
    )


def test_next_unvisited_from_level2_prefers_higher():
    # LEVEL2 → LEVEL3 first (nearest higher), then LEVEL4, then LEVEL1 (lower)
    assert _next_unvisited_tier(TierLevel.LEVEL2, frozenset()) == TierLevel.LEVEL3


def test_next_unvisited_from_level2_level3_visited_returns_level4():
    assert (
        _next_unvisited_tier(TierLevel.LEVEL2, frozenset({TierLevel.LEVEL3}))
        == TierLevel.LEVEL4
    )


def test_next_unvisited_from_level2_higher_visited_returns_lower():
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL2,
            frozenset({TierLevel.LEVEL3, TierLevel.LEVEL4, TierLevel.LEVEL5}),
        )
        == TierLevel.LEVEL1
    )


def test_next_unvisited_from_level2_all_others_visited_returns_none():
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL2,
            frozenset(
                {TierLevel.LEVEL1, TierLevel.LEVEL3, TierLevel.LEVEL4, TierLevel.LEVEL5}
            ),
        )
        is None
    )


def test_next_unvisited_from_level3_prefers_higher_level4():
    # LEVEL3 → LEVEL4 first (higher), then LEVEL2, then LEVEL1
    assert _next_unvisited_tier(TierLevel.LEVEL3, frozenset()) == TierLevel.LEVEL4


def test_next_unvisited_from_level3_level4_visited_returns_level2():
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL3, frozenset({TierLevel.LEVEL4, TierLevel.LEVEL5})
        )
        == TierLevel.LEVEL2
    )


def test_next_unvisited_from_level3_higher_and_level2_visited_returns_level1():
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL3,
            frozenset({TierLevel.LEVEL2, TierLevel.LEVEL4, TierLevel.LEVEL5}),
        )
        == TierLevel.LEVEL1
    )


def test_next_unvisited_from_level3_all_visited_returns_none():
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL3,
            frozenset(
                {TierLevel.LEVEL1, TierLevel.LEVEL2, TierLevel.LEVEL4, TierLevel.LEVEL5}
            ),
        )
        is None
    )


def test_next_unvisited_from_level4_prefers_higher_level5():
    # LEVEL4 → LEVEL5 first (higher), then LEVEL3, LEVEL2, LEVEL1
    assert _next_unvisited_tier(TierLevel.LEVEL4, frozenset()) == TierLevel.LEVEL5


def test_next_unvisited_from_level4_level5_visited_returns_level3():
    assert (
        _next_unvisited_tier(TierLevel.LEVEL4, frozenset({TierLevel.LEVEL5}))
        == TierLevel.LEVEL3
    )


def test_next_unvisited_from_level5_only_lower():
    # LEVEL5 → LEVEL4 (nearest lower), then LEVEL3, LEVEL2, LEVEL1
    assert _next_unvisited_tier(TierLevel.LEVEL5, frozenset()) == TierLevel.LEVEL4


def test_next_unvisited_from_level4_all_visited_returns_none():
    assert (
        _next_unvisited_tier(
            TierLevel.LEVEL4,
            frozenset(
                {TierLevel.LEVEL1, TierLevel.LEVEL2, TierLevel.LEVEL3, TierLevel.LEVEL5}
            ),
        )
        is None
    )


def test_next_unvisited_unknown_level_returns_none():
    # Defensive: if somehow a level not in _ALL_TIER_LEVELS is passed
    class UnknownLevel:
        value = "unknown"

    assert _next_unvisited_tier(UnknownLevel(), frozenset()) is None


# --------------------------------------------------------------------------- #
#  _ALL_TIER_LEVELS sync guard                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("member", tuple(TierLevel))
def test_all_tier_levels_enum_members_in_tuple(member: TierLevel) -> None:
    """Every TierLevel enum member is present in _ALL_TIER_LEVELS."""
    assert member in _ALL_TIER_LEVELS


@pytest.mark.parametrize("entry", _ALL_TIER_LEVELS)
def test_no_stale_entries_in_all_tier_levels_tuple(entry: TierLevel) -> None:
    """Every entry in _ALL_TIER_LEVELS is a valid TierLevel member."""
    assert entry in tuple(TierLevel)

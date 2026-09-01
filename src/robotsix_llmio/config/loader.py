"""Configuration loader — merges an explicit dict over the baked defaults
into a validated :class:`TierConfig`.

Usage::

    from robotsix_llmio.config import load_tier_config

    cfg = load_tier_config({"default": {"level2": {"model": "claudeSDK-sonnet"}}})

The dict shape mirrors :class:`~robotsix_llmio.config.tier.TierConfig`::

    {
      "default":  {"level1": {...}, "level2": {...}, "level3": {...}},
      "fallback": {"level1": {...}, "level2": {...}, "level3": {...}},
      "failover": {"failure_threshold": 3, "window_seconds": 900},
      "vision": {"model": "openrouter-<vision-model>"}
    }

Every key is optional; each per-level dict is merged field-by-field over the
corresponding baked default, so overriding one field (say ``model``) keeps
the default's ``provider_kwargs`` and ``max_tokens``. Unknown keys fail
validation loudly (``extra="forbid"`` on the schema) — there is no
environment-variable overlay.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from robotsix_llmio.config.tier import TierConfig, TierLevel
from robotsix_llmio.exceptions import RobotsixLLMIOError

# --------------------------------------------------------------------------- #
#  Exception
# --------------------------------------------------------------------------- #


class TierConfigLoadError(RobotsixLLMIOError):
    """Raised when tier configuration cannot be loaded.

    This wraps underlying pydantic :class:`~pydantic.ValidationError`\\(s) as
    ``__cause__`` so the original details are always available.
    """


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

_SLOT_KEYS = ("default", "fallback")


def load_tier_config(config_dict: dict[str, Any] | None = None) -> TierConfig:
    """Load a validated :class:`TierConfig`, merging *config_dict* over the
    baked defaults.

    Args:
        config_dict: Optional dict in the :class:`TierConfig` shape (see the
            module docstring). Per-level dicts merge field-by-field over the
            baked default for that slot+level; the ``failover`` dict merges
            field-by-field over the default policy. ``None`` (or ``{}``)
            yields the fully baked configuration.

    Returns:
        TierConfig: A fully-validated two-slot configuration.

    Raises:
        TierConfigLoadError: If the merged configuration fails validation —
            an unknown provider prefix in a supplied ``model``, an unknown
            key anywhere in the shape, or a slot/level value that is not a
            dict or pydantic model.

    """
    if not config_dict:
        return TierConfig()

    unknown = set(config_dict) - {*_SLOT_KEYS, "failover", "vision"}
    if unknown:
        raise TierConfigLoadError(
            f"Unknown top-level key(s) {sorted(unknown)!r} in tier config. "
            f"Expected 'default', 'fallback', 'failover', and/or 'vision' "
            f"(the flat level1..level5 shape was removed in the "
            f"provider-failover rework)."
        )

    baked = TierConfig()
    merged: dict[str, Any] = baked.model_dump()

    try:
        for slot in _SLOT_KEYS:
            if slot not in config_dict:
                continue
            slot_val = _to_dict(config_dict[slot])
            for tier in (m.value for m in TierLevel):
                if tier in slot_val:
                    merged[slot][tier].update(_to_dict(slot_val[tier]))
        if "failover" in config_dict:
            merged["failover"].update(_to_dict(config_dict["failover"]))
        if "vision" in config_dict:
            merged["vision"].update(_to_dict(config_dict["vision"]))
        return TierConfig.model_validate(merged)
    except (ValidationError, RobotsixLLMIOError) as exc:
        raise TierConfigLoadError(str(exc)) from exc


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert a config value to a plain dict for merging.

    Handles plain dicts and pydantic models (anything with ``model_dump()``).
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    raise TierConfigLoadError(
        f"Cannot merge config value of type {type(obj).__name__!r}; "
        f"expected a dict or pydantic model."
    )

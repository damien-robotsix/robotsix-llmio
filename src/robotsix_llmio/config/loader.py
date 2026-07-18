"""Configuration loader — merges environment variables, explicit dicts,
and baked defaults into a validated :class:`TierConfig`.

Usage::

    from robotsix_llmio.config import load_tier_config

    cfg = load_tier_config({"level1": {"model": "claudeSDK-opus"}})
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import ValidationError

from robotsix_llmio.config.tier import (
    LEVEL1_DEFAULT,
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    LEVEL4_DEFAULT,
    TierConfig,
    TierLevel,
)
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
#  Baked base dicts (imported constants, not redefined here)
# --------------------------------------------------------------------------- #

_BAKED_BASE: dict[str, dict[str, Any]] = {
    "level1": LEVEL1_DEFAULT.model_dump(),
    "level2": LEVEL2_DEFAULT.model_dump(),
    "level3": LEVEL3_DEFAULT.model_dump(),
    "level4": LEVEL4_DEFAULT.model_dump(),
}
"""Dict forms of the module-level baked defaults for every tier.

All four tiers have a baked default, so partial env/dict overrides merge
field-by-field over the corresponding default (and an omitted tier resolves
to its default outright).
"""

# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


def load_tier_config(
    config_dict: dict[str, Any] | None = None,
    *,
    env_prefix: str = "LLMIO_",
) -> TierConfig:
    """Load a validated :class:`TierConfig` by merging three sources.

    1. **Baked defaults** — the ``TierConfig`` model supplies a default for
       every tier (``level1``, ``level2``, ``level3``, ``level4``), so any
       omitted tier falls back to its baked default.
    2. **Environment variables** — every recognised variable under
       *env_prefix* (see table below).
    3. **Explicit dict** (*config_dict*) — highest precedence; merged per-tier
       so the caller can override individual fields.

    Args:
        config_dict: Optional dictionary whose keys are tier names
            (``"level1"``, ``"level2"``, ``"level3"``, ``"level4"``) and
            values are dicts of ``TierLevelConfig`` fields.  When
            ``None``, only environment variables and baked defaults are
            used.
        env_prefix: Prefix for environment variable names.  Defaults to
            ``"LLMIO_"``.

    Returns:
        TierConfig: A fully-validated four-tier configuration.

    Raises:
        TierConfigLoadError: If a ``*_PROVIDER_KWARGS`` environment
            variable contains invalid JSON, or if the merged
            configuration fails pydantic validation (e.g. an unknown
            provider prefix in a supplied ``model``).

    """
    # ---- 1.  Read environment variables -----------------------------------
    env_nested = _read_env_vars(env_prefix)

    # ---- 2.  Merge env + explicit dict ------------------------------------
    merged: dict[str, Any] = {}
    for tier in (m.value for m in TierLevel):
        tier_dict: dict[str, Any] = {}

        # Start with baked defaults for tiers that have them.
        if tier in _BAKED_BASE:
            tier_dict.update(_BAKED_BASE[tier])

        # Overlay environment-derived fields (only keys actually set via env).
        if tier in env_nested:
            tier_dict.update(env_nested[tier])

        # Overlay explicit-dict fields (highest precedence).
        if config_dict is not None and tier in config_dict:
            cfg_tier = config_dict[tier]
            if isinstance(cfg_tier, dict):
                tier_dict.update(cfg_tier)
            else:
                # If the caller passed a TierLevelConfig object or similar,
                # convert to dict so we can merge field-by-field.
                tier_dict.update(_to_dict(cfg_tier))

        # Only include the tier if we have *something* for it (otherwise
        # pydantic applies the per-tier ``default_factory``).
        if tier_dict:
            merged[tier] = tier_dict

    # ---- 3.  Validate -----------------------------------------------------
    try:
        return TierConfig.model_validate(merged)
    except (ValidationError, RobotsixLLMIOError) as exc:
        raise TierConfigLoadError(str(exc)) from exc


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert a tier-value to a plain dict for merging.

    Handles :class:`TierLevelConfig` instances and anything with a
    ``model_dump()`` or ``dict()`` method.
    """
    if isinstance(obj, dict):
        return obj
    try:
        result: Any = obj.model_dump()
    except AttributeError:
        pass
    else:
        if isinstance(result, dict):
            return result
    try:
        result = obj.dict()
    except AttributeError:
        pass
    else:
        if isinstance(result, dict):
            return result
    raise TierConfigLoadError(
        f"Cannot merge tier value of type {type(obj).__name__!r}; "
        f"expected a dict or pydantic model."
    )


def _read_env_vars(env_prefix: str) -> dict[str, dict[str, Any]]:
    """Read recognised environment variables into a nested tier→field dict.

    Returns only the keys that were actually set in the environment.
    Each tier dict contains the combined ``model`` identifier (used verbatim)
    and/or ``provider_kwargs`` (a JSON object).
    """
    nested: dict[str, dict[str, Any]] = {}

    def _set(tier: str, field: str, value: Any) -> None:
        nested.setdefault(tier, {})[field] = value

    for tier_upper, tier_lower in ((m.value.upper(), m.value) for m in TierLevel):
        # MODEL — full combined provider-model identifier, used verbatim.
        var_name = f"{env_prefix}{tier_upper}_MODEL"
        raw = os.environ.get(var_name)
        if raw is not None:
            _set(tier_lower, "model", raw)

        # PROVIDER_KWARGS — JSON object forwarded to the provider constructor.
        var_name = f"{env_prefix}{tier_upper}_PROVIDER_KWARGS"
        raw = os.environ.get(var_name)
        if raw is not None:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TierConfigLoadError(f"Invalid JSON in {var_name}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise TierConfigLoadError(
                    f"{var_name} must be a JSON object, got {type(parsed).__name__}"
                )
            _set(tier_lower, "provider_kwargs", parsed)

    return nested

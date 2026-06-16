"""Configuration loader — merges environment variables, explicit dicts,
and baked defaults into a validated :class:`TierConfig`.

Usage::

    from robotsix_llmio.config import load_tier_config

    cfg = load_tier_config({"level1": {"provider": "...", "model": "..."}})
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

from pydantic import ValidationError

from robotsix_llmio.config.tier import (
    LEVEL2_DEFAULT,
    LEVEL3_DEFAULT,
    TierConfig,
)

# --------------------------------------------------------------------------- #
#  Exception
# --------------------------------------------------------------------------- #


class TierConfigLoadError(Exception):
    """Raised when tier configuration cannot be loaded.

    This wraps underlying pydantic :class:`~pydantic.ValidationError`\\(s) as
    ``__cause__`` so the original details are always available.
    """


# --------------------------------------------------------------------------- #
#  Baked base dicts (imported constants, not redefined here)
# --------------------------------------------------------------------------- #

_BAKED_BASE: dict[str, dict[str, Any]] = {
    "level2": LEVEL2_DEFAULT.model_dump(),
    "level3": LEVEL3_DEFAULT.model_dump(),
}
"""Dict forms of the module-level baked defaults for levels that have them.

``level1`` is intentionally absent — it has no default and must be supplied
by the caller or environment.
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

    1. **Baked defaults** — the ``TierConfig`` model supplies ``level2`` and
       ``level3`` defaults; ``level1`` is *required* and has no default.
    2. **Environment variables** — every recognised variable under
       *env_prefix* (see table below).
    3. **Explicit dict** (*config_dict*) — highest precedence; merged per-tier
       so the caller can override individual fields.

    Parameters
    ----------
    config_dict:
        Optional dictionary whose keys are tier names (``"level1"``,
        ``"level2"``, ``"level3"``) and values are dicts of
        ``TierLevelConfig`` fields.  When ``None``, only environment
        variables and baked defaults are used.
    env_prefix:
        Prefix for environment variable names.  Defaults to ``"LLMIO_"``.

    Returns
    -------
    TierConfig
        A fully-validated three-tier configuration.

    Raises
    ------
    TierConfigLoadError
        If a ``*_PROVIDER_KWARGS`` environment variable contains invalid
        JSON, or if the merged configuration fails pydantic validation
        (e.g. because ``level1`` was not supplied).
    """
    # ---- 1.  Read environment variables -----------------------------------
    env_nested = _read_env_vars(env_prefix)

    # ---- 2.  Merge env + explicit dict ------------------------------------
    merged: dict[str, Any] = {}
    for tier in ("level1", "level2", "level3"):
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
        # pydantic applies the ``default_factory`` for level2/level3, or
        # raises ValidationError for the required level1).
        if tier_dict:
            merged[tier] = tier_dict

    # ---- 3.  Validate -----------------------------------------------------
    try:
        return TierConfig.model_validate(merged)
    except ValidationError as exc:
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
    if hasattr(obj, "model_dump"):
        result: Any = obj.model_dump()
        if isinstance(result, dict):
            return result
    if hasattr(obj, "dict"):
        result = obj.dict()
        if isinstance(result, dict):
            return result
    raise TierConfigLoadError(
        f"Cannot merge tier value of type {type(obj).__name__!r}; "
        f"expected a dict or pydantic model."
    )


def _read_env_vars(env_prefix: str) -> dict[str, dict[str, Any]]:
    """Read recognised environment variables into a nested tier→field dict.

    Returns only the keys that were actually set in the environment.
    """
    nested: dict[str, dict[str, Any]] = {}

    def _set(tier: str, field: str, value: Any) -> None:
        nested.setdefault(tier, {})[field] = value

    # -- new-style variables ------------------------------------------------
    for tier_upper, tier_lower in [
        ("LEVEL1", "level1"),
        ("LEVEL2", "level2"),
        ("LEVEL3", "level3"),
    ]:
        for field_upper, field_lower in [
            ("PROVIDER", "provider"),
            ("MODEL", "model"),
            ("PROVIDER_KWARGS", "provider_kwargs"),
        ]:
            var_name = f"{env_prefix}{tier_upper}_{field_upper}"
            raw = os.environ.get(var_name)
            if raw is None:
                continue
            if field_lower == "provider_kwargs":
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise TierConfigLoadError(
                        f"Invalid JSON in {var_name}: {exc}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise TierConfigLoadError(
                        f"{var_name} must be a JSON object, "
                        f"got {type(parsed).__name__}"
                    )
                _set(tier_lower, field_lower, parsed)
            else:
                _set(tier_lower, field_lower, raw)

    # -- legacy variables ---------------------------------------------------
    # Each legacy variable emits at most one FutureWarning per call and is
    # only consulted when the corresponding new-style variable is *unset*.

    # LLMIO_FLASH_MODEL → level1.model
    if "model" not in nested.get("level1", {}):
        legacy = os.environ.get("LLMIO_FLASH_MODEL")
        if legacy is not None:
            warnings.warn(
                "LLMIO_FLASH_MODEL is deprecated; use LLMIO_LEVEL1_MODEL instead",
                FutureWarning,
                stacklevel=3,  # caller → load_tier_config → _read_env_vars
            )
            _set("level1", "model", legacy)

    # LLMIO_FLASH_PROVIDER → level1.provider
    if "provider" not in nested.get("level1", {}):
        legacy = os.environ.get("LLMIO_FLASH_PROVIDER")
        if legacy is not None:
            warnings.warn(
                "LLMIO_FLASH_PROVIDER is deprecated; use LLMIO_LEVEL1_PROVIDER instead",
                FutureWarning,
                stacklevel=3,
            )
            _set("level1", "provider", legacy)

    # LLMIO_NORMAL_MODEL → level2.model
    if "model" not in nested.get("level2", {}):
        legacy = os.environ.get("LLMIO_NORMAL_MODEL")
        if legacy is not None:
            warnings.warn(
                "LLMIO_NORMAL_MODEL is deprecated; use LLMIO_LEVEL2_MODEL instead",
                FutureWarning,
                stacklevel=3,
            )
            _set("level2", "model", legacy)

    # LLMIO_NORMAL_PROVIDER → level2.provider
    if "provider" not in nested.get("level2", {}):
        legacy = os.environ.get("LLMIO_NORMAL_PROVIDER")
        if legacy is not None:
            warnings.warn(
                "LLMIO_NORMAL_PROVIDER is deprecated; "
                "use LLMIO_LEVEL2_PROVIDER instead",
                FutureWarning,
                stacklevel=3,
            )
            _set("level2", "provider", legacy)

    # LLMIO_PROVIDER → any level whose provider is still unset
    legacy_provider = os.environ.get("LLMIO_PROVIDER")
    if legacy_provider is not None:
        applied = False
        for tier in ("level1", "level2", "level3"):
            if "provider" not in nested.get(tier, {}):
                _set(tier, "provider", legacy_provider)
                applied = True
        if applied:
            warnings.warn(
                "LLMIO_PROVIDER is deprecated; set "
                "LLMIO_LEVEL{1,2,3}_PROVIDER per tier",
                FutureWarning,
                stacklevel=3,
            )

    return nested

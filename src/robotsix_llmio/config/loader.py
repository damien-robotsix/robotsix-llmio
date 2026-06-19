"""Configuration loader — merges environment variables, explicit dicts,
and baked defaults into a validated :class:`TierConfig`.

Usage::

    from robotsix_llmio.config import load_tier_config

    cfg = load_tier_config({"level1": {"model": "claudeSDK-opus"}})
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
from robotsix_llmio.config.transport import provider_to_transport
from robotsix_llmio.exceptions import RobotsixLLMIOError

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

# Pre-compute the provider-prefix (including sub-alias bracket) from each
# baked default so legacy model-only env vars can be combined with them.
_BAKED_PREFIX: dict[str, str] = {}
for _tier_name, _default in [("level2", LEVEL2_DEFAULT), ("level3", LEVEL3_DEFAULT)]:
    from robotsix_llmio.core.identifier import parse_model_identifier as _pmi

    _parsed = _pmi(_default.model)
    if _parsed.sub_alias:
        _BAKED_PREFIX[_tier_name] = f"{_parsed.provider}[{_parsed.sub_alias}]"
    else:
        _BAKED_PREFIX[_tier_name] = _parsed.provider

# --------------------------------------------------------------------------- #
#  Legacy transport → provider-prefix mapping (for combining env vars)
# --------------------------------------------------------------------------- #

_TRANSPORT_TO_PREFIX: dict[str, str] = {
    "claude-sdk": "claudeSDK",
}
"""Old transport alias → new hyphen-free provider prefix.

``"openrouter[deepseek]"`` is already in the canonical form and needs no
entry here.
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
                tier_dict.update(_normalise_old_shape(cfg_tier))
            else:
                # If the caller passed a TierLevelConfig object or similar,
                # convert to dict so we can merge field-by-field.
                tier_dict.update(_normalise_old_shape(_to_dict(cfg_tier)))

        # Normalise the merged dict — handle any transport/model/provider
        # keys that survived the merge (e.g. when an explicit dict overrides
        # with old-shape keys).
        tier_dict = _normalise_old_shape(tier_dict)

        # Only include the tier if we have *something* for it (otherwise
        # pydantic applies the ``default_factory`` for level2/level3, or
        # raises ValidationError for the required level1).
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


def _combine_transport_model(transport: str, model: str) -> str:
    """Combine a legacy transport alias and model name into a combined
    ``provider-model`` identifier."""
    prefix = _TRANSPORT_TO_PREFIX.get(transport, transport)
    return f"{prefix}-{model}"


def _normalise_old_shape(tier_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalise legacy tier dict shapes into the new combined-identifier form.

    Handles these legacy shapes:

    * ``{"provider": ..., "model": ...}`` — old registry-name shape.
    * ``{"transport": ..., "model": ...}`` — the two-field shape this
      child replaces.

    When both transport and model are present and *model* is already a
    combined identifier, the transport's prefix replaces the existing
    prefix (the transport wins).

    A dict already carrying a combined ``model`` identifier (with no
    transport/provider keys) is returned unchanged.
    """
    result = dict(tier_dict)

    # Already in new format — just ``model`` (combined identifier) and
    # no transport/provider keys.
    if "transport" not in result and "provider" not in result:
        return result

    transport: str | None = None
    model: str | None = None

    # Transport takes precedence over provider when both present.
    if "transport" in result:
        transport = result.pop("transport")
        # Also drop any stray provider so it doesn't leak through.
        result.pop("provider", None)
    elif "provider" in result:
        provider_val = result.pop("provider")
        transport = provider_to_transport(provider_val)

    if "model" in result:
        model = result.pop("model")

    if transport is not None and model is not None:
        # If model is already a combined identifier with a *known* prefix,
        # extract just the model-name portion and re-combine with the
        # transport prefix (transport wins).
        try:
            from robotsix_llmio.core.factory import _PROVIDER_PREFIX_MAP
            from robotsix_llmio.core.identifier import parse_model_identifier

            parsed = parse_model_identifier(model)
            if parsed.provider in _PROVIDER_PREFIX_MAP:
                model = parsed.model_name
        except Exception:
            pass
        result["model"] = _combine_transport_model(transport, model)
    elif model is not None:
        # Only model, no transport — keep as-is.
        result["model"] = model
    elif transport is not None:
        # Only transport — no model; leave the dict without a model field
        # so validation catches the missing field.
        pass

    return result


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
    Transport and model are combined into a single ``model`` field holding
    the combined ``provider-model`` identifier.
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
            ("TRANSPORT", "transport"),
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
                        f"{var_name} must be a JSON object, got {type(parsed).__name__}"
                    )
                _set(tier_lower, field_lower, parsed)
            else:
                _set(tier_lower, field_lower, raw)

        # ``LLMIO_LEVEL{n}_PROVIDER`` is a backward-compatible alias for
        # ``_TRANSPORT``; the converted transport is only applied when the
        # new-style ``_TRANSPORT`` variable is unset for this level.
        if "transport" not in nested.get(tier_lower, {}):
            legacy_provider = os.environ.get(f"{env_prefix}{tier_upper}_PROVIDER")
            if legacy_provider is not None:
                _set(tier_lower, "transport", provider_to_transport(legacy_provider))

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

    # LLMIO_FLASH_PROVIDER → level1.transport
    if "transport" not in nested.get("level1", {}):
        legacy = os.environ.get("LLMIO_FLASH_PROVIDER")
        if legacy is not None:
            warnings.warn(
                "LLMIO_FLASH_PROVIDER is deprecated; use LLMIO_LEVEL1_PROVIDER instead",
                FutureWarning,
                stacklevel=3,
            )
            _set("level1", "transport", provider_to_transport(legacy))

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

    # LLMIO_NORMAL_PROVIDER → level2.transport
    if "transport" not in nested.get("level2", {}):
        legacy = os.environ.get("LLMIO_NORMAL_PROVIDER")
        if legacy is not None:
            warnings.warn(
                "LLMIO_NORMAL_PROVIDER is deprecated; "
                "use LLMIO_LEVEL2_PROVIDER instead",
                FutureWarning,
                stacklevel=3,
            )
            _set("level2", "transport", provider_to_transport(legacy))

    # LLMIO_PROVIDER → any level whose transport is still unset
    legacy_provider = os.environ.get("LLMIO_PROVIDER")
    if legacy_provider is not None:
        applied = False
        for tier in ("level1", "level2", "level3"):
            if "transport" not in nested.get(tier, {}):
                _set(tier, "transport", provider_to_transport(legacy_provider))
                applied = True
        if applied:
            warnings.warn(
                "LLMIO_PROVIDER is deprecated; set "
                "LLMIO_LEVEL{1,2,3}_PROVIDER per tier",
                FutureWarning,
                stacklevel=3,
            )

    # -- post-process: combine transport + model into single model field -----
    for tier_lower in ("level1", "level2", "level3"):
        tier_data = nested.get(tier_lower)
        if tier_data is None:
            continue
        transport = tier_data.pop("transport", None)
        model = tier_data.pop("model", None)
        if transport is not None and model is not None:
            tier_data["model"] = _combine_transport_model(transport, model)
        elif model is not None and transport is None:
            # Only model, no transport — detect if it's already a valid
            # combined identifier (parseable AND has a known prefix).
            # If not, combine with baked prefix.
            try:
                from robotsix_llmio.core.factory import _PROVIDER_PREFIX_MAP
                from robotsix_llmio.core.identifier import (
                    parse_model_identifier,
                )

                parsed = parse_model_identifier(model)
                if parsed.provider in _PROVIDER_PREFIX_MAP:
                    # Already a valid combined identifier with known prefix.
                    tier_data["model"] = model
                else:
                    # Parsable but unknown prefix — treat as bare model name.
                    prefix = _BAKED_PREFIX.get(tier_lower)
                    if prefix is not None:
                        tier_data["model"] = f"{prefix}-{model}"
                    else:
                        tier_data["model"] = model
            except Exception:
                # Not parseable — treat as bare model name.
                prefix = _BAKED_PREFIX.get(tier_lower)
                if prefix is not None:
                    tier_data["model"] = f"{prefix}-{model}"
                else:
                    tier_data["model"] = model
        elif transport is not None:
            # Only transport — leave it; merge with baked model.
            tier_data["transport"] = transport

    return nested

"""Configuration loader — merges an explicit dict over the baked defaults
into a validated :class:`TierConfig`.

Usage::

    from robotsix_llmio.config import load_tier_config

    cfg = load_tier_config({"default": {"level2": {"model": "claudeSDK-sonnet"}}})

The dict shape mirrors :class:`~robotsix_llmio.config.tier.TierConfig`::

    {
      "chain": [
        {"level1": {...}, "level2": {...}, "level3": {...}},
        ...
      ],
      "failover": {"failure_threshold": 3, "window_seconds": 900},
      "vision": {"model": "openrouter-<vision-model>"}
    }

The legacy two-slot shape (top-level ``default`` / ``fallback`` slot dicts
instead of ``chain``) is still accepted; mixing it with ``chain`` is an
error.

Every key is optional; each per-level dict is merged field-by-field over the
corresponding baked default, so overriding one field (say ``model``) keeps
the default's ``provider_kwargs`` and ``max_tokens``. A ``chain`` override
sets the chain's length; positions inside the baked chain merge over it,
positions beyond it must be complete slots. As with
:class:`TierConfig`'s legacy-shape migration, supplying either legacy slot
key yields a two-slot chain whose baked bases are the Claude SDK
(``default``) and OpenRouter (``fallback``) slots; without them the baked
three-slot chain is used. Unknown keys fail
validation loudly (``extra="forbid"`` on the schema) — there is no
environment-variable overlay.

Persisted configs: :func:`load_tier_config_file` reads a JSON config from
disk, migrating a legacy two-slot file onto the chain and re-persisting it
in the ``chain`` shape; :func:`save_tier_config_file` /
:func:`dump_tier_config` always write the ``chain`` shape.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from robotsix_llmio.config.tier import (
    PROVIDER_SLOTS,
    TierConfig,
    TierLevel,
    _legacy_chain,
)
from robotsix_llmio.exceptions import RobotsixLLMIOError

_logger = logging.getLogger(__name__)

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
        TierConfig: A fully-validated configuration — the baked three-slot
        chain, or a two-slot (Claude SDK / OpenRouter) chain when the legacy
        ``default`` / ``fallback`` keys are supplied.

    Raises:
        TierConfigLoadError: If the merged configuration fails validation —
            an unknown provider prefix in a supplied ``model``, an unknown
            key anywhere in the shape, or a slot/level value that is not a
            dict or pydantic model.

    """
    if not config_dict:
        return TierConfig()

    unknown = set(config_dict) - {*PROVIDER_SLOTS, "chain", "failover", "vision"}
    if unknown:
        expected_slots = ", ".join(repr(slot) for slot in PROVIDER_SLOTS)
        raise TierConfigLoadError(
            f"Unknown top-level key(s) {sorted(unknown)!r} in tier config. "
            f"Expected 'chain' (or the legacy {expected_slots}), 'failover', "
            f"and/or 'vision' (the flat level1..level5 shape was removed in "
            f"the provider-failover rework)."
        )

    legacy_keys = [slot for slot in PROVIDER_SLOTS if slot in config_dict]
    if legacy_keys and "chain" in config_dict:
        raise TierConfigLoadError(
            f"Tier config mixes the legacy slot key(s) {legacy_keys!r} with "
            f"'chain'; use only the 'chain' shape."
        )

    baked = TierConfig()
    merged: dict[str, Any] = baked.model_dump()
    if legacy_keys:
        warnings.warn(
            f"Legacy tier config key(s) {legacy_keys!r} are deprecated; use "
            "'chain' instead. The legacy shape will be removed in the next "
            "major version.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Legacy slot keys keep their legacy meaning — ``default`` is the
        # Claude SDK slot and ``fallback`` the OpenRouter slot — exactly as
        # TierConfig's legacy-shape validator migrates them, so an existing
        # ``{"fallback": ...}`` override never lands on a different provider.
        merged["chain"] = [slot.model_dump() for slot in _legacy_chain()]

    try:
        for index, slot in enumerate(PROVIDER_SLOTS):
            if slot not in config_dict:
                continue
            _merge_slot(merged["chain"][index], config_dict[slot])
        if "chain" in config_dict:
            merged["chain"] = _merge_chain(merged["chain"], config_dict["chain"])
        if "failover" in config_dict:
            merged["failover"].update(_to_dict(config_dict["failover"]))
        if "vision" in config_dict:
            merged["vision"].update(_to_dict(config_dict["vision"]))
        return TierConfig.model_validate(merged)
    except (ValidationError, RobotsixLLMIOError) as exc:
        raise TierConfigLoadError(str(exc)) from exc


def dump_tier_config(config: TierConfig) -> dict[str, Any]:
    """Serialise *config* to a JSON-compatible dict in the ``chain`` shape.

    This is the canonical persisted form: the legacy ``default`` /
    ``fallback`` keys are never written, so a config saved with this helper
    round-trips through :func:`load_tier_config` and
    :meth:`TierConfig.model_validate` unchanged.
    """
    return config.model_dump(mode="json")


def save_tier_config_file(config: TierConfig, path: str | os.PathLike[str]) -> None:
    """Persist *config* as JSON in the ``chain`` shape at *path*.

    The write is atomic (temp file in the same directory, then
    :func:`os.replace`), so a crash mid-write never leaves a truncated
    config on disk for the next startup to choke on.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dump_tier_config(config), indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_tier_config_file(
    path: str | os.PathLike[str], *, migrate: bool = True
) -> TierConfig:
    """Load a persisted tier config from the JSON file at *path*.

    A missing file yields the baked configuration and — with *migrate* —
    is written out fresh in the ``chain`` shape. A file in the legacy
    two-slot shape (top-level ``default`` / ``fallback`` keys) is loaded
    through the same migration as :func:`load_tier_config` and — with
    *migrate* — re-persisted in the ``chain`` shape, so a deploy over an
    old on-disk config starts cleanly instead of crashlooping, and the
    cleanup happens once.

    Args:
        path: JSON file holding a :class:`TierConfig`-shaped dict.
        migrate: Rewrite a missing or legacy-shaped file in the ``chain``
            shape. Pass ``False`` for read-only access. The rewrite is
            best-effort: if it fails with :class:`OSError` (e.g. a
            read-only config directory) a warning is logged and the loaded
            config is still returned.

    Raises:
        TierConfigLoadError: If the file is not valid JSON, is not a JSON
            object, or fails validation.

    """
    target = Path(path).resolve()
    if not target.exists():
        config = TierConfig()
        if migrate:
            _try_persist(config, target)
        return config

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TierConfigLoadError(
            f"Cannot read tier config file {str(target)!r}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise TierConfigLoadError(
            f"Tier config file {str(target)!r} must hold a JSON object, "
            f"got {type(raw).__name__!r}."
        )

    config = load_tier_config(raw)
    if migrate and any(slot in raw for slot in PROVIDER_SLOTS):
        _try_persist(config, target)
    return config


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #


def _try_persist(config: TierConfig, target: Path) -> None:
    """Best-effort re-persist of *config* at *target*.

    The migration write is a cleanup, not a requirement: on a read-only
    config directory (ConfigMap/Secret mounts, read-only bind mounts,
    non-root containers) the ``OSError`` is logged and swallowed so the
    already-loaded config is still returned instead of failing startup.
    """
    try:
        save_tier_config_file(config, target)
    except OSError as exc:
        _logger.warning("Could not persist migrated tier config to %s: %s", target, exc)


def _merge_slot(base: dict[str, Any], override: Any) -> None:
    """Merge a slot override field-by-field into *base* (in place)."""
    slot_val = _to_dict(override)
    for tier in (m.value for m in TierLevel):
        if tier in slot_val:
            base[tier].update(_to_dict(slot_val[tier]))


def _merge_chain(base: list[dict[str, Any]], override: Any) -> list[dict[str, Any]]:
    """Merge a ``chain`` override over the baked *base* chain.

    The override's length decides the chain's length. Each position that
    exists in the baked chain merges field-by-field over it; positions
    beyond the baked chain are taken as-is (and must be complete slots).
    """
    if not isinstance(override, (list, tuple)):
        raise TierConfigLoadError(
            f"'chain' must be a list of provider slots, got "
            f"{type(override).__name__!r}."
        )
    result: list[dict[str, Any]] = []
    for index, slot in enumerate(override):
        if index < len(base):
            merged_slot = base[index]
            _merge_slot(merged_slot, slot)
            result.append(merged_slot)
        else:
            result.append(_to_dict(slot))
    return result


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

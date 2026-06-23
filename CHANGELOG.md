# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- Stale env var declarations from `.env.example`: `LLMIO_LEVEL{n}_TRANSPORT`, `LLMIO_LEVEL{n}_PROVIDER` (deprecated aliases), and `LLMIO_PROVIDER` (deprecated fallback). These vars are no longer read by the config loader as of the provider/transport refactor in PR #202.
- `Tier` enum (`robotsix_llmio.core.tier_enum.Tier`) — a backward-compatibility-only two-tier selector (`DEFAULT`/`CHEAP`) superseded by the three-level `TierLevel`/`TierConfig` system. The enum had zero runtime consumption; no `build_agent()` or `new_model()` method accepted a `tier=` parameter. Use `level=1|2|3` with a `TierConfig` instead. (mill: Remove backward-compat-only `Tier` enum — zero remaining production callers, deprecated docstring only (20260620T110305Z-remove-backward-compat-only-tier-enum-ze-f968))
- `__version__` module-level constant from `src/robotsix_llmio/__init__.py` (was `"0.1.0"`). The `pyproject.toml` `version` field is now the single source of truth. Consumers needing the version at runtime should use `importlib.metadata.version("robotsix-llmio")`.

### Changed

- Extracted duplicated tier-config resolution logic from `LLMProvider.build_agent` and `ClaudeSDKProvider.build_agent` into a shared `_resolve_model_name()` helper in `core/provider.py`.
- Extracted shared `_LangfuseReadClientBase` base class from `LangfuseReadClient` and `AsyncLangfuseReadClient`, deduplicating `__init__`, `base_url`, `url()`, and `auth_header()` methods.
- Moved weekly-pace config models (`WeeklyPaceConfig`, `ModelWeightConfig`) from `src/robotsix_llmio/config/weekly_pace.py` into `src/robotsix_llmio/weekly_pace/_config.py`, consolidating all weekly-pace source under the module directory. A compatibility re-export at the old path preserves existing imports.

### Added

- Registered `tests/core/test_identifier.py` in `docs/modules.yaml` under the `robotsix_llmio-core` module.
- Badge row in README.md: PyPI version, supported Python versions, CI status, and license badges.
- Documented identifier parsing (`MalformedIdentifierError`, `ParsedIdentifier`, `parse_model_identifier`), tier fallback (`call_with_tier_fallback`, `acall_with_tier_fallback`), `get_provider_for_identifier`, and `MODEL_LEVEL_TO_TIER` in `docs/core/index.md`.
- Root exception class `RobotsixLLMIOError` that all library-specific errors inherit from, allowing callers to catch library exceptions with a single `except` clause. Exported from top-level package.
- Three-tier configuration system: `TierLevel` (StrEnum with `LEVEL1`/`LEVEL2`/`LEVEL3`), `TierLevelConfig` (transport + model + provider_kwargs per level), and `TierConfig` (aggregates three levels with baked defaults for levels 2 and 3).
- `create_model()` consumer factory — the single entry point to obtain a configured `LLMProvider` by capability level without importing a concrete provider class.
- `load_tier_config()` — merges baked defaults, `LLMIO_LEVEL{1,2,3}_*` environment variables, and an explicit dict into a validated `TierConfig`.
- Transport alias system (`TRANSPORT_ALIASES`) — maps consumer-facing names (`claude-sdk`, `openrouter[deepseek]`) to provider registry names.
- Per-level `LLMIO_LEVEL{1,2,3}_TRANSPORT`, `_MODEL`, and `_PROVIDER_KWARGS` environment variables (deprecating the legacy `LLMIO_FLASH_*` / `LLMIO_NORMAL_*` / `LLMIO_PROVIDER` vars).
- Model registry validation (`PROVIDER_MODELS`, `validate_model()`) — misconfigured model names are caught at config parse time.

### Fixed

- Removed seven phantom symbols from `docs/core/index.md` (`PROVIDER_MODELS`, `TRANSPORT_ALIASES`, `UnknownModelError`, `UnknownTransportError`, `validate_model`, `get_provider`, `register_provider`) that were documented as public API exports but never implemented. Replaced `get_provider` with the actual function name `get_provider_for_identifier`.
- Registered `tests/tools/__init__.py` under the `robotsix_llmio-tools` module in `docs/modules.yaml` (replaced explicit `tests/tools/test_builtins.py` path with `tests/tools/**` glob).

### Changed
- **Breaking**: `ClaudeSDKTurnLimitError` and `ClaudeSDKQueryTimeout` now inherit from `RobotsixLLMIOError` instead of `RuntimeError` and `TimeoutError` respectively. Callers catching these exceptions by type must update to catch `RobotsixLLMIOError`.
- **Deprecated:** `Tier` enum (`CHEAP` / `DEFAULT`) — use `TierLevel` and the `level` parameter (1–3) instead.
- **Deprecated:** `MODEL_LEVEL_TO_TIER` mapping — use `TierConfig.for_level()`.
- **Removed:** `LEGACY_TIER_MAP` — the deprecated backward-compatibility mapping has been removed. Use `TierConfig.for_level()` instead.
- `LLMProvider.new_model()` signature changed: the old `tier` parameter is replaced by `level` (int) + `model` (str).
- `LLMProvider.build_agent()` accepts `level` (int, 1–3) and an optional `tier_config` instead of `tier`.

### Migration

| Old API | New API |
|---------|---------|
| `tier=Tier.CHEAP` | `level=1` |
| `tier=Tier.DEFAULT` | `level=2` |
| *(no equivalent)* | `level=3` |
| `LLMIO_FLASH_MODEL` | `LLMIO_LEVEL1_MODEL` |
| `LLMIO_FLASH_PROVIDER` | `LLMIO_LEVEL1_TRANSPORT` / `LLMIO_LEVEL1_PROVIDER` (deprecated compat alias) |
| `LLMIO_NORMAL_MODEL` | `LLMIO_LEVEL2_MODEL` |
| `LLMIO_NORMAL_PROVIDER` | `LLMIO_LEVEL2_TRANSPORT` / `LLMIO_LEVEL2_PROVIDER` (deprecated compat alias) |
| `LLMIO_PROVIDER` (blanket) | `LLMIO_LEVEL{1,2,3}_TRANSPORT` per level |
| `create_model(transport=..., model_level=...)` | `create_model(level=..., transport=...)` |
| `provider.new_model(tier=...)` | `provider.new_model(level=..., model=...)` |
| `TierConfig` from YAML with `provider` key | `TierConfig` with `transport` key (the old `provider` key is still accepted at input but emits a deprecation warning) |

The old APIs still work but emit `DeprecationWarning` (for `Tier`) or `FutureWarning` (for legacy env vars). Update at your own pace.

## [0.1.0] - 2026-06-13

Initial release.

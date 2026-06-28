# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Consolidated `openrouter_deepseek` into `openrouter`**: the DeepSeek model and
  provider classes (`OpenRouterDeepseekModel`, `OpenRouterDeepseekProvider`) now
  live in `robotsix_llmio.openrouter` (private modules `_deepseek_model` and
  `_deepseek_provider`). The `openrouter_deepseek` extra and its top-level
  package are deprecated — a backward-compat shim re-exports from `openrouter`
  with a deprecation warning. Import from `robotsix_llmio.openrouter` going
  forward.

### Fixed

- `src/robotsix_llmio/config/tier.py`: removed outdated module docstring claiming callers can still pass `Tier.DEFAULT` / `Tier.CHEAP` — the `Tier` StrEnum was fully deleted in PR #259.
- `ARCHITECTURE.md`: removed stale reference to the removed `Tier` enum at line 22, which was deleted in PR #259.
- `ARCHITECTURE.md`: removed second stale `Tier` enum reference at line 102 — replaced with `TierLevel` directly.
- `README.md`: removed stale reference to deleted `Tier.DEFAULT` enum.
- `.env.example`: corrected `LLMIO_LEVEL*_MODEL` example values to use the full `provider-prefix-model` format matching the baked tier defaults, preventing validation failures on onboarding.

### Added

- pytest-timeout plugin with a 30-second global timeout (``timeout_method = "thread"``) to catch hung tests in CI, and ``@pytest.mark.timeout(120)`` overrides on the 9 live Langfuse round-trip tests in ``tests/core/test_tracing_live.py``.

- Added `docs/tools/index.md` module guide for the `robotsix_llmio-tools` module

- Parallel test execution via `pytest-xdist` (`-n auto`) in CI and local config to reduce suite wall-clock time on multi-core runners.
- `src/robotsix_llmio/core/sqlite_utils.py`: `add_column_if_missing()` and `run_additive_migrations()` — additive SQLite column migration helpers that work with both raw `sqlite3.Connection` and SQLAlchemy connections. Lazy-exported from `robotsix_llmio.core`.

### Changed

- `src/robotsix_llmio/core/__init__.py`: replaced the 59-block `__getattr__` if-chain with a dict-driven dispatch using `importlib.import_module`, preserving identical lazy-import semantics.
- `src/robotsix_llmio/config/loader.py`: derive tier names from `TierLevel` enum instead of hardcoding `"level1"`/`"level2"`/`"level3"` string literals, making the tier set a single source of truth in `config/tier.py`.
- CONTRIBUTING.md: replaced pip-based local setup instructions with `uv sync --frozen` to match CI's dependency management, and updated the security-audit command from `pip-audit` to `uv audit`.
- `.github/workflows/release.yml`: switched trigger from `release: [published]` to `push: tags: ['v*']` and enabled `github-release` artifacts via `softprops/action-gh-release@v2`. A `git push --tags` now drives the full release pipeline (build → GitHub Release with dist assets → PyPI publish).

### Fixed

- README.md: replaced phantom symbols `get_provider` and `register_provider` with the actual public API (`get_provider_for_level`, `TierConfig`).
- docs/config/index.md: removed phantom symbol `TRANSPORT_ALIASES` and corrected `get_provider` to `get_provider_for_identifier`.
- docs/config/index.md: removed phantom `transport` parameter from `create_model()` doc signature — the parameter does not exist in the implementation.
- docs/core/index.md: removed stale reference to the removed `Tier` StrEnum (use `TierLevel`/`level` instead).
- README.md: removed the stale "Migrating from `tier` to `level`" section (claiming deprecated `Tier` enum, old env vars, and `DeprecationWarning`/`FutureWarning` still work) — all supporting code was fully removed in prior PRs.

### Added

- `TYPE_CHECKING` static re-declarations in `src/robotsix_llmio/config/__init__.py` to eliminate CodeQL `py/undefined-export` false positives from the PEP 562 lazy `__getattr__` exports.
- API reference pages for `tools`, `config`, `weekly_pace`, and `exceptions` modules in the documentation site.

### Removed

- `robotsix_llmio.tools` subpackage (`src/robotsix_llmio/tools/`, `tests/tools/`, `docs/tools/index.md`, `docs/reference/tools.md`) — the built-in example tools (`get_time`, `echo`, `calculator`, `roll_dice`) were entirely unused by any production code in the repository. The subpackage had zero imports from outside itself and served no purpose beyond maintenance overhead.
- `pip-audit` from the `dev` optional dependencies — CI now uses `uv audit --frozen` for dependency vulnerability auditing. CONTRIBUTING.md and AGENT.md updated to reference `uv audit` instead.
- `bandit` dependency and pre-commit hook — replaced by Ruff's `S` ruleset which covers the same security checks inline during `ruff check`.
- Stale `MODEL_LEVEL_TO_TIER` documentation references from `docs/config/index.md` and `docs/core/index.md`. The mapping was already removed from Python source; only doc references remained.
- Stale env var declarations from `.env.example`: `LLMIO_LEVEL{n}_TRANSPORT`, `LLMIO_LEVEL{n}_PROVIDER` (deprecated aliases), and `LLMIO_PROVIDER` (deprecated fallback). These vars are no longer read by the config loader as of the provider/transport refactor in PR #202.
- `Tier` enum (`robotsix_llmio.core.tier_enum.Tier`) — a backward-compatibility-only two-tier selector (`DEFAULT`/`CHEAP`) superseded by the three-level `TierLevel`/`TierConfig` system. The enum had zero runtime consumption; no `build_agent()` or `new_model()` method accepted a `tier=` parameter. Use `level=1|2|3` with a `TierConfig` instead. (mill: Remove backward-compat-only `Tier` enum — zero remaining production callers, deprecated docstring only (20260620T110305Z-remove-backward-compat-only-tier-enum-ze-f968))
- `__version__` module-level constant from `src/robotsix_llmio/__init__.py` (was `"0.1.0"`). The `pyproject.toml` `version` field is now the single source of truth. Consumers needing the version at runtime should use `importlib.metadata.version("robotsix-llmio")`.

### Changed

- `src/robotsix_llmio/core/tier_fallback.py`: replaced the manually-enumerated `_ALL_TIER_LEVELS` tuple with `tuple(TierLevel)`, so new enum members are automatically included in declaration order without a stale hardcoded list.
- Extracted duplicated tier-config resolution logic from `LLMProvider.build_agent` and `ClaudeSDKProvider.build_agent` into a shared `_resolve_model_name()` helper in `core/provider.py`.
- Extracted shared `_LangfuseReadClientBase` base class from `LangfuseReadClient` and `AsyncLangfuseReadClient`, deduplicating `__init__`, `base_url`, `url()`, and `auth_header()` methods.
- Moved weekly-pace config models (`WeeklyPaceConfig`, `ModelWeightConfig`) from `src/robotsix_llmio/config/weekly_pace.py` into `src/robotsix_llmio/weekly_pace/_config.py`, consolidating all weekly-pace source under the module directory. A compatibility re-export at the old path preserves existing imports.

### Added

- Enable Ruff's `S` (flake8-bandit-security) ruleset in `pyproject.toml`, replacing standalone bandit with 50+ security checks at Rust speed during `ruff check`.
- API reference page for `robotsix_llmio.tools` (`docs/reference/tools.md`) and nav entry in `mkdocs.yml`.
- Config narrative documentation (`docs/config/index.md`) to `mkdocs.yml` navigation, placed between Core and OpenRouter.
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
- **Removed:** `MODEL_LEVEL_TO_TIER` mapping — use `TierConfig.for_level()`.
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

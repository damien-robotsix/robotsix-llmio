# Agent guide

This repo follows the [robotsix stack standards](https://github.com/damien-robotsix/robotsix-standards).

## Project layout

Each robotsix_llmio module uses the per-module layout: code in `src/robotsix_llmio/<module>/`, tests in `tests/<module>/test_*.py`, and docs in `docs/<module>/index.md`. Never place test files in the flat `tests/` root, and register every module's `src`/`tests`/`docs` paths in `docs/modules.yaml`.

**Rule:** When adding a new test or source module under `tests/` or `src/robotsix_llmio/`, register its path in `docs/modules.yaml` in the same change — the manifest must stay in sync with the actual module tree.

## Testing conventions

**Rule:** When writing a new test that patches `httpx.Client` via `MockTransport`, use `install_transport` from `tests/core/conftest.py` with an explicit `module=` parameter pointing to the module under test (e.g. `module=langfuse_cost_module`). Do not define a private duplicate — this pattern of duplicated test helpers has been observed across multiple tickets and should not recur.

**Rule:** When two or more test files under the same `tests/<module>/` subtree share a fixture (defined identically in both files), extract it into a `tests/<module>/conftest.py` instead — pytest discovers conftest fixtures automatically for all sibling tests, avoiding duplication and maintenance drift.

## CI / workflows

**Rule:** A GitHub Actions step that uploads a *required* artifact (e.g. an SBOM) MUST set `if: always()` so it still runs when an earlier step in the same job exits non-zero. A non-zero exit from any preceding step (e.g. `uv audit`, lint, tests) skips all later steps in that job, which makes an `if-no-files-found: error` backstop unreachable and silently drops the artifact — do not rely on a preceding audit/lint/test step staying green to guarantee the upload runs.

## Configuration conventions

**Rule:** When a module reads an environment variable via ``os.environ.get()`` or ``os.environ[]``, define a module-level constant for the variable name (e.g. ``_ENV_LOG_LEVEL = "LOG_LEVEL"``) and reference the constant in every call-site. This centralises the string so a rename is a single-line change. The existing ``ENV_LANGFUSE_*`` constants in ``core/_otel.py`` established this pattern; ``logging.py``, ``openrouter/provider.py``, and ``refdocs/_settings.py`` follow it.

**Rule:** When adding a new environment variable consumed by the library at runtime (e.g. by `src/robotsix_llmio/config/loader.py` and related loader/config modules, or the logging/tracing/refdocs modules that also read env vars — `src/robotsix_llmio/logging.py`, `src/robotsix_llmio/core/tracing.py`, `src/robotsix_llmio/refdocs/_settings.py`), document it in the README's "Configuration" section in the same change — add a row to the runtime environment-variable table with its purpose and default. Derive default/example values from `src/robotsix_llmio/config/tier.py` (the baked tier defaults) or the consuming module's documentation. **Do not** add runtime variables to `tests/.env.example`: that file is scoped to credentials the opt-in live test suite (`pytest -m live`) needs (currently `OPENROUTER_API_KEY` and the `LANGFUSE_*` keys) — only extend it when a *new live test* requires a new credential. **Rationale:** the README is the discoverable home for the library's configuration interface, while `tests/.env.example` stays a minimal, test-only template; keeping each in sync prevents the recurring documentation drift that has required follow-up tickets to correct.

# Agent guide

## Project layout

Each robotsix_llmio module uses the per-module layout: code in `src/robotsix_llmio/<module>/`, tests in `tests/<module>/test_*.py`, and docs in `docs/<module>/index.md`. Never place test files in the flat `tests/` root, and register every module's `src`/`tests`/`docs` paths in `docs/modules.yaml`.

**Rule:** When adding a new test or source module under `tests/` or `src/robotsix_llmio/`, register its path in `docs/modules.yaml` in the same change — the manifest must stay in sync with the actual module tree.

## Testing conventions

**Rule:** When writing a new test that patches `httpx.Client` via `MockTransport`, use `install_transport` from `tests/core/conftest.py` with an explicit `module=` parameter pointing to the module under test (e.g. `module=langfuse_cost_module`). Do not define a private duplicate — this pattern of duplicated test helpers has been observed across multiple tickets and should not recur.

## CI / workflows

**Rule:** A GitHub Actions step that uploads a *required* artifact (e.g. an SBOM) MUST set `if: always()` so it still runs when an earlier step in the same job exits non-zero. A non-zero exit from any preceding step (e.g. `pip-audit`, lint, tests) skips all later steps in that job, which makes an `if-no-files-found: error` backstop unreachable and silently drops the artifact — do not rely on a preceding audit/lint/test step staying green to guarantee the upload runs.

## Configuration conventions

**Rule:** When adding new environment variables consumed by `src/robotsix_llmio/config/loader.py` or related loader/config modules (including the logging/tracing modules that also read env vars, e.g. `src/robotsix_llmio/logging.py`, `src/robotsix_llmio/core/tracing.py`), add corresponding entries to `.env.example` in the same change, as commented-out defaults. Follow the existing `.env.example` convention: derive example/default values from `src/robotsix_llmio/config/tier.py` (the baked tier defaults) or the loader's documentation, and format complex object values (e.g. the `LLMIO_LEVEL{1,2,3}_PROVIDER_KWARGS` JSON) as JSON examples. **Rationale:** keeping `.env.example` in sync makes every configurable setting discoverable and prevents the recurring drift that has required follow-up tickets to correct.

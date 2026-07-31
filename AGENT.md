# Agent guide

This repo follows the [robotsix stack standards](https://github.com/damien-robotsix/robotsix-standards) — fleet-wide conventions (layout, tests, CI, changelog, packaging) live there, not here.

robotsix-llmio is a **library** ([repo-baseline tier](https://damien-robotsix.github.io/robotsix-standards/repo-baseline/)): the stack's LLM provider abstraction — capability levels, cost tracking, tracing — for pydantic-ai agents. It ships no runnable service and is consumed from git by other stack packages. Read `docs/modules.yaml` first for the module map.

## Testing conventions

**Rule:** When writing a new test that patches `httpx.Client` via `MockTransport`, use `install_transport` from `tests/core/conftest.py` with an explicit `module=` parameter pointing to the module under test (e.g. `module=langfuse_cost_module`). Do not define a private duplicate.

**Rationale:** This pattern of duplicated test helpers has been observed across multiple tickets and should not recur.

See [python.md — Tests](https://github.com/damien-robotsix/robotsix-standards/blob/main/python.md) for the shared-fixture-in-conftest rule.

## Configuration conventions

**Rule:** When a module reads an environment variable via ``os.environ.get()`` or ``os.environ[]``, define a module-level constant for the variable name (e.g. ``_ENV_LOG_LEVEL = "LOG_LEVEL"``) and reference the constant in every call-site.

**Rationale:** Centralises the string so a rename is a single-line change. The existing ``ENV_LANGFUSE_*`` constants in ``core/_otel.py`` established this pattern; ``logging.py``, ``openrouter/provider.py``, and ``clients/refdocs/_settings.py`` follow it.

**Rule:** When adding a new environment variable consumed by the library at runtime (e.g. by `src/robotsix_llmio/config/loader.py` and related loader/config modules, or the logging/tracing/refdocs modules that also read env vars — `src/robotsix_llmio/logging.py`, `src/robotsix_llmio/core/tracing.py`, `src/robotsix_llmio/clients/refdocs/_settings.py`), document it in the README's "Configuration" section in the same change — add a row to the runtime environment-variable table with its purpose and default. Derive default/example values from `src/robotsix_llmio/config/tier.py` (the baked tier defaults) or the consuming module's documentation. **Do not** add runtime variables to `tests/.env.example`: that file is scoped to credentials the opt-in live test suite (`pytest -m live`) needs (currently `OPENROUTER_API_KEY` and the `LANGFUSE_*` keys) — only extend it when a *new live test* requires a new credential.

**Rationale:** The README is the discoverable home for the library's configuration interface, while `tests/.env.example` stays a minimal, test-only template; keeping each in sync prevents the recurring documentation drift that has required follow-up tickets to correct.

## CI / workflows

**Rule:** When adding a CVE ignore entry for a dependency vulnerability, update **both** ignore mechanisms: `[tool.uv.audit].ignore` in `pyproject.toml` (for the local `audit` job) **and** `pip-audit-ignore-vulns` in `.github/workflows/ci.yml`'s `security` job (for the shared `pip-audit` scan). CONTRIBUTING.md §4 documents this — keep it in sync.

**Rationale:** Two CI-fix tickets exhibited single-tool-only fixes because CONTRIBUTING.md documented only `uv audit` while CI runs both `uv audit` and `pip-audit` with separate ignore configs.

**Rule:** When adding a changelog.d newsfragment, do NOT add a `changelog.d/*.md` (or `**`) glob to any module's `paths` in `docs/modules.yaml` — the `core` module already claims the newsfragment files via its `changelog.d/*.md` glob (docs/modules.yaml:35), and `project-root` claims only `changelog.d/.gitkeep` (docs/modules.yaml:118). A duplicate glob makes the modules check (vulture/deptry) flag every newsfragment as multi-claimed and fail CI.

**Rationale:** On 2026-07-31 the implement stage of ticket 20260731T133725Z-remove-claude-sdk-model-py-backward-comp-78ca added `changelog.d/*.md` to the `core` module's paths in docs/modules.yaml (commit f3dd455), overlapping project-root's existing `changelog.d/**`; the modules check failed and a fixing_ci cycle reverted the one-line addition (commit 6b86b9a) — a net-zero round-trip. `core` has owned `changelog.d/*.md` since the structural fix on 20260731T134949Z; `project-root` retains only `changelog.d/.gitkeep`.

See [repo-baseline.md — CI gates](https://github.com/damien-robotsix/robotsix-standards/blob/main/repo-baseline.md) for the required-artifact `if: always()` rule.

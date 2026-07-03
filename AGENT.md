# Agent guide

This repo follows the [robotsix stack standards](https://github.com/damien-robotsix/robotsix-standards) — fleet-wide conventions (layout, tests, CI, changelog, packaging) live there, not here.

robotsix-llmio is a **library** ([repo-baseline tier](https://damien-robotsix.github.io/robotsix-standards/repo-baseline/)): the stack's LLM provider abstraction — capability levels, cost tracking, tracing — for pydantic-ai agents. It ships no runnable service and is consumed from git by other stack packages. Read `docs/modules.yaml` first for the module map.

## Testing conventions

**Rule:** When writing a new test that patches `httpx.Client` via `MockTransport`, use `install_transport` from `tests/core/conftest.py` with an explicit `module=` parameter pointing to the module under test (e.g. `module=langfuse_cost_module`). Do not define a private duplicate.

**Rationale:** This pattern of duplicated test helpers has been observed across multiple tickets and should not recur. (Fixture-extraction into `conftest.py` in general is a fleet-wide rule — see [python.md — Tests](https://damien-robotsix.github.io/robotsix-standards/python/#tests).)

## Configuration conventions

**Rule:** When a module reads an environment variable via ``os.environ.get()`` or ``os.environ[]``, define a module-level constant for the variable name (e.g. ``_ENV_LOG_LEVEL = "LOG_LEVEL"``) and reference the constant in every call-site.

**Rationale:** Centralises the string so a rename is a single-line change. The existing ``ENV_LANGFUSE_*`` constants in ``core/_otel.py`` established this pattern; ``logging.py``, ``openrouter/provider.py``, and ``refdocs/_settings.py`` follow it.

**Rule:** When adding a new environment variable consumed by the library at runtime (e.g. by `src/robotsix_llmio/config/loader.py` and related loader/config modules, or the logging/tracing/refdocs modules that also read env vars — `src/robotsix_llmio/logging.py`, `src/robotsix_llmio/core/tracing.py`, `src/robotsix_llmio/refdocs/_settings.py`), document it in the README's "Configuration" section in the same change — add a row to the runtime environment-variable table with its purpose and default. Derive default/example values from `src/robotsix_llmio/config/tier.py` (the baked tier defaults) or the consuming module's documentation. **Do not** add runtime variables to `tests/.env.example`: that file is scoped to credentials the opt-in live test suite (`pytest -m live`) needs (currently `OPENROUTER_API_KEY` and the `LANGFUSE_*` keys) — only extend it when a *new live test* requires a new credential.

**Rationale:** The README is the discoverable home for the library's configuration interface, while `tests/.env.example` stays a minimal, test-only template; keeping each in sync prevents the recurring documentation drift that has required follow-up tickets to correct.

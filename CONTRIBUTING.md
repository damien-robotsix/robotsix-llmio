# Contributing to robotsix-llmio

`robotsix-llmio` is a provider-agnostic LLM I/O layer for
[pydantic-ai](https://ai.pydantic.dev) agents. Contributions are welcome —
please open a GitHub PR against `main`.

## 1. Local development setup

Python **≥ 3.14** is required — the stack runtime baseline (see the
[robotsix stack standards](https://github.com/damien-robotsix/robotsix-standards)).
CI tests 3.14 only.

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (one-time — see https://docs.astral.sh/uv/#installation)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone <repo>
cd <repo>
uv sync --frozen

# Activate the virtual environment
source .venv/bin/activate
```

The `claude_sdk` extra additionally requires Node.js and a logged-in `claude`
CLI at runtime (see the README's "Alternative transport — Claude Agent SDK"
section). Tests that need the Claude CLI are skipped when it's absent —
contributors without Node can still run the rest of the suite.

Copy `tests/.env.example` to `tests/.env` and set `OPENROUTER_API_KEY` **only**
if you intend to run live API tests (see "Running tests" below). `tests/.env`
holds credentials for the opt-in live suite only — the library's runtime
environment variables are documented in the README's "Configuration" section.

## 2. Pre-commit hooks

The `pre-commit` tool is **not** included in the `dev` dependency group —
install it separately:

```bash
uv tool install pre-commit
pre-commit install
```

Optionally run every hook across the whole tree once:

```bash
pre-commit run --all-files
```

`.secrets.baseline` records known/audited dummy values (placeholder API keys in
`tests/.env.example` and literal test fixtures) so the `detect-secrets` hook doesn't
block on them; regenerate it via `detect-secrets scan` when needed.

Hooks pinned in `.pre-commit-config.yaml`:

| hook id                 | description                                  |
|-------------------------|----------------------------------------------|
| trailing-whitespace     | removes trailing whitespace                  |
| end-of-file-fixer       | ensures files end with a single newline      |
| check-yaml              | validates YAML syntax                        |
| check-toml              | validates TOML syntax                        |
| check-json              | validates JSON syntax                        |
| check-merge-conflict    | rejects files with unresolved merge markers  |
| check-added-large-files | rejects files over 1 MB (lockfile/baseline exempt) |
| check-ast               | rejects Python files that don't parse        |
| check-case-conflict     | rejects names that collide case-insensitively |
| debug-statements        | catches leftover `breakpoint()` / `pdb` etc. |
| detect-private-key      | rejects committed private keys               |
| ruff                    | linter (auto-fix on commit)                  |
| ruff-format             | formatter (auto-applied on commit)           |
| mypy                    | type-checks `src/`                           |
| vulture                 | dead-code check over `src/`                  |
| detect-secrets          | scans staged changes for plaintext secrets, audited against `.secrets.baseline` |
| actionlint              | lints GitHub Actions workflow files          |

## 3. Running tests

```bash
# Default suite — what CI runs, no network:
pytest

# Live API tests (opt-in, require OPENROUTER_API_KEY in the environment):
pytest -m live

# With coverage (as CI does):
pytest --cov=src/robotsix_llmio --cov-report=term-missing
```

`pyproject.toml` sets `addopts = "-m 'not live'"`, so plain `pytest` never hits
the network. Running `-m live` overrides that filter; individual live tests
still self-skip when `OPENROUTER_API_KEY` is unset.

## 4. Linting, type-checking, and security checks

Reproduce CI locally before pushing:

```bash
ruff check .                  # lint
ruff format --check .         # format verification (hook auto-formats)
mypy src/                     # type-check
uv audit                      # dependency vulnerability audit
```

`ruff format --check .` mirrors what the `ruff-format` pre-commit hook enforces;
it reports formatting issues without modifying files.

## 5. Code style

- **Line length**: 88 characters — `ruff format` enforces this (matches
  `[tool.ruff] line-length = 88` in `pyproject.toml`).
- **Type hints** are required on public APIs. `mypy src/` runs in CI under a
  strict-ish config (`ignore_missing_imports = false`,
  `warn_unused_configs = true`). The only relaxed area is
  `robotsix_llmio.openrouter.*`, where a mypy override silences
  errors that come from subclassing pydantic-ai's `OpenAIChatModel`.
- **Module layering**: core → openrouter (DeepSeek derived classes live inside
`openrouter`); `claude_sdk` is
  a sibling of openrouter (see the [README](README.md) for the architectural
  narrative). Don't introduce new top-level tunable knobs — timeout, retry, and
  backoff values are baked constants by design.

## 6. Pull request expectations

- Target **`main`**.
- Branch naming: short, kebab-case, topic-prefixed — e.g. `feat/…`, `fix/…`,
  `docs/…`, `chore/…`. This is a convention, not a CI gate.
- CI **must** pass — the test suite on Python 3.14, ruff, mypy,
  and uv audit. CI also
  runs a TruffleHog secret scan on pull requests to catch leaked credentials in
  the PR diff.
- **Every PR adds a changelog newsfragment** in `changelog.d/` — a short
  Markdown file named `<anything-unique>.<type>.md` with type one of
  `breaking`, `feature`, `bugfix`, `misc`. CI enforces this via `towncrier
  check`; apply the `skip-changelog` label for changes with nothing to record.
- Pre-commit hooks must pass locally before pushing.
- New behaviour should ship with tests; bug fixes should ship with a regression
  test. Tests that depend on a live API must be decorated `@pytest.mark.live`
  so they remain opt-in.
- Keep the dependency surface minimal. Prefer `pydantic-ai-slim` extras over the
  full meta-package (see the comment in `pyproject.toml`). Don't add a new
  top-level dependency unless it's genuinely required.

## 7. Reporting issues

Open a [GitHub issue](https://github.com/damien-robotsix/robotsix-llmio/issues)
with:

- a minimal reproducer,
- your Python version (`python --version`),
- the installed extras (`uv pip show robotsix-llmio`),
- and — for provider-specific bugs — which transport you're using (OpenRouter /
  Claude SDK).

## 8. Releasing

The stack publishes to **no package index** — consumers depend on this library
directly from git, pinned to a commit SHA (see the
[repo baseline](https://damien-robotsix.github.io/robotsix-standards/repo-baseline/)).
A release is a version bump, a `v0.X.Y` tag, and a compiled changelog — nothing
is published anywhere:

1. Bump `version` in `pyproject.toml`, then commit/merge the bump to `main`.
2. Tag the release and push the tag:
   ```bash
   git tag v0.2.0 && git push origin v0.2.0
   ```

`CHANGELOG.md` is never edited by hand — it is compiled from the
`changelog.d/` newsfragments with `towncrier build` when a release is cut
(see [changelog & releases](https://damien-robotsix.github.io/robotsix-standards/repo-baseline/#changelog-releases)).

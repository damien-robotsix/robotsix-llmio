# robotsix-llmio

[![CI](https://github.com/damien-robotsix/robotsix-llmio/actions/workflows/ci.yml/badge.svg)](https://github.com/damien-robotsix/robotsix-llmio/actions/workflows/ci.yml)

LLM provider abstraction — capability levels, cost tracking, tracing.
Provider-agnostic LLM I/O for [pydantic-ai](https://ai.pydantic.dev) agents,
with derived per-provider layers that bake in the known-working settings so a
consumer only ever picks a **level** (1, 2, 3, 4, or 5).

This repo is a **library** — imported by other stack packages, no runnable
service — and follows the
[robotsix stack standards](https://github.com/damien-robotsix/robotsix-standards)
([repo-baseline tier](https://damien-robotsix.github.io/robotsix-standards/repo-baseline/)).

**Docs site:** <https://damien-robotsix.github.io/robotsix-llmio/>

## Quickstart

```bash
git clone https://github.com/damien-robotsix/robotsix-llmio.git
cd robotsix-llmio
uv sync            # deps + dev group (pytest, ruff, mypy, ...)
uv run pytest      # offline suite; live tests are opt-in (pytest -m live)
```

## Layers

1. **`robotsix_llmio.core`** — provider-agnostic base: the `LLMProvider` ABC,
   the `TierLevel` enum, bounded retry/backoff
   (`call_with_retry`, `is_transient`, `is_rate_limited`), cost-on-span
   recording, a timeout HTTP client, and the generic pydantic-ai `Agent`
   assembler. All numeric parameters (timeouts, retry counts, backoff) are
   **baked constants** — not tunable.
2. **`robotsix_llmio.openrouter`** — OpenRouter transport: auth/base-url,
   `usage.include` opt-in, cost extraction from `usage.cost`, the
   OpenRouter upstream-error transient signature, and prompt caching of the
   stable request prefix (system prompt + tool schemas) via `cache_control`
   markers. Model-family agnostic.
3. **`robotsix_llmio.openrouter`** — the derived layer most consumers
   plug in. Extends the OpenRouter layer with DeepSeek specifics: pin the
   upstream provider to DeepSeek (warm prompt cache) and a level→reasoning
   policy (level 3→`effort: xhigh`; level 1→`reasoning disabled`).
   pydantic-ai round-trips reasoning natively, so this layer neither remaps
   reasoning nor adds a DeepSeek-specific transient signature (it inherits
   OpenRouter's). The models are **baked**:
   `level 3 = xiaomi/mimo-v2.5-pro`,
   `level 1 = deepseek/deepseek-v4-flash-20260731`.

### Alternative transport — Claude Agent SDK (subscription auth)

`robotsix_llmio.claude_sdk` is a **sibling of the OpenRouter layer** (both derive
from `core.LLMProvider`) that needs **no API key**: it drives the local `claude`
CLI through the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk),
so it authenticates with your `claude login` (Claude Code subscription / OAuth)
credentials. Models are resolved from the tier configuration: the baked
defaults bind `level 2` to `claudeSDK-haiku` (cheap flat-rate: monitors,
retrospects, classification), `level 4` to `claudeSDK-opus` and `level 5` to
`claudeSDK-claude-fable-5` (Claude Fable 5, the frontier tier).

Because the SDK runs its own agent loop and executes tools internally — returning
only final text, never raw `tool_use` blocks — this transport supports
`output_type=str` and pydantic-ai's `PromptedOutput` (JSON-in-text), but **not**
function/tool calling or the default tool-based structured output (those raise a
clear `UserError`). Each request also spawns a fresh CLI subprocess and pays
Claude Code's injected system-prompt overhead, so it's a convenience transport,
not a hot path. Runtime needs Node.js and a logged-in `claude` CLI.

```python
from pydantic import BaseModel
from pydantic_ai import PromptedOutput
from robotsix_llmio.claude_sdk import ClaudeSDKProvider

provider = ClaudeSDKProvider()  # no key — uses your `claude login` session


class City(BaseModel):
    name: str
    country: str


agent = provider.build_agent(
    level=4,
    system_prompt="Extract the city.",
    output_type=PromptedOutput(City),
    name="extract",
)  # level 5 resolves to claude-fable-5 via the baked tier defaults
result = provider.call_with_retry(lambda: agent.run_sync("Tell me about Kyoto."))
print(result.output)  # name='Kyoto' country='Japan'
agent.close()
```

> Auth note: Anthropic restricts offering claude.ai login to third-party *end
> users*; driving your *own* subscription from your own automation is the
> intended personal use. Keep this transport for your own tooling.

## Install

The stack publishes to **no package index** — consume this library directly
from git with [uv](https://docs.astral.sh/uv/), pinned to a commit SHA (see the
[repo baseline](https://damien-robotsix.github.io/robotsix-standards/repo-baseline/)):

```toml
[project]
dependencies = ["robotsix-llmio[openrouter]"]
# or, for the subscription-auth transport (also needs Node + `claude login`):
# dependencies = ["robotsix-llmio[claude_sdk]"]

[tool.uv.sources]
robotsix-llmio = { git = "https://github.com/damien-robotsix/robotsix-llmio.git", rev = "<commit-sha>" }
```

Update the pinned `rev` deliberately, through a reviewed bump — never track a
branch. pip is not a supported install path (it ignores `[tool.uv.sources]`).

## Configuration

The API key can be passed directly to the provider constructor or set via the
`OPENROUTER_API_KEY` environment variable.

The library reads all runtime configuration straight from the process
environment — it does **not** load any `.env` file itself. Set these in your
shell or deployment platform as needed:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | Key required by `OpenRouterProvider` and derived providers | — (required) |
| `REFDOCS_API_KEY` | Bearer token for the refdocs REST API | — |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Enable Langfuse trace/cost export when **both** are set | unset → tracing off |
| `LANGFUSE_BASE_URL` | Langfuse endpoint | `https://cloud.langfuse.com` |
| `LANGFUSE_PROJECT_ID` | Optional Langfuse project id | — |
| `LLMIO_LEVEL{1,2,3,4,5}_MODEL` | Override the baked tier model (see [Five-tier configuration](#five-tier-configuration)) | baked defaults |
| `LLMIO_LEVEL{1,2,3,4}_PROVIDER_KWARGS` | JSON object of provider kwargs for that tier | `{}` |
| `LLMIO_COOLDOWN_DURATION_SECONDS` | Cooldown window in seconds before re-probing a terminally-failing model | `21600` (6 hours) |
| `LLMIO_COOLDOWN_FAILURE_THRESHOLD` | Consecutive terminal failures before a model enters cooldown | `3` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Logging format | `console` |

If you keep these in a local `.env`, load it yourself (e.g. `set -a; source
.env; set +a`) before running your app — and keep it out of version control.

To route requests through a custom OpenRouter-compatible endpoint (e.g. a proxy
or mirror), pass `base_url=` when constructing the provider:

```python
provider = get_provider_for_level(
    level=1, api_key="sk-...", base_url="https://proxy.example/api/v1"
)
```

The default endpoint is `https://openrouter.ai/api/v1`.

## Five-tier configuration

The library uses a five-tier model selection system exposed through the
`level` parameter on `LLMProvider.build_agent()`.  Each level is backed by a
`TierLevelConfig` holding a combined `provider-model` identifier:

| Level | Intended use                           | Example env var (combined identifier)                    |
|-------|----------------------------------------|----------------------------------------------------------|
| 1     | Cheap, obvious, repetitive tasks (pay-per-token) | `LLMIO_LEVEL1_MODEL=openrouter-deepseek/deepseek-v4-flash-20260731` |
| 2     | Cheap flat-rate: monitors, retrospects, classification | `LLMIO_LEVEL2_MODEL=claudeSDK-haiku` |
| 3     | Intermediate (e.g. implementing code)  | `LLMIO_LEVEL3_MODEL=openrouter-xiaomi/mimo-v2.5-pro`   |
| 4     | High-level planning and refinement     | `LLMIO_LEVEL4_MODEL=claudeSDK-opus`                        |
| 5     | Frontier — hardest reasoning and long-horizon work | `LLMIO_LEVEL5_MODEL=claudeSDK-claude-fable-5`   |

Level 1 is the default — cheap and fast is the safe default.  The
configuration system (`TierConfig` / `load_tier_config` / `call_with_tier_fallback`)
supports all five levels end-to-end.

You can also set `LLMIO_LEVEL<N>_PROVIDER_KWARGS` as a JSON object for extra
constructor arguments (e.g. `{"base_url": "https://proxy.example/api/v1"}`).

### One-liner: pick a level, get an agent

For new code, the consumer never needs provider knowledge — just pick a level.
`build_agent_for_level` resolves that level's baked default *(provider, model)*
binding, lazy-imports the right backend, and returns a ready-to-run agent:

```python
from robotsix_llmio import build_agent_for_level

# Level 1 → OpenRouter DeepSeek (deepseek-v4-flash-20260731): cheap and fast.
cheap = build_agent_for_level(1, system_prompt="Classify this.", name="classify")

# Level 4 → Claude SDK (opus): high-level planning. Requires the
# `claude_sdk` extra.
planner = build_agent_for_level(
    4, system_prompt="Plan this epic.", tools=[], name="plan"
)

# Level 5 → Claude SDK (claude-fable-5): frontier tier for the hardest
# reasoning and long-horizon work. Requires the `claude_sdk` extra.
architect = build_agent_for_level(
    5, system_prompt="Design the migration strategy.", name="architect"
)
```

With everything left at its default the baked per-level defaults apply
(level 1 → `deepseek/deepseek-v4-flash-20260731`, level 2 → `haiku`, level 3 → `xiaomi/mimo-v2.5-pro`,
level 4 → `opus`, level 5 → `claude-fable-5`) — each on its **own** provider, so a DeepSeek model never runs
on the Claude transport. `model=` overrides only the model name (the provider
stays the one bound to the level); pass a custom `tier_config=` to override the
bindings, and `provider_kwargs=` for provider-constructor arguments. Two related
helpers are exported alongside it: `get_provider_for_level(level)` (just the
provider) and `default_tier_config()` (the baked binding as a `TierConfig`).


See [docs/config/index.md](docs/config/index.md) for the full `TierConfig`
schema and `create_model` factory API.

## Use

Obtain a provider through `get_provider_for_level` and pick a **level** (1, 2, 3, or 4).
Level 1 is the default — cheap and fast.

For new code, prefer `create_model` — it resolves the provider from your
tier configuration without naming a concrete backend:

```python
from robotsix_llmio.config import create_model

provider = create_model(level=2, api_key="sk-or-...")
```

```python
from robotsix_llmio.core import get_provider_for_level

provider = get_provider_for_level(
    level=1, api_key="sk-or-..."
)  # or OPENROUTER_API_KEY env

agent = provider.build_agent(
    level=2,
    system_prompt="You are a reviewer. Return a verdict.",
    tools=[],
    output_type=str,
    name="review",
)
result = provider.call_with_retry(lambda: agent.run_sync("Review this diff: ..."))
agent.close()
```

The backend is resolved from config — no consumer code change is needed to swap
it. By default `get_provider_for_level` resolves the provider bound to the
level in `TierConfig`. To override those bindings via environment variables
(`LLMIO_LEVEL<N>_PROVIDER` / `LLMIO_LEVEL<N>_MODEL`), pass
`tier_config=load_tier_config()` explicitly — the level factories do not read
the environment on their own. `get_provider_for_level`
forwards any extra keyword arguments to the chosen backend's constructor, so
pass the kwargs that backend accepts (e.g. `api_key=` for `openrouter-deepseek`,
nothing for `claude-sdk`).

The level-based `TierConfig` system eliminates the need for runtime provider
registration — add a new provider by contributing a `TierLevelConfig` and
setting the corresponding `LLMIO_LEVEL<N>_PROVIDER` variable. Importing a
concrete provider class directly still works (e.g.
`from robotsix_llmio.openrouter import OpenRouterDeepseekProvider`),
but `get_provider_for_level` is the preferred entry point.

## Error handling

All library-specific exceptions inherit from `RobotsixLLMIOError`, so you can
catch all library errors with a single clause:

```python
from robotsix_llmio import RobotsixLLMIOError

try:
    result = provider.call_with_retry(lambda: agent.run_sync("..."))
except RobotsixLLMIOError as e:
    print(f"Library error: {e}")
```

Specific exceptions include:
- `ClaudeSDKTurnLimitError` — Claude Agent SDK loop hit its turn cap without returning an answer.
- `ClaudeSDKQueryTimeout` — A Claude Agent SDK query exceeded the per-call timeout (a stall, typically transient).

## Tracing & cost (Langfuse)

Every provider model already stamps per-call cost onto the active OpenTelemetry
span. To ship those spans — traces **and** cost, for any provider — to a
[Langfuse](https://langfuse.com) project, call `setup_langfuse_tracing()` once at
startup. It wires an OTLP exporter to Langfuse and `Agent.instrument_all()`, so
every subsequent agent run is traced. It's a **no-op** unless
`LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are set (`LANGFUSE_BASE_URL`
defaults to Langfuse Cloud), so it's always safe to call.

Add the `tracing` extra — `robotsix-llmio[tracing]` — to pull in the OTLP
exporter (no langfuse SDK needed).

```python
from robotsix_llmio.core import setup_langfuse_tracing, langfuse_session, flush_tracing

setup_langfuse_tracing()  # reads LANGFUSE_* env; no-op without credentials

with langfuse_session("my-run-id"):  # groups the run's spans under one session
    result = provider.call_with_retry(lambda: agent.run_sync("..."))

flush_tracing()  # force-export before exit (or after a run you want shipped)
```

**Explicit root span** — group several agent runs (and non-agent steps) under one
trace with stage-level input/output via `start_trace`:

```python
from robotsix_llmio.core import start_trace

with start_trace("review-stage", session_id="ticket-42") as trace:
    trace.set_input({"ticket": 42})
    result = provider.call_with_retry(lambda: agent.run_sync("..."))
    trace.set_output(result.output)
    print(trace.trace_id)
```

**Multi-tenant** — one process exporting to several Langfuse projects: call
`setup_langfuse_tracing(public_key=..., secret_key=..., base_url=...)` once per
project, then route each unit of work to its project:

```python
from robotsix_llmio.core import langfuse_project

with langfuse_project("pk-projectB"):  # spans here ship to project B
    result = provider.call_with_retry(lambda: agent.run_sync("..."))
```

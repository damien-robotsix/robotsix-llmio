# robotsix_llmio config

Consumer-facing configuration layer — `TierConfig` schema, loader,
provider prefixes, and the `create_model` factory.

The config package is the **entry point consumers should import from**
to obtain a working model/agent without naming a concrete provider class
or pulling in `claude-agent-sdk` directly.

```python
from robotsix_llmio.config import create_model

provider = create_model(level=2)
agent = provider.build_agent(
    level=2,
    system_prompt="You are a helpful assistant.",
)
```

## Exports

### Factory

- `create_model` — the single consumer-facing entry point.  Signature:

  ```python
  def create_model(
      *,
      level: int = 1,
      tier_config: TierConfig | None = None,
      **provider_kwargs: Any,
  ) -> LLMProvider:
  ```

  - **`level`** (1, 2, or 3) — selects the capability level: 1 cheap/frequent,
    2 workhorse, 3 frontier. Defaults to 1. Resolution honours the provider
    slot the failover tracker currently designates as active.
  - **`tier_config`** — optional `TierConfig` instance supplying custom
    slot/level bindings.  When `None`, a default is built from the baked
    constants.
  - **`**provider_kwargs`** — forwarded to the provider constructor (e.g.
    `api_key=...`).  These override any `provider_kwargs` from the tier
    config.

  Returns a fully-instantiated `LLMProvider`.  Raises `ValueError` for
  unknown provider prefixes or invalid levels, and `ImportError` when a
  required optional extra is not installed.

### Provider prefixes

An identifier is `<provider>-<model-name>`. The provider prefix (before the
first hyphen) selects the backend known to `get_provider_for_identifier`:

| Provider prefix | Backend |
|---|---|
| `claudeSDK` | Claude Agent SDK |
| `openrouter` | OpenRouter (incl. DeepSeek models) |

### Schema & loader (tier configuration)

- `TierConfig` — pydantic model holding two provider slots (`default`,
  `fallback`), the `failover` policy, and the `vision` binding (the model
  answering `ask_image` questions for text-only transports)
- `ProviderSlotConfig` — one slot's binding of all three levels
  (`level1`, `level2`, `level3`)
- `FailoverConfig` — provider-failover policy (`failure_threshold`,
  `window_seconds`)
- `TierLevel` — `StrEnum` with `LEVEL1`, `LEVEL2`, `LEVEL3` selector values
- `TierLevelConfig` — pydantic model binding a single level to a
  provider-model identifier.  Each config carries:
  - **`model`** — the combined `provider-model` identifier.
  - **`provider_kwargs`** — extra constructor arguments forwarded to the provider.
  - **`max_tokens`** — optional output token cap (baked per level; see table below).
- `DEFAULT_LEVEL1..3`, `FALLBACK_LEVEL1..3` — the baked `TierLevelConfig`
  instances per slot and level
- `TierConfigLoadError` — raised when tier configuration cannot be loaded
- `load_tier_config` — merges an explicit dict over the baked defaults into
  a validated `TierConfig` (there is no environment-variable overlay)

## Consumer config shape

`load_tier_config` accepts a dict mirroring the `TierConfig` shape; every
key is optional and per-level dicts merge field-by-field over the baked
default for that slot+level:

```python
from robotsix_llmio.config import load_tier_config, create_model

cfg = load_tier_config(
    {
        "default": {"level2": {"model": "claudeSDK-sonnet"}},
        "fallback": {"level3": {"model": "openrouter-deepseek/deepseek-v4-pro"}},
        "failover": {"failure_threshold": 3, "window_seconds": 900},
        "vision": {"model": "openrouter-google/gemini-2-flash"},
    }
)
provider = create_model(level=2, tier_config=cfg)
```

Unknown keys fail validation loudly (`extra="forbid"`), including the
pre-rework flat `level1..level5` shape.

## Baked defaults

Two provider slots, three levels each; level 1 works out of the box:

| Slot / level | Identifier | `max_tokens` |
|--------------|------------|--------------|
| `default.level1` | `claudeSDK-haiku` | — |
| `default.level2` | `claudeSDK-opus` | — |
| `default.level3` | `claudeSDK-claude-fable-5` | — |
| `fallback.level1` | `openrouter-deepseek/deepseek-v4-flash-20260731` | 16 384 |
| `fallback.level2` | `openrouter-deepseek/deepseek-v4-pro-0813` (xhigh reasoning) | 131 072 |
| `fallback.level3` | `openrouter-deepseek/deepseek-v4-pro-0813` | 131 072 |
| `vision` | `openrouter-deepseek/deepseek-v4-flash-vision-exp` | 8 192 |

The Claude SDK levels deliberately carry no `max_tokens` (the SDK has no
per-response cap; the value could only become an advisory `task_budget`).
Failover between the slots is handled by
[`robotsix_llmio.core.failover`](../core/index.md) — after
`failure_threshold` consecutive default-slot failures (exhaustion arms
immediately) calls route to the fallback slot for `window_seconds`
(default 15 minutes), then automatically return to the default.

## Extra dependencies

The `claudeSDK` provider requires the `claude_sdk` extra:

```bash
pip install "robotsix-llmio[claude_sdk]"
```

The `openrouter` provider requires the `openrouter` extra:

```bash
pip install "robotsix-llmio[openrouter]"
```

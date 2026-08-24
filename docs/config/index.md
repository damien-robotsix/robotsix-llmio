# robotsix_llmio config

Consumer-facing configuration layer — `TierConfig` schema, loader,
provider prefixes, and the `create_model` factory.

The config package is the **entry point consumers should import from**
to obtain a working model/agent without naming a concrete provider class
or pulling in `claude-agent-sdk` directly.

```python
from robotsix_llmio.config import create_model

provider = create_model(level=3)
agent = provider.build_agent(
    level=3,
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

  - **`level`** (1, 2, 3, or 4) — selects the capability tier.  Level 1 picks the
    cheap/fast default; levels 2–4 pick progressively more capable defaults
    (level 4 is the frontier tier, `claudeSDK-claude-fable-5` by default).
    Defaults to 1.
  - **`tier_config`** — optional `TierConfig` instance supplying custom
    per-level defaults.  When `None`, a default is built from the baked
    module-level constants (`LEVEL1_DEFAULT`, `LEVEL2_DEFAULT`,
    `LEVEL3_DEFAULT`).
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

- `TierConfig` — pydantic model for four-tier provider+model configuration
- `TierLevel` — `StrEnum` with `LEVEL1`, `LEVEL2`, `LEVEL3`, `LEVEL4` tier-selector values
- `TierLevelConfig` — pydantic model binding a single tier to a provider-model identifier.
  Each config carries:
  - **`model`** — the combined `provider-model` identifier.
  - **`provider_kwargs`** — extra constructor arguments forwarded to the provider.
  - **`max_tokens`** — optional output token cap (baked per-level; see table below).
- `LEVEL1_DEFAULT`, `LEVEL2_DEFAULT`, `LEVEL3_DEFAULT`, `LEVEL4_DEFAULT` — default
  `TierLevelConfig` instances per level
- `TierConfigLoadError` — raised when tier configuration cannot be loaded
- `load_tier_config` — loads and validates a `TierConfig` from environment overrides and defaults

## Consumer config shape

Consumers can define tier configuration in two ways: programmatically with
`TierConfig` / `TierLevelConfig` constructors, or via environment variables
loaded through `load_tier_config`.

### Programmatic (dict / constructor)

```python
from robotsix_llmio.config import TierConfig, TierLevelConfig, create_model

cfg = TierConfig(
    level1=TierLevelConfig(model="openrouter-deepseek/deepseek-v4-flash-latest"),
    # level2 and level3 use baked defaults
)

provider = create_model(level=2, tier_config=cfg)
```

When `level2` and `level3` are omitted from the `TierConfig` constructor,
the baked defaults (`LEVEL2_DEFAULT` and `LEVEL3_DEFAULT`) are used
automatically.

### Environment variables (via `load_tier_config`)

Set per-level environment variables and call `load_tier_config` to merge
them with baked defaults:

```bash
export LLMIO_LEVEL1_MODEL="openrouter-deepseek/deepseek-v4-flash"
export LLMIO_LEVEL2_MODEL="openrouter-xiaomi/mimo-v2.5-pro"
export LLMIO_LEVEL2_PROVIDER_KWARGS='{"api_key":"sk-or-..."}'
```

```python
from robotsix_llmio.config import load_tier_config, create_model

cfg = load_tier_config()
provider = create_model(level=2, tier_config=cfg)
```

An explicit dict can override individual fields at highest precedence:

```python
cfg = load_tier_config(
    {
        "level2": {"model": "openrouter-xiaomi/mimo-v2.5-pro"},
    }
)
```

The loader merges three sources in order of increasing precedence:
1. Baked defaults (`LEVEL1_DEFAULT` … `LEVEL4_DEFAULT`).
2. Environment variables (`LLMIO_LEVEL{1,2,3,4}_*`).
3. Explicit dict argument.

## Default levels

The library ships with the following baked defaults so **level 1** works
out of the box and levels 2–4 have sensible fallbacks:

| Constant | Identifier | `max_tokens` |
|----------|------------|--------------|
| `LEVEL1_DEFAULT` | `openrouter-deepseek/deepseek-v4-flash-latest` | 16 384 |
| `LEVEL2_DEFAULT` | `openrouter-xiaomi/mimo-v2.5-pro` | 32 768 |
| `LEVEL3_DEFAULT` | `claudeSDK-opus` | 8 192 |
| `LEVEL4_DEFAULT` | `claudeSDK-claude-fable-5` | 16 384 |

Level 1 is the default when no level is specified — cheap and fast is the
safe default.

## Extra dependencies

The `claudeSDK` provider requires the `claude_sdk` extra:

```bash
pip install "robotsix-llmio[claude_sdk]"
```

The `openrouter` provider requires the `openrouter` extra:

```bash
pip install "robotsix-llmio[openrouter]"
```

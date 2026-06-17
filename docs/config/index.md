# robotsix_llmio config

Consumer-facing configuration layer — `TierConfig` schema, loader,
transport aliases, and the `create_model` factory.

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
      transport: str | None = None,
      tier_config: TierConfig | None = None,
      **provider_kwargs: Any,
  ) -> LLMProvider:
  ```

  - **`level`** (1, 2, or 3) — selects the capability tier.  Level 1 picks the
    cheap/fast default; levels 2 and 3 pick progressively more capable defaults.
    Defaults to 1.
  - **`transport`** — optional consumer-facing transport alias
    (`"claude-sdk"` or `"openrouter[deepseek]"`).  When supplied, it overrides
    the level-based provider choice — useful for pinning a specific backend
    regardless of tier.  When `None`, the provider is resolved from
    `tier_config.for_level(level).provider`.
  - **`tier_config`** — optional `TierConfig` instance supplying custom
    per-level defaults.  When `None`, a default is built from the baked
    module-level constants (`LEVEL1_DEFAULT`, `LEVEL2_DEFAULT`,
    `LEVEL3_DEFAULT`).
  - **`**provider_kwargs`** — forwarded to the provider constructor (e.g.
    `api_key=...`).  These override any `provider_kwargs` from the tier
    config.

  Returns a fully-instantiated `LLMProvider`.  Raises `ValueError` for
  unknown transports or invalid levels, and `ImportError` when a required
  optional extra is not installed.

### Transport alias mappings

- `TRANSPORT_ALIASES` — maps consumer-facing transport names to provider
  registry names known to `get_provider`.

  | Consumer alias | Provider registry name |
  |---|---|
  | `claude-sdk` | `claude-sdk` |
  | `openrouter[deepseek]` | `openrouter-deepseek` |

- `MODEL_LEVEL_TO_TIER` — **deprecated** mapping from `model_level` integers
  (1, 2, 3) to `Tier` enum values.  Level 1 → `Tier.CHEAP`, levels 2 and
  3 → `Tier.DEFAULT`.  Prefer `TierConfig.for_level()` which resolves
  directly to a `TierLevelConfig` without the two-tier round-trip.  This
  mapping is kept for backward compatibility only.

### Schema & loader (tier configuration)

- `TierConfig` — pydantic model for three-tier provider+model configuration
- `TierLevel` — `StrEnum` with `LEVEL1`, `LEVEL2`, `LEVEL3` tier-selector values
- `TierLevelConfig` — pydantic model binding a single tier's transport and model
- `LEVEL1_DEFAULT`, `LEVEL2_DEFAULT`, `LEVEL3_DEFAULT` — default `TierLevelConfig`
  instances per level
- `LEGACY_TIER_MAP` — **deprecated** — maps legacy `Tier.CHEAP`/`Tier.DEFAULT` to
  the new three-tier levels; use `TierConfig.for_level()` instead
- `TierConfigLoadError` — raised when tier configuration cannot be loaded
- `load_tier_config` — loads and validates a `TierConfig` from YAML and environment

### Weekly pace (Claude usage governor)

- `WeeklyPaceConfig` — configuration for the weekly Claude usage pace governor
- `ModelWeightConfig` — per-model weight mapping for weighted consumption calculations

## Consumer config shape

Consumers can define tier configuration in two ways: programmatically with
`TierConfig` / `TierLevelConfig` constructors, or via environment variables
loaded through `load_tier_config`.

### Programmatic (dict / constructor)

```python
from robotsix_llmio.config import TierConfig, TierLevelConfig, create_model

cfg = TierConfig(
    level1=TierLevelConfig(
        transport="openrouter[deepseek]",
        model="deepseek/deepseek-v4-flash",
    ),
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
export LLMIO_LEVEL1_TRANSPORT="openrouter[deepseek]"
export LLMIO_LEVEL1_MODEL="deepseek/deepseek-v4-flash"
export LLMIO_LEVEL2_TRANSPORT="openrouter[deepseek]"
export LLMIO_LEVEL2_MODEL="deepseek/deepseek-v4-pro"
export LLMIO_LEVEL2_PROVIDER_KWARGS='{"api_key":"sk-or-..."}'
```

```python
from robotsix_llmio.config import load_tier_config, create_model

cfg = load_tier_config()
provider = create_model(level=2, tier_config=cfg)
```

An explicit dict can override individual fields at highest precedence:

```python
cfg = load_tier_config({
    "level2": {"model": "deepseek/deepseek-v4-pro"},
})
```

The loader merges three sources in order of increasing precedence:
1. Baked defaults (`LEVEL2_DEFAULT`, `LEVEL3_DEFAULT`).
2. Environment variables (`LLMIO_LEVEL{1,2,3}_*`).
3. Explicit dict argument.

## Default levels

The library ships with the following baked defaults so **level 1** works
out of the box and levels 2+3 have sensible fallbacks:

| Constant | Transport | Model |
|----------|-----------|-------|
| `LEVEL1_DEFAULT` | `openrouter[deepseek]` | `deepseek/deepseek-v4-flash` |
| `LEVEL2_DEFAULT` | `openrouter[deepseek]` | `deepseek/deepseek-v4-pro` |
| `LEVEL3_DEFAULT` | `claude-sdk` | `opus` |

Level 1 is the default when no level is specified — cheap and fast is the
safe default.

## Extra dependencies

The `claude-sdk` transport requires the `claude_sdk` extra:

```bash
pip install "robotsix-llmio[claude_sdk]"
```

The `openrouter[deepseek]` transport requires the `openrouter_deepseek` extra
(which is the default when no extra is specified):

```bash
pip install "robotsix-llmio[openrouter_deepseek]"
```

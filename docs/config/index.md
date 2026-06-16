# robotsix_llmio config

Consumer-facing configuration layer — `TierConfig` schema, loader,
transport aliases, and the `create_model` factory.

The config package is the **entry point consumers should import from**
to obtain a working model/agent without naming a concrete provider class
or pulling in `claude-agent-sdk` directly.

```python
from robotsix_llmio.config import create_model, MODEL_LEVEL_TO_TIER

provider = create_model(transport="claude-sdk", model_level=3)
agent = provider.build_agent(
    tier=MODEL_LEVEL_TO_TIER[3],
    system_prompt="You are a helpful assistant.",
)
```

## Exports

### Factory

- `create_model` — the single consumer-facing entry point.  Takes a
  `transport` alias (e.g. `"claude-sdk"`, `"openrouter[deepseek]"`) and a
  `model_level` (1, 2, or 3), resolves them to a provider registry name and a
  `Tier`, and returns a fully-instantiated `LLMProvider`.  Forwarded keyword
  arguments (e.g. `api_key=`) are passed through to the provider constructor.
  Raises `ValueError` for unknown transports or invalid levels, and
  `ImportError` when the required optional extra is not installed.

### Transport alias mappings

- `TRANSPORT_ALIASES` — maps consumer-facing transport names to provider
  registry names known to `get_provider`.

  | Consumer alias | Provider registry name |
  |---|---|
  | `claude-sdk` | `claude-sdk` |
  | `openrouter[deepseek]` | `openrouter-deepseek` |

- `MODEL_LEVEL_TO_TIER` — maps `model_level` integers (1, 2, 3) to
  `Tier` enum values.  Level 1 → `Tier.CHEAP`, levels 2 and 3 → `Tier.DEFAULT`
  (the legacy `Tier` enum has no third tier, so level 3 falls back to `DEFAULT`).

### Schema & loader (tier configuration)

- `TierConfig` — pydantic model for three-tier provider+model configuration
- `TierLevel` — `StrEnum` with `LEVEL1`, `LEVEL2`, `LEVEL3` tier-selector values
- `TierLevelConfig` — pydantic model binding a single tier's provider and model
- `LEVEL1_DEFAULT`, `LEVEL2_DEFAULT`, `LEVEL3_DEFAULT` — default `TierLevelConfig`
  instances per level
- `LEGACY_TIER_MAP` — maps legacy `Tier.CHEAP`/`Tier.DEFAULT` to the new three-tier levels
- `TierConfigLoadError` — raised when tier configuration cannot be loaded
- `load_tier_config` — loads and validates a `TierConfig` from YAML and environment

### Weekly pace (Claude usage governor)

- `WeeklyPaceConfig` — configuration for the weekly Claude usage pace governor
- `ModelWeightConfig` — per-model weight mapping for weighted consumption calculations

## Consumer config shape

Consumers define `transport` and `model_level` in their own config file
(typically YAML) and pass them to `create_model`:

```yaml
llmio:
  transport: claude-sdk        # or: openrouter[deepseek]
  model_level: 1               # 1 | 2 | 3
```

The factory resolves these to a ready-to-use provider — no concrete provider
class is ever imported by consumer code.

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

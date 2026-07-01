# robotsix_llmio openrouter

OpenRouter transport layer — base model-family-agnostic provider, plus a
derived DeepSeek layer that pins to DeepSeek models and configures per-tier
reasoning policy.

## DeepSeek (derived layer)

The module also exports DeepSeek-specific classes that pin the generic
OpenRouter transport to DeepSeek models on OpenRouter:

- `OpenRouterDeepseekProvider` — provider that maps `level=2` to
  `"deepseek/deepseek-v4-pro"` (reasoning at `"xhigh"`) and `level=1`
  to `"deepseek/deepseek-v4-flash"` (reasoning disabled).
- `OpenRouterDeepseekModel` — model that injects
  `provider: {only: ["DeepSeek"], allow_fallbacks: false}` and per-level
  `reasoning` settings into every request.

The DeepSeek classes are imported from `robotsix_llmio.openrouter`:

```python
from robotsix_llmio.openrouter import (
    OpenRouterDeepseekModel,
    OpenRouterDeepseekProvider,
)
```

## Exports

### Provider & model

- `OpenRouterProvider` — base provider that builds cost-instrumented OpenRouter
  models; tier→model mapping is supplied by `TierConfig` — subclasses optionally
  inject provider-family quirks such as upstream pinning and per-tier reasoning policy
- `OpenRouterModel` — `OpenAIChatModel` subclass that opts into OpenRouter usage
  accounting (`usage.include`) and stamps `usage.cost` onto the active OTel span

### Transient error detection

- `is_openrouter_transient` — returns `True` for core transient errors *or* the
  OpenRouter upstream-error signature, walking the cause/context chain
- `is_openrouter_upstream_error` — detects OpenRouter's `finish_reason='error'`
  upstream-failure pattern (pydantic `ValidationError` mentioning
  `finish_reason` and `'error'`)

### Cost sources & recording

- `OpenRouterKeyCostSource` — reads per-key cumulative lifetime usage from
  `GET /api/v1/auth/key`; snapshot-and-diff to get a window's spend per key
- `OpenRouterProviderCostSource` — reads OpenRouter's billed spend for a time
  window via `GET /api/v1/activity`, summing across each UTC day the window
  covers
- `AsyncOpenRouterClient` — async REST client for per-key usage (`fetch_key_usage`)
  and account credits (`fetch_credits`); async counterpart of
  `OpenRouterKeyCostSource`
- `record_openrouter_cost` — stamps per-call cost, token counts, cache details,
  and gen_ai attributes onto the active OTel span (no-op without OTel or a
  recording span)

### Usage data types

- `KeyUsage` — frozen dataclass: cumulative lifetime usage and credit limit for
  an OpenRouter API key, as returned by `OpenRouterKeyCostSource`

# robotsix_llmio openrouter

OpenRouter transport layer — base model-family-agnostic provider, plus a
derived DeepSeek layer that pins to DeepSeek models and configures per-tier
reasoning policy.

## DeepSeek (derived layer)

The module also exports DeepSeek-specific classes that pin the generic
OpenRouter transport to DeepSeek models on OpenRouter:

- `OpenRouterDeepseekProvider` — provider that maps `level=2` to
  `"deepseek/deepseek-v4-pro"` (reasoning at `"xhigh"`) and `level=1` to
  `"deepseek/deepseek-v4-flash-latest"` (reasoning disabled).  Routing is configurable
  via ``provider_kwargs``: ``max_price_prompt``, ``max_price_completion``, and
  ``ignore_providers`` (a list of upstream providers to exclude from fallback
  routing — see ``OpenRouterDeepseekModel`` below).
- `OpenRouterDeepseekModel` — model that injects
  ``provider: {order: ["DeepSeek"], allow_fallbacks: true, max_price: …, ignore: …}``
  and per-level ``reasoning`` settings into every request. ``order`` keeps DeepSeek
  first (prompt-cache warmth) while ``allow_fallbacks`` lets OpenRouter route
  past it when it fails; ``max_price`` caps the fallback's sticker price per 1M
  tokens — each tier's ceiling is set from DeepSeek's own sticker price plus
  headroom so ``order`` and ``max_price`` never contradict. ``ignore`` bars
  providers whose cache-read rate is a large multiple of DeepSeek's — a cost
  ``max_price`` cannot see (OpenRouter's ``max_price`` accepts only ``prompt``,
  ``completion``, ``request`` and ``image``).

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
  accounting (`usage.include`) and stamps `usage.cost` onto the active OTel span.
  It also annotates the stable request prefix (last system message and last tool
  definition) with `cache_control: {"type": "ephemeral"}` markers so
  OpenRouter-compatible upstream providers (DeepSeek, Anthropic, …) apply
  prompt caching to the repeated system prompt + tool schemas. Subclasses or
  callers can opt out per-model by setting `_prompt_caching_enabled = False`.
  Cache semantics are documented in the `openrouter/model.py` module docstring.

### Transient error detection

- `is_openrouter_transient` — returns `True` for core transient errors *or* any
  of the OpenRouter upstream-error signatures below, walking the cause/context
  chain
- `is_openrouter_upstream_error` — detects OpenRouter's
  `finish_reason='error'` upstream-failure pattern (pydantic `ValidationError`
  mentioning `finish_reason` and `'error'`)
- `is_openrouter_upstream_payment_error` — detects a 402 raised by the
  *upstream provider* (OpenRouter routed to a provider with no balance of its
  own, carrying `metadata.provider_name` and `metadata.is_byok: false`), as
  distinct from our own account running out of credits — only the former is
  transient
- `is_deepseek_reasoning_400` — detects DeepSeek's reasoning-content 400
  (missing `reasoning_content` in thinking mode), treating it as an
  infrastructure hiccup to retry

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
  recording span). When the response carries cached-token details it also emits
  an INFO log line summarising the cached-vs-uncached input-token split (total /
  cached / `%` hit / cache-creation tokens) so the prompt-caching win is
  measurable in logs without a Langfuse UI.

### Usage data types

- `KeyUsage` — frozen dataclass: cumulative lifetime usage and credit limit for
  an OpenRouter API key, as returned by `OpenRouterKeyCostSource`

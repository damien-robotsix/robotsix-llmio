# robotsix_llmio deepseek

Direct-API **DeepSeek** transport — talks to the DeepSeek API
(`https://api.deepseek.com`) natively with a DeepSeek-issued key, bypassing
OpenRouter. DeepSeek's API is OpenAI-compatible, so the model is a pydantic-ai
`OpenAIChatModel` subclass (`DeepseekModel`) built on an `AsyncOpenAI` client
pointed at the DeepSeek base URL.

The layer keeps the DeepSeek-side behaviours — per-level thinking policy and
the `reasoning_content` round-trip — while dropping the OpenRouter-only
machinery: provider-routing kwargs, `usage.include`, and the server-side
`usage.cost` field. The direct API returns token counts but **no** cost, so
cost is computed client-side from configurable per-1M-token sticker prices.

## Model identifiers

Register a tier level with the `deepseek` prefix; the identifier is split on
the **first** hyphen, so everything after `deepseek-` is the DeepSeek model
id:

```yaml
default:
  level2:
    model: deepseek-deepseek-reasoner
```

`get_provider_for_identifier("deepseek-deepseek-reasoner")` returns a
`DeepseekProvider`; `new_model(model="deepseek-reasoner", level=2)` builds
the model.

## Authentication

The key is read from the `DEEPSEEK_API_KEY` environment variable (or passed
as `api_key=`). It is never stored in code or config — provision it via the
deploy environment. A missing key raises `DeepseekAPIError` at construction.
An optional `base_url` routes through a proxy, and `max_tokens` caps output
tokens on every model the provider builds.

## Thinking per level

DeepSeek selects thinking vs non-thinking purely by **model id** on the
direct API: `deepseek-chat` runs non-thinking and `deepseek-reasoner` runs
with thinking. `DeepseekProvider.new_model()` resolves the model id per
capability level — `level=1` maps to `deepseek-chat`, `level>=2` to
`deepseek-reasoner`, and `level=0` (a direct `new_model()` call) keeps the
configured model id unchanged. The thinking model also echoes
`reasoning_content` back on tool-call turns, so a reconstructed conversation
history (pre-seed, replayed `conversation_state`, pause/resume mid tool-loop)
is not rejected with HTTP 400.

## Cost

The direct API returns token counts — including DeepSeek's cache counters
(`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`) — but no dollar
cost. `DeepseekPricing` holds per-1M-token USD sticker prices (`input`,
`output`, `cache_read`); `DeepseekModel` records the cost client-side on the
active span via `core.cost.record_cost` with `provider="deepseek"`. Baked
defaults cover the current model ids:

| Model id | input | output | cache read |
| --- | --- | --- | --- |
| `deepseek-chat` | $0.27 | $1.10 | $0.07 |
| `deepseek-reasoner` | $0.55 | $2.19 | $0.14 |

Override or add prices via the `pricing` constructor argument — a
`DeepseekPricing` or a model-id → `{"input", "output", "cache_read"}`
mapping, e.g. routed from a tier's `provider_kwargs` — so a tariff change
needs no code change. Unpriced model ids record no cost.

## Exports

- `DeepseekProvider` — `LLMProvider` building direct-API DeepSeek models per level
- `DeepseekModel` — pydantic-ai `OpenAIChatModel` subclass with the `reasoning_content` round-trip and client-side cost
- `DeepseekPricing` / `ModelPrices` — per-1M-token sticker-price configuration
- `compute_cost` / `record_deepseek_cost` — client-side cost helpers
- `DeepseekAPIError` — configuration/auth error

## Manual smoke test

With a real key exported, one completion exercises the full path (request →
DeepSeek API → cost on span):

```python
import asyncio, os

from pydantic_ai import Agent

from robotsix_llmio.deepseek import DeepseekProvider

assert os.environ["DEEPSEEK_API_KEY"]
provider = DeepseekProvider()
model, http_client = provider.new_model(model="deepseek-chat", level=1)
try:
    print(asyncio.run(Agent(model).run("Say hi")).output)
finally:
    asyncio.run(http_client.aclose())
```

`new_model()` returns the timeout-configured `http_client` backing the model;
the caller owns closing it (as above). With Langfuse tracing configured, the
generation carries `gen_ai.usage.cost` and
`langfuse.observation.metadata.provider=deepseek`.
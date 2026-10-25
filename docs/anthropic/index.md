# robotsix_llmio anthropic

Direct first-party **Anthropic Messages API** transport
(`https://api.anthropic.com`). Unlike `claude_sdk` — which drives the local
`claude` CLI and consumes the Claude subscription seat — this transport
authenticates with an `ANTHROPIC_API_KEY` and spends first-party API credit.

Requires the `anthropic` extra:

```bash
pip install 'robotsix-llmio[anthropic]'
```

## Model identifiers

Register a tier level with the `anthropic` prefix; the identifier is split on
the **first** hyphen, so everything after `anthropic-` is the Anthropic model
id:

```yaml
default:
  level2:
    model: anthropic-claude-opus-5-5
```

`get_provider_for_identifier("anthropic-claude-opus-5-5")` returns an
`AnthropicProvider`; `new_model(model="claude-opus-5-5", level=2)` builds the
model.

## Authentication

The key is read from the `ANTHROPIC_API_KEY` environment variable (or passed
as `api_key=`). It is never stored in code or config — provision it via the
deploy environment. A missing key raises `AnthropicAPIError` at construction.

## Thinking / effort

Consistent with `claude_sdk`, no per-level thinking override is applied by
default: each model runs with the API default (adaptive thinking on current
Claude models). To tune reasoning depth per capability level, pass
`effort_by_level` (sent as `output_config.effort`):

```python
AnthropicProvider(effort_by_level={1: "low", 2: "high", 3: "xhigh"})
```

## Cost

The Messages API returns token counts (uncached input, output, cache read,
cache write) but no dollar cost. `AnthropicModel` computes the cost
client-side from per-1M-token sticker prices (`AnthropicPricing`) and records
it on the active span via `core.cost.record_cost` with
`provider="anthropic"`. Baked defaults cover the current Claude models
(Haiku, Sonnet, Opus, Fable); cache writes are priced at the 5-minute TTL rate
(1.25x input). Override or add prices via the `pricing` constructor argument
(a model-id → `{"input", "output", "cache_read", "cache_write"}` mapping,
e.g. from a tier's `provider_kwargs`). Unpriced models record no cost.

## Exports

- `AnthropicProvider` — `LLMProvider` building first-party API models
- `AnthropicModel` — pydantic-ai `AnthropicModel` that records cost
- `AnthropicPricing` / `ModelPrices` — sticker-price configuration
- `compute_cost` / `record_anthropic_cost` — client-side cost helpers
- `AnthropicAPIError` — configuration/auth error

## Manual smoke test

With a real key exported, one completion exercises the full path (request →
Messages API → cost on span):

```python
import asyncio, os
from pydantic_ai import Agent
from robotsix_llmio.core.factory import get_provider_for_identifier

assert os.environ["ANTHROPIC_API_KEY"]
provider = get_provider_for_identifier("anthropic-claude-haiku-5-5")
model, http_client = provider.new_model(model="claude-haiku-5-5", level=1)
print(asyncio.run(Agent(model).run("Say hi")).output)
```

With Langfuse tracing configured, the generation carries
`gen_ai.usage.cost` and `langfuse.observation.metadata.provider=anthropic`.

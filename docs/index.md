# robotsix-llmio

Provider-agnostic LLM I/O for [pydantic-ai](https://ai.pydantic.dev/) agents, with derived per-provider transport layers.

## Modules

- **[Core](core/index.md)** — Agent assembly, retry, cost recording, provider-cost reconciliation, tracing, and Langfuse integration.
- **[OpenRouter](openrouter/index.md)** — OpenRouter transport layer; base model-family-agnostic provider.
- **[OpenRouter DeepSeek](openrouter_deepseek/index.md)** — Derived DeepSeek provider that pins the generic OpenRouter transport to DeepSeek models.
- **[Claude SDK](claude_sdk/index.md)** — Claude Agent SDK transport; authenticates via the local `claude login` subscription.
- **[Config](config/index.md)** — Consumer-facing configuration layer: `TierConfig` schema, loader, transport aliases, and the `create_model` factory.

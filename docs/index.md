# robotsix-llmio

Provider-agnostic LLM I/O for [pydantic-ai](https://ai.pydantic.dev/) agents, with derived per-provider transport layers.

## Modules

- **[Core](core/index.md)** — Agent assembly, retry, cost recording, provider-cost reconciliation, tracing, and Langfuse integration.
- **[OpenRouter](openrouter/index.md)** — OpenRouter transport layer; base model-family-agnostic provider with a derived DeepSeek layer.
- **[Claude SDK](claude_sdk/index.md)** — Claude Agent SDK transport; authenticates via the local `claude login` subscription.
- **[Config](config/index.md)** — Consumer-facing configuration layer: `TierConfig` schema, loader, and the `create_model` factory.
- **[Knowledge](clients/knowledge/index.md)** — Direct-HTTP knowledge-store client with pydantic-ai tool integration.
- **[Refdocs](clients/refdocs/index.md)** — Direct-HTTP documentation search and retrieval.
- **[Self-Review](clients/self_review/index.md)** — Direct-HTTP agent activity review client.

# robotsix_llmio openrouter-deepseek (deprecated)

!!! warning "Deprecated"
    This module has been consolidated into `robotsix_llmio.openrouter`.
    Import from `robotsix_llmio.openrouter` instead:
    ```python
    from robotsix_llmio.openrouter import OpenRouterDeepseekProvider
    ```

Derived DeepSeek layer that pins the generic OpenRouter transport to
DeepSeek models on OpenRouter. Hard-codes the model names
(`deepseek/deepseek-v4-pro` → capable tier, `deepseek/deepseek-v4-flash`
→ cheap tier) and configures per-tier reasoning policy.

## Exports

- `OpenRouterDeepseekProvider` — provider that maps `level=2` to
  `"deepseek/deepseek-v4-pro"` (reasoning at `"xhigh"`) and `level=1`
  to `"deepseek/deepseek-v4-flash"` (reasoning disabled).
- `OpenRouterDeepseekModel` — model that injects
  `provider: {only: ["DeepSeek"], allow_fallbacks: false}` and per-level
  `reasoning` settings into every request.

# robotsix_llmio openrouter-deepseek

Derived DeepSeek layer that pins the generic OpenRouter transport to
DeepSeek models on OpenRouter. Hard-codes the model names
(`deepseek/deepseek-v4-pro` → capable tier, `deepseek/deepseek-v4-flash`
→ cheap tier) and configures per-tier reasoning policy.

## Exports

- `OpenRouterDeepseekProvider` — provider that maps `level=2` (legacy `Tier.DEFAULT`) to
  `"deepseek/deepseek-v4-pro"` (reasoning at `"xhigh"`) and `level=1` (legacy `Tier.CHEAP`)
  to `"deepseek/deepseek-v4-flash"` (reasoning disabled).
- `OpenRouterDeepseekModel` — model that injects
  `provider: {only: ["DeepSeek"], allow_fallbacks: false}` and per-level
  `reasoning` settings into every request.

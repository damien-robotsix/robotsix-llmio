# robotsix_llmio weekly pace governor

Spreads automated Claude (claude_sdk / subscription) usage across the week
so that weekly-limit headroom is preserved for interactive/other use.

## Exports

### PaceGovernor

- `PaceGovernor` — compares model-weighted Claude consumption this week
  (from the shared Langfuse project + in-process increments) against the
  elapsed week fraction. When consumption is ahead of pace, recommends
  falling back to DeepSeek; when behind pace, recommends Claude.

### Configuration

- `WeeklyPaceConfig` — pydantic model for all governor knobs (enabled,
  weekly_budget, week anchor, hysteresis margins, model weights, cache TTL,
  fail_open).

- `ModelWeightConfig` — per-model consumption weight multipliers (opus,
  sonnet, haiku) for calibrating against the interactive `/usage` meter.

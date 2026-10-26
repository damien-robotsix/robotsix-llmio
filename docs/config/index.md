# robotsix_llmio config

Consumer-facing configuration layer — `TierConfig` schema, loader,
provider prefixes, and the `create_model` factory.

The config package is the **entry point consumers should import from**
to obtain a working model/agent without naming a concrete provider class
or pulling in `claude-agent-sdk` directly.

```python
from robotsix_llmio.config import create_model

provider = create_model(level=2)
agent = provider.build_agent(
    level=2,
    system_prompt="You are a helpful assistant.",
)
```

## Exports

### Factory

- `create_model` — the single consumer-facing entry point.  Signature:

  ```python
  def create_model(
      *,
      level: int = 1,
      tier_config: TierConfig | None = None,
      **provider_kwargs: Any,
  ) -> LLMProvider:
  ```

  - **`level`** (1, 2, or 3) — selects the capability level: 1 cheap/frequent,
    2 workhorse, 3 frontier. Defaults to 1. Resolution honours the provider
    slot the failover tracker currently designates as active.
  - **`tier_config`** — optional `TierConfig` instance supplying custom
    slot/level bindings.  When `None`, a default is built from the baked
    constants.
  - **`**provider_kwargs`** — forwarded to the provider constructor (e.g.
    `api_key=...`).  These override any `provider_kwargs` from the tier
    config.

  Returns a fully-instantiated `LLMProvider`.  Raises `ValueError` for
  unknown provider prefixes or invalid levels, and `ImportError` when a
  required optional extra is not installed.

### Provider prefixes

An identifier is `<provider>-<model-name>`. The provider prefix (before the
first hyphen) selects the backend known to `get_provider_for_identifier`:

| Provider prefix | Backend |
|---|---|
| `anthropic` | Anthropic direct Messages API |
| `claudeSDK` | Claude Agent SDK |
| `openrouter` | OpenRouter (incl. DeepSeek models) |

### Schema & loader (tier configuration)

- `TierConfig` — pydantic model holding an ordered provider-slot `chain`
  (baked: direct Anthropic API → Claude SDK → OpenRouter; the legacy
  `default` / `fallback` keys still load, as the Claude SDK and OpenRouter
  slots of a two-slot chain), the `failover` policy, and the `vision` binding (the model
  answering `ask_image` questions for text-only transports)
- `ProviderSlotConfig` — one slot's binding of all three levels
  (`level1`, `level2`, `level3`)
- `FailoverConfig` — provider-failover policy (`failure_threshold`,
  `window_seconds`, the `reattempt_backoff_*` knobs, and the optional
  `max_request_cost_usd` / `cost_ratio_threshold` cost guards)
- `TierLevel` — `StrEnum` with `LEVEL1`, `LEVEL2`, `LEVEL3` selector values
- `TierLevelConfig` — pydantic model binding a single level to a
  provider-model identifier.  Each config carries:
  - **`model`** — the combined `provider-model` identifier.
  - **`provider_kwargs`** — extra constructor arguments forwarded to the provider.
  - **`max_tokens`** — optional output token cap (baked per level; see table below).
- `ANTHROPIC_LEVEL1..3`, `DEFAULT_LEVEL1..3`, `FALLBACK_LEVEL1..3` — the
  baked `TierLevelConfig` instances for chain positions 0 (direct Anthropic
  API), 1 (Claude SDK) and 2 (OpenRouter) respectively
- `TierConfigLoadError` — raised when tier configuration cannot be loaded
- `load_tier_config` — merges an explicit dict over the baked defaults into
  a validated `TierConfig` (there is no environment-variable overlay)
- `load_tier_config_file` / `save_tier_config_file` / `dump_tier_config` —
  read and write a persisted JSON tier config, always in the `chain` shape

## Consumer config shape

`TierConfig.chain` is an **ordered list of provider slots**. Position 0
serves normal operation; each later position is the next failover target,
tried in order. Every slot binds all three levels (`level1`, `level2`,
`level3`) — capability levels never fall back to one another, failover only
changes the *provider* serving a level. Any chain length >= 1 is supported.

`load_tier_config` accepts a dict mirroring the `TierConfig` shape; every
key is optional. Each `chain` entry merges field-by-field over the baked
slot at the same position (per-level dicts merge over the baked level), and
the length of the supplied list sets the chain length — entries beyond the
baked chain are taken as-is and must be complete slots:

```python
from robotsix_llmio.config import load_tier_config, create_model

cfg = load_tier_config(
    {
        "chain": [
            {},  # position 0: baked direct Anthropic API slot, unchanged
            {"level2": {"model": "claudeSDK-sonnet"}},  # position 1: Claude SDK
            {"level3": {"model": "openrouter-deepseek/deepseek-v4-pro"}},  # position 2
        ],
        "failover": {"failure_threshold": 3, "window_seconds": 900},
        "vision": {"model": "openrouter-google/gemini-2-flash"},
    }
)
provider = create_model(level=2, tier_config=cfg)
```

`TierConfig.slot()` / `for_level(level, slot=...)` accept a zero-based chain
index or the legacy names `"default"` (position 0) and `"fallback"`
(position 1).

Unknown keys fail validation loudly (`extra="forbid"`), including the
pre-rework flat `level1..level5` shape. Mixing `chain` with the legacy
`default` / `fallback` keys is rejected.

### Legacy two-slot configs

Configs written before the chain existed carry top-level `default` /
`fallback` slot dicts instead of `chain`. They still load: both
`load_tier_config` and `TierConfig` validation migrate them onto a
**two-slot chain** — `default` becomes position 0 and `fallback` position 1
— keeping their legacy meaning, so their baked bases are the Claude SDK slot
and the OpenRouter slot respectively (an omitted legacy key keeps that baked
slot). A legacy config therefore does *not* gain the direct Anthropic API
slot; switch it to the `chain` shape to opt in.

`load_tier_config_file(path)` reads a persisted JSON config. A missing file
is written fresh in the `chain` shape; a legacy `default` / `fallback` file
is migrated and re-persisted in the `chain` shape, so a deploy over an old
config neither crashloops nor silently changes provider order. The rewrite
is best-effort — on a read-only mount (e.g. a ConfigMap) the migrated config
is used in memory and the file is left untouched.

## Baked defaults

Three provider slots in chain order, three levels each:

1. **Direct Anthropic Messages API** (`anthropic-…`) — spends the Team
   subscription's monthly API dollar credit first.
2. **Claude SDK** (`claudeSDK-…`) — the Team subscription seat, used once
   the API credit is exhausted or the API is degraded.
3. **OpenRouter** (`openrouter-…`, DeepSeek) — the pay-as-you-go last
   resort.

| Slot / level | Identifier | `max_tokens` |
|--------------|------------|--------------|
| `chain[0].level1` | `anthropic-claude-haiku-5-5` | — |
| `chain[0].level2` | `anthropic-claude-opus-5-5` | — |
| `chain[0].level3` | `anthropic-claude-fable-5` | — |
| `chain[1].level1` | `claudeSDK-haiku` | — |
| `chain[1].level2` | `claudeSDK-opus` | — |
| `chain[1].level3` | `claudeSDK-claude-fable-5` | — |
| `chain[2].level1` | `openrouter-deepseek/deepseek-v4-flash-20260731` | 16 384 |
| `chain[2].level2` | `openrouter-deepseek/deepseek-v4-flash-20260731` (xhigh reasoning) | 65 536 |
| `chain[2].level3` | `openrouter-deepseek/deepseek-v4-pro-0813` | 131 072 |
| `vision` | `openrouter-deepseek/deepseek-v4-flash-vision-exp` | 8 192 |

The Anthropic API and Claude SDK levels carry no `max_tokens`; the Claude SDK levels deliberately so (the SDK has no
per-response cap; the value could only become an advisory `task_budget`).

## Failover down the chain

Failover between the slots is handled by
[`robotsix_llmio.core.failover`](../core/index.md) via
`call_with_failover` / `acall_with_failover`:

- **Per request** — a provider-shaped failure on the active slot retries the
  *same level* on the next slot, cascading until one succeeds. Each slot is
  tried at most once per request (at most `chain length − 1` switches, 2 for
  the baked chain); when the chain is exhausted the last slot's error
  propagates.
- **Sticky windows** — every slot except the last tracks its own health.
  After `failure_threshold` consecutive provider-shaped failures on a slot
  (a provider-wide exhaustion arms it immediately) the slot is skipped for
  `window_seconds` (default 15 minutes) and calls route straight to the next
  slot whose window is not armed. When an exhaustion error names its quota
  reset time, the window lasts until then instead.
- **Backoff** — when a window expires the next call probes the slot again;
  a still-broken slot re-arms a window grown by `reattempt_backoff_base`
  raised to the reattempt count (with `reattempt_backoff_jitter`), capped at
  `reattempt_backoff_cap_seconds` (default 4 hours). A success resets it.
- **Cost guards** — before each switch the candidate slot's estimated
  worst-case request cost is checked against `max_request_cost_usd` and
  against the slot the request *started on* times `cost_ratio_threshold`
  (comparing with the starting slot, not the previous hop, so a long cascade
  cannot ratchet the price up one tolerable step at a time). A refusal raises
  `FailoverCostGuardError` instead of switching. Both guards are off unless
  configured.

## Extra dependencies

The `claudeSDK` provider requires the `claude_sdk` extra:

```bash
pip install "robotsix-llmio[claude_sdk]"
```

The `openrouter` provider requires the `openrouter` extra:

```bash
pip install "robotsix-llmio[openrouter]"
```

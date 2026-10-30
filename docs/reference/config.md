# Configuration reference

Every setting `robotsix_llmio` reads, in one place. There are two kinds:

- **Tier configuration**: fields of the pydantic models in
  `src/robotsix_llmio/config/tier.py`. Consumers set them in the dict they
  pass to `load_tier_config` (`config/loader.py`) or in a file read by
  `load_tier_config_file`. Anything left out keeps its baked default. These
  fields have **no** environment-variable overlay.
- **Runtime environment variables**: read with `os.environ.get()` by the
  module named in the table. The library never loads a `.env` file itself.

The pages linked under "Details" explain the behaviour in full. This page
only lists each setting once.

## Tier configuration (`TierConfig`)

| Setting | Type | Default | Env var | Read by | Effect |
|---------|------|---------|---------|---------|--------|
| `chain` | `list[ProviderSlotConfig]` (≥ 1 slot) | 3 baked slots: direct Anthropic API, Claude SDK, OpenRouter (see [Baked defaults](../config/index.md#baked-defaults)) | — | `config/tier.py` (`TierConfig.for_level`), `config/factory.py` (`create_model`) | Ordered provider slots. Position 0 serves normal traffic. Later positions take over, in order, during failover. |
| `chain[i].level1` / `level2` / `level3` | `TierLevelConfig` | per slot, see [Baked defaults](../config/index.md#baked-defaults) | — | `config/tier.py` (`ProviderSlotConfig.for_level`) | Which model the slot uses for each capability level. |
| `default` / `fallback` (legacy) | `ProviderSlotConfig` | Claude SDK slot / OpenRouter slot | — | `config/loader.py` (`load_tier_config`), `config/tier.py` (`TierConfig` validation) | Old two-slot format, converted to `chain[0]` / `chain[1]` on load. Cannot be combined with `chain`. See [Legacy two-slot configs](../config/index.md#legacy-two-slot-configs). |
| `vision` | `TierLevelConfig` | `openrouter-deepseek/deepseek-v4-flash-vision-exp`, `max_tokens=8192` | — | `core/image_tool.py` | Model that answers `ask_image` tool calls. It is used directly and never goes through the chain or failover. |
| `failover` | `FailoverConfig` | see the next table | — | `core/failover.py` | Failover policy for the provider chain. |

### Per-level binding (`TierLevelConfig`)

| Setting | Type | Default | Env var | Read by | Effect |
|---------|------|---------|---------|---------|--------|
| `model` | `str` (required) | — | — | `core/identifier.py` (`parse_model_identifier`), `config/factory.py` | Provider and model name in the form `<provider>-<model-name>`. The prefix picks the backend (see [Provider prefixes](../config/index.md#provider-prefixes)). |
| `provider_kwargs` | `dict[str, Any]` | `{}` (baked OpenRouter levels set price ceilings and routing) | — | `config/factory.py` (`create_model`) | Keyword arguments passed to the provider constructor. Values passed directly to `create_model` override them. |
| `max_tokens` | `int \| None` | `None` (no cap); baked OpenRouter levels: 16 384 / 65 536 / 131 072 | — | `config/factory.py`, provider backends | Maximum output tokens. OpenRouter treats it as the `max_tokens` model setting; the Claude SDK treats it as a `task_budget`. |

### Failover policy (`FailoverConfig`)

| Setting | Type | Default | Env var | Read by | Effect |
|---------|------|---------|---------|---------|--------|
| `failure_threshold` | `int` (≥ 1) | `3` | — | `core/_failover_state.py` (`ProviderFailoverTracker`) | How many provider-related failures in a row a slot can have before the tracker skips it for a while (its window arms). Exhaustion arms the window right away. |
| `window_seconds` | `float` (> 0) | `900.0` (15 min) | — | `core/_failover_state.py` | How long the tracker skips a slot after its window arms. |
| `reattempt_backoff_base` | `float` (> 1) | `2.0` | — | `core/_failover_state.py` | Each time a slot that is still failing arms its window again, the window is multiplied by this factor. |
| `reattempt_backoff_cap_seconds` | `float` (> 0, ≥ `window_seconds`) | `14400.0` (4 h) | — | `core/_failover_state.py` | Maximum length of a backed-off window. |
| `reattempt_backoff_jitter` | `float` (0–1) | `0.5` | — | `core/_failover_state.py` | Random reduction applied to a backed-off window, as a fraction of its length. |
| `max_request_cost_usd` | `float \| None` (> 0) | `None` (off) | — | `core/failover.py` via `core/_cost_guard.py` | Blocks a switch to the next slot when that slot's estimated worst-case cost per request is above this limit. |
| `cost_ratio_threshold` | `float \| None` (> 0) | `None` (off) | — | `core/failover.py` via `core/_cost_guard.py` | Blocks a switch when the candidate slot would cost more than this multiple of the slot the request started on. |

Hard-coded failover limit (not configurable): each request switches slots at
most `len(chain) - 1` times (`_max_slot_switches` in `core/failover.py`).

Details: [Failover down the chain](../config/index.md#failover-down-the-chain),
[Failover runbook](../core/failover-runbook.md),
[Provider failover](../core/index.md#provider-failover).

## Runtime environment variables

| Variable | Type | Default | Read by | Effect |
|----------|------|---------|---------|--------|
| `OPENROUTER_API_KEY` | `str` | — (required for `openrouter-` models) | `openrouter/provider.py` | API key for `OpenRouterProvider`. A non-empty `api_key=` argument takes precedence. |
| `DEEPSEEK_API_KEY` | `str` | — (required for `deepseek` models) | `deepseek/provider.py` | API key for the direct-API `DeepseekProvider`. |
| `ANTHROPIC_API_KEY` | `str` | — (required for `anthropic-` models) | `anthropic/provider.py` | API key for the Anthropic Messages API provider. |
| `REFDOCS_API_KEY` | `str` | unset (no `Authorization` header) | `clients/refdocs/_settings.py` | Bearer token for refdocs. Used only when `RefdocsSettings.api_key` is `None`. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | `str` | unset (tracing off) | `core/tracing.py` (constants in `core/_otel.py`) | Langfuse trace and cost export runs only when both are set. |
| `LANGFUSE_BASE_URL` | `str` | `https://cloud.langfuse.com` | `core/tracing.py` | Langfuse endpoint. |
| `LANGFUSE_PROJECT_ID` | `str` | unset | `core/tracing.py` | Optional Langfuse project id. |
| `LOG_LEVEL` | `str` | `INFO` | `logging.py` | Logging level, unless one is passed explicitly. |
| `LOG_FORMAT` | `str` | `console` | `logging.py` | Logging format, unless one is passed explicitly. |

## Client settings (`RefdocsSettings`)

| Setting | Type | Default | Env var | Read by | Effect |
|---------|------|---------|---------|---------|--------|
| `base_url` | `str` | `http://localhost:9090` | — | `clients/refdocs/_settings.py` | Base URL of the refdocs REST API. |
| `api_key` | `str \| None` | `None` | `REFDOCS_API_KEY` (fallback) | `clients/refdocs/_settings.py` | Bearer token. |
| `request_timeout` | `float` | `30.0` | — | `clients/refdocs/_settings.py` | Timeout for each request, in seconds. |

Details: [Refdocs](../clients/refdocs/index.md#refdocssettings).

## API

::: robotsix_llmio.config

# robotsix_llmio core

Provider-agnostic LLM I/O base: agent assembly, retry, cost recording,
provider-cost reconciliation, tracing, and Langfuse integration.

## Exports

### Provider ABC & agent assembly

- `LLMProvider` — abstract base for every LLM provider; subclasses implement `new_model(*, model=None, level=0)`
- `AgentHandle` — wraps a pydantic-ai Agent with its httpx client, exposing `close()` for cleanup
- `build_agent` — assembles a pydantic-ai Agent from model, http_client, system_prompt, tools, and output_type; on the provider the public entry-point is `LLMProvider.build_agent(level=..., system_prompt=...)` where `level` is an integer 1–3

### Config-tier re-exports

- `DEFAULT_LEVEL1..3` — baked `TierLevelConfig` per level of the default slot (Claude SDK: haiku / opus / claude-fable-5)
- `FALLBACK_LEVEL1..3` — baked `TierLevelConfig` per level of the fallback slot (OpenRouter DeepSeek: flash / pro / pro)
- `TierConfig` — pydantic model holding two provider slots (`default`, `fallback`) plus the `failover` policy
- `ProviderSlotConfig` — one slot's binding of all three levels
- `FailoverConfig` — provider-failover policy (`failure_threshold`, `window_seconds`)
- `TierConfigLoadError` — raised when tier configuration cannot be loaded
- `TierLevel` — `StrEnum` with `LEVEL1` (→ `level=1`), `LEVEL2` (→ `level=2`), `LEVEL3` (→ `level=3`) selector values
- `TierLevelConfig` — pydantic model binding a single level's transport and model
- `create_model` — consumer-facing factory returning a configured `LLMProvider`
- `load_tier_config` — merges an explicit dict over the baked defaults into a validated `TierConfig`
### Agent runners

- `run_agent` — runs an `AgentHandle` under a trace span with bounded retry, always closing the handle
- `arun_agent` — async mirror of `run_agent`

### Identifier parsing

- `MalformedIdentifierError` — raised when a provider-model identifier string is malformed (e.g. unbalanced brackets, missing model part)
- `ParsedIdentifier` — `NamedTuple` holding the parsed components of a provider-model identifier: `provider` and `model_name`
- `parse_model_identifier` — parses a combined provider-model identifier (e.g. ``claudeSDK-opus`` or ``openrouter-deepseek/deepseek-v4-flash-latest``) into a `ParsedIdentifier`

### Factory

- `create_model` — preferred entry point: resolves a provider from level + transport + tier config, returns a configured `LLMProvider`
- `get_provider_for_identifier` — resolves and instantiates a provider from a combined provider-model identifier string (parsed via `parse_model_identifier`)

### Retry & transient errors

- `call_with_retry` — bounded retry on transient/rate-limit errors with optional fallback
- `call_with_retry_and_fallback` — retries locally then activates a fallback model on failure
- `is_rate_limited` — detects pydantic-ai `UsageLimitExceeded` (budget cap) exceptions
- `is_transient` — detects retryable infrastructure failures (httpx timeouts, 429/5xx, transport errors)
- `acall_with_retry` — async mirror of `call_with_retry`
- `acall_with_retry_and_fallback` — async mirror of `call_with_retry_and_fallback`

### Image questions

- `build_image_question_tool` — builds the async `ask_image(image_index, question)` tool over attached `(media_type, bytes)` images, answered by the tier config's `vision` binding; wired automatically by `build_agent(images=...)` on the OpenRouter (text-only) path; the Claude SDK transport serves images natively instead

### Provider failover

- `call_with_failover` — runs a callable at a fixed capability level with automatic provider-slot failover: a provider-shaped failure on the active slot retries the SAME level on the other slot; after `failure_threshold` consecutive default-slot failures (exhaustion immediately) all calls route to the fallback slot for `window_seconds`, then return to the default
- `acall_with_failover` — async mirror of `call_with_failover`
- `ProviderFailoverTracker` — process-wide tracker holding failover state (singleton via `get_failover_tracker`)
- `get_failover_status` / `FailoverStatus` — snapshot of the failover state, shaped for consumer status endpoints and UIs
- `is_provider_shaped` — classifies whether an exception points at the provider (failover-eligible) or the task
- `reset_failover_tracker` — resets the singleton (test teardown)

### Cost recording

- `CostLogSource` — protocol for reading logged cost; single method `fetch_logged_cost(window) -> LoggedCost`
- `CostRecord` — dataclass for one logged cost unit: `id`, `cost`, `timestamp`, optional `session_id` and `name`
- `CostWindow` — dataclass with `start`/`end` datetimes bounding a cost query (start inclusive, end exclusive)
- `LoggedCost` — dataclass aggregating a logged-cost query: `total_cost`, `record_count`, `records`

### Provider cost reconciliation

- `DEFAULT_TOLERANCE` — default tolerance (1.0) for cost reconciliation
- `Discrepancy` — dataclass with logged vs. provider totals, delta, and `within_tolerance` flag
- `ProviderCost` — dataclass for provider-billed cost: `total_cost`, `breakdown`, `request_count`
- `ProviderCostSource` — protocol for reading provider-billed cost; `fetch_provider_cost(window) -> ProviderCost`
- `reconcile` — compares logged vs. provider-billed cost, returning a `Discrepancy`

### Tracing & Langfuse

- `TraceSpan` — handle to a root span with `trace_id`, `set_input()`, `set_output()`
- `active_routing_key` — returns the active Langfuse public key from context, or `None`
- `current_session` — returns the session id active in the current context, or `None`
- `flush_tracing` — force-flushes pending OTel spans; no-op without OTel
- `get_recording_span` — returns the current OTel span if recording, else `None`
- `get_tracer` — returns an OTel tracer, or `None` when OpenTelemetry is not installed
- `install_signal_handlers` — installs SIGTERM/SIGINT handlers that flush tracing then exit
- `langfuse_project` — context manager routing spans to a registered Langfuse project (multi-tenant)
- `langfuse_session` — context manager grouping spans under a `session_id` in Langfuse
- `langfuse_trace_url` — builds the Langfuse web-UI URL for a trace ID
- `make_session_id` — returns a unique session id (`<kind>-<hex>`) for use with `langfuse_session`
- `setup_langfuse_tracing` — registers a Langfuse project and starts OTel instrumentation
- `start_span` — context manager starting a named span; yields `None` when OTel is absent
- `start_trace` — context manager opening a root span (trace), optionally under a session and project

### Langfuse cost log

- `LangfuseCostLogSource` — concrete `CostLogSource` fetching logged cost from Langfuse's REST API
- `LangfuseReadClient` — low-level Langfuse REST client for reading trace and session data

### HTTP

- `timeout_http_client` — returns a fresh `httpx.AsyncClient` with a hard per-request timeout

### Text utilities

- `html_to_text` — strips HTML markup down to whitespace-collapsed plaintext

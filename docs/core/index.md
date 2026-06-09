# robotsix_llmio core

Provider-agnostic LLM I/O base: agent assembly, retry, cost recording,
provider-cost reconciliation, tracing, and Langfuse integration.

## Exports

### Provider ABC & agent assembly

- `LLMProvider` — abstract base for every LLM provider; subclasses implement `new_model(tier)`
- `Tier` — `StrEnum` with `DEFAULT` (capable) and `CHEAP` (fast/cheap) model-selector values
- `AgentHandle` — wraps a pydantic-ai Agent with its httpx client, exposing `close()` for cleanup
- `build_agent` — assembles a pydantic-ai Agent from model, http_client, system_prompt, tools, and output_type

### Retry & transient errors

- `call_with_retry` — bounded retry on transient/rate-limit errors with optional fallback
- `call_with_retry_and_fallback` — retries locally then activates a fallback model on failure
- `is_rate_limited` — detects pydantic-ai `UsageLimitExceeded` (budget cap) exceptions
- `is_transient` — detects retryable infrastructure failures (httpx timeouts, 429/5xx, transport errors)

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

### HTTP

- `timeout_http_client` — returns a fresh `httpx.AsyncClient` with a hard per-request timeout

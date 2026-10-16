# Test fixtures & factories

This is the map of shared test helpers for the `robotsix-llmio` suite: the
pytest **fixtures** (defined in the per-area `conftest.py` files) and the
mock **factories** (centralized under `tests/fixtures/`). Reach for an
existing helper before writing a new inline mock — duplicated transport
helpers and ad-hoc `SimpleNamespace`/`httpx.Response` builders are a
recurring source of drift this document exists to prevent.

See also:

- [`AGENT.md`](../AGENT.md) — the normative testing rules (notably: use
  `install_transport` with an explicit `module=`; never define a private
  duplicate).
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) §3 — how to run the suite.
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) "Test coverage strategy" — the
  per-module test-file listing.

## Layout

| Path | Kind | Owns |
|------|------|------|
| `tests/conftest.py` | fixtures + helpers | process-wide isolation, `CostWindow`/transport helpers shared by every cost module |
| `tests/core/conftest.py` | helpers | the `install_*` transport installers, Langfuse adapter/window builders, retry-sleep stubs, live Langfuse helpers |
| `tests/config/conftest.py` | fixtures | config-layer env isolation and provider-factory mock |
| `tests/claude_sdk/conftest.py` | helpers | the fake Claude Agent SDK module and handle builders |
| `tests/fixtures/responses.py` | factories | `httpx` response / `MockTransport` handler builders |
| `tests/fixtures/builders.py` | factories | `SimpleNamespace` / mock-object builders |

Pytest auto-discovers each `conftest.py`. The functions prefixed with `_`
are **not** fixtures — they are plain helpers that test modules import and
call explicitly.

## `tests/conftest.py` (root — applies to the whole suite)

| Name | Kind | Scope | Purpose |
|------|------|-------|---------|
| `_reset_failover_tracker` | fixture | function, **autouse** | Resets the process-wide failover singleton before and after every test so failover state never leaks between tests. |
| `_window(start, end)` | helper | — | Builds a `CostWindow` from two ISO timestamp strings (UTC). |
| `_mock_client_factory(monkeypatch, module, handler)` | helper | — | Patches `module.httpx.Client` to use a `MockTransport(handler)`, capturing the real `Client` first to avoid recursion. |

## `tests/core/conftest.py` (transport installers & cost helpers)

These are the canonical transport installers. **Always** pass an explicit
`module=` pointing at the module under test (per `AGENT.md`).

| Name | Kind | Purpose |
|------|------|---------|
| `install_transport(monkeypatch, handler, module=langfuse_cost_module)` | helper | Patches the **sync** `httpx.Client` on `module`; returns a list that captures every issued `httpx.Request`. Default module is the Langfuse cost source. |
| `install_timeout_transport(monkeypatch, handler, module)` | helper | Patches `module.timeout_http_client` to return an `AsyncClient(transport=MockTransport(handler))`. Used by the refdocs client tests. |
| `install_async_transport(monkeypatch, handler, module)` | helper | Patches the **async** `httpx.AsyncClient`; returns the request-capture list. |
| `make_window()` | helper | Fixed `CostWindow` (2026-06-03 10:00–11:00 UTC). |
| `make_adapter()` | helper | A `LangfuseCostLogSource` with dummy public/secret keys and base URL. |
| `_HTTPErr` | helper | Lightweight HTTP-error stand-in for retry tests. |
| `_noop_sleep` / `_anoop_sleep` | helper | Sync/async no-op sleeps that skip real backoff delays in retry tests. |
| `_langfuse_creds`, `_require`, `_langfuse_traces`, `_langfuse_get` | helper | Env-gated helpers for the opt-in **live** Langfuse tests. |

### Usage

```python
import robotsix_llmio.core.langfuse_cost as langfuse_cost_module
from tests.core.conftest import install_transport
from tests.fixtures import make_langfuse_traces_response, json_handler

def test_fetch(monkeypatch):
    page = make_langfuse_traces_response([{"id": "t1", "totalCost": 0.5}])
    requests = install_transport(
        monkeypatch, json_handler(page), module=langfuse_cost_module
    )
    ...  # assert against `requests`
```

## `tests/config/conftest.py` (config-layer isolation)

| Name | Kind | Scope | Purpose |
|------|------|-------|---------|
| `clean_env` | fixture | function, **autouse** | Removes all `LLMIO_*` environment variables before each test so loader tests start from a clean configuration environment. |
| `mock_get_provider_for_identifier` | fixture | function | Yields a `MagicMock` that patches the factory's `get_provider_for_identifier`, letting config tests assert how the factory is called without constructing real providers. |

## `tests/claude_sdk/conftest.py` (fake Claude Agent SDK)

The Claude SDK is an optional runtime dependency, so these helpers stub it
out entirely. All are helpers (imported and called), not fixtures.

| Name | Kind | Purpose |
|------|------|---------|
| `_fake_sdk_module()` | helper | Returns a `SimpleNamespace` fake of the SDK with stub `Tool`, `TextBlock`, `AssistantMessage`, `ResultMessage`, `ClaudeAgentOptions`, `HookMatcher`, `create_sdk_mcp_server`, plus `_tool_regs` / `_server_calls` introspection lists. |
| `_install_fake_sdk(monkeypatch)` | helper | Installs the fake module into `sys.modules`. |
| `_echo_sync(text)` | helper | A trivial test tool that echoes its input. |
| `_capturing_query(fake, captured)` | helper | Fake SDK async generator yielding an `AssistantMessage` then `ResultMessage`, recording the prompt into `captured`. |
| `_tool_handle()` | helper | Builds a `ClaudeSDKProvider` agent handle at level 1 (Haiku). |
| `_make_minimal_handle(...)` | helper | Constructs a stub `_SdkToolAgentHandle` for unit tests. |
| `_Verdict`, `_AltVerdict` | helper | Pydantic models for structured-output tests. |
| `_FakeThinkingBlock` | helper | Fake thinking block (matched by class name in the reasoning test). |
| `_CLAUDE_SLOT`, `_HAIKU_AT_LEVEL1`, `_OPUS_AT_LEVEL2` | constants | A `ProviderSlotConfig` and two `TierConfig`s reused across handle builders. |

## `tests/fixtures/responses.py` (HTTP response factories)

Build `httpx.Response` objects and `MockTransport` handlers whose JSON
shapes mirror exactly what the production parsers read. Pair them with the
`install_*` installers above.

| Factory | Returns | Mirrors |
|---------|---------|---------|
| `make_langfuse_trace(...)` | `dict` | A trace record (`id`, `timestamp`, `totalCost`, `sessionId`, `name`) read by `LangfuseCostLogSource._to_record`. |
| `make_langfuse_observation(...)` | `dict` | An observation record (`id`, `startTime`, `traceId`, `name`, cost fields) read by `_observation_to_record` / `_observation_cost`. |
| `make_langfuse_page(records, total_pages=None)` | `dict` | The list-API envelope `{"data": [...], "meta": {"totalPages": N}}`. |
| `make_langfuse_traces_response(records, ...)` | `httpx.Response` | A full Langfuse list-API response. |
| `make_knowledge_result(...)` | `dict` | A knowledge search result (`id`/`title`/`snippet`/`score`). |
| `make_knowledge_client_response(results, ...)` | `httpx.Response` | `{"results": [...]}` for the knowledge client. |
| `make_refdocs_result(...)` | `dict` | A refdocs result (at least `path` and `title`). |
| `make_refdocs_response(results, ...)` | `httpx.Response` | `{"results": [...]}` for the refdocs client. |
| `make_error_response(status, message)` | `httpx.Response` | An `{"error": message}` body at the given status. |
| `json_handler(response)` | handler | A `MockTransport` handler that always returns `response`. |
| `paged_handler(pages)` | handler | A handler that returns each response in turn (pagination tests). |

## `tests/fixtures/builders.py` (mock-object factories)

| Factory | Returns | Mirrors |
|---------|---------|---------|
| `make_openrouter_usage(cost=..., model_extra=...)` | `SimpleNamespace` | An OpenRouter `usage` namespace; only the attributes passed are present. |
| `make_openrouter_response(usage=.../cost=.../model_extra=...)` | `SimpleNamespace` | The pydantic-ai response shape the cost helper reads (`response.usage.cost` / `response.usage.model_extra`). Pass `usage=None` to model a response with no usage block. |

### Usage

```python
from tests.fixtures import make_openrouter_response

def test_cost_extraction():
    resp = make_openrouter_response(cost=0.0123, model_extra=None)
    assert resp.usage.cost == 0.0123

    resp_no_usage = make_openrouter_response(usage=None)
    assert resp_no_usage.usage is None
```

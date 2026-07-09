# Refdocs

Direct HTTP access for documentation search and retrieval.

Replaces the agent-comm broker intermediary with direct REST calls to
the refdocs API.

## Quick start

```python
from robotsix_llmio.clients.refdocs import RefdocsSettings, build_refdocs_tools

settings = RefdocsSettings(base_url="http://refdocs:9090")
tools = build_refdocs_tools(settings)

# Pass tools to a pydantic-ai Agent
agent = Agent(model=..., tools=tools)
```

## API reference

### `RefdocsSettings`

Dataclass holding refdocs client configuration:

- `base_url` — Base URL of the refdocs REST API (default `http://localhost:9090`).
- `api_key` — Optional Bearer token. Falls back to the `REFDOCS_API_KEY`
  environment variable.
- `request_timeout` — Per-request timeout in seconds (default `30.0`).

### `build_refdocs_tools(settings)`

Builds pydantic-ai `Tool` objects wrapping direct-HTTP refdocs access.
Returns two tools:

- `search_refdocs(query)` — search the documentation index.
- `get_refdocs(path)` — retrieve full content of a documentation page.

### `AsyncRefdocsClient`

Low-level async HTTP client for the refdocs REST API. Useful for
programmatic access outside of an agent context.

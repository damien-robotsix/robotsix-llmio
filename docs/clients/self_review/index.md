# robotsix_llmio self_review

Direct-HTTP self-review client — no agent-comm broker intermediary.

## Overview

The self_review module provides direct HTTP access to a self-review /
recent-activity REST API, replacing the previous agent-comm broker
transport. It exposes:

- `SelfReviewClient` — async REST client for listing and retrieving
  agent activity
- `build_recent_activity_tools(client, conversation_store)` — wraps a
  client into two pydantic-ai-compatible async tool functions
- `SelfReviewClientError` — exception type for all self-review errors

## Quick start

```python
from robotsix_llmio.clients.self_review import (
    SelfReviewClient,
    build_recent_activity_tools,
)

client = SelfReviewClient(
    base_url="http://self-review:8000/api/v1",
    api_key="...",  # optional
)

# Direct use
activities = await client.list_activity(limit=10)
activity = await client.get_activity("act-42")

# As pydantic-ai tools
tools = build_recent_activity_tools(client)
# tools[0] = list_recent_activity(limit: int = 10) -> str
# tools[1] = get_recent_activity_detail(activity_id: str) -> str
```

## API reference

### SelfReviewClient

`SelfReviewClient(*, base_url="http://localhost:8000/api/v1", api_key=None)`

Async HTTP client for a self-review / recent-activity REST API.

**Methods:**

- `list_activity(limit: int = 10) -> list[dict]` — list recent agent
  activities. Returns a list of activity dicts with keys `id`, `agent`,
  `action`, `summary`, and `timestamp`.
- `get_activity(activity_id: str) -> dict` — retrieve a single activity
  by ID. Returns the activity with keys `id`, `agent`, `action`,
  `summary`, `timestamp`, and `detail`.

Both methods raise `SelfReviewClientError` on HTTP errors (4xx/5xx) or
network failures.

### build_recent_activity_tools

`build_recent_activity_tools(client: SelfReviewClient, conversation_store=None) -> list`

Returns two async tool functions suitable for passing to a pydantic-ai
`Agent`:

| Tool | Signature | Description |
|------|-----------|-------------|
| `list_recent_activity` | `(limit: int = 10) -> str` | List recent activities; returns formatted results |
| `get_recent_activity_detail` | `(activity_id: str) -> str` | Retrieve an activity; returns full detail |

Tool errors are returned as plain-text messages so the agent can interpret
them directly (no unhandled exceptions from within a tool invocation).

The `conversation_store` parameter is reserved for future use (e.g.
cross-referencing activity against conversation history) and is currently
accepted but unused.

### SelfReviewClientError

`SelfReviewClientError` extends `RobotsixLLMIOError`. Raised by
`SelfReviewClient` methods on any HTTP or network failure. Catch as
`RobotsixLLMIOError` for uniform error handling across the library.

## Endpoints

The client expects the following REST endpoints on the configured `base_url`:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/activity?limit=...` | List recent agent activities |
| `GET` | `/activity/{activity_id}` | Retrieve a single activity |

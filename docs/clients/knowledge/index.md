# robotsix_llmio knowledge

Direct-HTTP knowledge-store client — no agent-comm broker intermediary.

## Overview

The knowledge module provides direct HTTP access to a knowledge-store REST
API, replacing the previous agent-comm broker transport. It exposes:

- `KnowledgeClient` — async REST client for search and document retrieval
- `build_knowledge_tools(client)` — wraps a client into two pydantic-ai-
  compatible async tool functions
- `KnowledgeClientError` — exception type for all knowledge-store errors

## Quick start

```python
from robotsix_llmio.clients.knowledge import KnowledgeClient, build_knowledge_tools

client = KnowledgeClient(
    base_url="http://knowledge-store:8000/api/v1",
    api_key="...",      # optional
)

# Direct use
results = await client.search("how to configure tiers")
doc = await client.get_document("doc-42")

# As pydantic-ai tools
tools = build_knowledge_tools(client)
# tools[0] = search_knowledge(query: str) -> str
# tools[1] = get_knowledge_document(doc_id: str) -> str
```

## API reference

### KnowledgeClient

`KnowledgeClient(*, base_url="http://localhost:8000/api/v1", api_key=None)`

Async HTTP client for a knowledge-store REST API.

**Methods:**

- `search(query: str, top_k: int = 10) -> list[dict]` — full-text search.
  Returns a list of result dicts with keys `id`, `title`, `snippet`, and
  `score`.
- `get_document(doc_id: str) -> dict` — retrieve a document by ID. Returns
  the document with keys `id`, `title`, `content`, and `metadata`.

Both methods raise `KnowledgeClientError` on HTTP errors (4xx/5xx) or
network failures.

### build_knowledge_tools

`build_knowledge_tools(client: KnowledgeClient) -> list`

Returns two async tool functions suitable for passing to a pydantic-ai
`Agent`:

| Tool | Signature | Description |
|------|-----------|-------------|
| `search_knowledge` | `(query: str) -> str` | Full-text search; returns formatted results |
| `get_knowledge_document` | `(doc_id: str) -> str` | Retrieve a document; returns title + content |

Tool errors are returned as plain-text messages so the agent can interpret
them directly (no unhandled exceptions from within a tool invocation).

### KnowledgeClientError

`KnowledgeClientError` extends `RobotsixLLMIOError`. Raised by
`KnowledgeClient` methods on any HTTP or network failure. Catch as
`RobotsixLLMIOError` for uniform error handling across the library.

## Endpoints

The client expects the following REST endpoints on the configured `base_url`:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/search?q=...&top_k=...` | Full-text search |
| `GET` | `/documents/{doc_id}` | Retrieve a single document |

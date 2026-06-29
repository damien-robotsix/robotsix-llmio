"""Async HTTP client for a self-review REST API and pydantic-ai tool adapter.

``SelfReviewClient`` makes direct HTTP calls to a self-review / recent-activity
REST API (no agent-comm broker). ``build_recent_activity_tools(client,
conversation_store)`` wraps it into pydantic-ai-compatible async tool
functions so an agent can inspect recent agent activity.
"""

from __future__ import annotations

from typing import Any

from robotsix_llmio.core.http import timeout_http_client
from robotsix_llmio.self_review import SelfReviewClientError

_DEFAULT_BASE_URL = "http://localhost:8000/api/v1"


class SelfReviewClient:
    """Async HTTP client for a self-review / recent-activity REST API.

    Connects directly to the self-review HTTP API — no agent-comm
    broker intermediary. Creates a fresh, timeout-bounded
    ``httpx.AsyncClient`` per request (stateless per-call pattern).

    Parameters
    ----------
    base_url:
        Root URL of the self-review API (e.g. ``http://self-review:8000/api/v1``).
    api_key:
        Optional bearer token sent as ``Authorization: Bearer <api_key>``.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def list_activity(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent agent activities.

        Parameters
        ----------
        limit:
            Maximum number of activity entries to return (default 10).

        Returns
        -------
        list[dict]
            A list of activity dicts, each with keys ``id``, ``agent``,
            ``action``, ``summary``, and ``timestamp``. An empty list
            means no recent activity.
        """
        params: dict[str, str | int] = {"limit": limit}
        data = await self._get("/activity", params=params)
        results: list[dict[str, Any]] = data.get("activities", [])
        return results

    async def get_activity(self, activity_id: str) -> dict[str, Any]:
        """Retrieve a single activity entry by its identifier.

        Parameters
        ----------
        activity_id:
            The activity identifier.

        Returns
        -------
        dict
            The activity with keys ``id``, ``agent``, ``action``,
            ``summary``, ``timestamp``, and ``detail``.

        Raises
        ------
        SelfReviewClientError
            If the activity is not found (HTTP 404) or the request fails.
        """
        return await self._get(f"/activity/{activity_id}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _get(
        self,
        path: str,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        """Send a GET to *path* and return the JSON response body.

        Raises ``SelfReviewClientError`` on any non-2xx response or
        network failure.
        """
        url = f"{self._base_url}{path}"
        headers: dict[str, str] = {}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with timeout_http_client() as client:
                resp = await client.get(url, headers=headers, params=params)
        except Exception as exc:
            raise SelfReviewClientError(
                f"Self-review request to {path} failed: {exc}"
            ) from exc

        if not (200 <= resp.status_code < 300):
            raise SelfReviewClientError(
                f"Self-review {path} returned HTTP {resp.status_code}"
            )
        return resp.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------- #
# pydantic-ai tool adapter
# ---------------------------------------------------------------------- #


def build_recent_activity_tools(
    client: SelfReviewClient,
    conversation_store: Any | None = None,
) -> list[Any]:
    """Build pydantic-ai-compatible async tool functions from a
    :class:`SelfReviewClient`.

    Returns two async tool functions:

    * ``list_recent_activity(limit: int = 10) -> str`` — list recent
      agent activities, returns formatted results.
    * ``get_recent_activity_detail(activity_id: str) -> str`` — retrieve a
      single activity by ID, returns the activity detail.

    Each tool handles its own errors and returns a plain-text message so
    the agent can interpret the result directly.

    Parameters
    ----------
    client:
        The ``SelfReviewClient`` to wrap.
    conversation_store:
        Reserved for future use (e.g. cross-referencing activity against
        conversation history). Currently accepted but unused.
    """

    async def list_recent_activity(limit: int = 10) -> str:
        """List recent agent activities.

        Use this to see what other agents have been working on recently.
        Returns formatted activity entries with IDs, agents, actions,
        summaries, and timestamps.
        """
        try:
            activities = await client.list_activity(limit=limit)
        except SelfReviewClientError as exc:
            return f"List activity failed: {exc}"

        if not activities:
            return "No recent activity found."

        lines: list[str] = []
        for a in activities:
            activity_id = a.get("id", "?")
            agent = a.get("agent", "unknown")
            action = a.get("action", "unknown")
            summary = a.get("summary", "")
            timestamp = a.get("timestamp", "")
            lines.append(
                f"- [{activity_id}] {agent} — {action} ({timestamp})\n  {summary}"
            )
        return "\n".join(lines)

    async def get_recent_activity_detail(activity_id: str) -> str:
        """Retrieve a specific activity entry by ID.

        Use this after ``list_recent_activity`` to read the full detail
        of a particular activity. Returns the activity detail.
        """
        try:
            activity = await client.get_activity(activity_id)
        except SelfReviewClientError as exc:
            return f"Failed to retrieve activity {activity_id}: {exc}"

        agent = activity.get("agent", "unknown")
        action = activity.get("action", "unknown")
        summary = activity.get("summary", "")
        detail = activity.get("detail", "")
        timestamp = activity.get("timestamp", "")

        parts = [
            f"# Activity {activity_id}",
            f"Agent: {agent}",
            f"Action: {action}",
            f"Timestamp: {timestamp}",
            f"Summary: {summary}",
        ]
        if detail:
            parts.append(f"\n{detail}")
        return "\n".join(parts)

    return [list_recent_activity, get_recent_activity_detail]

"""Refdocs settings — configuration for the direct-HTTP refdocs client."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_DEFAULT_BASE_URL = "http://localhost:9090"

#: Environment variable name for the refdocs API key.
_ENV_REFDOCS_API_KEY = "REFDOCS_API_KEY"


@dataclass(frozen=True)
class RefdocsSettings:
    """Configuration for the refdocs HTTP client.

    Attributes
    ----------
    base_url:
        Base URL of the refdocs REST API. Defaults to
        ``http://localhost:9090``.
    api_key:
        Bearer-token for authentication. When ``None`` (default), falls
        back to the ``REFDOCS_API_KEY`` environment variable. If still
        ``None`` after the fallback, requests are sent without an
        ``Authorization`` header.
    request_timeout:
        Per-request timeout in seconds. Defaults to ``30.0``.

    """

    base_url: str = field(default=_DEFAULT_BASE_URL)
    api_key: str | None = field(default=None)
    request_timeout: float = field(default=30.0)

    @property
    def resolved_api_key(self) -> str | None:
        """Return the explicit *api_key* or the ``REFDOCS_API_KEY`` env var."""
        return self.api_key or os.environ.get(_ENV_REFDOCS_API_KEY)

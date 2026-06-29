"""Shared refdocs transport constants.

Single source of truth for the refdocs REST endpoint, imported by both the
async client and the tool factory so the URL is maintained in one place.
"""

from __future__ import annotations

_DEFAULT_BASE_URL = "http://localhost:9090"

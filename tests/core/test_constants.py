"""Pin the shared HTTP client timeout constant.

A typo or drift in the centralized timeout would silently change the
behaviour of every synchronous ``httpx.Client`` REST fetch, so pin its
literal value and confirm both consumers reference the shared constant.
"""

from __future__ import annotations

from robotsix_llmio.core import constants, langfuse_client
from robotsix_llmio.openrouter import provider_cost


def test_http_client_timeout_value() -> None:
    assert constants.HTTP_CLIENT_TIMEOUT == 20.0


def test_consumers_reference_shared_constant() -> None:
    assert langfuse_client.HTTP_CLIENT_TIMEOUT is constants.HTTP_CLIENT_TIMEOUT
    assert provider_cost.HTTP_CLIENT_TIMEOUT is constants.HTTP_CLIENT_TIMEOUT

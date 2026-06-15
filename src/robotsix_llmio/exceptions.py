"""Base exception hierarchy for robotsix-llmio."""

from __future__ import annotations


class RobotsixLLMIOError(Exception):
    """Base exception for all robotsix-llmio errors.

    All custom exceptions raised by this library (transport failures,
    configuration errors, provider-specific errors) inherit from this
    class so callers can catch them with a single ``except`` clause.
    """

"""OpenRouter-specific transient signatures, layered on the core set."""

from __future__ import annotations

from ..core import retry as _core_retry


def is_openrouter_upstream_error(exc: BaseException) -> bool:
    """Recognise OpenRouter's ``finish_reason='error'`` upstream-failure
    signature.

    When the provider behind OpenRouter errors mid-stream, OpenRouter returns
    a completion with ``finish_reason: "error"``. The OpenAI SDK then raises a
    pydantic ``ValidationError`` because ``"error"`` isn't in its
    ``finish_reason`` literal set. That's an upstream hiccup, not a bug in the
    prompt/schema — a re-run almost always succeeds, so ride it out.

    Matched by the exception type name (``ValidationError``) plus the
    distinctive ``finish_reason`` + ``'error'`` markers, so it does NOT catch
    genuine structured-output validation failures (those don't mention
    ``finish_reason``).
    """
    if type(exc).__name__ != "ValidationError":
        return False
    msg = str(exc)
    return "finish_reason" in msg and "'error'" in msg


def is_openrouter_upstream_payment_error(exc: BaseException) -> bool:
    """Recognise a 402 raised by the *upstream provider*, not by our account.

    OpenRouter reports two very different things as HTTP 402:

    * our own OpenRouter credits are exhausted — a real billing failure that
      must surface immediately, since retrying cannot help; and
    * the upstream provider OpenRouter routed to has no balance of its own. The
      body then carries ``"Provider returned error"`` with
      ``metadata.provider_name`` set and ``metadata.is_byok: False`` — meaning
      it is *OpenRouter's* account with that provider, nothing to do with ours.

    Only the second is transient: a retry lets OpenRouter route to one of the
    other providers serving the same model. Observed 2026-07-29, when the
    DeepSeek upstream ran dry and every ``deepseek/*`` request 402'd while ~17
    other providers were healthy.

    Matched on the ``provider_name`` + ``is_byok`` markers rather than the
    status alone, so a genuine "your account is out of credits" 402 (which
    carries neither) still fails fast. The body reaches us either as a Python
    dict repr (``'is_byok': False``, via ``ModelHTTPError``) or as raw JSON
    (``"is_byok":false``), so quotes/spacing are normalised before matching.
    """
    if _core_retry._status(exc) != 402:
        return False
    normalised = str(exc).replace('"', "").replace("'", "").replace(" ", "").lower()
    return "provider_name" in normalised and "is_byok:false" in normalised


def is_deepseek_reasoning_400(exc: BaseException) -> bool:
    """Recognise DeepSeek's reasoning-content 400.

    DeepSeek's capable tier runs in thinking mode and rejects assistant
    ``tool_calls`` turns that lack ``reasoning_content`` with HTTP 400
    (``"The reasoning_content in the thinking mode must be passed back to
    the API."``).  That is an infrastructure hiccup — the layer already
    injects reasoning content, so a re-run almost always succeeds.

    Matched on status 400 plus the distinctive ``reasoning_content`` +
    ``thinking mode`` markers, so it does NOT catch genuine schema/request
    400s.
    """
    if _core_retry._status(exc) != 400:
        return False
    msg = str(exc).lower()
    return "reasoning_content" in msg and "thinking mode" in msg


def is_openrouter_transient(exc: BaseException) -> bool:
    """Core transient set OR an OpenRouter upstream-failure signature (mid-stream
    provider error, upstream provider's 402, or DeepSeek reasoning-400),
    walking the cause/context chain for the latter."""
    if _core_retry.is_transient(exc):
        return True
    for cur in _core_retry._walk_cause_chain(exc):
        if (
            is_openrouter_upstream_error(cur)
            or is_openrouter_upstream_payment_error(cur)
            or is_deepseek_reasoning_400(cur)
        ):
            return True
    return False

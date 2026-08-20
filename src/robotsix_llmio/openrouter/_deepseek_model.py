"""DeepSeek-on-OpenRouter model — provider routing + per-tier reasoning policy +
reasoning round-trip.

Extends the OpenRouter transport model with DeepSeek's thinking-mode quirks:
- *prefer* the DeepSeek upstream provider (warms the per-provider prompt cache
  and keeps routing stable) while allowing OpenRouter to fall back to another
  provider, under a price ceiling — see "Why preference, not a hard pin" below;
- inject a per-tier reasoning policy into the request (set by the provider:
  ``{"effort": "xhigh"}`` for the capable tier, ``{"enabled": False}`` for the
  cheap tier);
- carry ``reasoning_content`` on every assistant tool-call turn so a thinking-
  mode request is accepted.

Why the round-trip is needed: DeepSeek's capable tier runs in thinking mode and
raises HTTP 400 ("The `reasoning_content` in the thinking mode must be passed
back to the API.") whenever the request carries an assistant ``tool_calls`` turn
with no ``reasoning_content``. pydantic-ai's native
``openai_chat_send_back_thinking_parts`` does NOT cover the case mill hits: a
history that ENDS at a pending tool-result and is continued
(``run_sync(None, message_history=…)``) — a pre-seeded ``read_file`` batch
(``build_preseed_history``), a replayed ``conversation_state``, or a pause/resume
mid tool-loop. Those assistant tool-call turns are synthetic or reconstructed
and carry no reasoning, so they 400. Reproduced live in
``tests/openrouter/test_openrouter_deepseek_live.py``
(``test_pro_resume_from_pending_tool_return_does_not_400``).

The fix: on the reasoning tier, every assistant tool-call turn carries a
``reasoning_content`` STRING — the turn's real reasoning (its ``ThinkingPart``s)
when present, else an empty string. DeepSeek requires the field to be a string;
an empty/placeholder string is accepted, a ``reasoning_details`` array is NOT
(both verified live). On the disabled (cheap) tier, all reasoning is stripped so
the sequence is consistently reasoning-free.

Why preference, not a hard pin: this layer used to send
``{"only": ["DeepSeek"], "allow_fallbacks": False}``, which forbids OpenRouter
from routing anywhere else. On 2026-07-29 the DeepSeek *upstream account* ran
out of balance and every ``deepseek/*`` request failed with HTTP 402
("Provider returned error" / ``provider_name: DeepSeek`` /
``is_byok: False``), blocking the whole mill board. Because the pin disabled
fallbacks, the ~17 healthy providers serving the same model — and OpenRouter's
own circuit breaker, which had already marked DeepSeek ``status: -5`` — were
bypassed by force, so the outage could not self-heal.

``order`` + ``allow_fallbacks: True`` keeps DeepSeek first (so the prompt cache
still warms in the common case) but lets OpenRouter route past it when it is
failing. ``max_price`` bounds what that fallback may cost: the same model is
served from ~$0.435/M to ~$1.740/M, so an unbounded fallback could silently
cost ~4x. The ceilings below were measured against the live endpoint list.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .model import OpenRouterModel, _resolve_model_settings

_PREFERRED_PROVIDER = "DeepSeek"
_PIN_MODEL_PREFIX = "deepseek/"
_REASONING_KEY = "reasoning"
_REASONING_CONTENT_KEY = "reasoning_content"
_TOOL_CALLS_KEY = "tool_calls"

#: Price ceilings in USD per 1M tokens, passed straight through to OpenRouter's
#: ``provider.max_price``. A ceiling that admits nobody makes the request fail
#: outright — OpenRouter answers ``404 No endpoints found that satisfy the max
#: price for this request`` — so these must be re-checked against the live
#: per-provider price list whenever DeepSeek pricing moves.
#:
#: Re-measured 2026-08-20 (previous values dated 2026-07-29):
#:
#: * capable tier (``deepseek-v4-pro``) — the old $0.70/$1.40 ceiling had
#:   decayed to admitting only 2 of 18 endpoints (StreamLake $0.624/$1.247,
#:   Baidu $0.625/$1.251).  DeepSeek itself had drifted out of its own
#:   ceiling — $0.435/$0.870 → $0.660/$1.980 — so ``order: ["DeepSeek"]``
#:   named a provider ``max_price`` then excluded, and any StreamLake/Baidu
#:   hiccup turned into a hard 404 instead of a fallback.  $0.90/$2.40 admits
#:   DeepSeek, StreamLake, Baidu, GMICloud ($0.792/$2.376) and DigitalOcean
#:   ($0.870/$1.740) — 5 healthy endpoints, the preferred one included —
#:   while still excluding the $1.13+/$2.26+ tail (Ionstream, CoreWeave,
#:   DeepInfra, Together, Fireworks, Azure, …) the cap exists to keep out.
#: * cheap tier (``deepseek-v4-flash``) — still healthy at $0.15/$0.30: the
#:   field runs $0.068/$0.168 (DigitalOcean) to $0.140/$0.280, so 13 of 18
#:   endpoints qualify.  Left unchanged.
DEFAULT_MAX_PRICE_CAPABLE: dict[str, float] = {"prompt": 0.90, "completion": 2.40}
DEFAULT_MAX_PRICE_CHEAP: dict[str, float] = {"prompt": 0.15, "completion": 0.30}


def build_provider_routing(
    *,
    preferred_provider: str | None = _PREFERRED_PROVIDER,
    allow_fallbacks: bool = True,
    max_price: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the OpenRouter ``provider`` routing preference block.

    Args:
        preferred_provider: Upstream provider to try first (``order``). Pass
            ``None`` to express no preference and let OpenRouter choose freely.
        allow_fallbacks: Whether OpenRouter may route to another provider when
            the preferred one fails. Keep ``True`` unless a caller genuinely
            needs a single provider and accepts that its outages become
            hard failures.
        max_price: Optional ``{"prompt": …, "completion": …}`` ceiling in USD
            per 1M tokens. Omitted entirely when ``None``.

    Returns:
        A dict suitable for ``extra_body["provider"]``.

    """
    routing: dict[str, Any] = {"allow_fallbacks": allow_fallbacks}
    if preferred_provider:
        routing["order"] = [preferred_provider]
    if max_price:
        routing["max_price"] = dict(max_price)
    return routing


def _reasoning_text(message: Any) -> str:
    """Concatenate the message's ``ThinkingPart`` contents into a string (the
    reasoning DeepSeek wants echoed back). Empty when the turn has no reasoning
    — e.g. a synthetic pre-seeded or reconstructed tool-call turn."""
    from pydantic_ai.messages import ThinkingPart

    parts = getattr(message, "parts", None) or []
    return "".join(
        p.content
        for p in parts
        if isinstance(p, ThinkingPart) and isinstance(getattr(p, "content", None), str)
    )


class OpenRouterDeepseekModel(OpenRouterModel):
    """OpenRouter model pinned to DeepSeek, with a per-tier reasoning policy and
    the thinking-mode ``reasoning_content`` round-trip.

    The provider stamps ``reasoning_setting`` and ``provider_routing`` per tier
    after construction (e.g. ``{"effort": "xhigh"}`` for the capable tier or
    ``{"enabled": False}`` for the cheap tier); sensible defaults (reasoning on
    at xhigh, DeepSeek preferred with fallbacks under the capable-tier price
    ceiling) apply if unset. The round-trip is active on every tier except the
    disabled one, derived from ``reasoning_setting`` (no separate flag needed).
    """

    reasoning_setting: ClassVar[dict[str, Any]] = {"effort": "xhigh"}
    #: OpenRouter ``provider`` routing preference, stamped per tier by the
    #: provider. Defaults to the capable-tier ceiling — the safe side, since a
    #: too-low ceiling fails the request outright.
    provider_routing: ClassVar[dict[str, Any]] = build_provider_routing(
        max_price=DEFAULT_MAX_PRICE_CAPABLE
    )

    @property
    def _echo_reasoning(self) -> bool:
        """Carry reasoning_content on tool-call turns iff reasoning is enabled —
        i.e. every tier except the cheap one's ``{"enabled": False}``."""
        return self.reasoning_setting.get("enabled", True) is not False

    def _inject_pin(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        model_name = str(getattr(self, "model_name", "") or "")
        if not model_name.startswith(_PIN_MODEL_PREFIX):
            return
        settings = _resolve_model_settings(args, kwargs)
        if settings is None:
            return
        extra_body = dict(settings.get("extra_body") or {})
        if "provider" not in extra_body:
            extra_body["provider"] = dict(self.provider_routing)
        if "reasoning" not in extra_body:
            extra_body["reasoning"] = dict(self.reasoning_setting)
        settings["extra_body"] = extra_body

    async def _completions_create(self, *args: Any, **kwargs: Any) -> Any:
        self._inject_pin(args, kwargs)
        # OpenRouterModel adds usage.include + records cost.
        return await super()._completions_create(*args, **kwargs)

    def _map_model_response(self, message: Any) -> Any:
        """Map a ModelResponse to an OpenAI assistant message, enforcing
        DeepSeek's thinking-mode reasoning rule (see module docstring).

        Reasoning tier: assistant tool-call turns carry ``reasoning_content`` (a
        string — the turn's real reasoning, else empty); non-tool-call turns and
        the disabled tier carry no reasoning at all. The ``reasoning`` /
        ``reasoning_details`` variants are always dropped (DeepSeek rejects an
        array; only the string ``reasoning_content`` is accepted)."""
        param = super()._map_model_response(message)
        if not (isinstance(param, dict) and param.get("role") == "assistant"):
            return param

        # Always clear the array/alias forms — DeepSeek only accepts the string.
        param.pop(_REASONING_KEY, None)
        param.pop("reasoning_details", None)

        if self._echo_reasoning and param.get(_TOOL_CALLS_KEY):
            # Present-but-possibly-empty string keeps the tool-call turn valid in
            # thinking mode even when the turn is synthetic/reconstructed.
            param[_REASONING_CONTENT_KEY] = _reasoning_text(message)
        else:
            param.pop(_REASONING_CONTENT_KEY, None)

        # DeepSeek rejects an assistant message with neither content nor
        # tool_calls (a thinking-only turn maps to no text and no tool calls). A
        # present empty string keeps such turns valid; this holds on every tier.
        if not param.get(_TOOL_CALLS_KEY) and not param.get("content"):
            param["content"] = ""
        return param

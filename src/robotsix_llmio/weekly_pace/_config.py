"""Weekly Claude usage pace governor — configuration models.

These are pure pydantic data models; the governor logic lives in
:mod:`robotsix_llmio.weekly_pace`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelWeightConfig(BaseModel):
    """Per-model consumption weight multipliers for calibration.

    Each weight scales the raw USD cost from Langfuse (which is already
    model-weighted by provider pricing) so the governor's budget accounting
    can be calibrated against what the interactive ``/usage`` meter actually
    reports over time.

    Defaults are 1.0 — raw USD cost is used as-is, which is a faithful
    approximation since Anthropic's per-model pricing already weights Opus
    higher than Sonnet/Haiku.
    """

    opus: float = Field(default=1.0, ge=0.0, description="Weight for Opus-tier models.")
    sonnet: float = Field(
        default=1.0, ge=0.0, description="Weight for Sonnet-tier models."
    )
    haiku: float = Field(
        default=1.0, ge=0.0, description="Weight for Haiku-tier models."
    )


class WeeklyPaceConfig(BaseModel):
    """Configuration for the weekly Claude usage pace governor.

    When *enabled* is True, the governor compares weighted Claude consumption
    this week against the elapsed week fraction and may route to DeepSeek
    when ahead of pace. When False (default), the governor always returns
    True (use Claude) — zero behaviour change.
    """

    enabled: bool = Field(
        default=False,
        description="Enable the pace governor. When False, always use Claude.",
    )

    weekly_budget: float = Field(
        default=10.0,
        ge=0.0,
        description="Weekly budget in weighted consumption units (USD-equivalent).",
    )

    week_anchor_day: int = Field(
        default=0,
        ge=0,
        le=6,
        description="Day of week for the weekly reset (0=Monday, 6=Sunday).",
    )

    week_anchor_time: str = Field(
        default="00:00",
        pattern=r"^\d{2}:\d{2}$",
        description="UTC time of day for the weekly reset (HH:MM).",
    )

    hysteresis_over: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Margin added to the pace line when deciding to fall back: "
            "fall back when budget_fraction > week_fraction + hysteresis_over."
        ),
    )

    hysteresis_under: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Margin subtracted from the pace line when deciding to stay on Claude: "
            "stay on Claude when budget_fraction < week_fraction - hysteresis_under."
        ),
    )

    model_weights: ModelWeightConfig = Field(
        default_factory=ModelWeightConfig,
        description="Per-model consumption weight multipliers for calibration.",
    )

    always_claude_agents: list[str] = Field(
        default_factory=list,
        description=(
            "Agent names that always use Claude regardless of pace. "
            "Critical/high-value agents should be listed here."
        ),
    )

    fail_open: bool = Field(
        default=True,
        description=(
            "When True and Langfuse is unreachable, default to using Claude "
            "and log a warning rather than blocking the pipeline."
        ),
    )

    cache_ttl_seconds: int = Field(
        default=60,
        ge=1,
        description="TTL in seconds for the cached Langfuse weekly cost query.",
    )

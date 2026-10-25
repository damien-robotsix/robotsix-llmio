# Failover cost-reduction operational runbook

Operate and tune the three deployed provider-failover cost-reduction
changes that cut failover-churn spend without regressing the request
completion rate:

1. the **per-request slot-switch cap** that bounds duplicate-prompt replay,
2. the **retry cap + exponential backoff** that stops a degraded primary
   from churning a doomed attempt every window, and
3. the **per-request cost guard** that refuses an expensive fallback slot.

All three are implemented in
[`robotsix_llmio.core.failover`](index.md#provider-failover) and driven by
[`FailoverConfig`](../config/index.md). This runbook covers how to
configure each change, which metrics to watch, and how to triage the
common failure modes. For the before/after-deploy acceptance procedure
(does spend actually drop, does completion stay flat) see
[Failover cost-reduction validation](failover-cost-validation.md).

## Where the configuration lives

The tunables are fields on `FailoverConfig`, set under the `failover` key
of the dict merged into the baked defaults by `load_tier_config`:

```python
from robotsix_llmio.core import load_tier_config

tier = load_tier_config(
    {
        "failover": {
            "failure_threshold": 3,
            "window_seconds": 900.0,
            "reattempt_backoff_base": 2.0,
            "reattempt_backoff_cap_seconds": 14400.0,
            "reattempt_backoff_jitter": 0.5,
            "max_request_cost_usd": None,
            "cost_ratio_threshold": None,
        }
    }
)
```

Any field omitted from the dict keeps its baked default (shown above).
These are configuration fields, not environment variables — there is no
`LLMIO_*` env knob for failover policy; change the dict a consumer passes
to `load_tier_config`.

All observability below is emitted as OpenTelemetry span attributes that
reach Langfuse (and the cost-monitor `/api/by-agent` aggregation). Two
per-attempt span names matter: `llmio.attempt.primary` for a
`default`-slot attempt (never counted as churn) and
`llmio.failover.attempt` for a genuine escalation to the `fallback` slot.

---

## 1. Per-request slot-switch cap (duplicate-prompt replay bound)

### What it does

A single request attempts its active slot and cascades down the tier chain,
trying **each slot at most once** — the cap `_max_slot_switches(len(chain))`
(`len(chain) - 1`; `2` for the baked three-slot chain) in `core/failover.py`. This is
the mechanism that removes the dominant duplicate-prompt replay cost: a
doomed request can never ping-pong between slots re-sending the full prompt
on each hop, so per-request churn spend is bounded to one attempt per slot.

### Enable / disable configuration

The cap is a **hard-coded constant, always on, and not runtime-tunable** —
there is deliberately no knob to raise it, because a higher cap directly
reinstates the duplicate-prompt cost it exists to prevent. The only
related switch is the `failover_enabled` parameter of
`call_with_failover` / `acall_with_failover` (default `True`): pass
`False` to drop the fallback attempt entirely, so the request tries only
its active slot and performs **zero** slot switches.

### Metrics to monitor

| Attribute (run span) | Meaning | Expected |
| --- | --- | --- |
| `llmio.failover.max_slot_switches` | the cap in force | always `1` |
| `llmio.failover.slot_switches` | slot switches this request | `0` (no switch) or `1` (one escalation) |
| `llmio.failover.activated` | request routed to the fallback slot | `true` only when a switch happened |

Watch the rate of `slot_switches == 1` (equivalently `activated == true`):
a sustained rise means the primary slot is degraded and requests are
leaning on the fallback. A value above `1` is impossible and signals a
regression of the cap itself.

### Troubleshooting

- **Switches look higher than expected across a window:** a per-request
  value never exceeds `1`; a high *aggregate* is many distinct requests
  each switching once — look to the primary slot's health
  (`llmio.attempt.primary` failures), not the cap.
- **No fallback ever happens on a real outage:** confirm the caller did
  not pass `failover_enabled=False`, which disables the single permitted
  switch.

---

## 2. Retry cap and exponential backoff

### What it does

Once the `default` slot fails in a provider-shaped way
`failure_threshold` times in a row (a provider-wide exhaustion arms
immediately), the tracker routes **all** calls to the `fallback` slot for
a window, skipping the doomed default attempt. The **first** arming uses
`window_seconds` unchanged; each subsequent re-arm (a still-degraded
default that failed its probe after the window expired) grows the window
with exponential backoff and jitter, so a persistently broken primary is
re-probed exponentially *less* often instead of churning a doomed attempt
every window.

### Config options

| Field | Default | Purpose |
| --- | --- | --- |
| `failure_threshold` | `3` | consecutive provider-shaped default-slot failures before failover arms (exhaustion arms immediately regardless) |
| `window_seconds` | `900.0` | base sticky-window length (15 min) the first time failover arms |
| `reattempt_backoff_base` | `2.0` | exponential base (`> 1`) for repeated degraded-primary re-arms |
| `reattempt_backoff_cap_seconds` | `14400.0` | ceiling on the backed-off window (4 h); must be `>= window_seconds` |
| `reattempt_backoff_jitter` | `0.5` | fraction of the window subtracted as random jitter (lands in `[window·(1−jitter), window]`) |

The backed-off window for the `n`-th re-arm is
`min(window_seconds · reattempt_backoff_base ^ n, reattempt_backoff_cap_seconds)`,
then multiplicatively de-jittered. A `default`-slot success resets the
reattempt count, so the next degradation starts again from the base
window. A parsed quota-reset hint ("resets 1:10pm (UTC)") arms to that
time verbatim, bypassing backoff.

### Tuning the backoff delays

- **Primary recovers quickly (brief blips):** keep `window_seconds` modest
  (minutes) and `reattempt_backoff_base` near `2.0` so probing resumes
  soon after recovery.
- **Primary outages are long (quota/billing):** raise
  `reattempt_backoff_cap_seconds` so a persistently broken primary is
  re-probed rarely, and/or raise `reattempt_backoff_base` so the window
  grows faster per re-arm — fewer doomed probe attempts, less churn.
- **A fleet re-probing in lockstep (thundering herd on recovery):** raise
  `reattempt_backoff_jitter` toward `1.0` to spread re-probes across
  instances. Jitter only applies to backed-off re-arms, never the first
  window.
- **Constraint:** `reattempt_backoff_cap_seconds` must be `>=
  window_seconds` (validated at load) — a cap below the base window is
  rejected.

### Metrics to monitor

| Attribute (run span) | Meaning |
| --- | --- |
| `llmio.failover.reattempt_count` | number of consecutive degraded-primary re-arms since the last default success |
| `llmio.failover.backoff_seconds` | the backed-off window actually applied on that re-arm |

A climbing `reattempt_count` with a `backoff_seconds` approaching
`reattempt_backoff_cap_seconds` is the signature of a *persistently* broken
primary — expected behaviour, and the point at which operators should chase
the primary provider rather than the failover config. The live state is
also available without spans via `get_failover_status()` /
[`FailoverStatus`](index.md#provider-failover) (active slot, armed-until).

### Troubleshooting

- **Calls parked on the fallback far longer than `window_seconds`:**
  expected once `reattempt_count > 0` — backoff has grown the window.
  Confirm against `backoff_seconds`; if undesired, lower
  `reattempt_backoff_base` / `reattempt_backoff_cap_seconds`.
- **Failover never arms despite primary errors:** the failures may not be
  provider-shaped (`is_provider_shaped` classifies task-shaped errors as
  non-failover) — these never increment the failure counter.
- **Window never backs off:** each `default`-slot success resets
  `reattempt_count` to `0`; an intermittently-recovering primary keeps
  re-arming from the base window by design.

---

## 3. Per-request cost guard

### What it does

Before a request spends money retrying on the candidate (fallback) slot,
the loop estimates that slot's worst-case per-request cost and **refuses**
the switch when it is too expensive, failing the request fast with
`FailoverCostGuardError` instead of running an expensive attempt. The
estimate is `estimate_request_cost(tlc)` — the slot's
`max_price_completion` (USD per 1M completion tokens) times its
`max_tokens` output cap. A slot carrying no price ceiling or no output cap
(e.g. the Claude SDK `default` slots) cannot be estimated and is **never**
blocked.

### Setting the per-request ceiling

Both knobs default to `None` (guard **off**); set either or both to arm it:

| Field | Default | Refuse the switch when… |
| --- | --- | --- |
| `max_request_cost_usd` | `None` (off) | the candidate estimate **strictly exceeds** this USD ceiling |
| `cost_ratio_threshold` | `None` (off) | the candidate estimate **strictly exceeds** the active slot's estimate × this ratio (candidate is materially pricier than staying put) |

Both boundaries are inclusive — an estimate exactly equal to the ceiling
(or exactly the primary × ratio) is **allowed**. A missing *primary*
estimate disables only the ratio check; the ceiling check still applies.
Example — cap any single failover attempt at $0.05 and never switch to a
slot more than 3× the primary's price:

```python
{"failover": {"max_request_cost_usd": 0.05, "cost_ratio_threshold": 3.0}}
```

### Impact of guard refusals on fallback behaviour

A refusal **aborts the failover** for that request: the switch does not
happen, `FailoverCostGuardError` is raised (chained from the original
provider failure), and the request **does not complete**. That is the
intended trade — spend is capped at the primary attempt — but it means an
over-tight ceiling converts recoverable requests into failures. Because a
guard refusal raises, it counts as a **non-completion** in the
[validation](failover-cost-validation.md) `completion_rate`, so the guard
can never inflate the completion rate to hide dropped requests.

### Monitoring and alerting setup

| Attribute (run span) | Meaning |
| --- | --- |
| `llmio.failover.cost_guard_blocked` | `true` when a slot switch was refused on cost grounds |
| `llmio.failover.cost_guard_reason` | human-readable refusal reason (which check tripped, the two dollar figures) |

Recommended alerts:

- **Over-blocking:** alert when the rate of `cost_guard_blocked == true`
  spans rises materially *and* correlates with a drop in
  `completion_rate` — the guard is refusing slots that used to succeed.
  This is also an explicit [rollback trigger](failover-cost-validation.md#rollback-triggers).
- **Silent miss:** if the ceiling is set but `cost_guard_blocked` is never
  `true` while fallback spend stays high, the candidate slot likely cannot
  be *estimated* (no `max_price_completion` / `max_tokens`) and is never
  blocked — verify the fallback slot exposes both.

### Troubleshooting

- **Guard appears to do nothing:** confirm at least one of
  `max_request_cost_usd` / `cost_ratio_threshold` is non-`None`, and that
  the fallback slot carries both a `max_price_completion` price ceiling and
  a `max_tokens` cap — an unestimatable slot is never blocked.
- **Requests failing with `FailoverCostGuardError` under a real outage:**
  the ceiling is tighter than the only available slot's cost. Raise
  `max_request_cost_usd` (or relax `cost_ratio_threshold`), or set both to
  `None` to disable the guard — no redeploy needed, it is pure config.
- **Ratio check never fires:** it is skipped whenever the *primary* slot's
  cost cannot be estimated; only the absolute ceiling applies in that case.

---

## Rollback (config-first)

All three behaviours are config-driven, so re-tuning needs no redeploy:
disable the cost guard by returning `max_request_cost_usd` and
`cost_ratio_threshold` to `null`; relax the backoff by lowering
`reattempt_backoff_base` / `reattempt_backoff_cap_seconds` toward plain
`window_seconds`. If a regression persists with the guard and backoff
disabled, revert the deploy. See the
[rollback triggers](failover-cost-validation.md#rollback-triggers) for the
quantitative pass/fail thresholds.

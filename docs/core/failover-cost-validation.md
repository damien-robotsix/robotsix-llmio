# Failover cost-reduction validation

A repeatable procedure for proving that the provider-failover changes
(partial-completion checkpointing, retry capping + backoff, and the
per-request cost guard) cut the 7-day cost of failover churn roughly in
half **without** lowering the request completion rate.

Run this before/after every deploy that touches
[`call_with_failover`](index.md#provider-failover) or the
[`FailoverConfig`](../config/index.md) knobs, and keep the results next to
the deploy record so the rollback decision is auditable.

## What is measured

Every genuine escalation to the fallback slot is emitted as an OTel span
named **`llmio.failover.attempt`** (primary-slot successes use
`llmio.attempt.primary` and are deliberately excluded — a primary success
is not failover churn). Provider cost is stamped onto these spans by the
transport cost recorders, so the cost-monitor service aggregates it per
agent through its `/api/by-agent` endpoint.

Relevant span attributes (all defined in
`robotsix_llmio.core.failover`):

| Attribute | Meaning |
| --- | --- |
| `llmio.failover.succeeded` | the attempt produced a usable result |
| `llmio.failover.activated` | the request routed to the fallback slot because failover was armed |
| `llmio.failover.slot_switches` | per-request slot switches (hard-capped at 1) |
| `llmio.failover.cost_guard_blocked` | a slot switch was refused on cost grounds |
| `llmio.failover.cost_guard_reason` | human-readable refusal reason |
| `llmio.failover.reattempt_count` / `llmio.failover.backoff_seconds` | degraded-primary re-arm count and the backed-off window applied |

## Comparison queries

The acceptance signal is the 7-day **cost** carried by
`llmio.failover.attempt` spans, before vs. after the deploy. Query
cost-monitor `/api/by-agent` for two equal-length windows that bracket the
deploy and keep every other variable (fleet size, traffic mix) as close as
possible:

```bash
# BEFORE: the 7 days ending at the deploy timestamp
curl -s "$COST_MONITOR_URL/api/by-agent?span=llmio.failover.attempt&window=7d&end=$DEPLOY_TS"

# AFTER: the 7 days starting at the deploy timestamp
curl -s "$COST_MONITOR_URL/api/by-agent?span=llmio.failover.attempt&window=7d&start=$DEPLOY_TS"
```

Sum the per-agent `cost` across each response to get the fleet-wide
failover spend for each window. **Acceptance:** the AFTER total drops
meaningfully relative to BEFORE — the epic target is roughly a halving of
`llmio.failover.attempt` 7-day cost (≈17.7% → ≈9% of fleet LLM spend).

### Failover spend share

Report the absolute drop as a share of total fleet LLM spend so the result
is comparable across weeks of differing traffic:

```
failover_share = cost(span=llmio.failover.attempt, 7d) / cost(all spans, 7d)
```

Query the denominator from the same `/api/by-agent` window without the
`span=` filter (all agents, all spans) and divide. Track `failover_share`
for both the BEFORE and AFTER windows; the share should fall by roughly the
same factor as the absolute cost.

## Completion rate (the guardrail)

The cost reduction only counts if requests still complete. Define the
completion rate over the same 7-day window as:

```
completion_rate = succeeded_runs / total_runs
```

Measure it from the run-level spans: `total_runs` is the count of
top-level agent runs and `succeeded_runs` is the subset whose
`llmio.failover.succeeded` is true (equivalently, runs that did not raise).
A request refused by the cost guard counts as a **non-completion** (it
raises `FailoverCostGuardError`), so the guard cannot inflate this number.

**Acceptance:** `completion_rate` in the AFTER window stays within the
chosen tolerance of the BEFORE window — default tolerance **±1.0 absolute
percentage point**. A larger drop means the cost savings came at the cost
of dropped requests and is a rollback trigger.

## Repeatable validation run

1. Record the deploy timestamp `$DEPLOY_TS`.
2. Wait for a full 7-day AFTER window to accumulate (or compare equal
   partial windows if an earlier read is needed — note the window length in
   the result).
3. Run the two `/api/by-agent` queries above; compute BEFORE/AFTER
   failover cost, `failover_share`, and `completion_rate`.
4. Record all six numbers (2× cost, 2× share, 2× completion rate) in the
   deploy record.
5. Apply the pass/fail criteria below.

| Metric | Pass |
| --- | --- |
| `llmio.failover.attempt` 7-day cost | AFTER drops meaningfully (target ≈ ½ of BEFORE) |
| `failover_share` | falls, roughly tracking the absolute cost drop |
| `completion_rate` | within ±1.0 pp of BEFORE |

## Rollback triggers

Roll back (or re-tune) when any of the following holds in the AFTER window:

- **Completion-rate regression:** `completion_rate` drops more than the
  ±1.0 pp tolerance below BEFORE — the churn reduction is shedding real
  requests.
- **Cost-guard over-blocking:** a material rise in
  `llmio.failover.cost_guard_blocked` spans that correlates with the
  completion-rate drop — the guard is refusing slots that used to succeed.
- **No cost benefit:** `llmio.failover.attempt` 7-day cost does not fall
  meaningfully, so the change is not paying for its risk.

Rollback is config-first — the behaviour is driven by
[`FailoverConfig`](../config/index.md), so no redeploy is required:

- Disable the cost guard by setting `max_request_cost_usd` and
  `cost_ratio_threshold` back to `null` (their default-off values).
- Relax the degraded-primary backoff by lowering
  `reattempt_backoff_base` / `reattempt_backoff_cap_seconds` toward the
  plain `window_seconds` behaviour.

If the regression persists with the guard and backoff disabled, revert the
deploy.

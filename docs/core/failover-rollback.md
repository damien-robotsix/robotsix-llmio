# Failover disable & rollback runbook

Every one of the three failover cost-reduction changes can be disabled or
relaxed **without a code redeploy** — the behaviour is driven by
[`FailoverConfig`](../config/index.md) (the `failover` key of the dict passed
to `load_tier_config`) and the per-call `failover_enabled` flag, not by
hard-coded policy. This runbook records the exact knob for each change so an
operator can roll back from configuration alone.

`FailoverConfig` is **not** set from environment variables: it is supplied
under the `failover` key of the dict handed to
[`load_tier_config`](../config/index.md#schema-loader-tier-configuration), so
"disable without redeployment" means editing the consumer's tier-config
source (the dict / file it feeds `load_tier_config`) and reloading, not
flipping an env var.

```python
cfg = load_tier_config(
    {
        "failover": {
            # ... rollback knobs documented below ...
        },
    }
)
```

## 1. Cost guard — disable the per-request cost checks

The per-request cost guard (see
[`estimate_request_cost` / `_cost_guard_verdict`](index.md#provider-failover))
is **off by default**: both knobs default to `None`, and
`_cost_guard_verdict` returns "allow" (no refusal) whenever both are unset.

| Knob | Default | Disable value |
| --- | --- | --- |
| `max_request_cost_usd` | `None` (off) | `None` / `null` |
| `cost_ratio_threshold` | `None` (off) | `None` / `null` |

To roll back the cost guard, set **both** back to `null`:

```python
"failover": {"max_request_cost_usd": None, "cost_ratio_threshold": None}
```

With both `None`, no slot switch is ever refused on cost grounds and
`FailoverCostGuardError` is never raised. Clearing only one knob disables
only that check (the ceiling and the ratio are independent); clear both to
fully disable the guard.

## 2. Retry cap & backoff — relax or neutralise the backoff

Two things were introduced here: the per-request **slot-switch cap** and the
degraded-primary **exponential backoff**.

### Backoff (config-tunable)

The degraded-primary re-arm backoff is fully configurable. To make it
**lenient** — i.e. keep re-probing a degraded primary on the plain, constant
`window_seconds` cadence instead of an exponentially growing one — pin the
cap to the base window and remove jitter:

```python
"failover": {
    "window_seconds": 900,
    "reattempt_backoff_cap_seconds": 900,  # == window_seconds → constant window
    "reattempt_backoff_jitter": 0.0,       # no jitter
}
```

| Knob | Default | Lenient / rollback value |
| --- | --- | --- |
| `reattempt_backoff_base` | `2.0` (must be `> 1.0`) | leave at default; the cap below neutralises growth |
| `reattempt_backoff_cap_seconds` | `14400` (4 h; must be `>= window_seconds`) | set equal to `window_seconds` to pin the window constant |
| `reattempt_backoff_jitter` | `0.5` | `0.0` to disable jitter |

Setting `reattempt_backoff_cap_seconds == window_seconds` caps every
backed-off window at the base window, so repeated re-arms never grow beyond
`window_seconds` — the pre-backoff behaviour. `reattempt_backoff_base` cannot
be set to `1.0` (the schema requires `> 1.0`), so the cap is the supported
way to neutralise growth. The validator also rejects a cap **below**
`window_seconds`.

### Slot-switch cap (bounded by two slots; disable failover entirely)

The per-request slot-switch cap `_MAX_SLOT_SWITCHES = 1` in
`robotsix_llmio.core.failover` is a hard constant, not a runtime knob. It is
already the natural maximum: there are exactly two provider slots (`default`,
`fallback`), so one switch exhausts them — raising the cap would have nothing
further to switch to, which is why it is not exposed as config.

To remove the *entire* failover attempt (and with it any slot switch), pass
`failover_enabled=False` to `call_with_failover` / `acall_with_failover` at
the call site. With it off, a provider-shaped failure on the active slot
raises straight through instead of retrying on the other slot.

## 3. Partial completion checkpointing — skip via `failover_enabled`

In this library "partial completion checkpointing" is **not** a separate
checkpoint/resume code path; it is realised as the one-slot-switch cap that
stops a single request from replaying its full prompt across slots more than
once (the dominant duplicate-prompt replay cost). The fallback path that
"skips checkpointing" — i.e. takes no fallback attempt at all — is the
per-call `failover_enabled` flag:

```python
result = call_with_failover(
    fn_factory,
    tier_config=cfg,
    level=2,
    failover_enabled=False,  # no fallback attempt; run the active slot only
)
```

`failover_enabled=False` makes the request attempt only the currently-active
slot (no switch, no replay). This is the per-call escape hatch for callers
that must not be re-run on the other provider.

## Rollback decision

Prefer config-first rollback — none of the above requires a redeploy:

1. **Cost-guard over-blocking** (a rise in
   `llmio.failover.cost_guard_blocked` spans correlated with a completion-rate
   drop) → clear `max_request_cost_usd` and `cost_ratio_threshold` (§1).
2. **Backoff parking calls on the fallback slot too long** → pin
   `reattempt_backoff_cap_seconds` to `window_seconds` and zero the jitter
   (§2).
3. **Need to take a provider entirely out of the failover path for a call**
   → `failover_enabled=False` (§2 / §3).

See [Failover cost-reduction validation](failover-cost-validation.md) for the
metrics (`failover_share`, `completion_rate`) that trigger these rollbacks,
and [`robotsix_llmio.core.failover`](index.md#provider-failover) for the
behaviour each knob governs. Only if a regression persists with the guard and
backoff disabled should the deploy itself be reverted.

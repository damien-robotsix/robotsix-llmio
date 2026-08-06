Drop `task_budget` and retry once when the API reports the model does not
support it, instead of failing the call. `task_budget` is a beta parameter only
some models accept; the transport was sending it unconditionally whenever a
tier configured `max_tokens`, so every call against any other model died with
`400 This model does not support user-configurable task budgets`. Observed on
2026-08-06 taking mill's refine stage down across five boards. The supported
set cannot be hardcoded here — callers configure a tier alias (`sonnet`,
`opus`) that the `claude` CLI resolves downstream — so the rejection itself is
the discovery mechanism: the budget is dropped, the request re-sent, and the
model remembered so later calls skip straight to the working shape.

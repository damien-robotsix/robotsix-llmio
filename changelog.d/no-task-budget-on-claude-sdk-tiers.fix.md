Stop sending a `task_budget` derived from `max_tokens` on the Claude SDK
tiers. `ClaudeAgentOptions` has no per-response cap, so the value could only
become an *advisory* whole-loop allowance the model is shown as a countdown —
and both baked defaults (8192 on level 3, 16384 on level 4) sat below the
API's 20,000 floor and were clamped **up**, so they capped nothing and simply
told the model it had a small allowance for the entire task. Observed
2026-08-06: agents abandoning work before starting it ("I'm out of token budget
for this task before I could load the required tools"), and a hard 400 on
models that reject the parameter outright. Below-floor values now send no
budget at all rather than being clamped up. The OpenRouter tiers keep
`max_tokens` — there it is a real enforced per-response cap.

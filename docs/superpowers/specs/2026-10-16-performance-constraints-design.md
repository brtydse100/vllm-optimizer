# Performance constraints

Users can maximize the existing objective while requiring every tested workload
to meet latency or throughput limits. Search runs its full configured budget.
The user authorized implementation, a new PR, and an independent judge review.

## Configuration

```yaml
optimization:
  maximize: output_tokens_per_second
  constraints:
    - metric: end_to_end_latency_ms
      statistic: p99
      max: "200 ms"
    - metric: requests_per_second
      statistic: average
      min: "50 req/s"
```

Each constraint has one bound (`min` or `max`), a canonical metric, and a
statistic (`average`, `p50`, `p95`, or `p99`). Latency metrics are TTFT, TPOT,
inter-token, and end-to-end latency; units are `us`, `ms`, and `s`.
Throughput metrics are requests, output tokens, and total tokens per second;
units are `req/s` and `tok/s`, respectively. Numeric bounds use canonical
units (ms, req/s, tok/s). Reject negative/nonfinite bounds, malformed mappings,
unknown metrics/statistics, and incompatible units. Equality satisfies a bound.

## Evaluation and search

Keep the existing maximize objective and request-failure eligibility policy.
For each named run and workload identity, average the selected statistic over
eligible repeats. Require the scoring minimum-repeat count, identical workload
coverage across repeats, and every required named run. Any missing/nonfinite
selected statistic in an eligible repeat fails that workload's constraint.
Percentiles are backend-supplied values, never pooled or synthesized.
Preserve p95 alongside p50/p99 in normalized metrics.

Retain objective scores for completed infeasible trials, with a signed normalized
violation per constraint (positive means violated; missing evidence uses a
positive finite sentinel). TPE receives these via its constraints callback and
still receives the original objective. Grid and Random also execute their full
budgets. Runtime failures remain failures, not performance violations.
Rank only feasible trials; a baseline that misses a limit cannot win.
Carry the policy through retries and fresh finalist validation. Finalist
decisions must not favor an infeasible candidate.

## Results and compatibility

Persist the original unit-bearing policy and per-workload assessment in run
results. Display measured values, canonical units, and target status in the
terminal summary, HTML, and CSV. Report explicitly when no tested configuration
meets all targets. Failed/incomplete trials cannot be presented as meeting them.
Offline regeneration verifies constraint eligibility against accepted evidence;
reclassification preserves constraints while changing request-failure policy.
Runs without constraints retain existing ranking and offline behavior.

## Boundaries and validation

Configuration owns parsing/units; a small evaluation module owns workload
aggregation; scoring owns feasible ranking; search owns sampler feedback;
reporting owns serialized evidence and presentation. No new dependencies.
New/edited source files stay below 200 lines by splitting responsibilities.
Private tests live outside the checkout and cover units, exact bounds, repeats,
coverage, missing metrics, TPE feedback, baseline selection, persistence,
finalist validation, retries, offline integrity, and reclassification.
Run the private and public suites, Ruff, and mypy before PR completion.
An independent judge reviews the final branch for correctness and regressions.

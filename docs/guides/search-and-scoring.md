# Search and scoring

Request failures are an eligibility gate controlled by
`benchmark.max_failure_percentage`. Among eligible trials, the configured
`optimization.maximize` metric is the primary objective in search, persisted
rankings, baseline comparison, terminal output, CSV, and HTML. Ties use failure
percentage, failure count, and trial identifier in that order.

Choose `grid` for exhaustive small spaces, `random` for a bounded sample, or
`tpe` for guided exploration. Random and TPE searches do not execute the same
resolved configuration twice.

Rankings describe observed scores. Parameter associations from adaptive search
are not causal effects. Optional [fresh finalist validation](benchmarks/benchmark-repeats.md#fresh-finalist-validation)
compares the baseline and a fixed set of top candidates after search. It can
report no clear winner when the evidence does not distinguish them. These fresh
scores do not replace observations already supplied to the search sampler.

```yaml
experiment:
  name: search-example
  seed: 42
optimization:
  maximize: output_tokens_per_second
  sampler: tpe
  trials: 20
```

`maximize` names the metric used to rank trials. The declared metric is always maximized;
there is no automatic direction inference or minimize mode. Do not select raw
latency expecting lower values to win. For each named benchmark, the `vllm-opt` CLI averages
the eligible workload metric values, takes the arithmetic mean when it was repeated,
then averages named benchmark scores into the trial score. A workload with an
errored or incomplete percentage above `benchmark.max_failure_percentage` is
excluded; the setting defaults to `0`. A trial without an eligible workload is
not ranked. Every configured named benchmark must also have at least
`benchmark.min_repeats` eligible repeat scores. Older stored runs without an
aggregation policy retain median scoring during offline regeneration.

Grid evaluates every unique configuration and is best for small spaces. TPE
uses completed trial results to choose promising configurations, so it is the
better default for larger spaces where exhaustive Grid search is impractical.

The fixed `server` configuration runs first as the baseline unless
`baseline.enabled: false` is set. The report
shows the best observed configuration and its difference from that baseline.

`experiment.seed` seeds Random and TPE suggestions. Parallel completion order
can still change the sequence of TPE suggestions.

TPE state is persisted in the run's SQLite study. Runs themselves remain
immutable: a retry creates a new linked run instead of rewriting history.

## Performance targets

Set `optimization.constraints` to require performance limits while maximizing
the existing objective. Search continues through its configured budget.

```yaml
optimization:
  maximize: output_tokens_per_second
  sampler: tpe
  trials: 20
  constraints:
    - metric: end_to_end_latency_ms
      statistic: p99
      max: "200 ms"
    - metric: requests_per_second
      statistic: average
      min: "50 req/s"
```

Each entry requires a metric and exactly one `min` or `max` bound. Equality
satisfies a limit. `statistic` defaults to `average`; latency also supports
`p50`, `p95`, and `p99`, when the backend supplies that percentile.

| Metrics | Accepted units | Numeric bound unit |
| --- | --- | --- |
| `end_to_end_latency_ms`, `time_to_first_token_ms`, `time_per_output_token_ms`, `inter_token_latency_ms` | `us`, `ms`, `s` | ms |
| `requests_per_second` | `req/s` | req/s |
| `output_tokens_per_second`, `total_tokens_per_second` | `tok/s` | tok/s |

For example, `max: "1 s"` and `max: "1000 ms"` are equivalent. Bare numbers use
the metric's canonical unit. Bounds must be finite and non-negative; unknown
metrics, statistics, and incompatible units are rejected before execution.

All limits must hold for every named benchmark and workload, using means across
eligible repeats. Missing percentile measurements cannot meet a target. JSON,
CSV, HTML, and the terminal summary show target status and measured evidence.
Retries and offline request-failure reclassification preserve the limits.
See [search and scoring](search-and-scoring.md#constrained-optimization) for
selection and repeat rules.

## Constrained optimization

Optional `optimization.constraints` adds performance limits alongside the
maximize objective. For example, maximize output throughput with end-to-end
p99 latency at or below `"200 ms"`, optionally also requiring requests per
second above `"50 req/s"`. See [performance targets](#performance-targets)
for the YAML and supported units.

All limits must pass for every named benchmark and matched workload identity.
The selected statistic is averaged over request-eligible repeats. Each workload
needs at least `benchmark.min_repeats` eligible observations; each eligible
observation must supply a finite selected statistic. Workload coverage must
match across repeats, and repeated records cannot reuse a repeat identifier.
Missing evidence fails the target rather than being treated as zero. P50/P95/P99
are means of backend-supplied percentiles, not pooled request percentiles.

Trials that miss a performance target retain their measurements and complete
normally, but cannot win overall or per benchmark. The baseline must satisfy
the same limits to win. Fresh finalist validation reevaluates all limits using
its full repeat budget; retries and offline exports preserve the policy.

TPE receives the original objective score and a separate signed normalized
violation for each limit. It uses violations to guide suggestions toward
feasible configurations. Grid and Random apply the same winner eligibility
rules. None of these modes stops when a target is first met: they execute the
configured search budget. Failed executions remain failures, not performance
constraint observations. Without constraints, existing behavior is unchanged.

If no tested configuration satisfies every limit, results explicitly state
that no configuration met all targets and do not select an infeasible winner.
Constraint evidence is stored in `result.json` and shown in the terminal,
HTML, and the CSV `targets_met`/`target_evidence` fields. Offline regeneration
checks target eligibility against accepted measurements and rejects a policy
that disagrees with its reproduction manifests.

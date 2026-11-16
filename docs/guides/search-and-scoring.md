# Search and scoring

One configured metric ranks eligible trials throughout search, baseline
comparison, and reporting. `optimization.maximize` names a metric that is
**always maximized**. To minimize latency, use `optimization.objective` instead.

## Choose an objective

```yaml
optimization:
  objective:
    metric: end_to_end_latency_ms
    percentile: 99
  sampler: tpe
  trials: 20
```

Latency metrics are `time_to_first_token_ms`, `time_per_output_token_ms`,
`inter_token_latency_ms`, and `end_to_end_latency_ms`. They are minimized;
omit `percentile` to minimize the mean. Throughput metrics are
`output_tokens_per_second`, `total_tokens_per_second`, and `requests_per_second`;
they are maximized and do not accept a percentile. Select one objective per
run; combining `objective` with `maximize` is rejected.

vLLM Bench Serve supports integer percentiles 1–100, including P55. The
selected percentile and the latency metrics are added to each run's
`metric-percentiles` and `percentile-metrics` arguments. GuideLLM supports the
percentiles it emits: P1, P5, P10, P25, P50, P75, P90, P95, and P99; others
fail validation with instructions to use `benchmark.engine: vllm`. A missing
measurement never falls back to another percentile or the mean.

Latency scores are the measured values in milliseconds; lower wins. A
percentile score is the mean of backend-reported percentiles across workloads
and repeats, not a percentile of pooled requests. Offline reports, retries, and
reclassification retain the objective. `maximize` keeps its earlier behavior,
including maximizing a latency mean if one is named, and rejects latency
statistic keys such as `end_to_end_latency_ms.p99`.

## Choose a search

| Sampler | Use it for | Budget |
| --- | --- | --- |
| `grid` | Exhaustive small spaces | Every unique configuration; omit `trials` |
| `random` | A bounded sample | Set `trials` |
| `tpe` | Guided exploration of larger spaces | Set `trials` |

```yaml
experiment:
  name: search-example
  seed: 42
optimization:
  maximize: output_tokens_per_second
  sampler: tpe
  trials: 20
```

Random and TPE do not execute a resolved configuration twice. The seed controls
suggestions, though parallel completion order can change TPE's sequence.
Random/TPE study state is saved in SQLite; retries create new linked runs.

## Score and baseline

The fixed `server` configuration runs first as the baseline unless
`baseline.enabled: false`. Request failures are a separate eligibility gate:
`benchmark.max_failure_percentage` defaults to zero.

Eligible workload metric values are averaged within a named benchmark;
eligible repeats use their arithmetic mean, then named benchmark scores are
averaged into the trial score. Every workload must have enough eligible repeats,
and every configured named benchmark must be covered. Ties use failure
percentage, failure count, then trial identifier. Historical stored runs without
an aggregation policy retain median scoring offline.

Rankings describe observed scores. Adaptive-search associations do not establish
causation. [Fresh finalist validation](benchmarks/benchmark-repeats.md#fresh-finalist-validation)
compares selected candidates against the baseline with new measurements and can
conclude **No clear winner**. These scores do not replace prior sampler feedback.

## Performance targets

Maximize throughput while requiring limits such as p99 TTFT ≤200 ms:

```yaml
optimization:
  maximize: output_tokens_per_second
  sampler: tpe
  trials: 20
  constraints:
    - metric: time_to_first_token_ms
      statistic: p99
      max: "200 ms"
    - metric: requests_per_second
      min: "50 req/s"
```

| Metrics | Accepted units | Bare-number unit |
| --- | --- | --- |
| `time_to_first_token_ms`, `time_per_output_token_ms`, `inter_token_latency_ms`, `end_to_end_latency_ms` | `us`, `ms`, `s` | ms |
| `requests_per_second` | `req/s` | req/s |
| `output_tokens_per_second`, `total_tokens_per_second` | `tok/s` | tok/s |

Each entry has exactly one `min` or `max`; equality satisfies the bound.
The statistic defaults to `average`. Latency also supports backend-supplied
`p50`, `p95`, and `p99`; throughput supports `average` only.
`"1 s"` and `"1000 ms"` are equivalent. Bounds must be finite and non-negative;
incompatible units and unknown metrics/statistics fail validation.

These are optimization targets. GuideLLM's per-run benchmark `constraints`
instead control execution, such as request counts or duration.

## Constrained optimization

All limits must pass for each named benchmark and matched workload. The selected
statistic is averaged over eligible repeats, with at least `min_repeats`
observations and matching workload coverage. Missing or nonfinite evidence and
duplicate repeat identifiers fail the target. Percentiles are averaged backend
values, **not pooled request percentiles**.

Target misses retain measurements but cannot win overall or per benchmark.
Baselines and fresh finalists must meet the same limits. If no configuration
meets every target, the result explicitly selects no feasible winner.

TPE receives the original objective and a separate signed violation per limit.
It seeks more throughput within the bounds; proximity to a latency boundary is
not an extra objective. Grid, Random, and TPE continue the configured search
budget after meeting targets. Failed executions remain failures.

Targets and measured evidence appear in terminal, JSON, HTML, and CSV
(`targets_met`, `target_evidence`). Retries and offline reclassification preserve
the limits; regeneration rejects a policy inconsistent with accepted manifests.

# Benchmark repeats and errors

The trusted default is four measured repeats with no discarded warmup; all four
are required for ranking. Add `warmup_repeats` only when the workload needs
warmup. Explicit lower values remain available for smoke tests.
Confidence intervals use Student's t distribution. Critical values through 30
degrees of freedom use a table; larger samples use a finite-degree Student's t
quantile expansion rather than switching to a normal interval. Drift detection requires at
least four sequential measurements and reports when it is unavailable.


```yaml
benchmark:
  repeats: 4
  warmup_repeats: 1  # Optional; omit when no warmup is needed.
  max_failure_percentage: 2
```

The `vllm-opt` CLI takes the arithmetic mean optimization score across repeats. Reported
latency and throughput comparisons use arithmetic means. It records successful,
errored, and incomplete request counts. `max_failure_percentage` accepts a
number from `0` through `100` and defaults to `0`. A workload at or below that
percentage is eligible when at least one request succeeded; a workload above it
is excluded. The decision is made after each benchmark finishes, using its
final request counts. Set `accept_any_request_failures: true` to ignore the
percentage while still requiring at least one successful request and a usable
metric. Request-count runs also require the expected total to be present. A
trial with no eligible workload is not ranked. Remaining trials are ordered by
lowest error percentage, lowest error count, then highest configured metric.

When failures occur, each benchmark artifact directory contains
`failed_requests.json` with the errored and incomplete details exposed by the
selected backend. Treat this file as workload data because it can contain
request arguments or outputs.

## Adaptive search repeats

Use an opt-in initial budget for search trials that are clearly behind a
completed reference:

```yaml
benchmark:
  repeats: 4
  min_repeats: 2
  adaptive_repeats:
    minimum_relative_score: 0.8
```

Each trial first runs `min_repeats` rounds. A round measures every named
benchmark once, so the decision never relies on a missing workload. If the
initial score is below 80% of the best completed tuned or baseline score
available when that trial starts, its remaining repeats are skipped. A score
equal to the threshold continues. The comparison runs only when both scores are
positive; otherwise the full repeat budget runs.

The first trial runs all repeats when no reference exists. In parallel mode,
the first simultaneous group normally has no completed reference and therefore
runs fully. Later trials use only results completed before they start. Completion
order can change which reference later trials receive.

`minimum_relative_score` must be greater than zero and at most one. Adaptive
repeats also require `min_repeats < repeats`. Trial JSON records the decision,
reason, initial and reference scores, planned budget, and actual repeat count by
workload. Fresh finalist validation always disables adaptive stopping and
requires its complete budget.

This policy saves whole benchmark repeats after the initial decision. It does
not interrupt a benchmark already running and does not yet enforce latency SLOs.

See the official
[GuideLLM dataset guide](https://github.com/vllm-project/guidellm/blob/main/docs/guides/datasets.md)
for file schemas and fields that vary by GuideLLM release.

## Fresh finalist validation

Enable a separate, fixed measurement budget after search:

```yaml
analysis:
  finalist_validation:
    top_k: 2
    repeats: 6
```

This selects the top two eligible tuned configurations from search, then reruns
the baseline (when present) and those candidates sequentially. Selection is
fixed before validation starts. Each candidate starts a fresh server and runs
six measured repeats of every named benchmark, retaining the configured warmup
policy. All six repeats are required for a validation score. Existing transient
failure retries can repeat an attempt; the budget applies to each attempt.

`top_k` must be a positive integer. `repeats` must be at least two and at least
`benchmark.min_repeats`. Defaults inside this section are two candidates and six
repeats. Omitting the section preserves the existing drift-triggered reruns.

Once validation starts, rankings use fresh measurements only. Initial search
scores and artifacts remain separate, and search-only candidates are not
promoted when a finalist fails. The report can say **No clear winner** if a
candidate fails, evidence is incomplete, workloads differ, drift is detected,
or measurement uncertainty overlaps. Close candidates do not trigger extra
repeats beyond the configured finalist budget in this stage. Mid-benchmark SLO
stopping and controlled ablations remain deferred.

Validation adds runtime, including server startups. Candidates run in fixed
order on their assigned devices; between-candidate drift or differences between
devices can still affect results. Keep the model, workload, and hardware
comparable. This is descriptive validation of selected candidates, not proof of
a global optimum or production performance.

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

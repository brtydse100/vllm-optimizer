# Reports

Open the printed `runs/<EXPERIMENT>/<RUN_ID>/report.html` path after a run.
It is a self-contained report; the same directory contains `result.json` and
`results.csv`. A backend's raw `results.json` is a different artifact.

## Read the result

Start with three questions:

1. **Did tuning help?** Compare baseline and best observed measurements.
2. **What should I try?** Inspect changed settings and the launch command.
3. **How much should I trust it?** Read the visible confidence verdict.

The summary compares **mean output throughput** and **average benchmark duration**
against the baseline. These measurements are separate from the configured
optimization score. Retaining the baseline does not claim an improvement.
Average benchmark duration excludes server startup and is not total trial time.

**Best observed** means the highest accepted score, not proof of a production
winner. Inconclusive evidence retains a warning. With fresh finalist validation,
the report shows **No clear winner** unless the validation supports one.
Use `selection_decision` for that conclusion in JSON, rather than inferring it
from `best`. Search associations are descriptive, not causal.

For constrained runs, **Performance targets** shows every workload's measured
statistic, unit, bound, and pass/fail status. Configurations missing any target
cannot win; the report explicitly says when no tested configuration qualifies.

## Use the settings

**Copy launch command** copies the selected settings. Expand **Full configuration
and launch command** for explicit vLLM YAML, environment values, and the POSIX
command. Replace redacted values before execution. Unspecified internal vLLM
defaults are not captured. Native server YAML is copied into accepted artifacts,
so reproduction uses that immutable copy rather than the original file.

## Inspect the evidence

| Report area | What to look for |
| --- | --- |
| Performance breakdown | Throughput, latency, failures, durations, baseline comparisons |
| Confidence and validation | Repeats, variability, drift, and accepted finalist evidence |
| All trials → Full details | Every trial, including failures/interruptions, settings, repeat timings |
| All recorded trial data (JSON) | Stored workload metrics, attempts, artifacts, and errors |
| Diagnostics and methodology | Missing evidence, failures, definitions, exploratory associations |

Comparisons require matching workload configurations and repeat coverage.
Missing samples, missing durations, or zero denominators make the affected
comparison unavailable. Different datasets/workloads are not combined to fill
a repeat budget. Backend P50/P95/P99 values are averaged across repeats;
they are **not pooled request percentiles**.

Confidence shows mean, sample standard deviation, range, repeat count, and
within-trial drift. Too few repeats, mismatched coverage, drift, or overlapping
ranges can make conclusions inconclusive. Range overlap is a descriptive
heuristic, not a significance test. Fresh validation keeps search observations
separate and excludes search-only candidates from accepted rankings.

Drift compares the first and second halves of at least four repeats within the
same trial/workload; `analysis.drift_threshold` defaults to `0.05` (5%). Without
explicit fresh finalist validation, the top two affected finalists can receive
drift-triggered reruns. Fresh validation requires separated repeat ranges and
95% Student's t intervals of workload means against every selected rival;
missing, failed, drifting, or mismatched evidence blocks a winner conclusion.
These intervals are descriptive guards, not a significance test of the aggregate
score. See [repeats and finalist validation](benchmarks/benchmark-repeats.md) for
budgets and adaptive stopping.

## Metric calculations

| Measurement | Meaning |
| --- | --- |
| Requests/s | Completed requests divided by measurement time |
| Output / total tokens/s | Generated tokens / prompt plus generated tokens per measurement time |
| TTFT | Request start to first generated token |
| TPOT | Average time per output token after the first token |
| ITL | Delay between consecutive streamed tokens |
| End-to-end latency | Request start through the complete response |
| P50 / P95 / P99 | Backend percentiles; P50 is the median |
| Elapsed | Benchmark subprocess launch through JSON parsing, excluding server startup |

The optimizer normalizes metric names/units without inventing missing
percentiles. Throughput change is `(selected - baseline) / baseline * 100`;
duration reduction is `(baseline - selected) / baseline * 100`.
The [scoring guide](search-and-scoring.md) explains the separate objective.

## Offline regeneration

Regenerate the report and exports without starting a model:

```bash
vllm-opt report --run runs/NAME/RUN_ID
```

Output goes to a separate directory. The source remains unchanged. Required
results/manifests are validated; accepted measurements determine rankings,
commands, charts, and conclusions. Missing validation manifests cannot fall back
to superseded search data. Stored scoring/constraint policies are checked
against accepted evidence. Legacy runs without those policies have fewer
integrity checks and retain their historical median scoring.

To reconsider request-failure eligibility using the same measurements:

```bash
vllm-opt reclassify --run runs/NAME/RUN_ID --max-failure-percentage 5
vllm-opt reclassify --run runs/NAME/RUN_ID --accept-any-request-failures
```

Reclassification writes new JSON/CSV/HTML below `reclassified` (or `--output`).
It preserves performance targets, accepted finalist budgets, and baseline
identity. Failed-repeat evidence must meet the new policy before becoming
eligible. [Retry and reproduction commands](../reference/cli.md) also preserve
the source run.

Each trial retains its resolved config, reproduction manifest, normalized
result, `vllm.log`, and benchmark logs. Raw backend JSON stays unchanged;
normalized results omit bulky generated text and token traces. Optional missing
or modified logs produce integrity warnings. Structured evidence errors stop
regeneration.

An optional [LLM summary](../reference/yaml/full-example-execution.md) uses a
named environment variable for its API key and requires HTTPS except on loopback.
The key is not persisted; name-based redaction cannot detect secrets in arbitrary
values. Summary failures become warnings and do not invalidate measurements.

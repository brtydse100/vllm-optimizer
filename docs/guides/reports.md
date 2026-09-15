# Reports

Every completed run writes a self-contained `report.html` alongside machine-
readable `result.json` and `results.csv` exports. `results.json` is reserved
for raw benchmark output where a backend writes one.

The report is designed to answer, at a glance:

1. Did tuning help?
2. Which configuration should I use?
3. How much should I trust it?

The first four sections are expanded: result overview, recommended configuration,
baseline comparison, and confidence. The overview identifies baseline wins,
the optimization metric, experiment wall time, and trial status counts.
The recommendation shows changed arguments and environment values, resolved
explicit vLLM YAML, selected environment variables, and a copyable POSIX launch
command. Unspecified vLLM internal defaults are not captured. Replace redacted
values before launching.

Comparisons preserve each workload's recorded configuration. They show output
and request throughput, TTFT, TPOT, end-to-end latency, and request failures,
with units and percentage changes. An overall summary and per-benchmark rows
also compare mean benchmark execution duration regardless of the optimization
objective. Values are arithmetic means across available
repeats; P50/P95/P99 columns are means of supplied backend percentiles, **not pooled
request percentiles**. Missing measurements and zero-denominator percentage
changes are unavailable. Different workload configurations are not paired.

Confidence shows individual repeats, mean, sample standard deviation, range,
repeat count, and signed drift percentage per workload. The drift value compares
the means of the first and second repeat halves and shows whether it exceeds the
configured threshold. Too few repeats, mismatched workloads,
drift, or overlapping repeat ranges make a comparison inconclusive. Range
overlap is a descriptive heuristic, not a significance test. Independent
production validation remains necessary. Initial search measurements are kept
separate from accepted finalist validation.

The sortable leaderboard keeps every trial, including duplicate configurations,
failures, and interruptions. It shows each trial's mean recorded benchmark
duration and percentage difference from the baseline; lower duration is labeled
better. Expand a row for individual benchmark execution durations. Missing
timings remain unavailable rather than inferred.

Detailed diagnostics and exploratory parameter associations are expandable.
Parameter importance describes associations across tested values, not causation.
The score-history chart uses recorded trial order, not elapsed time.
Scoring evidence includes each ranked trial's mean benchmark duration and its
percentage difference from the baseline; duration remains informational and does
not alter the configured optimization score.

## Metric calculations

- **Requests/s** is completed requests divided by benchmark measurement time.
- **Output tokens/s** is generated tokens divided by measurement time; total
  tokens/s includes prompt and output tokens.
- **TTFT** measures request start to first generated token.
- **TPOT** measures average time per output token after the first token.
- **ITL** measures delay between consecutive streamed tokens.
- **End-to-end latency** measures request start through complete response.
- **Median** is P50; **P99** means 99% of observations are at or below the value.
- **Elapsed** is measured by the `vllm-opt` CLI from benchmark subprocess launch through
  JSON parsing and excludes vLLM server startup.

Backends calculate request distributions and percentiles. vLLM Optimizer normalizes
names and units but does not derive missing percentiles. Eligible workloads are
averaged within an execution, repeated executions use the arithmetic mean score, and
named benchmark scores are averaged into the trial score.

When a finalist's sequential repeat means drift beyond `analysis.drift_threshold`
(5% by default), the top two affected finalists are rerun sequentially before
the recommendation is finalized. Their validation artifacts are kept under
`validation-001`; a failed validation removes that candidate from the ranking.

If `analysis.llm_summary` is configured, the report also includes a short
OpenAI-compatible summary. Its API key is read only from the named environment
variable and is never persisted. It requires HTTPS except for loopback HTTP.
Name-based redaction reduces accidental disclosure but cannot guarantee that
arbitrary user-provided values contain no secrets. An unavailable endpoint
becomes a report warning and never invalidates the experiment.

Each trial directory also contains its resolved configuration, normalized
result, reproduction manifest, `vllm.log`, and `benchmark.log`. Normalized
results omit token timings, request start times, and generated text. Input and
output lengths plus a sparse error map are retained in one compact `requests`
object. Raw
backend JSON remains available unchanged. Persistent values are secret-redacted.

## Offline regeneration

Regenerate `report.html`, `results.csv`, and a validated copy of `result.json`
from stored artifacts alone:

```bash
vllm-opt report --run runs/NAME/RUN_ID
```

The report uses reproduction manifests from the source run while writing all
new output to a separate directory. Required structured data (the run and
trial results and manifests) is validated and stops regeneration when invalid.
Rankings, displayed measurements, charts, YAML, and command export refer to the
accepted execution. A missing validation manifest never falls back to the
superseded search manifest. For runs retaining a scoring policy, regeneration
rejects ranking scores that disagree with accepted measurements. Legacy runs
without that policy cannot receive this additional score consistency check.

New runs record `benchmark_policy.repeat_aggregation: mean` and rank by the
arithmetic mean across repeats. Older runs without this field retain median
scoring when regenerated or reclassified, preserving their historical rankings.
This does not change backend P50/P95/P99 request percentiles.

To apply a different request-failure policy without starting vLLM or GuideLLM:

```bash
vllm-opt reclassify --run runs/RUN --max-failure-percentage 5
vllm-opt reclassify --run runs/RUN --accept-any-request-failures
```

This reads the stored benchmark measurements, evaluates request failures only
after each stored repeat is complete, recalculates eligibility and rankings, and
writes a new `result.json`, `results.csv`, and `report.html` below the source
run's `reclassified` directory (or `--output`). The source run is not modified.
Missing or changed optional logs and raw artifacts generate visible integrity
warnings. a5/a6 reports may have no execution assignment; a7 recomputes their
derived summaries from normalized trial data without changing the source run.

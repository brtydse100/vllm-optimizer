# Interpret results and explain why

## Establish what actually ran

Read `result.json` and `report.html` in the run directory. For a candidate, inspect
its trial `result.json`, resolved configuration, and accepted execution manifest.
The report also exports `results.csv`. Use the manifest corresponding to the
accepted measurement: fresh validation can supersede a trial's search execution.
Record model, GPU/device assignment, backend/version, workload, repeat count,
failure policy, and any concurrent system activity.

Check the exit code and run/trial statuses together. Exit `0` means at least one
trial completed; `1` means a runtime/artifact failure or no completed trials;
`2` indicates invalid arguments/configuration; `130` indicates interruption.
Completed and failed trials may coexist. Do not silently omit failures from the
explanation or treat a completed run as proof that all requests succeeded.

## Read the metrics in the user's terms

| Metric | Interpretation |
| --- | --- |
| `output_tokens_per_second` | Generated tokens/s; useful for batch generation when output lengths are representative. |
| `requests_per_second` | Completed request throughput; compare matching request-size distributions. |
| `total_tokens_per_second` | Input plus output token throughput, not identical to output generation speed. |
| TTFT (`time_to_first_token_ms`) | Delay until the first token, including effects such as queueing/prefill. Lower is usually better for interactive use. |
| TPOT (`time_per_output_token_ms`) | Time per generated token under backend accounting; useful for sustained streaming speed. |
| ITL (`inter_token_latency_ms`) | Gaps between streamed tokens; jitter/tails can matter even when average throughput is good. |
| End-to-end latency | Entire request duration. Canonical `end_to_end_latency_ms` is in milliseconds. |
| P50/P95/P99 | Backend request percentiles where available; verify the unit and source. Missing percentiles are unavailable, not zero. |
| Benchmark duration | Time to execute the benchmark; it is not an individual request's latency or the whole experiment's elapsed time. |
| Successful/errored/incomplete requests | Reliability and sample coverage. Inspect counts alongside any high score. |

New runs rank using the arithmetic mean across repeats, averaged across workload
points and named runs. They do not implement production traffic weighting.
Older stored runs without `benchmark_policy.repeat_aggregation` retain median
scoring during regeneration. Inspect recorded policy rather than rewriting history.

Report percentile comparisons are averages of backend percentiles across repeats,
not pooled percentiles over all requests. An average P99 is not a proof of a
production-wide P99 SLO. The normalizers may retain only selected percentiles;
use raw backend output for additional fields that exist there.

## Decide whether the comparison supports a recommendation

1. Verify matching model, hardware, workload configuration, route, data, and
   sampling/stopping settings. Changed inputs confound parameter comparisons.
2. Check every required workload's eligible repeats and request-failure policy.
   Errors, missing metrics, or incomplete coverage can exclude a candidate.
3. Compare baseline and candidate per workload: absolute throughput, latency,
   request counts, and repeat variability. For nonzero baseline `b`, change is
   `100 * (candidate - b) / b`. Positive latency change is worse.
4. Inspect mean, sample standard deviation, 95% Student's t interval, and drift.
   Drift compares early and late repeat halves and needs enough repeats (at least
   four). Two repeats and tiny workloads are exploratory.
5. Read `selection_decision.status`, `winner_trial_id`, and `reason` when present.
   `best` is the highest observed tuned score; it is not automatically a proven
   winner. Baseline can remain preferable. A missing selection decision in older
   artifacts limits the conclusion.
6. Apply the user's goals, including latency and reliability, to that evidence.
   A throughput leader that violates the latency target is not a suitable
   recommendation. Say when no measured candidate is acceptable.

With `analysis.finalist_validation`, selected candidates and available baseline
receive fresh sequential measurements. Search-only candidates cannot replace a
failed finalist. Incomplete, mismatched, drifting, or overlapping evidence can
yield **No clear winner**. The intervals are descriptive uncertainty guards,
not proof of a global optimum or a causal effect.

## Explain causes as testable hypotheses

Use this structure: **observation; possible mechanism; supporting evidence;
alternative explanation; smallest useful follow-up**. If the logs cannot
distinguish causes, say so instead of presenting one as established.

| Observation | Possible mechanism | Evidence or follow-up needed |
| --- | --- | --- |
| More sequences improve throughput but worsen latency | More batching or queueing increases utilization and waiting | Compare identical client load, TTFT/TPOT, queue metrics, and a controlled sequence-limit sweep. |
| Higher memory utilization helps concurrency | More KV-cache capacity reduces scheduling pressure | Inspect cache capacity, preemption/OOM logs, and matched workload runs. Other memory pressure can confound it. |
| Prefix caching helps long prompts | Reused prefixes avoid repeated prefill | Verify shared prefixes and cache evidence; compare cache on/off with identical requests and warmup policy. |
| Small-model tensor parallelism is slower | Communication overhead can exceed compute savings | Same model/load, matched hardware, TP-only comparison. Check topology and kernel behavior. |
| Throughput falls on longer outputs | More decode work or changed request mix | Compare achieved lengths, EOS handling, total duration, request rate, and per-token latency. |
| A candidate looks much faster with failures | Failed/short requests may change measured work | Inspect final counts, actual lengths, failure policy, and errors before interpreting speed. |
| Results vary or drift | Warmup, thermal/power behavior, external load, or order effects | Repeat under stable conditions; record device utilization/clocks and reverse or interleave candidate order if conducting a separate controlled experiment. |
| One attention/dtype choice changes speed | Different kernels or numerical precision | Confirm actual backend/dtype from startup logs; run one-factor comparisons and separate quality checks if precision changed. |

An adaptive search changes several parameters and chooses later trials based on
earlier scores. Parameter importance or correlations from that search are not
controlled ablations. Suggest a one-factor follow-up with the same model/data,
load, warmup, and repeat policy when explaining an individual parameter matters.
Obtain authorization before the additional GPU work.

## Investigate failed trials

Start with the recorded failure and artifact paths. `vllm.log` explains startup,
allocation, and server problems; `benchmark.log` explains backend invocation and
request completion. `failed_requests.json` may include sensitive request content.
Check actual paths in the manifest rather than assuming a flat directory layout.

Separate invalid configuration, unsupported upstream flags, OOM, startup failure,
benchmark timeout, HTTP/request errors, missing result metrics, and drain failure.
An OOM does not imply a bad model, and a timeout does not establish low throughput.
Never loosen the failure policy solely to manufacture a winning score.

## Useful follow-up commands

These operate on an existing run; replace the path and trial ID with real values:

```bash
vllm-opt report --run /absolute/run --output /absolute/new-report-directory
vllm-opt reproduce --run /absolute/run --trial trial-0001
vllm-opt export --run /absolute/run --trial trial-0001
```

`report` creates a new report/CSV/validated-result copy without GPU execution;
the destination must not exist. `reproduce` displays stored commands; `export`
prints the accepted execution's vLLM launch command to stdout. Neither executes
that command. Check redacted placeholders and native config dependencies before
using it. `export` does not accept `--output`.

```bash
vllm-opt reclassify --run /absolute/run --max-failure-percentage 2
```

This recalculates eligibility from existing measurements into a new destination.
It does not rerun the benchmark or fix failures. Explain how the policy change
affects conclusions and preserve the original evidence.

```bash
vllm-opt retry --run /absolute/run --trial trial-0001
```

`retry` performs new GPU work and needs authorization. It creates a linked run
from saved settings, preserving the accepted execution's budget. It does not
modify the source or accept `--config`. To change YAML, start a new experiment.

Finish with the best-supported recommendation, measured gains and costs, its
confidence limits, and a useful next step. Keep local evidence paths available;
do not upload workload data or logs without the user's authorization.

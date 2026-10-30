# Choose a workload and benchmark

## Translate the use case into measurable demand

Begin with an existing traffic trace or workload description. Record the model,
request route, input/output token distribution, arrival pattern, concurrency,
streaming requirements, repeated-prefix behavior, latency targets, error tolerance,
and run budget. Keep the workload the same across candidate server settings.

| Use case | Useful initial workload | Judge the result by |
| --- | --- | --- |
| Interactive chat | Representative chat prompts, realistic input/output lengths, expected concurrency or arrivals | TTFT and TPOT/tail latency, errors, then sustainable throughput |
| RAG or long documents | Long prompts with realistic output lengths and actual prefix reuse | TTFT/prefill cost, context capacity, cache behavior, and end-to-end latency |
| Offline/batch generation | Saturating load with representative output lengths | Output tokens/s or requests/s, completion rate, resource cost if measured |
| Fixed-rate API service | Constant arrivals or a Poisson arrival process at expected requests/s | Latency tails, failures, backlog/drain behavior at that rate |
| Recorded production traffic | Trace replay with matching request sizes/timestamps | Service targets under the trace, including bursts |
| Embeddings | Supported embedding model, route, and representative inputs | Requests/s and request latency; generation output tokens/s is usually inappropriate |

Synchronous load measures isolated requests. Fixed concurrency is closed-loop:
new work depends on completion of existing work. Constant/Poisson arrival rates
ask a different capacity question. Saturation throughput alone cannot show that
a service meets latency goals at production traffic levels.

If the user has no data, explain the synthetic assumption. A 32-input/16-output
token smoke test is useful for setup, but it does not represent a long-context
chat service. Prompt length, output length, early EOS, prefix reuse, and the
request route can each change measured results.

## Select the engine

**GuideLLM** is the optimizer's default and suits profiles, structured workload
data, and trace replay. A named run contains `profile`, optional `constraints`,
exactly one `data` item, and optionally `request_format` and `temperature`.
Use it when its load profiles match the user's demand.

**vLLM Bench Serve** suits existing `vllm bench serve` experiments or upstream
dataset/benchmark flags the user needs. A named run has `name` and `args`.
Use it when reproducing an upstream benchmark or needing a supported flag that
the GuideLLM adapter does not expose.

Both engines run real requests against the optimizer-managed server. They have
different workload generation and accounting; similar parameter names do not
make their scores interchangeable. Record engine/version and compare like with
like. Each experiment selects one engine.

## GuideLLM profiles

Choose one profile per run. These are profile fragments, not complete experiments:

```yaml
profile: {kind: synchronous}
```

```yaml
profile: {kind: concurrent, streams: [1, 4, 8], max_concurrency: 8}
```

```yaml
profile: {kind: throughput, max_concurrency: 8}
```

```yaml
profile: {kind: constant, rate: 5}
```

```yaml
profile: {kind: poisson, rate: 5}
```

`rate` is requests/s. A concurrent profile's `streams` can produce multiple
workload points in one invocation. A `sweep` uses `sweep_size` and
`strategy_type`; a `replay` uses timestamped data and `time_scale`. Their precise
semantics and accepted fields depend on the GuideLLM version. Consult its
[benchmark documentation](https://github.com/vllm-project/guidellm/blob/main/docs/getting-started/benchmark.md)
and the installed `guidellm run --help` when using them.

Bound a run with `max_requests.count`, `max_duration.seconds`, or both.
`max_requests` limits work; it does not serialize requests. GuideLLM may drain
in-flight requests before exiting. Use synchronous load for one-at-a-time work.
Multiple profile points and multiple named runs increase the total work.

## Dataset selection

- `synthetic_text`: specify prompt/output token lengths and optionally standard
  deviations and bounds. Include a realistic length distribution when known.
- `json_file`, `csv_file`, `text_file`, `parquet_file`, `arrow_file`, `hdf5_file`,
  `db_file`, or `tar_file`: use `path`. JSONL also uses `json_file`. Inspect the
  installed backend's required columns before promising compatibility; the
  optimizer does not expose GuideLLM's custom column mapper in this schema.
- `huggingface` (alias `hf`): use `source` and optional `load_kwargs`, such as a
  split. Remote access/downloads still need the user's authorization.
- `trace_synthetic`: use `path`, `timestamp_column`, `prompt_tokens_column`,
  and `output_tokens_column`, paired with replay. Other trace and multimodal
  kinds are upstream/version-specific and need matching model support.

Use the [GuideLLM dataset reference](https://github.com/vllm-project/guidellm/blob/main/docs/guides/datasets.md)
for the exact format. Do not fabricate file columns or place multiple dataset
entries in a single run. Add distinct named runs for separate datasets.

## vLLM alternative for the small smoke experiment

Replace the complete `benchmark` section of the [YAML example](yaml.md) with:

```yaml
benchmark:
  engine: vllm
  repeats: 2
  min_repeats: 2
  max_failure_percentage: 0
  runs:
    - name: short-completions
      args:
        dataset-name: random
        random-input-len: 32
        random-output-len: 16
        random-prefix-len: 0
        num-prompts: 8
        request-rate: inf
        max-concurrency: 2
        percentile-metrics: ttft,tpot,itl,e2el
        metric-percentiles: 50,95,99
        ignore-eos: true
        temperature: 0
        seed: 42
```

`request-rate: inf` offers requests as fast as allowed by concurrency. Set a finite
rate to study that arrival load. `ignore-eos` requests the full output length,
which improves length control but may differ from production stopping behavior.
With eight requests, tail percentiles are exploratory, not stable SLO evidence.

The optimizer owns `backend`, `model`, `host`, `port`, `base-url`, `save-result`,
`append-result`, `result-dir`, and `result-filename`; omit those from `args`.
Other flags pass through to the installed vLLM CLI. Here boolean `true` adds a
flag and `false` omits it; this differs from `server` boolean rendering.

Dataset choices such as `sharegpt`, `hf`, `custom`, and `prefix_repetition` have
version-specific arguments. Check [vLLM Bench Serve's reference](https://docs.vllm.ai/en/latest/cli/bench/serve/)
for the installed version before using them.

## Budget and comparison design

Use two measured repeats for an initial bounded smoke test; use at least the
default four for more informative comparisons, and enough requests to study the
requested percentile. Warmups are separate discarded executions. A rough budget
is candidates (plus baseline) times named runs times (warmups + measured repeats),
plus server startups, backend profile points, retries, and validation reruns.

An exhaustive grid's size is the product of candidate counts. Random/TPE need
`trials`; grid forbids it. Adaptive repeats can save measurement rounds for weak
candidates; they do not implement a live latency stopping rule.

## Optimize throughput within performance targets

In `0.1.0a21`, replace the example's `optimization` section with this fragment
when testing constrained optimization. These are illustrative smoke-test limits;
replace them with the user's actual requirements for a meaningful experiment:

```yaml
optimization:
  maximize: output_tokens_per_second
  sampler: grid
  constraints:
    - metric: time_to_first_token_ms
      statistic: p99
      max: "1 s"
    - metric: requests_per_second
      min: "1 req/s"
```

This maximizes output throughput among candidates whose mean backend P99 TTFT
is at most 1000 ms and mean request throughput is at least 1 request/s for every
matched workload. The throughput statistic defaults to `average`. Equality
passes. Use `max: "200 ms"` instead when that is the user's actual TTFT target.

`optimization.constraints` are performance targets; the GuideLLM run's
`constraints: [{kind: max_requests, count: 8}]` is a stopping condition. Keep
these in their own sections. Choose a backend that actually emits the metric
and percentile; unavailable measurements fail the performance target.

Targets are assessed across eligible repeats and do not stop a benchmark or
the search budget early. All limits must pass for every workload, including
the baseline and fresh finalists. Target misses retain measurements but cannot
win; if none pass, report that no tested configuration met all targets.
TPE receives objective scores and separate constraint violations; Grid and
Random also filter candidates by feasibility.

Do not set `optimization.maximize` to a latency metric: larger latency would win.
Direct latency minimization is not supported in this schema. A passing mean of
backend percentiles from a tiny smoke run does not establish a production SLO;
inspect repeat coverage, request counts, and variability in the results.

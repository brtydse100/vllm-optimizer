# Configuration

For automatic candidate selection and optional synthetic workload calibration,
see [automatic tuning](auto-tuning.md). Explicit search spaces remain supported.

Start with the [complete quick-start example](../getting-started/index.md),
then change the sections you need. The [YAML reference](../reference/yaml/full-example.md)
contains the full configuration.

| Section | Purpose |
| --- | --- |
| `server` | Local model and fixed vLLM flags, or a native vLLM YAML file |
| `tune` | vLLM flags to search |
| `env` / `tune_env` | Fixed / tunable server environment variables |
| `benchmark` | Backend, workloads, repeats, and request-failure policy |
| `optimization` | Objective, sampler, budget, and performance targets |
| `execution` | Process controls and optional parallel workers |
| `analysis` | Drift checks and optional fresh finalist validation |
| `logging` / `timeouts` | Verbosity and startup/benchmark limits |

## Model and server

`server.model` points to an existing local directory; models are not downloaded.
Other `server` entries become fixed vLLM flags. Use `tune` for values to vary:

```yaml
server:
  model: /models/qwen
  gpu-memory-utilization: 0.8
  enforce-eager: true
tune:
  max-num-seqs:
    values: [8, 16, 32]
env:
  CUDA_VISIBLE_DEVICES: "0"
```

To reuse native vLLM YAML, set `server.config: ./vllm-config.yaml`.
That path is relative to the experiment file. An inline `server.model`
overrides the native model; fixed flags, tuned values, and runtime assignments
override corresponding native settings through command-line arguments.
See [native configuration and LMCache](../reference/yaml/full-example-server.md#native-server-configuration-and-lmcache).

Unknown vLLM flags are allowed so new upstream options remain usable.
Hyphens and underscores identify the same flag; defining both spellings in
one section is rejected. Tuned settings override fixed aliases.
Environment names are case-sensitive.

### Fixed value rendering

Scalars emit flag/value pairs; lists emit one flag followed by all values.
`true` emits a flag, `false` its `--no-` form, and `null` omits it.
See the [server reference](../reference/yaml/full-example-server.md) for examples.

### Tunable arguments and environment

Use `values: [...]` for strings, numbers, booleans, or `null`.
Numeric searches may instead use inclusive `min`, `max`, and `step`.
`tune_env` uses the same forms; environment values are converted to strings.
Quote numeric-looking fixed values such as `"0"`.

## Parallel local trials

Execution is sequential by default. [Parallel trials](parallel-trials.md)
require explicit, non-overlapping GPU workers and a port range. The CLI assigns
each worker's GPU visibility and port; do not configure these yourself in
parallel mode. Each trial's tensor-parallel size must fit a worker.

Sequential runs use port 8000 unless overridden by `server.port` or tuning.
Fixed server settings run first as the baseline unless disabled.

## Benchmark runs

| Backend | Workload fields | Guide |
| --- | --- | --- |
| GuideLLM (default) | `profile`, `constraints`, `data` | [Benchmarking](benchmarks/benchmarking.md) |
| vLLM Bench Serve | `args` | [Complete example](../reference/yaml/full-example-vllm-bench.md) |

Every named run tests the same server configuration. The default is four
measured repeats, all required for ranking, with no discarded warmups.
[Repeats and errors](benchmarks/benchmark-repeats.md) explains `min_repeats`,
warmups, adaptive budgets, and fresh finalist validation.

Benchmark-run `constraints` control GuideLLM execution, such as request counts.
To set latency or throughput limits for optimization, use
[`optimization.constraints`](search-and-scoring.md#performance-targets).

### Request failure policy

The default `benchmark.max_failure_percentage: 0` excludes workloads with errored
or incomplete requests from ranking. Set a percentage from 0 to 100 to allow
up to that inclusive limit,
or `accept_any_request_failures: true` to ignore the percentage.
At least one successful request and a usable metric are always required;
request-count runs must also supply their expected total.
See [repeats and errors](benchmarks/benchmark-repeats.md).

## Logging and timeouts

```yaml
logging:
  level: INFO
timeouts:
  startup: 15m
  benchmark: 20m
```

Timeouts accept numeric seconds or durations such as `30s`, `15m`, and `1h`.
Omitted benchmark timeouts are derived from GuideLLM duration limits, use a
one-hour cap for request-only GuideLLM runs, or default to 180 seconds for
vLLM Bench Serve. The literal `auto` is invalid.

Use `--verbose` or `logging.level: DEBUG` to stream logs. Per-trial logs are
always saved. Failed-request details, when available, are saved beside raw JSON
as `failed_requests.json`.

After benchmarks, `execution.drain_grace` allows vLLM to become idle (default
15 seconds, must be positive). Missing request metrics or a busy server after
that window fails the trial. Other process controls are in the
[execution reference](../reference/yaml/full-example-execution.md).

`vllm-opt validate -c experiment.yaml` checks configuration and constructed
commands before a run is created. GPU capacity, model weights, and upstream
flag support are checked only by actual execution.

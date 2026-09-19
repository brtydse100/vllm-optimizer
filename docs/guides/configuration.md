# Configuration

vLLM Optimizer keeps server settings separate from benchmark workloads so every server
configuration can be compared under the same demand.

## Model and server

Choose either inline server settings or a native vLLM YAML file. For inline
settings, `server.model` is required and must point to an existing local model
directory. Every other `server` entry is a fixed vLLM argument. Top-level
`tune` defines the vLLM argument search space.

```yaml
server:
  model: /models/qwen
  tensor-parallel-size: 2
  enforce-eager: true
tune:
  max-num-seqs:
    values: [64, 128, 256]
  gpu-memory-utilization:
    min: 0.85
    max: 0.95
    step: 0.05
env:
  CUDA_VISIBLE_DEVICES: "0,1"
```

To reuse a native vLLM configuration, set `server.config`:

```yaml
server:
  config: ./vllm-config.yaml
  port: 8100  # Optional override.
```

The path is resolved relative to the vLLM Optimizer configuration file. The
native file must contain `model` unless supplied inline; either form is subject
to local model-directory validation. vLLM Optimizer launches `vllm serve` with both the
resolved model and `--config`. Other `server` settings, selected `tune` values,
and runtime-assigned settings are command-line arguments, so they override the
corresponding native values. If an inline `server.model` is supplied too, it
overrides the native model.

See [native server configuration and LMCache](../reference/yaml/full-example-server.md#native-server-configuration-and-lmcache)
for nested transfer settings and inline JSON examples.

Unknown vLLM flags are intentionally allowed. The `vllm-opt` CLI renders keys as CLI flags,
which keeps new vLLM options usable without a vLLM Optimizer release.
Underscores and hyphens identify the same CLI argument, and tuned values override
fixed aliases. Defining both spellings in the same section is rejected as
ambiguous. Environment-variable names remain exact and case-sensitive.

The complete configuration is validated before a run directory or process is
created. This includes nested benchmark options, search values, ports, GPU
assignments, timeouts, and the rendered vLLM and benchmark commands.

### Fixed value rendering

Fixed `server` values map directly to vLLM flags:

```yaml
server:
  model: /models/qwen
  enforce-eager: true       # emits --enforce-eager
  enable-prefix-caching: false  # emits --no-enable-prefix-caching
  lora-modules: [a=/a, b=/b]   # emits one flag with both values
```

`null` omits a flag. `true` emits a presence flag and `false` emits its
`--no-` form. Scalars emit a flag/value pair. Lists emit one flag followed by
all values so parsers retain the complete list.

### Tunable arguments and environment

`tune` and `tune_env` accept categorical `values` or inclusive `min`, `max`,
`step` ranges. Fixed environment values use `env`. Environment values are
converted to strings; quote `"0"` and `"1"` for clarity. See the
[server and tuning reference](../reference/yaml/full-example-server.md) for
complete categorical, boolean, range, and environment examples.

## Parallel local trials

Sequential execution is the default. To run separate vLLM instances at the
same time, configure explicit GPU workers and a port range:

```yaml
execution:
  mode: local_parallel
  max_parallel_trials: 2
  gpu_allocation:
    workers:
      - name: worker-0
        devices: [0, 1]
      - name: worker-1
        devices: [2, 3]
  ports:
    min: 8100
    max: 8199
```

GPU sets must not overlap. The `vllm-opt` CLI assigns `CUDA_VISIBLE_DEVICES` and one stable
port to each worker, so do not configure either yourself in parallel mode.
`max_parallel_trials` must equal the declared worker count, and every trial's
`tensor-parallel-size` must fit at least one worker. The baseline runs alone
first; tuned trials then run concurrently. See
[parallel trials](parallel-trials.md) for scheduling and measurement rules.

For sequential execution, including one vLLM server using tensor parallelism
across several GPUs, the `vllm-opt` CLI always passes a concrete port. It defaults to 8000;
set `server.port` only when another port is required. Tuned `host` and `port`
values drive both the launched server and its readiness endpoint. Local-parallel
worker ports override trial values as described above.

## Benchmark runs

`benchmark.engine` selects `guidellm` (default) or `vllm`. A single trial may
contain several benchmark runs, but they all evaluate the same running server
configuration. GuideLLM runs use `profile`, `constraints`, and `data`; vLLM
Bench Serve runs use `args`. The `vllm-opt` CLI preserves raw JSON and `benchmark.log`, then
exposes normalized metrics to scoring and reports.

Set `benchmark.warmup_repeats` to a positive number when discarded warmups are
needed; warmups are disabled when this setting is omitted. Set
`benchmark.min_repeats` to require enough measured repeats for ranking. The
default minimum is 4 for fixed-repeat runs and 2 when `adaptive_repeats` is
enabled. The effective default minimum is capped at `repeats`. Every configured run
must meet the minimum and failure policy or the
trial is not ranked. Set
`analysis.drift_threshold` to change the sequential finalist rerun threshold
(default 0.05).

The optional `benchmark.adaptive_repeats.minimum_relative_score` skips remaining
search repeats after `min_repeats` rounds when a positive initial score is more
than 30% below the baseline arithmetic mean. Its default threshold is `0.7`.
See [Benchmark repeats and
errors](benchmarks/benchmark-repeats.md#adaptive-search-repeats).

### Request failure policy

The default is strict: any errored or incomplete request excludes the
benchmark from ranking.

```yaml
benchmark:
  max_failure_percentage: 0
```

Set `max_failure_percentage` from `0` through `100` to accept a benchmark with
up to that percentage of errored or incomplete requests. The boundary is
inclusive, so this example accepts exactly 2% failures:

```yaml
benchmark:
  max_failure_percentage: 2
```

To ignore the percentage entirely, use:

```yaml
benchmark:
  accept_any_request_failures: true
```

The failure policy is evaluated only after the benchmark finishes. Every
accepted benchmark must still contain at least one successful request and a
usable metric. Normalized JSON, CSV, and HTML results show successful and
failed request counts and the failure percentage for every repeat.

Every supported profile, constraint, request format, and dataset form has
copyable examples in [benchmark configuration](benchmarks/benchmarking.md). See the
[complete YAML](../reference/yaml/full-example.md) for all configuration sections together.

## Logging and timeouts

```yaml
logging:
  level: INFO
timeouts:
  startup: 15m
  benchmark: 20m
```

Logging levels match GuideLLM: `DEBUG`, `INFO`, `WARNING`, `ERROR`, and
`CRITICAL`. Both timeouts accept seconds or values such as `30s`, `15m`, and
`1h`. For GuideLLM, omitting `timeouts.benchmark` derives it from the duration
constraint plus a safety margin. A run constrained only by `max_requests` uses
a documented one-hour hard cap when no explicit timeout is provided; set
`timeouts.benchmark` explicitly for longer workloads. vLLM Bench Serve uses a
180-second default when no explicit timeout is provided. The literal value
`auto` is not accepted.

`benchmark.log` is flushed continuously. Request-limited runs show processed
requests against their limit; duration-only runs show elapsed time against the
configured duration. If a backend reports failed or incomplete requests, their
available details are also saved beside the raw result as `failed_requests.json`.

After each benchmark, vLLM Optimizer polls vLLM's running and waiting request metrics.
`execution.drain_grace` controls this drain window and defaults to 15 seconds.
It must be positive. Missing metrics or a server that remains busy fails the trial.

# Build and explain optimization YAML

This reference covers optimizer-owned fields in the inspected `0.1.0a19` schema
and common upstream fields. `server`, `tune`, environment variables, and backend
options are open-ended: the installed upstream release determines valid names,
values, defaults, and hardware support. Look up an unfamiliar flag before
explaining it; configuration acceptance alone does not prove it works.

## A complete bounded starting point

This is an exploratory smoke experiment for a small local completion model.
Replace the model/output paths and resource limits for the actual host. Check
available VRAM before choosing `gpu-memory-utilization`; 0.4 is an example, not
a universal safe setting. Choose a workload using [benchmarks](benchmarks.md).

```yaml
schema_version: 1
experiment:
  name: first-optimization
  output_dir: /absolute/private/output
  seed: 42
server:
  model: /absolute/local/model
  host: 127.0.0.1
  port: 8000
  dtype: auto
  gpu-memory-utilization: 0.4
  max-model-len: 256
  max-num-seqs: 1
  enforce-eager: true
tune:
  max-num-seqs:
    values: [1, 2]
benchmark:
  engine: guidellm
  repeats: 2
  min_repeats: 2
  max_failure_percentage: 0
  runs:
    - name: short-completions
      request_format: /v1/completions
      temperature: 0
      profile: {kind: throughput, max_concurrency: 2}
      constraints: [{kind: max_requests, count: 8}]
      data: [{kind: synthetic_text, prompt_tokens: 32, output_tokens: 16}]
baseline:
  enabled: true
optimization:
  maximize: output_tokens_per_second
  sampler: grid
timeouts:
  startup: 5m
  benchmark: 3m
execution:
  mode: sequential
```

This tests a fixed baseline with one sequence, then two grid candidates. One
candidate deliberately repeats the baseline setting. The baseline is separate
from tuned rankings. Two repeats give limited confidence; four is the default
when `repeats` is omitted. Increase workload size and repeats only after the
smoke run succeeds and the user authorizes the larger budget.

`enforce-eager: true` avoids graph capture for this small smoke test; it changes
execution behavior and may reduce performance. Revisit it for a real search.
The CLI may perform drift-triggered reruns in addition to the initial budget.

## How to explain a user's file

For each requested field, give: **full YAML path; value; meaning and units;
default/required status; why it was chosen; interaction or trade-off**. For
example, `tune.max-num-seqs.values: [1, 2]` tests two server sequence limits;
it overrides the fixed `server.max-num-seqs: 1` for tuned trials. It does not
change the benchmark client's `max_concurrency: 2`.

When asked to explain the whole file, cover every populated field, including
nested lists and pass-through flags. Distinguish user choices, optimizer defaults,
and upstream defaults. Do not infer defaults that depend on GPU/model/version.

## Root, experiment, and model

| Field | Meaning, default, and interactions |
| --- | --- |
| `schema_version` | Only `1` is accepted; defaults to `1`. |
| `experiment` | Required experiment metadata mapping. |
| `experiment.name` | Required label starting with a letter/number; remaining characters may include letters, numbers, `_`, and `-`. Used in output layout. |
| `experiment.output_dir` | Output parent, default `runs`, relative to the process working directory. Use an absolute private path when needed. |
| `experiment.seed` | Optional integer sampler seed; it does not guarantee identical GPU timing, backend data, or parallel TPE completion order. |
| `server` | Required fixed vLLM configuration. Flags pass through to `vllm serve`. |
| `server.model` | Existing local model directory. Required inline or in native config; not a model-download ID. Inline relative paths resolve from the experiment YAML directory. Cannot be tuned. |
| `server.config` | Optional native vLLM YAML, relative to the experiment YAML. A model path inside it resolves relative to that native file. Inline model overrides native model. |

Explicit `server` values override corresponding native settings; `tune` selections
override fixed values; runtime worker assignments take precedence where applicable.
Use native YAML for nested upstream configurations (for example, KV transfer).
Inline mappings are not accepted as ordinary server flag values; JSON strings
may be used when the upstream flag expects JSON.

## Common server parameters

These effects describe the knobs, not guaranteed improvements. Confirm the
installed vLLM accepts the flag and the hardware/model supports the value.

| Field | Meaning and trade-off |
| --- | --- |
| `server.host` | Bind address. Optimizer defaults to loopback via `execution.host`; `0.0.0.0` exposes the service on all interfaces. |
| `server.port` | Sequential server port, default `8000`. Drives readiness and benchmark targeting. |
| `server.tensor-parallel-size` | GPUs cooperating on one model instance. Needs sufficient assigned devices; communication can outweigh gains for small models. |
| `server.dtype` | Model execution dtype. `auto` delegates selection to vLLM; supported precision depends on model/GPU. |
| `server.gpu-memory-utilization` | Fraction of total GPU memory budgeted for this instance; it is not a fraction of currently free memory. More cache capacity can improve concurrency but leaves less headroom. |
| `server.max-model-len` | Maximum sequence context in tokens, including prompt and generation. Must fit model/runtime constraints; larger contexts can require more cache. |
| `server.max-num-seqs` | Maximum sequences processed together. More parallelism can improve throughput while increasing memory use or latency. Distinct from client concurrency. |
| `server.max-num-batched-tokens` | Token budget per scheduler iteration. Affects batching/prefill-decode interaction; allowed combinations depend on chunked prefill and version. |
| `server.enforce-eager` | Eager execution instead of graph-based execution paths. Can simplify startup/memory behavior at a performance cost. |
| `server.enable-prefix-caching` | Reuses eligible shared prompt prefixes. Benefit requires actual prefix reuse; random unrelated prompts may not benefit. |
| `server.enable-chunked-prefill` | Splits prefill across iterations where supported. May change responsiveness, throughput, and batch-limit constraints. |
| `server.attention-backend` | Attention implementation. Availability depends on architecture, dtype, installation, and vLLM version. |
| `server.kv-cache-dtype` | KV cache precision. May change cache capacity, numerical behavior, and supported kernels; performance tests do not establish answer quality. |
| `server.quantization` | Model quantization method, requiring a compatible model and backend. This is not a generic way to quantize arbitrary weights. |
| `server.served-model-name` | API model alias. Check benchmark/model-name compatibility if overriding it. |
| `server.lora-modules` | List of adapter declarations accepted by vLLM; use compatible local adapters and upstream syntax. |

Server scalars render as flag/value pairs; lists as one flag with multiple values.
`true` emits `--flag`, `false` emits `--no-flag`, and `null` omits the flag.
Not every upstream flag supports a negative form; check its help. Hyphens and
underscores identify the same flag; duplicate aliases in one section are rejected.

## Search space and environment

| Field | Meaning, default, and interactions |
| --- | --- |
| `tune` | Mapping of vLLM flags to candidate definitions; default empty. Keep only meaningful, supported choices. |
| `tune.<flag>.values` | Nonempty categorical alternatives, including scalar/boolean choices, `null` where omission is intended, or supported list-valued alternatives. |
| `tune.<flag>.min` / `.max` / `.step` | Numeric range with positive step. Endpoints bound the range inclusively; the step determines which values are reached. Use this form or `values`, not both. |
| `env` | Fixed environment variables inherited by server trials. Values become strings; names are case-sensitive. |
| `env.CUDA_VISIBLE_DEVICES` | Selected device IDs for sequential execution; tensor parallelism uses these devices. Omit in local-parallel mode. |
| `tune_env` | Environment search space using the same `values` or numeric range form. Selected values override `env`. |
| `tune_env.<name>.values` / `.min` / `.max` / `.step` | Candidate strings or range values converted to environment strings. Quote numeric-looking categorical values such as `"0"` and `"1"`. |

The process inherits its host environment too. Do not put credentials in fixed or
tunable environment YAML. Unknown environment names may have no effect; verify
that the installed upstream runtime reads the variable.

## Benchmark-wide fields

| Field | Meaning, default, and interactions |
| --- | --- |
| `benchmark.engine` | `guidellm` (default) or `vllm`; determines the run schema. |
| `benchmark.repeats` | Positive measured repeat count per named run; default `4`. |
| `benchmark.warmup_repeats` | Nonnegative discarded runs before measurements, once per named run per trial attempt; default `0`. Adds runtime. |
| `benchmark.min_repeats` | Minimum eligible measured repeats per required workload. Defaults to `min(4, repeats)`, or `min(2, repeats)` with adaptive repeats. Must be positive and no greater than `repeats`. |
| `benchmark.max_failure_percentage` | Allowed errored + incomplete percentage, inclusive `0..100`, default `0`. Applied after a benchmark finishes, not as a live stopping rule. |
| `benchmark.accept_any_request_failures` | Boolean, default `false`; when true overrides the percentage threshold. Still requires a successful request and usable score. |
| `benchmark.adaptive_repeats` | Optional mapping enabling reduced repeat budgets for clearly weak search candidates. Requires `min_repeats < repeats`. |
| `benchmark.adaptive_repeats.minimum_relative_score` | Default `0.7`, must be `>0` and `<=1`. After initial rounds, a positive score below this fraction of a positive baseline mean skips remaining repeats. Missing/nonpositive baseline means full budget. |
| `benchmark.runs` | Required nonempty list of named workloads. All evaluate the same server settings in each trial. |
| `benchmark.runs[].name` | Required unique name with the same character restrictions as `experiment.name`. |

Failure acceptance trades strictness for sample availability; do not loosen it
just to make a failing candidate rank. Every required workload must meet its
repeat/failure policy. A successful exit does not prove every request succeeded.

## GuideLLM run fields

| Field under `benchmark.runs[]` | Meaning and interactions |
| --- | --- |
| `request_format` | Route, default `/v1/completions`. Chat, responses, and embeddings require compatible model/backend support. |
| `temperature` | Generation sampling temperature, default `0`, finite `0..2`. Nonzero values warn because output variability can affect timing. Explicit temperature on non-generation routes is rejected. |
| `profile.kind` | Required load profile, such as `synchronous`, `throughput`, `concurrent`, `constant`, `poisson`, `sweep`, or `replay`; verify upstream version support. |
| `profile.streams` | Concurrent-profile stream-count choices, potentially multiple workload points. |
| `profile.max_concurrency` | Maximum concurrent client requests where supported by the profile. It does not set the server's sequence limit. |
| `profile.rate` | Offered request rate in requests/s for constant/Poisson profiles. |
| `profile.rampup_duration` | Profile ramp-up duration in seconds where supported. |
| `profile.sweep_size` / `.strategy_type` | Number and strategy of sweep points; consult installed GuideLLM semantics. |
| `profile.time_scale` | Replay timing scale; `1.0` preserves original timing. Verify upstream semantics before changing it. |
| `constraints` | Optional list of backend stopping conditions. Add a request/duration bound for practical experiments. |
| `constraints[].kind: max_requests` / `.count` | Request-count stopping condition with a positive count; does not enforce serialization. |
| `constraints[].kind: max_duration` / `.seconds` | Duration bound per profile strategy, accepting seconds or duration strings such as `2m`. |
| `data` | Exactly one dataset mapping per named run. |
| `data[].kind` | Dataset loader/generator; see the benchmark reference for supported forms. |
| `data[].prompt_tokens` / `.output_tokens` | Synthetic input/output length parameters in tokens. Include context and formatting headroom. |
| `data[].prompt_tokens_stdev` / `.output_tokens_stdev` | Synthetic token-length standard deviations, where supported. |
| `data[].prompt_tokens_min` / `.prompt_tokens_max` | Bounds on synthetic input lengths. |
| `data[].output_tokens_min` / `.output_tokens_max` | Bounds on synthetic output lengths. |
| `data[].turns` | Synthetic conversation turn count where supported; choose it to match request format/model. |
| `data[].path` | Local file or trace path. Use absolute paths; the adapter passes these through. |
| `data[].source` | Hugging Face dataset ID or local source for `huggingface`/`hf`. |
| `data[].load_kwargs` | Dataset-loader options, such as `split`. Fields follow the installed backend. |
| `data[].timestamp_column` | Trace field holding arrival timestamps. |
| `data[].prompt_tokens_column` / `.output_tokens_column` | Trace fields holding input/output lengths. |

Nested backend fields pass through, so not all invalid profile/dataset values are
caught by optimizer validation. Check the backend docs/help and run a smoke test.

## vLLM Bench Serve run fields

Only `name` and `args` belong directly on each vLLM run. Use the replacement
benchmark example in [benchmarks](benchmarks.md).

| Field under `benchmark.runs[].args` | Meaning and interactions |
| --- | --- |
| `dataset-name` / `dataset-path` | Dataset loader and its input location; accepted loaders/formats depend on vLLM. |
| `random-input-len` / `random-output-len` / `random-prefix-len` | Random-workload input, output, and shared-prefix sizes in tokens. |
| `num-prompts` | Number of benchmark requests. Small values give noisy tail estimates. |
| `request-rate` | Offered requests/s, or `inf` for maximum offered load. |
| `max-concurrency` | Maximum outstanding client requests. |
| `percentile-metrics` / `metric-percentiles` | Metrics and percentiles requested from the backend. Reporting may normalize only a subset; retain raw output. |
| `ignore-eos` | Request full output length instead of early EOS stopping. Changes workload realism. |
| `temperature` | Sampling temperature, default `0`, finite `0..2`. Nonzero warns. |
| `seed` | Backend random-workload seed; separate from the optimizer sampler seed. |
| `extra-body` | JSON string of request-body overrides; malformed JSON is rejected. Effective temperature can override `args.temperature`. |
| `extra-body.<field>` | Dotted request-body field. When dotted fields coexist with a whole JSON body, the dotted object replaces that body under the inspected parser behavior. |

Boolean false omits a benchmark flag; server false emits a negative flag.
List values emit one flag followed by all items. Hyphen/underscore aliases are
normalized and duplicates rejected. Multiple completions (`n > 1`) need explicit
nonzero effective temperature. Omit optimizer-owned connection/output arguments
listed in the benchmark reference.

## Baseline, optimization, analysis

| Field | Meaning, default, and interactions |
| --- | --- |
| `baseline.enabled` | Boolean, default `true`; measures fixed settings separately before tuning. Set meaningful fixed values for a useful comparison. |
| `optimization.maximize` | Required metric name; larger values rank higher. Use a metric actually produced by the backend, e.g. `output_tokens_per_second` or `requests_per_second`. Latency minimization is not implemented in this schema. |
| `optimization.sampler` | `grid` (default), `random`, or `tpe`. Grid enumerates the space; TPE uses previous observations to guide proposals. |
| `optimization.trials` | Positive number required for random/TPE; forbidden for grid. Unique search configurations are not repeated; oversized budgets are capped with a warning. |
| `analysis.drift_threshold` | Finite nonnegative relative drift threshold, default `0.05` (5%). Too few repeats cannot establish drift. Raising it does not remove measurement instability. |
| `analysis.finalist_validation` | Optional fresh fixed-budget validation of selected finalists and available baseline. Omitting it retains drift-triggered reruns. |
| `analysis.finalist_validation.top_k` | Positive number of eligible tuned finalists, default `2` when enabled. |
| `analysis.finalist_validation.repeats` | Fresh repeats per workload/candidate, default `6`; at least `2` and at least the configured minimum repeats. All are required; adaptive stopping is disabled. |
| `analysis.llm_summary` | Optional API-generated summary. Sends ranking/parameter information; enable only with appropriate authorization. A summary failure is a report warning. |
| `analysis.llm_summary.base_url` | Required OpenAI-compatible API base URL; HTTPS except loopback HTTP. |
| `analysis.llm_summary.model` | Required summary model identifier, distinct from the benchmarked model. |
| `analysis.llm_summary.api_key_env` | Required name of the environment variable containing the credential, never the credential itself. |
| `analysis.llm_summary.timeout` | Positive API timeout in seconds, default `30`. |

The score uses arithmetic means across eligible repeated workload measurements,
then averages workload points and named runs. It is not weighted by production
traffic volume. Explain individual workloads as well as the aggregate.

## Timeouts, logging, execution

| Field | Meaning, default, and interactions |
| --- | --- |
| `timeouts.startup` | Server readiness timeout; default `15m`. Numeric seconds and strings like `30s`, `15m`, `1h` are accepted. |
| `timeouts.benchmark` | Per benchmark timeout in seconds/duration form. Omitted GuideLLM duration-bounded runs derive a timeout with margin; request-only runs have a 1h cap. vLLM defaults to 180 seconds. Literal `auto` is invalid. |
| `logging.level` | `DEBUG`, `INFO` (default), `WARNING`, `ERROR`, or `CRITICAL`. CLI `--verbose` streams logs and sets DEBUG for run/retry. |
| `execution.mode` | `sequential` (default) or `local_parallel` for independent simultaneous trial servers. This differs from one tensor-parallel model across GPUs. |
| `execution.host` | Default bind/connection host, `127.0.0.1`; explicit resolved server host takes precedence. |
| `execution.health_path` | Readiness path, default `/health`. |
| `execution.shutdown_grace` | Positive seconds for owned process shutdown, default `15`. |
| `execution.drain_grace` | Positive seconds to wait for empty running/waiting request queues after benchmarks, default `15`. Missing metrics or failure to drain fails the trial. |
| `execution.retry.max_attempts` | Positive attempt count, default `1`; retries selected transient failures, not arbitrary bad configurations. Adds work. |
| `execution.max_parallel_trials` | Required worker count in local-parallel mode; must equal the number of explicit workers. |
| `execution.gpu_allocation.strategy` | Only `explicit` is supported and is the default. |
| `execution.gpu_allocation.allow_sharing` | Must be `false` (default); workers cannot overlap GPUs. |
| `execution.gpu_allocation.workers` | Required nonempty list in local-parallel mode. |
| `execution.gpu_allocation.workers[].name` | Unique worker label with the experiment-name character restrictions. |
| `execution.gpu_allocation.workers[].devices` | Nonempty unique nonnegative integer GPU IDs. No overlap between workers; each candidate's tensor parallelism must fit a worker. |
| `execution.ports.min` / `.max` | Inclusive valid port range for local-parallel workers, with enough distinct ports. |

In local-parallel mode remove `server.port`, tuned ports, and
`env.CUDA_VISIBLE_DEVICES`/`tune_env.CUDA_VISIBLE_DEVICES`; the optimizer owns these
assignments. Baseline runs alone first. Hardware differences and concurrent work
can confound candidate comparisons. Multi-GPU behavior needs hardware validation.

## Validate the saved file

```bash
vllm-opt validate --config experiment.yaml
```

Fix the reported error, then validate again. Success checks model directory
existence, optimizer policy, and command rendering; upstream compatibility and
GPU capacity still require execution. Do not promise a global optimum, latency
SLO, or production improvement merely because YAML validates.

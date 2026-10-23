# Automatic tuning

Choose a local model and enable auto mode:

```yaml
server:
  model: /models/my-model
optimization:
  mode: auto
```

Run `vllm-opt validate -c experiment.yaml`, then `vllm-opt -c experiment.yaml`.
Validation reads configuration and model metadata without starting CUDA,
servers, or calibration. Models are never downloaded automatically.
Defaults are 20 TPE trials maximizing output tokens/s, experiment name
`auto-tuning`, and seed `42`. Override `experiment` to name runs or set an output path.

## Choose an objective

```yaml
optimization:
  mode: auto
  objective:
    metric: end_to_end_latency_ms
    percentile: 99
  trials: 20
```

Latency metrics are `time_to_first_token_ms`, `time_per_output_token_ms`,
`inter_token_latency_ms`, and `end_to_end_latency_ms`. They are minimized;
omit `percentile` to minimize the mean. Throughput metrics are
`output_tokens_per_second`, `total_tokens_per_second`, and `requests_per_second`;
they are maximized and do not accept a percentile. Select one objective per run.

vLLM Bench Serve supports integer percentiles 1–100, including P55.
GuideLLM 0.7.x supports its emitted integer percentiles P1, P5, P10, P25, P50,
P75, P90, P95, and P99. Unsupported selections fail validation with instructions
to use `benchmark.engine: vllm`. Missing measurements never fall back to another
percentile or the mean.

The objective syntax also works with manual `tune` spaces. Existing
`optimization.maximize` retains its historical behavior, including maximizing
latency if selected. Do not combine `maximize` with `objective`.

## Generated workloads and calibration

Omitting `benchmark.runs` generates reproducible small, medium, and large
synthetic workloads with target prompt/output lengths of 128/32, 512/64,
and 2048/128 tokens. Lengths are reduced to fit the model context limit.
The default backend is vLLM Bench Serve; each workload requests 256 completions
with the existing default of four measured repeats. Tail estimates from small
request counts are uncertain; these defaults are a starting point for tuning.

One baseline server calibrates each workload separately before optimization.
Concurrency starts at 1 and doubles to a cap. A warmup precedes the first
measurement. Calibration backs off when throughput gains fall below 5%, mean
end-to-end latency grows by more than 2x between probes, or a probe fails.
It records the last stable load and every stopping reason. Optimization does
not start without a usable concurrency-1 measurement.

```yaml
optimization:
  mode: auto
  calibration:
    max_concurrency: 64
    probe_duration: 30  # Seconds per probe, including benchmark startup/warmup.
    max_duration: 180  # Seconds per workload after server startup.
    latency_growth: 2
    throughput_gain: 0.05
```

Server startup uses `timeouts.startup`; probe cleanup can take additional time.
A reached cap is recorded as a limit, not evidence that saturation was found.
Selected concurrency, data, and requested lengths remain fixed for baseline
measurements, all candidates, and finalist validation. This compares candidates
at calibrated concurrency, rather than discovering each candidate's capacity
or holding an arrival rate constant.

## Custom benchmarks and search settings

Supply `benchmark.runs` using the existing [benchmark configuration](benchmarks/benchmarking.md)
to retain your data and concurrency/rate settings. Auto mode skips calibration.
It can request additional output statistics needed by the objective.

The automatic catalog searches `max-num-seqs` in `[8, 16, 32, 64, 128]`,
`max-num-batched-tokens` in `[512, 2048, 4096, 8192]`, and
`gpu-memory-utilization` in `[0.8, 0.85, 0.9]`. Token batch candidates increase
to fit the context when chunked prefill is explicitly disabled. Actual baseline
probes measure model/GPU capacity; the catalog is not a memory-fit guarantee.

Explicit fixed settings, including native `server.config` settings, are excluded
from automatic tuning. Explicit `tune` definitions remain user overrides.
Model, precision, quantization, and GPU assignments are not changed automatically.
Failed or infeasible candidates cannot win. Results identify the best measured
eligible configuration within the bounded search, not a guaranteed global optimum.

## Results and reuse

Auto runs add `auto.json` (calibration summary), `resolved.yaml` (frozen experiment),
and, when an eligible configuration exists, `best-server.yaml` and
`best-environment.yaml`. Use the server YAML with `vllm serve --config` and
apply environment settings separately. Secret values are redacted and need
restoring privately before reusing exports.

The HTML report includes calibration decisions and the objective. Percentile
scores across workloads and repeats are means of backend-supplied percentiles,
not pooled request percentiles. Latency scores remain actual positive values;
lower wins. Offline reports and retries retain the objective and benchmark;
retries do not recalibrate. Raw calibration artifacts live separately from
ranked trials under the experiment's `calibration/` directory.

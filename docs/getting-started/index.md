# Quick start

You need Linux or WSL, an NVIDIA GPU, and a local model directory. Native
Windows can inspect configurations and reports but cannot run vLLM.

Read this walkthrough in 5–10 minutes. Package installation, model preparation,
and benchmark execution can take longer. This is a small execution example;
use realistic workloads and repeat evidence before production decisions.

## 1. Install

Create and activate an isolated Python 3.11 or 3.12 environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "vllm-optimizer[runtime]"
vllm --help
guidellm --help
```

This installs vLLM Optimizer, vLLM, and GuideLLM together. Use plain `pip install vllm-optimizer`
only on machines that will inspect configurations, artifacts, and reports
without launching experiments. See [installation choices](installation.md).

## 2. Create `experiment.yaml`

Replace `/models/opt-125m` with your existing local model directory. The CLI
validates that the directory exists but does not download or verify model
weights; startup checks whether vLLM can load them.

```yaml
experiment:
  name: first-run
server:
  model: /models/opt-125m
  gpu-memory-utilization: 0.8
tune:
  max-num-seqs:
    values: [8, 16]
env:
  CUDA_VISIBLE_DEVICES: "0"
benchmark:
  runs:
    - name: throughput
      profile:
        kind: throughput
        max_concurrency: 16
      constraints:
        - kind: max_requests
          count: 10
      data:
        - kind: synthetic_text
          prompt_tokens: 32
          output_tokens: 16
optimization:
  maximize: output_tokens_per_second
  sampler: tpe
  trials: 2
timeouts:
  benchmark: 20m
```

## 3. Run

```bash
vllm-opt validate --config experiment.yaml
vllm-opt --config experiment.yaml
```

`validate` checks configuration and command construction without starting
servers or loading weights. It does not verify GPU capacity or upstream flag
support.

The fixed server configuration runs first as the baseline, followed by two
distinct tuned configurations. Each uses four measured repeats by default.
The ten-request workload keeps this example small; it does not establish a
production performance improvement.

## 4. Open the report

Open `runs/first-run/<RUN_ID>/report.html`; the CLI prints the exact path.
The same directory contains `result.json`, `results.csv`, and per-trial logs.

Start with the best observed settings and the confidence verdict. A higher
observed score is not automatically a supported winner conclusion; check
latency trade-offs and repeat evidence before adopting the settings.

Add `--verbose` to the run command to stream logs. If startup fails, use
[troubleshooting](../guides/troubleshooting.md) and the saved `vllm.log`.

Next: [change the configuration](../guides/configuration.md),
[set performance targets](../guides/search-and-scoring.md#performance-targets),
or [interpret the report](../guides/reports.md).

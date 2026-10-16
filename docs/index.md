# vLLM Optimizer

Tune vLLM serving settings against your own workloads. Describe an experiment
in YAML; `vllm-opt` runs a baseline, searches your settings, and writes a report.

**[Run your first experiment](getting-started/index.md)** ·
**[See a real report](results/rtx3080/index.md)**

The quick start takes 5–10 minutes to read and includes a complete configuration.
Installation and execution time depend on your model and hardware.

## Start here

| You want to… | Read |
| --- | --- |
| Install and run one experiment | [Quick start](getting-started/index.md) |
| Use an existing runtime or inspect results without a GPU | [Installation](getting-started/installation.md) |
| Choose settings and search ranges | [Configuration](guides/configuration.md) |
| Maximize throughput under a latency target | [Search and scoring](guides/search-and-scoring.md#performance-targets) |
| Choose a benchmark or dataset | [Benchmarking](guides/benchmarks/benchmarking.md) |
| Interpret the recommendation | [Reports](guides/reports.md) |
| Fix a failed run | [Troubleshooting](guides/troubleshooting.md) |

## What the tool does

- Explores settings with Grid, Random, or TPE and optional performance limits.
- Runs GuideLLM or native vLLM Bench Serve workloads with repeat evidence.
- Keeps trial artifacts, logs, reproduction commands, and JSON/CSV/HTML results.
- Shows the best observed settings and how much confidence the evidence supports.

For less common options, use the [CLI](reference/cli.md),
[complete YAML](reference/yaml/full-example.md), or
[parallel trials](guides/parallel-trials.md) guides.
Contributors can start with [architecture](project/ARCHITECTURE.md) and
[contributing](project/contributing.md).

!!! note "Alpha software"
    Execution needs Python 3.11–3.12, Linux/WSL, an NVIDIA GPU, and an existing
    local model. Native Windows supports configuration and result inspection.
    Check [dated compatibility evidence](reference/compatibility.md) for your
    setup; additional GPU/topology validation remains pending.

This independent community project is not affiliated with vLLM.

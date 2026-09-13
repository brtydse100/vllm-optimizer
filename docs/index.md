# vLLM Optimizer

vLLM Optimizer is an independent community project and is not affiliated with
the vLLM project.

Tune vLLM with evidence from controlled experiments in one YAML file. You choose
the settings and workload; the `vllm-opt` CLI handles process lifecycle, unique search trials,
failures, persistence, and a decision-focused report.

```bash
pip install "vllm-optimizer[runtime]"
vllm-opt --config experiment.yaml
```

Use plain `pip install vllm-optimizer` for configuration and report inspection without
local vLLM execution. [Understand the two installation choices](installation.md).

## What vLLM Optimizer gives you

- A baseline and Grid, Random, or TPE exploration of your chosen settings.
- One or more GuideLLM or vLLM Bench Serve workloads per server configuration.
- Immutable trial artifacts, concise terminal progress, and detailed logs.
- A self-contained HTML report showing the best configuration and observed
  parameter effects.
- Exact, secret-redacted commands for reproducing completed trials.

## Find what you need

### Start an experiment

- [Run your first experiment](getting-started.md)
- [Choose an installation](installation.md)
- [Configure a run](configuration.md)
- [See a complete YAML example](full-example.md)

### Tune and benchmark

- [Select a benchmark backend](benchmarking.md)
- [Choose benchmark data](benchmark-data.md)
- [Understand search and scoring](search-and-scoring.md)
- [Run trials in parallel](parallel-trials.md)

### Understand results and solve problems

- [Interpret and regenerate reports](reports.md)
- [Compare GuideLLM and vLLM Bench Serve](backend-comparison.md)
- [Troubleshoot common failures](troubleshooting.md)
- [Check platform compatibility](compatibility.md)

### Contribute and explore the design

- [Read the architecture overview](ARCHITECTURE.md)
- [Review the roadmap](ROADMAP.md)
- [Contribute to the project](contributing.md)

!!! note "Alpha software"
    Experiment execution targets Linux with NVIDIA GPUs. The universal Python
    package can be installed elsewhere for configuration and result inspection.

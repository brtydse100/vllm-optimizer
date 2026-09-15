# Real GPU experiment results

These reports are produced by actual inference runs. The
[experiment recipes](../experiments/README.md) include historical RTX 3080
reproduction and a separate H100 suite with user-selected models and GPUs.

- [RTX 3080 local results](rtx3080/index.md)
- [H100 results archive](h100/index.md) — reserved for real H100 executions;
  currently no H100 measurements are published.

The reports cover overview, recommendation, workload-specific baseline
comparison, confidence, the complete configuration leaderboard, and expandable
scoring evidence. Benchmark execution duration is shown for the baseline and
every trial, including mean and percentage differences. Missing backend fields
remain unavailable.

Each result includes its exact experiment, score/measurement exports, and an
evidence bundle. Compare baseline and candidates within the same model and
workloads. The RTX smoke workloads demonstrate reporting; the H100 large-model
scenario measures gains separately and requires fresh candidate validation.
Different models do not establish cross-GPU speedup claims.

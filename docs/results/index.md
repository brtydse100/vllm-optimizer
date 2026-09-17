# Real GPU experiment results

These reports are produced by actual inference runs. The
[experiment recipes](../experiments/README.md) include historical RTX 3080
reproduction and a separate H100 suite with user-selected models and GPUs.

- [RTX 3080 local results](rtx3080/index.md)
- [H100 results archive](h100/index.md) — reserved for real H100 executions;
  currently no H100 measurements are published.

The reports open with a concise throughput/duration summary, visible confidence,
and best-observed settings. All trials remain visible with their average
benchmark execution duration; expandable details preserve all recorded trial
data. Workload comparisons, validation evidence, charts, and methodology are
collapsed initially. Missing or incomparable measurements remain unavailable.

Each result includes its exact experiment, score/measurement exports, and an
evidence bundle. Compare baseline and candidates within the same model and
workloads. The RTX smoke workloads demonstrate reporting; the H100 large-model
scenario measures gains separately and requires fresh candidate validation.
Different models do not establish cross-GPU speedup claims.

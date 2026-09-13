# Real GPU experiment results

These reports are produced by actual local inference runs. The
[shared experiment files](../experiments/README.md) are intended to run
unchanged on the local RTX 3080 and on one rented H100.

- [RTX 3080 local results](rtx3080/index.md)
- [H100 results archive](h100/index.md) — reserved for real H100 executions;
  currently no H100 measurements are published.

The reports cover the five implemented decision sections: overview,
recommendation, workload-specific baseline comparison, confidence, and the
complete configuration leaderboard. Existing charts and diagnostics are
expandable. Missing timing or backend fields remain unavailable.

Each result includes its exact experiment, score/measurement exports, and an
evidence bundle. H100 runs must match the experiment and model checksums before
being labeled comparable. These small smoke workloads demonstrate reporting;
they do not establish production capacity or cross-GPU speedup claims.

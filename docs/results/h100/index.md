# H100 results archive

**No H100 experiments have been executed or published yet.** This folder is
reserved for real results from the rented H100 GPU machines, not simulated or
RTX 3080 measurements.

Run both [shared experiments](../../experiments/README.md) unchanged on one
H100. Record additional physical GPUs if present, but keep only device 0
visible for these single-GPU comparisons.

Store each actual run under `<date>/<experiment>/<run-id>/`, with:

- `report.html`, `result.json`, `results.csv`, and `experiment.yaml`;
- model/YAML checksums and software/hardware provenance;
- normalized results, manifests, raw benchmark output, and diagnostic logs
  in an evidence archive, with public redactions disclosed.

Add links here after execution, including unsuccessful runs. Do not copy the
local report into this folder or label unrun configurations H100-validated.

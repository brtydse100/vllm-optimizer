# H100 results archive

**No H100 experiments have been executed or published yet.** This folder is
reserved for real results from the rented H100 GPU machines, not simulated or
RTX 3080 measurements.

Use the [H100 scenario suite](../../experiments/h100/README.md) with your chosen
models and GPU IDs: parallel independent trials, one server spanning 2+ GPUs,
and large-model tuning followed by separate validation. H100 models do not
need to match the historical RTX 3080 model.

Store each actual run under `<date>/<experiment>/<run-id>/`, with:

- `report.html`, `result.json`, `results.csv`, and `experiment.yaml`;
- model/YAML checksums and software/hardware provenance;
- both search and validation reports for large-model tuning, with the baseline
  rationale and per-workload improvements or regressions;
- normalized results, manifests, raw benchmark output, and diagnostic logs
  in an evidence archive, with public redactions disclosed.

Add links here after execution, including unsuccessful runs. Do not copy the
local report into this folder or label unrun configurations H100-validated.

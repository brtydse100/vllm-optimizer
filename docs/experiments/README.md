# GPU experiment recipes

Choose a scenario, supply your model directory and GPU IDs, and generate a
resolved YAML before running it. H100 experiments do not need to use the model
or settings from the RTX 3080 demonstration.

- [H100 scenario suite](h100/README.md): parallel trials, 2+ GPU tensor
  parallelism, and large-model tuning with separate validation.
- [Reproduce the historical RTX 3080 examples](local-reproduction.md): pinned
  model, runtime, and unchanged YAMLs for the published reports.
- [Real results](../results/index.md): completed RTX examples and the archive
  reserved for future H100 measurements.

Keep the generated experiment YAML with each result. Baseline and candidates
within an experiment must use the same model and workloads. Comparisons across
different models describe separate experiments, not GPU speedups.

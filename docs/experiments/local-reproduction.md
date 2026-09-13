# Reproduce the recorded RTX 3080 examples

These two small, single-GPU experiments produced the published RTX 3080 reports.
Keep their YAML bytes unchanged when reproducing that historical setup. For new
H100 experiments with your own models, use the [H100 suite](h100/README.md).

| File | Purpose |
| --- | --- |
| [report-showcase.yaml](report-showcase.yaml) | Baseline, six valid grid settings, two deliberately invalid settings, two workloads, four repeats, parameter associations, and finalist validation when drift is observed. |
| [baseline-control.yaml](baseline-control.yaml) | Compare a concurrency-capable baseline with a deliberately restricted candidate; the measured winner is never forced. |

The showcase's `max-num-seqs: 0` cases intentionally fail server validation.
Its zero drift threshold deliberately exercises finalist validation whenever
the repeat halves differ. It is a demonstration policy, not a production
recommendation. Timing noise and the winning configuration can differ by GPU.

## Prepare the historical environment

Use Linux, one visible NVIDIA GPU, Python **3.12.3**, and a driver compatible
with the captured PyTorch **CUDA 13.0** runtime. Local evidence uses Ubuntu WSL
and RTX 3080. Other hardware produces new measurements, not the recorded results.
Do not run other inference jobs alongside these measurements.

From the repository root, create a fresh virtual environment and install the
captured package versions and this checkout:

```bash
python3.12 -m venv /var/tmp/vllm-optimizer-demo-venv
source /var/tmp/vllm-optimizer-demo-venv/bin/activate
python -m pip install -r docs/experiments/runtime-requirements.txt
python -m pip install --no-deps -e .
python docs/experiments/prepare_model.py
python -m pip check
```

The model is `facebook/opt-125m`, revision
`27dcfa74d334bc871f3234de431e71c6eeba5dd6`. The preparation helper verifies
[model and tokenizer checksums](model.sha256), including the weight file.
The YAML records fixed absolute model/output paths.

The two `VLLM_USE_*` flags capture local WSL compatibility settings. They belong
to this historical recipe; the separate H100 templates use native runtime defaults.

## Run unchanged

```bash
sha256sum docs/experiments/*.yaml
vllm-opt validate -c docs/experiments/report-showcase.yaml
vllm-opt validate -c docs/experiments/baseline-control.yaml
vllm-opt -c docs/experiments/report-showcase.yaml
vllm-opt -c docs/experiments/baseline-control.yaml
```

The showcase intentionally reports failures. Inspect the persisted run status
and its two invalid configurations before treating a nonzero exit as unexpected.
Each run prints its timestamped directory under `/var/tmp/vllm-optimizer-demo/runs`.
Copy the results off the rented machine **before shutting it down**.

For exact reproduction, preserve the model, workloads, seeds, and serving settings.
New H100 scenarios can use different models and GPU allocations.

## Publish the evidence

See [recorded GPU results](../results/index.md). Retain the original YAML,
`result.json`, `results.csv`, self-contained `report.html`, normalized trial
results, manifests, raw benchmark output, and logs. Record the checkout commit,
YAML/model checksums, package versions, GPU model/count, driver, OS, and date.
Remove credentials, GPU UUIDs, and personal paths from the public copy; retain
the untouched original privately and disclose any redaction.

H100 results belong under `docs/results/h100/<date>/<experiment>/<run-id>/`.
Add a row to its index only after actual execution; include failures rather
than publishing only winning trials. Compare corresponding workloads and
accepted executions rather than treating these short runs as general hardware
performance rankings.

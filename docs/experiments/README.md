# Identical local and H100 experiments

These two small, single-GPU experiments demonstrate report behavior using real
inference. Run the **same YAML bytes** on both machines. They are not an H100
capacity or multi-GPU scaling study.

| File | Purpose |
| --- | --- |
| [report-showcase.yaml](report-showcase.yaml) | Baseline, six valid grid settings, two deliberately invalid settings, two workloads, four repeats, parameter associations, and finalist validation when drift is observed. |
| [baseline-control.yaml](baseline-control.yaml) | Compare a concurrency-capable baseline with a deliberately restricted candidate; the measured winner is never forced. |

The showcase's `max-num-seqs: 0` cases intentionally fail server validation.
Its zero drift threshold deliberately exercises finalist validation whenever
the repeat halves differ. It is a demonstration policy, not a production
recommendation. Timing noise and the winning configuration can differ by GPU.

## Prepare either machine

Use Linux, one visible NVIDIA GPU, Python **3.12.3**, and a driver compatible
with the captured PyTorch **CUDA 13.0** runtime. Local evidence uses Ubuntu WSL
and RTX 3080; H100 evidence will use the rented host's recorded OS and driver.
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
The YAML uses the same absolute model/output paths on both hosts.

Keep the specified environment flags on H100 too. The two `VLLM_USE_*` flags
provide local WSL compatibility; retaining them preserves the serving setup.
`CUDA_VISIBLE_DEVICES=0` and tensor parallel size 1 use one GPU even if the
rented machine has several. A separate multi-GPU study must use different
experiment names and must not be labeled identical to this local run.

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

Do not change model, seeds, precision, batch limits, workload lengths,
concurrency, request count, repeats, warmups, failure policy, or sampler for the
H100 rerun. Runtime, GPU memory capacity, OS, driver, and measurements are
recorded separately; identical settings do not imply identical measurements.

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

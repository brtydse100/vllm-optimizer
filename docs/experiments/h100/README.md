# H100 experiments with your models

These templates are configuration-tested, **not H100 performance-tested**.
Choose any locally downloaded model supported by your installed vLLM runtime.
Model fit and multi-GPU startup must be checked on the rented hardware.

| Scenario | GPUs | Model | What it exercises |
| --- | --- | --- | --- |
| [parallel-trials.yaml](parallel-trials.yaml) | 2+ | Small, fitting on each GPU | Independent configurations run concurrently, one server per GPU; each benchmark also sends concurrent requests. |
| [multi-gpu.yaml](multi-gpu.yaml) | 2+ | Small, supporting the requested TP size | One vLLM server spans all selected GPUs through `tensor-parallel-size`; trials run sequentially. |
| [large-model.yaml](large-model.yaml) | 1+ | Your large deployment model | Searches batching settings across three workloads, then validates the selected candidate against the baseline again. |

Parallel trials are independent experiments scheduled at the same time. Named
workloads within each trial run sequentially; request concurrency creates load
inside each workload. The multi-GPU scenario tests startup and serving; a small
model is not expected to demonstrate a multi-GPU speedup.

## Prepare and run

Use the project's [Linux runtime installation](../../getting-started/installation.md).
Install this checkout, download your chosen model, and supply its local directory.
The preparation helper does not download models or start inference. It validates
the resolved configuration and refuses to overwrite an existing output file.

From the repository root:

```bash
python docs/experiments/prepare_h100.py parallel-trials \
  --model-path /models/your-small-model --gpus 0,1 --output prepared/parallel.yaml
python docs/experiments/prepare_h100.py multi-gpu \
  --model-path /models/your-small-model --gpus 0,1 --output prepared/tp2.yaml
python docs/experiments/prepare_h100.py large-model \
  --model-path /models/your-large-model --gpus 0,1 --output prepared/large.yaml
```

For TP=4, supply `--gpus 0,1,2,3`. Noncontiguous IDs such as `2,4` also work.
Parallel trials use one worker per listed GPU, while multi-GPU and large-model
experiments set TP to the number of listed GPUs. The large-model scenario also
accepts a single GPU if the selected model fits.

Review the generated YAML: context length, precision, memory budget, workloads,
and baseline must suit your model. `server.model` can be changed to another
local model directory. Revalidate edits, then run the scenarios one at a time:

```bash
vllm-opt validate -c prepared/parallel.yaml
vllm-opt -c prepared/parallel.yaml
vllm-opt validate -c prepared/tp2.yaml
vllm-opt -c prepared/tp2.yaml
vllm-opt validate -c prepared/large.yaml
vllm-opt -c prepared/large.yaml
```

Results default to `/var/tmp/vllm-optimizer-h100/runs`; override this with
`--runs-dir`. Physical GPU IDs refer to the host's devices. The scheduler owns
GPU visibility and ports for parallel trials. Use an otherwise idle GPU allocation.

## Measure useful improvements

The large-model baseline uses 128 sequences, an 8,192-token batch budget,
90% GPU memory utilization, chunked prefill, and native graph behavior.
Use your real deployment configuration as the baseline if available. Do not
artificially restrict it to manufacture a gain.

The grid tests 16 combinations: sequences `[32, 64, 128, 256]` and batched
tokens `[2048, 4096, 8192, 16384]`. It includes the baseline settings as a
search candidate. All configurations retain the same model, precision, TP,
memory budget, context length, and workloads.

| Workload | Input / output tokens | Requests per repeat | Concurrency |
| --- | --- | --- | --- |
| Interactive | 512 / 256 | 512 | 16 |
| Batch generation | 1024 / 512 | 512 | 128 |
| Long prompt | 3072 / 256 | 256 | 32 |

All use seeded random inputs, unlimited request rate, one excluded warmup,
four measured repeats, and zero allowed request failures. This is 17 initial
configurations including the baseline, before any drift-triggered reruns.
Runtime depends on your model; this is substantially longer than the smoke cases.
Choose a representative dataset and request rate when validating your deployment.

The objective maximizes the mean of per-workload mean output throughput.
Inspect each workload's throughput and P50/P95/P99 latency, request failures,
and repeat variability. An aggregate gain can hide a workload regression.

## Validate the selected candidate separately

After the large-model search finishes, use its printed run directory:

```bash
python docs/experiments/prepare_h100_validation.py \
  --run /var/tmp/vllm-optimizer-h100/runs/h100-large-model/RUN_ID \
  --output prepared/validation.yaml
vllm-opt -c prepared/validation.yaml
```

The helper checks stored scores and artifact integrity through offline report
regeneration. It reads accepted manifests, including finalist validation, and
creates exactly one candidate against the original baseline. Both receive two
warmups and eight measured repeats, with each workload's seed incremented by
one to provide fresh synthetic requests. The source run is retained; the
integrity check adds an offline regenerated report below it.

Baseline and candidate execute sequentially on the same allocation. This is a
separate validation run, not randomized interleaving or proof of significance.
If the baseline still wins, retain it. Treat overlapping ranges or drift as
inconclusive, and report workload regressions. The result identifies the best
observed tested setting; it does not establish a global optimum.

Publish both search and validation reports, including failures and no-gain
results, under the [H100 results archive](../../results/h100/index.md). Record
the model revision/checksums, GPU count and memory, driver, package versions,
checkout, exact generated YAML, workload definitions, and baseline rationale.
Back up the evidence before ending the GPU rental. Do not label these results
comparable to the RTX report merely because both were generated by this tool.

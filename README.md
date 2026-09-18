# vLLM Optimizer

> **Alpha status:** native Linux, L40/H100,
> tensor-parallel, and multi-GPU evidence remain **hardware validation pending**.

vLLM Optimizer is a local-first benchmarking and optimization tool for vLLM
serving configurations. Users define the parameters and workloads they care
about; the `vllm-opt` CLI manages the server lifecycle, runs repeatable benchmarks,
explores the search space, and reports which configurations performed best.

vLLM Optimizer is alpha software targeting Linux with NVIDIA GPUs and
Python 3.11–3.12. Install `vllm-optimizer`, import `vllm_optimizer`, and run
`vllm-opt`. The former `vtune` aliases are deprecated; see the [migration guide](docs/guides/migration-vllm-optimizer.md).

This independent community project is not affiliated with the vLLM project.

**[Documentation](https://brtydse100.github.io/vllm-optimizer/)** ·
**[Quick start](https://brtydse100.github.io/vllm-optimizer/getting-started/)** ·
**[Full YAML example](docs/reference/yaml/full-example.yaml)** ·
**[PyPI](https://pypi.org/project/vllm-optimizer/)**

## See a real optimization report

[**Open the interactive RTX 3080 report**](https://brtydse100.github.io/vllm-optimizer/results/rtx3080/report-showcase/report.html)
· [Baseline-control example](https://brtydse100.github.io/vllm-optimizer/results/rtx3080/baseline-control/report.html)
· [Exact experiment YAML](docs/results/rtx3080/report-showcase/experiment.yaml)
· [All published results](https://brtydse100.github.io/vllm-optimizer/results/rtx3080/)

[![Concise report with throughput and duration summary, confidence, and all trials](docs/results/rtx3080/overview.png)](https://brtydse100.github.io/vllm-optimizer/results/rtx3080/report-showcase/report.html)

This measured example increased the configured score from **194.34 to 1,082.27
output tokens/s (+456.89%)** on its deliberately small RTX 3080 workloads. It
also reports latency trade-offs, repeat evidence, validation drift, and failed
configurations; it is a reporting demonstration, not a production benchmark.

Published runtime evidence records vLLM 0.28.0 and GuideLLM 0.7.3 on WSL2
with an RTX 3080; see the [dated compatibility matrix](docs/reference/compatibility.md). That host required
  `VLLM_USE_V2_MODEL_RUNNER: "0"` because UVA was unavailable and
  `VLLM_USE_FLASHINFER_SAMPLER: "0"` because the CUDA compiler toolkit was
  not installed. Native Linux systems may not require these settings.

Other combinations may work but are not yet verified.

## Installation

Choose the installation that matches what you want to do:

| Goal | Command | Platform |
| --- | --- | --- |
| Run complete experiments | `pip install "vllm-optimizer[runtime]"` | Linux/WSL with NVIDIA GPU |
| Read configs, results, and reports | `pip install vllm-optimizer` | Linux, Windows, or macOS |

The core package intentionally does not install GPU frameworks. The `runtime`
extra adds vLLM and GuideLLM, which select large PyTorch/CUDA dependencies for
the machine. See the [installation guide](https://brtydse100.github.io/vllm-optimizer/getting-started/installation/)
for virtual environments, CUDA guidance, and verification commands.

## Quick start

Create and activate a Python 3.11 or 3.12 virtual environment on Linux or WSL,
then install the complete experiment runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "vllm-optimizer[runtime]"
vllm --help
guidellm --help
```

Replace `/models/opt-125m` with an existing local model directory, then create
`experiment.yaml`. The CLI does not download models:

```yaml
experiment:
  name: first-run
server:
  model: /models/opt-125m
  gpu-memory-utilization: 0.8
tune:
  max-num-seqs:
    values: [8, 16]
benchmark:
  engine: guidellm  # Default. Use vllm for `vllm bench serve`.
  max_failure_percentage: 2  # Accept up to 2% errored or incomplete requests.
  # accept_any_request_failures: true  # Ignore the percentage (one success still required).
  runs:
    - name: throughput
      profile:
        kind: throughput
        max_concurrency: 16
      constraints:
        - kind: max_requests
          count: 10
      data:
        - kind: synthetic_text
          prompt_tokens: 32
          output_tokens: 16
optimization:
  maximize: output_tokens_per_second
  sampler: tpe
  trials: 2
timeouts:
  benchmark: 20m
```

Run it:

```bash
vllm-opt validate --config experiment.yaml
vllm-opt --config experiment.yaml
```

The short form is `vllm-opt -c experiment.yaml`. The command validates the file,
runs the experiment, persists results, and generates its exports and report.
The `vllm-opt` CLI binds vLLM to `127.0.0.1` by default. Set `server.host` explicitly
only when the benchmark server must be reachable from another host.

Already have a native `vllm serve` YAML file? Use `server.config` to reuse it.
See [native server configuration and LMCache](docs/reference/yaml/full-example-server.md#native-server-configuration-and-lmcache)
for precedence, paths, and copyable examples.

Fixed vLLM flags go directly under `server`; tunable flags use top-level
`tune`. Fixed and tunable environment variables use `env` and `tune_env`.
See the [configuration guide](https://brtydse100.github.io/vllm-optimizer/guides/configuration/)
for categorical, boolean, integer-range, float-range, list, and environment
examples. The [complete YAML](https://brtydse100.github.io/vllm-optimizer/reference/yaml/full-example/)
and [benchmark guide](https://brtydse100.github.io/vllm-optimizer/guides/benchmarks/benchmarking/) show
every supported control with copyable examples.

To use vLLM's native benchmark, set `benchmark.engine: vllm` and configure
`args` for each run. See the [vLLM Bench Serve example](docs/reference/yaml/full-example-vllm-bench.md).
The CLI supplies the model, server address, and JSON output path.

Interactive terminal output uses color and remains concise by default. Set the
standard `NO_COLOR` environment variable to disable color. To stream server and
benchmark logs:

```bash
vllm-opt --config experiment.yaml --verbose
```

Set `logging.level: DEBUG` in YAML for persistent verbose output. Full logs are
saved for every trial regardless of terminal verbosity. See
[logging and timeouts](docs/guides/configuration.md#logging-and-timeouts).

Retry one or more selected trials into a new immutable linked run:

```bash
vllm-opt retry --run runs/EXPERIMENT/RUN_ID \
  --trial trial-0001 --trial trial-0004
```

The source run is never modified.

Display every stored vLLM and benchmark command for a trial without executing
anything:

```bash
vllm-opt reproduce --run runs/EXPERIMENT/RUN_ID --trial trial-0001
```

Each completed run also contains a self-contained `report.html` decision
dashboard with a one-line throughput and average benchmark duration summary,
a visible confidence verdict, and the best observed settings. Every trial and
its average benchmark duration remain visible; expand details for all recorded
trial data. Comparisons, latency percentiles, repeat statistics, charts, and
methodology are available on demand. See the [reports guide](docs/guides/reports.md).

Random and TPE runs never execute the same resolved configuration twice. If
`optimization.trials` exceeds the unique search space, the `vllm-opt` CLI warns and runs
every unique configuration once.

Multiple independent trials can run on explicitly assigned, non-overlapping
GPU sets and ports. A sequential or tensor-parallel server receives port 8000
unless `server.port` overrides it; local-parallel trials use their configured
port range. Sequential execution remains the default. See
[parallel trials](https://brtydse100.github.io/vllm-optimizer/guides/parallel-trials/) for the
YAML and measurement caveats.

## Product documents

- [First MVP specification](docs/project/mvp/MVP_SPEC.md)
- [Future implementation roadmap](docs/project/roadmap/ROADMAP.md)
- [Architecture overview and early sketch](docs/project/ARCHITECTURE.md)
- [Editable Draw.io architecture diagram](docs/vllm-optimizer-architecture.drawio)
- [Contributor guide](CONTRIBUTING.md)
- [Release notes](CHANGELOG.md)

The MVP specification defines the first releasable version and its acceptance
criteria. The roadmap describes capabilities that should be designed for now
but implemented after the core experiment loop is reliable.

vLLM Optimizer is available under the [MIT License](LICENSE).

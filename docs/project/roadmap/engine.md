# Experiment engine and benchmark coverage

## Phase 1: Harden the local experiment engine

This phase mixes completed foundations with future extensions. Code examples
under unfinished features are proposals and are not accepted by the current CLI.

### Offline report regeneration — completed in v0.1.0a2

- Rebuilds HTML from one completed run and its validated trial artifacts.
- Requires no GPU, vLLM, or GuideLLM process.
- Preserves the original report and refuses destination overwrites.
- Validates artifact schema and integrity before rendering.

### Conditional search spaces — next

Allow parameters to be active only when another choice makes them valid:

```yaml
tune:
  attention-backend:
    values: [FLASH_ATTN, FLASHINFER]

  flashinfer-option:
    values: [a, b]
    when:
      attention-backend: FLASHINFER
```

Requirements:

- Deterministic normalization and fingerprinting.
- Clear validation of missing or cyclic dependencies.
- Compatible behavior across grid, random, and TPE.
- Reports distinguish inactive parameters from missing values.

### Advanced retry policies

Extend the MVP retry model with:

- Configurable exponential backoff and jitter.
- Backend-specific retry classification.
- Per-phase attempt limits.
- Retry budgets shared across a run.
- Retry statistics and policy recommendations.

Retries must remain part of one trial rather than inflating the study with
duplicate trials. OOM, invalid arguments, and deterministic incompatibilities
should not retry by default.

### Baseline scheduling and drift detection

Implemented: within-trial sequential drift checks, drift-triggered reruns, and
optional fixed-budget finalist validation. The following extensions are future work:

- Periodically rerun the baseline during long experiments.
- Compare baseline measurements over time.
- Warn when performance drift exceeds a configured threshold.
- Optionally normalize candidates against the nearest baseline in time.
- Display time-correlated drift in the report.

### Statistical measurement controls

Implemented: repeat summaries, Student's t intervals, descriptive overlap
guards, and adaptive stopping for weak search candidates. Future extensions:

- Bootstrap comparisons.
- Outlier policies.
- Adaptive repeat counts when variance is high.
- Minimum meaningful improvement thresholds.
- Statistical tie handling instead of false precision.
- Host-noise metadata such as utilization and temperature sampling.

### Richer experiment comparison

- Compare two completed runs.
- Detect changes in vLLM, CUDA, drivers, model revisions, and benchmark versions.
- Show regressions and improvements by scenario.
- Export a machine-readable comparison artifact for CI.

### Experiment forking

Allow an existing immutable run to seed a changed experiment:

```bash
vllm-opt fork runs/qwen-h100 --config revised.yaml
```

The fork records ancestry, reuses compatible cached evaluations, and clearly
separates the new experiment identity.

## Phase 2: Expand benchmark coverage

### vLLM Bench Serve adapter

Implemented in the development version using the same adapter contract as
GuideLLM:

- Configuration validation.
- Command construction without shell interpolation.
- Expected-duration calculation.
- Raw output preservation.
- Metric normalization with explicit units.
- Backend version capture.
- Forward-compatible flat argument pass-through.

Future work: capability discovery, schema-drift fixtures across vLLM versions,
and comparability studies between equivalent GuideLLM and vLLM workloads.

Reports must show backend provenance. Metrics from different backends should
not be treated as directly interchangeable unless their definitions and
measurement procedures are verified compatible.

### User-defined command adapter

Allow an advanced user to supply an external benchmark command that emits a
documented JSON result schema.

Safety and correctness requirements:

- Structured executable and argument fields rather than raw shell text.
- Explicit timeout and exit-code behavior.
- Version/provenance metadata.
- JSON Schema validation.
- No implicit trust of emitted file paths.
- Clear statement that vLLM Optimizer cannot guarantee comparability across arbitrary
  benchmark implementations.

### Dataset adapters

Current backends already support local files, Hugging Face references, and
synthetic distributions through their dataset options. This list describes
broader adapter contracts and remaining extensions:

- Multiple datasets within one named benchmark run.
- Hugging Face dataset references with pinned revisions.
- Local JSONL and compatible benchmark formats.
- Synthetic prompt and output length distributions.
- Dataset sampling seeds.
- Dataset fingerprints stored in experiment metadata.
- Optional preprocessing cache.

### Correctness and quality gates

Performance alone is insufficient for settings that may affect output quality.
Request-failure thresholds already gate ranking. Additional evaluators could cover:

- Response validity.
- Exact-match or task-specific correctness.
- Output truncation and completion length.
- User-provided evaluation commands.

Quality metrics may act as constraints or later as multi-objective targets.

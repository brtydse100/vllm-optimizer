# Automatic Tuning Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. No subagents are proposed.

**Goal:** Tune a local model from minimal YAML, with a throughput or percentile-latency objective and an optional custom benchmark.

**Architecture:** Put automatic configuration preparation, default workloads, calibration, and exports in a new `auto` package. Pass the resolved configuration into the existing orchestrator. Put objective interpretation in one shared policy; existing consumers delegate to it instead of implementing separate direction and percentile rules.

**Tech stack:** Existing Python 3.11–3.12, dataclasses, Optuna, PyYAML, vLLM, and benchmark workers. No new dependencies.

**Spec:** The approved design is captured below. This document revises the implementation layout to minimize changes to existing files; implementation still requires approval.

## Approved behavior

- `optimization.mode: auto` selects automatic tuning; manual configurations remain supported.
- `optimization.objective.metric` selects throughput, TTFT, time per output token, inter-token latency, or end-to-end latency.
- `optimization.objective.percentile` selects a latency percentile, including P55 and P99. Missing measurements never fall back to an average or a different percentile.
- Throughput is maximized; latency is minimized. Existing `optimization.maximize` semantics remain unchanged.
- A supplied benchmark retains its data and load settings. Omitting benchmark runs generates reproducible small, medium, and large workloads with explicit output lengths.
- Calibrate the baseline once for each generated workload, then freeze load for the baseline, every candidate, and finalist validation.
- Bound concurrency, calibration duration, and search trials. Stop calibration on saturation, excessive latency growth, request failures, or limits.
- Respect fixed inline and native-server settings. Do not automatically change precision, quantization, or model identity.
- Report actual metric values and units, calibration decisions, and reusable winning settings. Label averages of workload percentiles accurately.
- Persist the resolved benchmark so retries use the same workload without recalibration.

## Global constraints

- Keep new implementation and tests outside existing modules wherever practical; existing edits must be short integration or correctness changes.
- No orchestrator rewrite, duplicate orchestration pipeline, monkey-patching, global objective state, or encoded negative latency presented as a measured value.
- Keep files normally below 150 lines and always below 200 lines; split additions by responsibility.
- Preserve unrelated working-tree changes. Do not reformat unrelated code or change dependencies, package versions, CI, or release files.
- Keep tests and GPU artifacts outside the repository, under `C:/Users/Ido/Desktop/vTune-private-tests`.
- Run focused private tests after each implementation stage and broader regressions for shared scoring/reporting changes.
- Stop if required tests or GPU validation cannot run; do not claim completion without successful real-GPU evidence.

## Review focus

- An unavailable requested percentile must produce an explicit failure, never silently select another statistic.
- Latency direction must survive finalist scoring, persistence, offline reports, reclassification, and retries.
- Calibration failures, cancellation, and timeouts must clean up only processes owned by this run.
- Small context limits and native configuration overrides must produce valid, identical workloads across candidates.
- Calibration must not be repeated per candidate or counted as optimization evidence.

## Task 1: Introduce the shared objective policy

**New files:** `src/vllm_optimizer/objectives.py` for the typed objective and comparison helpers; `src/vllm_optimizer/benchmarks/percentiles.py` for backend percentile extraction.

**Expected existing edits:** `config/runtime.py`, `benchmarks/metrics.py`, `managers/scoring.py`, `search/factory.py`, `search/optuna_session.py`.

**Interfaces:** `Objective(metric: str, percentile: int | None, direction: str)`; `objective_from_config(config: VTuneConfig) -> Objective`; `extract_objective(metrics: Mapping[str, object], objective: Objective) -> float | None`.

- [ ] Add private tests for legacy maximize behavior, throughput, latency minimization, P55/P99, missing/non-finite values, invalid percentiles, and conflicting objective syntax.
- [ ] Run the tests and confirm they fail for the intended missing behavior.
- [ ] Implement the policy and narrow integration calls. Preserve positive latency measurements; give Optuna the actual objective direction.
- [ ] Preserve all supplied percentile measurements in normalization. For requested percentiles, configure supported backend output or reject unsupported output explicitly; never invent a percentile from aggregate statistics.
- [ ] Run the focused tests and existing private metric/search/scoring tests successfully.

## Task 2: Carry objective semantics through existing results

**New file:** `src/vllm_optimizer/reporting/objective.py` for objective labels, serialization helpers, and direction-aware comparisons shared by existing report consumers.

**Expected existing edits:** `orchestrator_setup.py`, `execution/trial_executor.py`, `managers/run_documents.py`, `managers/run_results.py`, `measurement.py`, `lifecycle/retry.py`.

**Expected reporting edits:** `context.py`, `accepted_scores.py`, `offline.py`, `reclassify.py`, `reclassify_scores.py`, `dashboard_selection.py`, `finalist_decision.py`, `leaderboard.py`, `dashboard.py`, `confidence.py`, `llm_summary.py`, and `reporter.py`, under `src/vllm_optimizer/reporting/`.

These are small call-site changes, not replacements of these modules. Each edit must correct an existing maximize/average assumption or preserve the objective during reconstruction; remove any edit that is unnecessary after the shared policy is implemented.

**Interfaces:** Persist explicit objective metadata with backward-compatible defaults for old runs; use Task 1's policy for selection and extraction everywhere. Preserve that metadata when cloning scorers and sanitizing scores.

- [ ] Add private tests where lower P99 wins while mean latency disagrees; assert correct baseline comparison, finalist selection, positive measured values, and units.
- [ ] Add round-trip tests for stored scores, offline regeneration, reclassification, and retry. Legacy runs must keep their original semantics.
- [ ] Check adaptive-repeat decisions for latency objectives and make the smallest necessary direction-aware change in `workers/adaptive_repeats.py` if its tests expose the current throughput assumption.
- [ ] Run the tests to confirm the missing behavior, implement the integration changes, then rerun focused and broader reporting/lifecycle regressions.

## Task 3: Prepare auto configurations outside the orchestrator

**New package:** `src/vllm_optimizer/auto/`, with `__init__.py`, `configuration.py`, `hardware.py`, `space.py`, and `workloads.py`.

**Expected existing edits:** `config/loader.py`, `config/models.py`, and `cli.py`. Reuse Task 1's search validation changes rather than introducing another search factory.

**Interfaces:** `prepare_auto_config(raw: Mapping[str, object]) -> Mapping[str, object]` supplies structural defaults before normal loading; `resolve_auto_config(config: VTuneConfig) -> VTuneConfig` produces ordinary tune definitions and workload plans before orchestration.

- [ ] Add private tests for minimal YAML, fixed-setting precedence including aliases/native files, user-supplied benchmarks, bounded unique candidates, and context-limited synthetic workloads.
- [ ] Run the tests to establish failures, then implement preparation and a conservative, documented candidate catalog using existing grid/TPE definitions.
- [ ] Keep hardware access and server startup out of `validate`; defer runtime-only checks and calibration until execution.
- [ ] Default to 20 TPE trials, preserve explicit budgets, and retain explicit benchmark engine selection. No automatic model downloads.
- [ ] Run focused configuration/search tests and manual-mode regression tests.

## Task 4: Calibrate once, then call the existing orchestrator

**New files:** `auto/calibration.py` for bounded probe decisions; `auto/runner.py` for lifecycle integration; `auto/artifacts.py` for private evidence and resolved-config exports.

**Expected existing edits:** The same `cli.py` integration point from Task 3 and a small optional auto-summary hook in `reporting/reporter.py`. Avoid edits to `orchestrator.py` and its scheduling loop.

**Interfaces:** `run_auto(config: VTuneConfig) -> RunOutcome` is async; it calibrates only generated workloads, resolves ordinary configuration, and invokes `Orchestrator(resolved).run()` once. `select_load(probes: Sequence[ProbeResult]) -> CalibrationDecision` is pure.

- [ ] Add private tests for increasing probe loads, plateau/latency/error stopping, hard limits, no usable probe, cancellation, and cleanup.
- [ ] Add integration tests asserting byte-equivalent resolved workloads across baseline/candidates/finalists, zero recalibration for custom benchmarks or retries, and separate calibration evidence.
- [ ] Establish failures, then reuse existing configuration, server lifecycle, and benchmark workers for probes; do not duplicate subprocess management.
- [ ] Select the last stable load below detected saturation; record when a hard limit prevents finding saturation. Persist every decision and stopping reason.
- [ ] Save the resolved YAML and calibration evidence outside the repository; link them from the report through the optional hook. Ensure offline reports retain the summary.
- [ ] Run focused calibration/orchestration tests and all relevant lifecycle/report regressions.

## Task 5: Document and verify the complete feature

**Existing documentation edits:** `README.md` gets one minimal example and a link; `docs/guides/configuration.md` gets a short link to new `docs/guides/auto-tuning.md`. Avoid rewriting the quick start or full YAML reference.

**Private tests:** Add `test_auto_objectives.py`, `test_auto_configuration.py`, `test_auto_calibration.py`, `test_auto_lifecycle.py`, and a GPU harness under the existing private test directory, split further if required by file limits.

- [ ] Document minimal throughput/percentile YAML, custom workloads, calibration bounds, fixed-load interpretation, aggregation, and export paths.
- [ ] Run focused private tests after each change; run the full relevant private regression selection after integration, plus Ruff and mypy on the implementation.
- [ ] Run real GPU experiments for generated-workload calibration with throughput and percentile-latency objectives, plus a supplied benchmark that bypasses calibration. Verify P55 and P99 are actually collected and ranked.
- [ ] Use the discovered Ubuntu/WSL RTX 3080 10 GB environment, vLLM 0.30.0, GuideLLM 0.7.4, and local OPT-125M model. Record actual runtime versions and hardware availability when executing.
- [ ] Store commands, benchmark configuration, resolved loads, measurements, model/backend details, and source diff identity under `C:/Users/Ido/Desktop/vTune-private-tests/auto-tuning-evidence/`.
- [ ] Review the final diff for unnecessary existing-file edits, unsafe commands, artifacts, and file-length violations. Report exact changed files and verification results; do not create a PR or release without a separate request.

## Approval boundary

Only this plan document has been created. No application code or existing documentation has been changed for this feature. Ask for approval of this revised plan before implementation; request scope approval if additional existing files prove necessary.

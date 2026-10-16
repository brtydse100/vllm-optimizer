# Performance constraints implementation plan

Spec: ../specs/2026-10-16-performance-constraints-design.md
Execution: primary agent implements; one user-requested judge reviews the PR.
Workspace: isolated `feat/performance-constraints` branch from origin/main.
Private tests: C:/Users/Ido/Desktop/vTune-private-tests (synthetic fixtures).

## Task 1: Parse and evaluate limits

- Add config/constraints.py and config/constraint_units.py for typed policy,
  strict validation, canonical units, and serialization.
- Add managers/constraints.py for matched-workload repeat aggregation and
  per-workload evidence with finite normalized sampler violations.
- Extend benchmarks/metrics.py to preserve backend-supplied p95.
- Wire validation in config/preflight.py and search/factory.py.
- First write private unit/evaluation tests, observe failures, implement,
  then run those tests and the relevant private metric/config regressions.

## Task 2: Search and winner selection

- Add violation vectors to TrialScore and filter infeasible scores in ranking.
- Carry vectors through trial_executor, run_session, finalist validation, and
  baseline selection. Keep the original score for sampler feedback.
- Add optional constraints to SearchSession.complete and every implementation;
  configure Optuna's TPE constraints callback only for constrained runs.
- Preserve constraints when constructing fresh scoring policies.
- First test feasible ranking, baseline rejection, actual Optuna study evidence,
  full search budget, and finalist behavior; implement and run regressions.

## Task 3: Artifacts, reports, and offline compatibility

- Persist policy/evidence through run_results and orchestrator wiring.
- Add reporting/constraints.py; include a target evidence table and CSV fields.
- Pass policy through ReportContext and finalization, offline loading,
  accepted score checks, and reclassification. Recompute eligibility from
  accepted measurements rather than trusting recorded pass flags.
- Test no feasible candidate, feasible baseline, tampering, offline rendering,
  and reclassification before implementing each behavior.
- Run full private/public suites; split touched modules exceeding 200 lines.

## Task 4: Documentation, PR, and judge

- Update docs/guides/configuration.md and docs/guides/search-and-scoring.md.
- Run Ruff, mypy, both test suites, and inspect the final diff.
- Push the feature branch; open/update a draft PR and register it with T3 Code.
- Dispatch a read-only judge with the PR, spec, diff, and test evidence.
- Validate and fix material findings with regression tests; report limitations.
- No merge or release is requested.

## Review focus

Check workload coverage, unavailable percentiles, repeat/request eligibility,
nonfinite values, unit conversion/overflow, zero bounds, signed sampler feedback,
infeasible baselines, finalist reselection, persistence/retry fidelity, offline
tampering, legacy compatibility, and HTML escaping.

## Execution record

- Tasks 1–3 complete: parsing/evaluation, constrained search, feasible selection,
  artifacts/retries/finalists, and offline evidence verification implemented.
- Task 4 complete: user-facing docs updated; private tests, public tests, Ruff,
  mypy, strict MkDocs, and source review passed. Combined suite: 690 tests and
  12 subtests passed, with two environment-dependent skips.
- Judge findings reproduced and fixed: duplicate repeat/workload identities,
  overflow, latency evidence redaction, and offline required-run integrity.
- Tests remain outside the checkout; no GPU/model benchmark was executed.
- PR #33: https://github.com/brtydse100/vllm-optimizer/pull/33

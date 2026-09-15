"""Fixed-budget finalist validation, with the legacy drift-only fallback."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

from vllm_optimizer.config.finalist_validation import FinalistPolicy
from vllm_optimizer.domain.results import WorkerStatus
from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.execution.slots import WorkerSlot
from vllm_optimizer.managers.run_documents import score_document
from vllm_optimizer.managers.run_results import RunResultsManager
from vllm_optimizer.managers.run_session import RunAccumulator
from vllm_optimizer.managers.scoring import TrialScore
from vllm_optimizer.measurement import benchmark_samples, sequentially_drifted
from vllm_optimizer.search.grid import TrialParameters

TrialRun = Callable[
    [Path, TrialParameters, WorkerSlot | None, str | None],
    Awaitable[tuple[TrialReport, TrialScore | None, Mapping[str, float]]],
]


async def validate_drifted_finalists(
    directory: Path,
    session: RunAccumulator,
    results: RunResultsManager,
    run_id: str,
    started_at: str,
    metric: str,
    threshold: float,
    parameters_by_id: Mapping[str, TrialParameters],
    slots_by_id: Mapping[str, WorkerSlot | None],
    run_trial: TrialRun,
    warn: Callable[[str], None],
    source_run_id: str | None,
    sources: Mapping[str, Mapping[str, str]],
    policy: FinalistPolicy | None = None,
) -> None:
    reports = {report.trial_id: report for report in session.reports}
    selected = tuple(item.trial_id for item in session.ranking[: policy.top_k if policy else 2])
    if policy is not None:
        if "baseline" in reports:
            selected = ("baseline", *selected)
        session.validation = {
            "status": "running",
            "selected_trials": list(selected),
            "repeats": policy.repeats,
            "drift_threshold": threshold,
            "search_ranking": [score_document(item) for item in session.ranking],
            "search_baseline": score_document(session.baseline) if session.baseline else None,
        }
        session.baseline = None
        session.persist(results, run_id, metric, "running", started_at, None, source_run_id, sources)
    for trial_id in selected:
        report = reports.get(trial_id)
        if policy is None and (report is None or not _report_has_drift(report, metric, threshold)):
            continue
        parameters = parameters_by_id[trial_id]
        warn(f"Sequentially validating finalist {trial_id}")
        validated, score, by_benchmark = await run_trial(
            directory, parameters, slots_by_id.get(trial_id), "finalist-validation" if policy else "validation-001"
        )
        session.replace(parameters, validated, score, by_benchmark)
        if validated.status is WorkerStatus.INTERRUPTED:
            if policy:
                session.validation["status"] = "interrupted"
            session.persist(results, run_id, metric, "interrupted", started_at, None, source_run_id, sources)
            return
        session.persist(results, run_id, metric, "running", started_at, None, source_run_id, sources)
    if policy:
        session.validation["status"] = "completed"
        session.persist(results, run_id, metric, "running", started_at, None, source_run_id, sources)


def _report_has_drift(report: TrialReport, metric: str, threshold: float) -> bool:
    return any(
        sequentially_drifted(values, threshold) for values in benchmark_samples(report.benchmarks, metric).values()
    )

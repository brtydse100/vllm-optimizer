"""Re-evaluate stored benchmark evidence without launching runtime processes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vllm_optimizer.benchmarks.policy import BenchmarkPolicy
from vllm_optimizer.domain.results import Failure, WorkerStatus
from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.run_results import RunResultsManager
from vllm_optimizer.managers.scoring import ScoringManager
from vllm_optimizer.reporting.context import ReportContext
from vllm_optimizer.reporting.offline import _benchmark_policy, _load_trial, _read_object
from vllm_optimizer.reporting.reclassify_scores import benchmark_score as _benchmark_score
from vllm_optimizer.reporting.reclassify_scores import results as _results
from vllm_optimizer.reporting.reclassify_scores import trial_score as _trial_score
from vllm_optimizer.reporting.reporter import Reporter
from vllm_optimizer.reproduction.accepted import accepted_manifest
from vllm_optimizer.workers.completion import max_requests, reported_request_total, request_count_failure


@dataclass(frozen=True, slots=True)
class ReclassifiedReport:
    directory: Path
    result: Path
    csv: Path
    html: Path


def reclassify_run(run: Path, maximum: float, output: Path | None = None) -> ReclassifiedReport:
    if isinstance(maximum, bool) or not 0 <= maximum <= 100:
        raise ValueError("maximum failure percentage must be between 0 and 100")
    source = Path(run).resolve()
    document = _read_object(source / "result.json", "run result")
    summaries = document.get("trials")
    if not isinstance(summaries, list):
        raise ValueError("run result has invalid trials")
    warnings: list[str] = []
    stored = tuple(_load_trial(source, item, warnings) for item in summaries if isinstance(item, Mapping))
    destination = Path(output).resolve() if output else _destination(source)
    if destination == source or destination.exists():
        raise ValueError(f"re-evaluation output must be a new directory: {destination}")
    benchmark_policy = _benchmark_policy(document, source, stored).with_maximum(maximum)
    policy = _policy(source, stored, str(document.get("maximize", "")), benchmark_policy)
    expected_requests = _expected_requests(source, stored)
    raw_validation = document.get("finalist_validation")
    validation = raw_validation if isinstance(raw_validation, Mapping) else {}
    if validation:
        repeats = validation.get("repeats")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
            raise ValueError("invalid finalist validation repeat budget")
        policy = ScoringManager(policy.metric, repeats, policy.required_runs, maximum, policy.repeat_aggregation)

    def is_accepted(item: TrialReport) -> bool:
        return not validation or item.execution.get("artifact_subdirectory") == "finalist-validation"

    trials = tuple(_reclassify(item, policy, expected_requests) if is_accepted(item) else item for item in stored)
    accepted = tuple(item for item in trials if is_accepted(item))
    scores = [_trial_score(source, item, policy) for item in accepted]
    valid = [item for item in scores if item is not None]
    baseline_id = "baseline" if any(item.trial_id == "baseline" for item in stored) else _baseline_id(document)
    baseline = next((item for item in valid if item.trial_id == baseline_id), None)
    ranking = policy.rank([item for item in valid if item.trial_id != baseline_id])
    by_benchmark = {
        name: policy.rank([score for trial in accepted if (score := _benchmark_score(source, trial, policy, name))])
        for name in policy.required_runs
    }
    destination.mkdir(parents=True)
    result = RunResultsManager(destination / "result.json", benchmark_policy=benchmark_policy.to_dict()).save(
        str(document.get("run_id", source.name)),
        policy.metric,
        trials,
        ranking,
        by_benchmark,
        baseline,
        status="completed",
        source_run_id=source.name,
        finalist_validation=validation,
    )
    context = ReportContext(
        str(document.get("run_id", source.name)),
        "completed",
        source_run_id=source.name,
        benchmark_rankings=by_benchmark,
        benchmark_names=policy.required_runs,
        minimum_repeats=policy.minimum_repeats,
        drift_threshold=benchmark_policy.drift_threshold,
        maximum_failure_percentage=maximum,
        finalist_validation=validation,
        repeat_aggregation=benchmark_policy.repeat_aggregation,
    )
    csv_path, html = Reporter(destination, source).write(policy.metric, trials, ranking, baseline, context)
    return ReclassifiedReport(destination, result, csv_path, html)


def _policy(
    run: Path, trials: tuple[TrialReport, ...], metric: str, benchmark_policy: BenchmarkPolicy
) -> ScoringManager:
    if not trials:
        raise ValueError("source run contains no trials")
    benchmark = accepted_manifest(run, trials[0].trial_id, trials[0].execution).get("benchmark", {})
    if not isinstance(benchmark, Mapping):
        raise ValueError("source manifest has invalid benchmark policy")
    names = tuple(
        str(item.get("name")) for item in benchmark.get("runs", ()) if isinstance(item, Mapping) and item.get("name")
    )
    return ScoringManager(
        metric,
        benchmark_policy.minimum_repeats,
        names,
        benchmark_policy.maximum_failure_percentage,
        benchmark_policy.repeat_aggregation,
    )


def _reclassify(
    report: TrialReport, policy: ScoringManager, expected_requests: Mapping[str, int] | None = None
) -> TrialReport:
    measurements = _results(report)
    score = policy.score(measurements)
    quality = policy.quality(measurements)
    counts_valid = _request_counts_valid(measurements, expected_requests or {}, policy.max_failure_percentage)
    request_failure = report.failure and report.failure.code in {
        "benchmark_requests_incomplete",
        "benchmark_no_completed_requests",
    }
    if (
        score is not None
        and counts_valid
        and not quality.excluded_workloads
        and (report.status is WorkerStatus.COMPLETED or request_failure)
    ):
        return TrialReport(
            1,
            report.trial_id,
            WorkerStatus.COMPLETED,
            report.benchmarks,
            report.artifacts,
            report.attempts,
            None,
            report.execution,
        )
    if report.status is WorkerStatus.COMPLETED:
        failure = Failure("benchmark_requests_incomplete", "Stored requests exceed the selected failure policy")
        return TrialReport(
            1,
            report.trial_id,
            WorkerStatus.FAILED,
            report.benchmarks,
            report.artifacts,
            report.attempts,
            failure,
            report.execution,
        )
    return report


def _request_counts_valid(
    measurements: tuple[object, ...], expected_requests: Mapping[str, int], maximum: float
) -> bool:
    for result in measurements:
        name = str(getattr(result, "run_name", ""))
        backend = str(getattr(result, "backend", "unknown"))
        expected = expected_requests.get(name)
        is_vllm = backend.lower().startswith("vllm")
        if expected is None and is_vllm:
            expected = reported_request_total(result)
        if (expected is not None or is_vllm) and request_count_failure(result, expected, backend, maximum) is not None:
            return False
    return True


def _expected_requests(run: Path, trials: tuple[TrialReport, ...]) -> dict[str, int]:
    manifest = accepted_manifest(run, trials[0].trial_id, trials[0].execution)
    benchmark = manifest.get("benchmark", {})
    runs = benchmark.get("runs", ()) if isinstance(benchmark, Mapping) else ()
    expected: dict[str, int] = {}
    for item in runs if isinstance(runs, list | tuple) else ():
        if not isinstance(item, Mapping) or not item.get("name"):
            continue
        has_limit, count = max_requests(item)
        if has_limit and count is not None:
            expected[str(item["name"])] = count
    return expected


def _baseline_id(document: Mapping[str, object]) -> str | None:
    value = document.get("baseline")
    return str(value.get("trial_id")) if isinstance(value, Mapping) else None


def _destination(run: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    return run / "reclassified" / stamp

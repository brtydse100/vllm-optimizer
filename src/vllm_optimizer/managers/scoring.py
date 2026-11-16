"""Selection of the best completed trial for one configured metric."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from statistics import fmean, median

from vllm_optimizer.config.constraints import Constraint
from vllm_optimizer.domain.benchmark import BenchmarkResult
from vllm_optimizer.managers.scoring_values import _eligible, _request_counts, _workload_identity
from vllm_optimizer.objectives import extract_objective, objective_from_key


@dataclass(frozen=True, slots=True)
class TrialScore:
    trial_id: str
    value: float
    server_args: Mapping[str, object]
    server_env: Mapping[str, object]
    successful_requests: int = 0
    errored_requests: int = 0
    incomplete_requests: int = 0
    excluded_workloads: int = 0
    constraint_violations: tuple[float, ...] = ()
    direction: str = "maximize"

    @property
    def feasible(self) -> bool:
        return all(value <= 0 for value in self.constraint_violations)

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("score direction must be maximize or minimize")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float) or not isfinite(self.value):
            raise ValueError("trial score must be finite")
        if any(
            isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v) for v in self.constraint_violations
        ):
            raise ValueError("constraint violations must be finite numbers")

    @property
    def error_rate(self) -> float:
        total = self.successful_requests + self.errored_requests + self.incomplete_requests
        return (self.errored_requests + self.incomplete_requests) / total if total else 0.0


@dataclass(frozen=True, slots=True)
class QualitySummary:
    successful: int = 0
    errored: int = 0
    incomplete: int = 0
    excluded_workloads: int = 0


class ScoringManager:
    def __init__(
        self,
        metric: str,
        minimum_repeats: int = 1,
        required_runs: tuple[str, ...] = (),
        max_failure_percentage: float = 0,
        repeat_aggregation: str = "mean",
        constraints: tuple[Constraint, ...] = (),
        direction: str | None = None,
    ) -> None:
        if not metric.strip():
            raise ValueError("optimization.maximize must not be empty")
        if not isinstance(minimum_repeats, int) or isinstance(minimum_repeats, bool) or minimum_repeats < 1:
            raise ValueError("minimum repeats must be positive")
        self.constraints = constraints
        self.direction = direction or objective_from_key(metric).direction
        self.metric = metric
        self.minimum_repeats = minimum_repeats
        self.required_runs = required_runs
        if repeat_aggregation not in {"mean", "median"}:
            raise ValueError("repeat aggregation must be mean or median")
        self.repeat_aggregation = repeat_aggregation
        if (
            isinstance(max_failure_percentage, bool)
            or not isinstance(max_failure_percentage, int | float)
            or not 0 <= max_failure_percentage <= 100
        ):
            raise ValueError("maximum failure percentage must be between 0 and 100")
        self.max_failure_percentage = float(max_failure_percentage)

    def score(self, results: tuple[BenchmarkResult, ...]) -> float | None:
        scores = self.score_each(results)
        if self.required_runs and any(name not in scores for name in self.required_runs):
            return None
        values = tuple(scores.values())
        return fmean(values) if values else None

    def score_each(self, results: tuple[BenchmarkResult, ...]) -> dict[str, float]:
        grouped: dict[str, list[dict[str, float | None]]] = {}
        for result in results:
            result_values = {
                _workload_identity(getattr(workload, "index", 0), getattr(workload, "configuration", {})): (
                    extract_objective(workload.metrics, objective_from_key(self.metric))
                    if _eligible(workload.metrics, self.max_failure_percentage)
                    else None
                )
                for workload in result.workloads
            }
            if result_values:
                grouped.setdefault(result.run_name, []).append(result_values)
        scores: dict[str, float] = {}
        for name, repeats in grouped.items():
            identities = set(repeats[0])
            if any(set(repeat) != identities for repeat in repeats[1:]):
                continue
            if self.repeat_aggregation == "median":
                repeat_scores = [
                    fmean(values)
                    for repeat in repeats
                    if (values := [value for value in repeat.values() if value is not None])
                ]
                if len(repeat_scores) >= self.minimum_repeats:
                    scores[name] = float(median(repeat_scores))
                continue
            workload_scores: list[float] = []
            for identity in identities:
                repeat_values = [value for repeat in repeats if (value := repeat[identity]) is not None]
                if len(repeat_values) < self.minimum_repeats:
                    break
                aggregate = fmean(repeat_values) if self.repeat_aggregation == "mean" else median(repeat_values)
                workload_scores.append(float(aggregate))
            else:
                if workload_scores:
                    scores[name] = fmean(workload_scores)
        return scores

    @staticmethod
    def rank(scores: list[TrialScore]) -> tuple[TrialScore, ...]:
        return tuple(
            sorted(
                (item for item in scores if item.feasible),
                key=lambda item: (
                    item.value if item.direction == "minimize" else -item.value,
                    item.error_rate,
                    item.errored_requests + item.incomplete_requests,
                    item.trial_id,
                ),
            )
        )

    def quality(self, results: tuple[BenchmarkResult, ...]) -> QualitySummary:
        successful = errored = incomplete = excluded = 0
        for result in results:
            for workload in result.workloads:
                counts = _request_counts(workload.metrics)
                successful += counts[0]
                errored += counts[1]
                incomplete += counts[2]
                excluded += not self._eligible(workload.metrics)
        return QualitySummary(successful, errored, incomplete, excluded)

    def _eligible(self, metrics: Mapping[str, object]) -> bool:
        return _eligible(metrics, self.max_failure_percentage)

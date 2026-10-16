"""Evaluate limits per matched workload, retaining evidence and sampler feedback."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isfinite
from statistics import fmean

from vllm_optimizer.config.constraints import Constraint
from vllm_optimizer.domain.benchmark import BenchmarkResult, WorkloadResult
from vllm_optimizer.managers.scoring_values import _eligible, _workload_identity


@dataclass(frozen=True, slots=True)
class ConstraintAssessment:
    violations: tuple[float, ...]
    evidence: tuple[dict[str, object], ...]

    @property
    def feasible(self) -> bool:
        return all(value <= 0 for value in self.violations)

    def to_dict(self) -> dict[str, object]:
        return {"met": self.feasible, "violations": list(self.violations), "workloads": list(self.evidence)}


def assess_constraints(
    results: tuple[BenchmarkResult, ...],
    constraints: tuple[Constraint, ...],
    minimum_repeats: int = 1,
    required_runs: tuple[str, ...] = (),
    maximum_failure_percentage: float = 0,
) -> ConstraintAssessment:
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.run_name, []).append(result)
    violations: list[float] = []
    evidence: list[dict[str, object]] = []
    for position, constraint in enumerate(constraints):
        values: list[float] = []
        for name in sorted(set(grouped) | set(required_runs)):
            repeats = grouped.get(name, [])
            repeat_ids = [result.repeat_index for result in repeats]
            distinct = len(repeat_ids) == len(set(repeat_ids))
            workloads = [
                {_workload_identity(w.index, w.configuration): w for w in result.workloads} for result in repeats
            ]
            distinct = distinct and all(
                len(items) == len(result.workloads) for items, result in zip(workloads, repeats, strict=True)
            )
            identities = set().union(*(set(items) for items in workloads))
            for identity in sorted(identities) if identities else ("missing",):
                matched = [items.get(identity) for items in workloads]
                eligible = [
                    item for item in matched if item is not None and _eligible(item.metrics, maximum_failure_percentage)
                ]
                samples = [_statistic(item, constraint) for item in eligible]
                valid = (
                    distinct
                    and all(item is not None for item in matched)
                    and len(eligible) >= minimum_repeats
                    and all(value is not None for value in samples)
                )
                measured = _mean([value for value in samples if value is not None]) if valid else None
                if measured is not None and not isfinite(measured):
                    measured = None
                violation = _violation(measured, constraint)
                values.append(violation)
                evidence.append(
                    {
                        "constraint": position,
                        "benchmark": name,
                        "workload": eligible[0].index if eligible else None,
                        "metric": constraint.metric,
                        "statistic": constraint.statistic,
                        "operator": constraint.operator,
                        "bound": constraint.bound,
                        "unit": constraint.unit,
                        "measured": measured,
                        "repeats": len(eligible),
                        "met": violation <= 0,
                    }
                )
        violations.append(max(values) if values else 1.0)
    return ConstraintAssessment(tuple(violations), tuple(evidence))


def _mean(values: list[float]) -> float:
    try:
        return fmean(values)
    except OverflowError:
        return fsum(value / len(values) for value in values)


def _statistic(workload: WorkloadResult, constraint: Constraint) -> float | None:
    value = workload.metrics.get(constraint.metric)
    if isinstance(value, Mapping):
        key = "median" if constraint.statistic == "p50" else constraint.statistic
        value = value.get(key)
    elif constraint.statistic != "average":
        return None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value) else None


def _violation(measured: float | None, constraint: Constraint) -> float:
    if measured is None:
        return 1.0
    difference = measured - constraint.bound if constraint.operator == "max" else constraint.bound - measured
    return max(-1e12, min(1e12, difference / max(1.0, constraint.bound)))

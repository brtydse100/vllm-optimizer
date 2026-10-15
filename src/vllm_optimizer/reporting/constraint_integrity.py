"""Reconstruct constraint eligibility from accepted immutable measurements."""

from collections.abc import Mapping
from pathlib import Path

from vllm_optimizer.benchmarks.policy import BenchmarkPolicy
from vllm_optimizer.config.constraints import Constraint, stored_constraints
from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.scoring import TrialScore
from vllm_optimizer.reporting.constraints import assess_report
from vllm_optimizer.reporting.context import ReportContext
from vllm_optimizer.reproduction.accepted import accepted_manifest


def load_constraints(
    document: Mapping[str, object], run: Path, trials: tuple[TrialReport, ...]
) -> tuple[Constraint, ...]:
    manifests = tuple(accepted_manifest(run, item.trial_id, item.execution) for item in trials)
    return stored_constraints(document, manifests)


def validate_constraint_scores(
    document: Mapping[str, object],
    trials: tuple[TrialReport, ...],
    ranking: tuple[TrialScore, ...],
    baseline: TrialScore | None,
    by_benchmark: Mapping[str, tuple[TrialScore, ...]],
    policy: BenchmarkPolicy,
    constraints: tuple[Constraint, ...],
) -> None:
    if not constraints:
        return
    validation = document.get("finalist_validation", {})
    context = ReportContext(
        constraints=constraints,
        minimum_repeats=policy.minimum_repeats,
        maximum_failure_percentage=policy.maximum_failure_percentage,
        benchmark_names=tuple(by_benchmark),
        finalist_validation=validation if isinstance(validation, Mapping) else {},
    )
    assessments = {item.trial_id: assess_report(item, context) for item in trials}
    recorded = document.get("constraint_assessments")
    if recorded != {name: item.to_dict() for name, item in assessments.items()}:
        raise ValueError("stored performance target evidence disagrees with accepted measurements")
    for score in (*ranking, *(score for values in by_benchmark.values() for score in values)):
        assessment = assessments[score.trial_id]
        if not assessment.feasible or score.constraint_violations != assessment.violations:
            raise ValueError(f"stored ranking violates performance constraints: {score.trial_id}")
    if baseline and baseline.constraint_violations != assessments[baseline.trial_id].violations:
        raise ValueError("stored baseline performance constraints disagree with accepted measurements")

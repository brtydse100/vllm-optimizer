"""Reject stored scores that disagree with their accepted execution's measurements."""

from collections.abc import Mapping
from math import isclose

from vllm_optimizer.benchmarks.policy import BenchmarkPolicy
from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.scoring import ScoringManager, TrialScore
from vllm_optimizer.reporting.reclassify_scores import results


def validate_accepted_scores(
    document: Mapping[str, object],
    trials: tuple[TrialReport, ...],
    ranking: tuple[TrialScore, ...],
    baseline: TrialScore | None,
    policy: BenchmarkPolicy,
) -> None:
    # Legacy exports do not retain enough policy to reconstruct their original score.
    if not document.get("benchmark_policy"):
        return
    metric = str(document.get("maximize", "unknown"))
    # Verify the recorded value from available evidence; offline rendering does not
    # reclassify sparse legacy runs under a newly recorded minimum-repeat policy.
    scoring = ScoringManager(
        metric,
        1,
        max_failure_percentage=policy.maximum_failure_percentage,
        repeat_aggregation=policy.repeat_aggregation,
    )
    reports = {item.trial_id: item for item in trials}
    for score in (*ranking, *((baseline,) if baseline else ())):
        measured = scoring.score(results(reports[score.trial_id]))
        if measured is None or not isclose(measured, score.value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Stored score disagrees with accepted execution: {score.trial_id}")

"""Conservative descriptive decisions from a completed, fixed validation budget."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite

from vllm_optimizer.domain.results import WorkerStatus
from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.scoring import ScoringManager, TrialScore
from vllm_optimizer.measurement import sequentially_drifted, summarize
from vllm_optimizer.reporting.workloads import samples, scenarios


@dataclass(frozen=True, slots=True)
class FinalistDecision:
    status: str
    winner_trial_id: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decide(
    trials: tuple[TrialReport, ...],
    ranking: tuple[TrialScore, ...],
    baseline: TrialScore | None,
    metric: str,
    validation: Mapping[str, object],
) -> FinalistDecision:
    if not validation:
        return FinalistDecision("not_validated", None, "No clear winner: fixed-budget validation was not run.")
    if validation.get("status") != "completed":
        return _unclear("finalist validation did not finish")
    selected = validation.get("selected_trials")
    repeats, threshold = validation.get("repeats"), validation.get("drift_threshold", 0.05)
    if (
        not isinstance(selected, list | tuple)
        or not all(isinstance(item, str) for item in selected)
        or len(selected) < 2
        or len(set(selected)) != len(selected)
        or isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 2
        or isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not isfinite(threshold)
        or threshold < 0
    ):
        return _unclear("at least two candidates and a valid repeat budget are required")
    scores = {item.trial_id: item for item in (*ranking, *((baseline,) if baseline else ()))}
    reports = {item.trial_id: item for item in trials}
    for trial_id in selected:
        report, score = reports.get(trial_id), scores.get(trial_id)
        if (
            report is None
            or report.status is not WorkerStatus.COMPLETED
            or report.execution.get("artifact_subdirectory") != "finalist-validation"
            or score is None
            or not isfinite(score.value)
            or score.excluded_workloads
        ):
            return _unclear("a selected candidate has missing, failed, or ineligible validation evidence")
    ordered = ScoringManager.rank([scores[trial_id] for trial_id in selected])
    best = ordered[0]
    for other in ordered[1:]:
        if best.value <= other.value:
            return _unclear("validated objective scores are tied")
        issue = _compare(reports[best.trial_id], reports[other.trial_id], metric, repeats, float(threshold))
        if issue:
            return _unclear(f"{best.trial_id} vs {other.trial_id}: {issue}")
    return FinalistDecision(
        "evidence_favors",
        best.trial_id,
        f"Validation favors {best.trial_id} among the selected candidates on every matched workload. "
        "This is descriptive evidence, not a statistical significance claim or a global optimum.",
    )


def _unclear(reason: str) -> FinalistDecision:
    return FinalistDecision("inconclusive", None, f"No clear winner: {reason}.")


def _compare(best: TrialReport, other: TrialReport, metric: str, repeats: int, threshold: float) -> str | None:
    a, b = scenarios(best), scenarios(other)
    if not a or a.keys() != b.keys():
        return "workload coverage differs or is unavailable"
    for key in a:
        x, y = samples(a[key], (metric,)), samples(b[key], (metric,))
        if any(
            len(values) != repeats
            or len(observations) != repeats
            or len({item.repeat for item in observations}) != repeats
            or any(item.repeat == "Unavailable" for item in observations)
            for values, observations in ((x, a[key]), (y, b[key]))
        ):
            return "the full budget of distinct, finite repeats is unavailable"
        if any(sequentially_drifted(values, threshold) for values in (x, y)):
            return "repeat measurements show drift"
        left, right = summarize(x), summarize(y)
        assert left.confidence_low is not None and right.confidence_high is not None
        # Require separation of both observed ranges and t intervals of workload means.
        # This is a per-workload guard, not an interval for the aggregate scoring objective.
        if min(min(x), left.confidence_low) <= max(max(y), right.confidence_high):
            return "measurement uncertainty overlaps or a workload does not favor the leading score"
    return None

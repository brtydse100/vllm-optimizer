"""Objective labels and comparisons shared by live and offline reports."""

from vllm_optimizer.managers.scoring import TrialScore
from vllm_optimizer.objectives import objective_from_key


def score_loss(score: TrialScore) -> float:
    return score.value if score.direction == "minimize" else -score.value


def improvement(candidate: TrialScore | None, baseline: TrialScore | None) -> float | None:
    if candidate is None or baseline is None or baseline.value == 0:
        return None
    change = (candidate.value - baseline.value) / abs(baseline.value) * 100
    return -change if candidate.direction == "minimize" else change


def objective_label(metric: str) -> str:
    objective = objective_from_key(metric)
    unit = "ms" if objective.direction == "minimize" else ""
    statistic = f"P{objective.percentile}" if objective.percentile else "mean"
    return f"{objective.direction.title()} {objective.metric} ({statistic}{', ' + unit if unit else ''})"

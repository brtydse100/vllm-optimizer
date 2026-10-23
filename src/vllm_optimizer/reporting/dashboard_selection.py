"""Select and reproduce dashboard recommendations."""

from __future__ import annotations

from vllm_optimizer.managers.scoring import TrialScore
from vllm_optimizer.reporting.objective import improvement as improvement
from vllm_optimizer.reporting.objective import score_loss


def best_observed(best_tuned: TrialScore | None, baseline: TrialScore | None) -> TrialScore | None:
    candidates = [item for item in (baseline, best_tuned) if item is not None and item.feasible]
    return (
        min(
            candidates,
            key=lambda item: (score_loss(item), item.error_rate, item.errored_requests + item.incomplete_requests),
        )
        if candidates
        else None
    )

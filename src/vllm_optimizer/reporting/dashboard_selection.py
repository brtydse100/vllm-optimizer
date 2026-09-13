"""Select and reproduce dashboard recommendations."""

from __future__ import annotations

from vllm_optimizer.managers.scoring import TrialScore


def improvement(best: TrialScore | None, baseline: TrialScore | None) -> float | None:
    if best is None or baseline is None or baseline.value == 0:
        return None
    return (best.value - baseline.value) / abs(baseline.value) * 100


def best_observed(best_tuned: TrialScore | None, baseline: TrialScore | None) -> TrialScore | None:
    candidates = [item for item in (baseline, best_tuned) if item is not None]
    return (
        min(
            candidates,
            key=lambda item: (-item.value, item.error_rate, item.errored_requests + item.incomplete_requests),
        )
        if candidates
        else None
    )

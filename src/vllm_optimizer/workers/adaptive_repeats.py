"""Stop optional repeat workers when initial evidence is clearly weak."""

from dataclasses import dataclass, field

from vllm_optimizer.config.adaptive_repeats import AdaptiveRepeatPolicy
from vllm_optimizer.domain.benchmark import BenchmarkResult
from vllm_optimizer.domain.results import WorkerResult
from vllm_optimizer.managers.scoring import ScoringManager
from vllm_optimizer.workers.base import TrialContext, Worker

_STOP = "adaptive_repeats_stopped"


@dataclass(slots=True)
class AdaptiveRepeatGate:
    scorer: ScoringManager
    policy: AdaptiveRepeatPolicy
    baseline_mean_score: float | None
    planned_repeats: int
    completed_repeats: int
    name: str = "adaptive_repeat_gate"

    async def execute(self, context: TrialContext) -> WorkerResult[None]:
        raw = context.values.get("benchmark_results", ())
        results = raw if isinstance(raw, tuple) and all(isinstance(item, BenchmarkResult) for item in raw) else ()
        score = self.scorer.score(results)
        baseline_mean = self.baseline_mean_score
        status = "continued"
        reason = "initial score is competitive"
        if baseline_mean is None or baseline_mean <= 0:
            status, reason = "continued_without_reference", "no positive baseline mean score was available"
        elif score is None or score <= 0:
            status, reason = "continued_without_comparison", "initial score was unavailable or non-positive"
        elif score < baseline_mean * self.policy.minimum_relative_score:
            shortfall = (1 - self.policy.minimum_relative_score) * 100
            status, reason = "stopped", f"initial score was more than {shortfall:g}% below the baseline mean score"
            context.values[_STOP] = True
        context.execution["adaptive_repeats"] = {
            "status": status,
            "reason": reason,
            "decision_repeats": self.completed_repeats,
            "planned_repeats": self.planned_repeats,
            "initial_score": score,
            "baseline_mean_score": baseline_mean,
            "minimum_relative_score": self.policy.minimum_relative_score,
        }
        return WorkerResult.completed()

    async def cleanup(self, context: TrialContext) -> None:
        return None


@dataclass(slots=True)
class OptionalRepeatWorker:
    worker: Worker
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.worker.name

    async def execute(self, context: TrialContext) -> WorkerResult[None]:
        return WorkerResult.completed() if context.values.get(_STOP) is True else await self.worker.execute(context)

    async def cleanup(self, context: TrialContext) -> None:
        await self.worker.cleanup(context)

import json
import math
from pathlib import Path

import pytest

from vllm_optimizer.benchmarks.vllm import parse_result
from vllm_optimizer.domain.benchmark import BenchmarkResult, WorkloadResult
from vllm_optimizer.domain.results import Failure, WorkerResult, WorkerStatus
from vllm_optimizer.managers.results import ResultsManager
from vllm_optimizer.managers.scoring import ScoringManager, TrialScore
from vllm_optimizer.reporting.reclassify import _reclassify
from vllm_optimizer.workers.base import TrialContext
from vllm_optimizer.workers.benchmark_state import remember_result


def _result(tmp_path: Path, repeat: int, dataset: str, value: float) -> BenchmarkResult:
    metrics = {
        "requests_per_second": {"average": value},
        "request_totals": {"successful": 1, "errored": 0, "incomplete": 0},
    }
    return BenchmarkResult(
        "requests", "synthetic", "1", (WorkloadResult(0, {"dataset": dataset}, metrics),), tmp_path, repeat
    )


def _vllm_result(tmp_path: Path, value: float, successful: int, failed: int) -> BenchmarkResult:
    path = tmp_path / f"result-{successful}-{failed}.json"
    path.write_text(
        json.dumps(
            {
                "backend": "vllm",
                "model_id": "demo",
                "num_prompts": successful + failed,
                "completed": successful,
                "failed": failed,
                "request_throughput": value,
            }
        ),
        encoding="utf-8",
    )
    return parse_result(path, "requests")


def test_mismatched_workload_configurations_do_not_satisfy_repeat_budget(tmp_path: Path) -> None:
    score = ScoringManager("requests_per_second", 2, ("requests",)).score(
        (_result(tmp_path, 1, "dataset-a", 10), _result(tmp_path, 2, "dataset-b", 20))
    )
    assert score is None


def test_failed_repeat_is_persisted_and_blocks_reclassification(tmp_path: Path) -> None:
    context = TrialContext("trial")
    remember_result(context, _vllm_result(tmp_path, 5, 2, 0))
    remember_result(context, _vllm_result(tmp_path, 50, 1, 1), observed=True)
    report = ResultsManager(tmp_path / "result.json").save(
        context, WorkerResult.failed(Failure("benchmark_requests_incomplete", "one repeat failed"))
    )

    assert len(report.benchmarks) == 2
    assert report.benchmarks[1]["errored_requests"] == 1
    assert _reclassify(report, ScoringManager("requests_per_second", 1)).status is WorkerStatus.FAILED


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_metrics_and_trial_scores_are_rejected(tmp_path: Path, value: float) -> None:
    assert ScoringManager("requests_per_second").score((_vllm_result(tmp_path, value, 1, 0),)) is None
    with pytest.raises(ValueError, match="finite"):
        TrialScore("trial", value, {}, {})

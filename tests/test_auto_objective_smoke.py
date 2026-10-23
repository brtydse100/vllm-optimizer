"""Public objective checks using fabricated metrics; no captured benchmark fixtures."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_optimizer.benchmarks.metrics import normalize_guidellm_metrics, normalize_vllm_metrics
from vllm_optimizer.config.models import ExperimentConfig, VTuneConfig
from vllm_optimizer.domain.benchmark import BenchmarkResult, WorkloadResult
from vllm_optimizer.managers.scoring import ScoringManager, TrialScore
from vllm_optimizer.objectives import Objective, extract_objective, objective_from_config
from vllm_optimizer.reporting.dashboard_selection import best_observed, improvement


def test_latency_uses_requested_percentile_and_lower_values_win():
    config = VTuneConfig(
        1,
        ExperimentConfig("fabricated"),
        {"model": "."},
        optimization={"objective": {"metric": "end_to_end_latency_ms", "percentile": 55}},
    )
    objective = objective_from_config(config)
    assert objective.key == "end_to_end_latency_ms.p55" and objective.direction == "minimize"
    metrics = normalize_vllm_metrics({"mean_e2el_ms": 50, "p55_e2el_ms": 5, "completed": 1})
    result = BenchmarkResult("sample", "vllm", "example", (WorkloadResult(0, {}, metrics),), Path("not-read.json"))
    assert ScoringManager(objective.key).score((result,)) == 5
    baseline = TrialScore("baseline", 10, {}, {}, direction="minimize")
    candidate = replace(baseline, trial_id="candidate", value=5)
    assert ScoringManager.rank([baseline, candidate]) == (candidate, baseline)
    assert best_observed(candidate, baseline) == candidate
    assert improvement(candidate, baseline) == 50


def test_legacy_maximize_and_mean_latency_remain_distinct():
    config = VTuneConfig(
        1, ExperimentConfig("fabricated"), {"model": "."}, optimization={"maximize": "end_to_end_latency_ms"}
    )
    assert objective_from_config(config).direction == "maximize"
    config = replace(config, optimization={"objective": {"metric": "end_to_end_latency_ms"}})
    assert objective_from_config(config).key == "end_to_end_latency_ms.average"
    assert extract_objective({"end_to_end_latency_ms": {"average": 7}}, objective_from_config(config)) == 7


@pytest.mark.parametrize("percentile", [True, 0, 101, "99", 99.5])
def test_invalid_percentiles_are_rejected(percentile):
    with pytest.raises(ValueError):
        Objective("end_to_end_latency_ms", percentile, "minimize")


def test_missing_percentile_and_non_finite_values_never_fall_back():
    objective = Objective("end_to_end_latency_ms", 55, "minimize")
    assert extract_objective({"end_to_end_latency_ms": {"average": 7, "p99": 10}}, objective) is None
    assert extract_objective({objective.key: float("inf")}, objective) is None
    preserved = {"time_to_first_token_ms.p55": "<redacted>", "time_to_first_token_ms": {"average": 50, "p55": 5}}
    assert extract_objective(preserved, Objective("time_to_first_token_ms", 55, "minimize")) == 5


def test_guidellm_percentile_names_and_units_are_normalized():
    metrics = normalize_guidellm_metrics(
        {"request_latency": {"successful": {"percentiles": {"p01": 0.001, "p05": 0.005, "p99": 0.01}}}}
    )
    assert metrics["end_to_end_latency_ms.p1"] == 1
    assert metrics["end_to_end_latency_ms.p5"] == 5
    assert metrics["end_to_end_latency_ms.p99"] == 10


def test_numeric_percentile_evidence_survives_secret_redaction():
    from vllm_optimizer.reproduction.metric_redaction import redact_benchmarks

    metrics = normalize_vllm_metrics({"mean_ttft_ms": 10, "p55_ttft_ms": 5})
    metrics["api_token"] = "fabricated-value"
    stored = redact_benchmarks([{"workloads": [{"metrics": metrics}]}])[0]["workloads"][0]["metrics"]
    assert stored["time_to_first_token_ms.p55"] == 5
    assert stored["api_token"] == "<redacted>"


def test_probe_extracts_fabricated_summaries_without_gpu(monkeypatch, tmp_path):
    from vllm_optimizer.auto import probes
    from vllm_optimizer.domain.results import WorkerResult
    from vllm_optimizer.workers.base import TrialContext

    metrics = {
        "output_tokens_per_second": {"average": 10},
        "end_to_end_latency_ms": {"average": 1},
        "request_totals": {"successful": 1},
    }
    result = BenchmarkResult(
        "sample", "example", "example", (WorkloadResult(0, {}, metrics),), tmp_path / "not-read.json"
    )

    class FabricatedMeasurements:
        name = "fabricated-measurements"

        async def execute(self, context):
            context.values["benchmark_results"] = (result,)
            return WorkerResult.completed()

        async def cleanup(self, context):
            pass

    monkeypatch.setattr(probes, "build_trial_workers", lambda *args: (None,) * 4 + (FabricatedMeasurements(),))
    config = VTuneConfig(1, ExperimentConfig("example"), {"model": "."}, benchmark={"engine": "vllm"})
    measured = asyncio.run(probes.measure_probe(config, {"name": "sample"}, TrialContext("example"), tmp_path, 1, 10))
    assert measured.throughput == 10 and measured.latency == 1 and measured.failure is None


def test_calibration_freezes_last_stable_fabricated_load(monkeypatch, tmp_path):
    from vllm_optimizer.auto import calibration_worker
    from vllm_optimizer.auto.calibration import ProbeResult
    from vllm_optimizer.workers.base import TrialContext

    async def fabricated_probe(config, run, context, directory, concurrency, timeout, warmup=False):
        return {1: ProbeResult(1, 10, 1), 2: ProbeResult(2, 19, 1.1), 4: ProbeResult(4, 19.1, 1.2)}[concurrency]

    monkeypatch.setattr(calibration_worker, "measure_probe", fabricated_probe)
    run = {"name": "example", "args": {"num-prompts": 8, "max-concurrency": 1}}
    config = VTuneConfig(1, ExperimentConfig("example"), {"model": "."}, benchmark={"engine": "vllm", "runs": [run]})
    worker = calibration_worker.CalibrationWorker(config, tmp_path)
    asyncio.run(worker.execute(TrialContext("example")))
    assert worker.decisions[0]["reason"] == "throughput_plateau"
    assert worker.frozen_runs[0]["args"] == {"num-prompts": 8, "max-concurrency": 2}

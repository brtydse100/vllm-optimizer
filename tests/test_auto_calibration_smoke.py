"""Public probe and calibration checks using fabricated measurements; no GPU data."""

import asyncio

from vllm_optimizer.config.models import ExperimentConfig, VTuneConfig
from vllm_optimizer.domain.benchmark import BenchmarkResult, WorkloadResult


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

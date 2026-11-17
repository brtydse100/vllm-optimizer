"""Construct and measure calibration probes using ordinary benchmark workers."""

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

from vllm_optimizer.auto.calibration import ProbeResult
from vllm_optimizer.config.models import VTuneConfig
from vllm_optimizer.domain.benchmark import BenchmarkResult
from vllm_optimizer.domain.results import WorkerStatus
from vllm_optimizer.managers.results import ResultsManager
from vllm_optimizer.managers.trial import TrialManager
from vllm_optimizer.objectives import Objective, extract_objective
from vllm_optimizer.search.grid import TrialParameters
from vllm_optimizer.workers.base import TrialContext
from vllm_optimizer.workers.factory import build_trial_workers


def with_load(run: Mapping[str, Any], engine: str, concurrency: int, probe: bool = False) -> dict[str, Any]:
    if engine == "vllm":
        args = dict(run.get("args", {}))
        args["max-concurrency"] = concurrency
        if probe:
            args["num-prompts"] = min(512, max(16, concurrency * 2))
        return {**run, "args": args}
    return {
        **run,
        "profile": {"kind": "concurrent", "streams": concurrency, "max_concurrency": concurrency},
        **({"constraints": [{"kind": "max_requests", "count": max(16, concurrency * 2)}]} if probe else {}),
    }


async def measure_probe(
    config: VTuneConfig,
    run: Mapping[str, object],
    context: TrialContext,
    directory: Path,
    concurrency: int,
    timeout: float,
    warmup: bool = False,
) -> ProbeResult:
    benchmark = {key: value for key, value in config.benchmark.items() if key != "adaptive_repeats"}
    benchmark.update(runs=[dict(run)], repeats=1, min_repeats=1, warmup_repeats=int(warmup), max_failure_percentage=0)
    probe_config = replace(config, benchmark=benchmark, timeouts={**config.timeouts, "benchmark": timeout})
    probe_context = TrialContext("calibration", values=dict(context.values))
    workers = build_trial_workers(probe_config, TrialParameters("calibration", {}, {}), directory)[4:]
    timer = asyncio.timeout(timeout)
    async with timer:
        outcome = await TrialManager(workers).execute(probe_context)
    directory.mkdir(parents=True, exist_ok=True)
    ResultsManager(directory / "result.json").save(probe_context, outcome)
    if timer.expired():
        return ProbeResult(concurrency, 0, 0, "probe_timeout")
    if outcome.status is WorkerStatus.INTERRUPTED:
        raise asyncio.CancelledError
    if outcome.status is not WorkerStatus.COMPLETED:
        return ProbeResult(concurrency, 0, 0, outcome.failure.code if outcome.failure else "probe_failed")
    results = probe_context.values.get("benchmark_results", ())
    if not isinstance(results, tuple):
        return ProbeResult(concurrency, 0, 0, "missing_measurements")
    throughputs, latencies = [], []
    for result in results:
        if not isinstance(result, BenchmarkResult):
            continue
        for workload in result.workloads:
            totals = workload.metrics.get("request_totals", {})
            if (
                not isinstance(totals, Mapping)
                or not totals.get("successful")
                or totals.get("errored")
                or totals.get("incomplete")
            ):
                return ProbeResult(concurrency, 0, 0, "requests_failed")
            throughput = extract_objective(workload.metrics, Objective("output_tokens_per_second"))
            latency = extract_objective(workload.metrics, Objective("end_to_end_latency_ms", direction="minimize"))
            if throughput is None or latency is None:
                return ProbeResult(concurrency, 0, 0, "missing_measurements")
            throughputs.append(throughput)
            latencies.append(latency)
    return (
        ProbeResult(concurrency, fmean(throughputs), fmean(latencies))
        if throughputs
        else ProbeResult(concurrency, 0, 0, "missing_measurements")
    )

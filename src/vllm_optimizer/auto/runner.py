"""Prepare auto runs, then hand execution to the unchanged orchestrator."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vllm_optimizer.auto.artifacts import finish_auto_run
from vllm_optimizer.auto.calibration_worker import CalibrationWorker
from vllm_optimizer.auto.hardware import gpu_inventory
from vllm_optimizer.config.models import VTuneConfig
from vllm_optimizer.domain.results import WorkerStatus
from vllm_optimizer.execution.slots import worker_slots
from vllm_optimizer.managers.run_finalization import RunOutcome
from vllm_optimizer.managers.trial import TrialManager
from vllm_optimizer.orchestrator import Orchestrator
from vllm_optimizer.search.grid import TrialParameters
from vllm_optimizer.workers.base import TrialContext
from vllm_optimizer.workers.factory import build_trial_workers


async def run_auto(config: VTuneConfig) -> RunOutcome:
    inventory = gpu_inventory()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    directory = Path(config.experiment.output_dir) / config.experiment.name / "calibration" / stamp
    evidence: dict[str, Any] = {"hardware": inventory, "status": "custom_benchmark", "workloads": []}
    if config.auto.get("generated_benchmark"):
        directory.mkdir(parents=True, exist_ok=True)
        worker = CalibrationWorker(config, directory)
        slot = next((slot for slot in worker_slots(config) if slot.supports({}, config.server)), None)
        startup = build_trial_workers(config, TrialParameters("calibration", {}, {}), directory, slot)[:4]
        context = TrialContext("calibration")
        calibration = await TrialManager((*startup, worker)).execute(context)
        if calibration.status is WorkerStatus.INTERRUPTED:
            raise KeyboardInterrupt("Calibration interrupted")
        if calibration.status is not WorkerStatus.COMPLETED:
            raise ValueError(
                f"Calibration failed: {calibration.failure.message if calibration.failure else 'unknown failure'}"
            )
        config = replace(config, benchmark={**config.benchmark, "runs": worker.frozen_runs})
        evidence.update(status="calibrated", workloads=worker.decisions, directory=str(directory.resolve()))
    # Resolved retries must behave as ordinary experiments and never recalibrate.
    resolved = replace(
        config,
        auto={},
        optimization={key: value for key, value in config.optimization.items() if key not in {"mode", "calibration"}},
    )
    outcome = await Orchestrator(resolved).run()
    finish_auto_run(outcome, resolved, evidence)
    return outcome

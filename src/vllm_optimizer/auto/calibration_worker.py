"""Calibrate workloads on one owned baseline server before optimization."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

from vllm_optimizer.auto.calibration import ProbeResult, calibration_policy, select_load
from vllm_optimizer.auto.probes import measure_probe, with_load
from vllm_optimizer.config.models import VTuneConfig
from vllm_optimizer.domain.results import WorkerResult
from vllm_optimizer.workers.base import TrialContext


@dataclass(slots=True)
class CalibrationWorker:
    config: VTuneConfig
    directory: Path
    decisions: list[dict[str, Any]] = field(default_factory=list)
    frozen_runs: list[dict[str, Any]] = field(default_factory=list)
    name: str = "auto_calibration"

    async def execute(self, context: TrialContext) -> WorkerResult[None]:
        policy = calibration_policy(self.config)
        engine = str(self.config.benchmark.get("engine", "vllm"))
        for run in self.config.benchmark["runs"]:
            probes: list[ProbeResult] = []
            started = monotonic()
            for concurrency in policy.loads:
                remaining = policy.max_duration - (monotonic() - started)
                if remaining <= 0:
                    break
                probe_run = with_load(run, engine, concurrency, probe=True)
                print(f"Calibrating {run['name']}: concurrency={concurrency}", flush=True)
                probe = await measure_probe(
                    self.config,
                    probe_run,
                    context,
                    self.directory / str(run["name"]) / str(concurrency),
                    concurrency,
                    min(policy.probe_duration, remaining),
                    warmup=not probes,
                )
                probes.append(probe)
                # Persist partial evidence, including failures, before deciding whether to continue.
                self._save(run["name"], probes)
                if select_load(probes, policy).reason != "limit_reached":
                    break
            decision = select_load(probes, policy)
            reason = decision.reason
            if reason == "limit_reached" and monotonic() - started >= policy.max_duration:
                reason = "time_limit"
            self.decisions.append(
                {
                    "workload": run["name"],
                    "concurrency": decision.concurrency,
                    "reason": reason,
                    "probes": [asdict(probe) for probe in probes],
                }
            )
            self.frozen_runs.append(with_load(run, engine, decision.concurrency))
            print(f"Frozen {run['name']}: concurrency={decision.concurrency}, reason={reason}", flush=True)
        return WorkerResult.completed()

    def _save(self, name: str, probes: list[ProbeResult]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{name}-probes.json").write_text(json.dumps([asdict(item) for item in probes], indent=2))

    async def cleanup(self, context: TrialContext) -> None:
        """Server and benchmark workers own process cleanup."""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from vllm_optimizer.config.models import ExperimentConfig, VTuneConfig
from vllm_optimizer.domain.results import WorkerStatus
from vllm_optimizer.execution.slots import WorkerSlot
from vllm_optimizer.search.grid import TrialParameters
from vllm_optimizer.workers.base import TrialContext
from vllm_optimizer.workers.configuration import ConfigurationBuilderWorker
from vllm_optimizer.workers.factory import build_trial_workers
from vllm_optimizer.workers.process import ProcessRunner, ProcessSpec
from vllm_optimizer.workers.readiness import EndpointGuardWorker, ReadinessWorker


def _config(**server: object) -> VTuneConfig:
    return VTuneConfig(
        1,
        ExperimentConfig("readiness"),
        {"model": "demo", **server},
        tune={"host": {"values": ["127.0.0.2"]}, "port": {"values": [8123]}},
        benchmark={
            "repeats": 1,
            "min_repeats": 1,
            "runs": [{"name": "requests", "profile": {}, "constraints": [], "data": [{}]}],
        },
        optimization={"maximize": "requests_per_second"},
    )


def test_endpoint_guard_rejects_an_already_healthy_server() -> None:
    async def healthy(url: str, timeout: float) -> bool:
        return True

    worker = EndpointGuardWorker("127.0.0.1", 8000, health_probe=healthy)

    result = asyncio.run(worker.execute(TrialContext("trial")))

    assert result.status is WorkerStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == "server_endpoint_in_use"


def test_readiness_rejects_unrelated_server_that_becomes_healthy(tmp_path: Path) -> None:
    async def scenario():
        process = await ProcessRunner().start(
            ProcessSpec((sys.executable, "-c", "import time; time.sleep(30)")), tmp_path / "server.log"
        )
        probes = iter((False, True))
        ownership_checks = []

        async def health(url: str, timeout: float) -> bool:
            return next(probes)

        async def unrelated(pid: int, host: str, port: int) -> bool:
            ownership_checks.append((host, port))
            return False

        context = TrialContext("trial", {"server_process": process, "attempt_index": 1})
        worker = ReadinessWorker(
            startup_timeout=1, poll_interval=0.001, request_timeout=0.1, health_probe=health, ownership_probe=unrelated
        )
        try:
            return await worker.execute(context), ownership_checks
        finally:
            await process.stop(0.05)

    result, ownership_checks = asyncio.run(scenario())

    assert result.status is WorkerStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == "server_endpoint_in_use"
    assert ownership_checks == [("127.0.0.1", 8000)]


def test_tuned_endpoint_drives_launch_guard_and_readiness(tmp_path: Path) -> None:
    parameters = TrialParameters("trial", {"host": "127.0.0.2", "port": 8123}, {})
    workers = build_trial_workers(_config(), parameters, tmp_path)
    builder = next(worker for worker in workers if isinstance(worker, ConfigurationBuilderWorker))
    guard = next(worker for worker in workers if isinstance(worker, EndpointGuardWorker))
    readiness = next(worker for worker in workers if isinstance(worker, ReadinessWorker))
    context = TrialContext("trial")

    assert asyncio.run(builder.execute(context)).status is WorkerStatus.COMPLETED
    argv = context.values["process_spec"].argv
    assert argv[argv.index("--host") + 1] == "127.0.0.2"
    assert argv[argv.index("--port") + 1] == "8123"
    assert guard.endpoint == readiness.endpoint == "http://127.0.0.2:8123"


def test_fixed_host_and_parallel_port_share_the_resolved_endpoint(tmp_path: Path) -> None:
    slot = WorkerSlot("gpu", (0,), 9001)
    config = replace(_config(host="127.0.0.3"), execution={"host": "127.0.0.9"})
    workers = build_trial_workers(config, TrialParameters("trial", {}, {}), tmp_path, slot)
    builder = next(worker for worker in workers if isinstance(worker, ConfigurationBuilderWorker))
    readiness = next(worker for worker in workers if isinstance(worker, ReadinessWorker))
    context = TrialContext("trial")

    assert asyncio.run(builder.execute(context)).status is WorkerStatus.COMPLETED
    argv = context.values["process_spec"].argv
    assert argv[argv.index("--host") + 1] == "127.0.0.3"
    assert argv[argv.index("--port") + 1] == "9001"
    assert readiness.endpoint == "http://127.0.0.3:9001"

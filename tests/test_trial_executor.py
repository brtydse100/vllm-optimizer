import asyncio
import json
from pathlib import Path

import pytest

import vllm_optimizer.execution.trial_executor as module
from vllm_optimizer.config.loader import load_config
from vllm_optimizer.config.models import ExperimentConfig, VTuneConfig
from vllm_optimizer.domain.benchmark import BenchmarkResult, WorkloadResult
from vllm_optimizer.domain.results import Failure, WorkerResult
from vllm_optimizer.execution.slots import WorkerSlot
from vllm_optimizer.managers.scoring import ScoringManager
from vllm_optimizer.reproduction.export import export_vllm_command
from vllm_optimizer.reproduction.manifest import ManifestWriter
from vllm_optimizer.reproduction.models import CommandRecord
from vllm_optimizer.search.grid import TrialParameters
from vllm_optimizer.terminal import TerminalLogger
from vllm_optimizer.workers.configuration import build_process_spec


def _config(tmp_path: Path) -> VTuneConfig:
    return VTuneConfig(
        1,
        ExperimentConfig("executor", str(tmp_path)),
        {"model": "demo", "dtype": "float16"},
        env={"SAFE": "yes"},
        benchmark={
            "repeats": 1,
            "min_repeats": 1,
            "warmup_repeats": 0,
            "runs": [{"name": "requests", "profile": {}, "constraints": [], "data": [{}]}],
        },
        optimization={"maximize": "requests_per_second"},
    )


def test_trial_executor_persists_and_scores_synthetic_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Worker:
        name = "synthetic_benchmark:requests:repeat-1"

        async def execute(self, context):
            context.values["benchmark_results"] = (
                BenchmarkResult(
                    "requests",
                    "synthetic",
                    "1",
                    (
                        WorkloadResult(
                            0,
                            {},
                            {
                                "requests_per_second": {"average": 5.0},
                                "request_totals": {"successful": 1, "errored": 0, "incomplete": 0},
                            },
                        ),
                    ),
                    tmp_path / "raw.json",
                    1,
                    0.1,
                ),
            )
            return WorkerResult.completed()

        async def cleanup(self, context):
            return None

    monkeypatch.setattr(module, "build_trial_workers", lambda *args, **kwargs: (Worker(),))
    executor = module.TrialExecutor(
        _config(tmp_path),
        ScoringManager("requests_per_second", 1, ("requests",)),
        TerminalLogger("WARNING"),
        ManifestWriter({}),
        {},
    )

    report, score, by_benchmark = asyncio.run(
        executor.execute(
            tmp_path / "run",
            TrialParameters("trial-1", {"max-num-seqs": 2}, {}),
            WorkerSlot("gpu", (0,), 8100),
            "validation-001",
        )
    )

    assert report.status.value == "completed"
    assert report.execution["artifact_subdirectory"] == "validation-001"
    assert score and score.value == 5 and score.server_args["dtype"] == "float16"
    assert by_benchmark == {"requests": 5.0}
    artifact_directory = tmp_path / "run" / "trials" / "trial-1" / "validation-001"
    assert (artifact_directory / "result.json").exists()
    assert (artifact_directory / "manifest.json").exists()


def test_trial_executor_excludes_failed_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Worker:
        name = "failure"

        async def execute(self, context):
            return WorkerResult.failed(Failure("synthetic_failure", "failed"))

        async def cleanup(self, context):
            return None

    monkeypatch.setattr(module, "build_trial_workers", lambda *args, **kwargs: (Worker(),))
    executor = module.TrialExecutor(
        _config(tmp_path), ScoringManager("requests_per_second"), TerminalLogger("WARNING"), ManifestWriter({}), {}
    )
    report, score, by_benchmark = asyncio.run(
        executor.execute(tmp_path / "run", TrialParameters("trial-failed", {}, {}))
    )
    assert report.failure and score is None and by_benchmark == {}


def test_trial_executor_uses_external_config_captured_at_load_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    native = tmp_path / "vllm.yaml"
    native.write_text(f"model: {model}\nkv-transfer-config:\n  kv_role: original\n", encoding="utf-8")
    source = tmp_path / "experiment.yaml"
    source.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "experiment:",
                "  name: snapshot",
                "server:",
                "  config: vllm.yaml",
                "benchmark:",
                "  engine: vllm",
                "  repeats: 1",
                "  min_repeats: 1",
                "  runs:",
                "    - name: requests",
                "      args:",
                "        dataset_name: random",
                "optimization:",
                "  maximize: requests_per_second",
            )
        ),
        encoding="utf-8",
    )
    config = load_config(source)
    native.write_text(f"model: {model}\nkv-transfer-config:\n  kv_role: changed\n", encoding="utf-8")
    native.unlink()
    captured: dict[str, object] = {}

    def workers(worker_config, *args, **kwargs):
        captured.update(worker_config.server)
        process = build_process_spec(worker_config)

        class Worker:
            name = "capture"

            async def execute(self, context):
                context.commands.append(CommandRecord("vllm", process.argv, 1, process.env))
                return WorkerResult.completed()

            async def cleanup(self, context):
                return None

        return (Worker(),)

    monkeypatch.setattr(module, "build_trial_workers", workers)
    executor = module.TrialExecutor(
        config,
        ScoringManager("requests_per_second", 1, ("requests",)),
        TerminalLogger("WARNING"),
        ManifestWriter({}),
        {},
    )

    asyncio.run(executor.execute(tmp_path / "run", TrialParameters("trial-1", {}, {})))

    snapshot = Path(str(captured["config"]))
    assert snapshot.parent == tmp_path / "run" / "trials" / "trial-1"
    assert "kv_role: original" in snapshot.read_text(encoding="utf-8")
    manifest = json.loads((snapshot.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["external_config"]["settings"]["kv-transfer-config"] == {"kv_role": "original"}
    exported = export_vllm_command(tmp_path / "run", "trial-1")
    assert str(snapshot) in exported
    assert str(native) not in exported

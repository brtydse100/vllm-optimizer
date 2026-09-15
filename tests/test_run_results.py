import json
from pathlib import Path

from vllm_optimizer.domain.results import Failure, WorkerStatus
from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.run_results import RunResultsManager
from vllm_optimizer.managers.scoring import TrialScore


def test_run_results_persist_optional_metadata_and_summary_branches(tmp_path: Path) -> None:
    report = TrialReport(1, "failed-trial", WorkerStatus.FAILED, (), {}, failure=Failure("startup", "server failed"))
    score = TrialScore("failed-trial", 2.0, {"api-key": "secret"}, {"TOKEN": "secret"}, 3, 1, 2)
    baseline = TrialScore("baseline", 1.0, {}, {}, 4)
    validation = {"status": "pending", "selected_trials": ["failed-trial"], "repeats": 2}
    manager = RunResultsManager(tmp_path / "result.json", benchmark_policy={"repeats": 2})

    path = manager.save(
        "run-1",
        "requests_per_second",
        (report,),
        (score,),
        {"requests": (score,)},
        baseline,
        status="failed",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        source_run_id="source-1",
        sources={"failed-trial": {"result": "source.json"}},
        analysis_summary="analysis",
        run_failure={"code": "startup", "message": "server failed"},
        finalist_validation=validation,
    )

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["source_run_id"] == "source-1"
    assert document["analysis_summary"] == "analysis"
    assert document["run_failure"]["code"] == "startup"
    assert document["finalist_validation"] == validation
    assert document["selection_decision"]["status"] == "inconclusive"
    summary = manager.summary("requests_per_second", (report,), (score,), {"requests": (score,)}, baseline, validation)
    assert "Best observed tuned score" in summary
    assert "Observed score change vs baseline" in summary
    assert "failed-trial: startup: server failed" in summary
    assert "Ranked scores describe search observations" in summary

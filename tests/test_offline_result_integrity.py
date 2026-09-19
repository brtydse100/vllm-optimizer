import json
from pathlib import Path

import pytest

from tests.test_offline_commands import _run, _write
from vllm_optimizer.reporting.offline import regenerate_report
from vllm_optimizer.reporting.offline_loading import load_trial, read_object
from vllm_optimizer.reporting.reclassify import reclassify_run


def test_offline_loading_follows_accepted_validation_execution(tmp_path: Path) -> None:
    run = _run(tmp_path)
    result = read_object(run / "result.json", "run result")
    summary = result["trials"][0]
    assert isinstance(summary, dict)
    execution = {"mode": "sequential", "artifact_subdirectory": "validation-001"}
    summary["execution"] = execution
    result["ranking"][0]["score"] = 80.0
    _write(run / "result.json", result)

    original = read_object(run / "trials" / "trial-1" / "result.json", "trial result")
    original["benchmarks"][0]["workloads"][0]["metrics"]["requests_per_second"]["average"] = 150.0
    _write(run / "trials" / "trial-1" / "result.json", original)
    accepted = json.loads(json.dumps(original))
    accepted["execution"] = execution
    accepted["benchmarks"][0]["workloads"][0]["metrics"]["requests_per_second"]["average"] = 80.0
    manifest = read_object(run / "trials" / "trial-1" / "manifest.json", "manifest")
    manifest["execution"] = execution
    _write(run / "trials" / "trial-1" / "validation-001" / "result.json", accepted)
    _write(run / "trials" / "trial-1" / "validation-001" / "manifest.json", manifest)

    loaded = load_trial(run, summary, [])
    average = loaded.benchmarks[0]["workloads"][0]["metrics"]["requests_per_second"]["average"]
    assert average == 80.0
    assert regenerate_report(run, tmp_path / "validated-report").html.exists()


def test_reclassification_restores_an_initially_unranked_baseline(tmp_path: Path) -> None:
    run = _run(tmp_path)
    result = read_object(run / "result.json", "run result")
    result["trials"][0]["trial_id"] = "baseline"
    result["ranking"] = []
    result["baseline"] = None
    _write(run / "result.json", result)
    trial = read_object(run / "trials" / "trial-1" / "result.json", "trial result")
    trial["trial_id"] = "baseline"
    manifest = read_object(run / "trials" / "trial-1" / "manifest.json", "manifest")
    manifest["trial_id"] = "baseline"
    _write(run / "trials" / "baseline" / "result.json", trial)
    _write(run / "trials" / "baseline" / "manifest.json", manifest)

    reclassified = reclassify_run(run, 10, tmp_path / "baseline-reclassified")
    document = read_object(reclassified.result, "reclassified result")

    assert document["baseline"]["trial_id"] == "baseline"
    assert document["ranking"] == []


def test_offline_regeneration_rejects_non_finite_stored_scores(tmp_path: Path) -> None:
    run = _run(tmp_path)
    result = read_object(run / "result.json", "run result")
    result["ranking"][0]["score"] = float("nan")
    _write(run / "result.json", result)

    with pytest.raises(ValueError, match="finite"):
        regenerate_report(run, tmp_path / "non-finite")

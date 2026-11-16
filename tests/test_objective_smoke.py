"""Public objective checks using fabricated metrics; no captured benchmark fixtures."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from vllm_optimizer.benchmarks.metrics import normalize_guidellm_metrics, normalize_vllm_metrics
from vllm_optimizer.config.errors import ConfigValidationError
from vllm_optimizer.config.loader import load_config
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


def _load(tmp_path, engine, args, optimization):
    run = {"name": "custom", "args": args}
    document = {
        "experiment": {"name": "fabricated"},
        "server": {"model": "."},
        "tune": {"max-num-seqs": {"values": [8, 16]}},
        "benchmark": {"engine": engine, "runs": [run]},
        "optimization": optimization,
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_config(path)


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ({}, "50,95,99,55"),
        ({"metric_percentiles": "90,99"}, "90,99,55"),
        ({"--metric-percentiles=90": True}, "90,55"),
        ({"metric-percentiles": [50, 55]}, "50,55"),
    ],
)
def test_manual_space_requests_objective_percentile_once(tmp_path, spelling, expected):
    args = {"dataset-name": "random", "num-prompts": 8, "max-concurrency": 3, **spelling}
    objective = {"metric": "time_to_first_token_ms", "percentile": 55}
    config = _load(tmp_path, "vllm", args, {"objective": objective})
    resolved = config.benchmark["runs"][0]["args"]
    assert resolved["max-concurrency"] == 3 and resolved["num-prompts"] == 8
    assert resolved["metric-percentiles"] == expected
    assert resolved["percentile-metrics"] == "ttft,tpot,itl,e2el"
    assert not set(spelling) - {"metric-percentiles"} & set(resolved)


def test_legacy_maximize_arguments_are_not_modified(tmp_path):
    args = {"dataset-name": "random", "num-prompts": 8}
    config = _load(tmp_path, "vllm", args, {"maximize": "output_tokens_per_second"})
    assert config.benchmark["runs"][0]["args"] == args


@pytest.mark.parametrize(
    ("engine", "optimization", "message"),
    [
        ("guidellm", {"objective": {"metric": "end_to_end_latency_ms", "percentile": 55}}, "GuideLLM.*55"),
        ("vllm", {"maximize": "end_to_end_latency_ms.p99"}, "use optimization.objective"),
        ("vllm", {"maximize": "x", "objective": {"metric": "end_to_end_latency_ms"}}, "not both"),
        ("vllm", {"objective": {"metric": "output_tokens_per_second", "percentile": 99}}, "percentiles"),
        ("vllm", {"objective": {"metric": "unknown"}}, "canonical"),
    ],
)
def test_unsupported_objectives_fail_validation(tmp_path, engine, optimization, message):
    args = {"dataset-name": "random", "num-prompts": 8} if engine == "vllm" else {}
    with pytest.raises(ConfigValidationError, match=message):
        _load(tmp_path, engine, args, optimization)

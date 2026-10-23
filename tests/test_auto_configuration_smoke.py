"""Public smoke tests using fabricated settings, without model weights or GPU data."""

import json

import pytest
import yaml

from vllm_optimizer.auto.calibration import ProbeResult, calibration_policy, select_load
from vllm_optimizer.auto.probes import with_load
from vllm_optimizer.config.errors import ConfigValidationError
from vllm_optimizer.config.loader import load_config


def configuration(tmp_path, **sections):
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 128}))
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({"server": {"model": "."}, "optimization": {"mode": "auto"}, **sections}))
    return load_config(path)


@pytest.mark.parametrize("engine", ["vllm", "guidellm"])
def test_minimal_auto_generates_bounded_workloads_and_space(tmp_path, engine):
    config = configuration(tmp_path, benchmark={"engine": engine})
    assert config.optimization["trials"] == 20 and config.optimization["sampler"] == "tpe"
    assert config.auto["generated_benchmark"]
    assert [run["name"] for run in config.benchmark["runs"]] == ["small", "medium", "large"]
    assert set(config.tune) == {"max-num-seqs", "max-num-batched-tokens", "gpu-memory-utilization"}
    for run in config.benchmark["runs"]:
        if engine == "vllm":
            args = run["args"]
            assert args["random-input-len"] + args["random-output-len"] <= 128
            assert "e2el" in args["percentile-metrics"]
        else:
            dataset = run["data"][0]
            assert dataset["prompt_tokens"] + dataset["output_tokens"] <= 128


def test_native_and_inline_fixed_settings_are_excluded(tmp_path):
    (tmp_path / "native.yaml").write_text("model: .\nmax_num_seqs: 7\ngpu-memory-utilization: 0.5\n")
    config = configuration(tmp_path, server={"config": "native.yaml", "max_num_batched_tokens": 256})
    assert config.tune == {}


def test_custom_data_and_load_are_preserved_while_requesting_p55(tmp_path):
    run = {"name": "custom", "args": {"dataset-name": "random", "num-prompts": 8, "max-concurrency": 3}}
    config = configuration(
        tmp_path,
        benchmark={"engine": "vllm", "runs": [run]},
        optimization={"mode": "auto", "objective": {"metric": "time_to_first_token_ms", "percentile": 55}},
    )
    assert not config.auto["generated_benchmark"]
    args = config.benchmark["runs"][0]["args"]
    assert args["max-concurrency"] == 3 and args["num-prompts"] == 8
    assert "55" in args["metric-percentiles"]


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "invalid"},
        {"mode": "auto", "calibration": {"max_concurrency": 0}},
        {"mode": "auto", "calibration": {"latency_growth": 1}},
        {"mode": "auto", "calibration": {"throughput_gain": 2}},
    ],
)
def test_invalid_controls_fail_before_execution(tmp_path, settings):
    with pytest.raises(ConfigValidationError):
        configuration(tmp_path, optimization=settings)


def test_guidellm_unavailable_percentile_fails_explicitly(tmp_path):
    with pytest.raises(ConfigValidationError, match="GuideLLM.*55"):
        configuration(
            tmp_path,
            benchmark={"engine": "guidellm"},
            optimization={"mode": "auto", "objective": {"metric": "end_to_end_latency_ms", "percentile": 55}},
        )


def test_calibration_backs_off_at_plateau_and_failures():
    initial = ProbeResult(1, 10, 1)
    stable = ProbeResult(2, 19, 1.1)
    decision = select_load([initial, stable, ProbeResult(4, 19.1, 1.2)])
    assert (decision.concurrency, decision.reason) == (2, "throughput_plateau")
    assert select_load([initial, ProbeResult(2, 20, 3)]).reason == "latency_growth"
    assert select_load([initial, ProbeResult(2, 0, 0, "probe_timeout")]).concurrency == 1
    assert select_load([initial, stable]).reason == "limit_reached"
    with pytest.raises(ValueError, match="usable"):
        select_load([ProbeResult(1, 0, 0, "failure")])
    with pytest.raises(ValueError, match="usable"):
        select_load([])


def test_non_power_of_two_load_cap_and_frozen_workload(tmp_path):
    config = configuration(tmp_path, optimization={"mode": "auto", "calibration": {"max_concurrency": 6}})
    assert calibration_policy(config).loads == (1, 2, 4, 6)
    run = config.benchmark["runs"][0]
    frozen = with_load(run, "vllm", 4)
    assert frozen["args"]["max-concurrency"] == 4
    assert run["args"]["max-concurrency"] == 1
    assert frozen["args"]["random-input-len"] == run["args"]["random-input-len"]

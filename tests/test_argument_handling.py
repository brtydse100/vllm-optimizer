from pathlib import Path

import pytest

from vllm_optimizer.benchmarks.vllm import build_plan
from vllm_optimizer.config.models import ExperimentConfig, VTuneConfig
from vllm_optimizer.workers.configuration import build_process_spec


def _config(server: dict[str, object], tune: dict[str, object] | None = None) -> VTuneConfig:
    return VTuneConfig(1, ExperimentConfig("arguments"), {"model": "/models/demo", **server}, tune=tune or {})


@pytest.mark.parametrize(
    ("fixed_name", "tuned_name"), (("max_num_seqs", "max-num-seqs"), ("max-num-seqs", "max_num_seqs"))
)
def test_tuned_alias_overrides_fixed_value_once(fixed_name: str, tuned_name: str) -> None:
    config = _config({fixed_name: 1}, {tuned_name: {"values": [8]}})

    argv = build_process_spec(config, {tuned_name: 8}).argv

    assert argv.count("--max-num-seqs") == 1
    assert argv[argv.index("--max-num-seqs") + 1] == "8"


def test_runtime_alias_has_highest_precedence() -> None:
    config = _config({"max_num_seqs": 1}, {"max-num-seqs": {"values": [8]}})

    argv = build_process_spec(config, {"max-num-seqs": 8}, runtime_args={"max_num_seqs": 16}).argv

    assert argv.count("--max-num-seqs") == 1
    assert argv[argv.index("--max-num-seqs") + 1] == "16"


def test_duplicate_aliases_in_one_layer_are_rejected() -> None:
    config = _config({"max-num-seqs": 1, "max_num_seqs": 2})

    with pytest.raises(ValueError, match="duplicate aliases.*max-num-seqs"):
        build_process_spec(config)


def test_environment_variable_names_do_not_use_argument_aliases() -> None:
    config = VTuneConfig(
        1, ExperimentConfig("arguments"), {"model": "/models/demo"}, tune_env={"TOKEN_NAME": {"values": ["x"]}}
    )

    with pytest.raises(ValueError, match="Unknown tunable environment.*TOKEN-NAME"):
        build_process_spec(config, selected_env={"TOKEN-NAME": "x"})


def test_server_list_argument_keeps_all_values() -> None:
    argv = build_process_spec(_config({"lora-modules": ["a=/a", "b=/b"]})).argv
    index = argv.index("--lora-modules")

    assert argv.count("--lora-modules") == 1
    assert argv[index : index + 3] == ("--lora-modules", "a=/a", "b=/b")


def test_vllm_benchmark_list_argument_keeps_all_values(tmp_path: Path) -> None:
    run = {"name": "requests", "args": {"served-model-name": ["first", "second"]}}

    argv = build_plan(_config({}), run, "http://127.0.0.1:8000", tmp_path).argv
    index = argv.index("--served-model-name")

    assert argv.count("--served-model-name") == 1
    assert argv[index : index + 3] == ("--served-model-name", "first", "second")

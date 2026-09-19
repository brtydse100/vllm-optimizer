import pytest

from vllm_optimizer.config.errors import ConfigValidationError
from vllm_optimizer.config.models import ExperimentConfig, VTuneConfig
from vllm_optimizer.config.preflight import validate_config


def _parallel_search_config(
    tensor_parallel_sizes: list[int],
    worker_devices: list[int],
    name: str = "tensor-parallel-size",
    sampler: str = "tpe",
) -> VTuneConfig:
    return VTuneConfig(
        1,
        ExperimentConfig("search-validation", seed=7),
        {"model": "demo"},
        tune={name: {"values": tensor_parallel_sizes}},
        benchmark={"engine": "vllm", "runs": [{"name": "requests", "args": {}}]},
        optimization={"maximize": "score", "sampler": sampler, "trials": 1},
        execution={
            "mode": "local_parallel",
            "max_parallel_trials": 1,
            "ports": {"min": 8100, "max": 8100},
            "gpu_allocation": {
                "strategy": "explicit",
                "allow_sharing": False,
                "workers": [{"name": "worker", "devices": worker_devices}],
            },
        },
    )


@pytest.mark.parametrize("name", ["tensor-parallel-size", "tensor_parallel_size", "--tensor-parallel-size"])
@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_search_preflight_rejects_any_tensor_parallel_size_without_a_worker(name: str, sampler: str) -> None:
    config = _parallel_search_config([1, 2], [0], name, sampler)

    with pytest.raises(ConfigValidationError, match="tensor-parallel-size exceeds every parallel worker"):
        validate_config(config)


def test_tpe_preflight_accepts_tensor_parallel_sizes_supported_by_a_worker() -> None:
    validate_config(_parallel_search_config([1, 2], [0, 1]))


def test_search_preflight_rejects_duplicate_tensor_parallel_aliases() -> None:
    config = _parallel_search_config([1], [0])
    tune = dict(config.tune)
    tune["--tensor_parallel_size"] = {"values": [1]}
    config = VTuneConfig(
        config.schema_version,
        config.experiment,
        config.server,
        tune=tune,
        benchmark=config.benchmark,
        optimization=config.optimization,
        execution=config.execution,
    )

    with pytest.raises(ConfigValidationError, match="duplicate aliases.*tensor-parallel-size"):
        validate_config(config)

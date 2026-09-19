import pytest

from vllm_optimizer.config.errors import ConfigValidationError
from vllm_optimizer.config.models import ExperimentConfig, VTuneConfig
from vllm_optimizer.config.preflight import validate_config


def _parallel_tpe_config(tensor_parallel_sizes: list[int], worker_devices: list[int]) -> VTuneConfig:
    return VTuneConfig(
        1,
        ExperimentConfig("search-validation", seed=7),
        {"model": "demo"},
        tune={"tensor-parallel-size": {"values": tensor_parallel_sizes}},
        benchmark={"engine": "vllm", "runs": [{"name": "requests", "args": {}}]},
        optimization={"maximize": "score", "sampler": "tpe", "trials": 1},
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


def test_tpe_preflight_rejects_any_tensor_parallel_size_without_a_worker() -> None:
    config = _parallel_tpe_config([1, 2], [0])

    with pytest.raises(ConfigValidationError, match="tensor-parallel-size exceeds every parallel worker"):
        validate_config(config)


def test_tpe_preflight_accepts_tensor_parallel_sizes_supported_by_a_worker() -> None:
    validate_config(_parallel_tpe_config([1, 2], [0, 1]))

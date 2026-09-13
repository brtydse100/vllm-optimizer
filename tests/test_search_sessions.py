from pathlib import Path

from vllm_optimizer.config.models import ExperimentConfig, VTuneConfig
from vllm_optimizer.search.factory import create_search, validate_search


def _config(tmp_path: Path, values: list[int], trials: int) -> VTuneConfig:
    return VTuneConfig(
        1,
        ExperimentConfig("search", str(tmp_path), 7),
        {"model": "demo"},
        tune={"max-num-seqs": {"values": values}},
        optimization={"maximize": "score", "sampler": "tpe", "trials": trials},
    )


def test_tpe_stops_when_duplicate_search_space_is_exhausted(tmp_path: Path) -> None:
    config = _config(tmp_path, [8, 8], 2)
    session = create_search(config, tmp_path)

    assert validate_search(config) == ("tpe", 1)
    assert session.total == 1
    trial = session.suggest()
    assert trial is not None and trial.server_args == {"max-num-seqs": 8}
    session.complete(trial, 1.0)
    assert session.suggest() is None


def test_tpe_only_materializes_requested_trials(tmp_path: Path) -> None:
    config = VTuneConfig(
        1,
        ExperimentConfig("large", str(tmp_path), 7),
        {"model": "demo"},
        tune={f"choice-{index}": {"values": list(range(10))} for index in range(5)},
        optimization={"maximize": "score", "sampler": "tpe", "trials": 2},
    )

    session = create_search(config, tmp_path)
    first = session.suggest()
    second = session.suggest()

    assert session.total == 2
    assert first is not None and second is not None
    assert first.server_args != second.server_args

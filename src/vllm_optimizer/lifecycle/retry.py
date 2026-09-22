"""Reconstruct selected trials from an immutable source run."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from vllm_optimizer.config.models import ExperimentConfig, VTuneConfig
from vllm_optimizer.lifecycle.integrity import load_retry_source
from vllm_optimizer.reproduction.redaction import REDACTED
from vllm_optimizer.search.grid import TrialParameters

_TRIAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class RetryPlan:
    config: VTuneConfig
    trials: tuple[TrialParameters, ...]
    source_run_id: str
    sources: Mapping[str, Mapping[str, str]]
    warnings: tuple[str, ...] = ()


def load_retry_plan(run: Path, trial_ids: list[str]) -> RetryPlan:
    source = Path(run).resolve()
    if (
        not trial_ids
        or len(set(trial_ids)) != len(trial_ids)
        or any(not _TRIAL_ID.fullmatch(trial) for trial in trial_ids)
    ):
        raise ValueError("retry requires one or more unique --trial values")
    result, manifests, warnings = load_retry_source(source, trial_ids)
    model = _same(manifests, "model_path")
    benchmark = _same(manifests, "benchmark")
    selected = [_parameters(manifest) for manifest in manifests]
    fixed_args, external_server_config = _fixed_arguments(manifests, selected)
    fixed_env = _restore_env(_mapping(_same(selected, "fixed_env"), "fixed_env"))
    tune = _definitions(selected, "selected_args")
    tune_env = _definitions(selected, "selected_env")
    model_path = Path(_text(model, "model_path")).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f"source model path is not a directory: {model_path}")
    config = VTuneConfig(
        1,
        ExperimentConfig(source.parent.name, str(source.parent.parent)),
        {"model": str(model_path), **fixed_args},
        tune,
        fixed_env,
        tune_env,
        benchmark=_mapping(benchmark, "benchmark"),
        baseline={"enabled": False},
        optimization={"maximize": _text(result.get("maximize"), "maximize")},
        timeouts=_policy(manifests[0], "timeouts"),
        execution=_policy(manifests[0], "execution"),
        external_server_config=external_server_config,
    )
    trials = tuple(
        TrialParameters(
            trial_id,
            _mapping(values.get("selected_args"), "selected_args"),
            _restore_env(_mapping(values.get("selected_env"), "selected_env")),
        )
        for trial_id, values in zip(trial_ids, selected, strict=True)
    )
    run_id = _text(result.get("run_id"), "run_id")
    sources = {trial: {"run_id": run_id, "trial_id": trial} for trial in trial_ids}
    return RetryPlan(config, trials, run_id, sources, warnings)


def _parameters(manifest: Mapping[str, object]) -> dict[str, object]:
    return _mapping(manifest.get("parameters"), "parameters")


def _fixed_arguments(
    manifests: list[dict[str, object]], selected: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, object] | None]:
    arguments = [_mapping(item.get("fixed_args"), "fixed_args") for item in selected]
    external = [manifest.get("external_config") for manifest in manifests]
    if not any(item is not None for item in external):
        return _mapping(_same(selected, "fixed_args"), "fixed_args"), None
    if any(not isinstance(item, dict) for item in external):
        raise ValueError("selected trials have incompatible external_config")
    snapshots = [_mapping(item, "external_config") for item in external]
    settings = _mapping(_same(snapshots, "settings"), "external_config.settings")
    if redacted_path := _redacted_path(settings, "external_config.settings"):
        raise ValueError(
            f"retry cannot restore redacted external YAML setting '{redacted_path}'; "
            "start a new run with the original secret"
        )
    paths = [_text(item.pop("config", None), "fixed_args.config") for item in arguments]
    for snapshot, path in zip(snapshots, paths, strict=True):
        if _text(snapshot.get("path"), "external_config.path") != path:
            raise ValueError("external_config path does not match fixed_args.config")
    if any(item != arguments[0] for item in arguments[1:]):
        raise ValueError("selected trials have incompatible fixed_args")
    return {**arguments[0], "config": paths[0]}, settings


def _redacted_path(value: object, path: str, headers: bool = False) -> str | None:
    if value == REDACTED or headers and isinstance(value, str) and REDACTED in value:
        return path
    if isinstance(value, Mapping):
        for name, item in value.items():
            header_value = headers or str(name).lower().replace("_", "-") in {"header", "headers"}
            if found := _redacted_path(item, f"{path}.{name}", header_value):
                return found
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            if found := _redacted_path(item, f"{path}.{index}", headers):
                return found
    return None


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _same(documents: list[dict[str, object]], key: str) -> object:
    value = documents[0].get(key)
    if any(document.get(key) != value for document in documents[1:]):
        raise ValueError(f"selected trials have incompatible {key}")
    return value


def _restore_env(values: Mapping[str, object]) -> dict[str, str]:
    restored = {}
    for name, value in values.items():
        if value == REDACTED:
            if name not in os.environ:
                raise ValueError(f"retry requires environment variable '{name}'")
            value = os.environ[name]
        restored[name] = str(value)
    return restored


def _definitions(items: list[dict[str, object]], key: str) -> dict[str, object]:
    names = {name for item in items for name in _mapping(item.get(key), key)}
    return {
        name: {
            "values": [item.get(name) for item in (_mapping(values.get(key), key) for values in items) if name in item]
        }
        for name in sorted(names)
    }


def _policy(manifest: Mapping[str, object], name: str) -> dict[str, object]:
    policy = manifest.get("policy", {})
    return _mapping(_mapping(policy, "policy").get(name, {}), name)

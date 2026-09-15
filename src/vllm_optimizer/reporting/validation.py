"""Validation helpers for stored report data."""

from __future__ import annotations

import re
from collections.abc import Mapping
from math import isfinite

_ARTIFACT_SUBDIRECTORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def execution(value: object, trial_id: str) -> Mapping[str, object]:
    """Validate an optional resolved trial execution assignment."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"trial {trial_id} has invalid execution")
    allowed = {"mode", "worker", "devices", "port", "artifact_subdirectory", "adaptive_repeats"}
    require(set(value) <= allowed, f"trial {trial_id} has unknown execution fields")
    mode = value.get("mode")
    require(mode in {"sequential", "local_parallel"}, f"trial {trial_id} has invalid execution mode")
    result: dict[str, object] = {"mode": mode}
    adaptive = value.get("adaptive_repeats")
    if adaptive is not None:
        result["adaptive_repeats"] = _adaptive_evidence(adaptive, trial_id)
    artifact_subdirectory = value.get("artifact_subdirectory")
    if artifact_subdirectory is not None:
        require(
            isinstance(artifact_subdirectory, str) and bool(_ARTIFACT_SUBDIRECTORY.fullmatch(artifact_subdirectory)),
            f"trial {trial_id} has invalid execution artifact subdirectory",
        )
        result["artifact_subdirectory"] = artifact_subdirectory
    if mode == "sequential":
        require(
            set(value) <= {"mode", "artifact_subdirectory", "adaptive_repeats"},
            f"trial {trial_id} has invalid sequential execution",
        )
        return result
    worker, devices, port = value.get("worker"), value.get("devices"), value.get("port")
    require(isinstance(worker, str) and worker.strip(), f"trial {trial_id} has invalid execution worker")
    require(isinstance(devices, list) and devices, f"trial {trial_id} has invalid execution devices")
    if isinstance(devices, list):
        devices = value["devices"]
        require(
            all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in devices)
            and len(set(devices)) == len(devices),
            f"trial {trial_id} has invalid execution devices",
        )
    require(
        isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535,
        f"trial {trial_id} has invalid execution port",
    )
    result.update({"worker": worker, "devices": devices, "port": port})
    return result


def _adaptive_evidence(value: object, trial_id: str) -> dict[str, object]:
    require(isinstance(value, Mapping), f"trial {trial_id} has invalid adaptive repeat evidence")
    evidence = dict(value) if isinstance(value, Mapping) else {}
    allowed = {
        "status",
        "reason",
        "decision_repeats",
        "planned_repeats",
        "initial_score",
        "reference_score",
        "minimum_relative_score",
        "actual_repeats_by_run",
    }
    require(set(evidence) <= allowed, f"trial {trial_id} has unknown adaptive repeat evidence")
    require(
        evidence.get("status")
        in {"not_reached", "stopped", "continued", "continued_without_reference", "continued_without_comparison"},
        f"trial {trial_id} has invalid adaptive repeat status",
    )
    require(isinstance(evidence.get("reason"), str), f"trial {trial_id} has invalid adaptive repeat reason")
    for name in ("decision_repeats", "planned_repeats"):
        item = evidence.get(name)
        require(
            isinstance(item, int) and not isinstance(item, bool) and item >= 1,
            f"trial {trial_id} has invalid adaptive repeat {name}",
        )
    for name in ("initial_score", "reference_score"):
        item = evidence.get(name)
        require(
            item is None or isinstance(item, int | float) and not isinstance(item, bool) and isfinite(item),
            f"trial {trial_id} has invalid adaptive repeat {name}",
        )
    ratio = evidence.get("minimum_relative_score")
    require(
        isinstance(ratio, int | float) and not isinstance(ratio, bool) and isfinite(ratio) and 0 < ratio <= 1,
        f"trial {trial_id} has invalid adaptive repeat threshold",
    )
    counts = evidence.get("actual_repeats_by_run", {})
    require(
        isinstance(counts, Mapping)
        and all(
            isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool) and count >= 0
            for name, count in counts.items()
        ),
        f"trial {trial_id} has invalid adaptive repeat counts",
    )
    evidence["actual_repeats_by_run"] = dict(counts) if isinstance(counts, Mapping) else {}
    return evidence


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def object_list(document: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    value = document.get(name)
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"run result has invalid {name}")
    return [item for item in value if isinstance(item, Mapping)]


def mapping(document: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = document.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"stored {name} must be an object")
    return value


def text(document: Mapping[str, object], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"stored {name} must be text")
    return value


def optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None

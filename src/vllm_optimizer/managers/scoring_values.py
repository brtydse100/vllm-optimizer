"""Shared finite metric extraction, request eligibility, and workload identity."""

import json
from collections.abc import Mapping
from math import isfinite


def _metric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _finite(value)
    if not isinstance(value, Mapping):
        return None
    average = value.get("average")
    if isinstance(average, int | float) and not isinstance(average, bool):
        return _finite(average)
    successful = value.get("successful")
    if isinstance(successful, Mapping):
        mean = successful.get("mean")
        if isinstance(mean, int | float) and not isinstance(mean, bool):
            return _finite(mean)
    mean = value.get("mean")
    return _finite(mean)


def _finite(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value) else None


def _workload_identity(index: int, configuration: Mapping[str, object]) -> str:
    return f"{index}:{json.dumps(_plain(configuration), sort_keys=True, separators=(',', ':'))}"


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def _request_counts(metrics: Mapping[str, object]) -> tuple[int, int, int]:
    totals = metrics.get("request_totals")
    if not isinstance(totals, Mapping):
        return 0, 0, 0
    return tuple(_count(totals.get(name)) for name in ("successful", "errored", "incomplete"))  # type: ignore[return-value]


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _eligible(metrics: Mapping[str, object], max_failure_percentage: float = 0) -> bool:
    successful, errored, incomplete = _request_counts(metrics)
    total = successful + errored + incomplete
    return successful > 0 and total > 0 and 100 * (errored + incomplete) / total <= max_failure_percentage

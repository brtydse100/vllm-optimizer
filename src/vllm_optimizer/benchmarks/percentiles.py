"""Preserve backend statistics and expose explicit scalar statistic keys."""

from collections.abc import Mapping
from math import isfinite


def percentile_values(source: Mapping[str, object], scale: float = 1) -> dict[str, float]:
    nested = source.get("percentiles")
    values = {**(nested if isinstance(nested, Mapping) else {}), **source}
    return {
        {"p01": "p1", "p05": "p5"}.get(name, name): float(value) * scale
        for name, value in values.items()
        if isinstance(name, str)
        and name.startswith("p")
        and name[1:].isdigit()
        and isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
    }


def scalar_statistics(metrics: dict[str, object]) -> dict[str, object]:
    for name, summary in tuple(metrics.items()):
        if not name.endswith("_ms") or not isinstance(summary, Mapping):
            continue
        for statistic, value in summary.items():
            if statistic in {"average", "median"} or statistic.startswith("p"):
                key = "p50" if statistic == "median" else statistic
                metrics[f"{name}.{key}"] = value
    return metrics

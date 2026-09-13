"""Keep scenario identity, repeat order and missing measurements explicit."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from statistics import median

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.reporting.analysis import workload_metric_summary
from vllm_optimizer.reproduction.redaction import redact


@dataclass(frozen=True)
class Observation:
    repeat: str
    metrics: Mapping[str, object]


def scenarios(report: TrialReport | None) -> dict[str, list[Observation]]:
    grouped: dict[str, list[Observation]] = {}
    if report is None:
        return grouped
    for benchmark in report.benchmarks:
        workloads = benchmark.get("workloads", ())
        if not isinstance(workloads, tuple | list):
            continue
        for workload in workloads:
            if not isinstance(workload, Mapping):
                continue
            configuration = json.dumps(redact(workload.get("configuration", {})), sort_keys=True)
            key = f"{benchmark.get('name', 'Unavailable')} / workload {workload.get('index', '?')} / {configuration}"
            metrics = workload.get("metrics", {})
            if isinstance(metrics, Mapping):
                grouped.setdefault(key, []).append(Observation(str(benchmark.get("repeat", "Unavailable")), metrics))
    return grouped


def number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value):
        return float(value)
    return None


def samples(observations: list[Observation], aliases: tuple[str, ...], statistic: str = "average") -> list[float]:
    return [
        value
        for observation in observations
        if (value := number(workload_metric_summary(observation.metrics, aliases).get(statistic))) is not None
    ]


def center(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def formatted(value: float | None, unit: str = "") -> str:
    return f"{value:,.4g}{(' ' + unit) if unit else ''}" if value is not None else "Unavailable"


def delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline == 0:
        return "Unavailable"
    return f"{(value - baseline) / abs(baseline) * 100:+.2f}%"


def failure_samples(observations: list[Observation]) -> list[float]:
    result = []
    for observation in observations:
        totals = observation.metrics.get("request_totals")
        if not isinstance(totals, Mapping):
            continue
        errored, incomplete = number(totals.get("errored")), number(totals.get("incomplete"))
        if errored is not None and incomplete is not None:
            result.append(errored + incomplete)
    return result

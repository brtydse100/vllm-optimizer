"""Keep scenario identity, repeat order and missing measurements explicit."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import fmean

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.reporting.analysis import workload_metric_summary
from vllm_optimizer.reproduction.redaction import redact


@dataclass(frozen=True)
class Observation:
    repeat: str
    metrics: Mapping[str, object]


class ScenarioKey(str):
    configuration: str

    def __new__(cls, benchmark: str, index: str, configuration: str) -> ScenarioKey:
        summary = configuration_summary(json.loads(configuration))
        suffix = f" — {summary}" if summary else ""
        instance = super().__new__(cls, f"{benchmark} / workload {index}{suffix}")
        instance.configuration = configuration
        return instance

    def pretty_configuration(self) -> str:
        return json.dumps(json.loads(self.configuration), indent=2, sort_keys=True)


def scenarios(report: TrialReport | None) -> dict[ScenarioKey, list[Observation]]:
    grouped: dict[ScenarioKey, list[Observation]] = {}
    if report is None:
        return grouped
    for benchmark in report.benchmarks:
        workloads = benchmark.get("workloads", ())
        if not isinstance(workloads, tuple | list):
            continue
        for workload in workloads:
            if not isinstance(workload, Mapping):
                continue
            raw_configuration = redact(workload.get("configuration", {}))
            configuration = json.dumps(raw_configuration, sort_keys=True)
            key = ScenarioKey(str(benchmark.get("name", "Unavailable")), str(workload.get("index", "?")), configuration)
            metrics = workload.get("metrics", {})
            if isinstance(metrics, Mapping):
                grouped.setdefault(key, []).append(Observation(str(benchmark.get("repeat", "Unavailable")), metrics))
    return grouped


def configuration_summary(configuration: object) -> str:
    if not isinstance(configuration, Mapping):
        return ""
    aliases = (
        (("dataset", "dataset-name"), "dataset"),
        (("prompt_tokens", "random-input-len", "input_len"), "prompt"),
        (("output_tokens", "random-output-len", "output_len"), "output"),
        (("num_prompts", "num-prompts"), "requests"),
        (("max_concurrency", "max-concurrency", "concurrency"), "concurrency"),
        (("request_rate", "request-rate"), "request rate"),
        (("burstiness",), "burstiness"),
    )
    parts = []
    for names, label in aliases:
        value = next((configuration[name] for name in names if name in configuration), None)
        if value is not None:
            parts.append(f"{label}: {_readable_value(value)}")
    return " · ".join(parts)


def _readable_value(value: object) -> str:
    if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
        return "unlimited"
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return Path(value).name or value
    return str(value)


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
    return fmean(values) if values else None


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

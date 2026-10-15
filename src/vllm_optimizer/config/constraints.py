"""Typed, unit-aware performance constraints shared by runtime and offline tools."""

from collections.abc import Mapping
from dataclasses import dataclass

from vllm_optimizer.config.constraint_units import METRIC_UNITS, parse_bound
from vllm_optimizer.config.models import VTuneConfig


@dataclass(frozen=True, slots=True)
class Constraint:
    metric: str
    statistic: str
    operator: str
    bound: float
    unit: str
    quantity: str | float

    def to_dict(self) -> dict[str, object]:
        return {"metric": self.metric, "statistic": self.statistic, self.operator: self.quantity}


def configured_constraints(config: VTuneConfig) -> tuple[Constraint, ...]:
    return parse_constraints(config.optimization.get("constraints", ()))


def parse_constraints(raw: object) -> tuple[Constraint, ...]:
    if not isinstance(raw, list | tuple):
        raise ValueError("optimization.constraints must be a list")
    return tuple(_constraint(item) for item in raw)


def stored_constraints(
    document: Mapping[str, object], manifests: tuple[Mapping[str, object], ...]
) -> tuple[Constraint, ...]:
    constraints = parse_constraints(document.get("constraints", ()))
    for manifest in manifests:
        if parse_constraints(manifest.get("constraints", ())) != constraints:
            raise ValueError("stored performance constraints disagree with accepted manifest")
    return constraints


def _constraint(raw: object) -> Constraint:
    if not isinstance(raw, Mapping) or set(raw) - {"metric", "statistic", "min", "max"}:
        raise ValueError("each constraint supports only metric, statistic, and one min/max bound")
    metric, statistic = raw.get("metric"), raw.get("statistic", "average")
    if not isinstance(metric, str) or metric not in METRIC_UNITS:
        raise ValueError("constraint metric must be a canonical latency or throughput metric")
    unit = METRIC_UNITS[metric]
    if not isinstance(statistic, str) or statistic not in {"average", "p50", "p95", "p99"}:
        raise ValueError("constraint statistic must be average, p50, p95, or p99")
    if unit != "ms" and statistic != "average":
        raise ValueError("throughput constraints support only the average statistic")
    if ("min" in raw) == ("max" in raw):
        raise ValueError("each constraint requires exactly one min or max bound")
    operator = "min" if "min" in raw else "max"
    value = raw[operator]
    bound = parse_bound(value, unit)
    return Constraint(metric, statistic, operator, bound, unit, value if isinstance(value, str) else bound)

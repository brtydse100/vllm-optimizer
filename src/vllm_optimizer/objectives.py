"""One objective policy for configuration, search, and stored measurements."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite

from vllm_optimizer.config.constraint_units import METRIC_UNITS
from vllm_optimizer.config.models import VTuneConfig


@dataclass(frozen=True, slots=True)
class Objective:
    metric: str
    percentile: int | None = None
    direction: str = "maximize"

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("objective metric must be a non-empty string")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("objective direction must be maximize or minimize")
        if self.percentile is not None and (
            isinstance(self.percentile, bool) or not isinstance(self.percentile, int) or not 1 <= self.percentile <= 100
        ):
            raise ValueError("objective percentile must be an integer from 1 to 100")

    @property
    def key(self) -> str:
        if self.percentile is not None:
            return f"{self.metric}.p{self.percentile}"
        return f"{self.metric}.average" if self.direction == "minimize" else self.metric

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def loss(self, value: float) -> float:
        """A comparison key, never a replacement for the measured score."""
        return value if self.direction == "minimize" else -value


def objective_from_config(config: VTuneConfig) -> Objective:
    values = config.optimization
    if "objective" not in values:
        metric = values.get("maximize")
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("optimization requires a non-empty 'maximize' metric or 'objective'")
        return Objective(metric)
    if "maximize" in values:
        raise ValueError("choose optimization.objective or optimization.maximize, not both")
    raw = values["objective"]
    if not isinstance(raw, Mapping) or set(raw) - {"metric", "percentile"}:
        raise ValueError("optimization.objective supports only metric and percentile")
    metric = raw.get("metric")
    if not isinstance(metric, str) or metric not in METRIC_UNITS:
        raise ValueError("objective metric must be a canonical latency or throughput metric")
    latency = METRIC_UNITS[metric] == "ms"
    if "percentile" in raw and not latency:
        raise ValueError("throughput objectives do not support percentiles")
    return Objective(metric, raw.get("percentile"), "minimize" if latency else "maximize")


def objective_from_key(key: str) -> Objective:
    """Resolve derived statistic keys; legacy unsuffixed metrics still maximize."""
    metric, _, statistic = key.rpartition(".")
    if METRIC_UNITS.get(metric) == "ms":
        if statistic == "average":
            return Objective(metric, direction="minimize")
        if statistic.startswith("p") and statistic[1:].isdigit():
            return Objective(metric, int(statistic[1:]), "minimize")
    return Objective(key)


def extract_objective(metrics: Mapping[str, object], objective: Objective) -> float | None:
    value = metrics.get(objective.key)
    if value is None or isinstance(value, str):
        value = metrics.get(objective.metric)
        if not isinstance(value, Mapping) and objective.percentile is not None:
            value = None
    if isinstance(value, Mapping):
        successful = value.get("successful")
        source = successful if isinstance(successful, Mapping) else value
        if objective.percentile is not None:
            key = f"p{objective.percentile}"
            percentiles = source.get("percentiles", {})
            value = source.get(key, percentiles.get(key) if isinstance(percentiles, Mapping) else None)
            if value is None and objective.percentile == 50:
                value = source.get("median")
        else:
            value = source.get("average", source.get("mean"))
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value) else None

"""Pure bounded load-probe policy; execution lives in calibration_worker."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from vllm_optimizer.config.models import VTuneConfig


@dataclass(frozen=True, slots=True)
class CalibrationPolicy:
    max_concurrency: int = 64
    probe_duration: float = 30
    max_duration: float = 180
    latency_growth: float = 2
    throughput_gain: float = 0.05

    @property
    def loads(self) -> tuple[int, ...]:
        values = [1]
        while values[-1] < self.max_concurrency:
            values.append(min(values[-1] * 2, self.max_concurrency))
        return tuple(values)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    concurrency: int
    throughput: float
    latency: float
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    concurrency: int
    reason: str


def calibration_policy(config: VTuneConfig) -> CalibrationPolicy:
    raw = config.optimization.get("calibration", {})
    defaults = CalibrationPolicy()
    names = ("max_concurrency", "probe_duration", "max_duration", "latency_growth", "throughput_gain")
    if not isinstance(raw, Mapping) or set(raw) - set(names):
        raise ValueError("unsupported optimization.calibration setting")
    values = {name: raw.get(name, getattr(defaults, name)) for name in names}
    if any(
        isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value) or value <= 0
        for value in values.values()
    ):
        raise ValueError("calibration values must be finite positive numbers")
    cap = values["max_concurrency"]
    if not isinstance(cap, int) or cap > 4096:
        raise ValueError("calibration.max_concurrency must be an integer from 1 to 4096")
    if values["latency_growth"] <= 1 or values["throughput_gain"] >= 1:
        raise ValueError("calibration requires latency_growth > 1 and throughput_gain < 1")
    return CalibrationPolicy(cap, *(float(values[name]) for name in names[1:]))


def select_load(probes: Sequence[ProbeResult], policy: CalibrationPolicy | None = None) -> CalibrationDecision:
    policy = policy or CalibrationPolicy()
    previous: ProbeResult | None = None
    for probe in probes:
        if probe.failure or probe.throughput <= 0 or probe.latency <= 0:
            if previous is None:
                raise ValueError("calibration did not produce a usable load at concurrency 1")
            return CalibrationDecision(previous.concurrency, probe.failure or "invalid_measurement")
        if previous is not None:
            if probe.latency > previous.latency * policy.latency_growth:
                return CalibrationDecision(previous.concurrency, "latency_growth")
            if probe.throughput < previous.throughput * (1 + policy.throughput_gain):
                return CalibrationDecision(previous.concurrency, "throughput_plateau")
        previous = probe
    if previous is None:
        raise ValueError("calibration did not produce a usable load")
    return CalibrationDecision(previous.concurrency, "limit_reached")

"""Configuration for conservative repeat allocation during search."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite

from vllm_optimizer.benchmarks.configuration import configured_min_repeats, configured_repeats
from vllm_optimizer.config.models import VTuneConfig


@dataclass(frozen=True, slots=True)
class AdaptiveRepeatPolicy:
    minimum_relative_score: float = 0.7


def adaptive_repeat_policy(config: VTuneConfig) -> AdaptiveRepeatPolicy | None:
    raw = config.benchmark.get("adaptive_repeats")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) - {"minimum_relative_score"}:
        raise ValueError("benchmark.adaptive_repeats supports only minimum_relative_score")
    value = raw.get("minimum_relative_score", 0.7)
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value) or not 0 < value <= 1:
        raise ValueError("benchmark.adaptive_repeats.minimum_relative_score must be finite and between 0 and 1")
    if configured_min_repeats(config) >= configured_repeats(config):
        raise ValueError("benchmark.adaptive_repeats requires min_repeats to be less than repeats")
    return AdaptiveRepeatPolicy(float(value))


def without_adaptive_repeats(config: VTuneConfig) -> VTuneConfig:
    return replace(
        config, benchmark={name: value for name, value in config.benchmark.items() if name != "adaptive_repeats"}
    )

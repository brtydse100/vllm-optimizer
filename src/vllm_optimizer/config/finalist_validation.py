"""Opt-in, fixed-budget validation policy, independent of search measurements."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite

from vllm_optimizer.benchmarks.configuration import configured_min_repeats
from vllm_optimizer.config.adaptive_repeats import without_adaptive_repeats
from vllm_optimizer.config.models import VTuneConfig


@dataclass(frozen=True, slots=True)
class FinalistPolicy:
    top_k: int = 2
    repeats: int = 6


def finalist_policy(config: VTuneConfig) -> FinalistPolicy | None:
    raw = config.analysis.get("finalist_validation")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) - {"top_k", "repeats"}:
        raise ValueError("analysis.finalist_validation supports only top_k and repeats")
    drift = config.analysis.get("drift_threshold", 0.05)
    if isinstance(drift, bool) or not isinstance(drift, int | float) or not isfinite(drift) or drift < 0:
        raise ValueError("analysis.drift_threshold must be finite and non-negative")
    top_k, repeats = raw.get("top_k", 2), raw.get("repeats", 6)
    for name, value, minimum in (("top_k", top_k, 1), ("repeats", repeats, max(2, configured_min_repeats(config)))):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"analysis.finalist_validation.{name} must be an integer >= {minimum}")
    return FinalistPolicy(top_k, repeats)


def validation_config(config: VTuneConfig, policy: FinalistPolicy) -> VTuneConfig:
    # Require every planned repeat to finish; never score a partial validation budget.
    fixed = without_adaptive_repeats(config)
    benchmark = {**fixed.benchmark, "repeats": policy.repeats, "min_repeats": policy.repeats}
    return replace(fixed, benchmark=benchmark)

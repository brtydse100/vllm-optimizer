"""Preserve numeric canonical latency evidence while redacting surrounding data."""

from collections.abc import Mapping

from vllm_optimizer.benchmarks.metrics import metric_summary
from vllm_optimizer.config.constraint_units import METRIC_UNITS
from vllm_optimizer.reproduction.redaction import redact


def redact_benchmarks(values: object) -> list[object]:
    if not isinstance(values, list):
        raise ValueError("benchmark evidence must be a list")
    safe: list[object] = []
    for benchmark in values:
        if not isinstance(benchmark, Mapping):
            raise ValueError("benchmark evidence must contain mappings")
        clean = redact(benchmark)
        assert isinstance(clean, dict)
        workloads = benchmark.get("workloads", [])
        clean_workloads = clean.get("workloads", [])
        if isinstance(workloads, list) and isinstance(clean_workloads, list):
            for original, target in zip(workloads, clean_workloads, strict=True):
                if not isinstance(original, Mapping) or not isinstance(target, dict):
                    continue
                metrics = original.get("metrics")
                target_metrics = target.get("metrics")
                if isinstance(metrics, Mapping) and isinstance(target_metrics, dict):
                    for name, unit in METRIC_UNITS.items():
                        if unit == "ms" and (summary := metric_summary(metrics.get(name))):
                            target_metrics[name] = summary
        safe.append(clean)
    return safe

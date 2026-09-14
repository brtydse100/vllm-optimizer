"""Shared benchmark-duration summaries for report sections."""

from __future__ import annotations

from statistics import fmean

from vllm_optimizer.domain.trial_report import TrialReport


def mean_duration(report: TrialReport | None, benchmark_name: str | None = None) -> float | None:
    if report is None:
        return None
    values = [
        float(value)
        for benchmark in report.benchmarks
        if benchmark_name is None or benchmark.get("name") == benchmark_name
        if isinstance((value := benchmark.get("elapsed_seconds")), int | float) and not isinstance(value, bool)
    ]
    return fmean(values) if values else None


def relative_duration(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return (value - baseline) / abs(baseline) * 100


def duration_change(value: float | None, baseline: float | None) -> str:
    difference = relative_duration(value, baseline)
    if difference is None:
        return "Unavailable"
    status = "better" if difference < 0 else ("worse" if difference > 0 else "same")
    return f"{difference:+.2f}% ({status})"

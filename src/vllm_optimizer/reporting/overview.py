"""Concise, coverage-checked summary of observed throughput and execution time."""

from html import escape

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.reporting.comparison import _metric_mean
from vllm_optimizer.reporting.comparison_evidence import matching_durations, matching_workloads
from vllm_optimizer.reporting.durations import mean_duration, relative_duration
from vllm_optimizer.reporting.workloads import formatted


def result_summary(baseline: TrialReport | None, selected: TrialReport | None) -> str:
    before = _metric_mean(baseline, "throughput_tokens_per_second")
    after = _metric_mean(selected, "throughput_tokens_per_second")
    base_time, selected_time = mean_duration(baseline), mean_duration(selected)
    throughput = relative_duration(after, before) if matching_workloads(baseline, selected) else None
    duration = relative_duration(selected_time, base_time) if matching_durations(baseline, selected) else None
    if baseline and selected and baseline.trial_id == selected.trial_id:
        sentence = "Baseline retained; no observed improvement is claimed."
    else:
        parts = []
        if throughput is not None:
            parts.append(_change(throughput, "higher", "lower", "output throughput"))
        if duration is not None:
            parts.append(_change(duration, "longer", "shorter", "average benchmark duration"))
        sentence = "Results show " + " and ".join(parts) + " versus baseline." if parts else "Comparison unavailable."
    unavailable = []
    if throughput is None:
        unavailable.append("Throughput change unavailable")
    if duration is None:
        unavailable.append("Duration change unavailable")
    rows = "".join(
        f"<tr><th>{label}</th><td>{formatted(left, unit)}</td><td>{formatted(right, unit)}</td></tr>"
        for label, left, right, unit in (
            ("Mean output throughput", before, after, "tok/s"),
            ("Average benchmark duration", base_time, selected_time, "s"),
        )
    )
    return (
        f"<p class='result-summary'>{escape(sentence)}</p>"
        + (
            f"<p class='muted'>{'; '.join(unavailable)}: matching, complete measurements are required.</p>"
            if unavailable
            else ""
        )
        + "<div class='table'><table class='summary-table'><thead><tr><th>Measurement</th>"
        "<th>Baseline</th><th>Best observed</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        "<p class='note'>Arithmetic means of recorded measurements; benchmark duration excludes server startup. "
        "The configured optimization score may use a different aggregation.</p>"
    )


def _change(value: float, higher: str, lower: str, label: str) -> str:
    if value == 0:
        return f"unchanged {label}"
    return f"{abs(value):.2f}% {higher if value > 0 else lower} {label}"

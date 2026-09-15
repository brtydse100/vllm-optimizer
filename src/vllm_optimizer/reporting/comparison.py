"""Side-by-side metrics for matching workloads, never pooled percentiles."""

from html import escape

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.reporting.analysis import DEFAULT_METRICS
from vllm_optimizer.reporting.comparison_evidence import matching_durations, matching_workloads
from vllm_optimizer.reporting.durations import mean_duration
from vllm_optimizer.reporting.tables import _table
from vllm_optimizer.reporting.workloads import center, delta, failure_samples, formatted, samples, scenarios

METRICS = (
    ("Output throughput", "throughput_tokens_per_second", "tok/s"),
    ("Request throughput", "requests_per_second", "req/s"),
    ("TTFT", "ttft_ms", "ms"),
    ("Time per output token", "tpot_ms", "ms"),
    ("Inter-token latency", "itl_ms", "ms"),
    ("End-to-end latency", "end_to_end_ms", "ms"),
)


def comparison(baseline: TrialReport | None, recommended: TrialReport | None) -> str:
    before, after = scenarios(baseline), scenarios(recommended)
    summary = _summary(baseline, recommended)
    durations = _duration_comparison(baseline, recommended)
    blocks = []
    for key in dict.fromkeys((*before, *after)):
        left, right = before.get(key, []), after.get(key, [])
        rows = []
        for label, metric, unit in METRICS:
            aliases = DEFAULT_METRICS[metric]
            for statistic, title in (("average", "Mean"), ("median", "P50"), ("p95", "P95"), ("p99", "P99")):
                a, b = samples(left, aliases, statistic), samples(right, aliases, statistic)
                if statistic != "average" and not (a or b):
                    continue
                rows.append(
                    _row(
                        f"{label} · {title}",
                        center(a),
                        center(b),
                        unit,
                        len(a),
                        len(b),
                        len(a) == len(b) == len(left) == len(right),
                    )
                )
        a, b = failure_samples(left), failure_samples(right)
        rows.append(
            _row(
                "Failed + incomplete requests",
                center(a),
                center(b),
                "requests",
                len(a),
                len(b),
                len(a) == len(b) == len(left) == len(right),
            )
        )
        blocks.append(
            f"<h3>{escape(str(key))}</h3>"
            "<details><summary>Full workload configuration JSON</summary>"
            f"<pre>{escape(key.pretty_configuration())}</pre></details>"
            + _table(("Metric", "Baseline", "Recommended", "Change", "Repeats (base / rec)"), "".join(rows))
        )
    return (
        "<section id='comparison'><h2>3. Baseline vs recommended</h2>"
        "<details open><summary>Latency, throughput, and benchmark duration comparison</summary>"
        "<p class='warning'>This comparison exposes latency trade-offs, but tuning a different objective is not "
        "the best way to search for the lowest latency. Use a latency objective for that purpose.</p>"
        "<p>Each value is the arithmetic mean of available repeat measurements for this exact workload. "
        "P50/P95/P99 are means of backend-supplied percentiles, not pooled request percentiles. "
        "Positive throughput changes improve performance; negative latency/failure changes improve performance. "
        "Percentage changes are unavailable for mismatched workloads, unequal repeat counts, or missing measurements. "
        "Overall means retain the same workload weighting only when coverage matches.</p>"
        + summary
        + durations
        + ("".join(blocks) or "<p>Unavailable: no workload measurements.</p>")
        + "</details></section>"
    )


def _row(
    label: str, baseline: float | None, recommended: float | None, unit: str, n: int, m: int, comparable: bool = True
) -> str:
    change = delta(recommended, baseline) if comparable else "Unavailable (incomplete coverage)"
    cells = (label, formatted(baseline, unit), formatted(recommended, unit), change, f"{n} / {m}")
    return "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>"


def _summary(baseline: TrialReport | None, recommended: TrialReport | None) -> str:
    rows = []
    for label, metric, unit, lower_is_better in (
        ("Mean E2E latency", "end_to_end_ms", "ms", True),
        ("Mean output throughput", "throughput_tokens_per_second", "tok/s", False),
        ("Mean benchmark duration", None, "s", True),
    ):
        before = mean_duration(baseline) if metric is None else _metric_mean(baseline, metric)
        after = mean_duration(recommended) if metric is None else _metric_mean(recommended, metric)
        comparable = (
            matching_durations(baseline, recommended) if metric is None else matching_workloads(baseline, recommended)
        )
        cells = (
            label,
            formatted(before, unit),
            formatted(after, unit),
            _change(after, before, lower_is_better) if comparable else "Unavailable (incomplete coverage)",
        )
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>")
    return "<h3>Overall means</h3>" + _table(("Metric", "Baseline", "Recommended", "Difference"), "".join(rows))


def _metric_mean(report: TrialReport | None, metric: str) -> float | None:
    aliases = DEFAULT_METRICS[metric]
    groups = scenarios(report)
    if any(len(samples(observations, aliases)) != len(observations) for observations in groups.values()):
        return None
    values = [value for observations in groups.values() for value in samples(observations, aliases)]
    return center(values)


def _duration_comparison(baseline: TrialReport | None, recommended: TrialReport | None) -> str:
    names = dict.fromkeys(
        str(item.get("name", "Unavailable"))
        for report in (baseline, recommended)
        if report
        for item in report.benchmarks
    )
    rows = []
    for name in names:
        before, after = mean_duration(baseline, name), mean_duration(recommended, name)
        cells = (
            name,
            formatted(before, "s"),
            formatted(after, "s"),
            _change(after, before, True)
            if matching_durations(baseline, recommended, name)
            else "Unavailable (incomplete coverage)",
        )
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>")
    headings = ("Benchmark", "Baseline", "Recommended", "Difference")
    return "<h3>Mean duration by benchmark</h3>" + _table(headings, "".join(rows))


def _change(value: float | None, baseline: float | None, lower_is_better: bool) -> str:
    if value is None or baseline is None or baseline == 0:
        return "Unavailable"
    percent = (value - baseline) / abs(baseline) * 100
    better = percent < 0 if lower_is_better else percent > 0
    status = "better" if better else ("worse" if percent else "same")
    return f"{percent:+.2f}% ({status})"

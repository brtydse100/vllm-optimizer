"""Side-by-side metrics for matching workloads, never pooled percentiles."""

from html import escape

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.reporting.analysis import DEFAULT_METRICS
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
    matching = bool(before) and set(before) == set(after)
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
                rows.append(_row(f"{label} · {title}", center(a), center(b), unit, len(a), len(b)))
        a, b = failure_samples(left), failure_samples(right)
        rows.append(_row("Failed + incomplete requests", center(a), center(b), "requests", len(a), len(b)))
        blocks.append(
            f"<h3>{escape(key)}</h3>"
            + _table(("Metric", "Baseline", "Recommended", "Change", "Repeats (base / rec)"), "".join(rows))
        )
    return (
        "<section id='comparison'><h2>3. Baseline vs recommended</h2>"
        f"<details{' open' if matching else ''}><summary>Latency, throughput, and benchmark duration comparison</summary>"
        "<p class='warning'>This comparison exposes latency trade-offs, but tuning a different objective is not "
        "the best way to search for the lowest latency. Use a latency objective for that purpose.</p>"
        "<p>Each value is the arithmetic mean of available repeat measurements for this exact workload. "
        "P50/P95/P99 are means of backend-supplied percentiles, not pooled request percentiles. "
        "Positive throughput changes improve performance; negative latency/failure changes improve performance. "
        "Missing or unmatched workloads remain unavailable.</p>"
        + summary
        + durations
        + ("".join(blocks) or "<p>Unavailable: no workload measurements.</p>")
        + "</details></section>"
    )


def _row(label: str, baseline: float | None, recommended: float | None, unit: str, n: int, m: int) -> str:
    difference = delta(recommended, baseline)
    if (not n or not m or n != m) and difference != "Unavailable":
        difference = "Unavailable (incomplete coverage)"
    cells = (label, formatted(baseline, unit), formatted(recommended, unit), difference, f"{n} / {m}")
    return "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>"


def _summary(baseline: TrialReport | None, recommended: TrialReport | None) -> str:
    rows = []
    before, after = scenarios(baseline), scenarios(recommended)
    for label, metric, unit, lower_is_better in (
        ("Mean E2E latency", "end_to_end_ms", "ms", True),
        ("Mean output throughput", "throughput_tokens_per_second", "tok/s", False),
        ("Mean benchmark duration", None, "s", True),
    ):
        if metric is None:
            before_value, after_value, complete = _paired_durations(baseline, recommended)
        else:
            before_value, after_value, complete = _paired_metric_means(before, after, metric)
        cells = (
            label,
            formatted(before_value, unit),
            formatted(after_value, unit),
            _change(after_value, before_value, lower_is_better, incomplete=not complete),
        )
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>")
    return "<h3>Overall means</h3>" + _table(("Metric", "Baseline", "Recommended", "Difference"), "".join(rows))


def _metric_mean(report: TrialReport | None, metric: str) -> float | None:
    aliases = DEFAULT_METRICS[metric]
    values = [value for observations in scenarios(report).values() for value in samples(observations, aliases)]
    return center(values)


def _paired_metric_means(
    baseline: dict[str, list], recommended: dict[str, list], metric: str
) -> tuple[float | None, float | None, bool]:
    if not baseline or baseline.keys() != recommended.keys():
        return None, None, False
    aliases = DEFAULT_METRICS[metric]
    left, right = [], []
    for key in baseline:
        a, b = samples(baseline[key], aliases), samples(recommended[key], aliases)
        if not a or len(a) != len(b):
            return None, None, False
        left.extend(a)
        right.extend(b)
    return center(left), center(right), True


def _paired_durations(
    baseline: TrialReport | None, recommended: TrialReport | None
) -> tuple[float | None, float | None, bool]:
    if (
        baseline is None
        or recommended is None
        or set(scenarios(baseline)) != set(scenarios(recommended))
        or len(baseline.benchmarks) != len(recommended.benchmarks)
    ):
        return None, None, False
    left = [item.get("elapsed_seconds") for item in baseline.benchmarks]
    right = [item.get("elapsed_seconds") for item in recommended.benchmarks]
    if not all(isinstance(value, int | float) and not isinstance(value, bool) for value in (*left, *right)):
        return None, None, False
    numeric_left: list[float] = []
    numeric_right: list[float] = []
    for value in left:
        assert isinstance(value, int | float) and not isinstance(value, bool)
        numeric_left.append(float(value))
    for value in right:
        assert isinstance(value, int | float) and not isinstance(value, bool)
        numeric_right.append(float(value))
    return sum(numeric_left) / len(numeric_left), sum(numeric_right) / len(numeric_right), True


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
        cells = (name, formatted(before, "s"), formatted(after, "s"), _change(after, before, True))
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>")
    return "<h3>Mean duration by benchmark</h3>" + _table(
        ("Benchmark", "Baseline", "Recommended", "Difference"), "".join(rows)
    )


def _change(value: float | None, baseline: float | None, lower_is_better: bool, incomplete: bool = False) -> str:
    if value is None or baseline is None or baseline == 0:
        return "Unavailable (incomplete coverage)" if incomplete else "Unavailable"
    percent = (value - baseline) / abs(baseline) * 100
    better = percent < 0 if lower_is_better else percent > 0
    status = "better" if better else ("worse" if percent else "same")
    return f"{percent:+.2f}% ({status})"

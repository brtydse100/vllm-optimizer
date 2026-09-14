"""Side-by-side metrics for matching workloads, never pooled percentiles."""

from html import escape

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.reporting.analysis import DEFAULT_METRICS
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
        "<details><summary>Latency and throughput comparison</summary>"
        "<p class='warning'>This comparison exposes latency trade-offs, but tuning a different objective is not "
        "the best way to search for the lowest latency. Use a latency objective for that purpose.</p>"
        "<p>Each value is the arithmetic mean of available repeat measurements for this exact workload. "
        "P50/P95/P99 are means of backend-supplied percentiles, not pooled request percentiles. "
        "Positive throughput changes improve performance; negative latency/failure changes improve performance. "
        "Missing or unmatched workloads remain unavailable.</p>"
        + ("".join(blocks) or "<p>Unavailable: no workload measurements.</p>")
        + "</details></section>"
    )


def _row(label: str, baseline: float | None, recommended: float | None, unit: str, n: int, m: int) -> str:
    cells = (label, formatted(baseline, unit), formatted(recommended, unit), delta(recommended, baseline), f"{n} / {m}")
    return "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>"

"""Compose the self-contained HTML decision report from accepted trial evidence."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.scoring import TrialScore
from vllm_optimizer.reporting.charts import comparison_chart, effect_charts, history_chart
from vllm_optimizer.reporting.comparison import comparison
from vllm_optimizer.reporting.confidence import confidence, verdict
from vllm_optimizer.reporting.context import ReportContext
from vllm_optimizer.reporting.dashboard_selection import best_observed, improvement
from vllm_optimizer.reporting.importance import importance_section
from vllm_optimizer.reporting.interactions import report_script
from vllm_optimizer.reporting.leaderboard import leaderboard
from vllm_optimizer.reporting.methodology import metric_methodology
from vllm_optimizer.reporting.recommendation import recommendation
from vllm_optimizer.reporting.styles import dashboard_css
from vllm_optimizer.reporting.tables import evidence_table, failures
from vllm_optimizer.reporting.workloads import formatted


def render_dashboard(
    directory: Path,
    metric: str,
    trials: tuple[TrialReport, ...],
    ranking: tuple[TrialScore, ...],
    baseline: TrialScore | None,
    context: ReportContext,
) -> str:
    best = best_observed(ranking[0] if ranking else None, baseline)
    reports = {report.trial_id: report for report in trials}
    selected = reports.get(best.trial_id) if best else None
    base_report = reports.get(baseline.trial_id) if baseline else None
    finalist = reports.get(ranking[0].trial_id) if ranking else selected
    change = improvement(best, baseline)
    conclusion = verdict(base_report, finalist, metric, context)
    outcome = "Best observed configuration; baseline comparison unavailable"
    if best is None:
        outcome = "No eligible configuration"
    elif baseline and best.trial_id == baseline.trial_id:
        outcome = "Baseline wins — keep the baseline"
    elif change is not None:
        outcome = (
            f"Tuning improved the observed score by {change:.2f}%"
            if change > 0
            else "Score tied with baseline; recommendation selected by request quality"
        )
    counts = {
        state: sum(item.status.value == state for item in trials) for state in ("completed", "failed", "interrupted")
    }
    cards = "".join(
        (
            _card("Best observed", best.trial_id if best else "Unavailable", formatted(best.value if best else None)),
            _card(
                "Improvement vs baseline", f"{change:+.2f}%" if change is not None else "Unavailable", "Accepted score"
            ),
            _card("Experiment time", _duration(context), "Wall clock"),
            _card(
                "Trials",
                f"{counts['completed']} completed / {counts['failed']} failed",
                f"{counts['interrupted']} interrupted · {context.status}",
            ),
        )
    )
    contention = (
        "<p class='warning'>Parallel search can introduce shared-resource contention. "
        "Finalist validation is identified separately below when available.</p>"
        if context.execution_mode == "local_parallel"
        else ""
    )
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>vLLM Optimizer · {escape(context.run_id)}</title><style>{dashboard_css()}</style></head><body>
<header><div><p class='eyebrow'>vLLM Optimizer decision report</p><h1>{escape(context.run_id)}</h1>
<p>Maximize <code>{escape(metric)}</code> · Mode {escape(context.execution_mode)}</p>
<nav aria-label='Report sections'><a href='#overview'>Overview</a> · <a href='#recommendation'>Configuration</a> ·
<a href='#comparison'>Comparison</a> · <a href='#confidence'>Confidence</a> · <a href='#leaderboard'>All trials</a></nav>
</div></header><main><section id='overview'><h2>1. Result overview</h2><h3>{escape(outcome)}</h3>
<p>{escape(conclusion)}</p><div class='cards'>{cards}</div>{contention}</section>
{recommendation(directory, best, baseline, selected)}
{comparison(base_report, selected)}
{confidence(directory, base_report, finalist, metric, context)}
{leaderboard(trials, ranking, baseline, metric, directory)}{_diagnostics(trials, ranking, baseline, context)}
</main><footer>Generated from stored experiment artifacts. Missing evidence is shown as unavailable.</footer>
{report_script()}</body></html>"""


def _diagnostics(
    trials: tuple[TrialReport, ...],
    ranking: tuple[TrialScore, ...],
    baseline: TrialScore | None,
    context: ReportContext,
) -> str:
    llm = context.llm_summary or context.llm_summary_error
    return (
        "<details><summary>Detailed diagnostics and exploratory analysis</summary>"
        "<section><h2>Accepted score comparison</h2>"
        + comparison_chart(ranking, baseline)
        + "<h3>Accepted scores in recorded trial order</h3>"
        + history_chart(trials, ranking)
        + "<p>This is trial order, not elapsed time. Superseded search executions are excluded.</p></section>"
        "<section><h2>Parameter associations</h2>"
        + importance_section(ranking)
        + "<p>Observed score means by tested value, with sample counts; associations, not controlled ablations.</p>"
        + effect_charts(ranking)
        + "</section><section><h2>Failures and interruptions</h2>"
        + failures(trials)
        + "</section>"
        + metric_methodology()
        + "<section><h2>Scoring evidence</h2>"
        + evidence_table(ranking)
        + f"<p>A workload is excluded when failed/incomplete requests exceed\n{context.maximum_failure_percentage:g}% of all requests. "
        "Eligible workload metrics use their mean per named run, the median across repeats, then the mean of named runs. "
        "This configured objective is separate from the workload comparisons above.</p></section>"
        + (f"<section><h2>Optional LLM summary</h2><p>{escape(llm)}</p></section>" if llm else "")
        + "</details>"
    )


def _card(label: str, value: str, detail: str) -> str:
    return (
        f"<article><span>{escape(label)}</span><strong>{escape(value)}</strong>"
        f"<small>{escape(detail)}</small></article>"
    )


def _duration(context: ReportContext) -> str:
    if not context.started_at or not context.completed_at:
        return "Unavailable"
    try:
        elapsed = (
            datetime.fromisoformat(context.completed_at) - datetime.fromisoformat(context.started_at)
        ).total_seconds()
        return formatted(elapsed, "s") if elapsed >= 0 else "Unavailable"
    except (ValueError, TypeError):
        return "Unavailable"

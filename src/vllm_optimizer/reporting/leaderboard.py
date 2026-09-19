"""Every trial stays visible, with sortable values and expandable execution details."""

from __future__ import annotations

import json
from collections.abc import Mapping
from html import escape
from pathlib import Path
from statistics import stdev

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.scoring import ScoringManager, TrialScore
from vllm_optimizer.reporting.durations import duration_change, mean_duration
from vllm_optimizer.reporting.recommendation import manifest_settings, setting_changes, settings
from vllm_optimizer.reporting.workloads import delta, failure_samples, formatted, samples, scenarios
from vllm_optimizer.reproduction.accepted import accepted_manifest
from vllm_optimizer.reproduction.redaction import redact_values


def leaderboard(
    trials: tuple[TrialReport, ...],
    ranking: tuple[TrialScore, ...],
    baseline: TrialScore | None,
    metric: str,
    directory: Path | None = None,
) -> str:
    scores = {item.trial_id: item for item in ranking}
    if baseline:
        scores[baseline.trial_id] = baseline
    ordered = ScoringManager.rank(list(scores.values()))
    ranks = {item.trial_id: index for index, item in enumerate(ordered, 1)}
    reports = {item.trial_id: item for item in trials}
    baseline_duration = mean_duration(reports.get(baseline.trial_id) if baseline else None)
    rows = []
    for trial in sorted(trials, key=lambda item: ranks.get(item.trial_id, len(trials) + 1)):
        score = scores.get(trial.trial_id)
        workloads = scenarios(trial)
        variability = []
        counts = []
        details = []
        for name, observations in workloads.items():
            values = samples(observations, (metric,))
            sd = stdev(values) if len(values) >= 2 else None
            if sd is not None:
                variability.append(sd)
            failures = failure_samples(observations)
            if len(failures) == len(observations):
                counts.extend(failures)
            else:
                counts.append(float("nan"))
            details.append(f"<p>{escape(str(name))}: score SD {formatted(sd)}; n={len(values)}</p>")
        duration = mean_duration(trial)
        details.extend(_duration_details(trial))
        error_count = sum(counts) if counts and all(value == value for value in counts) else None
        selected_settings = settings(score) if score else None
        configuration = json.dumps(selected_settings, indent=2)
        if score is None and directory is not None:
            try:
                manifest = accepted_manifest(directory, trial.trial_id, trial.execution)
                selected_settings = manifest_settings(manifest)
                configuration = json.dumps(selected_settings, indent=2)
            except ValueError:
                configuration = "Configuration unavailable"
        diff = (
            setting_changes(settings(baseline), selected_settings)
            if baseline and selected_settings is not None
            else None
        )
        detail = (
            "<details><summary>Full details</summary>" + f"<p>Failed requests (all repeats): {formatted(error_count)}; "
            f"Score SD (largest workload): {formatted(max(variability) if variability else None)}</p>"
            + f"<p>Duration vs baseline: {duration_change(duration, baseline_duration)}</p>"
            + f"<p>Changed settings: {escape(json.dumps(diff, sort_keys=True)) if diff is not None else 'Unavailable'}</p>"
            + "".join(details)
            + f"<pre>{escape(configuration)}</pre>"
            + _adaptive_details(trial.execution.get("adaptive_repeats"))
            + "<details><summary>All recorded trial data (JSON)</summary>"
            + f"<pre>{escape(json.dumps(redact_values(trial.to_dict()), indent=2))}</pre></details>"
            + "</details>"
        )
        cells = [
            _cell(str(ranks.get(trial.trial_id, "Unranked")), ranks.get(trial.trial_id)),
            _cell(
                trial.trial_id + (" (baseline)" if baseline and trial.trial_id == baseline.trial_id else ""),
                trial.trial_id,
            ),
            _cell(trial.status.value, trial.status.value),
            _cell(formatted(score.value if score else None), score.value if score else None),
            _cell(
                delta(score.value if score else None, baseline.value if baseline else None),
                (score.value - baseline.value) / abs(baseline.value) * 100
                if score and baseline and baseline.value
                else None,
            ),
            _cell(formatted(duration, "s"), duration),
            f"<td>{detail}</td>",
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    labels = ("Rank", "Trial", "Status", "Score", "Baseline delta", "Mean benchmark duration", "Details")
    headers = "".join(
        f"<th><button data-sort='{index}' data-type='{'text' if index in (1, 2) else 'number'}'>"
        f"{escape(label)}</button></th>"
        if index < 6
        else f"<th>{label}</th>"
        for index, label in enumerate(labels)
    )
    return (
        "<section id='leaderboard'><h2>All trials</h2>"
        "<p>Every trial, including failures. Mean benchmark duration excludes server startup. "
        "Expand Full details for repeat timings, settings, errors, and all recorded metrics. Click a heading to sort.</p>"
        f"<div class='table'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _cell(label: str, value: object) -> str:
    return f"<td data-value='{escape(str(value) if value is not None else '', quote=True)}'>{escape(label)}</td>"


def _adaptive_details(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    status = escape(str(value.get("status", "Unavailable")))
    counts = (
        escape(str(dict(value.get("actual_repeats_by_run", {}))))
        if isinstance(value.get("actual_repeats_by_run"), Mapping)
        else "Unavailable"
    )
    reason = escape(str(value.get("reason", "Unavailable")))
    return f"<p>Adaptive repeats: {status}; actual by workload: {counts}; {reason}.</p>"


def _duration_details(report: TrialReport) -> list[str]:
    rows = []
    for benchmark in report.benchmarks:
        value = benchmark.get("elapsed_seconds")
        if isinstance(value, int | float) and not isinstance(value, bool):
            name = escape(str(benchmark.get("name", "unknown")))
            repeat = escape(str(benchmark.get("repeat", "Unavailable")))
            rows.append(f"<p>{name}, repeat {repeat}: duration {formatted(float(value), 's')}</p>")
    return rows

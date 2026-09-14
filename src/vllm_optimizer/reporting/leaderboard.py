"""Every trial stays visible, with sortable values and expandable execution details."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from statistics import stdev

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.scoring import ScoringManager, TrialScore
from vllm_optimizer.reporting.recommendation import manifest_settings, setting_changes, settings
from vllm_optimizer.reporting.workloads import center, delta, failure_samples, formatted, samples, scenarios
from vllm_optimizer.reproduction.accepted import accepted_manifest


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
    baseline_duration = _mean_duration(reports.get(baseline.trial_id) if baseline else None)
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
            details.append(f"<p>{escape(name)}: score SD {formatted(sd)}; n={len(values)}</p>")
        duration = _mean_duration(trial)
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
            "<details><summary>Full details</summary>"
            + "".join(details)
            + f"<pre>{escape(configuration)}</pre>"
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
            _cell(_duration_delta(duration, baseline_duration), _relative(duration, baseline_duration)),
            _cell(formatted(error_count), error_count),
            _cell(formatted(max(variability) if variability else None), max(variability) if variability else None),
            _cell(
                json.dumps(diff, sort_keys=True) if diff is not None else "Unavailable",
                json.dumps(diff, sort_keys=True) if diff is not None else None,
            ),
            f"<td>{detail}</td>",
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    labels = (
        "Rank",
        "Trial",
        "Status",
        "Score",
        "Baseline delta",
        "Mean benchmark duration",
        "Duration vs baseline",
        "Failed requests (all repeats)",
        "Score SD (largest workload)",
        "Changed settings",
        "Details",
    )
    headers = "".join(
        f"<th><button data-sort='{index}' data-type='{'text' if index in (1, 2, 9) else 'number'}'>"
        f"{escape(label)}</button></th>"
        if index < 10
        else f"<th>{label}</th>"
        for index, label in enumerate(labels)
    )
    return (
        "<section id='leaderboard'><h2>5. Configuration leaderboard</h2>"
        "<p>All trials, including duplicate configurations and unranked failures. Benchmark duration is the mean "
        "of recorded benchmark execution times; its baseline difference labels lower duration as better. Expand a "
        "row for individual execution durations. Click a heading to sort; missing values stay last.</p>"
        f"<div class='table'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _cell(label: str, value: object) -> str:
    return f"<td data-value='{escape(str(value) if value is not None else '', quote=True)}'>{escape(label)}</td>"


def _mean_duration(report: TrialReport | None) -> float | None:
    if report is None:
        return None
    values = [
        float(value)
        for benchmark in report.benchmarks
        if isinstance((value := benchmark.get("elapsed_seconds")), int | float) and not isinstance(value, bool)
    ]
    return center(values)


def _relative(value: float | None, baseline: float | None) -> float | None:
    return (value - baseline) / abs(baseline) * 100 if value is not None and baseline not in (None, 0) else None


def _duration_delta(value: float | None, baseline: float | None) -> str:
    difference = _relative(value, baseline)
    if difference is None:
        return "Unavailable"
    status = "better" if difference < 0 else ("worse" if difference > 0 else "same")
    return f"{difference:+.2f}% ({status})"


def _duration_details(report: TrialReport) -> list[str]:
    rows = []
    for benchmark in report.benchmarks:
        value = benchmark.get("elapsed_seconds")
        if isinstance(value, int | float) and not isinstance(value, bool):
            name = escape(str(benchmark.get("name", "unknown")))
            repeat = escape(str(benchmark.get("repeat", "Unavailable")))
            rows.append(f"<p>{name}, repeat {repeat}: duration {formatted(float(value), 's')}</p>")
    return rows

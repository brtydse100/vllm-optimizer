"""Every trial stays visible, with sortable values and expandable execution details."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from statistics import stdev

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.scoring import ScoringManager, TrialScore
from vllm_optimizer.reporting.analysis import DEFAULT_METRICS
from vllm_optimizer.reporting.recommendation import manifest_settings, setting_changes, settings
from vllm_optimizer.reporting.workloads import center, delta, failure_samples, formatted, samples, scenarios
from vllm_optimizer.reproduction.accepted import accepted_manifest
from vllm_optimizer.reproduction.redaction import redact


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
    rows = []
    for trial in sorted(trials, key=lambda item: ranks.get(item.trial_id, len(trials) + 1)):
        score = scores.get(trial.trial_id)
        workloads = scenarios(trial)
        latencies, variability = [], []
        counts = []
        details = []
        for name, observations in workloads.items():
            latency = center(samples(observations, DEFAULT_METRICS["end_to_end_ms"]))
            values = samples(observations, (metric,))
            sd = stdev(values) if len(values) >= 2 else None
            if latency is not None:
                latencies.append(latency)
            if sd is not None:
                variability.append(sd)
            failures = failure_samples(observations)
            if len(failures) == len(observations):
                counts.extend(failures)
            else:
                counts.append(float("nan"))
            details.append(
                f"<p>{escape(name)}: E2E {formatted(latency, 'ms')}; score SD {formatted(sd)}; n={len(values)}</p>"
            )
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
            + f"<pre>{escape(json.dumps(redact(trial.to_dict()), indent=2))}</pre></details>"
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
            _cell(formatted(max(latencies) if latencies else None, "ms"), max(latencies) if latencies else None),
            _cell(formatted(error_count), error_count),
            _cell(formatted(max(variability) if variability else None), max(variability) if variability else None),
            _cell("Unavailable", None),
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
        "E2E (worst workload median)",
        "Failed requests (all repeats)",
        "Score SD (largest workload)",
        "Total runtime",
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
        "<p>All trials, including duplicate configurations and unranked failures. Latency and variability columns "
        "show the largest available workload summary; expand for workload differences. Total trial runtime is "
        "unavailable in existing artifacts. Click a heading to sort; missing values stay last.</p>"
        f"<div class='table'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _cell(label: str, value: object) -> str:
    return f"<td data-value='{escape(str(value) if value is not None else '', quote=True)}'>{escape(label)}</td>"

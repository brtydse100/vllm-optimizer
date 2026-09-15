"""Require comparable workload coverage before reporting improvement percentages."""

from dataclasses import replace

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.reporting.workloads import number, scenarios


def matching_workloads(before: TrialReport | None, after: TrialReport | None, name: str | None = None) -> bool:
    def counts(report: TrialReport | None) -> dict[str, int]:
        if report and name is not None:
            report = replace(report, benchmarks=tuple(b for b in report.benchmarks if b.get("name") == name))
        return {key: len(values) for key, values in scenarios(report).items()}

    left, right = counts(before), counts(after)
    return bool(left) and left == right


def matching_durations(before: TrialReport | None, after: TrialReport | None, name: str | None = None) -> bool:
    if not matching_workloads(before, after, name):
        return False
    lengths = []
    for report in (before, after):
        if report is None:
            return False
        benchmarks = [b for b in report.benchmarks if name is None or b.get("name") == name]
        if not benchmarks or any((value := number(b.get("elapsed_seconds"))) is None or value < 0 for b in benchmarks):
            return False
        lengths.append(len(benchmarks))
    return lengths[0] == lengths[1]

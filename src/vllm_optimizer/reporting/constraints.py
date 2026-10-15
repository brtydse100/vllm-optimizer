"""Constraint evidence shared by stored run documents, CSV, and HTML."""

from html import escape

from vllm_optimizer.domain.results import WorkerStatus
from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.constraints import ConstraintAssessment, assess_constraints
from vllm_optimizer.reporting.context import ReportContext
from vllm_optimizer.reporting.reclassify_scores import results

NO_TARGET_MATCH = "No tested configuration met all targets."


def assess_report(report: TrialReport, context: ReportContext) -> ConstraintAssessment:
    minimum = context.minimum_repeats
    if report.execution.get("artifact_subdirectory") == "finalist-validation" and context.finalist_validation:
        repeats = context.finalist_validation.get("repeats", minimum)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("invalid finalist repeat budget for performance constraints")
        minimum = repeats
    return assess_constraints(
        results(report) if report.status is WorkerStatus.COMPLETED else (),
        context.constraints,
        minimum,
        context.benchmark_names,
        context.maximum_failure_percentage,
    )


def constraint_documents(trials: tuple[TrialReport, ...], context: ReportContext) -> dict[str, object]:
    if not context.constraints:
        return {}
    return {
        "constraints": [item.to_dict() for item in context.constraints],
        "constraint_assessments": {item.trial_id: assess_report(item, context).to_dict() for item in trials},
    }


def render_constraints(trials: tuple[TrialReport, ...], context: ReportContext) -> str:
    if not context.constraints:
        return ""
    rows = []
    for trial in trials:
        assessment = assess_report(trial, context)
        for item in assessment.evidence:
            measured = f"{item['measured']:g}" if isinstance(item["measured"], int | float) else "Unavailable"
            operator = "≤" if item["operator"] == "max" else "≥"
            cells = (
                trial.trial_id,
                item["benchmark"],
                item["workload"],
                item["metric"],
                item["statistic"],
                f"{measured} {item['unit']}",
                f"{operator} {item['bound']:g} {item['unit']}",
                "Met" if item["met"] else "Not met",
            )
            rows.append("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells) + "</tr>")
    headers = ("Trial", "Benchmark", "Workload", "Metric", "Statistic", "Measured", "Limit", "Target")
    return (
        "<section><h2>Performance targets</h2><p>All limits must hold for every workload. "
        "Values are means of eligible repeats; percentiles are supplied by the backend, not pooled.</p>"
        "<div class='table'><table><thead><tr>"
        + "".join(f"<th>{label}</th>" for label in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def target_summary(trials: tuple[TrialReport, ...], context: ReportContext) -> list[str]:
    if not context.constraints:
        return []
    lines = []
    for trial in trials:
        assessment = assess_report(trial, context)
        lines.append(f"Performance targets {trial.trial_id}: {'met' if assessment.feasible else 'not met'}")
        for item in assessment.evidence:
            measured = f"{item['measured']:g}" if isinstance(item["measured"], int | float) else "Unavailable"
            lines.append(
                f"  {item['benchmark']} workload {item['workload']}: {item['metric']} "
                f"{item['statistic']}={measured} {item['unit']}; "
                f"{item['operator']} {item['bound']:g} {item['unit']}; {'met' if item['met'] else 'not met'}"
            )
    return lines

"""Persist resolved auto settings and link calibration evidence in reports."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from html import escape
from pathlib import Path

import yaml

from vllm_optimizer.config.models import VTuneConfig
from vllm_optimizer.managers.run_finalization import RunOutcome
from vllm_optimizer.objectives import objective_from_config
from vllm_optimizer.reproduction.redaction import redact, redact_environment


def finish_auto_run(outcome: RunOutcome, config: VTuneConfig, evidence: Mapping[str, object]) -> None:
    directory = outcome.directory
    document: dict[str, object] = {"schema_version": 1, "experiment": asdict(config.experiment)}
    for name in (
        "server",
        "tune",
        "tune_env",
        "benchmark",
        "baseline",
        "optimization",
        "analysis",
        "timeouts",
        "logging",
        "execution",
    ):
        document[name] = dict(getattr(config, name))
    document["env"] = redact_environment({key: str(value) for key, value in config.env.items()})
    (directory / "resolved.yaml").write_text(yaml.safe_dump(redact(document), sort_keys=False), encoding="utf-8")
    summary = {**evidence, "objective": objective_from_config(config).to_dict()}
    (directory / "auto.json").write_text(json.dumps(redact(summary), indent=2) + "\n", encoding="utf-8")
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    best = result.get("best")
    if isinstance(best, dict):
        server = {
            **(config.external_server_config or {}),
            **best.get("server_args", {}),
            "model": config.server["model"],
        }
        server.pop("config", None)
        (directory / "best-server.yaml").write_text(yaml.safe_dump(server, sort_keys=False), encoding="utf-8")
        (directory / "best-environment.yaml").write_text(yaml.safe_dump(best.get("server_env", {})), encoding="utf-8")
    add_auto_summary(directory / "report.html", directory)


def add_auto_summary(html_path: Path, artifacts: Path) -> None:
    source = artifacts / "auto.json"
    if not source.is_file():
        return
    document = json.loads(source.read_text(encoding="utf-8"))
    html = html_path.read_text(encoding="utf-8")
    if 'id="auto-calibration"' in html:
        return
    rows = "".join(
        f"<li>{escape(str(item['workload']))}: concurrency={escape(str(item['concurrency']))}, {escape(str(item['reason']))}</li>"
        for item in document.get("workloads", [])
    )
    objective = escape(json.dumps(document.get("objective", {})))
    section = (
        '<section id="auto-calibration"><h2>Automatic tuning</h2>'
        f"<p>Objective: {objective}</p><p>{escape(str(document.get('status')))}</p><ul>{rows}</ul>"
        "<p>Load is fixed across candidates. Reported percentile aggregates are means of workload percentiles, "
        "not pooled request percentiles. Tail estimates from small request counts are uncertain.</p>"
        f"<p>Resolved settings and calibration evidence: {escape(str(artifacts))}</p></section>"
    )
    html_path.write_text(html.replace("</body>", section + "</body>"), encoding="utf-8")

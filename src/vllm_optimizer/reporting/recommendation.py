"""Render baseline changes and the accepted execution's reproduction settings."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path

import yaml

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.managers.scoring import TrialScore
from vllm_optimizer.reporting.tables import _table
from vllm_optimizer.reproduction.accepted import accepted_manifest
from vllm_optimizer.reproduction.reader import commands, render_command
from vllm_optimizer.reproduction.redaction import redact_arguments, redact_environment, redact_values


def settings(score: TrialScore | None) -> dict[str, object]:
    if score is None:
        return {}
    return {
        **redact_values(score.server_args),
        **{f"env.{name}": value for name, value in redact_values(score.server_env).items()},
    }


def changes(best: TrialScore | None, baseline: TrialScore | None) -> dict[str, tuple[object, object]]:
    return setting_changes(settings(baseline), settings(best))


def setting_changes(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, tuple[object, object]]:
    return {
        name: (before.get(name, "Not explicitly set"), after.get(name, "Not explicitly set"))
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    }


def manifest_settings(manifest: Mapping[str, object]) -> dict[str, object]:
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Configuration unavailable")
    values: dict[str, object] = {}
    for name in ("fixed_args", "selected_args", "fixed_env", "selected_env"):
        entries = parameters.get(name, {})
        if not isinstance(entries, Mapping):
            raise ValueError("Invalid configuration mapping")
        values.update({("env." if name.endswith("env") else "") + str(k): v for k, v in entries.items()})
    return redact_values(values)


def recommendation(
    directory: Path,
    best: TrialScore | None,
    baseline: TrialScore | None,
    report: TrialReport | None,
    provisional: bool = False,
) -> str:
    heading = "Candidate configuration" if provisional else "Best observed settings"
    if best is None or report is None:
        return f"<section id='recommendation'><h2>{heading}</h2><p>Unavailable.</p></section>"
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (name, *pair)) + "</tr>"
        for name, pair in changes(best, baseline).items()
    )
    changed = _table(("Changed setting", "Baseline", "Recommended"), rows) if rows else "<p>No settings changed.</p>"
    if baseline is None:
        changed = "<p>Baseline settings unavailable; a change-only comparison cannot be established.</p>"
    try:
        manifest = accepted_manifest(directory, best.trial_id, report.execution)
        command = next((item for item in reversed(commands(manifest)) if item.get("kind") == "vllm"), None)
        if command is None:
            raise ValueError("No stored vLLM command")
        argv = command.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError("Invalid stored command")
        raw_env = command.get("environment", {})
        if not isinstance(raw_env, Mapping):
            raise ValueError("Invalid stored environment")
        env = redact_environment({str(k): str(v) for k, v in raw_env.items()})
        launch = render_command({**command, "argv": redact_arguments(argv), "environment": env})
        resolved = resolved_yaml(manifest, argv)
        reproduction = (
            "<h3>Full resolved vLLM YAML</h3><p>Experiment settings, snapshotted external settings, and runtime host/port; "
            "unspecified vLLM internal defaults are not captured. Redacted values require replacement.</p>"
            + copy_block("resolved-yaml", resolved)
            + "<h3>Selected environment variables</h3>"
            + copy_block("resolved-env", yaml.safe_dump(env, sort_keys=True))
            + "<h3>Launch command (POSIX shell)</h3>"
            + copy_block("launch-command", launch)
        )
    except (ValueError, TypeError, yaml.YAMLError) as error:
        reproduction = f"<p>Reproduction unavailable: {escape(str(error))}</p>"
    return (
        f"<section id='recommendation'><h2>{heading}</h2>"
        f"<p><strong>{escape(best.trial_id)}</strong> · Best observed eligible configuration.</p>"
        + changed
        + (
            "<button type='button' data-copy='launch-command'>Copy launch command</button>"
            "<span class='copy-status' role='status'></span>"
            if "id='launch-command'" in reproduction
            else ""
        )
        + "<details><summary>Full configuration and launch command</summary>"
        + reproduction
        + "</details>"
        + "</section>"
    )


def resolved_yaml(manifest: Mapping[str, object], argv: list[str]) -> str:
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping) or not manifest.get("model_path"):
        raise ValueError("Resolved settings unavailable in this manifest")
    fixed, selected = parameters.get("fixed_args"), parameters.get("selected_args")
    if not isinstance(fixed, Mapping) or not isinstance(selected, Mapping):
        raise ValueError("Resolved argument mappings unavailable")
    external = manifest.get("external_config", {})
    if not isinstance(external, Mapping):
        raise ValueError("Invalid external configuration snapshot")
    inherited = external.get("settings", {})
    if not isinstance(inherited, Mapping):
        raise ValueError("Invalid external configuration settings")
    resolved = {str(k).removeprefix("--").replace("_", "-"): v for k, v in inherited.items()}
    explicit = {str(k).removeprefix("--").replace("_", "-"): v for k, v in {**fixed, **selected}.items()}
    if inherited:
        explicit.pop("config", None)
    resolved.update(explicit)
    resolved["model"] = manifest["model_path"]
    for index, token in enumerate(argv[:-1]):
        if token in {"--host", "--port"}:
            resolved[token[2:]] = int(argv[index + 1]) if token == "--port" else argv[index + 1]
    return yaml.safe_dump(redact_values(resolved), sort_keys=True)


def copy_block(identifier: str, value: str) -> str:
    return (
        f"<button type='button' data-copy='{identifier}'>Copy</button>"
        f"<pre id='{identifier}'>{escape(value)}</pre><span class='copy-status' role='status'></span>"
    )

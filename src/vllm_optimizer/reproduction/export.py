"""Render a stored vLLM command for a POSIX shell."""

from collections.abc import Mapping
from pathlib import Path

from vllm_optimizer.reproduction.accepted import accepted_manifest
from vllm_optimizer.reproduction.reader import commands, render_command


def export_vllm_command(run: Path, trial_id: str, execution: Mapping[str, object] | None = None) -> str:
    document = accepted_manifest(run, trial_id, execution)
    command = next((item for item in reversed(commands(document)) if item.get("kind") == "vllm"), None)
    if command is None:
        raise ValueError(f"Trial '{trial_id}' has no stored vLLM command")
    return render_command(command)

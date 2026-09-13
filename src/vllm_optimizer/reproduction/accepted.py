"""Resolve the execution selected by a run without falling back to stale artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from vllm_optimizer.reproduction.reader import load_manifest


def accepted_manifest(run: Path, trial_id: str, execution: Mapping[str, object] | None = None) -> dict[str, object]:
    if execution is None:
        path = Path(run) / "result.json"
        if path.exists():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                summaries = document["trials"]
                matches = [item for item in summaries if item["trial_id"] == trial_id]
                if len(matches) != 1:
                    raise ValueError(f"Expected one accepted execution for trial '{trial_id}'")
                execution = matches[0].get("execution", {})
            except (OSError, UnicodeError, KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Cannot resolve accepted execution: {error}") from error
        else:
            execution = {}
    if not isinstance(execution, Mapping):
        raise ValueError("Accepted execution must be a mapping")
    subdirectory = execution.get("artifact_subdirectory")
    if subdirectory is not None and not isinstance(subdirectory, str):
        raise ValueError("Invalid accepted artifact subdirectory")
    manifest = load_manifest(run, trial_id, subdirectory)
    if execution and manifest.get("execution") != dict(execution):
        # Frozen TrialReport device lists are tuples; stored JSON uses lists.
        actual = json.dumps(manifest.get("execution"), sort_keys=True)
        expected = json.dumps(dict(execution), sort_keys=True)
        if actual != expected:
            raise ValueError(f"Accepted manifest execution mismatch: {trial_id}")
    return manifest

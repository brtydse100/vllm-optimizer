"""Capture external vLLM configuration before a trial starts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from vllm_optimizer.config.models import VTuneConfig


def snapshot_external_config(config: VTuneConfig, directory: Path) -> VTuneConfig:
    configured = config.server.get("config")
    if configured is None:
        return config
    settings = config.external_server_config
    if settings is None:
        try:
            loaded = yaml.safe_load(Path(str(configured)).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"Cannot snapshot external vLLM configuration '{configured}': {error}") from error
        if not isinstance(loaded, dict):
            raise ValueError(f"External vLLM configuration '{configured}' must be a mapping")
        settings = loaded
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "vllm-config.yaml"
    snapshot.write_text(yaml.safe_dump(dict(settings), sort_keys=True), encoding="utf-8")
    return replace(config, server={**config.server, "config": str(snapshot)})

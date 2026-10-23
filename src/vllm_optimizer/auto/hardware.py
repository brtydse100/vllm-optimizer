"""Read local model metadata without loading weights or initializing CUDA."""

import json
import os
import subprocess
from pathlib import Path

from vllm_optimizer.config.models import VTuneConfig


def context_limit(config: VTuneConfig) -> int:
    settings = {**(config.external_server_config or {}), **config.server}
    explicit = settings.get("max-model-len")
    path = Path(str(config.server["model"])) / "config.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        if explicit is None:
            raise ValueError("auto mode needs model config.json or explicit server.max-model-len") from error
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("model config.json must contain an object")
    metadata = metadata.get("text_config", metadata)
    if not isinstance(metadata, dict):
        raise ValueError("model text_config must contain an object")
    inferred = next(
        (
            metadata[key]
            for key in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length")
            if isinstance(metadata.get(key), int) and metadata[key] > 0
        ),
        None,
    )
    limit = explicit if explicit is not None else inferred
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 16:
        raise ValueError("auto mode requires a model context limit of at least 16 tokens")
    return limit


def gpu_inventory() -> dict[str, object]:
    """Record available devices; visibility/assignment is owned by existing workers."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("auto execution requires accessible NVIDIA GPUs via nvidia-smi") from error
    return {"devices": result.stdout.strip().splitlines(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

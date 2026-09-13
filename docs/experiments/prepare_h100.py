"""Resolve an H100 scenario for a user-selected local model and physical GPUs."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import yaml

from vllm_optimizer.config.loader import load_config


def gpu_ids(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated integer GPU IDs") from error
    if min(devices) < 0 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("GPU IDs must be nonnegative and unique")
    return devices


def write_experiment(document: dict[str, Any], output: Path) -> None:
    """Validate before exclusively creating a resolved YAML; never start inference."""
    body = yaml.safe_dump(document, sort_keys=False)
    with tempfile.TemporaryDirectory() as temporary:
        check = Path(temporary) / "experiment.yaml"
        check.write_text(body, encoding="utf-8")
        load_config(check)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(body)


def prepare(scenario: str, model: Path, devices: list[int], runs: Path) -> dict[str, Any]:
    model = model.expanduser().resolve()
    if not model.is_dir():
        raise ValueError(f"Model must be an existing local directory: {model}")
    if scenario != "large-model" and len(devices) < 2:
        raise ValueError(f"{scenario} requires at least two GPUs")
    template = Path(__file__).parent / "h100" / f"{scenario}.yaml"
    document = yaml.safe_load(template.read_text(encoding="utf-8"))
    document["server"]["model"] = str(model)
    document["experiment"]["output_dir"] = str(runs.expanduser().resolve())
    if scenario == "parallel-trials":
        if len(devices) > 100:
            raise ValueError("The template port range supports at most 100 workers")
        document["execution"]["max_parallel_trials"] = len(devices)
        document["execution"]["gpu_allocation"]["workers"] = [
            {"name": f"gpu-{device}", "devices": [device]} for device in devices
        ]
    else:
        document["server"]["tensor-parallel-size"] = len(devices)
        document["env"]["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("parallel-trials", "multi-gpu", "large-model"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpus", type=gpu_ids, required=True, help="Physical GPU IDs, e.g. 0,1")
    parser.add_argument("--output", type=Path, required=True, help="New resolved experiment YAML")
    parser.add_argument("--runs-dir", type=Path, default=Path("/var/tmp/vllm-optimizer-h100/runs"))
    args = parser.parse_args()
    try:
        write_experiment(prepare(args.scenario, args.model_path, args.gpus, args.runs_dir), args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Prepared {args.output}. Review it, then run: vllm-opt -c {args.output}")


if __name__ == "__main__":
    main()

"""Prepare fresh sequential baseline/candidate measurements from accepted evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from prepare_h100 import write_experiment

from vllm_optimizer.reporting.offline import regenerate_report
from vllm_optimizer.reproduction.accepted import accepted_manifest


def configuration(manifest: dict[str, Any], kind: str) -> dict[str, Any]:
    parameters = manifest["parameters"]
    return {**parameters[f"fixed_{kind}"], **parameters[f"selected_{kind}"]}


def validation_experiment(run: Path) -> dict[str, Any]:
    run = run.resolve()
    document = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if document.get("status") not in {"completed", "completed_with_failures"}:
        raise ValueError("The source experiment must be finished")
    if document.get("execution_mode") != "sequential":
        raise ValueError("Use the sequential large-model experiment for this validation helper")
    if not document.get("baseline") or not document.get("ranking"):
        raise ValueError("A scored baseline and eligible candidate are required")
    checked = regenerate_report(run)
    if checked.warnings:
        raise ValueError("Source integrity warnings must be resolved before validation")
    baseline = accepted_manifest(run, document["baseline"]["trial_id"])
    candidate = accepted_manifest(run, document["ranking"][0]["trial_id"])
    if baseline["model_path"] != candidate["model_path"] or baseline["benchmark"] != candidate["benchmark"]:
        raise ValueError("Baseline and candidate must use the same model and workloads")
    server, selected = configuration(baseline, "args"), configuration(candidate, "args")
    env, selected_env = configuration(baseline, "env"), configuration(candidate, "env")
    if server.keys() - selected.keys() or env.keys() - selected_env.keys():
        raise ValueError("Removing baseline settings requires a manually reviewed validation YAML")
    benchmark = copy.deepcopy(baseline["benchmark"])
    benchmark.update(warmup_repeats=2, repeats=8, min_repeats=8)
    for workload in benchmark["runs"]:
        workload["args"]["seed"] = int(workload["args"].get("seed", 42)) + 1
    return {
        "experiment": {"name": f"h100-validation-{run.name}", "output_dir": str(run.parents[1]), "seed": 43},
        "server": {**server, "model": baseline["model_path"]},
        "env": env,
        "tune": {key: {"values": [value]} for key, value in selected.items() if server.get(key) != value},
        "tune_env": {key: {"values": [value]} for key, value in selected_env.items() if env.get(key) != value},
        "benchmark": benchmark,
        "baseline": {"enabled": True},
        "optimization": {"maximize": document["maximize"], "sampler": "grid"},
        "analysis": {"drift_threshold": 0.05},
        "timeouts": baseline["policy"]["timeouts"],
        "execution": {"mode": "sequential", "retry": {"max_attempts": 1}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed sequential search directory")
    parser.add_argument("--output", type=Path, required=True, help="New validation YAML")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists")
        write_experiment(validation_experiment(args.run), args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Prepared {args.output}: baseline plus one candidate, eight repeats, fresh workload seeds.")


if __name__ == "__main__":
    main()

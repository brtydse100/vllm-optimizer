"""Expand auto-mode convenience syntax into the existing experiment contract."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from vllm_optimizer.auto.hardware import context_limit
from vllm_optimizer.auto.space import automatic_space
from vllm_optimizer.auto.workloads import default_workloads
from vllm_optimizer.config.models import VTuneConfig
from vllm_optimizer.objectives import objective_from_config


def prepare_auto_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    root = dict(raw)
    optimization = root.get("optimization", {})
    if not isinstance(optimization, Mapping):
        return root  # Normal loading supplies the section-specific validation error.
    mode = optimization.get("mode")
    if mode is not None and mode != "auto":
        raise ValueError("optimization.mode must be auto when supplied")
    if mode is None:
        if "calibration" in optimization:
            raise ValueError("optimization.calibration requires mode: auto")
        return root
    options = {"sampler": "tpe", **optimization}
    if options["sampler"] != "grid":
        options.setdefault("trials", 20)
    if "objective" not in options and "maximize" not in options:
        options["objective"] = {"metric": "output_tokens_per_second"}
    root["optimization"] = options
    root.setdefault("experiment", {"name": "auto-tuning", "seed": 42})
    benchmark = root.get("benchmark", {})
    if isinstance(benchmark, Mapping) and "runs" not in benchmark:
        engine = benchmark.get("engine", "vllm")
        root["benchmark"] = {**benchmark, "engine": engine, "runs": default_workloads(2048, str(engine))}
    return root


def resolve_auto_config(config: VTuneConfig) -> VTuneConfig:
    if config.optimization.get("mode") == "auto":
        from vllm_optimizer.auto.calibration import calibration_policy

        calibration_policy(config)
        limit = context_limit(config)
        benchmark = dict(config.benchmark)
        if config.auto.get("generated_benchmark"):
            benchmark["runs"] = default_workloads(limit, str(benchmark.get("engine", "vllm")))
        config = replace(config, tune=automatic_space(config, limit), benchmark=benchmark)
    objective = objective_from_config(config)
    if objective.percentile is None:
        return config
    benchmark = dict(config.benchmark)
    if benchmark.get("engine", "guidellm") == "guidellm":
        if objective.percentile not in {1, 5, 10, 25, 50, 75, 90, 95, 99}:
            raise ValueError(f"GuideLLM does not emit P{objective.percentile}; select benchmark.engine: vllm")
        return config
    runs = []
    source_runs = benchmark.get("runs", [])
    if not isinstance(source_runs, list):
        raise ValueError("benchmark.runs must be a list")
    for run in source_runs:
        if not isinstance(run, Mapping) or not isinstance(run.get("args", {}), Mapping):
            raise ValueError("each benchmark run and its args must be mappings")
        args = dict(run.get("args", {}))
        percentiles = str(args.get("metric-percentiles", "50,95,99")).split(",")
        args["metric-percentiles"] = ",".join(dict.fromkeys([*percentiles, str(objective.percentile)]))
        metrics = str(args.get("percentile-metrics", "ttft,tpot,itl,e2el")).split(",")
        args["percentile-metrics"] = ",".join(dict.fromkeys([*metrics, "ttft", "tpot", "itl", "e2el"]))
        runs.append({**run, "args": args})
    return replace(config, benchmark={**benchmark, "runs": runs})

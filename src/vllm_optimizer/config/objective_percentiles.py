"""Make each benchmark run emit the percentile the configured objective reads."""

from collections.abc import Mapping
from dataclasses import replace

from vllm_optimizer.config.arguments import canonical_argument_name
from vllm_optimizer.config.models import VTuneConfig
from vllm_optimizer.objectives import objective_from_config

GUIDELLM_PERCENTILES = frozenset({1, 5, 10, 25, 50, 75, 90, 95, 99})
_LATENCY_METRICS = ("ttft", "tpot", "itl", "e2el")


def request_objective_percentile(config: VTuneConfig) -> VTuneConfig:
    """Reject percentiles the backend cannot emit; extend vLLM Bench Serve arguments."""
    if "objective" not in config.optimization:
        return config  # Missing or legacy settings are validated by preflight.
    percentile = objective_from_config(config).percentile
    if percentile is None:
        return config
    benchmark = dict(config.benchmark)
    if benchmark.get("engine", "guidellm") == "guidellm":
        if percentile not in GUIDELLM_PERCENTILES:
            raise ValueError(f"GuideLLM does not emit P{percentile}; select benchmark.engine: vllm")
        return config
    source_runs = benchmark.get("runs", [])
    if not isinstance(source_runs, list):
        raise ValueError("benchmark.runs must be a list")
    runs = []
    for run in source_runs:
        if not isinstance(run, Mapping) or not isinstance(run.get("args", {}), Mapping):
            raise ValueError("each benchmark run and its args must be mappings")
        args = dict(run.get("args", {}))
        _extend(args, "metric-percentiles", ("50", "95", "99"), (str(percentile),))
        _extend(args, "percentile-metrics", _LATENCY_METRICS, _LATENCY_METRICS)
        runs.append({**run, "args": args})
    return replace(config, benchmark={**benchmark, "runs": runs})


def _extend(args: dict[str, object], name: str, default: tuple[str, ...], required: tuple[str, ...]) -> None:
    """Merge required items into whichever spelling the user wrote, keeping one key."""
    current: list[str] = list(default)
    for raw in tuple(args):
        key, inline, embedded = str(raw).partition("=")
        if canonical_argument_name(key) != name:
            continue
        value = args.pop(raw)
        if inline and value is not True:
            continue  # A disabled inline argument is absent.
        items = value if isinstance(value, list) else str(embedded if inline else value).split(",")
        current = [str(item).strip() for item in items]
    args[name] = ",".join(dict.fromkeys([*current, *required]))

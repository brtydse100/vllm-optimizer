"""Validate request-count evidence when reclassifying stored benchmark results."""

from collections.abc import Mapping
from pathlib import Path

from vllm_optimizer.domain.trial_report import TrialReport
from vllm_optimizer.reproduction.accepted import accepted_manifest
from vllm_optimizer.workers.completion import max_requests, reported_request_total, request_count_failure


def _request_counts_valid(
    measurements: tuple[object, ...], expected_requests: Mapping[str, int], maximum: float
) -> bool:
    for result in measurements:
        name = str(getattr(result, "run_name", ""))
        backend = str(getattr(result, "backend", "unknown"))
        expected = expected_requests.get(name)
        is_vllm = backend.lower().startswith("vllm")
        if expected is None and is_vllm:
            expected = reported_request_total(result)
        if (expected is not None or is_vllm) and request_count_failure(result, expected, backend, maximum) is not None:
            return False
    return True


def _expected_requests(run: Path, trials: tuple[TrialReport, ...]) -> dict[str, int]:
    manifest = accepted_manifest(run, trials[0].trial_id, trials[0].execution)
    benchmark = manifest.get("benchmark", {})
    runs = benchmark.get("runs", ()) if isinstance(benchmark, Mapping) else ()
    expected: dict[str, int] = {}
    for item in runs if isinstance(runs, list | tuple) else ():
        if not isinstance(item, Mapping) or not item.get("name"):
            continue
        has_limit, count = max_requests(item)
        if has_limit and count is not None:
            expected[str(item["name"])] = count
    return expected

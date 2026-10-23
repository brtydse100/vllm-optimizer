"""Reproducible generated workloads with explicit prompt and output lengths."""

from collections.abc import Mapping


def default_workloads(limit: int, engine: str, average_request: object) -> list[dict[str, object]]:
    if not isinstance(average_request, Mapping) or set(average_request) != {"input_tokens", "output_tokens"}:
        raise ValueError("benchmark.average_request requires input_tokens and output_tokens for generated data")
    inputs, outputs = average_request["input_tokens"], average_request["output_tokens"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (inputs, outputs)):
        raise ValueError("benchmark.average_request input_tokens and output_tokens must be positive integers")
    required = inputs * 3 + outputs
    if required > limit:
        raise ValueError(
            f"generated large workload requires {required} context tokens, but model limit is {limit}; "
            "reduce benchmark.average_request or choose sufficient model context"
        )
    runs: list[dict[str, object]] = []
    for name, prompt in (("small", max(1, inputs // 4)), ("medium", inputs), ("large", inputs * 3)):
        output = outputs
        if engine == "vllm":
            runs.append(
                {
                    "name": name,
                    "args": {
                        "dataset-name": "random",
                        "random-input-len": prompt,
                        "random-output-len": output,
                        "num-prompts": 256,
                        "max-concurrency": 1,
                        "request-rate": "inf",
                        "seed": 42,
                        "temperature": 0,
                        "percentile-metrics": "ttft,tpot,itl,e2el",
                        "metric-percentiles": "50,95,99",
                        "ignore-eos": True,
                    },
                }
            )
        else:
            runs.append(
                {
                    "name": name,
                    "profile": {"kind": "concurrent", "streams": 1, "max_concurrency": 1},
                    "constraints": [{"kind": "max_requests", "count": 256}],
                    "data": [{"kind": "synthetic_text", "prompt_tokens": prompt, "output_tokens": output}],
                }
            )
    return runs

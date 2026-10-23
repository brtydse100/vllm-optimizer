"""Reproducible generated workloads with explicit prompt and output lengths."""


def default_workloads(limit: int, engine: str) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for name, prompt, output, divisor in (("small", 128, 32, 8), ("medium", 512, 64, 3), ("large", 2048, 128, 1)):
        prompt = min(prompt, (limit - 4) // divisor if divisor != 1 else (limit - 4) * 3 // 4)
        output = min(output, max(1, (limit - prompt) // 2))
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

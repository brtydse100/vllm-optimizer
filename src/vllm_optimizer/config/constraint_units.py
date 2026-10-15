"""Parse performance bounds into canonical metric units without evaluating code."""

import re
from math import isfinite

METRIC_UNITS = {
    "time_to_first_token_ms": "ms",
    "time_per_output_token_ms": "ms",
    "inter_token_latency_ms": "ms",
    "end_to_end_latency_ms": "ms",
    "requests_per_second": "req/s",
    "output_tokens_per_second": "tok/s",
    "total_tokens_per_second": "tok/s",
}
_QUANTITY = re.compile(r"^\s*([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(us|ms|s|req/s|tok/s)\s*$")


def parse_bound(value: object, unit: str) -> float:
    """Numbers use canonical units; strings must declare compatible units."""
    if isinstance(value, bool):
        raise ValueError("constraint bounds must be non-negative finite quantities")
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str) and (match := _QUANTITY.fullmatch(value)):
        source = match[2]
        factors = {"us": 0.001, "ms": 1.0, "s": 1000.0} if unit == "ms" else {unit: 1.0}
        if source not in factors:
            raise ValueError(f"constraint unit '{source}' is incompatible with {unit}")
        number = float(match[1]) * factors[source]
    else:
        raise ValueError(f"constraint bound must be a number or quantity with units ({unit})")
    if not isfinite(number) or number < 0:
        raise ValueError("constraint bounds must be non-negative and finite")
    return number

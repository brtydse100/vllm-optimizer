"""Deterministic grid expansion for vLLM arguments and environment values."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from math import prod

from vllm_optimizer.config.models import VTuneConfig


@dataclass(frozen=True, slots=True)
class TrialParameters:
    trial_id: str
    server_args: Mapping[str, object]
    server_env: Mapping[str, object]


def expand_grid(config: VTuneConfig) -> tuple[TrialParameters, ...]:
    return tuple(iter_grid(config))


def iter_grid(config: VTuneConfig) -> Iterator[TrialParameters]:
    """Yield each unique configuration without materializing the Cartesian product."""
    options = [
        *(("arg", key, definition_values(value, f"tune.{key}")) for key, value in sorted(config.tune.items())),
        *(("env", key, definition_values(value, f"tune_env.{key}")) for key, value in sorted(config.tune_env.items())),
    ]
    for index, combination in enumerate(product(*(entry[2] for entry in options)), start=1):
        arguments: dict[str, object] = {}
        environment: dict[str, object] = {}
        for (kind, name, _), value in zip(options, combination, strict=True):
            (arguments if kind == "arg" else environment)[name] = value
        yield TrialParameters(f"trial-{index:04d}", arguments, environment)


def space_cardinality(config: VTuneConfig) -> int:
    """Return the unique Cartesian search-space size without expanding it."""
    definitions = [
        *((value, f"tune.{key}") for key, value in sorted(config.tune.items())),
        *((value, f"tune_env.{key}") for key, value in sorted(config.tune_env.items())),
    ]
    return prod(len(definition_values(definition, label)) for definition, label in definitions)


def definition_values(definition: object, label: str) -> tuple[object, ...]:
    if not isinstance(definition, Mapping):
        raise ValueError(f"'{label}' must be a mapping")
    if set(definition) == {"values"}:
        values = definition["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"'{label}.values' must be a non-empty list")
        unique: list[object] = []
        for value in values:
            if not any(type(value) is type(existing) and value == existing for existing in unique):
                unique.append(value)
        return tuple(unique)
    if set(definition) == {"min", "max", "step"}:
        return _range(definition, label)
    raise ValueError(f"'{label}' requires either values or min/max/step")


def _range(definition: Mapping[str, object], label: str) -> tuple[object, ...]:
    try:
        start, stop = Decimal(str(definition["min"])), Decimal(str(definition["max"]))
        step = Decimal(str(definition["step"]))
    except Exception as error:
        raise ValueError(f"'{label}' range values must be numeric") from error
    if step <= 0 or start > stop:
        raise ValueError(f"'{label}' requires step > 0 and min <= max")
    values = []
    current = start
    integral = all(isinstance(definition[key], int) for key in ("min", "max", "step"))
    while current <= stop:
        values.append(int(current) if integral else float(current))
        current += step
    return tuple(values)

"""Canonical vLLM CLI argument names shared by validation paths."""

from collections.abc import Mapping


def canonical_argument_name(name: object) -> str:
    """Strip an optional CLI prefix and normalize underscores to hyphens."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("vLLM argument names must be non-empty strings")
    normalized = name.removeprefix("--").replace("_", "-")
    if not normalized:
        raise ValueError("vLLM argument names must be non-empty strings")
    return normalized


def normalized_arguments(values: Mapping[str, object], label: str) -> dict[str, object]:
    """Canonicalize one argument layer and reject ambiguous aliases."""
    normalized: dict[str, object] = {}
    sources: dict[str, str] = {}
    for raw_name, value in values.items():
        name = canonical_argument_name(raw_name)
        if name in normalized:
            raise ValueError(f"{label} contains duplicate aliases for '{name}': '{sources[name]}' and '{raw_name}'")
        normalized[name] = value
        sources[name] = raw_name
    return normalized

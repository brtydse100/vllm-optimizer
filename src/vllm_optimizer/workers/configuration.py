"""Build a safe process specification for one vLLM trial."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from vllm_optimizer.config.models import VTuneConfig
from vllm_optimizer.config.runtime import logging_level, model_path, server_port
from vllm_optimizer.domain.results import Failure, WorkerResult
from vllm_optimizer.workers.base import TrialContext
from vllm_optimizer.workers.process import ProcessSpec


def build_process_spec(
    config: VTuneConfig,
    selected_args: Mapping[str, object] | None = None,
    selected_env: Mapping[str, object] | None = None,
    runtime_args: Mapping[str, object] | None = None,
    runtime_env: Mapping[str, object] | None = None,
) -> ProcessSpec:
    """Resolve fixed and selected values into a shell-free process spec."""
    chosen_args = dict(selected_args or {})
    chosen_env = dict(selected_env or {})
    _validate_selected_keys(chosen_args, config.tune, "argument")
    _validate_selected_keys(chosen_env, config.tune_env, "environment")

    arguments = {name: value for name, value in config.server.items() if name != "model"}
    arguments.update(chosen_args)
    arguments.update(runtime_args or {})
    host, port = resolved_server_endpoint(config, chosen_args, runtime_args)
    arguments.update({"host": host, "port": port})
    argv = ["vllm", "serve", model_path(config)]
    for name in sorted(arguments):
        argv.extend(_render_argument(name, arguments[name]))

    environment = _string_environment(config.env)
    environment.update(_string_environment(chosen_env))
    environment.update(_string_environment(runtime_env or {}))
    environment["VLLM_LOGGING_LEVEL"] = logging_level(config)
    return ProcessSpec(argv=tuple(argv), env=environment)


def resolved_server_endpoint(
    config: VTuneConfig,
    selected_args: Mapping[str, object] | None = None,
    runtime_args: Mapping[str, object] | None = None,
) -> tuple[str, int]:
    values = {
        "host": config.execution.get("host", "127.0.0.1"),
        "port": server_port(config),
        **{name: value for name, value in config.server.items() if name in {"host", "port"}},
        **{name: value for name, value in (selected_args or {}).items() if name in {"host", "port"}},
        **{name: value for name, value in (runtime_args or {}).items() if name in {"host", "port"}},
    }
    host, port = values["host"], values["port"]
    if not isinstance(host, str) or not host.strip():
        raise ValueError("resolved server host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("resolved server port must be a valid integer port")
    return host, port


@dataclass(slots=True)
class ConfigurationBuilderWorker:
    """Place the resolved vLLM process specification in a trial context."""

    config: VTuneConfig
    selected_args: Mapping[str, object] = field(default_factory=dict)
    selected_env: Mapping[str, object] = field(default_factory=dict)
    runtime_args: Mapping[str, object] = field(default_factory=dict)
    runtime_env: Mapping[str, object] = field(default_factory=dict)
    name: str = "configuration_builder"

    async def execute(self, context: TrialContext) -> WorkerResult[None]:
        try:
            process_spec = build_process_spec(
                self.config, self.selected_args, self.selected_env, self.runtime_args, self.runtime_env
            )
        except (TypeError, ValueError) as error:
            return WorkerResult.failed(Failure(code="configuration_invalid", message=str(error)))
        context.values["process_spec"] = process_spec
        return WorkerResult.completed()

    async def cleanup(self, context: TrialContext) -> None:
        """Configuration construction owns no external resources."""


def _validate_selected_keys(selected: Mapping[str, object], allowed: Mapping[str, object], label: str) -> None:
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown tunable {label} key(s): {names}")


def _render_argument(name: str, value: object) -> list[str]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("vLLM argument names must be non-empty strings")
    normalized = name[2:] if name.startswith("--") else name
    flag = f"--{normalized.replace('_', '-')}"
    if value is True:
        return [flag]
    if value is False:
        return [f"--no-{normalized.replace('_', '-')}"]
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rendered: list[str] = []
        for item in value:
            if item is None or isinstance(item, (Mapping, Sequence)) and not isinstance(item, (str, bytes, bytearray)):
                raise ValueError(f"Argument '{name}' contains a non-scalar value")
            rendered.extend((flag, str(item)))
        return rendered
    if isinstance(value, Mapping):
        raise ValueError(f"Argument '{name}' must be a scalar or list")
    return [flag, str(value)]


def _string_environment(values: Mapping[str, object]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Environment variable names must be non-empty strings")
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, str):
            raise ValueError(f"Environment variable '{name}' must be a scalar")
        if value is None:
            raise ValueError(f"Environment variable '{name}' must not be null")
        environment[name] = str(value)
    return environment

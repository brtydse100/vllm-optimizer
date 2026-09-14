from pathlib import Path

import pytest

from vllm_optimizer.benchmarks.configuration import (
    configured_min_repeats,
    configured_repeats,
    configured_runs,
    configured_warmup_repeats,
)
from vllm_optimizer.benchmarks.timing import timeout_for_run
from vllm_optimizer.config.errors import ConfigValidationError, ConfigYAMLError
from vllm_optimizer.config.loader import load_config
from vllm_optimizer.workers.configuration import build_process_spec


def _config_text(timeout: str | None = "20m") -> str:
    timeout_section = f"timeouts:\n  benchmark: {timeout}\n" if timeout else ""
    return f"""experiment:
  name: public-test
server:
  model: model
benchmark:
  runs:
    - name: requests
      profile: {{kind: synchronous}}
      constraints: [{{kind: max_requests, count: 2}}]
      data: [{{kind: synthetic_text, prompt_tokens: 4, output_tokens: 2}}]
optimization:
  maximize: output_tokens_per_second
{timeout_section}"""


def test_load_config_resolves_model_path_and_run(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    source = tmp_path / "experiment.yaml"
    source.write_text(_config_text(), encoding="utf-8")

    config = load_config(source)

    assert config.server["model"] == str((tmp_path / "model").resolve())
    assert configured_runs(config)[0]["name"] == "requests"


def test_request_count_run_uses_conservative_timeout_cap(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    source = tmp_path / "experiment.yaml"
    source.write_text(_config_text(None), encoding="utf-8")

    config = load_config(source)

    assert timeout_for_run(config.benchmark["runs"][0]) == 3600.0


def test_measurement_policy_accepts_warmups_and_minimum_repeats(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    source = tmp_path / "experiment.yaml"
    source.write_text(
        _config_text().replace("benchmark:\n", "benchmark:\n  repeats: 3\n  min_repeats: 2\n  warmup_repeats: 1\n", 1),
        encoding="utf-8",
    )

    config = load_config(source)

    assert configured_min_repeats(config) == 2
    assert configured_warmup_repeats(config) == 1


def test_measurement_defaults_are_trustworthy(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    source = tmp_path / "experiment.yaml"
    source.write_text(_config_text(), encoding="utf-8")
    config = load_config(source)

    assert configured_repeats(config) == 4
    assert configured_min_repeats(config) == 4
    assert configured_warmup_repeats(config) == 0


def test_implicit_minimum_tracks_explicitly_lowered_repeats(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    for repeats in (1, 2):
        source = tmp_path / f"experiment-{repeats}.yaml"
        source.write_text(
            _config_text().replace("benchmark:\n", f"benchmark:\n  repeats: {repeats}\n", 1), encoding="utf-8"
        )
        assert configured_min_repeats(load_config(source)) == repeats


def test_native_vllm_config_supplies_model_and_preserves_nested_lmcache(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    native = tmp_path / "vllm-config.yaml"
    native.write_text(
        """model: model
port: 8100
kv-transfer-config:
  kv_connector: LMCacheConnectorV1
  kv_role: kv_both
""",
        encoding="utf-8",
    )
    source = tmp_path / "experiment.yaml"
    source.write_text(_config_text().replace("  model: model", "  config: ./vllm-config.yaml"), encoding="utf-8")

    config = load_config(source)
    spec = build_process_spec(config)

    assert config.server["config"] == str(native.resolve())
    assert config.server["model"] == str((tmp_path / "model").resolve())
    assert spec.argv[:3] == ("vllm", "serve", str((tmp_path / "model").resolve()))
    assert spec.argv[spec.argv.index("--config") + 1] == str(native.resolve())
    assert spec.argv[spec.argv.index("--port") + 1] == "8100"
    assert "--kv-transfer-config" not in spec.argv


def test_server_values_override_native_vllm_config(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "vllm.yaml").write_text("model: model\nport: 8100\ngpu-memory-utilization: 0.5\n", encoding="utf-8")
    source = tmp_path / "experiment.yaml"
    source.write_text(
        _config_text().replace("  model: model", "  config: vllm.yaml\n  port: 8200\n  gpu-memory-utilization: 0.8"),
        encoding="utf-8",
    )

    argv = build_process_spec(load_config(source)).argv

    assert argv[argv.index("--port") + 1] == "8200"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.8"


def test_legacy_inline_lmcache_configuration_is_unchanged(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    transfer = '\'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}\''
    source = tmp_path / "experiment.yaml"
    source.write_text(
        _config_text()
        .replace("  model: model", f"  model: model\n  kv-transfer-config: {transfer}")
        .replace("benchmark:\n  runs:", "env:\n  LMCACHE_CONFIG_FILE: ./lmcache.yaml\nbenchmark:\n  runs:"),
        encoding="utf-8",
    )

    spec = build_process_spec(load_config(source))

    assert spec.argv[spec.argv.index("--kv-transfer-config") + 1] == transfer.strip("'")
    assert spec.env["LMCACHE_CONFIG_FILE"] == "./lmcache.yaml"


@pytest.mark.parametrize("contents", ["- not-a-mapping\n", "model: [\n"])
def test_native_vllm_config_rejects_invalid_yaml(tmp_path: Path, contents: str) -> None:
    native = tmp_path / "vllm.yaml"
    native.write_text(contents, encoding="utf-8")
    source = tmp_path / "experiment.yaml"
    source.write_text(_config_text().replace("  model: model", "  config: vllm.yaml"), encoding="utf-8")

    with pytest.raises((ConfigValidationError, ConfigYAMLError)):
        load_config(source)


def test_native_vllm_config_requires_an_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "experiment.yaml"
    source.write_text(_config_text().replace("  model: model", "  config: missing.yaml"), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="server.config.*is not a file"):
        load_config(source)


def test_native_vllm_config_requires_a_model(tmp_path: Path) -> None:
    (tmp_path / "vllm.yaml").write_text("port: 8100\n", encoding="utf-8")
    source = tmp_path / "experiment.yaml"
    source.write_text(_config_text().replace("  model: model", "  config: vllm.yaml"), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="native vLLM config model"):
        load_config(source)

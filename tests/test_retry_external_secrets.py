import re

import pytest

from vllm_optimizer.lifecycle.retry import _fixed_arguments
from vllm_optimizer.reproduction.redaction import REDACTED


@pytest.mark.parametrize(
    ("settings", "path"),
    [
        ({"hf-token": REDACTED}, "hf-token"),
        ({"nested": {"headers": [{"Authorization": REDACTED}]}}, "nested.headers.0.Authorization"),
    ],
)
def test_retry_rejects_redacted_external_yaml_settings(settings: dict[str, object], path: str) -> None:
    snapshot = "/run/trials/trial-1/vllm-config.yaml"
    manifests = [{"external_config": {"path": snapshot, "settings": settings}}]
    selected = [{"fixed_args": {"config": snapshot, "dtype": "float16"}}]

    expected = rf"redacted external YAML setting 'external_config\.settings\.{re.escape(path)}'"
    with pytest.raises(ValueError, match=expected):
        _fixed_arguments(manifests, selected)

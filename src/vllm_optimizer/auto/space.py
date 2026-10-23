"""A bounded catalog of serving knobs, respecting fixed configuration ownership."""

from vllm_optimizer.config.arguments import normalized_arguments
from vllm_optimizer.config.models import VTuneConfig


def automatic_space(config: VTuneConfig, limit: int) -> dict[str, object]:
    fixed = normalized_arguments({**(config.external_server_config or {}), **config.server}, "fixed settings")
    selected = normalized_arguments(config.tune, "tuned settings")
    chunked = fixed.get("enable-chunked-prefill", True)
    minimum = 256 if chunked else max(256, limit)
    catalog = {
        "max-num-seqs": {"values": [8, 16, 32, 64, 128]},
        "max-num-batched-tokens": {"values": sorted({max(minimum, value) for value in (512, 2048, 4096, 8192)})},
        "gpu-memory-utilization": {"values": [0.8, 0.85, 0.9]},
    }
    return {**{key: value for key, value in catalog.items() if key not in fixed and key not in selected}, **selected}

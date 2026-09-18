# Experiment, server, and tuning

```yaml
experiment:
  name: complete-example  # Letters, numbers, underscores, and hyphens only.
  output_dir: runs        # Default: runs. Relative to the current directory.
  seed: 42                # Optional sampler seed; parallel completion order can affect TPE.

# Use model for inline settings, or replace it with config: ./vllm-config.yaml.
# Define a local model inline or in the native config. Other server
# keys are CLI overrides of native values.
server:
  model: /models/Qwen3-32B
  port: 8000
  tensor-parallel-size: 4
  dtype: bfloat16
  gpu-memory-utilization: 0.90
  max-model-len: 32768
  enforce-eager: false          # false emits --no-enforce-eager; null omits it.
  enable-prefix-caching: true   # true emits a presence-only flag.
  # served-model-name: qwen     # Scalars emit --flag value.
  # lora-modules:               # Lists repeat the flag for every item.
  #   - adapter-a=/models/a
  #   - adapter-b=/models/b

# Tunable vLLM arguments live here, never under server.
# Hyphens and underscores are both accepted in argument names.
tune:
  attention-backend:            # Categorical strings.
    values: [FLASH_ATTN, FLASHINFER]
  max-num-seqs:                 # Categorical integers.
    values: [64, 128, 256]
  enforce-eager:                # Categorical booleans.
    values: [true, false]
  max_num_batched_tokens:       # Inclusive integer range.
    min: 4096
    max: 16384
    step: 4096
  gpu-memory-utilization:       # Inclusive float range.
    min: 0.85
    max: 0.95
    step: 0.05
  # kv-cache-dtype:             # null may be tested to omit a flag.
  #   values: [auto, fp8, null]

# Fixed environment variables are inherited by every server trial.
env:
  CUDA_VISIBLE_DEVICES: "0,1,2,3"
  VLLM_LOG_STATS_INTERVAL: "5"
  # Quote numeric-looking environment values, such as "0" and "1".

# Tunable environment variables use the same values or range syntax.
tune_env:
  VLLM_USE_FLASHINFER_SAMPLER:
    values: ["0", "1"]
  # WORKER_COUNT:
  #   min: 1
  #   max: 4
  #   step: 1

```

## Native server configuration and LMCache

Use `server.config: ./vllm-config.yaml` to load native server settings. The
config path is relative to the experiment YAML; a model path inside the native
file is relative to that file. An inline `server.model` overrides the native
model and is relative to the experiment YAML. Either model path must exist.
Explicit server values, tuned values, and runtime assignments override native
settings through command-line arguments. Do not rely on unspecified internal
vLLM defaults being captured in reproduction exports.

A native vLLM YAML can preserve LMCache's nested transfer configuration:

```yaml
# vllm-config.yaml
model: /models/qwen
kv-transfer-config:
  kv_connector: LMCacheConnectorV1
  kv_role: kv_both
```

```yaml
# experiment.yaml
server:
  config: ./vllm-config.yaml
env:
  LMCACHE_CONFIG_FILE: /configs/lmcache-config.yaml
```

The legacy inline form remains supported by quoting the transfer configuration
as JSON so it is rendered as one CLI value:

```yaml
server:
  model: /models/qwen
  kv-transfer-config: '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
env:
  LMCACHE_CONFIG_FILE: /configs/lmcache-config.yaml
```

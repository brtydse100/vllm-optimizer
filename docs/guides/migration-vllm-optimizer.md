# Migrating from vtune

The project was renamed before its first stable release.

| Before | Now |
| --- | --- |
| vLLM Config Tuner | vLLM Optimizer |
| `brtydse100/vllm-config-tuner` | `brtydse100/vllm-optimizer` |
| `pip install vtune` | `pip install vllm-optimizer` |
| `import vtune` | `import vllm_optimizer` |
| `vtune --config experiment.yaml` | `vllm-opt --config experiment.yaml` |

The `vtune` import and command are deprecated compatibility aliases still
present in the current package. Use `vllm_optimizer` and `vllm-opt`; this guide
does not promise an alias removal version. New code, automation, and documentation should use the new names.

Existing run directories and YAML files do not need conversion. Reproduction
metadata retains the legacy `vtune_version` field and adds
`vllm_optimizer_version` so older readers remain usable.

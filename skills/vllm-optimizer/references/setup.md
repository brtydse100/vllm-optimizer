# Setup and environment checks

## Choose what the user needs

The package is `vllm-optimizer`, the import is `vllm_optimizer`, and the command
is `vllm-opt`. The older `vtune` aliases are deprecated.

- To build/validate YAML or inspect stored reports, install the core package.
  It works on Linux, Windows, and macOS without GPU frameworks.
- To execute experiments, use a Linux host with NVIDIA GPUs and install the
  runtime extra. Native Windows vLLM execution is unsupported. WSL2 has limited
  recorded evidence and may need environment-specific workarounds.

The inspected package requires Python 3.11 or 3.12. Its runtime dependencies are
vLLM `>=0.30,<0.31` and GuideLLM `>=0.7.3,<0.8`. These are version-specific
requirements, not a promise that arbitrary CUDA/GPU combinations work. Native
Linux L40/H100 and multi-GPU validation remain pending in the inspected release.
Check the version-specific [compatibility documentation](https://brtydse100.github.io/vllm-optimizer/reference/compatibility/)
when preparing another release.

## Inspect before installing

Determine which machine will execute the benchmark. A desktop agent can prepare
files for a separate Linux host; do not silently install a GPU runtime on the
desktop instead. Inspect an existing environment before changing it:

```bash
python --version
python -m pip show vllm-optimizer vllm guidellm torch
nvidia-smi
```

`nvidia-smi` reports the driver, GPU, memory use, and driver-supported CUDA level;
that CUDA label does not prove a matching CUDA toolkit is installed. Check GPU
availability and other users/processes. An imported package or successful
`--help` command does not prove inference works.

## Install in an isolated environment

For a new Linux/WSL execution environment, after authorization:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "vllm-optimizer[runtime]"
python -m pip check
vllm-opt --help
vllm --help
guidellm --help
```

Use `python3.11` if that is the compatible interpreter available. Inspect the
installed versions after installation; package indexes can offer a release newer
than this reference. If reproducing an older experiment, use its exact package
versions in a separate environment rather than upgrading it in place.

For offline work, replace the installation command with:

```bash
python -m pip install vllm-optimizer
```

On Windows, create the environment with `py -3.12 -m venv .venv` and invoke
`.venv\Scripts\python.exe -m pip install vllm-optimizer`; direct interpreter
invocation avoids changing PowerShell execution policy. The corresponding CLI
is `.venv\Scripts\vllm-opt.exe`.

## Confirm the model and workload

`server.model` must be an existing local model directory, with model/tokenizer
files usable by the installed vLLM. A Hugging Face repository ID is not a valid
substitute. The optimizer does not download models. An empty directory may pass
the path check but cannot serve a model. Obtain permission separately for a
download, remote code, or access to gated models when needed.

Use the same host's path conventions: a Windows model path is not a Linux path.
Keep input tokens plus requested output tokens within the model's supported
context and `max-model-len`, allowing for formatting/chat-template overhead.
Use `/v1/completions` for the small plain completion example; choose chat only
when the model/tokenizer supports the intended chat template.

Prepare YAML using the [field reference](yaml.md), then:

```bash
vllm-opt validate --config experiment.yaml
vllm-opt --config experiment.yaml
```

The first command checks configuration and command construction without starting
processes. It does not load weights, check GPU capacity, or validate every
upstream CLI option. The second runs real GPU work and requires the user's
execution authorization. Follow it with the checks in [results](results.md).

## Diagnose setup failures from the actual log

| Symptom | Check and bounded response |
| --- | --- |
| Command not found | Verify the environment's interpreter, `pip show`, and PATH. Use its CLI directly. |
| Model path rejected | Check existence on the execution host and relative-path resolution. |
| Unknown flag or unsupported model | Compare the installed vLLM version's help/docs with the configured option or architecture. |
| Insufficient free memory/OOM | Inspect other allocations, model size, context, batch limits, and cache allocation. Propose a smaller budget; do not kill other processes. |
| Startup timeout | Read `vllm.log` first. Distinguish a slow healthy load from a crash before extending the timeout. |
| Address already in use | Choose an unused `server.port` for sequential execution. Do not reuse an unrelated server. |
| WSL UVA or compiler failure | The historical WSL evidence used `VLLM_USE_V2_MODEL_RUNNER: "0"` and `VLLM_USE_FLASHINFER_SAMPLER: "0"`. Apply only for a matching error/version, not as universal defaults. |
| Missing CUDA libraries/compiler | Follow the installed vLLM release's GPU installation instructions; do not replace drivers or CUDA installations speculatively. |

For upstream compatibility, consult the [official vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
for the installed release. Preserve the failing command and diagnostic log.

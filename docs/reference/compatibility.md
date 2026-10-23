# Compatibility

vLLM Optimizer targets Python 3.11–3.12 and native Linux experiment hosts
with NVIDIA GPUs. This page records dated evidence rather than promising that
every vLLM/CUDA combination works.

The runtime extra targets stable vLLM **0.32.x** (`>=0.32,<0.33`) and GuideLLM
`>=0.8.1,<0.9`, checked against the published releases on 2026-10-23. See the
[vLLM release notes](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)
and [GuideLLM release notes](https://github.com/vllm-project/guidellm/releases/tag/v0.8.1).
This runtime pair passed the RTX 3080/WSL2 backend checks recorded below;
earlier rows retain their tested versions.
Install the current checkout on Linux with `pip install -e '.[runtime]'`.
Historical evidence and captured requirement files below retain their original
versions.

## Validation matrix

The required L40 and H100 native-Linux runs need dedicated hosts and are not
available in this workspace. They remain explicitly unvalidated:

| Date | GPU | GPUs | Mode | Driver / CUDA | Python | vLLM / GuideLLM | Model | Outcome |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| 2026-09-01 | L40 | 1 | native Linux | — | — | — | — | Not run: host unavailable |
| 2026-09-01 | L40 | 2+ | tensor parallel | — | — | — | — | Not run: host unavailable |
| 2026-09-01 | H100 | 1 | native Linux | — | — | — | — | Not run: host unavailable |
| 2026-09-01 | H100 | 2+ | tensor parallel | — | — | — | — | Not run: host unavailable |
| 2026-09-01 | RTX 3080 | 1 | WSL2 attempt | 596.36 / CUDA 13.2 | 3.12.3 | 0.28.0 / 0.7.3 | facebook/opt-125m | Failed: vLLM reports `UVA is not available` |
| 2026-09-02 | RTX 3080 | 1 | WSL2 V0 smoke | 596.36 | 3.12.3 | 0.19.0 / 0.7.3 | local OPT-125M | Passed: two repeats/backend, 10/10 requests/backend, clean drains |
| 2026-09-13 | RTX 3080 | 1 | WSL2 showcase and baseline control | 616.64 / CUDA 13.0 | 3.12.3 | 0.28.0 / 0.7.3 installed | facebook/opt-125m | Completed: published vLLM Bench Serve evidence; showcase drift remains inconclusive |
| 2026-10-04 | RTX 3080 | 1 | WSL2 temperature smoke | 616.92 / CUDA 13.0 | 3.12.3 | 0.30.0 / 0.7.4 | local OPT-125M | Passed: default zero, explicit zero, nonzero startup warnings, 6/6 vLLM and 4/4 GuideLLM requests, JSON/CSV/HTML, clean shutdown |
| 2026-10-23 | RTX 3080 | 1 | WSL2 runtime upgrade | 616.92 / CUDA 13.0 | 3.12.3 | 0.32.0 / 0.8.1 | local OPT-125M FP16 | Passed: 64 measured requests/backend, temperature 0, P99 objective, offline reporting/reclassification/retries, clean shutdown; short-context compatibility check |

The RTX 3080 rows are environment diagnostics, not supported native-Linux
validation. The September 1 attempt produced no accepted benchmark result;
the later September 13 runs did. See the [published results and provenance](../results/rtx3080/index.md).
The V0
smoke verified streaming, final counters, request totals, reports, cleanup, and
offline reclassification, but used a vLLM version outside the supported range.

The matrix still needs evidence for explicit ports, interruption cleanup, long
generation, and at least one multi-GPU run on each supported native-Linux GPU.

Both GuideLLM 0.7.3 and `vllm bench serve` from vLLM 0.28.0 are the versions
used by the reproducibility procedure. Workload arguments are passed through,
so consult the documentation for the installed engine version when adopting
newer options.

The published WSL2 experiments used these environment workarounds:

```yaml
env:
  VLLM_USE_V2_MODEL_RUNNER: "0"
  VLLM_USE_FLASHINFER_SAMPLER: "0"
```

The first disables a runner that required unavailable unified virtual
addressing; the second avoids a sampler requiring a local CUDA compiler.
Native Linux systems may not need either setting.

## CUDA compatibility

vLLM Optimizer does not bundle CUDA and does not require one exact CUDA release. The
working combination is determined by vLLM, its PyTorch wheel, the NVIDIA
driver, the GPU architecture, and any locally compiled kernels. Therefore,
vLLM Optimizer cannot promise support for every CUDA version.

Follow the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
for the vLLM release you install. Also check NVIDIA's
[CUDA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html):
newer CUDA toolkits require a sufficiently recent driver, and some PTX or new
features require a newer driver even within a compatible major family.

The `py3-none-any` wheel is one universal pure-Python artifact for Linux,
Windows, and macOS. It is not three OS-specific binaries. Native vLLM execution
is unsupported on Windows; configuration and stored-result inspection work.

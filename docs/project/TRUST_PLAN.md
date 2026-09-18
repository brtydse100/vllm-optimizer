# Trust plan

Current status: software checks and published single-GPU WSL2 evidence are
available; native-Linux L40/H100 and multi-GPU validation remain pending.

## Software gates completed

- Request-count runs must account for their expected requests and meet the configured
  failure tolerance; missing or inconsistent totals cannot be ranked.
- Explicit benchmark timeouts remain authoritative; duration-based limits are derived conservatively.
- Server drain evidence is required and persisted before a trial completes.
- Public synthetic tests cover configuration, execution, scheduling, retry, process ownership,
  crash finalization, redaction, scoring, reporting, and offline workflows.
- The failure percentage is an eligibility gate. The configured maximize metric is the primary
  objective everywhere; request failure rate and count are deterministic tie-breakers.
- Fixed-repeat defaults use no warmup and four measurements; adaptive search can stop
  weak candidates after its initial budget. Fresh finalist validation uses a fixed budget.
  Reports distinguish uncertainty, drift, and incomplete evidence.
- Coordinator failures atomically finalize recoverable runs as `failed`; keyboard interruption
  finalizes them as `interrupted`.
- Conventionally named credentials are recursively redacted from terminal output, manifests,
  run results, CSV, and HTML. Arbitrary values and raw backend files are not guaranteed secret-free.
- CI owns Ruff, formatting, type, coverage, strict documentation, wheel installation, dependency
  audit, SBOM, package metadata, and tag/version gates on Python 3.11 and 3.12.

## Hardware validation pending

The [RTX 3080 WSL2 archive](../results/rtx3080/index.md) contains real vLLM 0.28.0
measurements and provenance. These are reporting demonstrations and do not
establish native-Linux, L40/H100, multi-GPU, or production performance. The
remaining evidence needed before broader hardware claims includes:

- Native-Linux vLLM 0.28 end-to-end execution.
- L40 and H100 runs, including tensor parallel and at least one multi-GPU run.
- Real long-generation timeout and interruption cleanup.
- Raw GuideLLM versus `vllm bench serve` comparison evidence using identical requests.
- Execution of the repository GPU smoke workflow.

Record GPU, driver, CUDA, Python, vLLM, GuideLLM, model, configuration, raw artifacts, and dated
pass/fail outcomes in the [compatibility matrix](../reference/compatibility.md). Until those checks succeed, the project remains explicitly
**hardware validation pending**.

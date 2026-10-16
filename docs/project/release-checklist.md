# Release checklist

A release is ready only after all of these checks have recorded artifacts:

- Required public package CI passes on Python 3.11 and 3.12, including the
  behavioral tests and clean wheel checks.
- Repository and risk-focused coverage thresholds pass on the designated
  Python 3.12 coverage job; Python 3.11 still runs the complete test suite.
- The external private regression suite passes; its fixtures and output remain
  outside this repository.
- The real GPU smoke passes against the exact release commit, either through the
  manually dispatched [GPU workflow](https://github.com/brtydse100/vllm-optimizer/actions/workflows/gpu-smoke.yml)
  on a native-Linux self-hosted runner or through an equivalent local Linux/WSL
  run. It must complete the one-request OPT-125M serving and benchmark cycle,
  verify the generated JSON, CSV, and HTML, and confirm clean process shutdown.
  Local evidence records the commit SHA, GPU, driver, CUDA, Python, vLLM, and
  GuideLLM versions together with the command and verification results.
- The [compatibility matrix](../reference/compatibility.md) contains dated evidence for the
  target GPU, driver, CUDA, Python, vLLM, GuideLLM, model, ports, cleanup,
  long-generation, and tensor-parallel cases.
- `pip-audit`, the SBOM, package checks, and the build provenance attestation
  are retained with the release artifacts.
- The working tree contains no generated `dist/`, `build/`, `site/`, runtime
  output, credentials, or private benchmark artifacts.

A version bump merged to `main` starts the release pipeline. After software
and private-regression gates pass, it creates the tag at that exact merged SHA,
attaches the wheel/source distribution/SBOM to the GitHub release, and publishes
to PyPI. Dependency-only edits without a version change skip release work.
Existing `release: published` and immutable-tag manual dispatch remain supported.
Retries reject tags on another commit and never overwrite mismatched artifacts.

The workflow checks that `vX.Y.Z` matches package and runtime versions and that
release notes exist. Creating the release and publishing the package are jobs
in the same workflow; they do not depend on bot-created tag/release events.
Configure branch protection with an external required `private-regression`
status check and configure the `pypi` environment to require it. After that
check passes, set the protected environment variable `PRIVATE_REGRESSION_COMMIT`
to the exact release commit SHA. The repository-visible publish job verifies
that SHA before building, so missing or stale private evidence blocks release.
For a merged PR, this means validating the merged commit, updating the marker,
then rerunning the failed release workflow; evidence for the PR head alone is
insufficient. The workflow does not update the marker itself.
If release assets already exist, use **Re-run failed jobs** to reuse the original
artifacts. Rebuilding all jobs can change distribution/SBOM bytes; immutable
digest checks intentionally reject replacing different files at the same version.
Hardware evidence remains an explicit manual gate and must not be marked passed
until one of the supported GPU paths completes against the release commit.

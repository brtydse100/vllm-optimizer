# Release notes

## Unreleased

## v0.1.0a21 — Performance targets and simpler guides

- Maximize throughput under per-workload latency or throughput limits, with
  inclusive min/max bounds, percentiles, and explicit or numeric units.
- Keep target eligibility consistent across TPE feedback, baseline/finalist
  selection, terminal/JSON/CSV/HTML results, retries, and offline reports.
- Match repeated GuideLLM workloads independently of invocation metadata.
- Shorten the first-run path and link advanced configuration and report details.
- Start a gated release pipeline when a package-version bump merges to main.
- Bump package/runtime versions from `0.1.0a20` to `0.1.0a21`; the release
  attaches a wheel, source archive, and SBOM. After publication, install with
  `pip install "vllm-optimizer==0.1.0a21"` or
  `pip install "vllm-optimizer[runtime]==0.1.0a21"` on Linux/WSL.

See [release details](docs/project/releases/v0.1.0a21.md).

## v0.1.0a20 — vLLM 0.30 sampling and trustworthy benchmark feedback

_October 6, 2026_

- Updated the runtime extra to vLLM 0.30.x; GuideLLM remains compatible with 0.7.x.
- Defaulted generation benchmarks to temperature zero and added startup warnings for explicit nonzero sampling.
- Validated temperature bounds, effective extra-body overrides, and multiple-completion sampling before execution.
- Normalized prefixed, inline, and dotted benchmark arguments while preserving JSON field names and detecting duplicate aliases.
- Preserved request-completion failure codes for offline reclassification while adding benchmark log diagnostics.
- Retained confidence warnings for insufficient repeats and drifting baselines, and rejected non-finite drift thresholds.
- Matched progress tracking to CLI arguments and vLLM's disabled-by-default readiness check; omitted disabled extra-body flags.
- Bumped package and runtime versions to `0.1.0a20`. The release attaches the universal wheel, source distribution, and SBOM;
  install with `pip install "vllm-optimizer==0.1.0a20"` or `pip install "vllm-optimizer[runtime]==0.1.0a20"` on Linux/WSL.

## v0.1.0a19 — Audited runtime and reporting correctness

_September 22, 2026_

- Revalidated request completion during offline reclassification for GuideLLM and vLLM, including normalized totals and accounted requests.
- Normalized configuration layers before merging, preserved inherited external YAML values when overrides are omitted, bracketed IPv6 literals,
  and rejected embedded redaction placeholders during retries.
- Awaited parallel scheduler cleanup before finalization and rejected non-finite search bounds before range expansion.
- Preserved legacy median aggregation order while retaining workload-identity validation.
- Made winner selection and baseline comparisons consistent across terminal, JSON, and HTML output, and included baseline data in CSV exports.
- Added a real GPU smoke workflow that serves OPT-125M, runs a request through vLLM Bench Serve, verifies reports, and checks cleanup.
- Made cleanup cancellation-resistant so owned processes reach forced termination, bounded retry preflight to selected trials, rejected inline
  redaction placeholders, and aligned resolved YAML alias/null handling with launched commands.

## Earlier releases

See the [release archive](docs/project/releases/CHANGELOG_ARCHIVE.md) for older notes.
The [complete historical changelog](https://github.com/brtydse100/vllm-optimizer/blob/v0.1.0a20/CHANGELOG.md)
retains the original entries through `v0.1.0a20`.

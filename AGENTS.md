# vLLM Optimizer Project Rules

## Workflow and approval

- Before editing files, state a concise plan containing:
  - Intended changes.
  - Files expected to change.
  - Tests that will be run.
  - Any proposed subagent work.
- Wait for explicit user approval before making changes.
- Read-only inspection is allowed when needed to prepare the plan.
- Do not exceed the approved scope. Request approval for additional changes.
- Keep the user informed about current work and edited files.
- Be concise and do not exceed what the user requested.

## Subagents

- Subagents may only be used after the user approves a plan that includes them.
- Give every subagent a specific, bounded task.
- Review all subagent changes, reasoning, and test results before accepting them.
- The primary agent remains responsible for the final result.

## Engineering

- Prefer the simplest implementation that satisfies current requirements.
- Avoid speculative features, abstractions, dependencies, and infrastructure.
- Follow SOLID principles pragmatically.
- Keep responsibilities and ownership boundaries explicit.
- Files should normally remain under 150 lines and must not exceed 200 lines.
- Generated or vendored files are exempt from the line limit.
- Split files by responsibility rather than hiding complexity.

## Testing and security

- Every implementation stage and behavior must have tests.
- Every feature requires a successful benchmark on a real GPU that exercises
  the new behavior before it can be marked complete or its PR ready for review
  or merge. Synthetic tests, mocks, and CPU-only CI are not substitutes.
- Report the GPU/model/backend setup, benchmark configuration, measured results,
  and private evidence location. Keep GPU benchmark artifacts outside the repository.
- If the required GPU benchmark cannot run, stop and inform the user; do not
  claim the feature is verified or mark its PR ready.
- Public tests, fixtures, snapshots, and expected results are allowed only when
  they contain no models, secrets, GPU data, or private fixtures. Keep sensitive
  regression tests outside the repository.
- Keep the test suite in a private location outside the repository.
- Run relevant private tests after every change.
- Run broader regression checks when shared behavior is affected.
- If required tests are unavailable or cannot run, stop and inform the user.
- Never claim a change is verified when its tests did not run successfully.
- Tests do not replace source review. Inspect changes, dependencies, commands,
  and generated files for unsafe or unrelated behavior.

## Contributor experience

- Use clear names, small modules, typed boundaries, and focused responsibilities.
- Avoid unnecessary indirection.
- Document extension points and non-obvious decisions.
- A new contributor should quickly understand where to add a worker, manager,
  adapter, search strategy, or reporter.

## Documentation

- Keep documentation concise and synchronized with behavior.
- The quick start must take no more than 5–10 minutes to understand and follow.
- Examples should demonstrate the simplest supported workflow.
- Update relevant documentation when user-facing behavior changes.
- When creating a new release note, include the package version updates and
  release-artifact details, including install commands and attached artifacts.

## Release publication

- Before triggering a release, identify the exact final release commit SHA,
  including the merged SHA when applicable. A tested PR head does not validate
  a different merge commit.
- Run the required private regression tests and real GPU benchmarks against that
  SHA. Keep the evidence outside the repository and record its location.
- Only after those checks pass, set `PRIVATE_REGRESSION_COMMIT` in the protected
  `pypi` environment to that SHA and read it back to confirm an exact match.
  Complete this before creating or pushing the tag for a tag-triggered release.
- For release-on-merge, ensure the workflow waits for final merged-SHA validation
  and the marker update before evaluating the publication gate. If this cannot
  be guaranteed, report the blocker before merging the release version bump.
- Never bypass the gate or reuse an earlier release's SHA or a PR-head marker
  for a different release commit.
- Monitor the publish workflow to completion. If it fails, inspect the logs,
  resolve the cause within approved scope, and rerun the failed jobs against
  the same immutable release commit; do not move an existing tag.
- Verify the exact version on PyPI and compare the wheel and source archive
  SHA-256 hashes with the GitHub release assets before reporting release success.
  A tag or GitHub release alone is not proof of successful publication.
- If validation or publication cannot complete, report the blocker and current
  release state; do not report the release as complete.

## Completion report

After approved work, report:

- What changed.
- Which files changed.
- Which tests ran and their results.
- Any limitations, risks, or unverified behavior.

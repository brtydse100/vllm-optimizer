# CLI reference

## Run or validate an experiment

```bash
vllm-opt validate --config experiment.yaml
vllm-opt --config experiment.yaml
vllm-opt -c experiment.yaml --verbose
```

`validate` checks the YAML, local model-directory existence, runtime policy,
and command construction. It creates no run and launches no server or benchmark.
It does not load model weights, verify GPU capacity, or check whether installed
upstream tools accept every pass-through flag.

`--verbose` streams server and benchmark output and overrides `logging.level`
with `DEBUG`. It is available only for starting experiments and retrying trials.

## Retry selected trials

```bash
vllm-opt retry --run runs/NAME/RUN_ID \
  --trial trial-0001 --trial trial-0004
```

The command validates the source artifacts and creates a new linked run with
only the requested saved configurations. It does not accept `--config` or
modify the source run. To change settings, edit your YAML and start a new
experiment instead.

## Display or export reproduction commands

```bash
vllm-opt reproduce --run runs/NAME/RUN_ID --trial trial-0001
vllm-opt export --run runs/NAME/RUN_ID --trial trial-0001
```

`reproduce` displays stored server and benchmark commands. `export` prints only
the accepted execution's vLLM launch command in POSIX shell syntax. Each requires
exactly one trial. Neither executes the commands; replace redacted placeholders
before using them. `export` writes to stdout, not an `--output` file.

## Regenerate a report offline

```bash
vllm-opt report --run runs/NAME/RUN_ID
vllm-opt report --run runs/NAME/RUN_ID --output /tmp/vllm-opt-report
```

This validates stored evidence and writes `report.html`, `results.csv`, and a
validated `result.json` copy without loading a model or starting a benchmark.
The default destination is a timestamped directory under `RUN_ID/regenerated/`.
An existing destination is never overwritten.

## Reclassify stored request failures

```bash
vllm-opt reclassify --run runs/NAME/RUN_ID --max-failure-percentage 2
vllm-opt reclassify --run runs/NAME/RUN_ID --accept-any-request-failures
vllm-opt reclassify --run runs/NAME/RUN_ID \
  --max-failure-percentage 5 --output /tmp/reclassified-run
```

Choose exactly one failure-policy option. The percentage must be between 0 and
100, inclusive. Accepting any failure percentage still requires a successful
request and usable metric. This recalculates eligibility and rankings from
stored measurements without executing benchmarks or modifying the source.
The default destination is under `RUN_ID/reclassified/`; an explicit output
directory must not already exist. See [offline reports](../guides/reports.md#offline-regeneration)
for evidence and legacy aggregation rules.

## Exit codes and help

| Code | Meaning |
| --- | --- |
| 0 | Command completed; for experiments, at least one trial completed |
| 1 | Runtime/artifact error, or an experiment with no completed trials |
| 2 | Invalid CLI arguments or configuration |
| 130 | Experiment interrupted |

An experiment can return 0 with some failed trials; inspect the run status,
trial counts, and report before treating it as a fully successful experiment.

`vllm-opt --help` lists the shared parser's options. An action followed by
`--help` shows the same help, not an action-specific option subset. The examples
above show valid combinations; `--config` is for run/validate, `--output` is for
report/reclassify, and `--trial` is for retry/reproduce/export.

---
name: vllm-optimizer
description: Use when installing vllm-optimizer, choosing a vLLM serving benchmark for a workload, creating or explaining optimization YAML, or interpreting vllm-opt results, failures, and performance trade-offs.
---

# vLLM Optimizer

Help users measure and improve vLLM serving configurations with the
`vllm-optimizer` Python package and `vllm-opt` CLI. Ground recommendations in
their workload, installed version, and recorded measurements. This is an
independent community project; it is not affiliated with the vLLM project.

## Choose the relevant reference

- Installation or runtime trouble: [setup](references/setup.md).
- Workload, traffic, benchmark engine, or dataset selection:
  [benchmarks](references/benchmarks.md).
- Creating, changing, or explaining YAML: [YAML fields](references/yaml.md).
- Reports, failures, rankings, and explanations of performance:
  [results](references/results.md).

Read only the references needed for the request. Answer a field question
directly; do not force a new installation or benchmark. Use existing files and
conversation context before asking for missing information.

## Establish the environment and goal

For a new experiment, establish:

1. The inference use case: interactive chat, long-context/RAG, batch generation,
   embeddings, or another concrete workload; request route and local model path.
2. Hardware and software: execution host, GPU count/VRAM, available resources,
   Python, vLLM Optimizer, vLLM, and benchmark-backend versions.
3. Demand: representative input/output lengths, concurrency or arrival rate,
   request distribution, and any reusable prompt prefixes or real trace.
4. Success criteria: throughput metric, latency percentile/limit, permitted
   failure rate, and the user's time/GPU budget.

Ask only for facts that materially change the configuration. When a workload
is unknown, offer a labeled synthetic exploratory workload and state its limits.
Use the benchmark reference to explain why the selected load matches the goal.

These references describe the inspected `0.1.0a21` schema. Identify the installed
version before relying on its fields. For other versions, check their release
documentation and `vllm-opt validate`; do not assume features from an unreleased
branch exist. The objective is `optimization.maximize`; express latency and
throughput requirements with `optimization.constraints`. These targets filter
eligible measurements; they do not stop requests or end the search early.
Do not invent `minimize`, automatic calibration, or early-stop fields.

## Build a reviewable experiment

Start from the complete example in the YAML reference, adapting it to the
actual model and host. Preserve an existing YAML's intent and unrelated values.

- Establish explicit fixed baseline settings. Change a small, explained set of
  tunable parameters while keeping the model and benchmark workload comparable.
- Translate measured performance requirements into `optimization.constraints`.
  Distinguish those targets from GuideLLM run `constraints`, which bound work.
  Use the user's actual limits; exploratory example limits are not service SLOs.
- Describe each field the user asks about using its full path, current value,
  units/default, effect, and relevant interaction. Explain every field you add
  when the user requests a walkthrough; see the field reference for the format.
- Count search combinations, baseline work, warmups, repeats, retries, and
  finalist validation. Explain the budget before running it. Timeouts are per
  operation, not a global experiment deadline.
- Use absolute model/data/output paths when portability or working directories
  could be ambiguous. Models must already exist locally; model acquisition is
  a separate user-authorized task.
- Show the resulting YAML and rationale. Validate the actual saved file with
  `vllm-opt validate --config experiment.yaml` before GPU execution.

## Execute within the user's authorization

Installation and GPU execution require authorization covering the environment,
resource use, and workload. Reuse authorization already given; configuration
help alone is not authorization to install, download models, rent GPUs, or run a
large search. Keep the server on loopback unless remote access is requested.

The CLI starts and stops its own vLLM instances. Check available VRAM and ports;
do not stop unrelated services to make room. Run a bounded smoke experiment
before a larger search. Record the exact command, configuration, versions, and
run directory. If execution is unavailable, deliver the YAML and checks that
did run, clearly marking the experiment unverified.

Treat model files, prompts, logs, reports, and benchmark outputs as user data.
Keep them local unless the user authorizes sharing. Optional `analysis.llm_summary`
sends ranking information to an external or local API; explain that before
enabling it. Credentials belong in the user's secret mechanism, never in YAML.

## Explain the outcome

Read structured run/trial results and the accepted execution manifests as well
as the report. Follow the results reference to check failures, workload matches,
repeat budgets, drift, latency, target assessments, and the selection decision before recommending
a configuration. An exit code of zero means at least one trial completed.

Give the user a concise account of:

- What ran and whether the evidence is usable.
- What changed versus the baseline, including throughput, latency, and failures.
- Which configuration best fits their stated goals and what remains uncertain.
- Whether every performance target passed for every matched workload. If none
  passed, explain that no tested configuration met all targets; retain the
  measurements and do not recommend an infeasible throughput leader.
- Why the outcome might have occurred, with evidence for each hypothesis and
  one focused follow-up measurement when causality is unresolved.

Never turn an association in a parameter search into a proven causal explanation.
Use "best observed" when the evidence cannot establish a clear winner. A small
synthetic smoke test verifies the workflow; it does not establish production
performance or answer quality.

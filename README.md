# Reliable Task Agent

Reliable Task Agent is a compact Python runtime for tool-using agents that need durable execution state, deterministic verification, and explicit handling of recoverable side effects. Its premise is simple: **Agent output != task completion**, and **checkpoint absence != proof that an external effect did not happen**. The runtime combines checkpoint/resume, verifier-driven bounded repair, workspace-scoped tool safety, persistent traces, and a durable Effect Boundary for explicitly registered side effects.

> **Agent output != task completion.**
>
> **Checkpoint absence != proof that an external effect did not happen.**

**Checkpoint/Resume · Verifier-driven Repair · Effect Boundary · Fault Benchmark**

## Why this project exists

A tool-using agent can return a confident answer while its artifact is wrong. It can also crash after an external operation succeeds but before its checkpoint records the completed tool call. These are different failure modes:

```text
model writes artifact -> deterministic verification fails

external side effect succeeds -> process crashes -> completed result is absent
```

Reliable Task Agent treats model output as a proposal, verifies important results with deterministic code, and persists enough execution state to make recovery decisions explicit.

## Key capabilities

- **Agent runtime:** tool-calling loop, OpenAI-compatible client, Pydantic argument validation, bounded steps, and retry with exponential backoff.
- **Durable execution:** persistent traces, checkpoints, resume by `run_id`, pending-call recovery, and reuse of completed Tool Call results.
- **Verifier-driven Repair:** structured verification errors, runtime hard feedback, bounded repair attempts, and crash/resume-safe repair bookkeeping.
- **Effect Boundary:** durable effect identity, receipts, reconciliation, and fail-closed handling for explicitly registered side-effecting tools.
- **Safety and testing:** workspace-scoped file tools, path-traversal protection, deterministic verification, and failure-injection hooks.

## Architecture

```mermaid
flowchart TD
    M["Model"] --> A["Agent Loop"]
    A --> R["Tool Registry + Pydantic validation"]
    A --> C["Checkpoint / Resume"]
    A --> T["Persistent Trace"]
    R --> O["Ordinary workspace tools"]
    R --> V["Deterministic verifier"]
    V -->|"FAIL: structured hard feedback"| A
    V -->|"PASS"| S["SUCCESS"]
    R --> E["Effect Executor"]
    E --> L["Durable Effect Ledger"]
    E --> B["External business system"]
    L -->|"PREPARED: reconcile"| B
```

Ordinary tools remain compatible with the Tool Registry. Effect-managed tools must be registered explicitly and can execute only through the Effect Executor.

## Verifier-driven Repair

For verified workflows, a model's final response is not sufficient. The deterministic verifier independently recomputes expected results from source inputs and returns both the backward-compatible `errors` list and structured `error_details` such as `type`, `field`, `expected`, and `actual`.

When verification fails, the runtime:

1. Detects `verification_passed=false` from the verifier result.
2. Appends runtime-generated hard feedback containing `errors` and `error_details`.
3. Persists the incremented `repair_count`, feedback message, and handled verifier Tool Call identity together.
4. Gives the model a bounded opportunity to repair the artifact.
5. Re-verifies the repaired artifact before allowing `SUCCESS`.

`max_repair_attempts` bounds the loop. Persisted `handled_verification_tool_call_ids` make each failed verifier call consume at most one repair cycle, including a crash after the verifier result is saved but before repair bookkeeping finishes.

A real configured-model smoke test exercised the complete path:

```text
Verify FAIL
-> repair_requested
-> structured hard feedback
-> model repairs artifact
-> Verify PASS
-> SUCCESS
```

This demonstrates the implemented workflow; it is not a general self-healing guarantee.

## Durable Side-effect Recovery

A checkpoint alone cannot resolve this crash window:

```text
external side effect succeeds
-> process crashes
-> Agent checkpoint has not persisted CompletedToolCall
```

The missing checkpoint result does not prove that the external operation failed. Blindly invoking the handler again can duplicate the effect.

Effect-managed tools therefore execute through a separate durable Effect Boundary. Before calling the external handler, the runtime persists a `PREPARED` record in the Effect Ledger. The record uses a stable identity derived from `run_id + tool_call_id`, a canonical hash of validated Pydantic arguments, and a stable idempotency key.

A successful execution stores the complete serialized `ToolExecutionResult` receipt and transitions the record to `COMMITTED`. If resume finds `PREPARED`, the runtime reconciles against the business system:

- `FOUND` reconstructs the expected tool result and commits the ledger without rerunning the handler.
- `NOT_FOUND` permits the registered handler to run.
- `UNKNOWN`, including reconciliation errors, is persisted and fails closed.

`COMMITTED` and `UNKNOWN` are terminal for automatic recovery. The included SQLite `create_ticket` workload demonstrates **duplicate-safe recovery for explicitly registered, idempotent/reconcilable side effects under the implemented and tested SQLite semantics.**

## Reliability benchmark

The frozen benchmark compares three configurations:

1. **LangGraph checkpoint-only:** an experimental baseline, not recommended production LangGraph practice.
2. **LangGraph + application idempotency:** the fair baseline following documented idempotent-side-effect guidance.
3. **Reliable Task Agent Effect Boundary:** the integrated Agent Loop, Tool Registry, Effect Executor, Effect Ledger, business SQLite, and checkpoint/resume path.

The final run used CPython 3.13.14, SQLite 3.50.4, LangGraph 1.2.11, `langgraph-checkpoint` 4.2.0, and `langgraph-checkpoint-sqlite` 3.1.1. It completed 150/150 ranked trials and 10/10 separate RTA F5 trials, with no exclusions or harness failures. Every ranked cell was uniform across 10 repetitions.

| Configuration | Duplicate trials | Handler invocations | Final success | F2 handlers | F4 handlers |
|---|---:|---:|---:|---:|---:|
| LangGraph checkpoint-only¹ | 20/50 | 80 | 30/50 | 20 | 20 |
| LangGraph + application idempotency | 0/50 | 80 | 50/50 | 20 | 20 |
| RTA Effect Boundary | 0/50 | 60 | 50/50 | 10 | 10 |

¹ Checkpoint-only is an experimental baseline and is not recommended production LangGraph practice.

The application-idempotent LangGraph baseline also achieved **0/50 duplicate trials and 50/50 final success**. The observed distinction is recovery semantics and handler re-entry, not a general reliability ranking.

In each tested F2/F4 10-trial cell, RTA recorded 10 handler invocations versus 20 for the application-idempotent LangGraph baseline (-50%), while both maintained zero duplicate trials and 10/10 final success. This statement is scoped only to those tested ambiguity-window cells; the overall ranked handler counts were 60 for RTA and 80 for the application-idempotent baseline.

F1 provides important context: both systems may execute the handler again when reconciliation determines that the effect did not happen. F3 uses analogous rather than identical internal durability boundaries.

### Descriptive F5 result

F5 is separate RTA-only evidence and is not part of the LangGraph comparison. In 10/10 trials, reconciliation was unavailable, the effect transitioned to `UNKNOWN`, the Agent checkpoint failed, no final `SUCCESS` was returned, and no duplicate business effect was created.

Full tracked evidence:

- [Measured report](benchmarks/results/report.md)
- [Summary CSV](benchmarks/results/summary.csv)
- [Summary JSON](benchmarks/results/summary.json)
- [Pinned environment](benchmarks/results/environment.json)
- [Representative F2 trials](benchmarks/results/representative_trials/)

## Quick start

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create `.env` from `.env.example` and configure an OpenAI-compatible endpoint with tool calling:

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=your_openai_compatible_base_url
LLM_MODEL=your_model_name
```

Run the included wireless-link analysis demo:

```bash
uv run reliable-task-agent demo
```

The Agent reads `demo_workspace/config.json`, `experiment_notes.md`, and `results.csv`; writes `analysis_report.md`; and verifies the report against the original inputs. The experiment can fail while the Agent's analysis is correct—the verifier judges the report, not whether the measured system passed its thresholds.

Resume an interrupted run with its printed `run_id`:

```bash
uv run reliable-task-agent resume <run_id>
```

## Built-in tools

| Tool | Purpose |
|---|---|
| `calculate_shannon_capacity` | Calculate Shannon theoretical channel capacity |
| `read_text_file` | Read a text file inside the workspace boundary |
| `list_workspace_files` | Discover workspace files |
| `search_text` | Search text across workspace files |
| `analyze_csv` | Deterministically analyze CSV data |
| `write_analysis_report` | Generate a structured Markdown report |
| `verify_analysis_report` | Independently verify the generated report |

The registry validates arguments before execution and exports tool schemas for the model. Applications register effect-managed tools separately with execute and reconcile handlers.

## Tests

```bash
uv run pytest -q
```

Current complete suite:

```text
92 passed
```

Tests cover tool validation, workspace boundaries, retries, traces, checkpoints, resume, completed-result reuse, verifier-driven repair, repair crash windows, Effect Ledger transitions, reconciliation, fail-closed UNKNOWN behavior, real SQLite fault injection, and the cross-runtime benchmark harness.

## Limitations and non-goals

- Effect recovery applies only to explicitly registered tools whose handlers support the required idempotency and reconciliation contract.
- The Effect Boundary does not provide distributed transactions, rollback, compensation, or safety for arbitrary external APIs.
- It does not establish multi-worker global serialization.
- `UNKNOWN` intentionally requires operator or application-level resolution.
- Verifier-driven repair is bounded and task-specific; a successful smoke test is not a general self-healing guarantee.
- Deterministic verification is only as complete as the verifier's encoded rules.
- The benchmark uses one deterministic ticket workload, local SQLite, a scripted model client, one Windows host, and pinned dependency versions.
- F3 compares analogous, not identical, internal durability boundaries.
- The benchmark does not measure latency, throughput, concurrency, network partitions, or distributed databases.
- F5 evaluates the configured RTA fail-closed path only and has no ranked LangGraph counterpart.

## Design scope

Reliable Task Agent is an engineering reference implementation for making selected agent workflows more observable, verifiable, and recoverable. It favors explicit contracts—validated tools, deterministic verifiers, bounded repair, durable effect state, and fail-closed ambiguity—over broad claims about autonomous correctness.

The repository contains the runtime under `src/reliable_task_agent/`, deterministic tests under `tests/`, the demo inputs under `demo_workspace/`, and the benchmark plus compact evidence under `benchmarks/`.

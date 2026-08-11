# Reliable Task Agent

A recoverable and verifiable Agent harness for reliable engineering tasks.

Reliable Task Agent is a lightweight Python Agent runtime focused on three engineering problems:

**Recovery, Safety, and Verification.**

Instead of treating an LLM's final answer as task success, the runtime persists execution state, supports checkpoint-based resume, constrains tool access to a workspace, and uses deterministic verification to decide whether a task has actually succeeded.

---

## Why this project?

A basic tool-calling Agent often looks like:

```text
LLM
 ↓
Tool
 ↓
Answer
```

This works for simple demos, but engineering tasks introduce additional problems:

- What happens if the process crashes after a tool has executed?
- How do we resume without blindly restarting the whole task?
- How do we prevent tools from accessing files outside the workspace?
- How do we know the Agent's final answer is actually correct?
- How can we inspect and reproduce the execution afterwards?

Reliable Task Agent adds a reliability layer around the Agent loop:

```text
                    ┌──────────────────────┐
                    │         LLM          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Agent Loop      │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      Tool Registry       Run Trace        Checkpoint
      + Validation       Persistence        Persistence
             │                                   │
             ▼                                   ▼
         Tool Call                         Crash / Resume
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
                    Deterministic Verifier
                               │
                    ┌──────────┴──────────┐
                    │                     │
                  FAIL                  SUCCESS
```

The final `SUCCESS` is therefore not determined only by the language model.

For verified workflows, the deterministic verifier must pass.

---

## Core Features

### Agent Runtime

- Tool-calling Agent loop
- OpenAI-compatible model client
- Pydantic-based tool argument validation
- Model request retry with exponential backoff
- Maximum-step protection

### Recovery

- Persistent run trace
- Persistent checkpoint
- Resume by `run_id`
- Recovery of pending tool calls
- Reuse of checkpointed completed-tool results
- Failure injection for crash-recovery testing

### Safety

- Workspace-scoped file access
- Path traversal protection
- Restricted file-writing tools
- Default protection against overwriting existing reports
- Atomic-style file persistence using temporary files and replace

### Verification

- Deterministic CSV analysis
- Structured analysis report generation
- Deterministic report verifier
- Verifier-gated CLI success

### Engineering

- CLI interface
- Reproducible demo workspace
- Automated tests
- Fake model clients for deterministic Agent tests
- Real-model end-to-end execution

---

## Built-in Tools

The current runtime contains seven built-in tools:

| Tool | Purpose |
|---|---|
| `calculate_shannon_capacity` | Calculate Shannon theoretical channel capacity |
| `read_text_file` | Safely read a text file inside the workspace |
| `list_workspace_files` | Discover workspace files |
| `search_text` | Search text across workspace files |
| `analyze_csv` | Deterministically analyze CSV data |
| `write_analysis_report` | Generate a structured Markdown analysis report |
| `verify_analysis_report` | Independently verify the generated report |

The tool registry validates arguments before execution and converts registered tools into schemas that can be provided to the LLM.

---

## Demo: Wireless Link Reliability Analysis

The repository includes a small engineering workspace:

```text
demo_workspace/
├── config.json
├── experiment_notes.md
└── results.csv
```

The task is to evaluate whether five wireless-link experiment runs satisfy the configured requirements.

Example thresholds:

```text
throughput >= 80 Mbps
latency <= 20 ms
packet_loss <= 1.0 %
required_runs = 5
```

The Agent must not simply trust the `status` column in the CSV.

Instead, it must inspect the configuration and data, calculate the metrics, identify threshold violations, write a report, and pass deterministic verification.

For the included dataset:

```text
run_003
└── throughput violation

run_005
├── throughput violation
├── latency violation
└── packet-loss violation
```

The experiment itself therefore has an overall status of:

```text
FAIL
```

But if the Agent correctly identifies the failures and produces accurate aggregate statistics, the report verification result is:

```text
verification_passed = true
```

This distinction is intentional:

> The experiment may fail while the Agent's analysis is still correct.

---

## Verified Execution Flow

A successful demo follows this flow:

```text
Discover workspace
        ↓
Read configuration and notes
        ↓
Analyze results.csv
        ↓
Determine threshold violations
        ↓
write_analysis_report
        ↓
analysis_report.md
        ↓
verify_analysis_report
        ↓
Recompute expected results from config + CSV
        ↓
verification_passed = true
        ↓
SUCCESS
```

The verifier independently recomputes the expected status, failed runs, and aggregate metrics from the original inputs.

It does not ask the LLM whether its own answer is correct.

---

## Crash Recovery

Each run receives a unique `run_id`.

Execution state is persisted under the run directory using:

```text
runs/<run_id>/
├── trace.json
└── checkpoint.json
```

If execution is interrupted, the task can be resumed using the same `run_id`.

Checkpointed completed tool calls can reuse their stored results instead of being executed again.

The test suite also contains failure-injection scenarios that intentionally crash the Agent during execution and verify that resume continues the original run correctly.

---

## Run the Project

### 1. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### 2. Configure the model

Create a `.env` file based on `.env.example`.

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=your_openai_compatible_base_url
LLM_MODEL=your_model_name
```

The model endpoint must support OpenAI-compatible chat completions and tool calling.

### 3. Run the demo

Make sure an old generated report is not present:

```powershell
Remove-Item demo_workspace\analysis_report.md -ErrorAction SilentlyContinue
```

Then run:

```bash
uv run reliable-task-agent demo
```

Example output:

```text
Starting Reliable Task Agent demo...

VERIFICATION PASSED

SUCCESS
run_id: <run_id>
Agent answer: ...
```

The generated report will appear at:

```text
demo_workspace/analysis_report.md
```

---

## Resume an Interrupted Run

If a run is interrupted, the CLI prints its `run_id`.

Resume it with:

```bash
uv run reliable-task-agent resume <run_id>
```

The runtime loads the persisted checkpoint and trace, reconstructs the pending state, and continues the original execution.

---

## Run Tests

```bash
uv run pytest -q
```

Current V0.1 test suite:

```text
51 passed
```

The tests cover:

- tool registration and validation
- workspace path protection
- CSV analysis
- trace persistence
- checkpoint persistence
- model retries
- resume behavior
- completed-tool result reuse
- failure injection
- safe report writing
- deterministic verification
- end-to-end crash recovery and verification

---

## Project Structure

```text
reliable-task-agent/
├── demo_workspace/
│   ├── config.json
│   ├── experiment_notes.md
│   └── results.csv
│
├── src/reliable_task_agent/
│   ├── agent_loop.py
│   ├── checkpoint.py
│   ├── checkpoint_store.py
│   ├── cli.py
│   ├── model_client.py
│   ├── trace.py
│   ├── trace_store.py
│   └── tools/
│       ├── builtin.py
│       └── registry.py
│
├── tests/
│   ├── test_agent_loop.py
│   ├── test_checkpoint.py
│   ├── test_checkpoint_store.py
│   ├── test_tools.py
│   └── test_trace_store.py
│
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Design Principles

### 1. Do not trust model output as ground truth

The LLM proposes actions and conclusions.

Deterministic code verifies critical results.

### 2. Persist before assuming progress is safe

Trace and checkpoint state are continuously stored so execution can be inspected and resumed.

### 3. Restrict side effects

File tools operate inside an explicit workspace boundary.

### 4. Prefer structured tool inputs

For example, `write_analysis_report` receives structured analysis fields instead of asking the LLM to directly generate an arbitrary file.

This makes downstream verification more reliable.

### 5. Make failures observable

Retries, tool calls, results, errors, and final answers are stored in the run trace.

---

## Roadmap

V0.1 focuses on the minimum reliable Agent harness:

```text
Recovery + Safety + Verification
```

Possible future work includes:

- replay support
- evaluation runner
- fault-injection matrix
- additional deterministic verifiers
- tool approval / permission policies
- harness ablation experiments
- richer engineering-domain tools

---

## Status

**V0.1**

Core runtime and end-to-end demo are implemented.

```text
Automated tests: 51 passing
CLI: available
Checkpoint / Resume: available
Failure Injection: available
Deterministic Verification: available
Real-model Demo: verified
```

# Reliability benchmark evidence

Source commit: `4ad16e767da1bb998e0a2e9bcc670e52c28839f8`

The final run completed all 150 ranked trials and all 10 separate RTA F5 descriptive trials. No trial was excluded or rerun individually. All 16 cells were uniform across their 10 repetitions. The detailed raw counts, rates, and contributing trial IDs are in `summary.csv` and `summary.json`.

## A. Measured behavior

The ranked matrix used three configurations, faults F0 through F4, and 10 repetitions per configuration/fault cell. F1, F2, and F3 used real child-process termination. F3 is labeled analogous because the frameworks expose different internal durability boundaries.

| Configuration | Fault | Business effects per trial | Handler calls per trial | Recovery successes | Duplicate trials | Final successes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| langgraph_checkpoint_only | F0 | 1 | 1 | N/A | 0/10 | 10/10 |
| langgraph_checkpoint_only | F1 | 1 | 2 | 10/10 | 0/10 | 10/10 |
| langgraph_checkpoint_only | F2 | 2 | 2 | 10/10 | 10/10 | 0/10 |
| langgraph_checkpoint_only | F3 | 1 | 1 | 10/10 | 0/10 | 10/10 |
| langgraph_checkpoint_only | F4 | 2 | 2 | 10/10 | 10/10 | 0/10 |
| langgraph_idempotent | F0 | 1 | 1 | N/A | 0/10 | 10/10 |
| langgraph_idempotent | F1 | 1 | 2 | 10/10 | 0/10 | 10/10 |
| langgraph_idempotent | F2 | 1 | 2 | 10/10 | 0/10 | 10/10 |
| langgraph_idempotent | F3 | 1 | 1 | 10/10 | 0/10 | 10/10 |
| langgraph_idempotent | F4 | 1 | 2 | 10/10 | 0/10 | 10/10 |
| reliable_task_agent_effect_boundary | F0 | 1 | 1 | N/A | 0/10 | 10/10 |
| reliable_task_agent_effect_boundary | F1 | 1 | 2 | 10/10 | 0/10 | 10/10 |
| reliable_task_agent_effect_boundary | F2 | 1 | 1 | 10/10 | 0/10 | 10/10 |
| reliable_task_agent_effect_boundary | F3 | 1 | 1 | 10/10 | 0/10 | 10/10 |
| reliable_task_agent_effect_boundary | F4 | 1 | 1 | 10/10 | 0/10 | 10/10 |

Across the ranked trials, identity matched in 150/150 and the terminal receipt was consistent with the durable business row in 150/150. All 50 ranked RTA effects ended COMMITTED. RTA reconciliation occurred once per F1, F2, and F4 trial, and did not occur in F0 or F3.

The separate RTA F5 cell produced one business row and one handler invocation per trial. All 10 effects ended UNKNOWN, all 10 checkpoints failed, all 10 reconciliation attempts failed closed, and no F5 recovery or final task was reported as successful. F5 is descriptive and is not included in comparative rankings.

## B. Interpretation

Under the pinned local SQLite workload and process-crash scenarios, LangGraph with application-level idempotency preserved a single business effect while incomplete tasks could re-enter the handler. RTA used durable effect state and reconciliation to recover already-applied effects without re-entering the handler in the corresponding tested ambiguity windows.

The checkpoint-only configuration is an experimental baseline and is not recommended LangGraph production practice. Its F2 and F4 recoveries completed at the workflow level, but each of those trials contained two durable business rows, so they did not satisfy this benchmark's final-task-success metric.

## C. Limitations

- The workload is a single deterministic ticket insertion backed by local SQLite.
- Results cover 10 repetitions per cell on one Windows host and the pinned package versions in `environment.json`.
- F3 compares analogous, not identical, internal durability boundaries.
- The model client is scripted; no external model or API behavior is measured.
- The benchmark does not measure latency, throughput, distributed databases, network partitions, concurrent callers, or other application idempotency designs.
- F5 evaluates the configured RTA fail-closed path only and has no ranked LangGraph counterpart.

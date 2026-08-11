# Wireless Link Reliability Experiment

## Objective

Evaluate whether the wireless link satisfies the configured
throughput, latency, and packet-loss requirements.

## Analysis Rules

The acceptance thresholds in `config.json` are the source of truth.

Do not determine experiment success only from the `status` column in
`results.csv`.

The final analysis must:

- summarize the CSV results;
- identify every run that violates at least one configured threshold;
- explain which metric caused each violation;
- report the aggregate statistics;
- preserve the run IDs as evidence.

## Observations

Previous manual inspection suggested that `run_003` may contain a
throughput regression.

`run_005` may contain more serious link instability.

These notes are only hints. The final conclusion must be based on the
actual configuration and CSV data.
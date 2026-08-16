from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def aggregate_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(
            (result["configuration"], result["scenario"]), []
        ).append(result)

    summary: list[dict[str, Any]] = []
    for (configuration, scenario), trials in sorted(groups.items()):
        metrics = [trial["metrics"] for trial in trials]
        recovery_attempted = [
            item for item in metrics if item["recovery_success"] is not None
        ]
        successful = sum(item["final_task_success"] for item in metrics)
        recovery_successes = sum(
            item["recovery_success"] is True for item in metrics
        )
        duplicates = sum(item["duplicate_side_effect"] for item in metrics)
        invocations = [item["handler_invocation_count"] for item in metrics]
        uniform_signatures = {
            json.dumps(item, sort_keys=True) for item in metrics
        }
        summary.append(
            {
                "configuration": configuration,
                "scenario": scenario,
                "comparability": metrics[0]["comparability"],
                "trials": len(trials),
                "successful_trials": successful,
                "final_task_success_rate": successful / len(trials),
                "recovery_attempted_count": len(recovery_attempted),
                "recovery_success_count": recovery_successes,
                "recovery_success_rate": (
                    recovery_successes / len(recovery_attempted)
                    if recovery_attempted
                    else None
                ),
                "duplicate_side_effect_count": duplicates,
                "duplicate_side_effect_rate": duplicates / len(trials),
                "total_handler_invocations": sum(invocations),
                "min_handler_invocations": min(invocations),
                "max_handler_invocations": max(invocations),
                "business_effect_counts": [
                    item["business_effect_count"] for item in metrics
                ],
                "identity_match_count": sum(
                    item["identity_matches"] for item in metrics
                ),
                "receipt_consistent_count": sum(
                    item["receipt_consistent"] for item in metrics
                ),
                "terminal_effect_states": [
                    item["terminal_effect_state"] for item in metrics
                ],
                "reconciliation_counts": [
                    item["reconciliation_count"] for item in metrics
                ],
                "uniform": len(uniform_signatures) == 1,
                "contributing_trial_ids": [
                    trial["trial_id"] for trial in trials
                ],
            }
        )
    return summary


def write_summary(
    output_root: str | Path,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    root = Path(output_root)
    summary = aggregate_results(results)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if summary:
        with (root / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            for row in summary:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, list)
                        else value
                        for key, value in row.items()
                    }
                )
    return summary

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.aggregate import write_summary
from benchmarks.business_store import BenchmarkBusinessStore
from benchmarks.fault_protocol import wait_for_marker
from benchmarks.identity import (
    TOOL_CALL_ID,
    build_logical_action_id,
    build_run_id,
    expected_effect_identity,
    payload_hash,
)


CONFIGURATIONS = (
    "langgraph_checkpoint_only",
    "langgraph_idempotent",
    "reliable_task_agent_effect_boundary",
)
RANKED_SCENARIOS = ("F0", "F1", "F2", "F3", "F4")
KILL_SCENARIOS = frozenset({"F1", "F2", "F3", "F5"})
PAYLOAD = {
    "title": "Deterministic link outage",
    "description": "Investigate benchmark link outage.",
}
LABELS = {
    "langgraph_checkpoint_only": (
        "checkpoint-only experimental baseline, not recommended "
        "LangGraph production practice"
    ),
    "langgraph_idempotent": (
        "LangGraph with benchmark-neutral application idempotency"
    ),
    "reliable_task_agent_effect_boundary": (
        "Reliable Task Agent integrated Effect Boundary"
    ),
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(
    configuration: str,
    scenario: str,
    repetition: int,
) -> dict[str, Any]:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration: {configuration}")
    if scenario not in (*RANKED_SCENARIOS, "F5"):
        raise ValueError(f"Unknown scenario: {scenario}")
    if scenario == "F5" and configuration != CONFIGURATIONS[2]:
        raise ValueError("F5 is descriptive and RTA-only.")

    run_id = build_run_id(scenario, repetition, PAYLOAD)
    effect_id, idempotency_key = expected_effect_identity(
        run_id,
        TOOL_CALL_ID,
    )
    return {
        "schema_version": 2,
        "configuration": configuration,
        "scenario": scenario,
        "repetition": repetition,
        "trial_id": f"{configuration}/{scenario}/trial-{repetition:03d}",
        "run_id": run_id,
        "tool_call_id": TOOL_CALL_ID,
        "logical_action_id": build_logical_action_id(
            run_id,
            TOOL_CALL_ID,
        ),
        "expected_effect_id": effect_id,
        "expected_idempotency_key": idempotency_key,
        "payload": PAYLOAD,
        "label": LABELS[configuration],
        "comparability": (
            "descriptive"
            if scenario == "F5"
            else "analogous"
            if scenario == "F3"
            else "exact"
        ),
        "uses_external_model_api": False,
    }


def validate_fairness(manifests: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for manifest in manifests:
        grouped.setdefault(
            (manifest["scenario"], manifest["repetition"]), []
        ).append(manifest)

    for group in grouped.values():
        reference = group[0]
        for manifest in group[1:]:
            for field in (
                "payload",
                "run_id",
                "tool_call_id",
                "logical_action_id",
                "expected_idempotency_key",
            ):
                if manifest[field] != reference[field]:
                    raise AssertionError(
                        f"Unfair manifest field {field}: {group}"
                    )
        if any(item["uses_external_model_api"] for item in group):
            raise AssertionError("Benchmark must not use an external model API.")


def environment_data() -> dict[str, Any]:
    def version(package: str) -> str:
        return importlib.metadata.version(package)

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
        "reliable_task_agent": version("reliable-task-agent"),
        "langgraph": version("langgraph"),
        "langgraph_checkpoint_sqlite": version(
            "langgraph-checkpoint-sqlite"
        ),
    }


def run_worker(
    manifest_path: Path,
    phase: str,
) -> subprocess.Popen[str]:
    environment = os.environ.copy()
    environment["LANGGRAPH_STRICT_MSGPACK"] = "true"
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmarks.worker",
            "--manifest",
            str(manifest_path),
            "--phase",
            phase,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def finish_process(
    process: subprocess.Popen[str],
    timeout: float = 30.0,
) -> dict[str, Any]:
    stdout, stderr = process.communicate(timeout=timeout)
    return {
        "pid": process.pid,
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def kill_marked_worker(
    process: subprocess.Popen[str],
    marker_data: dict[str, Any],
) -> int:
    worker_pid = int(marker_data["pid"])
    if worker_pid == process.pid:
        process.kill()
    else:
        os.kill(worker_pid, signal.SIGTERM)
    return worker_pid


def _receipt_consistent(
    receipt: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> bool:
    return receipt is not None and receipt in rows


def run_trial(
    output_root: str | Path,
    configuration: str,
    scenario: str,
    repetition: int = 0,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    trial_directory = (
        root / configuration / scenario / f"trial-{repetition:03d}"
    )
    trial_directory.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest(configuration, scenario, repetition)
    manifest_path = trial_directory / "manifest.json"
    write_json(manifest_path, manifest)

    initial = run_worker(manifest_path, "initial")
    marker_data: dict[str, Any] | None = None
    terminated_pid: int | None = None
    if scenario in KILL_SCENARIOS:
        marker_data = wait_for_marker(
            trial_directory / "fault-marker.json", initial
        )
        terminated_pid = kill_marked_worker(initial, marker_data)
    initial_process = finish_process(initial)
    initial_process["kill_requested"] = scenario in KILL_SCENARIOS
    initial_process["terminated_pid"] = terminated_pid

    recovery_process: dict[str, Any] | None = None
    recovery_result: dict[str, Any] | None = None
    if scenario != "F0":
        recovery = run_worker(manifest_path, "recovery")
        recovery_process = finish_process(recovery)
        recovery_path = trial_directory / "worker-recovery.json"
        if recovery_path.is_file():
            recovery_result = load_json(recovery_path)

    initial_path = trial_directory / "worker-initial.json"
    initial_result = load_json(initial_path) if initial_path.is_file() else None
    terminal_result = recovery_result or initial_result

    store = BenchmarkBusinessStore(trial_directory / "business.sqlite3")
    rows = store.ticket_rows(manifest["logical_action_id"])
    handler_invocations = store.handler_invocation_count(
        manifest["logical_action_id"]
    )
    reconciliation_count = store.reconciliation_invocation_count(
        manifest["logical_action_id"]
    )
    observed_keys = sorted(
        {
            row["idempotency_key"]
            for row in rows
            if row["idempotency_key"] is not None
        }
    )
    rows_match_payload = all(
        row["logical_action_id"] == manifest["logical_action_id"]
        and row["payload_hash"] == payload_hash(manifest["payload"])
        for row in rows
    )
    identity_matches = rows_match_payload and (
        observed_keys == []
        if configuration == "langgraph_checkpoint_only"
        else observed_keys == [manifest["expected_idempotency_key"]]
    )

    receipt = (
        terminal_result.get("receipt")
        if terminal_result is not None
        else None
    )
    receipt_consistent = _receipt_consistent(receipt, rows)
    terminal_success = bool(
        terminal_result is not None
        and terminal_result.get("status") == "SUCCESS"
    )

    terminal_effect_state: str | None = None
    checkpoint_status: str | None = None
    if configuration == "reliable_task_agent_effect_boundary":
        from reliable_task_agent.checkpoint_store import CheckpointStore
        from reliable_task_agent.effects import EffectStore

        effect = EffectStore(trial_directory / "effects.sqlite3").get(
            manifest["expected_effect_id"]
        )
        terminal_effect_state = effect.state if effect else None
        checkpoint = CheckpointStore(trial_directory / "runs").load(
            manifest["run_id"]
        )
        checkpoint_status = checkpoint.status

    recovery_success = (
        None
        if scenario == "F0"
        else bool(
            recovery_process
            and recovery_process["returncode"] == 0
            and terminal_success
        )
    )
    metrics = {
        "business_effect_count": len(rows),
        "duplicate_side_effect": len(rows) > 1,
        "handler_invocation_count": handler_invocations,
        "recovery_success": recovery_success,
        "final_task_success": bool(
            terminal_success
            and len(rows) == 1
            and identity_matches
            and receipt_consistent
        ),
        "identity_matches": identity_matches,
        "comparability": manifest["comparability"],
        "terminal_effect_state": terminal_effect_state,
        "reconciliation_count": reconciliation_count,
        "receipt_consistent": receipt_consistent,
        "checkpoint_status": checkpoint_status,
        "fail_closed_correctly": bool(
            scenario == "F5"
            and terminal_effect_state == "UNKNOWN"
            and checkpoint_status == "failed"
            and recovery_process
            and recovery_process["returncode"] != 0
        ),
    }
    result = {
        "schema_version": 2,
        "trial_id": manifest["trial_id"],
        "configuration": configuration,
        "scenario": scenario,
        "repetition": repetition,
        "manifest": manifest,
        "fault_marker": marker_data,
        "initial_process": initial_process,
        "recovery_process": recovery_process,
        "initial_result": initial_result,
        "recovery_result": recovery_result,
        "business_rows": rows,
        "observed_idempotency_keys": observed_keys,
        "metrics": metrics,
    }
    write_json(trial_directory / "result.json", result)
    return result


def run_matrix(
    output_root: str | Path,
    repetitions: int,
) -> list[dict[str, Any]]:
    root = Path(output_root).resolve()
    manifests = [
        build_manifest(configuration, scenario, repetition)
        for configuration in CONFIGURATIONS
        for scenario in RANKED_SCENARIOS
        for repetition in range(repetitions)
    ]
    validate_fairness(manifests)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "environment.json", environment_data())
    results = [
        run_trial(
            root,
            manifest["configuration"],
            manifest["scenario"],
            manifest["repetition"],
        )
        for manifest in manifests
    ]
    write_summary(root, results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="benchmark_results/phase3b-development"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    results = run_matrix(args.output, args.repetitions)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

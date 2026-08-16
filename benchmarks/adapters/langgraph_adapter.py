from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.func import entrypoint, task

from benchmarks.business_store import BenchmarkBusinessStore
from benchmarks.fault_protocol import FaultMarker


def run_langgraph_worker(
    manifest: dict[str, Any],
    trial_directory: str | Path,
    worker_phase: str,
) -> dict[str, Any]:
    trial_path = Path(trial_directory)
    store = BenchmarkBusinessStore(
        trial_path / "business.sqlite3"
    )
    marker = FaultMarker(
        trial_path / "fault-marker.json",
        {
            "F1": "langgraph_before_business_write",
            "F2": "langgraph_after_business_commit",
            "F3": "langgraph_after_task_result",
        }.get(manifest["scenario"])
        if worker_phase == "initial"
        else None,
    )
    write_mode = (
        "plain"
        if manifest["configuration"]
        == "langgraph_checkpoint_only"
        else "idempotent"
    )

    connection = sqlite3.connect(
        trial_path / "langgraph-checkpoints.sqlite3",
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(connection)

    @task
    def create_ticket_task(
        payload: dict[str, str],
    ) -> dict[str, Any]:
        def before_write() -> None:
            marker.reach("langgraph_before_business_write")

        def after_write() -> None:
            marker.reach("langgraph_after_business_commit")
            if (
                manifest["scenario"] == "F4"
                and worker_phase == "initial"
            ):
                raise RuntimeError(
                    "injected ambiguous failure after business commit"
                )

        receipt = store.create_ticket(
            payload=payload,
            logical_action_id=manifest["logical_action_id"],
            idempotency_key=(
                None
                if write_mode == "plain"
                else manifest["expected_idempotency_key"]
            ),
            write_mode=write_mode,
            worker_phase=worker_phase,
            process_id=os.getpid(),
            before_write=before_write,
            after_write=after_write,
        )
        return receipt

    @entrypoint(checkpointer=checkpointer)
    def create_ticket_workflow(
        payload: dict[str, str] | None,
    ) -> dict[str, Any]:
        stable_payload = payload or manifest["payload"]
        receipt = create_ticket_task(stable_payload).result()
        marker.reach("langgraph_after_task_result")
        return {"status": "SUCCESS", "receipt": receipt}

    config = {
        "configurable": {
            "thread_id": manifest["run_id"],
        }
    }

    try:
        result = create_ticket_workflow.invoke(
            manifest["payload"]
            if worker_phase == "initial"
            else None,
            config,
        )
    finally:
        connection.close()

    return {
        "status": result["status"],
        "receipt": result["receipt"],
        "framework": "langgraph",
        "worker_phase": worker_phase,
    }

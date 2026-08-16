from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.effects import EffectStore, build_effect_identity
from reliable_task_agent.mcp_adapter import (
    MCPEffectToolPolicy,
    MCPStdioServer,
    MCPToolPolicy,
    register_mcp_effect_tools,
)
from reliable_task_agent.model_client import create_client
from reliable_task_agent.tools.registry import ToolRegistry
from reliable_task_agent.tools.tickets import CreateTicketArgs
from reliable_task_agent.trace_store import TraceStore


TASK = (
    "Create a ticket for database latency being above the SLA. "
    "Use the available tools to create exactly one ticket. "
    "After the tool succeeds, reply exactly SUCCESS."
)


def read_tickets(database: Path) -> list[dict[str, Any]]:
    if not database.exists():
        return []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT * FROM tickets ORDER BY ticket_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="smoke_runs/mcp-effect-real-model",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    attempt_directory = output_root / f"attempt-{uuid4().hex[:12]}"
    attempt_directory.mkdir(parents=True, exist_ok=False)
    business_database = attempt_directory / "business.sqlite3"
    effect_database = attempt_directory / "effects.sqlite3"
    runs_directory = attempt_directory / "runs"

    server_environment = os.environ.copy()
    for key in tuple(server_environment):
        if key.startswith("LLM_"):
            server_environment.pop(key)
    server_environment["RTA_MCP_DEMO_DB"] = str(business_database)
    server_environment["RTA_MCP_DEMO_RECONCILIATION"] = "normal"

    demo_server = Path(__file__).with_name("mcp_demo_server.py")
    server = MCPStdioServer(
        command=sys.executable,
        args=(str(demo_server),),
        env=server_environment,
        cwd=demo_server.parent,
    )
    policy = MCPToolPolicy(
        effect_tools=(
            MCPEffectToolPolicy(
                tool_name="create_ticket",
                description=(
                    "Create one ticket for an operational problem. "
                    "This tool is protected by RTA Effect Boundary."
                ),
                args_model=CreateTicketArgs,
                reconciliation_tool_name=(
                    "get_ticket_by_idempotency_key"
                ),
            ),
        )
    )
    registry = ToolRegistry()
    register_mcp_effect_tools(registry, server, policy)

    client, model = create_client()
    agent = AgentLoop(
        registry,
        client=client,
        model=model,
        max_steps=4,
        checkpoint_store=CheckpointStore(runs_directory),
        trace_store=TraceStore(runs_directory),
        effect_store=EffectStore(effect_database),
    )

    final_answer: str | None = None
    execution_error: dict[str, str] | None = None
    try:
        final_answer = agent.run(TASK)
    except Exception as exc:
        execution_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    checkpoint = agent.last_checkpoint
    trace = agent.last_trace
    selected_call = None
    if checkpoint is not None:
        selected_call = next(
            (
                call
                for call in checkpoint.completed_tool_calls.values()
                if call.tool_name == "create_ticket"
            ),
            None,
        )

    effect = None
    if checkpoint is not None and selected_call is not None:
        effect_id, _ = build_effect_identity(
            checkpoint.run_id,
            selected_call.tool_call_id,
        )
        effect = EffectStore(effect_database).get(effect_id)

    tickets = read_tickets(business_database)
    transitions = (
        [
            event.details
            for event in trace.events
            if event.event_type == "effect_transition"
        ]
        if trace is not None
        else []
    )
    completed_normally = (
        execution_error is None
        and checkpoint is not None
        and checkpoint.status == "completed"
    )
    success = bool(
        completed_normally
        and final_answer is not None
        and final_answer.strip() == "SUCCESS"
        and selected_call is not None
        and effect is not None
        and effect.state == "COMMITTED"
        and len(tickets) == 1
    )
    result = {
        "attempt_directory": str(attempt_directory),
        "run_id": checkpoint.run_id if checkpoint is not None else None,
        "model": model,
        "mcp_tool_selected": (
            selected_call.tool_name if selected_call is not None else None
        ),
        "effect_id": effect.effect_id if effect is not None else None,
        "effect_state": effect.state if effect is not None else None,
        "effect_transitions": transitions,
        "ticket": tickets[0] if len(tickets) == 1 else None,
        "ticket_count": len(tickets),
        "duplicate_side_effect": len(tickets) > 1,
        "checkpoint_status": (
            checkpoint.status if checkpoint is not None else None
        ),
        "completed_normally": completed_normally,
        "final_answer": final_answer,
        "success": success,
        "execution_error": execution_error,
    }
    result_path = attempt_directory / "smoke-result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

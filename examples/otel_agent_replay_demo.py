from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.effects import EffectStore, build_effect_identity
from reliable_task_agent.telemetry import Telemetry
from reliable_task_agent.tools.registry import ToolRegistry
from reliable_task_agent.tools.tickets import register_ticket_tool
from reliable_task_agent.trace_store import TraceStore


TASK = "Create one operational ticket, then report success."
TOOL_ARGUMENTS = json.dumps(
    {
        "title": "Database latency above SLA",
        "description": "Investigate the deterministic demo alert.",
    },
    separators=(",", ":"),
)


@dataclass(frozen=True)
class ScriptedFunction:
    name: str
    arguments: str


@dataclass(frozen=True)
class ScriptedToolCall:
    id: str
    function: ScriptedFunction
    type: str = "function"
    index: int = 0


class ScriptedMessage:
    def __init__(
        self,
        *,
        content: str | None,
        tool_calls: list[ScriptedToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, **_: Any) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant"}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "index": call.index,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return message


class ScriptedClient:
    def __init__(self) -> None:
        responses = iter(
            [
                ScriptedMessage(
                    content="",
                    tool_calls=[
                        ScriptedToolCall(
                            id="call_create_ticket",
                            function=ScriptedFunction(
                                name="create_ticket",
                                arguments=TOOL_ARGUMENTS,
                            ),
                        )
                    ],
                ),
                ScriptedMessage(content="SUCCESS"),
            ]
        )
        self.requests: list[dict[str, Any]] = []

        def create(**kwargs: Any) -> Any:
            self.requests.append(deepcopy(kwargs))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=next(responses))]
            )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:4318/v1/traces",
    )
    parser.add_argument(
        "--output-root",
        default="smoke_runs/agent-replay-interop/rta-runs",
    )
    args = parser.parse_args()

    attempt_directory = (
        Path(args.output_root).resolve()
        / f"attempt-{uuid4().hex[:12]}"
    )
    attempt_directory.mkdir(parents=True, exist_ok=False)
    runs_directory = attempt_directory / "runs"
    effect_database = attempt_directory / "effects.sqlite3"
    business_database = attempt_directory / "tickets.sqlite3"

    registry = ToolRegistry()
    tickets = register_ticket_tool(registry, business_database)
    effect_store = EffectStore(effect_database)
    telemetry = Telemetry.from_otlp_http(args.endpoint)
    agent = AgentLoop(
        registry,
        client=ScriptedClient(),
        model="scripted-local-model",
        max_steps=3,
        checkpoint_store=CheckpointStore(runs_directory),
        trace_store=TraceStore(runs_directory),
        effect_store=effect_store,
        telemetry=telemetry,
    )

    final_answer: str | None = None
    error_type: str | None = None
    try:
        final_answer = agent.run(TASK)
    except Exception as exc:
        error_type = type(exc).__name__
    finally:
        telemetry_flushed = telemetry.force_flush(timeout_millis=10_000)
        telemetry.shutdown()

    checkpoint = agent.last_checkpoint
    completed_call = (
        checkpoint.completed_tool_calls.get("call_create_ticket")
        if checkpoint is not None
        else None
    )
    effect = None
    if checkpoint is not None and completed_call is not None:
        effect_id, _ = build_effect_identity(
            checkpoint.run_id,
            completed_call.tool_call_id,
        )
        effect = effect_store.get(effect_id)

    success = bool(
        error_type is None
        and checkpoint is not None
        and checkpoint.status == "completed"
        and final_answer == "SUCCESS"
        and completed_call is not None
        and effect is not None
        and effect.state == "COMMITTED"
        and tickets.count() == 1
        and telemetry_flushed
    )
    result = {
        "run_id": checkpoint.run_id if checkpoint is not None else None,
        "model": "scripted-local-model",
        "tool_name": (
            completed_call.tool_name if completed_call is not None else None
        ),
        "effect_state": effect.state if effect is not None else None,
        "ticket_count": tickets.count(),
        "checkpoint_status": (
            checkpoint.status if checkpoint is not None else None
        ),
        "final_answer": final_answer,
        "telemetry_flushed": telemetry_flushed,
        "success": success,
        "error_type": error_type,
    }
    print(json.dumps(result, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

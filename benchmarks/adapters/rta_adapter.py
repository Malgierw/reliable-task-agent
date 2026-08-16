from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.effects import EffectStore
from reliable_task_agent.tools.builtin import build_default_registry
from reliable_task_agent.tools.tickets import CreateTicketArgs
from reliable_task_agent.trace_store import TraceStore

from benchmarks.business_store import BenchmarkBusinessStore
from benchmarks.fault_protocol import FaultMarker


@dataclass
class ScriptedFunction:
    name: str
    arguments: str


@dataclass
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
        data: dict[str, Any] = {"role": "assistant"}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
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
        return data


class ScriptedCompletions:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(deepcopy(kwargs))
        has_tool_result = any(
            message.get("role") == "tool"
            and message.get("tool_call_id")
            == self.manifest["tool_call_id"]
            for message in kwargs["messages"]
        )

        if has_tool_result:
            message = ScriptedMessage(content="SUCCESS")
        else:
            message = ScriptedMessage(
                content="",
                tool_calls=[
                    ScriptedToolCall(
                        id=self.manifest["tool_call_id"],
                        function=ScriptedFunction(
                            name="create_ticket",
                            arguments=json.dumps(
                                self.manifest["payload"],
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        ),
                    )
                ],
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


class ScriptedClient:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.completions = ScriptedCompletions(manifest)
        self.chat = SimpleNamespace(completions=self.completions)


def run_rta_worker(
    manifest: dict[str, Any],
    trial_directory: str | Path,
    worker_phase: str,
) -> dict[str, Any]:
    trial_path = Path(trial_directory)
    business_store = BenchmarkBusinessStore(
        trial_path / "business.sqlite3"
    )
    marker = FaultMarker(
        trial_path / "fault-marker.json",
        {
            "F1": "rta_before_business_write",
            "F2": "rta_after_effect_execute",
            "F3": "rta_after_effect_committed",
            "F5": "rta_after_effect_execute",
        }.get(manifest["scenario"])
        if worker_phase == "initial"
        else None,
    )
    registry = build_default_registry()

    def execute(
        args: CreateTicketArgs,
        idempotency_key: str,
    ) -> dict[str, Any]:
        def before_write() -> None:
            marker.reach("rta_before_business_write")

        def after_write() -> None:
            if (
                manifest["scenario"] == "F4"
                and worker_phase == "initial"
            ):
                raise RuntimeError(
                    "injected ambiguous failure after business commit"
                )

        return business_store.create_ticket(
            payload=args.model_dump(mode="json"),
            logical_action_id=manifest["logical_action_id"],
            idempotency_key=idempotency_key,
            write_mode="idempotent",
            worker_phase=worker_phase,
            process_id=os.getpid(),
            before_write=before_write,
            after_write=after_write,
        )

    def reconcile(
        args: CreateTicketArgs,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if (
            manifest["scenario"] == "F5"
            and worker_phase == "recovery"
        ):
            business_store.record_reconciliation_invocation(
                logical_action_id=manifest["logical_action_id"],
                worker_phase=worker_phase,
                process_id=os.getpid(),
            )
            raise RuntimeError(
                "injected reconciliation failure"
            )
        return business_store.reconcile(
            payload=args.model_dump(mode="json"),
            logical_action_id=manifest["logical_action_id"],
            idempotency_key=idempotency_key,
            worker_phase=worker_phase,
            process_id=os.getpid(),
        )

    registry.register_effect(
        name="create_ticket",
        description="Create the deterministic benchmark ticket.",
        args_model=CreateTicketArgs,
        execute=execute,
        reconcile=reconcile,
    )
    client = ScriptedClient(manifest)

    def fault_hook(stage: str) -> None:
        marker.reach(f"rta_{stage}")

    agent = AgentLoop(
        registry,
        client=client,
        model="deterministic-benchmark-client",
        max_steps=3,
        max_model_retries=0,
        checkpoint_store=CheckpointStore(trial_path / "runs"),
        trace_store=TraceStore(trial_path / "runs"),
        effect_store=EffectStore(
            trial_path / "effects.sqlite3"
        ),
        fault_hook=fault_hook,
    )

    if worker_phase == "initial":
        final_answer = agent.run(
            "Create exactly one benchmark ticket.",
            run_id=manifest["run_id"],
        )
    else:
        final_answer = agent.resume(manifest["run_id"])

    rows = business_store.ticket_rows(
        manifest["logical_action_id"]
    )
    actual_key = rows[0]["idempotency_key"] if rows else None
    if actual_key != manifest["expected_idempotency_key"]:
        raise AssertionError(
            "Benchmark expected key does not match RTA-observed key: "
            f"expected={manifest['expected_idempotency_key']}, "
            f"actual={actual_key}"
        )

    effect = EffectStore(
        trial_path / "effects.sqlite3"
    ).get(manifest["expected_effect_id"])
    completed = (
        agent.last_checkpoint.completed_tool_calls.get(
            manifest["tool_call_id"]
        )
        if agent.last_checkpoint is not None
        else None
    )
    return {
        "status": final_answer,
        "framework": "reliable_task_agent",
        "worker_phase": worker_phase,
        "actual_idempotency_key": actual_key,
        "effect_state": effect.state if effect else None,
        "receipt": (
            completed.result.get("data")
            if completed is not None
            else None
        ),
        "checkpoint_status": (
            agent.last_checkpoint.status
            if agent.last_checkpoint is not None
            else None
        ),
        "model_request_count": len(client.completions.requests),
    }

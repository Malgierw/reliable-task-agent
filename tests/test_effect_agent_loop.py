from __future__ import annotations

import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.effects import (
    EffectExecutionAmbiguousError,
    EffectSafetyError,
    EffectStateUnknownError,
    EffectStore,
    ReconciliationResult,
    build_effect_identity,
)
from reliable_task_agent.tools.builtin import build_default_registry
from reliable_task_agent.tools.tickets import (
    CreateTicketArgs,
    TicketStore,
)
from reliable_task_agent.trace_store import TraceStore


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"
    index: int = 0


class FakeMessage:
    def __init__(
        self,
        *,
        content: str | None,
        tool_calls: list[FakeToolCall] | None = None,
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


class FakeCompletions:
    def __init__(self, messages: list[FakeMessage]) -> None:
        self._messages = iter(messages)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=next(self._messages))
            ]
        )


class FakeClient:
    def __init__(self, messages: list[FakeMessage]) -> None:
        self.completions = FakeCompletions(messages)
        self.chat = SimpleNamespace(
            completions=self.completions
        )


TICKET_ARGUMENTS = (
    '{"title":"Link outage",'
    '"description":"Investigate the failed link."}'
)


def ticket_call(call_id: str = "call_ticket") -> FakeMessage:
    return FakeMessage(
        content="",
        tool_calls=[
            FakeToolCall(
                id=call_id,
                function=FakeFunction(
                    name="create_ticket",
                    arguments=TICKET_ARGUMENTS,
                ),
            )
        ],
    )


def build_registry_with_counters(
    business_database,
    *,
    unknown_reconciliation: bool = False,
    execute_exception: Exception | None = None,
    reconcile_exception: Exception | None = None,
):
    registry = build_default_registry()
    tickets = TicketStore(business_database)
    execute_calls: list[str] = []
    reconcile_calls: list[str] = []

    def execute(
        args: CreateTicketArgs,
        idempotency_key: str,
    ) -> dict[str, object]:
        execute_calls.append(idempotency_key)
        if execute_exception is not None:
            raise execute_exception
        return tickets.create(args, idempotency_key)

    def reconcile(
        args: CreateTicketArgs,
        idempotency_key: str,
    ) -> ReconciliationResult:
        reconcile_calls.append(idempotency_key)
        if reconcile_exception is not None:
            raise reconcile_exception
        if unknown_reconciliation:
            return ReconciliationResult(
                status="UNKNOWN",
                reason="business database unavailable",
            )
        return tickets.reconcile(args, idempotency_key)

    registry.register_effect(
        name="create_ticket",
        description="Create a protected SQLite ticket.",
        args_model=CreateTicketArgs,
        execute=execute,
        reconcile=reconcile,
    )
    return registry, tickets, execute_calls, reconcile_calls


def build_agent(
    *,
    registry,
    client,
    run_directory,
    ledger_path,
    fault_hook=None,
) -> AgentLoop:
    return AgentLoop(
        registry,
        client=client,
        model="fake-model",
        max_steps=4,
        checkpoint_store=CheckpointStore(run_directory),
        trace_store=TraceStore(run_directory),
        effect_store=EffectStore(ledger_path),
        fault_hook=fault_hook,
    )


def effect_record(agent: AgentLoop, ledger_path, call_id="call_ticket"):
    effect_id, _ = build_effect_identity(
        agent.last_checkpoint.run_id,
        call_id,
    )
    return EffectStore(ledger_path).get(effect_id)


def test_normal_effect_moves_prepared_to_committed(tmp_path) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    registry, tickets, executes, reconciles = (
        build_registry_with_counters(
            tmp_path / "business" / "tickets.sqlite3"
        )
    )
    agent = build_agent(
        registry=registry,
        client=FakeClient(
            [ticket_call(), FakeMessage(content="done")]
        ),
        run_directory=tmp_path / "runs",
        ledger_path=ledger,
    )

    assert agent.run("Create a ticket.") == "done"
    record = effect_record(agent, ledger)

    assert tickets.count() == 1
    assert len(executes) == 1
    assert reconciles == []
    assert record.state == "COMMITTED"
    assert record.result == (
        agent.last_checkpoint.completed_tool_calls[
            "call_ticket"
        ].result
    )
    transitions = [
        event.details
        for event in agent.last_trace.events
        if event.event_type == "effect_transition"
    ]
    assert [item["to_state"] for item in transitions] == [
        "PREPARED",
        "COMMITTED",
    ]


def test_resume_after_prepared_executes_once(tmp_path) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    runs = tmp_path / "runs"
    registry, tickets, executes, reconciles = (
        build_registry_with_counters(
            tmp_path / "business" / "tickets.sqlite3"
        )
    )

    def crash(stage: str) -> None:
        if stage == "after_effect_prepared":
            raise RuntimeError("crash after PREPARED")

    first = build_agent(
        registry=registry,
        client=FakeClient([ticket_call()]),
        run_directory=runs,
        ledger_path=ledger,
        fault_hook=crash,
    )
    with pytest.raises(RuntimeError, match="after PREPARED"):
        first.run("Create a ticket.")

    assert effect_record(first, ledger).state == "PREPARED"
    assert tickets.count() == 0

    resumed = build_agent(
        registry=registry,
        client=FakeClient([FakeMessage(content="resumed")]),
        run_directory=runs,
        ledger_path=ledger,
    )
    assert resumed.resume(first.last_checkpoint.run_id) == "resumed"
    assert tickets.count() == 1
    assert len(executes) == 1
    assert len(reconciles) == 1
    assert effect_record(resumed, ledger).state == "COMMITTED"


def test_resume_reconciles_effect_after_external_commit(
    tmp_path,
) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    runs = tmp_path / "runs"
    registry, tickets, executes, reconciles = (
        build_registry_with_counters(
            tmp_path / "business" / "tickets.sqlite3"
        )
    )

    def crash(stage: str) -> None:
        if stage == "after_effect_execute":
            raise RuntimeError("crash after external effect")

    first = build_agent(
        registry=registry,
        client=FakeClient([ticket_call()]),
        run_directory=runs,
        ledger_path=ledger,
        fault_hook=crash,
    )
    with pytest.raises(RuntimeError, match="external effect"):
        first.run("Create a ticket.")

    assert effect_record(first, ledger).state == "PREPARED"
    assert tickets.count() == 1
    assert len(executes) == 1

    resumed = build_agent(
        registry=registry,
        client=FakeClient([FakeMessage(content="reconciled")]),
        run_directory=runs,
        ledger_path=ledger,
    )
    assert resumed.resume(first.last_checkpoint.run_id) == (
        "reconciled"
    )
    assert tickets.count() == 1
    assert len(executes) == 1
    assert len(reconciles) == 1
    assert effect_record(resumed, ledger).state == "COMMITTED"


def test_resume_uses_committed_result_without_callbacks(
    tmp_path,
) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    runs = tmp_path / "runs"
    registry, tickets, executes, reconciles = (
        build_registry_with_counters(
            tmp_path / "business" / "tickets.sqlite3"
        )
    )

    def crash(stage: str) -> None:
        if stage == "after_effect_committed":
            raise RuntimeError("crash after COMMITTED")

    first = build_agent(
        registry=registry,
        client=FakeClient([ticket_call()]),
        run_directory=runs,
        ledger_path=ledger,
        fault_hook=crash,
    )
    with pytest.raises(RuntimeError, match="after COMMITTED"):
        first.run("Create a ticket.")

    committed = effect_record(first, ledger)
    assert committed.state == "COMMITTED"
    assert "call_ticket" not in (
        first.last_checkpoint.completed_tool_calls
    )

    resumed = build_agent(
        registry=registry,
        client=FakeClient([FakeMessage(content="reused")]),
        run_directory=runs,
        ledger_path=ledger,
    )
    assert resumed.resume(first.last_checkpoint.run_id) == "reused"
    assert tickets.count() == 1
    assert len(executes) == 1
    assert reconciles == []
    assert (
        resumed.last_checkpoint.completed_tool_calls[
            "call_ticket"
        ].result
        == committed.result
    )


def test_repeated_resume_keeps_one_business_effect(tmp_path) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    runs = tmp_path / "runs"
    registry, tickets, executes, reconciles = (
        build_registry_with_counters(
            tmp_path / "business" / "tickets.sqlite3"
        )
    )

    first = build_agent(
        registry=registry,
        client=FakeClient([ticket_call()]),
        run_directory=runs,
        ledger_path=ledger,
        fault_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("first crash"))
            if stage == "after_effect_execute"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="first crash"):
        first.run("Create a ticket.")

    second = build_agent(
        registry=registry,
        client=FakeClient([]),
        run_directory=runs,
        ledger_path=ledger,
        fault_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("second crash"))
            if stage == "after_effect_committed"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="second crash"):
        second.resume(first.last_checkpoint.run_id)

    third = build_agent(
        registry=registry,
        client=FakeClient([FakeMessage(content="finished")]),
        run_directory=runs,
        ledger_path=ledger,
    )
    assert third.resume(first.last_checkpoint.run_id) == "finished"
    assert tickets.count() == 1
    assert len(executes) == 1
    assert len(reconciles) == 1


def test_unknown_reconciliation_fails_closed(tmp_path) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    runs = tmp_path / "runs"
    business = tmp_path / "business" / "tickets.sqlite3"
    first_registry, tickets, _, _ = build_registry_with_counters(
        business
    )

    first = build_agent(
        registry=first_registry,
        client=FakeClient([ticket_call()]),
        run_directory=runs,
        ledger_path=ledger,
        fault_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("prepared crash"))
            if stage == "after_effect_prepared"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="prepared crash"):
        first.run("Create a ticket.")

    unknown_registry, _, executes, reconciles = (
        build_registry_with_counters(
            business,
            unknown_reconciliation=True,
        )
    )
    resumed = build_agent(
        registry=unknown_registry,
        client=FakeClient([]),
        run_directory=runs,
        ledger_path=ledger,
    )

    with pytest.raises(
        EffectStateUnknownError,
        match="business database unavailable",
    ):
        resumed.resume(first.last_checkpoint.run_id)

    assert tickets.count() == 0
    assert executes == []
    assert len(reconciles) == 1
    assert resumed.last_checkpoint.status == "failed"
    assert effect_record(resumed, ledger).state == "UNKNOWN"
    assert any(
        event.event_type == "effect_transition"
        and event.details["to_state"] == "UNKNOWN"
        for event in resumed.last_trace.events
    )
    assert resumed.last_trace.events[-1].event_type == "error"


def test_live_value_error_from_effect_handler_fails_closed(
    tmp_path,
) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    client = FakeClient(
        [ticket_call(), FakeMessage(content="must not continue")]
    )
    registry, tickets, executes, reconciles = (
        build_registry_with_counters(
            tmp_path / "business" / "tickets.sqlite3",
            execute_exception=ValueError("ambiguous handler failure"),
        )
    )
    agent = build_agent(
        registry=registry,
        client=client,
        run_directory=tmp_path / "runs",
        ledger_path=ledger,
    )

    with pytest.raises(
        EffectExecutionAmbiguousError,
        match="ValueError: ambiguous handler failure",
    ):
        agent.run("Create a ticket.")

    assert tickets.count() == 0
    assert len(executes) == 1
    assert reconciles == []
    assert effect_record(agent, ledger).state == "PREPARED"
    assert agent.last_checkpoint.status == "failed"
    assert len(client.completions.requests) == 1
    assert not any(
        message.get("role") == "tool"
        for message in agent.last_checkpoint.messages
    )
    assert agent.last_trace.events[-1].details["stage"] == (
        "effect_boundary"
    )


def test_resume_value_error_from_effect_handler_fails_closed(
    tmp_path,
) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    runs = tmp_path / "runs"
    business = tmp_path / "business" / "tickets.sqlite3"
    first_registry, tickets, _, _ = build_registry_with_counters(
        business
    )
    first = build_agent(
        registry=first_registry,
        client=FakeClient([ticket_call()]),
        run_directory=runs,
        ledger_path=ledger,
        fault_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("prepared crash"))
            if stage == "after_effect_prepared"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="prepared crash"):
        first.run("Create a ticket.")

    resumed_registry, _, executes, reconciles = (
        build_registry_with_counters(
            business,
            execute_exception=ValueError("resume handler failure"),
        )
    )
    client = FakeClient([FakeMessage(content="must not continue")])
    resumed = build_agent(
        registry=resumed_registry,
        client=client,
        run_directory=runs,
        ledger_path=ledger,
    )

    with pytest.raises(
        EffectExecutionAmbiguousError,
        match="ValueError: resume handler failure",
    ):
        resumed.resume(first.last_checkpoint.run_id)

    assert tickets.count() == 0
    assert len(reconciles) == 1
    assert len(executes) == 1
    assert effect_record(resumed, ledger).state == "PREPARED"
    assert resumed.last_checkpoint.status == "failed"
    assert client.completions.requests == []


def test_reconciler_exception_transitions_unknown_and_stops(
    tmp_path,
) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    runs = tmp_path / "runs"
    business = tmp_path / "business" / "tickets.sqlite3"
    first_registry, tickets, _, _ = build_registry_with_counters(
        business
    )
    first = build_agent(
        registry=first_registry,
        client=FakeClient([ticket_call()]),
        run_directory=runs,
        ledger_path=ledger,
        fault_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("prepared crash"))
            if stage == "after_effect_prepared"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="prepared crash"):
        first.run("Create a ticket.")

    resumed_registry, _, executes, reconciles = (
        build_registry_with_counters(
            business,
            reconcile_exception=RuntimeError("reconcile unavailable"),
        )
    )
    client = FakeClient([FakeMessage(content="must not continue")])
    resumed = build_agent(
        registry=resumed_registry,
        client=client,
        run_directory=runs,
        ledger_path=ledger,
    )

    with pytest.raises(
        EffectStateUnknownError,
        match="Reconciliation raised RuntimeError",
    ):
        resumed.resume(first.last_checkpoint.run_id)

    assert tickets.count() == 0
    assert len(reconciles) == 1
    assert executes == []
    assert effect_record(resumed, ledger).state == "UNKNOWN"
    assert resumed.last_checkpoint.status == "failed"
    assert client.completions.requests == []
    assert any(
        event.event_type == "effect_transition"
        and event.details.get("reconciliation_status") == "UNKNOWN"
        for event in resumed.last_trace.events
    )
    assert not any(
        event.event_type == "effect_transition"
        and event.details.get("reconciliation_status") == "NOT_FOUND"
        for event in resumed.last_trace.events
    )


def test_corrupt_committed_result_fails_without_callbacks(
    tmp_path,
) -> None:
    ledger = tmp_path / "runtime" / "effects.sqlite3"
    runs = tmp_path / "runs"
    registry, tickets, executes, reconciles = (
        build_registry_with_counters(
            tmp_path / "business" / "tickets.sqlite3"
        )
    )
    first = build_agent(
        registry=registry,
        client=FakeClient([ticket_call()]),
        run_directory=runs,
        ledger_path=ledger,
        fault_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("committed crash"))
            if stage == "after_effect_committed"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="committed crash"):
        first.run("Create a ticket.")

    effect_id, _ = build_effect_identity(
        first.last_checkpoint.run_id,
        "call_ticket",
    )
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "UPDATE effects SET result_json = ? WHERE effect_id = ?",
            ("{malformed", effect_id),
        )

    client = FakeClient([FakeMessage(content="must not continue")])
    resumed = build_agent(
        registry=registry,
        client=client,
        run_directory=runs,
        ledger_path=ledger,
    )

    with pytest.raises(
        EffectSafetyError,
        match="Effect ledger record is corrupt",
    ):
        resumed.resume(first.last_checkpoint.run_id)

    assert tickets.count() == 1
    assert len(executes) == 1
    assert reconciles == []
    assert resumed.last_checkpoint.status == "failed"
    assert client.completions.requests == []
    assert resumed.last_trace.events[-1].details["stage"] == (
        "effect_boundary"
    )
    with sqlite3.connect(ledger) as connection:
        state = connection.execute(
            "SELECT state FROM effects WHERE effect_id = ?",
            (effect_id,),
        ).fetchone()[0]
    assert state == "COMMITTED"

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.effects import (
    EffectExecutor,
    EffectStore,
    ReconciliationResult,
    build_effect_identity,
    hash_arguments,
)
from reliable_task_agent.mcp_adapter import (
    MCPStdioServer,
    MCPToolPolicy,
    invoke_stdio_tool,
)
from reliable_task_agent.telemetry import Telemetry
from reliable_task_agent.tools.registry import ToolRegistry


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


class FakeClient:
    def __init__(self, messages: list[FakeMessage]) -> None:
        message_iterator = iter(messages)

        def create(**_: Any) -> Any:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=next(message_iterator))]
            )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )


class EchoArgs(BaseModel):
    value: str


class EmptyArgs(BaseModel):
    pass


class EffectArgs(BaseModel):
    title: str


def telemetry_fixture() -> tuple[Telemetry, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Telemetry.from_tracer_provider(provider), exporter


def span_by_name(exporter: InMemorySpanExporter, name: str):
    return [span for span in exporter.get_finished_spans() if span.name == name]


def exported_text(exporter: InMemorySpanExporter) -> str:
    return repr(
        [
            (
                span.name,
                dict(span.attributes or {}),
                [
                    (event.name, dict(event.attributes or {}))
                    for event in span.events
                ],
            )
            for span in exporter.get_finished_spans()
        ]
    )


def test_tracing_disabled_preserves_agent_behavior() -> None:
    agent = AgentLoop(
        ToolRegistry(),
        client=FakeClient([FakeMessage(content="done")]),
        model="test-model",
    )

    assert agent.run("ordinary task") == "done"
    assert agent.last_checkpoint is not None
    assert agent.last_checkpoint.status == "completed"


def test_agent_root_llm_and_tool_spans_are_parented_and_sanitized() -> None:
    telemetry, exporter = telemetry_fixture()
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Echo a value.",
        args_model=EchoArgs,
        handler=lambda args: {"value": args.value},
    )
    secret = "sk-secret-prompt-argument-result"
    client = FakeClient(
        [
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_echo",
                        function=FakeFunction(
                            name="echo",
                            arguments=f'{{"value":"{secret}"}}',
                        ),
                    )
                ],
            ),
            FakeMessage(content=f"done {secret}"),
        ]
    )
    agent = AgentLoop(
        registry,
        client=client,
        model="test-model",
        telemetry=telemetry,
    )

    assert agent.run(f"prompt {secret}") == f"done {secret}"

    roots = span_by_name(exporter, "rta.agent.run")
    tools = span_by_name(exporter, "rta.tool.execute")
    llm_calls = span_by_name(exporter, "rta.llm.call")
    assert len(roots) == 1
    assert len(tools) == 1
    assert len(llm_calls) == 2
    assert roots[0].parent is None
    assert tools[0].parent.span_id == roots[0].context.span_id
    assert tools[0].attributes["rta.tool.name"] == "echo"
    assert tools[0].attributes["rta.tool_call.id"] == "call_echo"
    assert tools[0].attributes["rta.tool.ok"] is True
    assert secret not in exported_text(exporter)


def test_mcp_invocation_emits_sanitized_span() -> None:
    telemetry, exporter = telemetry_fixture()
    server_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "mcp_demo_server.py"
    )
    secret = "ticket-secret-credential"
    result = asyncio.run(
        invoke_stdio_tool(
            MCPStdioServer(
                command=sys.executable,
                args=(str(server_path),),
                cwd=server_path.parent,
            ),
            tool_name="get_ticket",
            arguments={"ticket_id": secret},
            policy=MCPToolPolicy(
                ordinary_tool_names=frozenset({"get_ticket"})
            ),
            telemetry=telemetry,
        )
    )

    assert result.ok is True
    spans = span_by_name(exporter, "rta.mcp.call")
    assert len(spans) == 1
    assert spans[0].attributes["rta.tool.name"] == "get_ticket"
    assert spans[0].attributes["rta.mcp.transport"] == "stdio"
    assert secret not in exported_text(exporter)


def test_effect_transitions_and_reconciliation_are_observable(tmp_path) -> None:
    telemetry, exporter = telemetry_fixture()
    store = EffectStore(tmp_path / "effects.sqlite3")
    registry = ToolRegistry()
    registry.register_effect(
        name="create_item",
        description="Create an item.",
        args_model=EffectArgs,
        execute=lambda args, key: {"title": args.title, "key": key},
        reconcile=lambda args, key: ReconciliationResult(
            status="FOUND",
            receipt={"title": args.title, "key": key},
        ),
    )
    executor = EffectExecutor(store, telemetry=telemetry)
    transitions: list[dict[str, Any]] = []

    result = executor.execute(
        run_id="run-normal",
        tool_call_id="call-normal",
        tool=registry.get("create_item"),
        arguments={"title": "normal"},
        transition_hook=transitions.append,
        fault_hook=lambda _: None,
    )

    assert result.ok is True
    effect_span = span_by_name(exporter, "rta.effect")[0]
    states = [
        event.attributes["rta.effect.to_state"]
        for event in effect_span.events
        if event.name == "rta.effect.transition"
    ]
    assert states == ["PREPARED", "COMMITTED"]
    assert effect_span.attributes["rta.effect.state"] == "COMMITTED"

    effect_id, idempotency_key = build_effect_identity(
        "run-reconcile",
        "call-reconcile",
    )
    store.prepare(
        effect_id=effect_id,
        run_id="run-reconcile",
        tool_call_id="call-reconcile",
        tool_name="create_item",
        arguments_hash=hash_arguments(EffectArgs(title="recovered")),
        idempotency_key=idempotency_key,
    )
    recovered = executor.execute(
        run_id="run-reconcile",
        tool_call_id="call-reconcile",
        tool=registry.get("create_item"),
        arguments={"title": "recovered"},
        transition_hook=transitions.append,
        fault_hook=lambda _: None,
    )

    assert recovered.ok is True
    reconciliation = span_by_name(exporter, "rta.reconciliation")
    assert len(reconciliation) == 1
    assert (
        reconciliation[0].attributes["rta.reconciliation.outcome"]
        == "FOUND"
    )


def test_verifier_and_repair_spans_are_observable() -> None:
    telemetry, exporter = telemetry_fixture()
    registry = ToolRegistry()
    outcomes = iter(
        [
            {
                "verification_passed": False,
                "errors": ["wrong count"],
                "error_details": [
                    {
                        "type": "mismatch",
                        "field": "count",
                        "expected": 1,
                        "actual": 2,
                    }
                ],
            },
            {"verification_passed": True, "errors": []},
        ]
    )
    registry.register(
        name="verify_analysis_report",
        description="Verify a report.",
        args_model=EmptyArgs,
        handler=lambda _: next(outcomes),
    )
    client = FakeClient(
        [
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="verify-1",
                        function=FakeFunction(
                            name="verify_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            ),
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="verify-2",
                        function=FakeFunction(
                            name="verify_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            ),
            FakeMessage(content="SUCCESS"),
        ]
    )
    agent = AgentLoop(
        registry,
        client=client,
        model="test-model",
        telemetry=telemetry,
    )

    assert agent.run("verify") == "SUCCESS"
    verifier_spans = span_by_name(exporter, "rta.verifier")
    repair_spans = span_by_name(exporter, "rta.repair")
    assert [
        span.attributes["rta.verifier.passed"]
        for span in verifier_spans
    ] == [False, True]
    assert len(repair_spans) == 1
    assert repair_spans[0].attributes["rta.repair.count"] == 1
    assert repair_spans[0].attributes["rta.repair.max_attempts"] == 2
    assert "wrong count" not in exported_text(exporter)


class RaisingExporter(SpanExporter):
    def __init__(self) -> None:
        self.calls = 0

    def export(self, spans):
        self.calls += 1
        raise RuntimeError("exporter failure with secret payload")

    def shutdown(self) -> None:
        return None


def test_exporter_failure_does_not_break_agent_execution() -> None:
    provider = TracerProvider()
    exporter = RaisingExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry = Telemetry.from_tracer_provider(provider)
    agent = AgentLoop(
        ToolRegistry(),
        client=FakeClient([FakeMessage(content="done")]),
        model="test-model",
        telemetry=telemetry,
    )

    assert agent.run("ordinary task") == "done"
    assert exporter.calls > 0
    assert agent.last_checkpoint is not None
    assert agent.last_checkpoint.status == "completed"

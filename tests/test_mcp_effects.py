from __future__ import annotations

import json
import os
import sqlite3
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.effects import (
    EffectStateUnknownError,
    EffectStore,
    build_effect_identity,
)
from reliable_task_agent.mcp_adapter import (
    MCPEffectToolPolicy,
    MCPStdioServer,
    MCPToolPolicy,
    register_mcp_effect_tools,
)
from reliable_task_agent.tools.registry import ToolRegistry
from reliable_task_agent.tools.tickets import CreateTicketArgs
from reliable_task_agent.trace_store import TraceStore


DEMO_SERVER = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mcp_demo_server.py"
)
TICKET_ARGUMENTS = {
    "title": "MCP link outage",
    "description": "Investigate the MCP-managed link.",
}


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
            choices=[SimpleNamespace(message=next(self._messages))]
        )


class FakeClient:
    def __init__(self, messages: list[FakeMessage]) -> None:
        self.completions = FakeCompletions(messages)
        self.chat = SimpleNamespace(completions=self.completions)


def ticket_call() -> FakeMessage:
    return FakeMessage(
        content="",
        tool_calls=[
            FakeToolCall(
                id="mcp_create_ticket_call",
                function=FakeFunction(
                    name="create_ticket",
                    arguments=json.dumps(TICKET_ARGUMENTS),
                ),
            )
        ],
    )


def mcp_policy() -> MCPToolPolicy:
    return MCPToolPolicy(
        ordinary_tool_names=frozenset({"get_ticket"}),
        effect_tools=(
            MCPEffectToolPolicy(
                tool_name="create_ticket",
                description="Create an effect-managed MCP ticket.",
                args_model=CreateTicketArgs,
                reconciliation_tool_name=(
                    "get_ticket_by_idempotency_key"
                ),
            ),
        ),
    )


def mcp_server(
    business_database: Path,
    *,
    reconciliation: str = "normal",
) -> MCPStdioServer:
    environment = os.environ.copy()
    environment["RTA_MCP_DEMO_DB"] = str(business_database)
    environment["RTA_MCP_DEMO_RECONCILIATION"] = reconciliation
    return MCPStdioServer(
        command=sys.executable,
        args=(str(DEMO_SERVER),),
        env=environment,
        cwd=DEMO_SERVER.parent,
    )


def mcp_registry(server: MCPStdioServer) -> ToolRegistry:
    registry = ToolRegistry()
    register_mcp_effect_tools(registry, server, mcp_policy())
    return registry


def build_agent(
    *,
    server: MCPStdioServer,
    client: FakeClient,
    runs: Path,
    ledger: Path,
    fault_hook=None,
) -> AgentLoop:
    return AgentLoop(
        mcp_registry(server),
        client=client,
        model="fake-model",
        max_steps=3,
        checkpoint_store=CheckpointStore(runs),
        trace_store=TraceStore(runs),
        effect_store=EffectStore(ledger),
        fault_hook=fault_hook,
    )


def database_count(database: Path, table: str) -> int:
    if not database.exists():
        return 0
    with sqlite3.connect(database) as connection:
        try:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row[0]) if row is not None else 0


def invocation_count(database: Path, tool_name: str) -> int:
    if not database.exists():
        return 0
    with sqlite3.connect(database) as connection:
        try:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM tool_invocations
                WHERE tool_name = ?
                """,
                (tool_name,),
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row[0]) if row is not None else 0


def effect_record(agent: AgentLoop, ledger: Path):
    effect_id, _ = build_effect_identity(
        agent.last_checkpoint.run_id,
        "mcp_create_ticket_call",
    )
    return EffectStore(ledger).get(effect_id)


def test_normal_mcp_effect_commits_one_business_ticket(tmp_path) -> None:
    business = tmp_path / "business.sqlite3"
    ledger = tmp_path / "effects.sqlite3"
    server = mcp_server(business)
    registry = mcp_registry(server)

    bypass = registry.execute("create_ticket", TICKET_ARGUMENTS)
    assert bypass.ok is False
    assert database_count(business, "tickets") == 0

    agent = AgentLoop(
        registry,
        client=FakeClient(
            [ticket_call(), FakeMessage(content="done")]
        ),
        model="fake-model",
        max_steps=3,
        checkpoint_store=CheckpointStore(tmp_path / "runs"),
        trace_store=TraceStore(tmp_path / "runs"),
        effect_store=EffectStore(ledger),
    )

    assert agent.run("Create one MCP ticket.") == "done"
    assert database_count(business, "tickets") == 1
    assert invocation_count(business, "create_ticket") == 1
    assert effect_record(agent, ledger).state == "COMMITTED"
    assert agent.last_checkpoint.status == "completed"


def test_mcp_effect_crash_after_commit_recovers_without_reentry(
    tmp_path,
) -> None:
    business = tmp_path / "business.sqlite3"
    ledger = tmp_path / "effects.sqlite3"
    runs = tmp_path / "runs"
    server = mcp_server(business)

    def crash(stage: str) -> None:
        if stage == "after_effect_execute":
            raise RuntimeError("crash after MCP business commit")

    first = build_agent(
        server=server,
        client=FakeClient([ticket_call()]),
        runs=runs,
        ledger=ledger,
        fault_hook=crash,
    )
    with pytest.raises(RuntimeError, match="MCP business commit"):
        first.run("Create one MCP ticket.")

    assert effect_record(first, ledger).state == "PREPARED"
    assert database_count(business, "tickets") == 1
    assert invocation_count(business, "create_ticket") == 1

    resumed = build_agent(
        server=server,
        client=FakeClient([FakeMessage(content="recovered")]),
        runs=runs,
        ledger=ledger,
    )
    assert resumed.resume(first.last_checkpoint.run_id) == "recovered"
    assert database_count(business, "tickets") == 1
    assert invocation_count(business, "create_ticket") == 1
    assert invocation_count(
        business, "get_ticket_by_idempotency_key"
    ) == 1
    assert effect_record(resumed, ledger).state == "COMMITTED"


def test_mcp_effect_not_found_allows_safe_execution(tmp_path) -> None:
    business = tmp_path / "business.sqlite3"
    ledger = tmp_path / "effects.sqlite3"
    runs = tmp_path / "runs"
    server = mcp_server(business)

    def crash(stage: str) -> None:
        if stage == "after_effect_prepared":
            raise RuntimeError("crash after MCP PREPARED")

    first = build_agent(
        server=server,
        client=FakeClient([ticket_call()]),
        runs=runs,
        ledger=ledger,
        fault_hook=crash,
    )
    with pytest.raises(RuntimeError, match="MCP PREPARED"):
        first.run("Create one MCP ticket.")

    assert database_count(business, "tickets") == 0
    resumed = build_agent(
        server=server,
        client=FakeClient([FakeMessage(content="created")]),
        runs=runs,
        ledger=ledger,
    )
    assert resumed.resume(first.last_checkpoint.run_id) == "created"
    assert database_count(business, "tickets") == 1
    assert invocation_count(business, "create_ticket") == 1
    assert invocation_count(
        business, "get_ticket_by_idempotency_key"
    ) == 1
    assert effect_record(resumed, ledger).state == "COMMITTED"


def test_mcp_reconciliation_failure_becomes_unknown(tmp_path) -> None:
    business = tmp_path / "business.sqlite3"
    ledger = tmp_path / "effects.sqlite3"
    runs = tmp_path / "runs"
    normal_server = mcp_server(business)

    def crash(stage: str) -> None:
        if stage == "after_effect_prepared":
            raise RuntimeError("crash before MCP execution")

    first = build_agent(
        server=normal_server,
        client=FakeClient([ticket_call()]),
        runs=runs,
        ledger=ledger,
        fault_hook=crash,
    )
    with pytest.raises(RuntimeError, match="before MCP execution"):
        first.run("Create one MCP ticket.")

    resumed = build_agent(
        server=mcp_server(business, reconciliation="error"),
        client=FakeClient([]),
        runs=runs,
        ledger=ledger,
    )
    with pytest.raises(
        EffectStateUnknownError,
        match="Reconciliation raised MCPInvocationError",
    ):
        resumed.resume(first.last_checkpoint.run_id)

    assert database_count(business, "tickets") == 0
    assert invocation_count(business, "create_ticket") == 0
    assert invocation_count(
        business, "get_ticket_by_idempotency_key"
    ) == 1
    assert effect_record(resumed, ledger).state == "UNKNOWN"
    assert resumed.last_checkpoint.status == "failed"

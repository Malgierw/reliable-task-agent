from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from reliable_task_agent.mcp_adapter import (
    MCPDiscoveryError,
    MCPEffectToolPolicy,
    MCPInvocationError,
    MCPStdioServer,
    MCPToolPolicy,
    discover_stdio_tools,
    invoke_stdio_tool,
)
from reliable_task_agent.tools.tickets import CreateTicketArgs


DEMO_SERVER = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mcp_demo_server.py"
)


def demo_server() -> MCPStdioServer:
    return MCPStdioServer(
        command=sys.executable,
        args=(str(DEMO_SERVER),),
        cwd=DEMO_SERVER.parent,
    )


def demo_policy() -> MCPToolPolicy:
    return MCPToolPolicy(
        ordinary_tool_names=frozenset({"get_ticket"}),
        effect_tools=(
            MCPEffectToolPolicy(
                tool_name="create_ticket",
                description="Create a protected MCP ticket.",
                args_model=CreateTicketArgs,
                reconciliation_tool_name=(
                    "get_ticket_by_idempotency_key"
                ),
            ),
        ),
    )


def discover_demo_tools():
    return asyncio.run(
        discover_stdio_tools(
            demo_server()
        )
    )


def test_discovers_and_maps_demo_mcp_tools() -> None:
    tools = {tool.name: tool for tool in discover_demo_tools()}

    assert set(tools) == {
        "get_ticket",
        "create_ticket",
        "get_ticket_by_idempotency_key",
    }
    assert tools["get_ticket"].description == (
        "Look up a ticket by its identifier."
    )
    assert tools["create_ticket"].description == (
        "Create a ticket in the demo service."
    )

    get_schema = tools["get_ticket"].input_schema
    assert get_schema["type"] == "object"
    assert get_schema["properties"]["ticket_id"]["type"] == "string"
    assert get_schema["required"] == ["ticket_id"]

    create_schema = tools["create_ticket"].input_schema
    assert create_schema["type"] == "object"
    assert create_schema["properties"]["title"]["type"] == "string"
    assert create_schema["properties"]["description"]["type"] == "string"
    assert set(create_schema["required"]) == {
        "title",
        "description",
        "idempotency_key",
    }

    assert tools["get_ticket"].to_openai_schema() == {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": "Look up a ticket by its identifier.",
            "parameters": get_schema,
        },
    }


def test_preserves_annotations_and_metadata_as_untrusted_hints() -> None:
    tools = {tool.name: tool for tool in discover_demo_tools()}

    assert tools["get_ticket"].annotations == {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
    assert tools["create_ticket"].annotations["readOnlyHint"] is False
    assert tools["create_ticket"].annotations["idempotentHint"] is False
    assert tools["get_ticket"].metadata == {
        "demo": True,
        "category": "ticket",
    }


def test_discovery_failure_has_adapter_level_error() -> None:
    server = MCPStdioServer(
        command=sys.executable,
        args=("-c", "raise SystemExit(23)"),
    )

    with pytest.raises(
        MCPDiscoveryError,
        match="MCP tool discovery failed for stdio server",
    ):
        asyncio.run(discover_stdio_tools(server))


def test_invokes_get_ticket_and_maps_mcp_result() -> None:
    result = asyncio.run(
        invoke_stdio_tool(
            demo_server(),
            tool_name="get_ticket",
            arguments={"ticket_id": "ticket-417"},
            policy=demo_policy(),
        )
    )

    assert result.ok is True
    assert result.tool_name == "get_ticket"
    assert result.error is None
    assert result.data["isError"] is False
    assert result.data["structuredContent"] == {
        "ticket_id": "ticket-417",
        "status": "open",
    }
    assert result.data["content"]


def test_maps_mcp_tool_error_without_transport_failure() -> None:
    result = asyncio.run(
        invoke_stdio_tool(
            demo_server(),
            tool_name="get_ticket",
            arguments={"ticket_id": "raise-error"},
            policy=demo_policy(),
        )
    )

    assert result.ok is False
    assert result.tool_name == "get_ticket"
    assert result.data["isError"] is True
    assert "demo ticket lookup failed" in result.error


def test_invocation_transport_failure_has_adapter_level_error() -> None:
    server = MCPStdioServer(
        command=sys.executable,
        args=("-c", "raise SystemExit(24)"),
    )

    with pytest.raises(
        MCPInvocationError,
        match="MCP tool invocation failed for stdio server",
    ):
        asyncio.run(
            invoke_stdio_tool(
                server,
                tool_name="get_ticket",
                arguments={"ticket_id": "ticket-1"},
                policy=demo_policy(),
            )
        )


def test_create_ticket_is_not_exposed_as_an_ordinary_tool() -> None:
    result = asyncio.run(
        invoke_stdio_tool(
            MCPStdioServer(command="must-not-be-launched"),
            tool_name="create_ticket",
            arguments={"title": "A", "description": "B"},
            policy=demo_policy(),
        )
    )

    assert result.ok is False
    assert result.tool_name == "create_ticket"
    assert "must execute through RTA Effect Boundary" in result.error


def test_policy_rejects_ordinary_effect_overlap() -> None:
    with pytest.raises(
        ValueError,
        match="both ordinary and effect-managed",
    ):
        MCPToolPolicy(
            ordinary_tool_names=frozenset({"create_ticket"}),
            effect_tools=demo_policy().effect_tools,
        )

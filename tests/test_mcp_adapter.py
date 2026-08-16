from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from reliable_task_agent.mcp_adapter import (
    MCPDiscoveryError,
    MCPStdioServer,
    discover_stdio_tools,
)


DEMO_SERVER = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mcp_demo_server.py"
)


def discover_demo_tools():
    return asyncio.run(
        discover_stdio_tools(
            MCPStdioServer(
                command=sys.executable,
                args=(str(DEMO_SERVER),),
                cwd=DEMO_SERVER.parent,
            )
        )
    )


def test_discovers_and_maps_demo_mcp_tools() -> None:
    tools = {tool.name: tool for tool in discover_demo_tools()}

    assert set(tools) == {"get_ticket", "create_ticket"}
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
    assert set(create_schema["required"]) == {"title", "description"}

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

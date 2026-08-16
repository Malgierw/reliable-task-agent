from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations


server = MCPServer(
    "reliable-task-agent-mcp-demo",
    description="Local MCP server for RTA discovery tests.",
)


@server.tool(
    description="Look up a ticket by its identifier.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    meta={"demo": True, "category": "ticket"},
)
def get_ticket(ticket_id: str) -> dict[str, Any]:
    """Return a deterministic demo ticket."""

    return {
        "ticket_id": ticket_id,
        "status": "open",
    }


@server.tool(
    description="Create a ticket in the demo service.",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    meta={"demo": True, "category": "ticket"},
)
def create_ticket(title: str, description: str) -> dict[str, Any]:
    """Return a deterministic demo creation receipt."""

    return {
        "ticket_id": "demo-001",
        "title": title,
        "description": description,
    }


if __name__ == "__main__":
    server.run(transport="stdio")

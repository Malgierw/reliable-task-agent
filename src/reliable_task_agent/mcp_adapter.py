from __future__ import annotations

from copy import deepcopy
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from reliable_task_agent.tools.registry import ToolExecutionResult


class MCPDiscoveryError(RuntimeError):
    """Raised when an MCP stdio server cannot provide its tool list."""


class MCPInvocationError(RuntimeError):
    """Raised when MCP transport or protocol failure prevents invocation."""


@dataclass(frozen=True)
class MCPStdioServer:
    """Connection settings for one local stdio MCP server."""

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | Path | None = None

    def to_sdk_parameters(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=self.command,
            args=list(self.args),
            env=deepcopy(self.env),
            cwd=self.cwd,
        )


@dataclass(frozen=True)
class MCPToolDefinition:
    """RTA-facing metadata discovered through MCP tools/list.

    MCP annotations are retained as untrusted hints. This Phase A model does
    not classify or execute tools and does not opt them into Effect Boundary.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        """Project the discovery record into RTA's function-schema shape."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.input_schema),
            },
        }


def _map_tool(tool: Tool) -> MCPToolDefinition:
    annotations = (
        tool.annotations.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        if tool.annotations is not None
        else {}
    )
    return MCPToolDefinition(
        name=tool.name,
        description=tool.description or "",
        input_schema=deepcopy(tool.input_schema),
        annotations=annotations,
        metadata=deepcopy(tool.meta or {}),
    )


async def discover_stdio_tools(
    server: MCPStdioServer,
) -> list[MCPToolDefinition]:
    """Connect to one stdio server and discover all tools/list pages."""

    try:
        async with stdio_client(server.to_sdk_parameters()) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                discovered: list[MCPToolDefinition] = []
                cursor: str | None = None

                while True:
                    result = await session.list_tools(
                        params=(
                            None
                            if cursor is None
                            else PaginatedRequestParams(cursor=cursor)
                        )
                    )
                    discovered.extend(_map_tool(tool) for tool in result.tools)
                    cursor = result.next_cursor
                    if cursor is None:
                        return discovered
    except Exception as exc:
        raise MCPDiscoveryError(
            "MCP tool discovery failed for stdio server "
            f"{server.command!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _map_call_result(
    tool_name: str,
    result: CallToolResult,
) -> ToolExecutionResult:
    data = result.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    if not result.is_error:
        return ToolExecutionResult(
            ok=True,
            tool_name=tool_name,
            data=data,
        )

    text_errors = [
        content.text
        for content in result.content
        if isinstance(content, TextContent)
    ]
    return ToolExecutionResult(
        ok=False,
        tool_name=tool_name,
        data=data,
        error=(
            "\n".join(text_errors)
            if text_errors
            else "MCP tool returned isError=true."
        ),
    )


async def invoke_stdio_tool(
    server: MCPStdioServer,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    ordinary_tool_names: Collection[str],
) -> ToolExecutionResult:
    """Invoke an explicitly approved ordinary MCP tool through tools/call.

    The allowlist is an application decision. MCP annotations never add a
    tool to it and are not consulted by this function.
    """

    if tool_name not in ordinary_tool_names:
        return ToolExecutionResult(
            ok=False,
            tool_name=tool_name,
            error=(
                "MCP tool is not explicitly approved for ordinary "
                f"invocation: {tool_name}"
            ),
        )

    try:
        async with stdio_client(server.to_sdk_parameters()) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if not isinstance(result, CallToolResult):
                    raise TypeError(
                        "MCP tools/call returned an unsupported result "
                        f"type: {type(result).__name__}"
                    )
                return _map_call_result(tool_name, result)
    except Exception as exc:
        raise MCPInvocationError(
            "MCP tool invocation failed for stdio server "
            f"{server.command!r}, tool {tool_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

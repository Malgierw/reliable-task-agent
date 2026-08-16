from __future__ import annotations

import asyncio
from copy import deepcopy
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
from pydantic import BaseModel

from reliable_task_agent.effects import ReconciliationResult
from reliable_task_agent.telemetry import NOOP_TELEMETRY, Telemetry
from reliable_task_agent.tools.registry import ToolExecutionResult
from reliable_task_agent.tools.registry import ToolRegistry


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

    MCP annotations are retained as untrusted hints. Discovery does not
    classify or execute tools and does not opt them into Effect Boundary.
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


@dataclass(frozen=True)
class MCPEffectToolPolicy:
    """Explicit local policy for one MCP tool protected by Effect Boundary."""

    tool_name: str
    description: str
    args_model: type[BaseModel]
    reconciliation_tool_name: str
    idempotency_argument: str = "idempotency_key"


@dataclass(frozen=True)
class MCPToolPolicy:
    """Local execution policy; MCP annotations never populate this model."""

    ordinary_tool_names: frozenset[str] = frozenset()
    effect_tools: tuple[MCPEffectToolPolicy, ...] = ()

    def __post_init__(self) -> None:
        effect_names = [item.tool_name for item in self.effect_tools]
        if len(effect_names) != len(set(effect_names)):
            raise ValueError("Duplicate MCP effect tool policy names.")
        overlap = self.ordinary_tool_names.intersection(effect_names)
        if overlap:
            raise ValueError(
                "MCP tools cannot be both ordinary and effect-managed: "
                f"{', '.join(sorted(overlap))}"
            )

    @property
    def effect_tool_names(self) -> frozenset[str]:
        return frozenset(item.tool_name for item in self.effect_tools)


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


async def _call_stdio_tool(
    server: MCPStdioServer,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    telemetry: Telemetry | None = None,
) -> ToolExecutionResult:
    observer = telemetry or NOOP_TELEMETRY
    with observer.span(
        "rta.mcp.call",
        {
            "rta.tool.name": tool_name,
            "rta.mcp.transport": "stdio",
        },
        error_category="mcp_invocation",
    ) as telemetry_span:
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
                    mapped = _map_call_result(tool_name, result)
                    telemetry_span.set_attribute("rta.tool.ok", mapped.ok)
                    return mapped
        except Exception as exc:
            raise MCPInvocationError(
                "MCP tool invocation failed for stdio server "
                f"{server.command!r}, tool {tool_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


async def invoke_stdio_tool(
    server: MCPStdioServer,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    policy: MCPToolPolicy,
    telemetry: Telemetry | None = None,
) -> ToolExecutionResult:
    """Invoke an explicitly approved ordinary MCP tool through tools/call.

    The policy is an application decision. MCP annotations never add a tool
    to it and are not consulted by this function.
    """

    if tool_name in policy.effect_tool_names:
        return ToolExecutionResult(
            ok=False,
            tool_name=tool_name,
            error=(
                "Effect-managed MCP tool must execute through RTA Effect "
                f"Boundary: {tool_name}"
            ),
        )
    if tool_name not in policy.ordinary_tool_names:
        return ToolExecutionResult(
            ok=False,
            tool_name=tool_name,
            error=(
                "MCP tool is not explicitly approved for ordinary "
                f"invocation: {tool_name}"
            ),
        )
    return await _call_stdio_tool(
        server,
        tool_name=tool_name,
        arguments=arguments,
        telemetry=telemetry,
    )


def _structured_result(
    result: ToolExecutionResult,
    *,
    purpose: str,
) -> dict[str, Any]:
    if not result.ok:
        raise MCPInvocationError(
            f"MCP {purpose} tool returned an error: {result.error}"
        )
    if not isinstance(result.data, dict):
        raise MCPInvocationError(
            f"MCP {purpose} result is missing protocol data."
        )
    structured = result.data.get("structuredContent")
    if not isinstance(structured, dict):
        raise MCPInvocationError(
            f"MCP {purpose} result requires object structuredContent."
        )
    return deepcopy(structured)


def register_mcp_effect_tools(
    registry: ToolRegistry,
    server: MCPStdioServer,
    policy: MCPToolPolicy,
    *,
    telemetry: Telemetry | None = None,
) -> None:
    """Register explicit MCP effects with the existing RTA Effect Boundary."""

    for effect_policy in policy.effect_tools:

        def execute(
            args: BaseModel,
            idempotency_key: str,
            *,
            _policy: MCPEffectToolPolicy = effect_policy,
        ) -> dict[str, Any]:
            arguments = args.model_dump(mode="json")
            arguments[_policy.idempotency_argument] = idempotency_key
            result = asyncio.run(
                _call_stdio_tool(
                    server,
                    tool_name=_policy.tool_name,
                    arguments=arguments,
                    telemetry=telemetry,
                )
            )
            return _structured_result(result, purpose="effect")

        def reconcile(
            _: BaseModel,
            idempotency_key: str,
            *,
            _policy: MCPEffectToolPolicy = effect_policy,
        ) -> ReconciliationResult:
            result = asyncio.run(
                _call_stdio_tool(
                    server,
                    tool_name=_policy.reconciliation_tool_name,
                    arguments={
                        _policy.idempotency_argument: idempotency_key,
                    },
                    telemetry=telemetry,
                )
            )
            structured = _structured_result(
                result,
                purpose="reconciliation",
            )
            return ReconciliationResult.model_validate(structured)

        registry.register_effect(
            name=effect_policy.tool_name,
            description=effect_policy.description,
            args_model=effect_policy.args_model,
            execute=execute,
            reconcile=reconcile,
        )

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.checkpoint import AgentCheckpoint, CompletedToolCall
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.effects import (
    EffectExecutor,
    EffectStateUnknownError,
    EffectStore,
    ReconciliationResult,
    build_effect_identity,
    hash_arguments,
)
from reliable_task_agent.mcp_adapter import _map_call_result
from reliable_task_agent.tools.registry import ToolRegistry
from reliable_task_agent.trace_store import TraceStore


SECRETS = (
    "sk-test-secret-123",
    "Authorization: Bearer test-token",
    "https://user:pass@example.test/path?token=secret",
    "query_token=abc123",
    "MCP server error text containing secrets",
)
SECRET_TEXT = " | ".join(SECRETS)


def assert_secrets_absent(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text


class SecretModelError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(SECRET_TEXT)
        self.status_code = 401


class FailingCompletions:
    def create(self, **_: Any) -> Any:
        raise SecretModelError()


class FailingClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FailingCompletions())


def test_model_exception_is_sanitized_in_checkpoint_and_trace(
    tmp_path,
) -> None:
    run_id = "a" * 32
    agent = AgentLoop(
        ToolRegistry(),
        client=FailingClient(),
        model="fake-model",
        checkpoint_store=CheckpointStore(tmp_path),
        trace_store=TraceStore(tmp_path),
    )

    with pytest.raises(SecretModelError, match="sk-test-secret-123"):
        agent.run("safe task", run_id=run_id)

    checkpoint_text = (
        tmp_path / run_id / "checkpoint.json"
    ).read_text(encoding="utf-8")
    trace_text = (tmp_path / run_id / "trace.json").read_text(
        encoding="utf-8"
    )
    assert_secrets_absent(checkpoint_text)
    assert_secrets_absent(trace_text)
    checkpoint = json.loads(checkpoint_text)
    checkpoint_error = json.loads(checkpoint["error_message"])
    trace = json.loads(trace_text)
    trace_error = trace["events"][0]["details"]
    assert checkpoint_error["error_category"] == "model_request"
    assert checkpoint_error["status_code"] == 401
    assert trace_error["error_category"] == "model_request"


class ValidationArgs(BaseModel):
    count: int


class HandlerArgs(BaseModel):
    label: str


def test_persisted_tool_errors_exclude_validation_values_and_exceptions(
    tmp_path,
) -> None:
    registry = ToolRegistry()
    registry.register(
        name="validate",
        description="Validate an integer.",
        args_model=ValidationArgs,
        handler=lambda args: {"count": args.count},
    )
    registry.register(
        name="fail",
        description="Raise from a handler.",
        args_model=HandlerArgs,
        handler=lambda _: (_ for _ in ()).throw(RuntimeError(SECRET_TEXT)),
    )

    validation = registry.execute("validate", {"count": SECRET_TEXT})
    handler = registry.execute("fail", {"label": "safe"})
    checkpoint = AgentCheckpoint(run_id="b" * 32)
    checkpoint.record_tool_call(
        CompletedToolCall(
            tool_call_id="validation",
            tool_name="validate",
            arguments={"count": "intentionally omitted from assertion"},
            result=validation.model_dump(mode="json"),
        )
    )
    checkpoint.record_tool_call(
        CompletedToolCall(
            tool_call_id="handler",
            tool_name="fail",
            arguments={"label": "safe"},
            result=handler.model_dump(mode="json"),
        )
    )
    path = CheckpointStore(tmp_path).save(checkpoint)
    persisted = json.loads(path.read_text(encoding="utf-8"))

    validation_error = persisted["completed_tool_calls"]["validation"][
        "result"
    ]["error"]
    handler_error = persisted["completed_tool_calls"]["handler"]["result"][
        "error"
    ]
    assert_secrets_absent(validation_error)
    assert_secrets_absent(handler_error)
    assert '"field":"count"' in validation_error
    assert '"code":"int_parsing"' in validation_error
    assert '"error_category":"tool_execution"' in handler_error


class EffectArgs(BaseModel):
    title: str


@pytest.mark.parametrize("reconciler_raises", [False, True])
def test_unknown_reason_and_transition_exclude_untrusted_text(
    tmp_path,
    reconciler_raises: bool,
) -> None:
    database_path = tmp_path / "effects.sqlite3"
    store = EffectStore(database_path)
    registry = ToolRegistry()

    def reconcile(_: BaseModel, __: str) -> ReconciliationResult:
        if reconciler_raises:
            raise RuntimeError(SECRET_TEXT)
        return ReconciliationResult(status="UNKNOWN", reason=SECRET_TEXT)

    registry.register_effect(
        name="create_ticket",
        description="Create a ticket.",
        args_model=EffectArgs,
        execute=lambda _args, _key: pytest.fail("must not execute"),
        reconcile=reconcile,
    )
    args = EffectArgs(title="safe")
    run_id = "c" * 32
    tool_call_id = "call-effect"
    effect_id, idempotency_key = build_effect_identity(run_id, tool_call_id)
    store.prepare(
        effect_id=effect_id,
        run_id=run_id,
        tool_call_id=tool_call_id,
        tool_name="create_ticket",
        arguments_hash=hash_arguments(args),
        idempotency_key=idempotency_key,
    )
    transitions: list[dict[str, Any]] = []

    with pytest.raises(EffectStateUnknownError):
        EffectExecutor(store).execute(
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool=registry.get("create_ticket"),
            arguments=args.model_dump(),
            transition_hook=transitions.append,
            fault_hook=lambda _: None,
        )

    record = store.get(effect_id)
    assert record is not None
    assert record.state == "UNKNOWN"
    assert record.unknown_reason is not None
    assert_secrets_absent(record.unknown_reason)
    assert_secrets_absent(json.dumps(transitions))
    assert_secrets_absent(database_path.read_bytes().decode("latin-1"))
    expected_category = (
        "reconciliation_exception"
        if reconciler_raises
        else "reconciliation_unknown"
    )
    assert expected_category in record.unknown_reason


def test_mcp_is_error_content_is_not_mapped_into_persistable_result() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text=SECRET_TEXT)],
        isError=True,
    )

    mapped = _map_call_result("get_ticket", result)
    persisted = mapped.model_dump_json()

    assert mapped.ok is False
    assert mapped.data == {"isError": True}
    assert_secrets_absent(persisted)
    assert mapped.error is not None
    error = json.loads(mapped.error)
    assert error["error_category"] == "mcp_tool_error"

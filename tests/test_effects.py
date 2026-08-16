from __future__ import annotations

from collections import OrderedDict

import pytest
from pydantic import BaseModel

from reliable_task_agent.effects import (
    EffectIdentityMismatchError,
    EffectSafetyError,
    EffectStateUnknownError,
    EffectStore,
    build_effect_identity,
    hash_arguments,
)
from reliable_task_agent.tools.builtin import build_default_registry
from reliable_task_agent.tools.registry import ToolExecutionResult
from reliable_task_agent.tools.tickets import register_ticket_tool


class HashArgs(BaseModel):
    name: str
    metadata: dict[str, int]


def test_argument_hash_uses_canonical_json() -> None:
    first = HashArgs(
        name="ticket",
        metadata=OrderedDict([("b", 2), ("a", 1)]),
    )
    second = HashArgs(
        name="ticket",
        metadata=OrderedDict([("a", 1), ("b", 2)]),
    )

    assert hash_arguments(first) == hash_arguments(second)


def test_effect_store_reopens_committed_result(tmp_path) -> None:
    database_path = tmp_path / "runtime" / "effects.sqlite3"
    effect_id, idempotency_key = build_effect_identity(
        "a" * 32,
        "call_ticket",
    )
    store = EffectStore(database_path)
    record, created = store.prepare(
        effect_id=effect_id,
        run_id="a" * 32,
        tool_call_id="call_ticket",
        tool_name="create_ticket",
        arguments_hash="hash-1",
        idempotency_key=idempotency_key,
    )

    assert created is True
    assert record.state == "PREPARED"

    result = ToolExecutionResult(
        ok=True,
        tool_name="create_ticket",
        data={"ticket_id": 1},
    )
    store.commit(effect_id, result)

    reopened = EffectStore(database_path)
    committed = reopened.get(effect_id)

    assert committed is not None
    assert committed.state == "COMMITTED"
    assert committed.result == result.model_dump(mode="json")


def test_effect_store_rejects_identity_hash_mismatch(
    tmp_path,
) -> None:
    store = EffectStore(tmp_path / "effects.sqlite3")
    effect_id, idempotency_key = build_effect_identity(
        "b" * 32,
        "same_call",
    )
    base = {
        "effect_id": effect_id,
        "run_id": "b" * 32,
        "tool_call_id": "same_call",
        "tool_name": "create_ticket",
        "idempotency_key": idempotency_key,
    }
    store.prepare(arguments_hash="hash-1", **base)

    with pytest.raises(
        EffectIdentityMismatchError,
        match="identity/arguments mismatch",
    ):
        store.prepare(arguments_hash="hash-2", **base)


def test_effect_store_prepare_is_idempotent_while_prepared(
    tmp_path,
) -> None:
    store = EffectStore(tmp_path / "effects.sqlite3")
    effect_id, idempotency_key = build_effect_identity(
        "c" * 32,
        "same_call",
    )
    arguments = {
        "effect_id": effect_id,
        "run_id": "c" * 32,
        "tool_call_id": "same_call",
        "tool_name": "create_ticket",
        "arguments_hash": "hash-1",
        "idempotency_key": idempotency_key,
    }

    first, first_created = store.prepare(**arguments)
    second, second_created = store.prepare(**arguments)

    assert first_created is True
    assert second_created is False
    assert first.state == second.state == "PREPARED"


@pytest.mark.parametrize("terminal_state", ["COMMITTED", "UNKNOWN"])
def test_effect_store_rejects_prepare_for_terminal_record(
    tmp_path,
    terminal_state,
) -> None:
    store = EffectStore(tmp_path / terminal_state / "effects.sqlite3")
    effect_id, idempotency_key = build_effect_identity(
        "d" * 32,
        f"call_{terminal_state.lower()}",
    )
    arguments = {
        "effect_id": effect_id,
        "run_id": "d" * 32,
        "tool_call_id": f"call_{terminal_state.lower()}",
        "tool_name": "create_ticket",
        "arguments_hash": "hash-1",
        "idempotency_key": idempotency_key,
    }
    store.prepare(**arguments)

    if terminal_state == "COMMITTED":
        store.commit(
            effect_id,
            ToolExecutionResult(
                ok=True,
                tool_name="create_ticket",
                data={"ticket_id": 1},
            ),
        )
    else:
        store.mark_unknown(effect_id, "ambiguous")

    with pytest.raises(
        EffectSafetyError,
        match="Cannot prepare terminal Effect",
    ):
        store.prepare(**arguments)

    assert store.get(effect_id).state == terminal_state


def test_effect_store_rejects_unknown_to_committed(tmp_path) -> None:
    store = EffectStore(tmp_path / "effects.sqlite3")
    effect_id, idempotency_key = build_effect_identity(
        "e" * 32,
        "call_unknown",
    )
    store.prepare(
        effect_id=effect_id,
        run_id="e" * 32,
        tool_call_id="call_unknown",
        tool_name="create_ticket",
        arguments_hash="hash-1",
        idempotency_key=idempotency_key,
    )
    store.mark_unknown(effect_id, "ambiguous")

    with pytest.raises(EffectStateUnknownError):
        store.commit(
            effect_id,
            ToolExecutionResult(
                ok=True,
                tool_name="create_ticket",
                data={"ticket_id": 1},
            ),
        )

    assert store.get(effect_id).state == "UNKNOWN"


def test_effect_store_rejects_committed_to_unknown(tmp_path) -> None:
    store = EffectStore(tmp_path / "effects.sqlite3")
    effect_id, idempotency_key = build_effect_identity(
        "f" * 32,
        "call_committed",
    )
    store.prepare(
        effect_id=effect_id,
        run_id="f" * 32,
        tool_call_id="call_committed",
        tool_name="create_ticket",
        arguments_hash="hash-1",
        idempotency_key=idempotency_key,
    )
    store.commit(
        effect_id,
        ToolExecutionResult(
            ok=True,
            tool_name="create_ticket",
            data={"ticket_id": 1},
        ),
    )

    with pytest.raises(EffectSafetyError):
        store.mark_unknown(effect_id, "ambiguous")

    assert store.get(effect_id).state == "COMMITTED"


def test_registry_blocks_direct_effect_execution(tmp_path) -> None:
    registry = build_default_registry()
    tickets = register_ticket_tool(
        registry,
        tmp_path / "business" / "tickets.sqlite3",
    )

    result = registry.execute(
        "create_ticket",
        {
            "title": "Outage",
            "description": "Investigate link outage.",
        },
    )

    assert result.ok is False
    assert "Effect Boundary" in result.error
    assert tickets.count() == 0


def test_registry_ordinary_tools_remain_compatible() -> None:
    registry = build_default_registry()
    result = registry.execute(
        "calculate_shannon_capacity",
        {
            "bandwidth_hz": 20_000_000,
            "snr_db": 10,
        },
    )

    assert result.ok is True
    assert result.data["capacity_mbps"] == pytest.approx(
        69.1886,
        rel=1e-4,
    )

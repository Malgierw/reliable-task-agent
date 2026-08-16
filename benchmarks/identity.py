from __future__ import annotations

import hashlib
import json
from typing import Any


TOOL_CALL_ID = "create_ticket:0"


def canonical_json(value: dict[str, Any]) -> str:
    """Serialize benchmark inputs with stable ordering."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


def build_run_id(
    scenario: str,
    repetition: int,
    payload: dict[str, Any],
) -> str:
    """Build a configuration-neutral run identity."""

    material = (
        "reliability-benchmark:v1:"
        f"{scenario}:{repetition}:{payload_hash(payload)}"
    )
    return hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()[:32]


def build_logical_action_id(
    run_id: str,
    tool_call_id: str = TOOL_CALL_ID,
) -> str:
    material = f"logical-action:v1:{run_id}:{tool_call_id}"
    return hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()


def expected_effect_identity(
    run_id: str,
    tool_call_id: str = TOOL_CALL_ID,
) -> tuple[str, str]:
    """Mirror the documented RTA v1 identity algorithm independently."""

    material = (
        f"reliable-task-agent:v1:{run_id}:{tool_call_id}"
    )
    effect_id = hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()
    return effect_id, f"rta:{effect_id}"

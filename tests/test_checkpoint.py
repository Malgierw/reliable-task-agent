from __future__ import annotations

import pytest
from pydantic import ValidationError

from reliable_task_agent.checkpoint import (
    AgentCheckpoint,
    CompletedToolCall,
)


def test_checkpoint_records_completed_tool_call() -> None:
    """Checkpoint 应记录已经完成的工具调用。"""

    checkpoint = AgentCheckpoint(
        run_id="a" * 32,
        messages=[
            {
                "role": "user",
                "content": "计算信道容量。",
            }
        ],
    )

    tool_call = CompletedToolCall(
        tool_call_id="call_001",
        tool_name="calculate_shannon_capacity",
        arguments={
            "bandwidth_hz": 20_000_000,
            "snr_db": 10,
        },
        result={
            "ok": True,
            "capacity_mbps": 69.1886,
        },
    )

    checkpoint.record_tool_call(tool_call)
    checkpoint.advance_to(2)

    assert checkpoint.next_step == 2

    assert "call_001" in (
        checkpoint.completed_tool_calls
    )

    saved_call = checkpoint.completed_tool_calls[
        "call_001"
    ]

    assert saved_call.tool_name == (
        "calculate_shannon_capacity"
    )
    assert saved_call.result["ok"] is True


def test_checkpoint_can_be_marked_completed() -> None:
    """任务完成后，应保存最终答案和完成状态。"""

    checkpoint = AgentCheckpoint(
        run_id="b" * 32,
    )

    checkpoint.mark_completed(
        "香农理论容量约为 69.19 Mbps。"
    )

    assert checkpoint.status == "completed"
    assert checkpoint.final_answer == (
        "香农理论容量约为 69.19 Mbps。"
    )
    assert checkpoint.error_message is None


def test_checkpoint_rejects_invalid_run_id() -> None:
    """Checkpoint 不接受不合法的 run_id。"""

    with pytest.raises(ValidationError):
        AgentCheckpoint(
            run_id="../outside",
        )
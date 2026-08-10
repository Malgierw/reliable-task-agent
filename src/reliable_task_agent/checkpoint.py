from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


CheckpointStatus = Literal[
    "running",
    "completed",
    "failed",
]


def utc_now() -> datetime:
    """返回带时区信息的 UTC 时间。"""

    return datetime.now(timezone.utc)


class CompletedToolCall(BaseModel):
    """保存一次已经完成的工具调用。"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    result: dict[str, Any]


class AgentCheckpoint(BaseModel):
    """保存 Agent 恢复执行所需要的完整状态。"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    run_id: str = Field(
        pattern=r"^[0-9a-f]{32}$",
        description="与本次任务 Trace 相同的运行编号。",
    )

    next_step: int = Field(
        default=1,
        ge=1,
        description="恢复任务后应从第几轮继续执行。",
    )

    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="截至当前的完整模型消息上下文。",
    )

    completed_tool_calls: dict[str, CompletedToolCall] = Field(
        default_factory=dict,
        description="已经成功执行过的工具调用。",
    )

    status: CheckpointStatus = "running"

    final_answer: str | None = None
    error_message: str | None = None

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        """更新 Checkpoint 的最后修改时间。"""

        self.updated_at = utc_now()

    def record_tool_call(
        self,
        tool_call: CompletedToolCall,
    ) -> None:
        """记录一个已经完成的工具调用。"""

        self.completed_tool_calls[
            tool_call.tool_call_id
        ] = tool_call

        self.touch()

    def advance_to(self, next_step: int) -> None:
        """更新恢复时应继续执行的轮次。"""

        if next_step < 1:
            raise ValueError("next_step 必须大于等于 1。")

        self.next_step = next_step
        self.touch()

    def mark_running(self) -> None:
        """将任务重新标记为运行中。"""

        self.status = "running"
        self.final_answer = None
        self.error_message = None
        self.touch()

    def mark_completed(self, final_answer: str) -> None:
        """将任务标记为已完成。"""

        if not final_answer.strip():
            raise ValueError("最终答案不能为空。")

        self.status = "completed"
        self.final_answer = final_answer
        self.error_message = None
        self.touch()

    def mark_failed(self, error_message: str) -> None:
        """将任务标记为失败。"""

        if not error_message.strip():
            raise ValueError("错误信息不能为空。")

        self.status = "failed"
        self.error_message = error_message
        self.touch()
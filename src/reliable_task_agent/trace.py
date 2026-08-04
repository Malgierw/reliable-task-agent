from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


TraceEventType = Literal[
    "model_response",
    "tool_call",
    "tool_result",
    "final_answer",
    "error",
]


def utc_now() -> datetime:
    """返回带时区信息的 UTC 时间。"""

    return datetime.now(timezone.utc)


class TraceEvent(BaseModel):
    """Agent 运行过程中的一个事件。"""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(
        ge=1,
        description="事件发生在第几轮 Agent Loop。",
    )
    event_type: TraceEventType
    timestamp: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class RunTrace(BaseModel):
    """保存一次 Agent 任务的完整执行轨迹。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(
        default_factory=lambda: uuid4().hex
    )
    started_at: datetime = Field(default_factory=utc_now)
    events: list[TraceEvent] = Field(default_factory=list)

    def add(
        self,
        *,
        step: int,
        event_type: TraceEventType,
        details: dict[str, Any] | None = None,
    ) -> None:
        """向执行轨迹中增加一个事件。"""

        self.events.append(
            TraceEvent(
                step=step,
                event_type=event_type,
                details=details or {},
            )
        )

    def to_pretty_json(self) -> str:
        """把执行轨迹转换为便于阅读的 JSON。"""

        return self.model_dump_json(indent=2)
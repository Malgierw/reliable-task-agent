from __future__ import annotations

import json
from typing import Any

import time
from collections.abc import Callable

from openai import OpenAI

from reliable_task_agent.model_client import create_client
from reliable_task_agent.tools.registry import ToolRegistry
# from reliable_task_agent.trace import RunTrace
from reliable_task_agent.trace import RunTrace, TraceEventType
from reliable_task_agent.trace_store import TraceStore

SYSTEM_PROMPT = """
你是一个可靠的工程任务智能体。

你可以调用程序提供的工具完成任务。

规则：
1. 当已有工具能够完成计算、文件读取或其他确定性任务时，必须调用工具，
   不要依靠语言模型自行猜测结果。
2. 工具执行失败时，应根据错误信息决定是否修正参数并重新调用。
3. 最终回答应说明使用了哪些工具，并基于工具结果作答。
""".strip()

RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    409,  # Conflict
    429,  # Too Many Requests
    500,
    502,
    503,
    504,
}

RETRYABLE_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
}


def is_retryable_model_error(exc: Exception) -> bool:
    """判断模型请求异常是否可能通过重试恢复。"""

    status_code = getattr(exc, "status_code", None)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    return type(exc).__name__ in RETRYABLE_ERROR_NAMES

class AgentLoop:
    """负责模型与工具之间的循环调用。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        client: OpenAI | None = None,
        model: str | None = None,
        max_steps: int = 5,
        max_model_retries: int = 2,
        retry_delay_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        trace_store: TraceStore | None = None,
    ) -> None:
        if client is None and model is None:
            client, model = create_client()
        elif client is None or model is None:
            raise ValueError(
                "client 和 model 必须同时提供，或者同时省略。"
            )

        if max_steps < 1:
            raise ValueError("max_steps 必须大于等于 1。")

        if max_model_retries < 0:
            raise ValueError(
                "max_model_retries 必须大于等于 0。"
            )

        if retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds 必须大于等于 0。"
            )
            
        self.registry = registry
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.max_model_retries = max_model_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.sleep_fn = sleep_fn
        self.last_trace: RunTrace | None = None
        self.trace_store = trace_store

    def _persist_trace(self, trace: RunTrace) -> None:
        """在配置了 TraceStore 时保存当前执行轨迹。"""

        if self.trace_store is not None:
            self.trace_store.save(trace)


    def _record_event(
        self,
        *,
        trace: RunTrace,
        step: int,
        event_type: TraceEventType,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录一个 Trace 事件并立即保存。"""

        trace.add(
            step=step,
            event_type=event_type,
            details=details,
        )

        self._persist_trace(trace)
    
    def _request_model(
        self,
        *,
        messages: list[dict[str, Any]],
        step: int,
        trace: RunTrace,
    ) -> Any:
        """请求模型，并对临时性错误进行指数退避重试。"""

        total_attempts = self.max_model_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.registry.to_openai_tools(),
                    tool_choice="auto",
                )

            except Exception as exc:
                retryable = is_retryable_model_error(exc)
                is_last_attempt = attempt == total_attempts

                if not retryable or is_last_attempt:
                    self._record_event(
                        trace=trace,
                        step=step,
                        event_type="error",
                        details={
                            "stage": "model_request",
                            "attempt": attempt,
                            "max_attempts": total_attempts,
                            "retryable": retryable,
                            "error_type": type(exc).__name__,
                            "status_code": getattr(
                                exc,
                                "status_code",
                                None,
                            ),
                            "message": str(exc),
                        },
                    )
                    raise

                delay = self.retry_delay_seconds * (
                    2 ** (attempt - 1)
                )

                self._record_event(
                    trace=trace,
                    step=step,
                    event_type="retry",
                    details={
                        "stage": "model_request",
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "delay_seconds": delay,
                        "error_type": type(exc).__name__,
                        "status_code": getattr(
                            exc,
                            "status_code",
                            None,
                        ),
                        "message": str(exc),
                    },
                )

                self.sleep_fn(delay)

        raise RuntimeError("模型请求重试流程异常结束。")
    
    def run(self, user_input: str) -> str:
        """执行一次完整的用户任务。"""

        if not user_input.strip():
            raise ValueError("用户输入不能为空。")

        trace = RunTrace()
        self.last_trace = trace
        self._persist_trace(trace)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]

        for step in range(1, self.max_steps + 1):
            response = self._request_model(
            messages=messages,
            step=step,
            trace=trace,
)


            assistant_message = response.choices[0].message

            message_data = assistant_message.model_dump(
                mode="json",
                exclude_none=True,
            )
            message_data.pop("reasoning_content", None)

            self._record_event(
                trace=trace,
                step=step,
                event_type="model_response",
                details={
                    "message": message_data,
                },
            )

            # 保存模型本轮回复，包括可能存在的 tool_calls
            messages.append(message_data)

            # 没有请求调用工具，说明模型已经给出最终答案
            if not assistant_message.tool_calls:
                if not assistant_message.content:
                    self._record_event(
                        trace=trace,
                        step=step,
                        event_type="error",
                        details={
                            "stage": "model_response",
                            "message": (
                                "模型没有返回文本内容，"
                                "也没有调用工具。"
                            ),
                        },
                    )
                    raise RuntimeError(
                        "模型没有返回文本内容，也没有调用工具。"
                    )

                self._record_event(
                    trace=trace,
                    step=step,
                    event_type="final_answer",
                    details={
                        "content": assistant_message.content,
                    },
                )

                return assistant_message.content

            # 模型一轮中可能请求调用一个或多个工具
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                try:
                    arguments = json.loads(raw_arguments)

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "工具参数必须是 JSON 对象。"
                        )

                    self._record_event(
                        trace=trace,
                        step=step,
                        event_type="tool_call",
                        details={
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_name,
                            "arguments": arguments,
                        },
                    )

                    result = self.registry.execute(
                        tool_name,
                        arguments,
                    )

                    result_data = result.model_dump(
                        mode="json"
                    )

                    self._record_event(
                        trace=trace,
                        step=step,
                        event_type="tool_result",
                        details={
                            "tool_call_id": tool_call.id,
                            **result_data,
                        },
                    )

                    tool_content = json.dumps(
                        result_data,
                        ensure_ascii=False,
                    )

                except (json.JSONDecodeError, ValueError) as exc:
                    failure_data = {
                        "ok": False,
                        "tool_name": tool_name,
                        "data": None,
                        "error": f"工具参数解析失败：{exc}",
                    }

                    self._record_event(
                        trace=trace,
                        step=step,
                        event_type="tool_call",
                        details={
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_name,
                            "raw_arguments": raw_arguments,
                        },
                    )

                    self._record_event(
                        trace=trace,
                        step=step,
                        event_type="tool_result",
                        details={
                            "tool_call_id": tool_call.id,
                            **failure_data,
                        },
                    )

                    tool_content = json.dumps(
                        failure_data,
                        ensure_ascii=False,
                    )

                # 把工具执行结果返回给模型
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    }
                )

        error_message = (
            f"Agent 在 {self.max_steps} 轮内未完成任务。"
        )

        self._record_event(
            trace=trace,
            step=self.max_steps,
            event_type="error",
            details={
                "stage": "agent_loop",
                "message": error_message,
            },
        )

        raise RuntimeError(error_message)
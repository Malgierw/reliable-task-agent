from __future__ import annotations



import time
import json
from typing import Any
from copy import deepcopy
from collections.abc import Callable

from openai import OpenAI

from reliable_task_agent.model_client import create_client
from reliable_task_agent.tools.registry import ToolRegistry
# from reliable_task_agent.trace import RunTrace
from reliable_task_agent.trace import RunTrace, TraceEventType
from reliable_task_agent.trace_store import TraceStore
from reliable_task_agent.checkpoint import (
    AgentCheckpoint,
    CompletedToolCall,
)
from reliable_task_agent.checkpoint_store import CheckpointStore



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
        checkpoint_store: CheckpointStore | None = None,
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
        self.checkpoint_store = checkpoint_store
        self.last_checkpoint: AgentCheckpoint | None = None

    def _persist_trace(self, trace: RunTrace) -> None:
        """在配置了 TraceStore 时保存当前执行轨迹。"""

        if self.trace_store is not None:
            self.trace_store.save(trace)

    def _persist_checkpoint(
        self,
        checkpoint: AgentCheckpoint,
    ) -> None:
        """配置了 CheckpointStore 时，保存当前任务状态。"""

        if self.checkpoint_store is not None:
            self.checkpoint_store.save(checkpoint)


    def _sync_checkpoint_messages(
        self,
        *,
        checkpoint: AgentCheckpoint,
        messages: list[dict[str, Any]],
        next_step: int | None = None,
    ) -> None:
        """同步消息上下文、下一轮位置，并保存 Checkpoint。"""

        checkpoint.messages = deepcopy(messages)

        if next_step is None:
            checkpoint.touch()
        else:
            checkpoint.advance_to(next_step)

        self._persist_checkpoint(checkpoint)

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
        checkpoint: AgentCheckpoint,
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
                    
                    checkpoint.mark_failed(str(exc))
                    self._persist_checkpoint(checkpoint)
                    
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
        """启动一个新的 Agent 任务。"""

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

        checkpoint = AgentCheckpoint(
            run_id=trace.run_id,
            next_step=1,
            messages=deepcopy(messages),
        )

        self.last_checkpoint = checkpoint
        self._persist_checkpoint(checkpoint)

        return self._continue_run(
            messages=messages,
            trace=trace,
            checkpoint=checkpoint,
            start_step=1,
        )

    def _load_or_create_trace(
        self,
        run_id: str,
    ) -> RunTrace:
        """读取历史 Trace；不存在时创建同 run_id 的 Trace。"""

        if self.trace_store is not None:
            try:
                return self.trace_store.load(run_id)
            except FileNotFoundError:
                pass

        return RunTrace(run_id=run_id)

    def _continue_run(
        self,
        *,
        messages: list[dict[str, Any]],
        trace: RunTrace,
        checkpoint: AgentCheckpoint,
        start_step: int,
    ) -> str:
        """执行一次完整的用户任务。"""

        for step in range(start_step, self.max_steps + 1):
            response = self._request_model(
            messages=messages,
            step=step,
            trace=trace,
            checkpoint=checkpoint,
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
            
            self._sync_checkpoint_messages(
                checkpoint=checkpoint,
                messages=messages,
                next_step=step,
            )
            
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

                checkpoint.mark_completed(
                    assistant_message.content
                )
                self._persist_checkpoint(checkpoint)

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

                    if result.ok:
                        checkpoint.record_tool_call(
                            CompletedToolCall(
                                tool_call_id=tool_call.id,
                                tool_name=tool_name,
                                arguments=arguments,
                                result=result_data,
                            )
                        )

                        self._persist_checkpoint(checkpoint)

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
                
                self._sync_checkpoint_messages(
                    checkpoint=checkpoint,
                    messages=messages,
                    next_step=step,
                )
            
            checkpoint.advance_to(step + 1)
            self._persist_checkpoint(checkpoint)
        
        error_message = (
            f"Agent 在 {self.max_steps} 轮内未完成任务。"
        )
        
        checkpoint.mark_failed(error_message)
        self._persist_checkpoint(checkpoint)
        
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

    def resume(self, run_id: str) -> str:
        """根据 run_id 恢复一个已保存的任务。"""

        if self.checkpoint_store is None:
            raise RuntimeError(
                "恢复任务前必须配置 CheckpointStore。"
            )

        checkpoint = self.checkpoint_store.load(run_id)
        self.last_checkpoint = checkpoint

        trace = self._load_or_create_trace(run_id)
        self.last_trace = trace
        self._persist_trace(trace)

        # 已完成任务不需要再次请求模型
        if checkpoint.status == "completed":
            if checkpoint.final_answer is None:
                raise RuntimeError(
                    "已完成的 Checkpoint 缺少 final_answer。"
                )

            return checkpoint.final_answer

        messages = deepcopy(checkpoint.messages)
        start_step = checkpoint.next_step

        checkpoint.mark_running()
        self._persist_checkpoint(checkpoint)

        return self._continue_run(
            messages=messages,
            trace=trace,
            checkpoint=checkpoint,
            start_step=start_step,
        )

    def test_agent_resumes_failed_checkpoint(
        tmp_path,
    ) -> None:
        """失败任务应能从保存的轮次和消息继续执行。"""

        run_id = "c" * 32

        checkpoint = AgentCheckpoint(
            run_id=run_id,
            next_step=2,
            messages=[
                {
                    "role": "system",
                    "content": "系统提示词",
                },
                {
                    "role": "user",
                    "content": "继续执行任务。",
                },
            ],
        )

        checkpoint.mark_failed("模拟上一次请求失败")

        store = CheckpointStore(tmp_path / "runs")
        store.save(checkpoint)

        fake_client = FakeClient(
            messages=[
                FakeMessage(
                    content="恢复后的任务已经完成。",
                )
            ]
        )

        agent = AgentLoop(
            build_default_registry(),
            client=fake_client,
            model="fake-model",
            checkpoint_store=store,
        )

        answer = agent.resume(run_id)

        assert answer == "恢复后的任务已经完成。"

        loaded_checkpoint = store.load(run_id)

        assert loaded_checkpoint.status == "completed"
        assert loaded_checkpoint.final_answer == (
            "恢复后的任务已经完成。"
        )
        assert loaded_checkpoint.error_message is None

        assert len(fake_client.completions.requests) == 1

    def test_agent_resume_returns_completed_answer(
        tmp_path,
    ) -> None:
        """已完成任务不应再次调用模型。"""

        run_id = "d" * 32

        checkpoint = AgentCheckpoint(
            run_id=run_id,
        )
        checkpoint.mark_completed("之前保存的最终答案。")

        store = CheckpointStore(tmp_path / "runs")
        store.save(checkpoint)

        fake_client = FakeClient(messages=[])

        agent = AgentLoop(
            build_default_registry(),
            client=fake_client,
            model="fake-model",
            checkpoint_store=store,
        )

        answer = agent.resume(run_id)

        assert answer == "之前保存的最终答案。"
        assert len(fake_client.completions.requests) == 0

    def test_agent_resume_requires_checkpoint_store() -> None:
        """未配置 CheckpointStore 时不能恢复任务。"""

        fake_client = FakeClient(messages=[])

        agent = AgentLoop(
            build_default_registry(),
            client=fake_client,
            model="fake-model",
        )

        with pytest.raises(
            RuntimeError,
            match="CheckpointStore",
        ):
            agent.resume("e" * 32)



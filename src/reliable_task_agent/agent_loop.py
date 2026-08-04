from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from reliable_task_agent.model_client import create_client
from reliable_task_agent.tools.registry import ToolRegistry
from reliable_task_agent.trace import RunTrace

SYSTEM_PROMPT = """
你是一个可靠的工程任务智能体。

你可以调用程序提供的工具完成任务。

规则：
1. 当已有工具能够完成计算、文件读取或其他确定性任务时，必须调用工具，
   不要依靠语言模型自行猜测结果。
2. 工具执行失败时，应根据错误信息决定是否修正参数并重新调用。
3. 最终回答应说明使用了哪些工具，并基于工具结果作答。
""".strip()


class AgentLoop:
    """负责模型与工具之间的循环调用。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        client: OpenAI | None = None,
        model: str | None = None,
        max_steps: int = 5,
    ) -> None:
        if client is None and model is None:
            client, model = create_client()
        elif client is None or model is None:
            raise ValueError(
                "client 和 model 必须同时提供，或者同时省略。"
            )

        if max_steps < 1:
            raise ValueError("max_steps 必须大于等于 1。")

        self.registry = registry
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.last_trace: RunTrace | None = None

    def run(self, user_input: str) -> str:
        """执行一次完整的用户任务。"""

        if not user_input.strip():
            raise ValueError("用户输入不能为空。")

        trace = RunTrace()
        self.last_trace = trace

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
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.registry.to_openai_tools(),
                    tool_choice="auto",
                )
            except Exception as exc:
                trace.add(
                    step=step,
                    event_type="error",
                    details={
                        "stage": "model_request",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise

            assistant_message = response.choices[0].message

            message_data = assistant_message.model_dump(
                mode="json",
                exclude_none=True,
            )
            message_data.pop("reasoning_content", None)

            trace.add(
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
                    trace.add(
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

                trace.add(
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

                    trace.add(
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

                    trace.add(
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

                    trace.add(
                        step=step,
                        event_type="tool_call",
                        details={
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_name,
                            "raw_arguments": raw_arguments,
                        },
                    )

                    trace.add(
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

        trace.add(
            step=self.max_steps,
            event_type="error",
            details={
                "stage": "agent_loop",
                "message": error_message,
            },
        )

        raise RuntimeError(error_message)
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.tools.builtin import build_default_registry


@dataclass
class FakeFunction:
    """模拟模型返回的工具函数信息。"""

    name: str
    arguments: str


@dataclass
class FakeToolCall:
    """模拟模型返回的一次工具调用。"""

    id: str
    function: FakeFunction
    type: str = "function"
    index: int = 0


class FakeMessage:
    """模拟 OpenAI SDK 返回的 assistant message。"""

    def __init__(
        self,
        *,
        content: str | None,
        tool_calls: list[FakeToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, **_: Any) -> dict[str, Any]:
        """模拟 SDK 对象的 model_dump()。"""

        data: dict[str, Any] = {
            "role": "assistant",
        }

        if self.content is not None:
            data["content"] = self.content

        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "index": tool_call.index,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]

        return data

class FakeCompletions:
    """按照预设顺序返回模型消息。"""

    def __init__(
        self,
        messages: list[FakeMessage],
    ) -> None:
        self._messages = iter(messages)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """模拟 chat.completions.create()。"""

        # 保存请求发出时的数据快照
        self.requests.append(deepcopy(kwargs))

        # 取出预设的下一条模型消息
        message = next(self._messages)

        # 模拟真实 SDK 的响应结构
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                )
            ]
        )

 
class FakeClient:
    """模拟 OpenAI 客户端的属性结构。"""

    def __init__(
        self,
        messages: list[FakeMessage],
    ) -> None:
        self.completions = FakeCompletions(messages)

        self.chat = SimpleNamespace(
            completions=self.completions,
        )


def test_agent_calls_tool_and_records_trace() -> None:
    """Agent 应调用工具，并记录完整执行轨迹。"""

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_test_001",
                        function=FakeFunction(
                            name="calculate_shannon_capacity",
                            arguments=(
                                '{"bandwidth_hz": 20000000, '
                                '"snr_db": 10}'
                            ),
                        ),
                    )
                ],
            ),
            FakeMessage(
                content="香农理论容量约为 69.19 Mbps。",
            ),
        ]
    )

    agent = AgentLoop(
        build_default_registry(),
        client=fake_client,
        model="fake-model",
    )

    answer = agent.run(
        "计算带宽为 20 MHz、SNR 为 10 dB 时的容量。"
    )

    assert answer == "香农理论容量约为 69.19 Mbps。"

    assert agent.last_trace is not None

    event_types = [
        event.event_type
        for event in agent.last_trace.events
    ]

    assert event_types == [
        "model_response",
        "tool_call",
        "tool_result",
        "model_response",
        "final_answer",
    ]

    tool_result = agent.last_trace.events[2]

    assert tool_result.details["ok"] is True
    assert tool_result.details["tool_name"] == (
        "calculate_shannon_capacity"
    )

    assert tool_result.details["data"]["capacity_mbps"] == (
        pytest.approx(69.1886, rel=1e-4)
    )

    # 模型应被请求两次：
    # 第一次选择工具，第二次根据工具结果回答。
    assert len(fake_client.completions.requests) == 2

    second_request_messages = (
        fake_client.completions.requests[1]["messages"]
    )

    assert second_request_messages[-1]["role"] == "tool"


def test_agent_returns_answer_without_tool_call() -> None:
    """模型不调用工具时，Agent 应直接返回答案。"""

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="这是一个不需要调用工具的回答。",
            )
        ]
    )

    agent = AgentLoop(
        build_default_registry(),
        client=fake_client,
        model="fake-model",
    )

    answer = agent.run("请简单介绍你自己。")

    assert answer == "这是一个不需要调用工具的回答。"

    assert agent.last_trace is not None

    event_types = [
        event.event_type
        for event in agent.last_trace.events
    ]

    assert event_types == [
        "model_response",
        "final_answer",
    ]

    assert len(fake_client.completions.requests) == 1

def test_agent_records_invalid_tool_arguments() -> None:
    """工具参数不是合法 JSON 时，应记录失败并继续运行。"""

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_invalid_json",
                        function=FakeFunction(
                            name="calculate_shannon_capacity",
                            # 故意缺少最后的右花括号
                            arguments=(
                                '{"bandwidth_hz": 20000000, '
                                '"snr_db": 10'
                            ),
                        ),
                    )
                ],
            ),
            FakeMessage(
                content="工具参数格式错误，无法完成计算。",
            ),
        ]
    )

    agent = AgentLoop(
        build_default_registry(),
        client=fake_client,
        model="fake-model",
    )

    answer = agent.run(
        "计算带宽为 20 MHz、SNR 为 10 dB 时的容量。"
    )

    assert answer == "工具参数格式错误，无法完成计算。"

    assert agent.last_trace is not None

    event_types = [
        event.event_type
        for event in agent.last_trace.events
    ]

    assert event_types == [
        "model_response",
        "tool_call",
        "tool_result",
        "model_response",
        "final_answer",
    ]

    tool_call = agent.last_trace.events[1]
    tool_result = agent.last_trace.events[2]

    assert tool_call.details["tool_name"] == (
        "calculate_shannon_capacity"
    )
    assert "raw_arguments" in tool_call.details

    assert tool_result.details["ok"] is False
    assert tool_result.details["data"] is None
    assert "工具参数解析失败" in tool_result.details["error"]

    # 即使参数错误，Agent 仍会把错误结果返回模型，
    # 让模型进行第二轮回答。
    assert len(fake_client.completions.requests) == 2

    second_request_messages = (
        fake_client.completions.requests[1]["messages"]
    )

    assert second_request_messages[-1]["role"] == "tool"

def test_agent_records_tool_validation_failure() -> None:
    """JSON 合法但参数值错误时，应记录工具校验失败。"""

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_invalid_bandwidth",
                        function=FakeFunction(
                            name="calculate_shannon_capacity",
                            arguments=(
                                '{"bandwidth_hz": -1, '
                                '"snr_db": 10}'
                            ),
                        ),
                    )
                ],
            ),
            FakeMessage(
                content="带宽必须大于零，请提供有效参数。",
            ),
        ]
    )

    agent = AgentLoop(
        build_default_registry(),
        client=fake_client,
        model="fake-model",
    )

    answer = agent.run("计算带宽为 -1 Hz 时的信道容量。")

    assert answer == "带宽必须大于零，请提供有效参数。"

    assert agent.last_trace is not None

    event_types = [
        event.event_type
        for event in agent.last_trace.events
    ]

    assert event_types == [
        "model_response",
        "tool_call",
        "tool_result",
        "model_response",
        "final_answer",
    ]

    tool_call = agent.last_trace.events[1]
    tool_result = agent.last_trace.events[2]

    # JSON 本身可以正常解析，所以 Trace 中记录的是 arguments，
    # 而不是 raw_arguments。
    assert tool_call.details["arguments"] == {
        "bandwidth_hz": -1,
        "snr_db": 10,
    }

    # Registry 的 Pydantic 校验应拒绝负数带宽。
    assert tool_result.details["ok"] is False
    assert tool_result.details["data"] is None
    assert "工具参数校验失败" in tool_result.details["error"]
    assert "greater than 0" in tool_result.details["error"]

    # 错误结果仍应返回给模型进行第二轮回答。
    assert len(fake_client.completions.requests) == 2

    second_request_messages = (
        fake_client.completions.requests[1]["messages"]
    )

    assert second_request_messages[-1]["role"] == "tool"
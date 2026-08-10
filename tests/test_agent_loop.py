from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.tools.builtin import build_default_registry
from reliable_task_agent.trace_store import TraceStore
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.checkpoint import AgentCheckpoint


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

class RetryableModelError(RuntimeError):
    """模拟服务器暂时不可用。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 503


class FlakyCompletions:
    """第一次请求失败，第二次请求成功。"""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.call_count = 0

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(deepcopy(kwargs))
        self.call_count += 1

        if self.call_count == 1:
            raise RetryableModelError(
                "模拟服务器暂时不可用"
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=FakeMessage(
                        content="重试后请求成功。",
                    )
                )
            ]
        )


class FlakyClient:
    """模拟一次失败后恢复的模型客户端。"""

    def __init__(self) -> None:
        self.completions = FlakyCompletions()
        self.chat = SimpleNamespace(
            completions=self.completions,
        )

class FailingCompletions:
    """模拟模型请求发生异常。"""

    def create(self, **_: Any) -> Any:
        raise RuntimeError("模拟网络错误")


class FailingClient:
    """提供与 OpenAI 客户端相同的属性结构。"""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=FailingCompletions(),
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

def test_agent_records_model_request_error() -> None:
    """模型请求失败时，应记录错误 Trace 并继续抛出异常。"""

    agent = AgentLoop(
        build_default_registry(),
        client=FailingClient(),
        model="fake-model",
    )

    with pytest.raises(
        RuntimeError,
        match="模拟网络错误",
    ):
        agent.run("执行一个任务。")

    assert agent.last_trace is not None

    assert len(agent.last_trace.events) == 1

    error_event = agent.last_trace.events[0]

    assert error_event.event_type == "error"
    assert error_event.step == 1
    assert error_event.details["stage"] == "model_request"
    assert error_event.details["error_type"] == "RuntimeError"
    assert error_event.details["message"] == "模拟网络错误"
    
def test_agent_retries_transient_model_error() -> None:
    """临时性模型错误应自动重试并记录 Trace。"""

    fake_client = FlakyClient()
    sleep_delays: list[float] = []

    agent = AgentLoop(
        build_default_registry(),
        client=fake_client,
        model="fake-model",
        max_model_retries=2,
        retry_delay_seconds=0.5,
        sleep_fn=sleep_delays.append,
    )

    answer = agent.run("执行一个任务。")

    assert answer == "重试后请求成功。"
    assert fake_client.completions.call_count == 2

    # 测试中不会真的等待，只记录本应等待多久。
    assert sleep_delays == [0.5]

    assert agent.last_trace is not None

    event_types = [
        event.event_type
        for event in agent.last_trace.events
    ]

    assert event_types == [
        "retry",
        "model_response",
        "final_answer",
    ]

    retry_event = agent.last_trace.events[0]

    assert retry_event.details["attempt"] == 1
    assert retry_event.details["next_attempt"] == 2
    assert retry_event.details["delay_seconds"] == 0.5
    assert retry_event.details["status_code"] == 503

def test_agent_persists_completed_trace(tmp_path) -> None:
    """完成任务后，Agent 应将 Trace 自动保存到磁盘。"""

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="任务执行完成。",
            )
        ]
    )

    store = TraceStore(tmp_path / "runs")

    agent = AgentLoop(
        build_default_registry(),
        client=fake_client,
        model="fake-model",
        trace_store=store,
    )

    answer = agent.run("执行一个简单任务。")

    assert answer == "任务执行完成。"
    assert agent.last_trace is not None

    run_id = agent.last_trace.run_id

    assert store.list_run_ids() == [run_id]

    loaded_trace = store.load(run_id)

    assert loaded_trace == agent.last_trace
    assert [
        event.event_type
        for event in loaded_trace.events
    ] == [
        "model_response",
        "final_answer",
    ]

def test_agent_persists_error_trace(tmp_path) -> None:
    """模型请求失败时，错误 Trace 也应保存到磁盘。"""

    store = TraceStore(tmp_path / "runs")

    agent = AgentLoop(
        build_default_registry(),
        client=FailingClient(),
        model="fake-model",
        trace_store=store,
    )

    with pytest.raises(
        RuntimeError,
        match="模拟网络错误",
    ):
        agent.run("执行一个任务。")

    assert agent.last_trace is not None

    run_id = agent.last_trace.run_id
    loaded_trace = store.load(run_id)

    assert len(loaded_trace.events) == 1

    error_event = loaded_trace.events[0]

    assert error_event.event_type == "error"
    assert error_event.details["stage"] == "model_request"
    assert error_event.details["message"] == "模拟网络错误"

def test_agent_persists_completed_checkpoint(
    tmp_path,
) -> None:
    """完成任务后，应保存 completed Checkpoint。"""

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="任务执行完成。",
            )
        ]
    )

    store = CheckpointStore(tmp_path / "runs")

    agent = AgentLoop(
        build_default_registry(),
        client=fake_client,
        model="fake-model",
        checkpoint_store=store,
    )

    answer = agent.run("执行一个简单任务。")

    assert answer == "任务执行完成。"
    assert agent.last_checkpoint is not None

    checkpoint = store.load(
        agent.last_checkpoint.run_id
    )

    assert checkpoint.status == "completed"
    assert checkpoint.final_answer == "任务执行完成。"
    assert checkpoint.error_message is None

    assert checkpoint.messages[-1]["role"] == (
        "assistant"
    )
    assert checkpoint.messages[-1]["content"] == (
        "任务执行完成。"
    )
    
def test_agent_records_completed_tool_in_checkpoint(
    tmp_path,
) -> None:
    """成功执行的工具应记录进 Checkpoint。"""

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_checkpoint_001",
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
                content="容量约为 69.19 Mbps。",
            ),
        ]
    )

    store = CheckpointStore(tmp_path / "runs")

    agent = AgentLoop(
        build_default_registry(),
        client=fake_client,
        model="fake-model",
        checkpoint_store=store,
    )

    agent.run("计算信道容量。")

    assert agent.last_checkpoint is not None

    checkpoint = store.load(
        agent.last_checkpoint.run_id
    )

    saved_call = checkpoint.completed_tool_calls[
        "call_checkpoint_001"
    ]

    assert saved_call.tool_name == (
        "calculate_shannon_capacity"
    )
    assert saved_call.arguments == {
        "bandwidth_hz": 20_000_000,
        "snr_db": 10,
    }
    assert saved_call.result["ok"] is True
    assert saved_call.result["data"][
        "capacity_mbps"
    ] == pytest.approx(
        69.1886,
        rel=1e-4,
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






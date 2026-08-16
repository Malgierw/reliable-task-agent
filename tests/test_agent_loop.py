from __future__ import annotations

import shutil
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.tools.builtin import build_default_registry
from reliable_task_agent.tools.registry import ToolExecutionResult
from reliable_task_agent.trace_store import TraceStore
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.checkpoint import (
    AgentCheckpoint,
    CompletedToolCall,
)


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


def test_verification_failure_requests_persisted_repair(
    tmp_path,
    monkeypatch,
) -> None:
    """验证失败应记录 repair 事件，并把硬反馈持久化给下一轮。"""

    registry = build_default_registry()

    monkeypatch.setattr(
        registry,
        "execute",
        lambda name, arguments: ToolExecutionResult(
            ok=True,
            tool_name=name,
            data={
                "verification_passed": False,
                "errors": ["throughput_mbps.count missing"],
                "error_details": [
                    {
                        "type": "missing_metric_field",
                        "field": "throughput_mbps.count",
                        "expected": 5,
                        "actual": None,
                    }
                ],
                "checks": {
                    "aggregate_metrics_match": False,
                },
            },
        ),
    )

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_verify_failed",
                        function=FakeFunction(
                            name="verify_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            ),
            FakeMessage(content="Repair requested."),
        ]
    )

    store = CheckpointStore(tmp_path / "runs")
    agent = AgentLoop(
        registry,
        client=fake_client,
        model="fake-model",
        checkpoint_store=store,
    )

    assert agent.run("Verify the report.") == "Repair requested."
    assert agent.last_checkpoint is not None

    checkpoint = store.load(agent.last_checkpoint.run_id)
    assert checkpoint.repair_count == 1

    repair_events = [
        event
        for event in agent.last_trace.events
        if event.event_type == "repair_requested"
    ]

    assert len(repair_events) == 1
    assert repair_events[0].details == {
        "repair_attempt": 1,
        "max_repair_attempts": 2,
        "verifier_errors": [
            "throughput_mbps.count missing"
        ],
        "verifier_error_details": [
            {
                "type": "missing_metric_field",
                "field": "throughput_mbps.count",
                "expected": 5,
                "actual": None,
            }
        ],
        "verifier_checks": {
            "aggregate_metrics_match": False,
        },
    }

    next_request_messages = (
        fake_client.completions.requests[1]["messages"]
    )

    assert next_request_messages[-2]["role"] == "tool"
    assert next_request_messages[-1]["role"] == "system"
    assert "deterministic verification failed" in (
        next_request_messages[-1]["content"]
    )
    assert "throughput_mbps.count missing" in (
        next_request_messages[-1]["content"]
    )
    for structured_value in (
        "missing_metric_field",
        "throughput_mbps.count",
        '"expected": 5',
        '"actual": null',
    ):
        assert structured_value in (
            next_request_messages[-1]["content"]
        )
    assert next_request_messages[-1] in checkpoint.messages


def test_verification_repair_then_passes(
    monkeypatch,
) -> None:
    """验证失败、修复、重新验证通过后应正常完成。"""

    registry = build_default_registry()
    verification_count = 0

    def execute(
        name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        nonlocal verification_count

        if name == "verify_analysis_report":
            verification_count += 1
            passed = verification_count == 2

            return ToolExecutionResult(
                ok=True,
                tool_name=name,
                data={
                    "verification_passed": passed,
                    "errors": [] if passed else ["bad metric"],
                    "checks": {
                        "aggregate_metrics_match": passed,
                    },
                },
            )

        return ToolExecutionResult(
            ok=True,
            tool_name=name,
            data={"repaired": True},
        )

    monkeypatch.setattr(registry, "execute", execute)

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="verify_1",
                        function=FakeFunction(
                            name="verify_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            ),
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="repair_1",
                        function=FakeFunction(
                            name="write_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            ),
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="verify_2",
                        function=FakeFunction(
                            name="verify_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            ),
            FakeMessage(content="SUCCESS"),
        ]
    )

    agent = AgentLoop(
        registry,
        client=fake_client,
        model="fake-model",
        max_steps=4,
    )

    assert agent.run("Repair and verify.") == "SUCCESS"
    assert verification_count == 2
    assert agent.last_checkpoint.repair_count == 1
    assert agent.last_checkpoint.status == "completed"


def test_repeated_verification_failures_exhaust_repairs(
    tmp_path,
    monkeypatch,
) -> None:
    """两次 repair 后的第三次验证失败应确定性终止。"""

    registry = build_default_registry()

    monkeypatch.setattr(
        registry,
        "execute",
        lambda name, arguments: ToolExecutionResult(
            ok=True,
            tool_name=name,
            data={
                "verification_passed": False,
                "errors": ["still invalid"],
                "checks": {"aggregate_metrics_match": False},
            },
        ),
    )

    verifier_message = lambda call_id: FakeMessage(
        content="",
        tool_calls=[
            FakeToolCall(
                id=call_id,
                function=FakeFunction(
                    name="verify_analysis_report",
                    arguments="{}",
                ),
            )
        ],
    )

    fake_client = FakeClient(
        messages=[
            verifier_message("verify_failure_1"),
            verifier_message("verify_failure_2"),
            verifier_message("verify_failure_3"),
        ]
    )

    store = CheckpointStore(tmp_path / "runs")
    agent = AgentLoop(
        registry,
        client=fake_client,
        model="fake-model",
        max_steps=4,
        max_repair_attempts=2,
        checkpoint_store=store,
    )

    with pytest.raises(
        RuntimeError,
        match="已用尽 2 次修复机会",
    ):
        agent.run("Keep verifying.")

    checkpoint = store.load(agent.last_checkpoint.run_id)
    assert checkpoint.status == "failed"
    assert checkpoint.repair_count == 2

    repair_events = [
        event
        for event in agent.last_trace.events
        if event.event_type == "repair_requested"
    ]
    assert [
        event.details["repair_attempt"]
        for event in repair_events
    ] == [1, 2]

    assert agent.last_trace.events[-1].event_type == "error"
    assert agent.last_trace.events[-1].details["stage"] == (
        "verification_repair"
    )


def test_resume_preserves_repair_count(
    tmp_path,
) -> None:
    """恢复已有 repair 周期时不得把计数重置为零。"""

    run_id = "a" * 32
    checkpoint = AgentCheckpoint(
        run_id=run_id,
        next_step=2,
        repair_count=1,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "repair"},
            {
                "role": "system",
                "content": "existing verifier repair guidance",
            },
        ],
    )
    checkpoint.mark_failed("simulated crash during repair")

    store = CheckpointStore(tmp_path / "runs")
    store.save(checkpoint)

    fake_client = FakeClient(
        messages=[FakeMessage(content="resumed")]
    )
    agent = AgentLoop(
        build_default_registry(),
        client=fake_client,
        model="fake-model",
        checkpoint_store=store,
    )

    assert agent.resume(run_id) == "resumed"
    assert store.load(run_id).repair_count == 1
    assert (
        fake_client.completions.requests[0]["messages"][-1][
            "content"
        ]
        == "existing verifier repair guidance"
    )


def test_resume_reconciles_crash_before_repair_bookkeeping(
    tmp_path,
    monkeypatch,
) -> None:
    """Verifier tool message 落盘后的崩溃应在恢复时只消费一次 repair。"""

    verifier_tool_call_id = "verify_before_repair_crash"
    registry = build_default_registry()
    execute_calls: list[str] = []

    def execute(
        name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        execute_calls.append(name)

        if name == "verify_analysis_report":
            return ToolExecutionResult(
                ok=True,
                tool_name=name,
                data={
                    "verification_passed": False,
                    "errors": ["throughput_mbps.count missing"],
                    "error_details": [
                        {
                            "type": "missing_metric_field",
                            "field": "throughput_mbps.count",
                            "expected": 5,
                            "actual": None,
                        }
                    ],
                    "checks": {
                        "aggregate_metrics_match": False,
                    },
                },
            )

        return ToolExecutionResult(
            ok=True,
            tool_name=name,
            data={"repaired": True},
        )

    monkeypatch.setattr(registry, "execute", execute)

    def crash_before_repair_bookkeeping(
        stage: str,
    ) -> None:
        if stage == "after_tool_message_checkpoint":
            raise RuntimeError(
                "simulated crash before repair bookkeeping"
            )

    store = CheckpointStore(tmp_path / "runs")
    first_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id=verifier_tool_call_id,
                        function=FakeFunction(
                            name="verify_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            )
        ]
    )
    first_agent = AgentLoop(
        registry,
        client=first_client,
        model="fake-model",
        checkpoint_store=store,
        fault_hook=crash_before_repair_bookkeeping,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated crash before repair bookkeeping",
    ):
        first_agent.run("Verify and repair.")

    run_id = first_agent.last_checkpoint.run_id
    crashed_checkpoint = store.load(run_id)

    assert crashed_checkpoint.repair_count == 0
    assert (
        verifier_tool_call_id
        not in crashed_checkpoint.handled_verification_tool_call_ids
    )
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == verifier_tool_call_id
        for message in crashed_checkpoint.messages
    )

    resumed_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="repair_after_resume",
                        function=FakeFunction(
                            name="write_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            ),
            FakeMessage(content="Repair path continued."),
        ]
    )
    resumed_agent = AgentLoop(
        registry,
        client=resumed_client,
        model="fake-model",
        checkpoint_store=store,
    )

    assert resumed_agent.resume(run_id) == (
        "Repair path continued."
    )

    final_checkpoint = store.load(run_id)
    guidance_messages = [
        message
        for message in final_checkpoint.messages
        if message.get("role") == "system"
        and "Runtime verifier guidance" in message.get(
            "content",
            "",
        )
    ]

    assert final_checkpoint.repair_count == 1
    assert guidance_messages and len(guidance_messages) == 1
    assert (
        verifier_tool_call_id
        in final_checkpoint.handled_verification_tool_call_ids
    )
    assert execute_calls.count("verify_analysis_report") == 1
    assert execute_calls.count("write_analysis_report") == 1


def test_verifier_execution_error_does_not_request_repair(
    monkeypatch,
) -> None:
    """普通工具执行错误（包括 verifier 错误）不属于 repair 触发条件。"""

    registry = build_default_registry()
    monkeypatch.setattr(
        registry,
        "execute",
        lambda name, arguments: ToolExecutionResult(
            ok=False,
            tool_name=name,
            error="missing report",
        ),
    )

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="verify_error",
                        function=FakeFunction(
                            name="verify_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            ),
            FakeMessage(content="Report is missing."),
        ]
    )
    agent = AgentLoop(
        registry,
        client=fake_client,
        model="fake-model",
    )

    assert agent.run("Verify.") == "Report is missing."
    assert agent.last_checkpoint.repair_count == 0
    assert "repair_requested" not in [
        event.event_type
        for event in agent.last_trace.events
    ]

def test_resume_reuses_completed_tool_result(
    tmp_path,
    monkeypatch,
) -> None:
    """恢复时不得重复执行已经完成的工具调用。"""

    run_id = "f" * 32
    tool_call_id = "call_replay_001"

    checkpoint = AgentCheckpoint(
        run_id=run_id,
        next_step=1,
        messages=[
            {
                "role": "system",
                "content": "系统提示词",
            },
            {
                "role": "user",
                "content": "计算容量。",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": (
                                "calculate_shannon_capacity"
                            ),
                            "arguments": (
                                '{"bandwidth_hz": '
                                '20000000, '
                                '"snr_db": 10}'
                            ),
                        },
                    }
                ],
            },
        ],
    )

    checkpoint.record_tool_call(
        CompletedToolCall(
            tool_call_id=tool_call_id,
            tool_name=(
                "calculate_shannon_capacity"
            ),
            arguments={
                "bandwidth_hz": 20_000_000,
                "snr_db": 10,
            },
            result={
                "ok": True,
                "tool_name": (
                    "calculate_shannon_capacity"
                ),
                "data": {
                    "capacity_mbps": 69.1886,
                },
                "error": None,
            },
        )
    )

    checkpoint.mark_failed(
        "模拟工具执行后程序中断"
    )

    store = CheckpointStore(
        tmp_path / "runs"
    )
    store.save(checkpoint)

    registry = build_default_registry()

    # 如果恢复过程再次调用工具，
    # 测试应立刻失败。
    def fail_if_executed(
        *_: Any,
        **__: Any,
    ) -> Any:
        raise AssertionError(
            "已经完成的工具不应重复执行"
        )

    monkeypatch.setattr(
        registry,
        "execute",
        fail_if_executed,
    )

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="恢复任务成功。",
            )
        ]
    )

    agent = AgentLoop(
        registry,
        client=fake_client,
        model="fake-model",
        checkpoint_store=store,
    )

    answer = agent.resume(run_id)

    assert answer == "恢复任务成功。"

    # 模型收到的上下文中应该已经补上 tool message。
    request_messages = (
        fake_client.completions
        .requests[0]["messages"]
    )

    assert request_messages[-1]["role"] == (
        "tool"
    )

    assert request_messages[-1][
        "tool_call_id"
    ] == tool_call_id

def test_resume_executes_pending_tool_once(
    tmp_path,
    monkeypatch,
) -> None:
    """尚未执行的 pending tool 应在恢复时执行一次。"""

    run_id = "1" * 32
    tool_call_id = "call_pending_001"

    checkpoint = AgentCheckpoint(
        run_id=run_id,
        next_step=1,
        messages=[
            {
                "role": "system",
                "content": "系统提示词",
            },
            {
                "role": "user",
                "content": "计算容量。",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": (
                                "calculate_shannon_capacity"
                            ),
                            "arguments": (
                                '{"bandwidth_hz": '
                                '20000000, '
                                '"snr_db": 10}'
                            ),
                        },
                    }
                ],
            },
        ],
    )

    checkpoint.mark_failed(
        "模拟工具执行前程序中断"
    )

    store = CheckpointStore(
        tmp_path / "runs"
    )
    store.save(checkpoint)

    registry = build_default_registry()

    original_execute = registry.execute
    execute_calls: list[str] = []

    def counted_execute(
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        execute_calls.append(name)

        return original_execute(
            name,
            arguments,
        )

    monkeypatch.setattr(
        registry,
        "execute",
        counted_execute,
    )

    fake_client = FakeClient(
        messages=[
            FakeMessage(
                content="恢复任务成功。",
            )
        ]
    )

    agent = AgentLoop(
        registry,
        client=fake_client,
        model="fake-model",
        checkpoint_store=store,
    )

    answer = agent.resume(run_id)

    assert answer == "恢复任务成功。"

    assert execute_calls == [
        "calculate_shannon_capacity"
    ]

def test_resume_after_crash_does_not_repeat_tool(
    tmp_path,
    monkeypatch,
) -> None:
    """工具执行并写入 Checkpoint 后崩溃，恢复时不得重复执行工具。"""

    store = CheckpointStore(
        tmp_path / "runs"
    )

    registry = build_default_registry()

    original_execute = registry.execute

    execute_calls: list[str] = []

    def counted_execute(
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        execute_calls.append(name)

        return original_execute(
            name,
            arguments,
        )

    monkeypatch.setattr(
        registry,
        "execute",
        counted_execute,
    )

    def crash_after_checkpoint(
        stage: str,
    ) -> None:
        if stage == "after_tool_checkpoint":
            raise RuntimeError(
                "模拟工具执行后的程序崩溃"
            )

    first_client = FakeClient(
        messages=[
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_crash_001",
                        function=FakeFunction(
                            name=(
                                "calculate_shannon_capacity"
                            ),
                            arguments=(
                                '{"bandwidth_hz": '
                                '20000000, '
                                '"snr_db": 10}'
                            ),
                        ),
                    )
                ],
            )
        ]
    )

    first_agent = AgentLoop(
        registry,
        client=first_client,
        model="fake-model",
        checkpoint_store=store,
        fault_hook=crash_after_checkpoint,
    )

    with pytest.raises(
        RuntimeError,
        match="模拟工具执行后的程序崩溃",
    ):
        first_agent.run(
            "计算 20 MHz、10 dB 时的香农容量。"
        )

    # 第一次运行时，工具确实执行过一次。
    assert execute_calls == [
        "calculate_shannon_capacity"
    ]

    assert first_agent.last_checkpoint is not None

    run_id = first_agent.last_checkpoint.run_id

    crashed_checkpoint = store.load(run_id)

    # 即使程序随后崩溃，
    # Checkpoint 中已经存在该工具的执行结果。
    assert (
        "call_crash_001"
        in crashed_checkpoint.completed_tool_calls
    )

    second_client = FakeClient(
        messages=[
            FakeMessage(
                content="恢复成功，容量约为 69.19 Mbps。",
            )
        ]
    )

    resumed_agent = AgentLoop(
        registry,
        client=second_client,
        model="fake-model",
        checkpoint_store=store,
    )

    answer = resumed_agent.resume(run_id)

    assert answer == (
        "恢复成功，容量约为 69.19 Mbps。"
    )

    # 核心断言：
    # resume 后工具调用次数仍然只有一次。
    assert execute_calls == [
        "calculate_shannon_capacity"
    ]

    # 恢复时，应把保存的工具结果补进 messages。
    request_messages = (
        second_client
        .completions
        .requests[0]["messages"]
    )

    assert request_messages[-1]["role"] == "tool"

    assert request_messages[-1][
        "tool_call_id"
    ] == "call_crash_001"

def test_full_demo_recovers_side_effect_and_verifies(
    tmp_path,
    monkeypatch,
) -> None:
    """完整 Demo 应在副作用后崩溃，并通过 Resume 与 Verifier 完成任务。"""

    # -------------------------------------------------
    # 1. 准备一份隔离的 demo_workspace
    # -------------------------------------------------

    demo_source = (
        Path(__file__).resolve().parents[1]
        / "demo_workspace"
    )

    workspace = (
        tmp_path / "demo_workspace"
    )
    workspace.mkdir()

    for filename in [
        "config.json",
        "experiment_notes.md",
        "results.csv",
    ]:
        shutil.copy2(
            demo_source / filename,
            workspace / filename,
        )

    # -------------------------------------------------
    # 2. Trace 和 Checkpoint 共用同一个 run 目录
    # -------------------------------------------------

    runs_dir = tmp_path / "runs"

    checkpoint_store = CheckpointStore(
        runs_dir
    )

    trace_store = TraceStore(
        runs_dir
    )

    registry = build_default_registry(
        workspace
    )

    # 记录所有真实发生过的工具执行。
    original_execute = registry.execute

    execute_calls: list[str] = []

    def counted_execute(
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        execute_calls.append(name)

        return original_execute(
            name,
            arguments,
        )

    monkeypatch.setattr(
        registry,
        "execute",
        counted_execute,
    )

    # -------------------------------------------------
    # 3. 第一次运行：
    #    发现文件 → 阅读配置/说明 → 搜索 → CSV 分析
    #    → 写报告 → 崩溃
    # -------------------------------------------------

    first_client = FakeClient(
        messages=[
            # Step 1: 发现文件
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_list_files",
                        function=FakeFunction(
                            name="list_workspace_files",
                            arguments=(
                                '{"path": ".", '
                                '"recursive": true}'
                            ),
                        ),
                    )
                ],
            ),

            # Step 2: 读取配置
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_read_config",
                        function=FakeFunction(
                            name="read_text_file",
                            arguments=(
                                '{"path": "config.json"}'
                            ),
                        ),
                    )
                ],
            ),

            # Step 3: 读取实验说明
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_read_notes",
                        function=FakeFunction(
                            name="read_text_file",
                            arguments=(
                                '{"path": '
                                '"experiment_notes.md"}'
                            ),
                        ),
                    )
                ],
            ),

            # Step 4: 搜索关键规则
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_search_rules",
                        function=FakeFunction(
                            name="search_text",
                            arguments=(
                                '{"query": '
                                '"source of truth"}'
                            ),
                        ),
                    )
                ],
            ),

            # Step 5: 确定性分析 CSV
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_analyze_csv",
                        function=FakeFunction(
                            name="analyze_csv",
                            arguments=(
                                '{"path": "results.csv"}'
                            ),
                        ),
                    )
                ],
            ),

            # Step 6: 产生真实副作用——写报告
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_write_report",
                        function=FakeFunction(
                            name="write_analysis_report",
                            arguments=(
                                "{"
                                '"path": "analysis_report.md", '
                                '"experiment_name": '
                                '"wireless_link_reliability_evaluation", '
                                '"overall_status": "FAIL", '
                                '"summary": '
                                '"5 runs were analyzed and '
                                '2 runs violated configured thresholds.", '
                                '"failed_runs": '
                                '["run_003", "run_005"], '
                                '"violations": ['
                                '"run_003: throughput_mbps 74 < 80", '
                                '"run_005: throughput_mbps 79 < 80", '
                                '"run_005: latency_ms 24 > 20", '
                                '"run_005: packet_loss_pct 1.4 > 1.0"'
                                "], "
                                '"aggregate_metrics": {'
                                '"throughput_mbps": {'
                                '"count": 5, '
                                '"min": 74.0, '
                                '"max": 92.0, '
                                '"mean": 83.6'
                                "}, "
                                '"latency_ms": {'
                                '"count": 5, '
                                '"min": 12.0, '
                                '"max": 24.0, '
                                '"mean": 16.8'
                                "}, "
                                '"packet_loss_pct": {'
                                '"count": 5, '
                                '"min": 0.2, '
                                '"max": 1.4, '
                                '"mean": 0.66'
                                "}"
                                "}"
                                "}"
                            ),
                        ),
                    )
                ],
            ),
        ]
    )

    # after_tool_checkpoint 会被很多工具经过。
    # 我们只在第 6 次，也就是 write_analysis_report
    # 已执行且结果已经落盘之后制造崩溃。
    checkpoint_count = 0

    def crash_after_report_checkpoint(
        stage: str,
    ) -> None:
        nonlocal checkpoint_count

        if stage != "after_tool_checkpoint":
            return

        checkpoint_count += 1

        if checkpoint_count == 6:
            raise RuntimeError(
                "模拟报告写入后的程序崩溃"
            )

    first_agent = AgentLoop(
        registry,
        client=first_client,
        model="fake-model",
        max_steps=10,
        checkpoint_store=checkpoint_store,
        trace_store=trace_store,
        fault_hook=crash_after_report_checkpoint,
    )

    with pytest.raises(
        RuntimeError,
        match="模拟报告写入后的程序崩溃",
    ):
        first_agent.run(
            "分析实验结果，生成报告并验证报告。"
        )

    # -------------------------------------------------
    # 4. 崩溃虽然发生了，但报告应该已经真实写入磁盘
    # -------------------------------------------------

    report_path = (
        workspace / "analysis_report.md"
    )

    assert report_path.is_file()

    report_content = report_path.read_text(
        encoding="utf-8"
    )

    assert "FAIL" in report_content
    assert "run_003" in report_content
    assert "run_005" in report_content

    assert first_agent.last_checkpoint is not None

    run_id = (
        first_agent.last_checkpoint.run_id
    )

    crashed_checkpoint = (
        checkpoint_store.load(run_id)
    )

    assert (
        "call_write_report"
        in crashed_checkpoint.completed_tool_calls
    )

    # 到崩溃时，写报告工具只能真正执行过一次。
    assert execute_calls.count(
        "write_analysis_report"
    ) == 1

    # -------------------------------------------------
    # 5. 模拟程序重启并 Resume
    #
    # Resume 应：
    #   复用 write_analysis_report 的旧结果
    #   不重新写报告
    #   然后执行 Verifier
    # -------------------------------------------------

    second_client = FakeClient(
        messages=[
            # Resume 后下一步调用 Verifier
            FakeMessage(
                content="",
                tool_calls=[
                    FakeToolCall(
                        id="call_verify_report",
                        function=FakeFunction(
                            name="verify_analysis_report",
                            arguments="{}",
                        ),
                    )
                ],
            ),

            # Verifier 通过以后结束任务
            FakeMessage(
                content="SUCCESS",
            ),
        ]
    )

    resumed_agent = AgentLoop(
        registry,
        client=second_client,
        model="fake-model",
        max_steps=10,
        checkpoint_store=checkpoint_store,
        trace_store=trace_store,
    )

    answer = resumed_agent.resume(
        run_id
    )

    assert answer == "SUCCESS"

    # -------------------------------------------------
    # 6. 最关键：
    #    Resume 没有重复产生副作用
    # -------------------------------------------------

    assert execute_calls.count(
        "write_analysis_report"
    ) == 1

    assert execute_calls.count(
        "verify_analysis_report"
    ) == 1

    # -------------------------------------------------
    # 7. Verifier 必须真的给出 verification_passed=True
    # -------------------------------------------------

    assert resumed_agent.last_trace is not None

    verification_events = [
        event
        for event in resumed_agent.last_trace.events
        if (
            event.event_type == "tool_result"
            and event.details.get("tool_name")
            == "verify_analysis_report"
        )
    ]

    assert len(
        verification_events
    ) == 1

    verification_result = (
        verification_events[0]
        .details["data"]
    )

    assert (
        verification_result[
            "verification_passed"
        ]
        is True
    )

    assert (
        verification_result["errors"]
        == []
    )

    # -------------------------------------------------
    # 8. 原来的同一个 run 最终进入 completed
    # -------------------------------------------------

    final_checkpoint = (
        checkpoint_store.load(run_id)
    )

    assert (
        final_checkpoint.status
        == "completed"
    )

    assert (
        final_checkpoint.final_answer
        == "SUCCESS"
    )


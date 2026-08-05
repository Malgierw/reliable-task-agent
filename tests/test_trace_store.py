from __future__ import annotations

import pytest

from reliable_task_agent.trace import RunTrace
from reliable_task_agent.trace_store import TraceStore


def test_trace_store_saves_and_loads_trace(tmp_path) -> None:
    """Trace 应能保存到磁盘并完整读取。"""

    trace = RunTrace()

    trace.add(
        step=1,
        event_type="tool_call",
        details={
            "tool_name": "calculate_shannon_capacity",
        },
    )

    store = TraceStore(tmp_path / "runs")

    trace_path = store.save(trace)

    assert trace_path.is_file()
    assert trace_path.name == "trace.json"
    assert trace_path.parent.name == trace.run_id

    loaded_trace = store.load(trace.run_id)

    assert loaded_trace == trace
    assert store.list_run_ids() == [trace.run_id]


def test_trace_store_rejects_invalid_run_id(tmp_path) -> None:
    """不合法的 run_id 不应被用于构造文件路径。"""

    store = TraceStore(tmp_path / "runs")

    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        store.load("../outside")


def test_trace_store_reports_missing_trace(tmp_path) -> None:
    """轨迹文件不存在时，应给出明确错误。"""

    store = TraceStore(tmp_path / "runs")

    missing_run_id = "0" * 32

    with pytest.raises(
        FileNotFoundError,
        match="未找到运行轨迹",
    ):
        store.load(missing_run_id)
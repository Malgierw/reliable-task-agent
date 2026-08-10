from __future__ import annotations

import pytest

from reliable_task_agent.checkpoint import (
    AgentCheckpoint,
    CompletedToolCall,
)
from reliable_task_agent.checkpoint_store import (
    CheckpointStore,
)


def test_checkpoint_store_saves_and_loads(
    tmp_path,
) -> None:
    """Checkpoint 应能保存并完整读取。"""

    checkpoint = AgentCheckpoint(
        run_id="a" * 32,
        next_step=2,
        messages=[
            {
                "role": "user",
                "content": "分析实验结果。",
            }
        ],
    )

    checkpoint.record_tool_call(
        CompletedToolCall(
            tool_call_id="call_001",
            tool_name="read_text_file",
            arguments={
                "path": "config.json",
            },
            result={
                "ok": True,
                "data": {
                    "content": "{}",
                },
            },
        )
    )

    store = CheckpointStore(tmp_path / "runs")

    checkpoint_path = store.save(checkpoint)

    assert checkpoint_path.is_file()
    assert checkpoint_path.name == "checkpoint.json"
    assert checkpoint_path.parent.name == (
        checkpoint.run_id
    )

    assert store.exists(checkpoint.run_id) is True

    loaded_checkpoint = store.load(
        checkpoint.run_id
    )

    assert loaded_checkpoint == checkpoint


def test_checkpoint_store_overwrites_existing_state(
    tmp_path,
) -> None:
    """再次保存时，应更新同一个 Checkpoint。"""

    checkpoint = AgentCheckpoint(
        run_id="b" * 32,
        next_step=1,
    )

    store = CheckpointStore(tmp_path / "runs")

    store.save(checkpoint)

    checkpoint.advance_to(3)
    checkpoint.mark_completed("任务已经完成。")

    store.save(checkpoint)

    loaded_checkpoint = store.load(
        checkpoint.run_id
    )

    assert loaded_checkpoint.next_step == 3
    assert loaded_checkpoint.status == "completed"
    assert loaded_checkpoint.final_answer == (
        "任务已经完成。"
    )


def test_checkpoint_store_rejects_invalid_run_id(
    tmp_path,
) -> None:
    """不合法的 run_id 不应被用于读取文件。"""

    store = CheckpointStore(tmp_path / "runs")

    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        store.load("../outside")
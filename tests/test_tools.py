import pytest

from reliable_task_agent.tools.builtin import build_default_registry


def test_calculate_shannon_capacity_success() -> None:
    """正常参数应当成功计算香农容量。"""

    registry = build_default_registry()

    result = registry.execute(
        "calculate_shannon_capacity",
        {
            "bandwidth_hz": 20_000_000,
            "snr_db": 10,
        },
    )

    assert result.ok is True
    assert result.error is None
    assert result.tool_name == "calculate_shannon_capacity"

    assert result.data["capacity_mbps"] == pytest.approx(
        69.1886,
        rel=1e-4,
    )


def test_calculate_shannon_capacity_rejects_invalid_bandwidth() -> None:
    """带宽小于或等于零时，应当被参数校验拦截。"""

    registry = build_default_registry()

    result = registry.execute(
        "calculate_shannon_capacity",
        {
            "bandwidth_hz": -1,
            "snr_db": 10,
        },
    )

    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert "greater than 0" in result.error

def test_read_text_file_success(tmp_path) -> None:
    """应当能够读取工作区内的文本文件。"""

    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "这是一次仿真实验记录。",
        encoding="utf-8",
    )

    registry = build_default_registry(tmp_path)

    result = registry.execute(
        "read_text_file",
        {
            "path": "notes.txt",
            "max_chars": 1000,
        },
    )

    assert result.ok is True
    assert result.error is None
    assert result.data["path"] == "notes.txt"
    assert result.data["content"] == "这是一次仿真实验记录。"
    assert result.data["truncated"] is False


def test_read_text_file_rejects_outside_path(tmp_path) -> None:
    """不应允许读取工作区之外的文件。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text(
        "工作区外的内容",
        encoding="utf-8",
    )

    registry = build_default_registry(workspace)

    result = registry.execute(
        "read_text_file",
        {
            "path": "../outside.txt",
        },
    )

    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert "禁止读取工作区之外" in result.error
    
def test_list_workspace_files_success(
    tmp_path,
) -> None:
    """应递归列出 workspace 内的文件。"""

    (tmp_path / "config.json").write_text(
        "{}",
        encoding="utf-8",
    )

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    (
        notes_dir / "experiment.md"
    ).write_text(
        "实验说明",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "list_workspace_files",
        {
            "path": ".",
            "recursive": True,
        },
    )

    assert result.ok is True
    assert result.data is not None

    assert result.data["count"] == 2

    assert result.data["files"] == [
        "config.json",
        "notes/experiment.md",
    ]

    assert result.data["truncated"] is False

def test_list_workspace_files_non_recursive(
    tmp_path,
) -> None:
    """recursive=False 时，不应进入子目录。"""

    (tmp_path / "root.txt").write_text(
        "root",
        encoding="utf-8",
    )

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()

    (
        nested_dir / "inside.txt"
    ).write_text(
        "inside",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "list_workspace_files",
        {
            "path": ".",
            "recursive": False,
        },
    )

    assert result.ok is True

    assert result.data["files"] == [
        "root.txt",
    ]

def test_list_workspace_files_rejects_outside_path(
    tmp_path,
) -> None:
    """不得通过 .. 逃出 workspace。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()

    registry = build_default_registry(
        workspace
    )

    result = registry.execute(
        "list_workspace_files",
        {
            "path": "../outside",
        },
    )

    assert result.ok is False
    assert result.data is None
    assert result.error is not None

    assert (
        "workspace 之外"
        in result.error
    )

def test_list_workspace_files_respects_max_files(
    tmp_path,
) -> None:
    """文件数量超过 max_files 时应截断结果。"""

    for index in range(5):
        (
            tmp_path / f"file_{index}.txt"
        ).write_text(
            str(index),
            encoding="utf-8",
        )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "list_workspace_files",
        {
            "path": ".",
            "recursive": True,
            "max_files": 3,
        },
    )

    assert result.ok is True

    assert result.data["count"] == 3
    assert len(result.data["files"]) == 3
    assert result.data["truncated"] is True
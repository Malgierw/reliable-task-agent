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
    assert '"field":"bandwidth_hz"' in result.error
    assert '"code":"greater_than"' in result.error

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
    assert '"error_category":"workspace_violation"' in result.error
    
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

    assert '"error_category":"workspace_violation"' in result.error

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

def test_search_text_success(
    tmp_path,
) -> None:
    """应返回匹配文本所在文件、行号和内容。"""

    (tmp_path / "config.txt").write_text(
        "mode=test\nthreshold=0.85\n",
        encoding="utf-8",
    )

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    (
        notes_dir / "experiment.md"
    ).write_text(
        "Experiment notes\n"
        "Threshold should remain stable.\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "search_text",
        {
            "query": "threshold",
        },
    )

    assert result.ok is True
    assert result.data is not None
    assert result.data["match_count"] == 2

    assert result.data["matches"] == [
        {
            "path": "config.txt",
            "line_number": 2,
            "line": "threshold=0.85",
        },
        {
            "path": "notes/experiment.md",
            "line_number": 2,
            "line": (
                "Threshold should remain stable."
            ),
        },
    ]

def test_search_text_case_sensitive(
    tmp_path,
) -> None:
    """case_sensitive=True 时应区分大小写。"""

    (tmp_path / "notes.txt").write_text(
        "Threshold=0.8\n"
        "threshold=0.9\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "search_text",
        {
            "query": "threshold",
            "case_sensitive": True,
        },
    )

    assert result.ok is True
    assert result.data["match_count"] == 1
    assert result.data["matches"][0][
        "line"
    ] == "threshold=0.9"
    
def test_search_text_rejects_outside_path(
    tmp_path,
) -> None:
    """搜索不得逃出 workspace。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()

    registry = build_default_registry(
        workspace
    )

    result = registry.execute(
        "search_text",
        {
            "query": "secret",
            "path": "../outside",
        },
    )

    assert result.ok is False
    assert result.data is None
    assert '"error_category":"workspace_violation"' in result.error

def test_search_text_respects_max_matches(
    tmp_path,
) -> None:
    """超过 max_matches 时应截断搜索结果。"""

    (tmp_path / "data.txt").write_text(
        "error one\n"
        "error two\n"
        "error three\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "search_text",
        {
            "query": "error",
            "max_matches": 2,
        },
    )

    assert result.ok is True
    assert result.data["match_count"] == 2
    assert result.data["truncated"] is True

def test_analyze_csv_success(
    tmp_path,
) -> None:
    """应正确分析 CSV 中的数值列。"""

    (
        tmp_path / "results.csv"
    ).write_text(
        "experiment_id,throughput_mbps,latency_ms\n"
        "1,80,12\n"
        "2,90,15\n"
        "3,70,18\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "analyze_csv",
        {
            "path": "results.csv",
        },
    )

    assert result.ok is True
    assert result.data is not None

    assert result.data["row_count"] == 3

    assert result.data["columns"] == [
        "experiment_id",
        "throughput_mbps",
        "latency_ms",
    ]

    throughput = result.data[
        "numeric_summary"
    ]["throughput_mbps"]

    assert throughput["count"] == 3
    assert throughput["min"] == 70.0
    assert throughput["max"] == 90.0
    assert throughput["mean"] == pytest.approx(
        80.0
    )

def test_analyze_csv_tracks_missing_values(
    tmp_path,
) -> None:
    """应统计缺失值，并避免把文本列当成数值列。"""

    (
        tmp_path / "results.csv"
    ).write_text(
        "name,score,status\n"
        "run_a,90,ok\n"
        "run_b,,failed\n"
        "run_c,80,ok\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "analyze_csv",
        {
            "path": "results.csv",
        },
    )

    assert result.ok is True

    assert result.data[
        "missing_values"
    ]["score"] == 1

    assert "score" in result.data[
        "numeric_summary"
    ]

    assert "name" not in result.data[
        "numeric_summary"
    ]

    assert "status" not in result.data[
        "numeric_summary"
    ]

def test_analyze_csv_tracks_missing_values(
    tmp_path,
) -> None:
    """应统计缺失值，并避免把文本列当成数值列。"""

    (
        tmp_path / "results.csv"
    ).write_text(
        "name,score,status\n"
        "run_a,90,ok\n"
        "run_b,,failed\n"
        "run_c,80,ok\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "analyze_csv",
        {
            "path": "results.csv",
        },
    )

    assert result.ok is True

    assert result.data[
        "missing_values"
    ]["score"] == 1

    assert "score" in result.data[
        "numeric_summary"
    ]

    assert "name" not in result.data[
        "numeric_summary"
    ]

    assert "status" not in result.data[
        "numeric_summary"
    ]

def test_analyze_csv_rejects_outside_path(
    tmp_path,
) -> None:
    """不得通过路径逃出 workspace。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside_file = tmp_path / "secret.csv"

    outside_file.write_text(
        "value\n123\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        workspace
    )

    result = registry.execute(
        "analyze_csv",
        {
            "path": "../secret.csv",
        },
    )

    assert result.ok is False
    assert result.data is None
    assert '"error_category":"workspace_violation"' in result.error

def test_analyze_csv_respects_max_rows(
    tmp_path,
) -> None:
    """超过 max_rows 后应停止分析并标记 truncated。"""

    (
        tmp_path / "results.csv"
    ).write_text(
        "value\n"
        "1\n"
        "2\n"
        "3\n"
        "4\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "analyze_csv",
        {
            "path": "results.csv",
            "max_rows": 2,
        },
    )

    assert result.ok is True
    assert result.data["row_count"] == 2
    assert result.data["truncated"] is True

    summary = result.data[
        "numeric_summary"
    ]["value"]

    assert summary["mean"] == pytest.approx(
        1.5
    )

def test_analyze_csv_rejects_non_csv(
    tmp_path,
) -> None:
    """analyze_csv 不应读取其他文件类型。"""

    (
        tmp_path / "notes.txt"
    ).write_text(
        "hello",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "analyze_csv",
        {
            "path": "notes.txt",
        },
    )

    assert result.ok is False
    assert result.data is None
    assert '"error_category":"invalid_file_type"' in result.error

def test_write_analysis_report_success(
    tmp_path,
) -> None:
    """应根据结构化输入真正创建 Markdown 报告。"""

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "write_analysis_report",
        {
            "path": "analysis_report.md",
            "experiment_name": (
                "wireless_link_reliability_evaluation"
            ),
            "overall_status": "FAIL",
            "summary": (
                "5 runs were analyzed and "
                "2 runs violated configured thresholds."
            ),
            "failed_runs": [
                "run_003",
                "run_005",
            ],
            "violations": [
                (
                    "run_003: throughput_mbps "
                    "74 < 80"
                ),
                (
                    "run_005: throughput_mbps "
                    "79 < 80"
                ),
                (
                    "run_005: latency_ms "
                    "24 > 20"
                ),
                (
                    "run_005: packet_loss_pct "
                    "1.4 > 1.0"
                ),
            ],
            "aggregate_metrics": {
                "throughput_mbps": {
                    "count": 5,
                    "min": 74.0,
                    "max": 92.0,
                    "mean": 83.6,
                }
            },
        },
    )

    assert result.ok is True
    assert result.data is not None

    report_path = (
        tmp_path / "analysis_report.md"
    )

    assert report_path.is_file()

    content = report_path.read_text(
        encoding="utf-8"
    )

    assert "## Overall Status" in content
    assert "FAIL" in content
    assert "run_003" in content
    assert "run_005" in content
    assert "83.6" in content

def test_write_analysis_report_rejects_existing_file(
    tmp_path,
) -> None:
    """默认不得覆盖已经存在的报告。"""

    report_path = (
        tmp_path / "analysis_report.md"
    )

    report_path.write_text(
        "existing report",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "write_analysis_report",
        {
            "path": "analysis_report.md",
            "experiment_name": "demo",
            "overall_status": "FAIL",
            "summary": "summary",
        },
    )

    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert '"error_category":"file_already_exists"' in result.error

    # 原文件不能被破坏。
    assert report_path.read_text(
        encoding="utf-8"
    ) == "existing report"

def test_write_analysis_report_rejects_outside_path(
    tmp_path,
) -> None:
    """不得通过 .. 在 workspace 外写报告。"""

    workspace = (
        tmp_path / "workspace"
    )
    workspace.mkdir()

    registry = build_default_registry(
        workspace
    )

    result = registry.execute(
        "write_analysis_report",
        {
            "path": "../outside.md",
            "experiment_name": "demo",
            "overall_status": "FAIL",
            "summary": "summary",
        },
    )

    assert result.ok is False
    assert result.data is None
    assert result.error is not None

    assert '"error_category":"workspace_violation"' in result.error

    assert not (
        tmp_path / "outside.md"
    ).exists()

def test_write_analysis_report_rejects_non_markdown(
    tmp_path,
) -> None:
    """报告工具不得写入非 Markdown 文件。"""

    registry = build_default_registry(
        tmp_path
    )

    result = registry.execute(
        "write_analysis_report",
        {
            "path": "config.json",
            "experiment_name": "demo",
            "overall_status": "FAIL",
            "summary": "summary",
        },
    )

    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert '"error_category":"invalid_output_type"' in result.error

    assert not (
        tmp_path / "config.json"
    ).exists()

def test_verify_analysis_report_success(
    tmp_path,
) -> None:
    """正确报告应通过确定性验证。"""

    (
        tmp_path / "config.json"
    ).write_text(
        """
{
  "throughput_target_mbps": 80.0,
  "latency_limit_ms": 20.0,
  "packet_loss_limit_pct": 1.0,
  "required_runs": 3
}
""".strip(),
        encoding="utf-8",
    )

    (
        tmp_path / "results.csv"
    ).write_text(
        "run_id,throughput_mbps,latency_ms,"
        "packet_loss_pct,status\n"
        "run_001,90,10,0.2,ok\n"
        "run_002,70,15,0.5,warning\n"
        "run_003,85,25,1.2,failed\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    write_result = registry.execute(
        "write_analysis_report",
        {
            "experiment_name": "demo",
            "overall_status": "FAIL",
            "summary": "Two runs violated thresholds.",
            "failed_runs": [
                "run_002",
                "run_003",
            ],
            "violations": [
                "run_002 throughput violation",
                "run_003 latency and loss violation",
            ],
            "aggregate_metrics": {
                "throughput_mbps": {
                    "count": 3,
                    "min": 70,
                    "max": 90,
                    "mean": 81.66666666666667,
                },
                "latency_ms": {
                    "count": 3,
                    "min": 10,
                    "max": 25,
                    "mean": 16.666666666666668,
                },
                "packet_loss_pct": {
                    "count": 3,
                    "min": 0.2,
                    "max": 1.2,
                    "mean": (
                        0.6333333333333333
                    ),
                },
            },
        },
    )

    assert write_result.ok is True

    verify_result = registry.execute(
        "verify_analysis_report",
        {},
    )

    assert verify_result.ok is True
    assert verify_result.data is not None

    assert (
        verify_result.data[
            "verification_passed"
        ]
        is True
    )

    assert verify_result.data[
        "expected_failed_runs"
    ] == [
        "run_002",
        "run_003",
    ]

    assert verify_result.data["errors"] == []
    assert verify_result.data["error_details"] == []

def test_verify_analysis_report_rejects_wrong_failed_runs(
    tmp_path,
) -> None:
    """报告漏掉失败 run 时，Verifier 必须拒绝。"""

    (
        tmp_path / "config.json"
    ).write_text(
        """
{
  "throughput_target_mbps": 80.0,
  "latency_limit_ms": 20.0,
  "packet_loss_limit_pct": 1.0,
  "required_runs": 1
}
""".strip(),
        encoding="utf-8",
    )

    (
        tmp_path / "results.csv"
    ).write_text(
        "run_id,throughput_mbps,latency_ms,"
        "packet_loss_pct\n"
        "run_bad,70,10,0.2\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    write_result = registry.execute(
        "write_analysis_report",
        {
            "experiment_name": "demo",
            "overall_status": "FAIL",
            "summary": "summary",
            "failed_runs": [],
            "aggregate_metrics": {
                "throughput_mbps": {
                    "count": 1,
                    "min": 70,
                    "max": 70,
                    "mean": 70,
                },
                "latency_ms": {
                    "count": 1,
                    "min": 10,
                    "max": 10,
                    "mean": 10,
                },
                "packet_loss_pct": {
                    "count": 1,
                    "min": 0.2,
                    "max": 0.2,
                    "mean": 0.2,
                },
            },
        },
    )

    assert write_result.ok is True

    result = registry.execute(
        "verify_analysis_report",
        {},
    )

    assert result.ok is True

    # 工具本身成功运行，
    # 但报告验证没有通过。
    assert (
        result.data[
            "verification_passed"
        ]
        is False
    )

    assert (
        result.data["checks"][
            "failed_runs_match"
        ]
        is False
    )

def test_verify_analysis_report_rejects_wrong_metrics(
    tmp_path,
) -> None:
    """报告统计指标错误时，Verifier 必须拒绝。"""

    (
        tmp_path / "config.json"
    ).write_text(
        """
{
  "throughput_target_mbps": 80.0,
  "latency_limit_ms": 20.0,
  "packet_loss_limit_pct": 1.0,
  "required_runs": 1
}
""".strip(),
        encoding="utf-8",
    )

    (
        tmp_path / "results.csv"
    ).write_text(
        "run_id,throughput_mbps,latency_ms,"
        "packet_loss_pct\n"
        "run_001,90,10,0.2\n",
        encoding="utf-8",
    )

    registry = build_default_registry(
        tmp_path
    )

    registry.execute(
        "write_analysis_report",
        {
            "experiment_name": "demo",
            "overall_status": "PASS",
            "summary": "summary",
            "failed_runs": [],
            "aggregate_metrics": {
                "throughput_mbps": {
                    "count": 1,
                    "min": 90,
                    "max": 90,
                    "mean": 999,
                },
                "latency_ms": {
                    "count": 1,
                    "min": 10,
                    "max": 10,
                    "mean": 10,
                },
                "packet_loss_pct": {
                    "count": 1,
                    "min": 0.2,
                    "max": 0.2,
                    "mean": 0.2,
                },
            },
        },
    )

    result = registry.execute(
        "verify_analysis_report",
        {},
    )

    assert result.ok is True

    assert (
        result.data[
            "verification_passed"
        ]
        is False
    )

    assert (
        result.data["checks"][
            "aggregate_metrics_match"
        ]
        is False
    )

    assert {
        "type": "metric_value_mismatch",
        "field": "throughput_mbps.mean",
        "expected": 90.0,
        "actual": 999.0,
    } in result.data["error_details"]

def test_verify_analysis_report_rejects_outside_path(
    tmp_path,
) -> None:
    """Verifier 不得读取 workspace 之外的报告。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = build_default_registry(
        workspace
    )

    result = registry.execute(
        "verify_analysis_report",
        {
            "report_path": "../outside.md",
        },
    )

    assert result.ok is False
    assert result.data is None
    assert result.error is not None

    assert '"error_category":"workspace_violation"' in result.error

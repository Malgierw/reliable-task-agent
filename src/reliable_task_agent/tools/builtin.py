from __future__ import annotations

import csv
import math
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from reliable_task_agent.tools.registry import ToolRegistry


class ShannonCapacityArgs(BaseModel):
    """香农容量计算工具的输入参数。"""

    model_config = ConfigDict(extra="forbid")

    bandwidth_hz: float = Field(
        gt=0,
        description="信道带宽，单位为 Hz，必须大于 0。",
    )
    snr_db: float = Field(
        description="信噪比，单位为 dB。",
    )


def calculate_shannon_capacity(
    args: ShannonCapacityArgs,
) -> dict[str, float]:
    """根据带宽和信噪比计算香农理论容量。"""

    snr_linear = 10 ** (args.snr_db / 10)
    capacity_bps = (
        args.bandwidth_hz
        * math.log2(1 + snr_linear)
    )

    return {
        "bandwidth_hz": args.bandwidth_hz,
        "snr_db": args.snr_db,
        "snr_linear": snr_linear,
        "capacity_bps": capacity_bps,
        "capacity_mbps": capacity_bps / 1_000_000,
    }


class ReadTextFileArgs(BaseModel):
    """文本文件读取工具的输入参数。"""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        description="相对于工作区根目录的文本文件路径。",
    )
    max_chars: int = Field(
        default=10_000,
        ge=1,
        le=100_000,
        description="最多返回的字符数量。",
    )


def read_text_file(
    args: ReadTextFileArgs,
    workspace: Path,
) -> dict[str, object]:
    """安全读取工作区内的文本文件。"""

    workspace = workspace.resolve()
    file_path = (
        workspace / args.path
    ).resolve()

    try:
        relative_path = file_path.relative_to(
            workspace
        )
    except ValueError as exc:
        raise PermissionError(
            "禁止读取工作区之外的文件。"
        ) from exc

    if not file_path.exists():
        raise FileNotFoundError(
            f"文件不存在：{relative_path}"
        )

    if not file_path.is_file():
        raise IsADirectoryError(
            f"目标不是文件：{relative_path}"
        )

    content = file_path.read_text(
        encoding="utf-8"
    )

    truncated = (
        len(content) > args.max_chars
    )

    return {
        "path": relative_path.as_posix(),
        "content": content[: args.max_chars],
        "total_chars": len(content),
        "truncated": truncated,
    }


class ListWorkspaceFilesArgs(BaseModel):
    """列出 workspace 中的文件。"""

    model_config = ConfigDict(extra="forbid")

    path: str = "."
    recursive: bool = True
    max_files: int = Field(
        default=100,
        ge=1,
        le=1000,
    )


def list_workspace_files(
    args: ListWorkspaceFilesArgs,
    workspace: Path,
) -> dict[str, object]:
    """安全列出 workspace 中指定目录下的文件。"""

    workspace = workspace.resolve()

    target = (
        workspace / args.path
    ).resolve()

    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(
            "不允许访问 workspace 之外的路径。"
        ) from exc

    if not target.exists():
        raise FileNotFoundError(
            f"路径不存在：{args.path}"
        )

    if not target.is_dir():
        raise ValueError(
            f"目标不是目录：{args.path}"
        )

    if args.recursive:
        candidates = target.rglob("*")
    else:
        candidates = target.iterdir()

    files: list[str] = []
    truncated = False

    for path in candidates:
        if not path.is_file():
            continue

        if len(files) >= args.max_files:
            truncated = True
            break

        relative_path = path.relative_to(
            workspace
        )

        files.append(
            relative_path.as_posix()
        )

    files.sort()

    return {
        "path": args.path,
        "recursive": args.recursive,
        "files": files,
        "count": len(files),
        "truncated": truncated,
    }

class SearchTextArgs(BaseModel):
    """工作区文本搜索工具的输入参数。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description="需要搜索的文本。",
    )
    path: str = Field(
        default=".",
        description="从 workspace 中哪个相对目录开始搜索。",
    )
    recursive: bool = True
    case_sensitive: bool = False
    max_matches: int = Field(
        default=100,
        ge=1,
        le=1000,
    )

def search_text(
    args: SearchTextArgs,
    workspace: Path,
) -> dict[str, object]:
    """安全搜索 workspace 内文本文件中的内容。"""

    workspace = workspace.resolve()

    target = (
        workspace / args.path
    ).resolve()

    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(
            "不允许访问 workspace 之外的路径。"
        ) from exc

    if not target.exists():
        raise FileNotFoundError(
            f"路径不存在：{args.path}"
        )

    if not target.is_dir():
        raise ValueError(
            f"目标不是目录：{args.path}"
        )

    if args.recursive:
        candidates = target.rglob("*")
    else:
        candidates = target.iterdir()

    matches: list[dict[str, object]] = []
    files_scanned = 0
    skipped_files: list[str] = []
    truncated = False

    search_query = (
        args.query
        if args.case_sensitive
        else args.query.lower()
    )

    for path in candidates:
        if not path.is_file():
            continue

        relative_path = path.relative_to(
            workspace
        ).as_posix()

        try:
            content = path.read_text(
                encoding="utf-8"
            )
        except (UnicodeDecodeError, OSError):
            skipped_files.append(relative_path)
            continue

        files_scanned += 1

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):
            searchable_line = (
                line
                if args.case_sensitive
                else line.lower()
            )

            if search_query not in searchable_line:
                continue

            if len(matches) >= args.max_matches:
                truncated = True
                break

            matches.append(
                {
                    "path": relative_path,
                    "line_number": line_number,
                    "line": line,
                }
            )

        if truncated:
            break

    return {
        "query": args.query,
        "path": args.path,
        "recursive": args.recursive,
        "case_sensitive": args.case_sensitive,
        "matches": matches,
        "match_count": len(matches),
        "files_scanned": files_scanned,
        "skipped_files": skipped_files,
        "truncated": truncated,
    }

class AnalyzeCsvArgs(BaseModel):
    """CSV 实验结果分析工具的输入参数。"""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        description="相对于 workspace 的 CSV 文件路径。",
    )

    max_rows: int = Field(
        default=10_000,
        ge=1,
        le=100_000,
        description="最多分析的数据行数。",
    )

def analyze_csv(
    args: AnalyzeCsvArgs,
    workspace: Path,
) -> dict[str, object]:
    """安全读取并分析 workspace 内的 CSV 文件。"""

    workspace = workspace.resolve()

    file_path = (
        workspace / args.path
    ).resolve()

    try:
        relative_path = file_path.relative_to(
            workspace
        )
    except ValueError as exc:
        raise ValueError(
            "不允许访问 workspace 之外的路径。"
        ) from exc

    if not file_path.exists():
        raise FileNotFoundError(
            f"文件不存在：{args.path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"目标不是文件：{args.path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "analyze_csv 只允许分析 .csv 文件。"
        )

    rows: list[dict[str, str]] = []
    truncated = False

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "CSV 文件缺少表头。"
            )

        columns = list(reader.fieldnames)

        for row in reader:
            if len(rows) >= args.max_rows:
                truncated = True
                break

            rows.append(
                {
                    key: (
                        value
                        if value is not None
                        else ""
                    )
                    for key, value in row.items()
                }
            )

    missing_values: dict[str, int] = {
        column: 0
        for column in columns
    }

    numeric_values: dict[str, list[float]] = {
        column: []
        for column in columns
    }

    non_empty_counts: dict[str, int] = {
        column: 0
        for column in columns
    }

    numeric_counts: dict[str, int] = {
        column: 0
        for column in columns
    }

    for row in rows:
        for column in columns:
            raw_value = row.get(
                column,
                "",
            ).strip()

            if raw_value == "":
                missing_values[column] += 1
                continue

            non_empty_counts[column] += 1

            try:
                numeric_value = float(
                    raw_value
                )
            except ValueError:
                continue

            numeric_values[column].append(
                numeric_value
            )
            numeric_counts[column] += 1

    numeric_summary: dict[
        str,
        dict[str, float | int],
    ] = {}

    for column in columns:
        values = numeric_values[column]

        # 只有所有非空值都能转换成数字，
        # 才把这一列认定为数值列。
        if (
            not values
            or numeric_counts[column]
            != non_empty_counts[column]
        ):
            continue

        numeric_summary[column] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "path": relative_path.as_posix(),
        "columns": columns,
        "row_count": len(rows),
        "missing_values": missing_values,
        "numeric_summary": numeric_summary,
        "truncated": truncated,
    }

class WriteAnalysisReportArgs(BaseModel):
    """分析报告写入工具的输入参数。"""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        default="analysis_report.md",
        min_length=1,
        description="相对于 workspace 的 Markdown 报告路径。",
    )

    experiment_name: str = Field(
        min_length=1,
        description="实验名称。",
    )

    overall_status: str = Field(
        min_length=1,
        description="总体结论，例如 PASS 或 FAIL。",
    )

    summary: str = Field(
        min_length=1,
        description="实验结果摘要。",
    )

    failed_runs: list[str] = Field(
        default_factory=list,
        description="违反验收条件的 run_id。",
    )

    violations: list[str] = Field(
        default_factory=list,
        description="具体违反条件的说明。",
    )

    aggregate_metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="确定性工具计算得到的聚合指标。",
    )

    overwrite: bool = Field(
        default=False,
        description="是否允许覆盖已经存在的报告。",
    )

def write_analysis_report(
    args: WriteAnalysisReportArgs,
    workspace: Path,
) -> dict[str, object]:
    """安全地将结构化分析结果写入 Markdown 报告。"""

    workspace = workspace.resolve()

    report_path = (
        workspace / args.path
    ).resolve()

    try:
        relative_path = report_path.relative_to(
            workspace
        )
    except ValueError as exc:
        raise ValueError(
            "不允许在 workspace 之外写入文件。"
        ) from exc

    if report_path.suffix.lower() != ".md":
        raise ValueError(
            "分析报告必须是 .md 文件。"
        )

    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"报告已经存在：{relative_path.as_posix()}"
        )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = [
        f"# Analysis Report: {args.experiment_name}",
        "",
        "## Overall Status",
        "",
        args.overall_status,
        "",
        "## Summary",
        "",
        args.summary,
        "",
        "## Failed Runs",
        "",
    ]

    if args.failed_runs:
        for run_id in args.failed_runs:
            lines.append(
                f"- {run_id}"
            )
    else:
        lines.append(
            "- None"
        )

    lines.extend(
        [
            "",
            "## Violations",
            "",
        ]
    )

    if args.violations:
        for violation in args.violations:
            lines.append(
                f"- {violation}"
            )
    else:
        lines.append(
            "- None"
        )

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
        ]
    )

    if args.aggregate_metrics:
        for metric_name in sorted(
            args.aggregate_metrics
        ):
            metrics = args.aggregate_metrics[
                metric_name
            ]

            lines.append(
                f"### {metric_name}"
            )
            lines.append("")

            for key in (
                "count",
                "min",
                "max",
                "mean",
            ):
                if key in metrics:
                    lines.append(
                        f"- {key}: {metrics[key]}"
                    )

            lines.append("")
    else:
        lines.append(
            "No aggregate metrics provided."
        )
        lines.append("")

    content = "\n".join(lines)

    temporary_path = report_path.with_suffix(
        report_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        content,
        encoding="utf-8",
    )

    temporary_path.replace(
        report_path
    )

    return {
        "path": relative_path.as_posix(),
        "bytes_written": len(
            content.encode("utf-8")
        ),
        "failed_run_count": len(
            args.failed_runs
        ),
        "overall_status": args.overall_status,
    }

class VerifyAnalysisReportArgs(BaseModel):
    """分析报告确定性验证工具的输入参数。"""

    model_config = ConfigDict(extra="forbid")

    report_path: str = Field(
        default="analysis_report.md",
        min_length=1,
        description="待验证的 Markdown 报告路径。",
    )

    config_path: str = Field(
        default="config.json",
        min_length=1,
        description="实验验收配置路径。",
    )

    results_path: str = Field(
        default="results.csv",
        min_length=1,
        description="实验结果 CSV 路径。",
    )

    tolerance: float = Field(
        default=1e-6,
        gt=0,
        le=1,
        description="数值指标比较时使用的绝对误差容限。",
    )

def _parse_analysis_report(
    content: str,
) -> dict[str, object]:
    """解析 write_analysis_report 生成的固定格式报告。"""

    lines = content.splitlines()

    reported_status: str | None = None
    failed_runs: list[str] = []
    aggregate_metrics: dict[
        str,
        dict[str, float],
    ] = {}

    current_section: str | None = None
    current_metric: str | None = None

    for raw_line in lines:
        line = raw_line.strip()

        if line == "## Overall Status":
            current_section = "status"
            current_metric = None
            continue

        if line == "## Failed Runs":
            current_section = "failed_runs"
            current_metric = None
            continue

        if line == "## Aggregate Metrics":
            current_section = "aggregate_metrics"
            current_metric = None
            continue

        if line.startswith("## "):
            current_section = None
            current_metric = None
            continue

        if not line:
            continue

        if (
            current_section == "status"
            and reported_status is None
        ):
            reported_status = line
            continue

        if current_section == "failed_runs":
            if line.startswith("- "):
                value = line[2:].strip()

                if value != "None":
                    failed_runs.append(value)

            continue

        if current_section == "aggregate_metrics":
            if line.startswith("### "):
                current_metric = line[4:].strip()

                aggregate_metrics[
                    current_metric
                ] = {}

                continue

            if (
                current_metric is not None
                and line.startswith("- ")
                and ":" in line
            ):
                key, value = line[2:].split(
                    ":",
                    maxsplit=1,
                )

                try:
                    aggregate_metrics[
                        current_metric
                    ][key.strip()] = float(
                        value.strip()
                    )
                except ValueError:
                    continue

    return {
        "overall_status": reported_status,
        "failed_runs": failed_runs,
        "aggregate_metrics": aggregate_metrics,
    }

def verify_analysis_report(
    args: VerifyAnalysisReportArgs,
    workspace: Path,
) -> dict[str, object]:
    """根据原始配置和 CSV 确定性验证分析报告。"""

    workspace = workspace.resolve()

    def resolve_inside_workspace(
        relative_path: str,
    ) -> Path:
        path = (
            workspace / relative_path
        ).resolve()

        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                "不允许访问 workspace 之外的路径。"
            ) from exc

        return path

    report_path = resolve_inside_workspace(
        args.report_path
    )
    config_path = resolve_inside_workspace(
        args.config_path
    )
    results_path = resolve_inside_workspace(
        args.results_path
    )

    if not report_path.is_file():
        raise FileNotFoundError(
            f"报告不存在：{args.report_path}"
        )

    if report_path.suffix.lower() != ".md":
        raise ValueError(
            "待验证报告必须是 .md 文件。"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在：{args.config_path}"
        )

    if not results_path.is_file():
        raise FileNotFoundError(
            f"结果文件不存在：{args.results_path}"
        )

    if results_path.suffix.lower() != ".csv":
        raise ValueError(
            "实验结果必须是 .csv 文件。"
        )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    throughput_target = float(
        config["throughput_target_mbps"]
    )
    latency_limit = float(
        config["latency_limit_ms"]
    )
    packet_loss_limit = float(
        config["packet_loss_limit_pct"]
    )
    required_runs = int(
        config["required_runs"]
    )

    rows: list[dict[str, str]] = []

    with results_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "run_id",
            "throughput_mbps",
            "latency_ms",
            "packet_loss_pct",
        }

        fieldnames = set(
            reader.fieldnames or []
        )

        if not required_columns.issubset(
            fieldnames
        ):
            raise ValueError(
                "CSV 缺少 Verifier 所需的列。"
            )

        rows = list(reader)

    failed_runs: list[str] = []

    metric_values: dict[
        str,
        list[float],
    ] = {
        "throughput_mbps": [],
        "latency_ms": [],
        "packet_loss_pct": [],
    }

    for row in rows:
        run_id = row["run_id"]

        throughput = float(
            row["throughput_mbps"]
        )
        latency = float(
            row["latency_ms"]
        )
        packet_loss = float(
            row["packet_loss_pct"]
        )

        metric_values[
            "throughput_mbps"
        ].append(throughput)

        metric_values[
            "latency_ms"
        ].append(latency)

        metric_values[
            "packet_loss_pct"
        ].append(packet_loss)

        violated = (
            throughput < throughput_target
            or latency > latency_limit
            or packet_loss > packet_loss_limit
        )

        if violated:
            failed_runs.append(run_id)

    row_count_matches = (
        len(rows) == required_runs
    )

    expected_status = (
        "PASS"
        if row_count_matches
        and not failed_runs
        else "FAIL"
    )

    expected_metrics: dict[
        str,
        dict[str, float],
    ] = {}

    for metric_name, values in (
        metric_values.items()
    ):
        if not values:
            continue

        expected_metrics[metric_name] = {
            "count": float(len(values)),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    report_content = report_path.read_text(
        encoding="utf-8"
    )

    parsed_report = _parse_analysis_report(
        report_content
    )

    reported_status = parsed_report[
        "overall_status"
    ]

    reported_failed_runs = parsed_report[
        "failed_runs"
    ]

    reported_metrics = parsed_report[
        "aggregate_metrics"
    ]

    status_matches = (
        reported_status == expected_status
    )

    failed_runs_match = (
        set(reported_failed_runs)
        == set(failed_runs)
    )

    metrics_match = True
    metric_errors: list[str] = []

    for (
        metric_name,
        expected_summary,
    ) in expected_metrics.items():

        reported_summary = (
            reported_metrics.get(
                metric_name
            )
        )

        if reported_summary is None:
            metrics_match = False
            metric_errors.append(
                f"报告缺少指标：{metric_name}"
            )
            continue

        for key, expected_value in (
            expected_summary.items()
        ):
            reported_value = (
                reported_summary.get(key)
            )

            if reported_value is None:
                metrics_match = False
                metric_errors.append(
                    f"{metric_name} 缺少 {key}"
                )
                continue

            if not math.isclose(
                float(reported_value),
                float(expected_value),
                abs_tol=args.tolerance,
                rel_tol=0.0,
            ):
                metrics_match = False
                metric_errors.append(
                    (
                        f"{metric_name}.{key} "
                        f"应为 {expected_value}，"
                        f"报告中为 {reported_value}"
                    )
                )

    verification_passed = (
        row_count_matches
        and status_matches
        and failed_runs_match
        and metrics_match
    )

    errors: list[str] = []

    if not row_count_matches:
        errors.append(
            (
                f"实验行数应为 {required_runs}，"
                f"实际为 {len(rows)}"
            )
        )

    if not status_matches:
        errors.append(
            (
                f"总体状态应为 {expected_status}，"
                f"报告中为 {reported_status}"
            )
        )

    if not failed_runs_match:
        errors.append(
            (
                "失败 run 集合不匹配："
                f"应为 {failed_runs}，"
                f"报告中为 {reported_failed_runs}"
            )
        )

    errors.extend(metric_errors)

    return {
        "verification_passed": (
            verification_passed
        ),
        "expected_status": expected_status,
        "reported_status": reported_status,
        "expected_failed_runs": failed_runs,
        "reported_failed_runs": (
            reported_failed_runs
        ),
        "checks": {
            "row_count_matches_config": (
                row_count_matches
            ),
            "overall_status_matches": (
                status_matches
            ),
            "failed_runs_match": (
                failed_runs_match
            ),
            "aggregate_metrics_match": (
                metrics_match
            ),
        },
        "errors": errors,
    }

def build_default_registry(
    workspace: str | Path = ".",
) -> ToolRegistry:
    """创建并返回装有默认工具的注册中心。"""

    registry = ToolRegistry()
    workspace_path = Path(workspace).resolve()

    registry.register(
        name="calculate_shannon_capacity",
        description=(
            "根据给定的信道带宽和信噪比，"
            "计算香农理论信道容量。"
        ),
        args_model=ShannonCapacityArgs,
        handler=calculate_shannon_capacity,
    )

    def handle_read_text_file(
        args: ReadTextFileArgs,
    ) -> dict[str, object]:
        return read_text_file(
            args,
            workspace_path,
        )

    registry.register(
        name="read_text_file",
        description=(
            "读取工作区内的 UTF-8 文本文件。"
            "不能读取工作区之外的路径。"
        ),
        args_model=ReadTextFileArgs,
        handler=handle_read_text_file,
    )

    def handle_list_workspace_files(
        args: ListWorkspaceFilesArgs,
    ) -> dict[str, object]:
        return list_workspace_files(
            args,
            workspace_path,
        )

    registry.register(
        name="list_workspace_files",
        description=(
            "列出工作区中的文件。"
            "可以指定相对目录、是否递归以及最大文件数。"
            "不能访问 workspace 之外的路径。"
        ),
        args_model=ListWorkspaceFilesArgs,
        handler=handle_list_workspace_files,
    )

    def handle_search_text(
        args: SearchTextArgs,
    ) -> dict[str, object]:
        return search_text(
            args,
            workspace_path,
        )
        
    registry.register(
        name="search_text",
        description=(
            "在工作区的 UTF-8 文本文件中搜索指定文本，"
            "返回匹配文件、行号和对应文本。"
            "支持递归搜索和大小写控制，"
            "不能访问 workspace 之外的路径。"
        ),
        args_model=SearchTextArgs,
        handler=handle_search_text,
    )

    def handle_analyze_csv(
        args: AnalyzeCsvArgs,
    ) -> dict[str, object]:
        return analyze_csv(
            args,
            workspace_path,
        )
        
    registry.register(
        name="analyze_csv",
        description=(
            "分析工作区内的 CSV 数据文件，"
            "返回列名、行数、缺失值统计以及"
            "数值列的 count、min、max 和 mean。"
            "不能访问 workspace 之外的路径。"
        ),
        args_model=AnalyzeCsvArgs,
        handler=handle_analyze_csv,
    )
    
    def handle_write_analysis_report(
        args: WriteAnalysisReportArgs,
    ) -> dict[str, object]:
        return write_analysis_report(
            args,
            workspace_path,
        )

    registry.register(
        name="write_analysis_report",
        description=(
            "将结构化实验分析结果写入 workspace 内的 Markdown 报告。"
            "这是一个具有文件写入副作用的工具。"
            "默认不覆盖已有文件，且不能写入 workspace 之外。"
        ),
        args_model=WriteAnalysisReportArgs,
        handler=handle_write_analysis_report,
    )

    def handle_verify_analysis_report(
        args: VerifyAnalysisReportArgs,
    ) -> dict[str, object]:
        return verify_analysis_report(
            args,
            workspace_path,
        )

    registry.register(
        name="verify_analysis_report",
        description=(
            "确定性验证分析报告。"
            "重新读取实验配置和 CSV 原始结果，"
            "独立计算正确状态、失败 run 和聚合指标，"
            "再与 Markdown 报告比较。"
            "verification_passed 决定报告是否真正通过验收。"
        ),
        args_model=VerifyAnalysisReportArgs,
        handler=handle_verify_analysis_report,
    )
    
    return registry



from __future__ import annotations

import csv
import math
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

    return registry



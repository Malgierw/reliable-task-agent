from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.checkpoint import AgentCheckpoint
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.tools.builtin import build_default_registry
from reliable_task_agent.trace_store import TraceStore


app = typer.Typer(
    no_args_is_help=True,
    help="Reliable Task Agent CLI",
)

console = Console()


DEMO_TASK = """
分析当前 workspace 中的无线链路实验。

要求：
1. 先发现 workspace 中有哪些文件；
2. 阅读实验配置和实验说明；
3. 必要时搜索实验规则；
4. 使用 analyze_csv 确定性分析 results.csv；
5. 根据 config.json 中的阈值判断失败的 run；
6. 使用 write_analysis_report 生成 analysis_report.md；
7. 必须调用 verify_analysis_report 验证报告；
8. 只有 verification_passed=true 时才能认为任务成功。

不要仅根据 results.csv 中的 status 列判断实验是否通过。
""".strip()


def _find_verification_result(
    checkpoint: AgentCheckpoint,
) -> dict[str, Any] | None:
    """从 Checkpoint 中寻找最近一次 Verifier 的结果。"""

    calls = list(
        checkpoint.completed_tool_calls.values()
    )

    for tool_call in reversed(calls):
        if (
            tool_call.tool_name
            != "verify_analysis_report"
        ):
            continue

        result = tool_call.result

        if not result.get("ok"):
            return None

        data = result.get("data")

        if isinstance(data, dict):
            return data

    return None


def _require_successful_verification(
    checkpoint: AgentCheckpoint,
) -> None:
    """只有 Deterministic Verifier 通过才认为任务成功。"""

    verification = _find_verification_result(
        checkpoint
    )

    if verification is None:
        console.print(
            "[red]FAILED:[/red] "
            "任务结束，但没有找到有效的 "
            "verify_analysis_report 结果。"
        )
        raise typer.Exit(code=2)

    if (
        verification.get(
            "verification_passed"
        )
        is not True
    ):
        console.print(
            "[red]VERIFICATION FAILED[/red]"
        )

        errors = verification.get(
            "errors",
            [],
        )

        for error in errors:
            console.print(
                f"- {error}"
            )

        raise typer.Exit(code=1)

    console.print(
        "[green]VERIFICATION PASSED[/green]"
    )


def _build_agent(
    *,
    workspace: Path,
    runs_dir: Path,
) -> AgentLoop:
    """创建 CLI 使用的 Agent。"""

    registry = build_default_registry(
        workspace
    )

    checkpoint_store = CheckpointStore(
        runs_dir
    )

    trace_store = TraceStore(
        runs_dir
    )

    return AgentLoop(
        registry,
        checkpoint_store=checkpoint_store,
        trace_store=trace_store,
        max_steps=12,
    )


@app.command()
def demo(
    workspace: Path = typer.Option(
        Path("demo_workspace"),
        "--workspace",
        "-w",
        help="Demo workspace 路径。",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    runs_dir: Path = typer.Option(
        Path("runs"),
        "--runs-dir",
        help="Trace 和 Checkpoint 保存目录。",
    ),
) -> None:
    """运行完整实验分析 Demo。"""

    agent = _build_agent(
        workspace=workspace,
        runs_dir=runs_dir,
    )

    console.print(
        "[bold]Starting Reliable Task Agent demo...[/bold]"
    )

    try:
        answer = agent.run(DEMO_TASK)

    except Exception as exc:
        console.print(
            f"[red]Task interrupted:[/red] {exc}"
        )

        if agent.last_checkpoint is not None:
            run_id = (
                agent.last_checkpoint.run_id
            )

            console.print(
                f"run_id: {run_id}"
            )

            console.print(
                "Resume with:"
            )

            console.print(
                "uv run reliable-task-agent "
                f"resume {run_id}"
            )

        raise typer.Exit(code=1) from exc

    if agent.last_checkpoint is None:
        raise RuntimeError(
            "任务结束后缺少 Checkpoint。"
        )

    _require_successful_verification(
        agent.last_checkpoint
    )

    console.print()
    console.print(
        "[bold green]SUCCESS[/bold green]"
    )
    console.print(
        f"run_id: {agent.last_checkpoint.run_id}"
    )
    console.print(
        f"Agent answer: {answer}"
    )


@app.command()
def resume(
    run_id: str = typer.Argument(
        ...,
        help="需要恢复的 run_id。",
    ),
    workspace: Path = typer.Option(
        Path("demo_workspace"),
        "--workspace",
        "-w",
        help="原任务对应的 workspace。",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    runs_dir: Path = typer.Option(
        Path("runs"),
        "--runs-dir",
        help="Trace 和 Checkpoint 保存目录。",
    ),
) -> None:
    """根据 run_id 恢复之前的任务。"""

    agent = _build_agent(
        workspace=workspace,
        runs_dir=runs_dir,
    )

    console.print(
        f"Resuming run: {run_id}"
    )

    try:
        answer = agent.resume(run_id)

    except Exception as exc:
        console.print(
            f"[red]Resume failed:[/red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    if agent.last_checkpoint is None:
        raise RuntimeError(
            "恢复结束后缺少 Checkpoint。"
        )

    _require_successful_verification(
        agent.last_checkpoint
    )

    console.print()
    console.print(
        "[bold green]SUCCESS[/bold green]"
    )
    console.print(
        f"run_id: {run_id}"
    )
    console.print(
        f"Agent answer: {answer}"
    )


if __name__ == "__main__":
    app()
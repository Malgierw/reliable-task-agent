from __future__ import annotations

import re
from pathlib import Path

from reliable_task_agent.trace import RunTrace


RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class TraceStore:
    """负责将 Agent 执行轨迹保存到磁盘并重新读取。"""

    def __init__(
        self,
        root_dir: str | Path = "runs",
    ) -> None:
        self.root_dir = Path(root_dir).resolve()

    def _get_run_dir(self, run_id: str) -> Path:
        """校验 run_id，并返回该任务的保存目录。"""

        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                "run_id 必须是由 32 个小写十六进制字符组成的字符串。"
            )

        return self.root_dir / run_id

    def save(self, trace: RunTrace) -> Path:
        """将一条完整 Trace 保存为 JSON 文件。"""

        run_dir = self._get_run_dir(trace.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        trace_path = run_dir / "trace.json"
        temporary_path = run_dir / "trace.json.tmp"

        # 先写临时文件，写完后再替换正式文件，
        # 避免写入中断时留下不完整的 trace.json。
        temporary_path.write_text(
            trace.to_pretty_json(),
            encoding="utf-8",
        )
        temporary_path.replace(trace_path)

        return trace_path

    def load(self, run_id: str) -> RunTrace:
        """根据 run_id 读取一条已经保存的 Trace。"""

        trace_path = self._get_run_dir(run_id) / "trace.json"

        if not trace_path.is_file():
            raise FileNotFoundError(
                f"未找到运行轨迹：{run_id}"
            )

        content = trace_path.read_text(encoding="utf-8")

        return RunTrace.model_validate_json(content)

    def list_run_ids(self) -> list[str]:
        """列出当前保存的所有有效 run_id。"""

        if not self.root_dir.exists():
            return []

        run_ids: list[str] = []

        for path in self.root_dir.iterdir():
            if (
                path.is_dir()
                and RUN_ID_PATTERN.fullmatch(path.name)
                and (path / "trace.json").is_file()
            ):
                run_ids.append(path.name)

        return sorted(run_ids)
from __future__ import annotations

from pathlib import Path

from reliable_task_agent.checkpoint import AgentCheckpoint
from reliable_task_agent.trace_store import RUN_ID_PATTERN


class CheckpointStore:
    """负责保存和读取 Agent Checkpoint。"""

    def __init__(
        self,
        root_dir: str | Path = "runs",
    ) -> None:
        self.root_dir = Path(root_dir).resolve()

    def _get_run_dir(self, run_id: str) -> Path:
        """校验 run_id，并返回对应的任务目录。"""

        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                "run_id 必须是由 32 个小写十六进制字符组成的字符串。"
            )

        return self.root_dir / run_id

    def save(
        self,
        checkpoint: AgentCheckpoint,
    ) -> Path:
        """将 Checkpoint 保存为 JSON 文件。"""

        run_dir = self._get_run_dir(checkpoint.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = run_dir / "checkpoint.json"
        temporary_path = run_dir / "checkpoint.json.tmp"

        temporary_path.write_text(
            checkpoint.model_dump_json(indent=2),
            encoding="utf-8",
        )

        temporary_path.replace(checkpoint_path)

        return checkpoint_path

    def load(self, run_id: str) -> AgentCheckpoint:
        """根据 run_id 读取 Checkpoint。"""

        checkpoint_path = (
            self._get_run_dir(run_id)
            / "checkpoint.json"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"未找到任务检查点：{run_id}"
            )

        content = checkpoint_path.read_text(
            encoding="utf-8"
        )

        return AgentCheckpoint.model_validate_json(
            content
        )

    def exists(self, run_id: str) -> bool:
        """判断某个运行是否存在 Checkpoint。"""

        checkpoint_path = (
            self._get_run_dir(run_id)
            / "checkpoint.json"
        )

        return checkpoint_path.is_file()
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from reliable_task_agent.effects import ReconciliationResult
from reliable_task_agent.tools.registry import ToolRegistry


def utc_now_iso() -> str:
    """返回可持久化的 UTC 时间。"""

    return datetime.now(timezone.utc).isoformat()


class CreateTicketArgs(BaseModel):
    """创建演示工单所需的业务参数。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


class TicketStore:
    """与 Runtime Effect Ledger 分离的 SQLite 业务数据库。"""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _receipt(row: sqlite3.Row) -> dict[str, object]:
        return {
            "ticket_id": row["ticket_id"],
            "idempotency_key": row["idempotency_key"],
            "title": row["title"],
            "description": row["description"],
            "created_at": row["created_at"],
        }

    def create(
        self,
        args: CreateTicketArgs,
        idempotency_key: str,
    ) -> dict[str, object]:
        """用业务唯一键创建或读取同一张工单。"""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO tickets (
                    idempotency_key,
                    title,
                    description,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    args.title,
                    args.description,
                    utc_now_iso(),
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM tickets
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

        if row is None:
            raise RuntimeError(
                "Ticket 插入提交后无法读取业务 receipt。"
            )
        return self._receipt(row)

    def reconcile(
        self,
        _: CreateTicketArgs,
        idempotency_key: str,
    ) -> ReconciliationResult:
        """按业务唯一键确定外部工单是否已经存在。"""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT * FROM tickets
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
        except sqlite3.Error as exc:
            return ReconciliationResult(
                status="UNKNOWN",
                reason=f"Ticket reconciliation failed: {exc}",
            )

        if row is None:
            return ReconciliationResult(
                status="NOT_FOUND"
            )

        return ReconciliationResult(
            status="FOUND",
            receipt=self._receipt(row),
        )

    def count(self) -> int:
        """返回业务工单数量，供演示和测试断言使用。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM tickets"
            ).fetchone()

        if row is None:
            return 0
        return int(row["count"])


def register_ticket_tool(
    registry: ToolRegistry,
    database_path: str | Path,
) -> TicketStore:
    """注册 opt-in 的 SQLite effect-managed 工单工具。"""

    ticket_store = TicketStore(database_path)
    registry.register_effect(
        name="create_ticket",
        description=(
            "在 SQLite 业务系统中创建一张工单。"
            "该工具具有外部副作用，只能由 Runtime Effect Boundary 执行。"
        ),
        args_model=CreateTicketArgs,
        execute=ticket_store.create,
        reconcile=ticket_store.reconcile,
    )
    return ticket_store

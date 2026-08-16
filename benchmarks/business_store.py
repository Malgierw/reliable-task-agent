from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

from benchmarks.identity import payload_hash


WriteMode = Literal["plain", "idempotent"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BenchmarkBusinessStore:
    """Shared SQLite workload and durable invocation instrumentation."""

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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logical_action_id TEXT NOT NULL,
                    idempotency_key TEXT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS
                    uq_tickets_idempotency_key
                ON tickets(idempotency_key)
                WHERE idempotency_key IS NOT NULL;

                CREATE TABLE IF NOT EXISTS handler_invocations (
                    invocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logical_action_id TEXT NOT NULL,
                    worker_phase TEXT NOT NULL,
                    process_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reconciliation_invocations (
                    invocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logical_action_id TEXT NOT NULL,
                    worker_phase TEXT NOT NULL,
                    process_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _receipt(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "ticket_id": row["ticket_id"],
            "logical_action_id": row["logical_action_id"],
            "idempotency_key": row["idempotency_key"],
            "title": row["title"],
            "description": row["description"],
            "payload_hash": row["payload_hash"],
            "created_at": row["created_at"],
        }

    def record_handler_invocation(
        self,
        *,
        logical_action_id: str,
        worker_phase: str,
        process_id: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO handler_invocations (
                    logical_action_id,
                    worker_phase,
                    process_id,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    logical_action_id,
                    worker_phase,
                    process_id,
                    utc_now_iso(),
                ),
            )

    def create_ticket(
        self,
        *,
        payload: dict[str, str],
        logical_action_id: str,
        idempotency_key: str | None,
        write_mode: WriteMode,
        worker_phase: str,
        process_id: int,
        before_write: Callable[[], None] | None = None,
        after_write: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        self.record_handler_invocation(
            logical_action_id=logical_action_id,
            worker_phase=worker_phase,
            process_id=process_id,
        )
        if before_write is not None:
            before_write()
        digest = payload_hash(payload)

        with self._connect() as connection:
            if write_mode == "plain":
                connection.execute(
                    """
                    INSERT INTO tickets (
                        logical_action_id,
                        idempotency_key,
                        title,
                        description,
                        payload_hash,
                        created_at
                    ) VALUES (?, NULL, ?, ?, ?, ?)
                    """,
                    (
                        logical_action_id,
                        payload["title"],
                        payload["description"],
                        digest,
                        utc_now_iso(),
                    ),
                )
                row = connection.execute(
                    """
                    SELECT * FROM tickets
                    WHERE ticket_id = last_insert_rowid()
                    """
                ).fetchone()
            else:
                if not idempotency_key:
                    raise ValueError(
                        "Idempotent writes require an idempotency key."
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO tickets (
                        logical_action_id,
                        idempotency_key,
                        title,
                        description,
                        payload_hash,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        logical_action_id,
                        idempotency_key,
                        payload["title"],
                        payload["description"],
                        digest,
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
            raise RuntimeError("Committed ticket could not be read.")

        receipt = self._receipt(row)
        expected = {
            "logical_action_id": logical_action_id,
            "title": payload["title"],
            "description": payload["description"],
            "payload_hash": digest,
        }
        actual = {name: receipt[name] for name in expected}
        if actual != expected:
            raise RuntimeError(
                "Idempotency key refers to a different ticket payload."
            )
        if after_write is not None:
            after_write()
        return receipt

    def record_reconciliation_invocation(
        self,
        *,
        logical_action_id: str,
        worker_phase: str,
        process_id: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO reconciliation_invocations (
                    logical_action_id,
                    worker_phase,
                    process_id,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    logical_action_id,
                    worker_phase,
                    process_id,
                    utc_now_iso(),
                ),
            )

    def reconcile(
        self,
        *,
        payload: dict[str, str],
        logical_action_id: str,
        idempotency_key: str,
        worker_phase: str,
        process_id: int,
    ) -> dict[str, Any]:
        self.record_reconciliation_invocation(
            logical_action_id=logical_action_id,
            worker_phase=worker_phase,
            process_id=process_id,
        )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM tickets
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

        if row is None:
            return {"status": "NOT_FOUND"}

        receipt = self._receipt(row)
        expected = {
            "logical_action_id": logical_action_id,
            "title": payload["title"],
            "description": payload["description"],
            "payload_hash": payload_hash(payload),
        }
        actual = {name: receipt[name] for name in expected}
        if actual != expected:
            return {
                "status": "UNKNOWN",
                "reason": "Existing ticket payload does not match.",
            }
        return {"status": "FOUND", "receipt": receipt}

    def ticket_rows(
        self,
        logical_action_id: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM tickets
                WHERE logical_action_id = ?
                ORDER BY ticket_id
                """,
                (logical_action_id,),
            ).fetchall()
        return [self._receipt(row) for row in rows]

    def handler_invocation_count(
        self,
        logical_action_id: str,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM handler_invocations
                WHERE logical_action_id = ?
                """,
                (logical_action_id,),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def reconciliation_invocation_count(
        self,
        logical_action_id: str,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM reconciliation_invocations
                WHERE logical_action_id = ?
                """,
                (logical_action_id,),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

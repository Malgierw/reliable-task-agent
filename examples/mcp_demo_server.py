from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations


server = MCPServer(
    "reliable-task-agent-mcp-demo",
    description="Local MCP server for RTA discovery tests.",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect_ticket_database() -> sqlite3.Connection:
    configured_path = os.environ.get("RTA_MCP_DEMO_DB")
    if not configured_path:
        raise RuntimeError("RTA_MCP_DEMO_DB is required for ticket effects")
    database_path = Path(configured_path).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tool_invocations (
            invocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    return connection


def _ticket_receipt(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "ticket_id": row["ticket_id"],
        "idempotency_key": row["idempotency_key"],
        "title": row["title"],
        "description": row["description"],
        "created_at": row["created_at"],
    }


@server.tool(
    description="Look up a ticket by its identifier.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    meta={"demo": True, "category": "ticket"},
)
def get_ticket(ticket_id: str) -> dict[str, Any]:
    """Return a deterministic demo ticket."""

    if ticket_id == "raise-error":
        raise ValueError("demo ticket lookup failed")

    return {
        "ticket_id": ticket_id,
        "status": "open",
    }


@server.tool(
    description="Create a ticket in the demo service.",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    meta={"demo": True, "category": "ticket"},
)
def create_ticket(
    title: str,
    description: str,
    idempotency_key: str,
) -> dict[str, Any]:
    """Create or read one durable ticket under a stable identity."""

    with _connect_ticket_database() as connection:
        connection.execute(
            """
            INSERT INTO tool_invocations (
                tool_name, idempotency_key, created_at
            ) VALUES ('create_ticket', ?, ?)
            """,
            (idempotency_key, _utc_now_iso()),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO tickets (
                idempotency_key, title, description, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                idempotency_key,
                title,
                description,
                _utc_now_iso(),
            ),
        )
        row = connection.execute(
            "SELECT * FROM tickets WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
    if row is None:
        raise RuntimeError("Committed demo ticket could not be read")
    return _ticket_receipt(row)


@server.tool(
    description="Find a ticket using its stable idempotency identity.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    meta={"demo": True, "category": "ticket-reconciliation"},
)
def get_ticket_by_idempotency_key(
    idempotency_key: str,
) -> dict[str, Any]:
    """Return an RTA reconciliation decision for one durable ticket."""

    with _connect_ticket_database() as connection:
        connection.execute(
            """
            INSERT INTO tool_invocations (
                tool_name, idempotency_key, created_at
            ) VALUES ('get_ticket_by_idempotency_key', ?, ?)
            """,
            (idempotency_key, _utc_now_iso()),
        )
        row = connection.execute(
            "SELECT * FROM tickets WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()

    reconciliation_mode = os.environ.get(
        "RTA_MCP_DEMO_RECONCILIATION",
        "normal",
    )
    if reconciliation_mode == "error":
        raise RuntimeError("demo reconciliation unavailable")
    if reconciliation_mode == "unknown":
        return {
            "status": "UNKNOWN",
            "reason": "demo reconciliation is ambiguous",
        }
    if row is None:
        return {"status": "NOT_FOUND"}
    return {
        "status": "FOUND",
        "receipt": _ticket_receipt(row),
    }


if __name__ == "__main__":
    server.run(transport="stdio")

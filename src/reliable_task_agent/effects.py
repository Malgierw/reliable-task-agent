from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError

from reliable_task_agent.tools.registry import (
    RegisteredTool,
    ToolExecutionResult,
)
from reliable_task_agent.telemetry import NOOP_TELEMETRY, Telemetry


EffectState = Literal[
    "PREPARED",
    "COMMITTED",
    "UNKNOWN",
]

ReconciliationStatus = Literal[
    "FOUND",
    "NOT_FOUND",
    "UNKNOWN",
]


def utc_now() -> datetime:
    """返回带时区信息的 UTC 时间。"""

    return datetime.now(timezone.utc)


def canonical_arguments_json(
    arguments: BaseModel,
) -> str:
    """稳定序列化已经通过 Pydantic 校验的工具参数。"""

    return json.dumps(
        arguments.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def hash_arguments(arguments: BaseModel) -> str:
    """计算规范化工具参数的 SHA-256。"""

    canonical = canonical_arguments_json(arguments)
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def build_effect_identity(
    run_id: str,
    tool_call_id: str,
) -> tuple[str, str]:
    """根据稳定的运行/工具调用身份生成 effect 与幂等键。"""

    material = (
        f"reliable-task-agent:v1:{run_id}:{tool_call_id}"
    )
    effect_id = hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()
    return effect_id, f"rta:{effect_id}"


class ReconciliationResult(BaseModel):
    """外部副作用查询的确定性结果。"""

    model_config = ConfigDict(extra="forbid")

    status: ReconciliationStatus
    receipt: dict[str, Any] | None = None
    reason: str | None = None


class EffectRecord(BaseModel):
    """Runtime Effect Ledger 中的一条持久化记录。"""

    model_config = ConfigDict(extra="forbid")

    effect_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments_hash: str
    idempotency_key: str
    state: EffectState
    result: dict[str, Any] | None = None
    unknown_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    committed_at: datetime | None = None


class EffectSafetyError(RuntimeError):
    """Effect Boundary 无法证明自动执行安全。"""


class EffectIdentityMismatchError(EffectSafetyError):
    """相同 effect 身份对应了不一致的工具或参数。"""


class EffectStateUnknownError(EffectSafetyError):
    """外部副作用状态无法可靠确定。"""


class EffectExecutionAmbiguousError(EffectSafetyError):
    """副作用处理器失败，无法确定外部副作用是否已经发生。"""


class EffectStore:
    """使用 SQLite 保存独立于 Agent Checkpoint 的 Effect Ledger。"""

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
                CREATE TABLE IF NOT EXISTS effects (
                    effect_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_hash TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL CHECK (
                        state IN ('PREPARED', 'COMMITTED', 'UNKNOWN')
                    ),
                    result_json TEXT,
                    unknown_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    committed_at TEXT,
                    UNIQUE (run_id, tool_call_id)
                )
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> EffectRecord:
        try:
            result_json = row["result_json"]
            return EffectRecord(
                effect_id=row["effect_id"],
                run_id=row["run_id"],
                tool_call_id=row["tool_call_id"],
                tool_name=row["tool_name"],
                arguments_hash=row["arguments_hash"],
                idempotency_key=row["idempotency_key"],
                state=row["state"],
                result=(
                    json.loads(result_json)
                    if result_json is not None
                    else None
                ),
                unknown_reason=row["unknown_reason"],
                created_at=datetime.fromisoformat(
                    row["created_at"]
                ),
                updated_at=datetime.fromisoformat(
                    row["updated_at"]
                ),
                committed_at=(
                    datetime.fromisoformat(row["committed_at"])
                    if row["committed_at"] is not None
                    else None
                ),
            )
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise EffectSafetyError(
                "Effect ledger record is corrupt for "
                f"effect_id={row['effect_id']}."
            ) from exc

    @staticmethod
    def assert_identity(
        record: EffectRecord,
        *,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments_hash: str,
        idempotency_key: str,
    ) -> None:
        """验证已有 Effect 记录仍对应同一稳定调用身份。"""

        expected_identity = (
            run_id,
            tool_call_id,
            tool_name,
            arguments_hash,
            idempotency_key,
        )
        actual_identity = (
            record.run_id,
            record.tool_call_id,
            record.tool_name,
            record.arguments_hash,
            record.idempotency_key,
        )

        if actual_identity != expected_identity:
            raise EffectIdentityMismatchError(
                "Effect identity/arguments mismatch for "
                f"effect_id={record.effect_id}."
            )

    def get(self, effect_id: str) -> EffectRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM effects WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()

        if row is None:
            return None
        return self._from_row(row)

    def prepare(
        self,
        *,
        effect_id: str,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments_hash: str,
        idempotency_key: str,
    ) -> tuple[EffectRecord, bool]:
        """持久化 PREPARED；已存在时校验其稳定身份。"""

        now = utc_now().isoformat()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO effects (
                    effect_id,
                    run_id,
                    tool_call_id,
                    tool_name,
                    arguments_hash,
                    idempotency_key,
                    state,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'PREPARED', ?, ?)
                """,
                (
                    effect_id,
                    run_id,
                    tool_call_id,
                    tool_name,
                    arguments_hash,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                "SELECT * FROM effects WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError(
                f"Effect PREPARED 持久化失败：{effect_id}"
            )

        record = self._from_row(row)
        self.assert_identity(
            record,
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments_hash=arguments_hash,
            idempotency_key=idempotency_key,
        )

        if not created and record.state != "PREPARED":
            raise EffectSafetyError(
                "Cannot prepare terminal Effect "
                f"effect_id={effect_id} in state={record.state}."
            )

        return record, created

    def commit(
        self,
        effect_id: str,
        result: ToolExecutionResult,
    ) -> EffectRecord:
        """把 PREPARED 原子转换为 COMMITTED 并保存完整工具结果。"""

        result_data = result.model_dump(mode="json")
        result_json = json.dumps(
            result_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = utc_now().isoformat()

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM effects WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()

            if row is None:
                raise KeyError(f"未找到 Effect：{effect_id}")

            record = self._from_row(row)

            if record.state == "UNKNOWN":
                raise EffectStateUnknownError(
                    f"Effect 已处于 UNKNOWN：{effect_id}"
                )

            if record.state == "COMMITTED":
                if record.result != result_data:
                    raise EffectIdentityMismatchError(
                        "COMMITTED Effect receipt mismatch for "
                        f"effect_id={effect_id}."
                    )
                return record

            connection.execute(
                """
                UPDATE effects
                SET state = 'COMMITTED',
                    result_json = ?,
                    unknown_reason = NULL,
                    updated_at = ?,
                    committed_at = ?
                WHERE effect_id = ? AND state = 'PREPARED'
                """,
                (result_json, now, now, effect_id),
            )
            row = connection.execute(
                "SELECT * FROM effects WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError(
                f"Effect COMMITTED 持久化失败：{effect_id}"
            )
        return self._from_row(row)

    def mark_unknown(
        self,
        effect_id: str,
        reason: str,
    ) -> EffectRecord:
        """把 PREPARED 原子转换为 UNKNOWN。"""

        now = utc_now().isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE effects
                SET state = 'UNKNOWN',
                    unknown_reason = ?,
                    updated_at = ?
                WHERE effect_id = ? AND state = 'PREPARED'
                """,
                (reason, now, effect_id),
            )
            row = connection.execute(
                "SELECT * FROM effects WHERE effect_id = ?",
                (effect_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"未找到 Effect：{effect_id}")

        record = self._from_row(row)
        if record.state != "UNKNOWN":
            raise EffectSafetyError(
                "只有 PREPARED Effect 可以转换为 UNKNOWN："
                f"{effect_id}"
            )
        return record


TransitionHook = Callable[[dict[str, Any]], None]
FaultHook = Callable[[str], None]


class EffectExecutor:
    """执行受保护副作用，并根据 Ledger 状态安全恢复。"""

    def __init__(
        self,
        store: EffectStore,
        *,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.store = store
        self.telemetry = telemetry or NOOP_TELEMETRY

    def execute(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        transition_hook: TransitionHook,
        fault_hook: FaultHook,
    ) -> ToolExecutionResult:
        effect_id, _ = build_effect_identity(run_id, tool_call_id)
        with self.telemetry.span(
            "rta.effect",
            {
                "rta.run.id": run_id,
                "rta.tool.name": tool.name,
                "rta.tool_call.id": tool_call_id,
                "rta.effect.id": effect_id,
            },
            error_category="effect_boundary",
        ) as effect_span:

            def observed_transition(details: dict[str, Any]) -> None:
                attributes = {
                    "rta.effect.id": details.get("effect_id"),
                    "rta.effect.from_state": details.get("from_state"),
                    "rta.effect.to_state": details.get("to_state"),
                    "rta.reconciliation.outcome": details.get(
                        "reconciliation_status"
                    ),
                }
                effect_span.add_event(
                    "rta.effect.transition",
                    attributes,
                )
                effect_span.set_attribute(
                    "rta.effect.state",
                    details.get("to_state"),
                )
                effect_span.set_attribute(
                    "rta.reconciliation.outcome",
                    details.get("reconciliation_status"),
                )
                transition_hook(details)

            return self._execute_unobserved(
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool=tool,
                arguments=arguments,
                transition_hook=observed_transition,
                fault_hook=fault_hook,
            )

    def _execute_unobserved(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        transition_hook: TransitionHook,
        fault_hook: FaultHook,
    ) -> ToolExecutionResult:
        if tool.effect_spec is None:
            raise ValueError(
                f"工具不是 effect-managed：{tool.name}"
            )

        try:
            validated_args = tool.args_model.model_validate(
                arguments
            )
        except ValidationError as exc:
            return ToolExecutionResult(
                ok=False,
                tool_name=tool.name,
                error=f"工具参数校验失败：{exc}",
            )

        arguments_hash = hash_arguments(validated_args)
        effect_id, idempotency_key = build_effect_identity(
            run_id,
            tool_call_id,
        )
        record = self.store.get(effect_id)
        created = False

        if record is None:
            record, created = self.store.prepare(
                effect_id=effect_id,
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name=tool.name,
                arguments_hash=arguments_hash,
                idempotency_key=idempotency_key,
            )
        else:
            self.store.assert_identity(
                record,
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name=tool.name,
                arguments_hash=arguments_hash,
                idempotency_key=idempotency_key,
            )

        if created:
            transition_hook(
                {
                    "effect_id": effect_id,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool.name,
                    "from_state": "ABSENT",
                    "to_state": "PREPARED",
                    "reason": "effect_prepared",
                }
            )
            fault_hook("after_effect_prepared")

        if record.state == "COMMITTED":
            if record.result is None:
                raise EffectSafetyError(
                    "COMMITTED Effect 缺少 ToolExecutionResult："
                    f"{effect_id}"
                )
            transition_hook(
                {
                    "effect_id": effect_id,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool.name,
                    "from_state": "COMMITTED",
                    "to_state": "COMMITTED",
                    "reason": "reused_committed_result",
                }
            )
            try:
                return ToolExecutionResult.model_validate(
                    record.result
                )
            except ValidationError as exc:
                raise EffectSafetyError(
                    "COMMITTED Effect contains an invalid "
                    f"ToolExecutionResult: {effect_id}"
                ) from exc

        if record.state == "UNKNOWN":
            raise EffectStateUnknownError(
                record.unknown_reason
                or f"Effect 状态为 UNKNOWN：{effect_id}"
            )

        if not created:
            with self.telemetry.span(
                "rta.reconciliation",
                {
                    "rta.run.id": run_id,
                    "rta.tool.name": tool.name,
                    "rta.tool_call.id": tool_call_id,
                    "rta.effect.id": effect_id,
                },
                error_category="reconciliation",
            ) as reconciliation_span:
                try:
                    reconciliation = tool.effect_spec.reconcile(
                        validated_args,
                        idempotency_key,
                    )
                    reconciliation = (
                        reconciliation
                        if isinstance(
                            reconciliation,
                            ReconciliationResult,
                        )
                        else ReconciliationResult.model_validate(
                            reconciliation
                        )
                    )
                except Exception as exc:
                    reconciliation_span.record_error(
                        exc,
                        category="reconciliation",
                    )
                    reconciliation = ReconciliationResult(
                        status="UNKNOWN",
                        reason=(
                            "Reconciliation raised "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                reconciliation_span.set_attribute(
                    "rta.reconciliation.outcome",
                    reconciliation.status,
                )

            if reconciliation.status == "FOUND":
                if reconciliation.receipt is None:
                    raise EffectSafetyError(
                        "FOUND reconciliation 缺少 receipt。"
                    )
                result = ToolExecutionResult(
                    ok=True,
                    tool_name=tool.name,
                    data=reconciliation.receipt,
                )
                self.store.commit(effect_id, result)
                transition_hook(
                    {
                        "effect_id": effect_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": tool.name,
                        "from_state": "PREPARED",
                        "to_state": "COMMITTED",
                        "reason": "reconciled_existing_effect",
                        "reconciliation_status": "FOUND",
                    }
                )
                fault_hook("after_effect_committed")
                return result

            if reconciliation.status == "UNKNOWN":
                reason = (
                    reconciliation.reason
                    or "外部副作用状态无法可靠确定。"
                )
                self.store.mark_unknown(effect_id, reason)
                transition_hook(
                    {
                        "effect_id": effect_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": tool.name,
                        "from_state": "PREPARED",
                        "to_state": "UNKNOWN",
                        "reason": reason,
                        "reconciliation_status": "UNKNOWN",
                    }
                )
                raise EffectStateUnknownError(reason)

            transition_hook(
                {
                    "effect_id": effect_id,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool.name,
                    "from_state": "PREPARED",
                    "to_state": "PREPARED",
                    "reason": "reconciled_not_found",
                    "reconciliation_status": "NOT_FOUND",
                }
            )

        try:
            receipt = tool.effect_spec.execute(
                validated_args,
                idempotency_key,
            )
        except Exception as exc:
            raise EffectExecutionAmbiguousError(
                "Effect handler raised after PREPARED; external "
                "effect status is ambiguous for "
                f"effect_id={effect_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        fault_hook("after_effect_execute")

        result = ToolExecutionResult(
            ok=True,
            tool_name=tool.name,
            data=receipt,
        )
        self.store.commit(effect_id, result)
        transition_hook(
            {
                "effect_id": effect_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool.name,
                "from_state": "PREPARED",
                "to_state": "COMMITTED",
                "reason": "external_effect_committed",
            }
        )
        fault_hook("after_effect_committed")
        return result

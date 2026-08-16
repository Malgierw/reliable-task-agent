from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from reliable_task_agent.durable_errors import (
    durable_error_message,
    durable_validation_error_message,
)

# 规定工具执行函数的形式：
# 接收一个经过 Pydantic 校验的参数对象，返回任意类型结果。
ToolHandler = Callable[[BaseModel], Any]
EffectHandler = Callable[[BaseModel, str], Any]
EffectReconciler = Callable[[BaseModel, str], Any]


@dataclass(frozen=True)
class EffectSpec:
    """Effect Boundary 执行与对账一个副作用所需的回调。"""

    execute: EffectHandler
    reconcile: EffectReconciler


@dataclass(frozen=True)
class RegisteredTool:
    """保存一个工具的完整定义。"""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler | None
    effect_spec: EffectSpec | None = None

    def to_openai_schema(self) -> dict[str, Any]:
        """将工具转换为大模型 Function Calling 所需的格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class ToolExecutionResult(BaseModel):
    """统一表示一次工具执行的结果。"""

    ok: bool
    tool_name: str
    data: Any | None = None
    error: str | None = None


class SafeToolFeedbackError(RuntimeError):
    """A built-in failure with a static, persistence-safe diagnostic code."""

    def __init__(self, error_category: str, message: str) -> None:
        super().__init__(message)
        self.error_category = error_category


class ToolRegistry:
    """负责工具的注册、查询、描述和执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: ToolHandler,
    ) -> None:
        """向注册中心添加一个工具。"""
        if name in self._tools:
            raise ValueError(f"工具已经存在：{name}")

        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            args_model=args_model,
            handler=handler,
        )

    def register_effect(
        self,
        *,
        name: str,
        description: str,
        args_model: type[BaseModel],
        execute: EffectHandler,
        reconcile: EffectReconciler,
    ) -> None:
        """注册只能通过 Runtime Effect Boundary 执行的工具。"""

        if name in self._tools:
            raise ValueError(f"工具已经存在：{name}")

        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            args_model=args_model,
            handler=None,
            effect_spec=EffectSpec(
                execute=execute,
                reconcile=reconcile,
            ),
        )

    def get(self, name: str) -> RegisteredTool:
        """根据名称获取工具。"""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"未找到工具：{name}") from exc

    def list_names(self) -> list[str]:
        """返回当前已经注册的全部工具名称。"""
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """生成可以直接传给大模型的工具列表。"""
        return [
            tool.to_openai_schema()
            for tool in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        """校验参数并执行指定工具。"""
        try:
            tool = self.get(name)
        except KeyError as exc:
            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error=durable_error_message(
                    exc,
                    category="tool_lookup",
                ),
            )

        if tool.effect_spec is not None:
            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error=(
                    "Effect-managed 工具不能通过 "
                    "ToolRegistry.execute() 直接执行；"
                    "必须使用 Runtime Effect Boundary。"
                ),
            )

        if tool.handler is None:
            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error="普通工具缺少 handler。",
            )

        try:
            validated_args = tool.args_model.model_validate(arguments)
            result = tool.handler(validated_args)

        except ValidationError as exc:
            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error=durable_validation_error_message(
                    exc,
                    category="tool_argument_validation",
                ),
            )

        except SafeToolFeedbackError as exc:
            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error=durable_error_message(
                    exc,
                    category=exc.error_category,
                ),
            )

        except Exception as exc:
            return ToolExecutionResult(
                ok=False,
                tool_name=name,
                error=durable_error_message(
                    exc,
                    category="tool_execution",
                ),
            )

        return ToolExecutionResult(
            ok=True,
            tool_name=name,
            data=result,
        )

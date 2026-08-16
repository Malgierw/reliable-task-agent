from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry.trace import Tracer


_SPAN_NAMES = frozenset(
    {
        "rta.agent.run",
        "rta.llm.call",
        "rta.tool.execute",
        "rta.mcp.call",
        "rta.effect",
        "rta.reconciliation",
        "rta.verifier",
        "rta.repair",
    }
)

_ATTRIBUTE_KEYS = frozenset(
    {
        "rta.run.id",
        "rta.agent.resumed",
        "rta.agent.outcome",
        "rta.model.name",
        "rta.step",
        "rta.attempt",
        "rta.tool.name",
        "rta.tool_call.id",
        "rta.tool.ok",
        "rta.mcp.transport",
        "rta.effect.id",
        "rta.effect.from_state",
        "rta.effect.to_state",
        "rta.effect.state",
        "rta.reconciliation.outcome",
        "rta.verifier.passed",
        "rta.repair.count",
        "rta.repair.max_attempts",
        "rta.error.type",
        "rta.error.category",
    }
)

_EVENT_NAMES = frozenset({"rta.effect.transition"})
_MAX_STRING_LENGTH = 256


def _safe_value(value: Any) -> bool | int | float | str | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_STRING_LENGTH]
    return None


class TelemetrySpan:
    """Fail-open access to one span through a strict attribute allowlist."""

    def __init__(self, span: Any | None) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is None or key not in _ATTRIBUTE_KEYS:
            return
        safe_value = _safe_value(value)
        if safe_value is None:
            return
        try:
            self._span.set_attribute(key, safe_value)
        except Exception:
            pass

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if self._span is None or name not in _EVENT_NAMES:
            return
        safe_attributes: dict[str, bool | int | float | str] = {}
        for key, value in (attributes or {}).items():
            if key not in _ATTRIBUTE_KEYS:
                continue
            safe_value = _safe_value(value)
            if safe_value is not None:
                safe_attributes[key] = safe_value
        try:
            self._span.add_event(name, attributes=safe_attributes)
        except Exception:
            pass

    def record_error(
        self,
        exc: BaseException,
        *,
        category: str,
    ) -> None:
        self.set_attribute("rta.error.type", type(exc).__name__)
        self.set_attribute("rta.error.category", category)


class Telemetry:
    """Optional manual tracing facade that can never affect RTA execution."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self._tracer = tracer

    @classmethod
    def from_tracer_provider(
        cls,
        tracer_provider: Any,
        *,
        instrumentation_name: str = "reliable_task_agent",
    ) -> Telemetry:
        try:
            tracer = tracer_provider.get_tracer(instrumentation_name)
        except Exception:
            tracer = None
        return cls(tracer)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        error_category: str = "runtime",
    ) -> Iterator[TelemetrySpan]:
        if self._tracer is None or name not in _SPAN_NAMES:
            yield TelemetrySpan(None)
            return

        manager: Any | None = None
        raw_span: Any | None = None
        entered = False
        try:
            manager = self._tracer.start_as_current_span(
                name,
                record_exception=False,
                set_status_on_exception=False,
            )
            raw_span = manager.__enter__()
            entered = True
        except Exception:
            raw_span = None

        span = TelemetrySpan(raw_span)
        span.set_attributes(attributes or {})
        try:
            yield span
        except BaseException as exc:
            span.record_error(exc, category=error_category)
            raise
        finally:
            if entered and manager is not None:
                try:
                    manager.__exit__(None, None, None)
                except Exception:
                    pass


NOOP_TELEMETRY = Telemetry()

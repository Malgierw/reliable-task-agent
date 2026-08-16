from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_SAFE_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


def _identifier(value: Any, fallback: str) -> str:
    if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value):
        return value
    return fallback


def _status_code(exc: BaseException) -> int | None:
    try:
        value = getattr(exc, "status_code", None)
    except Exception:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if 100 <= value <= 599:
            return value
    return None


def durable_error_details(
    exc: BaseException | None,
    *,
    category: str,
    error_type: str | None = None,
) -> dict[str, str | int]:
    """Return the only exception metadata allowed in durable RTA state."""

    details: dict[str, str | int] = {
        "error_type": _identifier(
            error_type or (type(exc).__name__ if exc is not None else "Error"),
            "Error",
        ),
        "error_category": _identifier(category, "runtime_error"),
    }
    if exc is not None:
        status_code = _status_code(exc)
        if status_code is not None:
            details["status_code"] = status_code
    return details


def durable_error_message(
    exc: BaseException | None,
    *,
    category: str,
    error_type: str | None = None,
) -> str:
    return json.dumps(
        durable_error_details(
            exc,
            category=category,
            error_type=error_type,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def durable_validation_error_message(
    exc: ValidationError,
    *,
    category: str,
) -> str:
    validation_errors: list[dict[str, str]] = []
    for item in exc.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        path: list[str] = []
        for segment in item.get("loc", ()):
            if isinstance(segment, int):
                path.append(str(segment))
            elif isinstance(segment, str) and _SAFE_FIELD.fullmatch(segment):
                path.append(segment)
            else:
                path.append("<field>")
        validation_errors.append(
            {
                "field": ".".join(path) or "<root>",
                "code": _identifier(item.get("type"), "validation_error"),
            }
        )

    payload: dict[str, Any] = durable_error_details(
        exc,
        category=category,
    )
    payload["validation_errors"] = validation_errors
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def normalize_durable_error_message(
    value: str,
    *,
    fallback_category: str,
) -> str:
    """Accept only canonical safe summaries at a string persistence field."""

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if not isinstance(parsed, dict):
        return durable_error_message(None, category=fallback_category)

    safe: dict[str, Any] = {
        "error_type": _identifier(parsed.get("error_type"), "Error"),
        "error_category": _identifier(
            parsed.get("error_category"),
            fallback_category,
        ),
    }
    status_code = parsed.get("status_code")
    if (
        isinstance(status_code, int)
        and not isinstance(status_code, bool)
        and 100 <= status_code <= 599
    ):
        safe["status_code"] = status_code

    validation_errors = parsed.get("validation_errors")
    if isinstance(validation_errors, list):
        safe_items: list[dict[str, str]] = []
        for item in validation_errors:
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            code = item.get("code")
            if not isinstance(field, str) or len(field) > 256:
                field = "<field>"
            if not all(
                part.isdigit() or _SAFE_FIELD.fullmatch(part)
                for part in field.split(".")
            ):
                field = "<field>"
            safe_items.append(
                {
                    "field": field,
                    "code": _identifier(code, "validation_error"),
                }
            )
        safe["validation_errors"] = safe_items

    return json.dumps(safe, sort_keys=True, separators=(",", ":"))

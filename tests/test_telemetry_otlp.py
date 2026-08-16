from __future__ import annotations

import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Iterator

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.telemetry import Telemetry
from reliable_task_agent.tools.registry import ToolRegistry


class FinalMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"role": "assistant", "content": self.content}


class FinalClient:
    def __init__(self, content: str) -> None:
        def create(**_: Any) -> Any:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=FinalMessage(content))]
            )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )


class CaptureServer(ThreadingHTTPServer):
    response_status: int
    requests: list[dict[str, Any]]


class CaptureHandler(BaseHTTPRequestHandler):
    server: CaptureServer

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        self.server.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": self.rfile.read(content_length),
            }
        )
        self.send_response(self.server.response_status)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return None


@contextmanager
def local_otlp_receiver(
    response_status: int = 200,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    server = CaptureServer(("127.0.0.1", 0), CaptureHandler)
    server.response_status = response_status
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1/traces", server.requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_agent(telemetry: Telemetry, *, secret: str) -> AgentLoop:
    agent = AgentLoop(
        ToolRegistry(),
        client=FinalClient(f"done {secret}"),
        model="test-model",
        telemetry=telemetry,
    )
    assert agent.run(f"prompt {secret}") == f"done {secret}"
    return agent


def decode_requests(
    requests: list[dict[str, Any]],
) -> list[ExportTraceServiceRequest]:
    decoded = []
    for request in requests:
        message = ExportTraceServiceRequest()
        message.ParseFromString(request["body"])
        decoded.append(message)
    return decoded


def test_otlp_export_is_opt_in_and_empty_endpoint_is_noop() -> None:
    telemetry = Telemetry.from_otlp_http("")

    agent = run_agent(telemetry, secret="not-exported")

    assert telemetry.force_flush()
    assert agent.last_checkpoint is not None
    assert agent.last_checkpoint.status == "completed"


def test_otlp_http_exports_rta_spans_and_safe_service_resource(
    monkeypatch,
) -> None:
    secret = "sk-sensitive-prompt-response-and-header"
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS",
        f"authorization={secret}",
    )
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        f"x-secret={secret}",
    )

    with local_otlp_receiver() as (endpoint, requests):
        telemetry = Telemetry.from_otlp_http(endpoint)
        run_agent(telemetry, secret=secret)
        assert telemetry.force_flush(timeout_millis=5_000)
        telemetry.shutdown()

    assert requests
    assert {request["path"] for request in requests} == {"/v1/traces"}
    decoded = decode_requests(requests)
    span_names: set[str] = set()
    service_names: set[str] = set()
    for export_request in decoded:
        for resource_spans in export_request.resource_spans:
            for attribute in resource_spans.resource.attributes:
                if attribute.key == "service.name":
                    service_names.add(attribute.value.string_value)
            for scope_spans in resource_spans.scope_spans:
                span_names.update(span.name for span in scope_spans.spans)

    assert service_names == {"reliable-task-agent"}
    assert "rta.agent.run" in span_names
    assert "rta.llm.call" in span_names
    wire_output = b"".join(request["body"] for request in requests)
    header_output = repr(
        [request["headers"] for request in requests]
    ).encode()
    assert secret.encode() not in wire_output
    assert secret.encode() not in header_output


def test_unavailable_otlp_http_does_not_break_agent_execution() -> None:
    with local_otlp_receiver(response_status=503) as (endpoint, requests):
        telemetry = Telemetry.from_otlp_http(
            endpoint,
            timeout_seconds=0.1,
        )
        agent = run_agent(telemetry, secret="delivery-failure-secret")
        telemetry.force_flush(timeout_millis=2_000)
        telemetry.shutdown()

    assert requests
    assert agent.last_checkpoint is not None
    assert agent.last_checkpoint.status == "completed"

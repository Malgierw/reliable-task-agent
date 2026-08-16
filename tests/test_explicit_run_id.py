from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from reliable_task_agent.agent_loop import AgentLoop
from reliable_task_agent.checkpoint_store import CheckpointStore
from reliable_task_agent.tools.builtin import build_default_registry
from reliable_task_agent.trace_store import TraceStore


class FinalMessage:
    content = "SUCCESS"
    tool_calls = None

    def model_dump(self, **_: Any) -> dict[str, str]:
        return {"role": "assistant", "content": self.content}


class FinalCompletions:
    def create(self, **_: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=FinalMessage())]
        )


def build_client() -> Any:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=FinalCompletions())
    )


def build_agent(root: Path) -> AgentLoop:
    return AgentLoop(
        build_default_registry(),
        client=build_client(),
        model="fake-model",
        checkpoint_store=CheckpointStore(root),
        trace_store=TraceStore(root),
    )


def test_run_default_still_generates_uuid_identity(tmp_path) -> None:
    agent = build_agent(tmp_path / "default")

    assert agent.run("Finish.") == "SUCCESS"
    assert agent.last_trace is not None
    assert re.fullmatch(r"[0-9a-f]{32}", agent.last_trace.run_id)


def test_explicit_run_id_persists_and_resumes(tmp_path) -> None:
    run_id = "a" * 32
    root = tmp_path / "explicit"
    agent = build_agent(root)

    assert agent.run("Finish.", run_id=run_id) == "SUCCESS"
    assert agent.last_trace is not None
    assert agent.last_trace.run_id == run_id
    assert TraceStore(root).load(run_id).run_id == run_id
    assert CheckpointStore(root).load(run_id).run_id == run_id

    resumed = build_agent(root)
    assert resumed.resume(run_id) == "SUCCESS"
    assert resumed.last_trace is not None
    assert resumed.last_trace.run_id == run_id

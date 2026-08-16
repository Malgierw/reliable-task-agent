from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FaultMarker:
    """Durably announces a kill point, then waits for parent termination."""

    def __init__(
        self,
        marker_path: str | Path,
        target_stage: str | None,
    ) -> None:
        self.marker_path = Path(marker_path)
        self.target_stage = target_stage

    def reach(self, stage: str) -> None:
        if stage != self.target_stage:
            return

        data = {
            "stage": stage,
            "pid": os.getpid(),
            "reached_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }
        temporary_path = self.marker_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.marker_path)

        while True:
            time.sleep(1)


def wait_for_marker(
    marker_path: str | Path,
    process: Any,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    path = Path(marker_path)
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        if process.poll() is not None:
            raise RuntimeError(
                "Worker exited before reaching the fault marker: "
                f"returncode={process.returncode}"
            )
        time.sleep(0.02)

    raise TimeoutError(
        f"Timed out waiting for fault marker: {path}"
    )

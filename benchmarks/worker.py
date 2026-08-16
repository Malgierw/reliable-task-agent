from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("initial", "recovery"),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    trial_directory = manifest_path.parent
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    result_path = trial_directory / f"worker-{args.phase}.json"

    try:
        if manifest["configuration"].startswith("langgraph_"):
            from benchmarks.adapters.langgraph_adapter import (
                run_langgraph_worker,
            )

            result = run_langgraph_worker(
                manifest,
                trial_directory,
                args.phase,
            )
        else:
            from benchmarks.adapters.rta_adapter import (
                run_rta_worker,
            )

            result = run_rta_worker(
                manifest,
                trial_directory,
                args.phase,
            )
        write_json(result_path, result)
        return 0
    except Exception as exc:
        write_json(
            result_path,
            {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "worker_phase": args.phase,
            },
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())

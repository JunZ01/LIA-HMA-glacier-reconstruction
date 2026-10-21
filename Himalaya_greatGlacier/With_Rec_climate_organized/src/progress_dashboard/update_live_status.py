"""Atomically update the persistent workspace monitor state."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--state", choices=["running", "complete", "error"])
    parser.add_argument("--phase")
    parser.add_argument("--activity")
    parser.add_argument("--last-action")
    parser.add_argument("--error")
    parser.add_argument("--clear-error", action="store_true")
    parser.add_argument("--event")
    parser.add_argument("--step", type=int)
    parser.add_argument(
        "--step-status", choices=["pending", "running", "complete", "error"]
    )
    args = parser.parse_args()

    path = args.path.resolve()
    value = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    for key, incoming in (
        ("state", args.state),
        ("phase", args.phase),
        ("current_activity", args.activity),
        ("last_action", args.last_action),
        ("last_error", args.error),
    ):
        if incoming is not None:
            value[key] = incoming
    if args.clear_error:
        value["last_error"] = None
    if args.step is not None:
        plan = value.get("plan", [])
        if args.step < 1 or args.step > len(plan):
            raise IndexError(f"step must be between 1 and {len(plan)}")
        if args.step_status is None:
            raise ValueError("--step-status is required with --step")
        plan[args.step - 1]["status"] = args.step_status
    if args.event:
        events = value.setdefault("events", [])
        timestamp = datetime.now(timezone.utc).isoformat()
        events.append(f"{timestamp} {args.event}")
        value["events"] = events[-30:]
    value["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    value["completed_step_count"] = sum(
        item.get("status") == "complete" for item in value.get("plan", [])
    )
    value["total_step_count"] = len(value.get("plan", []))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()

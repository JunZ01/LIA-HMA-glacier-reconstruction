"""Run the gated Stage 3 operator smoke and, if it passes, the 12-system pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import read_json, resolve_path, write_json_atomic


def update_live_status(path: Path, state: str, phase: str, activity: str, action: str, error: str | None = None) -> None:
    value = read_json(path) if path.exists() else {}
    value.update(
        {
            "state": state,
            "phase": phase,
            "current_activity": activity,
            "last_action": action,
            "last_error": error,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    events = value.setdefault("events", [])
    events.append(f"{value['updated_at_utc']} {action}")
    value["events"] = events[-30:]
    value["completed_step_count"] = sum(
        item.get("status") == "complete" for item in value.get("plan", [])
    )
    value["total_step_count"] = len(value.get("plan", []))
    write_json_atomic(path, value)


def run_wave(runner: Path, config: Path, wave: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(runner), "--config", str(config), "--wave", wave]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(runner.parents[2]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = resolve_path(config["output_dir"])
    monitor_path = output_dir.parent / "live_monitor" / "live_status.json"
    runner = Path(__file__).resolve().with_name("run_initial_state_pilot.py")
    supervisor_log = output_dir / "overnight_gate_v10.log"

    update_live_status(
        monitor_path,
        "running",
        "Stage 3 operator-smoke v10",
        "Running the four-system operator smoke with candidate-level process isolation.",
        "Overnight gate started: operator-smoke v10 is running; 12-system pilot is conditional on its semantic gate.",
    )
    operator_log = output_dir / "operator-smoke-v10" / "supervisor.stdout.log"
    operator_rc = run_wave(runner, config_path, "operator-smoke", operator_log)
    operator_dir = output_dir / config["wave_output_dirs"]["operator-smoke"]
    operator_summary_path = operator_dir / "run_summary.json"
    operator_summary = read_json(operator_summary_path) if operator_summary_path.exists() else {}
    operator_pass = operator_rc == 0 and bool(operator_summary.get("operator_semantic_gate_pass"))
    if not operator_pass:
        message = f"operator-smoke gate failed (returncode={operator_rc}, summary={operator_summary_path})"
        update_live_status(
            monitor_path,
            "error",
            "Stage 3 operator-smoke v10 stopped",
            "Operator smoke did not pass; the 12-system pilot was not started.",
            message,
            message,
        )
        return 1

    update_live_status(
        monitor_path,
        "running",
        "Stage 3 turnover-pilot v10",
        "Operator gate passed; running 12 systems across the three frozen turnover roles.",
        "operator-smoke v10 passed its semantic gate; turnover-pilot v10 started automatically.",
    )
    turnover_log = output_dir / "turnover-pilot-v10" / "supervisor.stdout.log"
    turnover_rc = run_wave(runner, config_path, "turnover-pilot", turnover_log)
    turnover_dir = output_dir / config["wave_output_dirs"]["turnover-pilot"]
    turnover_summary_path = turnover_dir / "run_summary.json"
    turnover_summary = read_json(turnover_summary_path) if turnover_summary_path.exists() else {}
    turnover_pass = turnover_rc == 0 and bool(turnover_summary.get("operator_semantic_gate_pass"))
    if turnover_pass:
        update_live_status(
            monitor_path,
            "complete",
            "Stage 3 turnover-pilot v10 complete",
            "The 12-system pilot completed and passed its semantic gate; no 191-system run was started.",
            "Overnight gate complete: operator-smoke and turnover-pilot passed.",
        )
        return 0
    message = f"turnover-pilot gate failed (returncode={turnover_rc}, summary={turnover_summary_path})"
    update_live_status(
        monitor_path,
        "error",
        "Stage 3 turnover-pilot v10 stopped",
        "The 12-system pilot ended without a passing semantic gate; no 191-system run was started.",
        message,
        message,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

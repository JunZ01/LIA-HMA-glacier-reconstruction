"""Run the 12-system turnover pilot with one serial shard per system."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import read_json, resolve_path, write_json_atomic


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def update_live_status(
    path: Path,
    state: str,
    phase: str,
    activity: str,
    action: str,
    error: str | None = None,
) -> None:
    value = read_json(path) if path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    value.update(
        {
            "state": state,
            "phase": phase,
            "current_activity": activity,
            "last_action": action,
            "last_error": error,
            "updated_at_utc": now,
        }
    )
    events = value.setdefault("events", [])
    if not events or not events[-1].endswith(action):
        events.append(f"{now} {action}")
    value["events"] = events[-30:]
    write_json_atomic(path, value)


def read_progress(path: Path) -> dict:
    try:
        return read_json(path)
    except (FileNotFoundError, ValueError):
        return {}


def merge_csv(shard_dirs: list[Path], name: str, target: Path) -> pd.DataFrame:
    frames = []
    for directory in shard_dirs:
        path = directory / name
        if path.exists() and path.stat().st_size:
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                pass
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "task_key" in frame:
        frame = frame.sort_values(
            [column for column in ("lia_unique_id", "task_key", "initial_state_offset_k") if column in frame]
        ).reset_index(drop=True)
    write_csv_atomic(target, frame)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-name", default="turnover-pilot-v12-parallel")
    parser.add_argument("--model-tag", default="s3p0v12p")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    started = time.time()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = resolve_path(config["output_dir"])
    canonical_dir = output_dir / args.output_name
    canonical_dir.mkdir(parents=True, exist_ok=True)
    shard_root = resolve_path(config["oggm_workdir_path"]) / "parallel_shards_v12"
    shard_root.mkdir(parents=True, exist_ok=True)
    monitor_path = output_dir.parent / "live_monitor" / "live_status.json"
    runner = Path(__file__).resolve().with_name("run_initial_state_pilot.py")

    operator_summary_path = output_dir / "operator-smoke-v10" / "run_summary.json"
    operator_summary = read_json(operator_summary_path)
    if not operator_summary.get("operator_semantic_gate_pass"):
        raise RuntimeError("operator-smoke-v10 semantic gate has not passed")

    selection = pd.read_csv(output_dir / "pilot_selection_12.csv")
    system_ids = sorted(
        int(value) for value in selection["lia_unique_id"].astype(int).unique()
    )
    if len(system_ids) != 12:
        raise RuntimeError(f"Expected 12 pilot systems, found {len(system_ids)}")
    members = pd.read_csv(output_dir / "pilot_stage2_member_ensemble.csv")
    ownership = members[["lia_unique_id", "rgi_id"]].drop_duplicates()
    overlap = ownership.groupby("rgi_id")["lia_unique_id"].nunique()
    if (overlap > 1).any():
        raise RuntimeError("Parallel systems share RGI members; safe sharding is impossible")

    shard_dirs = {system_id: shard_root / f"s{system_id}" for system_id in system_ids}
    completed_systems = set()
    for system_id, directory in shard_dirs.items():
        summary_path = directory / "run_summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            if summary.get("task_count") == 3 and summary.get("operator_semantic_gate_pass"):
                completed_systems.add(system_id)
    queue = [system_id for system_id in system_ids if system_id not in completed_systems]
    active: dict[int, tuple[subprocess.Popen, object, object]] = {}
    returncodes: dict[int, int] = {system_id: 0 for system_id in completed_systems}

    update_live_status(
        monitor_path,
        "running",
        "Stage 3 turnover-pilot v12 parallel",
        f"Running 12 systems with up to {args.workers} independent system workers.",
        "Parallel turnover pilot started after operator-smoke-v10 passed.",
    )

    last_live_update = 0.0
    while queue or active:
        while queue and len(active) < args.workers:
            system_id = queue.pop(0)
            shard_dir = shard_dirs[system_id]
            shard_dir.mkdir(parents=True, exist_ok=True)
            stdout = (shard_dir / "runner.stdout.log").open("a", encoding="utf-8")
            stderr = (shard_dir / "runner.stderr.log").open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(runner),
                "--config",
                str(config_path),
                "--wave",
                "turnover-pilot",
                "--system-id",
                str(system_id),
                "--wave-dir-override",
                str(shard_dir),
                "--model-filesuffix-tag-override",
                args.model_tag,
            ]
            process = subprocess.Popen(
                command,
                cwd=str(runner.parents[2]),
                stdout=stdout,
                stderr=stderr,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            active[system_id] = (process, stdout, stderr)

        time.sleep(3)
        for system_id, (process, stdout, stderr) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            stdout.close()
            stderr.close()
            returncodes[system_id] = int(returncode)
            del active[system_id]

        progresses = [
            read_progress(directory / "progress.json")
            for directory in shard_dirs.values()
        ]
        completed_tasks = sum(int(value.get("completed_task_count", 0)) for value in progresses)
        candidate_count = sum(int(value.get("completed_candidate_count", 0)) for value in progresses)
        active_tasks = [
            value.get("current_task_key")
            for value in progresses
            if value.get("current_task_key")
        ]
        write_json_atomic(
            canonical_dir / "progress.json",
            {
                "wave": "turnover-pilot-parallel",
                "state": "running",
                "worker_count": int(args.workers),
                "completed_task_count": int(completed_tasks),
                "total_task_count": 36,
                "completed_candidate_count": int(candidate_count),
                "active_system_ids": sorted(active),
                "current_task_key": "; ".join(active_tasks),
                "current_offset_k": None,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if time.time() - last_live_update >= 30:
            update_live_status(
                monitor_path,
                "running",
                "Stage 3 turnover-pilot v12 parallel",
                f"{completed_tasks}/36 tasks complete; active systems: {sorted(active)}.",
                "Parallel turnover pilot is running with checkpointed system shards.",
            )
            last_live_update = time.time()

    ordered_dirs = [shard_dirs[system_id] for system_id in system_ids]
    candidate_frame = merge_csv(ordered_dirs, "candidate_trace.csv", canonical_dir / "candidate_trace.csv")
    task_frame = merge_csv(ordered_dirs, "task_summary.csv", canonical_dir / "task_summary.csv")
    failure_frame = merge_csv(ordered_dirs, "dynamic_failures.csv", canonical_dir / "dynamic_failures.csv")
    state_frame = merge_csv(ordered_dirs, "selected_states.csv", canonical_dir / "selected_states.csv")
    shard_summaries = [read_progress(directory / "run_summary.json") for directory in ordered_dirs]
    gate_pass = bool(
        len(task_frame) == 36
        and task_frame["task_key"].nunique() == 36
        and not any(returncodes.get(system_id, 1) for system_id in system_ids)
        and all(summary.get("operator_semantic_gate_pass") for summary in shard_summaries)
        and len(failure_frame) == 0
        and all(summary.get("source_stage0_key_files_immutable") for summary in shard_summaries)
    )
    summary = {
        "run_id": config["run_id"],
        "wave": "turnover-pilot-parallel",
        "status": "completed-stage3-parallel-turnover-pilot",
        "worker_count": int(args.workers),
        "system_count": len(system_ids),
        "task_count": int(len(task_frame)),
        "selected_state_task_count": int(task_frame["status"].eq("selected-stage3-state").sum()),
        "classified_failure_task_count": int((~task_frame["status"].eq("selected-stage3-state")).sum()),
        "candidate_count": int(len(candidate_frame)),
        "dynamic_failure_count": int(len(failure_frame)),
        "selected_member_state_count": int(len(state_frame)),
        "source_stage0_key_files_immutable": bool(
            all(summary.get("source_stage0_key_files_immutable") for summary in shard_summaries)
        ),
        "operator_semantic_gate_pass": gate_pass,
        "scale_up_gate_pass": gate_pass,
        "shard_root": str(shard_root),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_atomic(canonical_dir / "run_summary.json", summary)
    progress = read_progress(canonical_dir / "progress.json")
    progress.update(
        {
            "state": "complete" if gate_pass else "error",
            "completed_task_count": int(len(task_frame)),
            "completed_candidate_count": int(len(candidate_frame)),
            "active_system_ids": [],
            "current_task_key": None,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json_atomic(canonical_dir / "progress.json", progress)
    if gate_pass:
        update_live_status(
            monitor_path,
            "complete",
            "Stage 3 turnover-pilot v12 parallel complete",
            f"The 12-system, 36-task pilot completed with {args.workers} system workers.",
            "Parallel turnover pilot passed its semantic gate; no 191-system run was started.",
        )
    else:
        update_live_status(
            monitor_path,
            "error",
            "Stage 3 turnover-pilot v12 parallel stopped",
            "The parallel pilot completed without a passing semantic gate.",
            "Parallel turnover pilot requires review; no 191-system run was started.",
            "One or more system shards failed the execution or provenance gate.",
        )
    print(summary)
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

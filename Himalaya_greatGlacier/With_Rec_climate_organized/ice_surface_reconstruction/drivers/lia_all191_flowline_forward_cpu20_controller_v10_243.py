"""CPU20 controller for 30-year-state forward members, preserving legacy results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from . import lia_all191_flowline_forward_cpu20_handoff_controller_v10_240 as common


WORKERS = 20
MEMORY_USED_CEILING = 0.80
NEW_WORKER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.lia_all191_flowline_forward_v10_243_30y"
)
OLD_WORKER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.lia_all191_flowline_forward_v10_237"
)


def read_matrix(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--handoff-snapshot", type=Path, required=True)
    args = parser.parse_args()
    manifest = common.read_json(args.transaction_dir / "transaction_manifest.json")
    if not manifest.get("launch_authorized"):
        raise ValueError("transaction-not-launch-authorized")
    if common.sha256(Path(__file__)) != manifest["runtime"]["controller_sha256"]:
        raise ValueError("controller-hash-mismatch")
    if common.sha256(Path(manifest["runtime"]["worker_path"])) != manifest["runtime"]["worker_sha256"]:
        raise ValueError("worker-hash-mismatch")
    matrix = read_matrix(args.transaction_dir / "forward_matrix.csv")
    old_matrix = read_matrix(args.transaction_dir / "source_forward_matrix.csv")
    if len(matrix) != 530 or len(old_matrix) != 530:
        raise ValueError("matrix-cardinality-failure")
    new_by_key = {row["production_member_key"]: row for row in matrix}
    old_by_key = {row["production_member_key"]: row for row in old_matrix}
    if set(new_by_key) != set(old_by_key) or len(new_by_key) != 530:
        raise ValueError("member-key-set-failure")
    root = Path(manifest["runtime"]["unified_short_root"]).resolve()
    if args.output_root.resolve() != root:
        raise ValueError("output-root-contract-mismatch")
    preflight_result = Path(manifest["runtime"]["preflight_root"]) / "p/p0000/c/result.json"
    preflight = common.read_json(preflight_result)
    if not (preflight.get("status") == "preflight-pass" and preflight.get("resume_check", {}).get("passed")):
        raise ValueError("30y-preflight-not-passed")
    handoff = common.read_json(args.handoff_snapshot)
    source_parent = int(handoff["source_parent_pid"])
    if psutil.pid_exists(source_parent):
        raise RuntimeError("source-controller-still-alive")

    def result_path(key: str) -> Path:
        return root / "m" / key / "c" / "result.json"

    def terminal_result(key: str) -> dict | None:
        path = result_path(key)
        if not path.is_file():
            return None
        try:
            result = common.read_json(path)
        except Exception:
            return None
        if not result.get("terminal"):
            return None
        allowed = {old_by_key[key]["forward_fingerprint"], new_by_key[key]["forward_fingerprint"]}
        if result.get("forward_fingerprint") not in allowed:
            raise ValueError(f"result-fingerprint-mismatch:{key}")
        return result

    completed: dict[str, dict] = {}
    for key in new_by_key:
        result = terminal_result(key)
        if result is not None:
            completed[key] = result
    external = {}
    for item in handoff["inherited_workers"]:
        key = item["member_key"]
        if key in completed:
            continue
        pid = int(item["pid"])
        if psutil.pid_exists(pid):
            external[key] = {"pid": pid}

    tasks = [{"kind": "new30", "key": key} for key in new_by_key if key not in completed and key not in external]
    next_index = 0
    owned = {}
    runtime_root = root / "r"
    log_root = root / "l/production_v10_243"
    runtime_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=False)
    events_path = runtime_root / "production_events_v10_243.jsonl"
    parent = {
        "schema": "lia-all191-forward-parent-v10.243", "pid": os.getpid(),
        "started_at_utc": common.now(), "workers": WORKERS,
        "inherited_workers": len(external), "inherited_completed": len(completed),
        "transaction": str(args.transaction_dir.resolve()), "output_root": str(root),
        "handoff_snapshot": str(args.handoff_snapshot.resolve()), "command": sys.argv,
    }
    common.atomic_json(runtime_root / "production_parent_v10_243.json", parent)
    env = os.environ.copy()
    env.update({k: str(v) for k, v in manifest["runtime"]["thread_environment"].items()})
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4])
    python = manifest["runtime"]["python_executable"]
    pause_reason = None
    peak_memory = 0.0

    def write_status() -> None:
        technical = sum(bool(item.get("technical_failure")) for item in completed.values())
        scientific = sum(bool(item.get("scientific_failure")) for item in completed.values())
        running = ([{"kind": "legacy-inflight", "pid": item["pid"], "member_key": key}
                    for key, item in external.items()] +
                   [{"kind": item["kind"], "slot": slot, "pid": item["process"].pid,
                     "member_key": item["key"]} for slot, item in owned.items()])
        common.resilient_json(runtime_root / "production_status_v10_243.json", {
            "updated_at_utc": common.now(), "parent_pid": os.getpid(), "workers": WORKERS,
            "running": running, "completed": len(completed),
            "pending": len(tasks) - next_index, "technical_failures": technical,
            "scientific_failures": scientific, "pause_reason": pause_reason,
            "full_state_checkpoint_interval_years_for_new_members": 30,
        })

    def launch(slot: int, task: dict) -> None:
        key = task["key"]
        legacy = task["kind"] == "legacy-resume"
        row = old_by_key[key] if legacy else new_by_key[key]
        module = OLD_WORKER if legacy else NEW_WORKER
        transaction = Path(manifest["source"]["transaction_dir"]) if legacy else args.transaction_dir
        out_path = log_root / f"{key}.{task['kind']}.out.log"
        err_path = log_root / f"{key}.{task['kind']}.err.log"
        out_handle = out_path.open("x", encoding="utf-8")
        err_handle = err_path.open("x", encoding="utf-8")
        command = [python, "-m", module, "--transaction-dir", str(transaction.resolve()),
                   "--fingerprint", row["forward_fingerprint"], "--output-root", str(root),
                   "--mode", "production"]
        process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[4]), env=env,
                                   stdout=out_handle, stderr=err_handle)
        owned[slot] = {"process": process, "key": key, "kind": task["kind"],
                       "out_handle": out_handle, "err_handle": err_handle, "err_path": err_path}
        with events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"event": "started", "at_utc": common.now(), "slot": slot,
                                      "pid": process.pid, "member_key": key, "kind": task["kind"]}) + "\n")

    while external or owned or next_index < len(tasks):
        for key, item in list(external.items()):
            result = terminal_result(key)
            if result is not None:
                completed[key] = result
                del external[key]
            elif not psutil.pid_exists(item["pid"]):
                tasks.append({"kind": "legacy-resume", "key": key})
                del external[key]
        for slot, item in list(owned.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["out_handle"].close(); item["err_handle"].close()
            result = terminal_result(item["key"])
            if result is None:
                result = {"terminal": True, "technical_failure": True, "scientific_failure": False,
                          "status": "technical-failure-missing-result", "forward_fingerprint": "controller-record"}
            completed[item["key"]] = result
            with events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"event": "finished", "at_utc": common.now(), "slot": slot,
                                          "pid": item["process"].pid, "member_key": item["key"],
                                          "kind": item["kind"], "returncode": code,
                                          "status": result.get("status"),
                                          "stderr_tail": common.stderr_tail(item["err_path"])},
                                         ensure_ascii=False) + "\n")
            del owned[slot]
        memory = psutil.virtual_memory().percent / 100.0
        peak_memory = max(peak_memory, memory)
        if memory > MEMORY_USED_CEILING and pause_reason is None:
            pause_reason = f"memory-needs-user-decision:{memory:.6f}>{MEMORY_USED_CEILING}"
        while pause_reason is None and next_index < len(tasks) and len(external) + len(owned) < WORKERS:
            task = tasks[next_index]; next_index += 1
            if task["key"] in completed:
                continue
            slot = next(index for index in range(WORKERS) if index not in owned)
            launch(slot, task)
        write_status()
        if not external and not owned and next_index >= len(tasks):
            break
        time.sleep(2.0)

    technical = sum(bool(item.get("technical_failure")) for item in completed.values())
    scientific = sum(bool(item.get("scientific_failure")) for item in completed.values())
    batch_completed = pause_reason is None and len(completed) == 530
    summary = {
        "schema": "lia-all191-forward-controller-result-v10.243",
        "status": "completed" if batch_completed else "paused", "parent_pid": os.getpid(),
        "started_at_utc": parent["started_at_utc"], "finished_at_utc": common.now(),
        "total_cells": 530, "completed_cells": len(completed),
        "technical_failures": technical, "scientific_failures": scientific,
        "inherited_completed_at_start": parent["inherited_completed"],
        "inherited_workers_at_start": parent["inherited_workers"],
        "full_state_checkpoint_interval_years_for_new_members": 30,
        "annual_diagnostics_retained": True, "peak_memory_used_fraction": peak_memory,
        "batch_completed": batch_completed, "pause_reason": pause_reason,
    }
    common.resilient_json(runtime_root / "production_summary_v10_243.json", summary)
    write_status()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if batch_completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

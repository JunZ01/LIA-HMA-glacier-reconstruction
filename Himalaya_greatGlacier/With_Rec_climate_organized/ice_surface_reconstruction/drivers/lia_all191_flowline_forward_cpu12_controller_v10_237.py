"""Resilient preflight/CPU12 controller for v10.237 forward curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


WORKERS = 12
MEMORY_USED_CEILING = 0.80
WORKER_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.lia_all191_flowline_forward_v10_237"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def resilient_json(path: Path, payload: object) -> Path:
    errors = []
    for attempt in range(8):
        try:
            atomic_json(path, payload); return path
        except OSError as exc:
            errors.append({"attempt": attempt + 1, "error": repr(exc), "at_utc": now()})
            time.sleep(min(2.0, 0.05 * (2 ** attempt)))
    fallback = path.parent / "fallback" / f"{path.stem}.{time.time_ns()}.{os.getpid()}.json"
    atomic_json(fallback, {"primary_path": str(path), "errors": errors, "payload": payload})
    return fallback


def load_contract(transaction_dir: Path, mode: str):
    manifest = json.loads((transaction_dir / "transaction_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("launch_authorized"):
        raise ValueError("transaction-not-launch-authorized")
    if manifest["runtime"]["controller_sha256"] != sha256(Path(__file__)):
        raise ValueError("controller-hash-mismatch")
    with (transaction_dir / "forward_matrix.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    if len(matrix) != 530 or len({row["forward_fingerprint"] for row in matrix}) != 530:
        raise ValueError("forward-matrix-cardinality-or-fingerprint-failure")
    selected = set(manifest[mode]["fingerprints"])
    rows = [row for row in matrix if row["forward_fingerprint"] in selected]
    if len(rows) != int(manifest[mode]["cell_count"]):
        raise ValueError(f"{mode}-selection-cardinality-failure")
    if mode == "production":
        preflight_path = Path(manifest["runtime"]["unified_short_root"]) / "r" / "preflight_summary.json"
        if not preflight_path.is_file():
            raise ValueError("preflight-summary-missing")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not (preflight.get("status") == "completed" and preflight.get("technical_failures") == 0
                and preflight.get("scientific_failures") == 0 and preflight.get("completed_cells") == 3):
            raise ValueError("preflight-not-passed")
    return manifest, rows


def stderr_tail(path: Path, maximum: int = 4000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-maximum:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "production"), required=True)
    args = parser.parse_args()
    manifest, pending = load_contract(args.transaction_dir, args.mode)
    expected_root = Path(manifest["runtime"]["unified_short_root"]).resolve()
    if args.output_root.resolve() != expected_root:
        raise ValueError("unified-short-root-contract-mismatch")
    group = "p" if args.mode == "preflight" else "m"
    if (args.output_root / group).exists():
        raise FileExistsError(f"{args.mode}-member-root-collision")
    args.output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = args.output_root / "r"; log_root = args.output_root / "l" / args.mode
    runtime_root.mkdir(parents=True, exist_ok=True); log_root.mkdir(parents=True, exist_ok=True)
    parent = {
        "schema": "lia-all191-forward-parent-v10.237", "mode": args.mode,
        "pid": os.getpid(), "started_at_utc": now(), "workers": min(WORKERS, len(pending)),
        "transaction": str(args.transaction_dir.resolve()), "output_root": str(args.output_root.resolve()),
        "command": sys.argv,
    }
    atomic_json(runtime_root / f"{args.mode}_parent.json", parent)

    env = os.environ.copy(); env.update({k: str(v) for k, v in manifest["runtime"]["thread_environment"].items()})
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4])
    python = manifest["runtime"]["python_executable"]
    running = {}; completed = []; next_index = 0; pause_reason = None; peak_memory = 0.0
    events_path = runtime_root / f"{args.mode}_events.jsonl"

    def write_status():
        technical = sum(item["technical_failure"] for item in completed)
        scientific = sum(item["scientific_failure"] for item in completed)
        resilient_json(runtime_root / f"{args.mode}_status.json", {
            "updated_at_utc": now(), "mode": args.mode, "parent_pid": os.getpid(),
            "running": [{"slot": slot, "pid": item["process"].pid,
                         "lia_unique_id": int(item["row"]["lia_unique_id"]),
                         "member_key": item["key"]} for slot, item in running.items()],
            "completed": len(completed), "pending": len(pending) - next_index,
            "technical_failures": technical, "scientific_failures": scientific,
            "pause_reason": pause_reason,
        })

    def launch(slot: int, row: dict):
        key = row["preflight_member_key"] if args.mode == "preflight" else row["production_member_key"]
        out_path = log_root / f"{key}.out.log"; err_path = log_root / f"{key}.err.log"
        out_handle = out_path.open("x", encoding="utf-8"); err_handle = err_path.open("x", encoding="utf-8")
        command = [python, "-m", WORKER_MODULE, "--transaction-dir", str(args.transaction_dir.resolve()),
                   "--fingerprint", row["forward_fingerprint"], "--output-root", str(args.output_root.resolve()),
                   "--mode", args.mode]
        process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[4]), env=env,
                                   stdout=out_handle, stderr=err_handle)
        running[slot] = {"row": row, "key": key, "process": process, "out_handle": out_handle,
                         "err_handle": err_handle, "out_path": out_path, "err_path": err_path,
                         "started_at_utc": now(), "command": command}
        with events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"event": "started", "at_utc": now(), "slot": slot,
                "pid": process.pid, "member_key": key, "lia_unique_id": int(row["lia_unique_id"]),
                "delta_t_eq_degC": float(row["delta_t_eq_degC"]), "command": command}) + "\n")

    worker_limit = min(WORKERS, len(pending))
    while running or next_index < len(pending):
        memory = psutil.virtual_memory().percent / 100.0; peak_memory = max(peak_memory, memory)
        if memory > MEMORY_USED_CEILING and pause_reason is None:
            pause_reason = f"memory-needs-user-decision:{memory:.6f}>{MEMORY_USED_CEILING}"
        while pause_reason is None and next_index < len(pending) and len(running) < worker_limit:
            slot = next(slot for slot in range(worker_limit) if slot not in running)
            launch(slot, pending[next_index]); next_index += 1
        write_status()
        if not running:
            if pause_reason is not None:
                break
            continue
        time.sleep(2.0)
        for slot, item in list(running.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["out_handle"].close(); item["err_handle"].close()
            group_root = "p" if args.mode == "preflight" else "m"
            result_path = args.output_root / group_root / item["key"] / "c" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
                "status": "technical-failure-missing-result", "technical_failure": True, "scientific_failure": False}
            record = {"slot": slot, "pid": item["process"].pid, "member_key": item["key"],
                "lia_unique_id": int(item["row"]["lia_unique_id"]),
                "delta_t_eq_degC": float(item["row"]["delta_t_eq_degC"]), "returncode": code,
                "status": result.get("status"), "technical_failure": bool(result.get("technical_failure", code == 2)),
                "scientific_failure": bool(result.get("scientific_failure", code == 3)),
                "stderr_tail": stderr_tail(item["err_path"]), "finished_at_utc": now()}
            completed.append(record)
            with events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"event": "finished", **record}, ensure_ascii=False) + "\n")
            del running[slot]

    technical = sum(item["technical_failure"] for item in completed)
    scientific = sum(item["scientific_failure"] for item in completed)
    batch_completed = pause_reason is None and len(completed) == len(pending)
    summary = {"schema": "lia-all191-forward-controller-result-v10.237", "mode": args.mode,
        "status": "completed" if batch_completed else "paused", "parent_pid": os.getpid(),
        "started_at_utc": parent["started_at_utc"], "finished_at_utc": now(),
        "requested_cells": len(pending), "dispatched_cells": next_index, "completed_cells": len(completed),
        "technical_failures": technical, "scientific_failures": scientific,
        "peak_memory_used_fraction": peak_memory, "memory_ceiling_fraction": MEMORY_USED_CEILING,
        "batch_completed": batch_completed, "pause_reason": pause_reason, "records": completed}
    resilient_json(runtime_root / f"{args.mode}_summary.json", summary); write_status()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.mode == "preflight" and (technical or scientific):
        return 2
    return 0 if batch_completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU12 controller for the immutable v10.216 all-191 flowline matrix."""

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
INVALID_RATE_MIN_CELLS = 120
INVALID_RATE_CEILING = 0.15
MEMORY_USED_CEILING = 0.80
CHAIN_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.lia_all191_flowline_spinup_v10_216"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def resilient_json(path: Path, payload: object) -> Path:
    errors = []
    for attempt in range(8):
        try:
            atomic_json(path, payload)
            return path
        except OSError as exc:
            errors.append({"attempt": attempt + 1, "error": repr(exc), "at_utc": now()})
            time.sleep(min(2.0, 0.05 * (2 ** attempt)))
    fallback = path.parent / "fallback" / f"{path.stem}.{time.time_ns()}.{os.getpid()}.json"
    atomic_json(fallback, {
        "primary_path": str(path), "primary_write_errors": errors, "payload": payload,
    })
    return fallback


def load_contract(transaction_dir: Path, mode: str) -> tuple[dict, list[dict]]:
    manifest_path = transaction_dir / "transaction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("launch_authorized"):
        raise ValueError("transaction-not-launch-authorized")
    if manifest["runtime"]["controller_sha256"] != sha256(Path(__file__)):
        raise ValueError("controller-hash-mismatch")
    with (transaction_dir / "level1_matrix.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    if len(matrix) != 1337 or len({row["fingerprint"] for row in matrix}) != 1337:
        raise ValueError("matrix-cardinality-or-fingerprint-failure")
    if mode != "production" or not manifest.get("production_authorized"):
        raise ValueError("v10.216-production-authorization-required")
    evidence = manifest["production"]["preflight_evidence"]
    evidence_path = Path(evidence["path"])
    if not evidence_path.is_file() or sha256(evidence_path) != evidence["sha256"]:
        raise ValueError("v10.196-preflight-evidence-hash-mismatch")
    preflight = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not (
        preflight.get("preflight_passed") is True
        and int(preflight.get("completed_cells", -1)) == 12
        and int(preflight.get("technical_failures", -1)) == 0
    ):
        raise ValueError("v10.196-preflight-not-passed")
    expected = int(manifest["production"]["cell_count"])
    selected = set(manifest["production"]["fingerprints"])
    matrix = [row for row in matrix if row["fingerprint"] in selected]
    if len(matrix) != expected or not all(
        str(row["launch_authorized"]).lower() == "true" for row in matrix
    ):
        raise ValueError(f"production-cardinality-or-authorization:{len(matrix)}")
    members = manifest["production"]["members"]
    if len(members) != expected or {item["fingerprint"] for item in members} != selected:
        raise ValueError("short-member-map-mismatch")
    keys = [item["key"] for item in members]
    if len(set(keys)) != expected or any(len(key) != 5 or not key.startswith("c") for key in keys):
        raise ValueError("short-member-key-collision")
    return manifest, matrix


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stderr_tail(path: Path, maximum: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-maximum:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("production",), required=True)
    args = parser.parse_args()
    manifest, pending = load_contract(args.transaction_dir, args.mode)
    expected_root = Path(manifest["runtime"]["unified_short_root"])
    if args.output_root.resolve() != expected_root.resolve():
        raise ValueError("unified-short-root-contract-mismatch")
    if (args.output_root / "r").exists() or (args.output_root / "m").exists():
        raise FileExistsError("v10.216-runtime-or-member-root-collision")
    args.output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = args.output_root / "r"
    log_root = args.output_root / "l"
    runtime_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    member_map = {item["fingerprint"]: item for item in manifest["production"]["members"]}
    parent_record = {
        "schema": "lia-all191-cpu12-parent-v10.216", "mode": args.mode,
        "pid": os.getpid(), "started_at_utc": now(), "workers": WORKERS,
        "transaction": str(args.transaction_dir.resolve()),
        "output_root": str(args.output_root.resolve()), "command": sys.argv,
    }
    atomic_json(runtime_root / "p.json", parent_record)

    thread_env = manifest["runtime"]["thread_environment"]
    env = os.environ.copy()
    env.update({key: str(value) for key, value in thread_env.items()})
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4])
    python = manifest["runtime"]["python_executable"]
    running: dict[int, dict] = {}
    completed: list[dict] = []
    next_index = 0
    stop_reason = None
    pause_reason = None
    peak_memory_used_fraction = 0.0
    worker_events = runtime_root / "events.jsonl"

    def write_status() -> None:
        slots = []
        for slot in range(WORKERS):
            item = running.get(slot)
            slots.append({
                "slot": slot, "state": "running" if item else "idle",
                "pid": None if item is None else item["process"].pid,
                "fingerprint": None if item is None else item["row"]["fingerprint"],
                "lia_unique_id": None if item is None else int(item["row"]["lia_unique_id"]),
                "delta_t_eq_degC": None if item is None else float(item["row"]["delta_t_eq_degC"]),
                "member_key": None if item is None else item["member"]["key"],
            })
        technical_failures = sum(bool(item["technical_failure"]) for item in completed)
        invalid_rate = technical_failures / len(completed) if completed else 0.0
        resilient_json(runtime_root / "status.json", {
            "updated_at_utc": now(), "mode": args.mode, "parent_pid": os.getpid(),
            "slots": slots, "completed": len(completed), "pending": len(pending) - next_index,
            "technical_failures": technical_failures,
            "technical_invalid_rate": invalid_rate,
            "technical_invalid_advisory": bool(
                len(completed) >= INVALID_RATE_MIN_CELLS and invalid_rate > INVALID_RATE_CEILING
            ),
            "pause_reason": pause_reason, "stop_reason": stop_reason,
        })

    def launch(slot: int, row: dict) -> None:
        fingerprint = row["fingerprint"]
        member = member_map[fingerprint]
        stdout_path = log_root / f"{member['key']}.out.log"
        stderr_path = log_root / f"{member['key']}.err.log"
        stdout_handle = stdout_path.open("x", encoding="utf-8")
        stderr_handle = stderr_path.open("x", encoding="utf-8")
        command = [
            python, "-m", CHAIN_MODULE,
            "--transaction-dir", str(args.transaction_dir.resolve()),
            "--fingerprint", fingerprint,
            "--output-root", str(args.output_root.resolve()),
            "--mode", args.mode,
        ]
        process = subprocess.Popen(
            command, cwd=str(Path(__file__).resolve().parents[4]), env=env,
            stdout=stdout_handle, stderr=stderr_handle,
        )
        item = {
            "slot": slot, "row": row, "member": member, "process": process, "command": command,
            "stdout_handle": stdout_handle, "stderr_handle": stderr_handle,
            "stdout_path": stdout_path, "stderr_path": stderr_path,
            "started_at_utc": now(),
        }
        running[slot] = item
        with worker_events.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({
                "event": "started", "at_utc": item["started_at_utc"], "slot": slot,
                "pid": process.pid, "fingerprint": fingerprint,
                "lia_unique_id": int(row["lia_unique_id"]),
                "delta_t_eq_degC": float(row["delta_t_eq_degC"]),
                "member_key": member["key"], "command": command,
            }) + "\n")

    while running or next_index < len(pending):
        memory_used = psutil.virtual_memory().percent / 100.0
        peak_memory_used_fraction = max(peak_memory_used_fraction, memory_used)
        if memory_used > MEMORY_USED_CEILING and pause_reason is None:
            pause_reason = f"memory-used-fraction-needs-user-decision:{memory_used:.6f}>{MEMORY_USED_CEILING}"
        decision_path = runtime_root / "user_decision.json"
        if pause_reason is not None and decision_path.exists():
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            action = decision.get("action")
            if action == "resume":
                pause_reason = None
                decision_path.rename(runtime_root / f"user_decision.applied.{time.time_ns()}.json")
            elif action == "terminate":
                stop_reason = "user-confirmed-termination"
                decision_path.rename(runtime_root / f"user_decision.applied.{time.time_ns()}.json")
        while stop_reason is None and pause_reason is None and next_index < len(pending) and len(running) < WORKERS:
            slot = next(slot for slot in range(WORKERS) if slot not in running)
            launch(slot, pending[next_index])
            next_index += 1
        write_status()
        if not running:
            if pause_reason is not None and stop_reason is None:
                time.sleep(2.0)
                continue
            break
        time.sleep(2.0)
        for slot, item in list(running.items()):
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["stdout_handle"].close()
            item["stderr_handle"].close()
            result_path = args.output_root / "m" / item["member"]["key"] / "c" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
                "status": "technical-failure-missing-result", "technical_failure": True,
            }
            record = {
                "slot": slot, "pid": item["process"].pid,
                "fingerprint": item["row"]["fingerprint"],
                "lia_unique_id": int(item["row"]["lia_unique_id"]),
                "delta_t_eq_degC": float(item["row"]["delta_t_eq_degC"]),
                "member_key": item["member"]["key"],
                "returncode": returncode, "status": result.get("status"),
                "technical_failure": bool(result.get("technical_failure", returncode != 0)),
                "scientific_nonconvergence": bool(result.get("scientific_nonconvergence", False)),
                "stderr_tail": stderr_tail(item["stderr_path"]), "finished_at_utc": now(),
            }
            completed.append(record)
            with worker_events.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"event": "finished", **record}, ensure_ascii=False) + "\n")
            del running[slot]

    technical_failures = sum(bool(item["technical_failure"]) for item in completed)
    batch_completed = (
        stop_reason is None and next_index == len(pending) and len(completed) == len(pending)
    )
    scientific_nonconvergence = sum(bool(item["scientific_nonconvergence"]) for item in completed)
    summary = {
        "schema": "lia-all191-cpu12-controller-result-v10.216", "mode": args.mode,
        "status": "completed" if batch_completed else "stopped",
        "parent_pid": os.getpid(), "started_at_utc": parent_record["started_at_utc"],
        "finished_at_utc": now(), "requested_cells": len(pending),
        "dispatched_cells": next_index, "completed_cells": len(completed),
        "technical_failures": technical_failures,
        "scientific_nonconvergence": scientific_nonconvergence,
        "technical_invalid_rate": technical_failures / len(completed) if completed else 0.0,
        "technical_invalid_rate_is_advisory_only": True,
        "peak_memory_used_fraction": peak_memory_used_fraction,
        "memory_ceiling_fraction": MEMORY_USED_CEILING, "batch_completed": batch_completed,
        "pause_reason": pause_reason, "stop_reason": stop_reason, "records": completed,
    }
    resilient_json(runtime_root / "summary.json", summary)
    write_status()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if batch_completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

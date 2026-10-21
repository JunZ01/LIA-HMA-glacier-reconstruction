"""Lossless CPU20 handoff controller for the active v10.237 forward batch."""

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


WORKERS = 20
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
            time.sleep(min(2.0, 0.05 * (2**attempt)))
    fallback = path.parent / "fallback" / f"{path.stem}.{time.time_ns()}.{os.getpid()}.json"
    atomic_json(fallback, {"primary_path": str(path), "errors": errors, "payload": payload})
    return fallback


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_contract(transaction_dir: Path) -> tuple[dict, list[dict]]:
    manifest = read_json(transaction_dir / "transaction_manifest.json")
    if not manifest.get("launch_authorized"):
        raise ValueError("transaction-not-launch-authorized")
    if manifest["runtime"]["controller_sha256"] != sha256(Path(__file__)):
        raise ValueError("controller-hash-mismatch")
    worker_path = Path(manifest["runtime"]["worker_path"])
    if manifest["runtime"]["worker_sha256"] != sha256(worker_path):
        raise ValueError("worker-hash-mismatch")
    with (transaction_dir / "forward_matrix.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    if len(matrix) != 530 or len({row["forward_fingerprint"] for row in matrix}) != 530:
        raise ValueError("forward-matrix-cardinality-or-fingerprint-failure")
    if set(manifest["production"]["fingerprints"]) != {row["forward_fingerprint"] for row in matrix}:
        raise ValueError("production-fingerprint-set-mismatch")
    return manifest, matrix


def result_path(root: Path, row: dict) -> Path:
    return root / "m" / row["production_member_key"] / "c" / "result.json"


def valid_terminal_result(root: Path, row: dict) -> dict | None:
    path = result_path(root, row)
    if not path.is_file():
        return None
    try:
        result = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not result.get("terminal"):
        return None
    if result.get("forward_fingerprint") != row["forward_fingerprint"]:
        raise ValueError(f"terminal-result-fingerprint-mismatch:{row['production_member_key']}")
    return result


def stderr_tail(path: Path, maximum: int = 4000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-maximum:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--handoff-snapshot", type=Path, required=True)
    args = parser.parse_args()

    manifest, matrix = load_contract(args.transaction_dir)
    root = Path(manifest["runtime"]["unified_short_root"]).resolve()
    if args.output_root.resolve() != root:
        raise ValueError("unified-short-root-contract-mismatch")
    handoff = read_json(args.handoff_snapshot)
    if int(handoff["source_parent_pid"]) != int(manifest["handoff"]["source_parent_pid"]):
        raise ValueError("handoff-parent-mismatch")
    if psutil.pid_exists(int(handoff["source_parent_pid"])):
        raise RuntimeError("source-controller-still-alive")

    rows_by_key = {row["production_member_key"]: row for row in matrix}
    completed_keys = {row["production_member_key"] for row in matrix if valid_terminal_result(root, row)}
    external: dict[str, dict] = {}
    for item in handoff["inherited_workers"]:
        key = item["member_key"]
        row = rows_by_key[key]
        if key in completed_keys:
            continue
        pid = int(item["pid"])
        if psutil.pid_exists(pid):
            external[key] = {"pid": pid, "row": row, "started_at_utc": item.get("started_at_utc")}

    pending = [row for row in matrix if row["production_member_key"] not in completed_keys | set(external)]
    runtime_root = root / "r"
    log_root = root / "l" / "production_v10_240"
    runtime_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=False)
    events_path = runtime_root / "production_events_v10_240.jsonl"
    parent = {
        "schema": "lia-all191-forward-parent-v10.240",
        "pid": os.getpid(),
        "started_at_utc": now(),
        "workers": WORKERS,
        "inherited_workers": len(external),
        "inherited_completed": len(completed_keys),
        "transaction": str(args.transaction_dir.resolve()),
        "handoff_snapshot": str(args.handoff_snapshot.resolve()),
        "output_root": str(root),
        "command": sys.argv,
    }
    atomic_json(runtime_root / "production_parent_v10_240.json", parent)

    env = os.environ.copy()
    env.update({k: str(v) for k, v in manifest["runtime"]["thread_environment"].items()})
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4])
    python = manifest["runtime"]["python_executable"]
    owned: dict[int, dict] = {}
    new_records: list[dict] = []
    next_index = 0
    pause_reason = None
    peak_memory = 0.0

    def recount() -> tuple[int, int, int]:
        terminal = []
        for row in matrix:
            result = valid_terminal_result(root, row)
            if result is not None:
                terminal.append(result)
        technical = sum(bool(item.get("technical_failure")) for item in terminal)
        scientific = sum(bool(item.get("scientific_failure")) for item in terminal)
        return len(terminal), technical, scientific

    def write_status() -> None:
        total_completed, technical, scientific = recount()
        active_rows = [
            {"kind": "inherited", "pid": item["pid"], "member_key": key,
             "lia_unique_id": int(item["row"]["lia_unique_id"])}
            for key, item in external.items()
        ] + [
            {"kind": "v10.240", "slot": slot, "pid": item["process"].pid,
             "member_key": item["key"], "lia_unique_id": int(item["row"]["lia_unique_id"])}
            for slot, item in owned.items()
        ]
        remaining = 530 - total_completed - len(active_rows)
        resilient_json(runtime_root / "production_status_v10_240.json", {
            "updated_at_utc": now(), "parent_pid": os.getpid(), "workers": WORKERS,
            "running": active_rows, "completed": total_completed,
            "pending": max(0, remaining), "technical_failures": technical,
            "scientific_failures": scientific, "pause_reason": pause_reason,
        })

    def launch(slot: int, row: dict) -> None:
        key = row["production_member_key"]
        out_path = log_root / f"{key}.out.log"
        err_path = log_root / f"{key}.err.log"
        out_handle = out_path.open("x", encoding="utf-8")
        err_handle = err_path.open("x", encoding="utf-8")
        command = [
            python, "-m", WORKER_MODULE, "--transaction-dir", str(args.transaction_dir.resolve()),
            "--fingerprint", row["forward_fingerprint"], "--output-root", str(root),
            "--mode", "production",
        ]
        process = subprocess.Popen(
            command, cwd=str(Path(__file__).resolve().parents[4]), env=env,
            stdout=out_handle, stderr=err_handle,
        )
        owned[slot] = {
            "row": row, "key": key, "process": process, "out_handle": out_handle,
            "err_handle": err_handle, "err_path": err_path, "started_at_utc": now(),
        }
        with events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({
                "event": "started", "at_utc": now(), "slot": slot, "pid": process.pid,
                "member_key": key, "lia_unique_id": int(row["lia_unique_id"]),
                "delta_t_eq_degC": float(row["delta_t_eq_degC"]), "command": command,
            }, ensure_ascii=False) + "\n")

    while external or owned or next_index < len(pending):
        for key, item in list(external.items()):
            result = valid_terminal_result(root, item["row"])
            if result is not None or not psutil.pid_exists(item["pid"]):
                if result is None:
                    pending.append(item["row"])
                with events_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps({
                        "event": "inherited-finished" if result is not None else "inherited-resume-queued",
                        "at_utc": now(), "pid": item["pid"], "member_key": key,
                    }) + "\n")
                del external[key]

        for slot, item in list(owned.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["out_handle"].close()
            item["err_handle"].close()
            result = valid_terminal_result(root, item["row"])
            record = {
                "slot": slot, "pid": item["process"].pid, "member_key": item["key"],
                "returncode": code,
                "status": result.get("status") if result else "technical-failure-missing-result",
                "technical_failure": bool(result.get("technical_failure")) if result else True,
                "scientific_failure": bool(result.get("scientific_failure")) if result else False,
                "stderr_tail": stderr_tail(item["err_path"]), "finished_at_utc": now(),
            }
            new_records.append(record)
            with events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"event": "finished", **record}, ensure_ascii=False) + "\n")
            del owned[slot]

        memory = psutil.virtual_memory().percent / 100.0
        peak_memory = max(peak_memory, memory)
        if memory > MEMORY_USED_CEILING and pause_reason is None:
            pause_reason = f"memory-needs-user-decision:{memory:.6f}>{MEMORY_USED_CEILING}"

        while pause_reason is None and next_index < len(pending) and len(external) + len(owned) < WORKERS:
            row = pending[next_index]
            next_index += 1
            if valid_terminal_result(root, row) is not None:
                continue
            used = set(owned)
            slot = next(index for index in range(WORKERS) if index not in used)
            launch(slot, row)

        write_status()
        if not external and not owned and next_index >= len(pending):
            break
        time.sleep(2.0)

    total_completed, technical, scientific = recount()
    batch_completed = pause_reason is None and total_completed == 530
    summary = {
        "schema": "lia-all191-forward-controller-result-v10.240",
        "status": "completed" if batch_completed else "paused",
        "parent_pid": os.getpid(), "started_at_utc": parent["started_at_utc"],
        "finished_at_utc": now(), "total_cells": 530, "completed_cells": total_completed,
        "technical_failures": technical, "scientific_failures": scientific,
        "inherited_completed_at_start": parent["inherited_completed"],
        "inherited_workers_at_start": parent["inherited_workers"],
        "new_records": new_records, "peak_memory_used_fraction": peak_memory,
        "memory_ceiling_fraction": MEMORY_USED_CEILING, "batch_completed": batch_completed,
        "pause_reason": pause_reason,
    }
    resilient_json(runtime_root / "production_summary_v10_240.json", summary)
    write_status()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if batch_completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

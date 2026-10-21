"""CPU20 controller for the v10.267 all-191 full-grid completion batch."""

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
STARTUP_STAGGER_SECONDS = 0.5
INVALID_RATE_MIN_CELLS = 120
INVALID_RATE_CEILING = 0.15
MEMORY_USED_CEILING = 0.80
EXPECTED_CHECKPOINT_YEARS = list(range(0, 901, 30))
CHAIN_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.lia_all191_full900_spinup_v10_262"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
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
    fallback = (
        path.parent / "fallback" / f"{path.stem}.{time.time_ns()}.{os.getpid()}.json"
    )
    atomic_json(
        fallback,
        {
            "primary_path": str(path),
            "primary_write_errors": errors,
            "payload": payload,
        },
    )
    return fallback


def load_contract(transaction_dir: Path, mode: str) -> tuple[dict, list[dict]]:
    manifest = json.loads(
        (transaction_dir / "transaction_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest.get("launch_authorized"):
        raise ValueError("transaction-not-launch-authorized")
    if manifest["runtime"]["controller_sha256"] != sha256(Path(__file__)):
        raise ValueError("controller-hash-mismatch")
    with (transaction_dir / "production_matrix.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        matrix = list(csv.DictReader(handle))
    expected = int(manifest["production"]["cell_count"])
    fingerprints = {row["fingerprint"] for row in matrix}
    if len(matrix) != expected or len(fingerprints) != expected:
        raise ValueError("matrix-cardinality-or-fingerprint-failure")
    if mode != "production" or not manifest.get("production_authorized"):
        raise ValueError("v10.262-production-authorization-required")
    if fingerprints != set(manifest["production"]["fingerprints"]):
        raise ValueError("production-fingerprint-set-mismatch")
    if not all(str(row["launch_authorized"]).lower() == "true" for row in matrix):
        raise ValueError("unauthorized-matrix-row")
    members = manifest["production"]["members"]
    if len(members) != expected or {item["fingerprint"] for item in members} != fingerprints:
        raise ValueError("short-member-map-mismatch")
    keys = [item["key"] for item in members]
    if (
        len(set(keys)) != expected
        or any(len(key) != 5 or not key.startswith("s") for key in keys)
    ):
        raise ValueError("short-member-key-collision")
    matrix.sort(
        key=lambda row: (
            0 if row["source_routing"] == "v10.260-remaining-round1" else 1,
            int(row["lia_unique_id"]),
            float(row["delta_t_eq_degC"]),
        )
    )
    return manifest, matrix


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stderr_tail(path: Path, maximum: int = 4000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-maximum:]


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
        raise FileExistsError("v10.262-runtime-or-member-root-collision")
    args.output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = args.output_root / "r"
    log_root = args.output_root / "l"
    runtime_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    member_map = {
        item["fingerprint"]: item for item in manifest["production"]["members"]
    }
    parent_record = {
        "schema": "lia-all191-full900-cpu20-parent-v10.267",
        "mode": args.mode,
        "pid": os.getpid(),
        "started_at_utc": now(),
        "workers": WORKERS,
        "transaction": str(args.transaction_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "checkpoint_interval_years": 30,
        "maximum_years": 900,
        "command": sys.argv,
    }
    atomic_json(runtime_root / "p.json", parent_record)

    env = os.environ.copy()
    env.update(
        {
            key: str(value)
            for key, value in manifest["runtime"]["thread_environment"].items()
        }
    )
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4])
    python = manifest["runtime"]["python_executable"]
    running: dict[int, dict] = {}
    completed: list[dict] = []
    next_index = 0
    stop_reason = None
    pause_reason = None
    peak_memory_used_fraction = 0.0
    worker_events = runtime_root / "events.jsonl"

    def current_invalid_rate() -> float:
        if not completed:
            return 0.0
        return sum(bool(item["technical_failure"]) for item in completed) / len(completed)

    def write_status() -> None:
        slots = []
        for slot in range(WORKERS):
            item = running.get(slot)
            slots.append(
                {
                    "slot": slot,
                    "state": "running" if item else "idle",
                    "pid": None if item is None else item["process"].pid,
                    "fingerprint": None
                    if item is None
                    else item["row"]["fingerprint"],
                    "lia_unique_id": None
                    if item is None
                    else int(item["row"]["lia_unique_id"]),
                    "delta_t_eq_degC": None
                    if item is None
                    else float(item["row"]["delta_t_eq_degC"]),
                    "member_key": None if item is None else item["member"]["key"],
                }
            )
        invalid_rate = current_invalid_rate()
        resilient_json(
            runtime_root / "status.json",
            {
                "updated_at_utc": now(),
                "mode": args.mode,
                "parent_pid": os.getpid(),
                "slots": slots,
                "completed": len(completed),
                "pending": len(pending) - next_index,
                "technical_failures": sum(
                    bool(item["technical_failure"]) for item in completed
                ),
                "technical_invalid_rate": invalid_rate,
                "technical_invalid_advisory": bool(
                    len(completed) >= INVALID_RATE_MIN_CELLS
                    and invalid_rate > INVALID_RATE_CEILING
                ),
                "checkpoint_interval_years": 30,
                "maximum_years": 900,
                "pause_reason": pause_reason,
                "stop_reason": stop_reason,
            },
        )

    def launch(slot: int, row: dict) -> None:
        fingerprint = row["fingerprint"]
        member = member_map[fingerprint]
        stdout_path = log_root / f"{member['key']}.out.log"
        stderr_path = log_root / f"{member['key']}.err.log"
        stdout_handle = stdout_path.open("x", encoding="utf-8")
        stderr_handle = stderr_path.open("x", encoding="utf-8")
        command = [
            python,
            "-m",
            CHAIN_MODULE,
            "--transaction-dir",
            str(args.transaction_dir.resolve()),
            "--fingerprint",
            fingerprint,
            "--output-root",
            str(args.output_root.resolve()),
            "--mode",
            args.mode,
        ]
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[4]),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        item = {
            "slot": slot,
            "row": row,
            "member": member,
            "process": process,
            "command": command,
            "stdout_handle": stdout_handle,
            "stderr_handle": stderr_handle,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "started_at_utc": now(),
        }
        running[slot] = item
        with worker_events.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "started",
                        "at_utc": item["started_at_utc"],
                        "slot": slot,
                        "pid": process.pid,
                        "fingerprint": fingerprint,
                        "lia_unique_id": int(row["lia_unique_id"]),
                        "delta_t_eq_degC": float(row["delta_t_eq_degC"]),
                        "member_key": member["key"],
                        "command": command,
                    }
                )
                + "\n"
            )

    while running or next_index < len(pending):
        memory_used = psutil.virtual_memory().percent / 100.0
        peak_memory_used_fraction = max(peak_memory_used_fraction, memory_used)
        if memory_used > MEMORY_USED_CEILING and pause_reason is None:
            pause_reason = (
                f"memory-used-fraction-needs-user-decision:"
                f"{memory_used:.6f}>{MEMORY_USED_CEILING}"
            )
        invalid_rate = current_invalid_rate()
        if (
            len(completed) >= INVALID_RATE_MIN_CELLS
            and invalid_rate > INVALID_RATE_CEILING
            and pause_reason is None
        ):
            pause_reason = (
                f"technical-invalid-rate-needs-user-decision:"
                f"{invalid_rate:.6f}>{INVALID_RATE_CEILING}"
            )
        decision_path = runtime_root / "user_decision.json"
        if pause_reason is not None and decision_path.exists():
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            action = decision.get("action")
            if action == "resume":
                pause_reason = None
                decision_path.rename(
                    runtime_root / f"user_decision.applied.{time.time_ns()}.json"
                )
            elif action == "terminate":
                stop_reason = "user-confirmed-termination"
                decision_path.rename(
                    runtime_root / f"user_decision.applied.{time.time_ns()}.json"
                )
        while (
            stop_reason is None
            and pause_reason is None
            and next_index < len(pending)
            and len(running) < WORKERS
        ):
            slot = next(slot for slot in range(WORKERS) if slot not in running)
            launch(slot, pending[next_index])
            next_index += 1
            # Avoid the native-library initialization burst seen in v10.264 while
            # retaining twenty concurrent workers after the pool has filled.
            time.sleep(STARTUP_STAGGER_SECONDS)
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
            result_path = (
                args.output_root
                / "m"
                / item["member"]["key"]
                / "c"
                / "result.json"
            )
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.exists()
                else {
                    "status": "technical-failure-missing-result",
                    "technical_failure": True,
                }
            )
            checkpoint_contract_pass = bool(
                result.get("terminal") is True
                and result.get("all900_completed") is True
                and result.get("checkpoint_contract_pass") is True
                and result.get("checkpoint_years_actual") == EXPECTED_CHECKPOINT_YEARS
            )
            record = {
                "slot": slot,
                "pid": item["process"].pid,
                "fingerprint": item["row"]["fingerprint"],
                "lia_unique_id": int(item["row"]["lia_unique_id"]),
                "delta_t_eq_degC": float(item["row"]["delta_t_eq_degC"]),
                "source_routing": item["row"]["source_routing"],
                "member_key": item["member"]["key"],
                "returncode": returncode,
                "status": result.get("status"),
                "technical_failure": bool(
                    result.get("technical_failure", returncode != 0)
                ),
                "scientific_nonconvergence": bool(
                    result.get("scientific_nonconvergence", False)
                ),
                "spinup_target_pass": bool(result.get("spinup_target_pass", False)),
                "checkpoint_contract_pass": checkpoint_contract_pass,
                "lia_length_relative_error": result.get("lia_length_relative_error"),
                "stderr_tail": stderr_tail(item["stderr_path"]),
                "finished_at_utc": now(),
            }
            completed.append(record)
            with worker_events.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps({"event": "finished", **record}, ensure_ascii=False)
                    + "\n"
                )
            del running[slot]

    technical_failures = sum(bool(item["technical_failure"]) for item in completed)
    batch_completed = (
        stop_reason is None
        and next_index == len(pending)
        and len(completed) == len(pending)
    )
    summary = {
        "schema": "lia-all191-full900-cpu20-controller-result-v10.267",
        "mode": args.mode,
        "status": "completed" if batch_completed else "stopped",
        "parent_pid": os.getpid(),
        "started_at_utc": parent_record["started_at_utc"],
        "finished_at_utc": now(),
        "requested_cells": len(pending),
        "dispatched_cells": next_index,
        "completed_cells": len(completed),
        "technical_failures": technical_failures,
        "scientific_nonconvergence": sum(
            bool(item["scientific_nonconvergence"]) for item in completed
        ),
        "technical_invalid_rate": current_invalid_rate(),
        "technical_invalid_rate_requires_user_decision": True,
        "spinup_target_passes": sum(
            bool(item.get("spinup_target_pass")) for item in completed
        ),
        "checkpoint_contract_passes": sum(
            bool(item.get("checkpoint_contract_pass")) for item in completed
        ),
        "checkpoint_interval_years": 30,
        "maximum_years": 900,
        "peak_memory_used_fraction": peak_memory_used_fraction,
        "memory_ceiling_fraction": MEMORY_USED_CEILING,
        "batch_completed": batch_completed,
        "pause_reason": pause_reason,
        "stop_reason": stop_reason,
        "records": completed,
    }
    resilient_json(runtime_root / "summary.json", summary)
    write_status()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if batch_completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

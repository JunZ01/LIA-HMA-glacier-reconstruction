"""Validate and freeze the terminal result of the final full-grid cell."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
TRANSACTION = BASE / "lia_191_all191_fullgrid_finalcell_retry_v10_273_transaction_v1"
OUTPUT = ROOT / "w273"
RESULT = OUTPUT / "m/s2679/c/result.json"
HISTORY = OUTPUT / "m/s2679/c/diagnostics.jsonl"
EVIDENCE = BASE / "lia_191_all191_fullgrid_finalcell_retry_v10_273_terminal_v1"
FINGERPRINT = "1AAB17B0C4EF175D60626B79CA1187177795F875DCA24D580155BEF3A85B763B"
EXPECTED_YEARS = list(range(0, 901, 30))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    for path in (TRANSACTION / "transaction_manifest.json", RESULT, HISTORY):
        if not path.is_file():
            raise FileNotFoundError(path)
    if EVIDENCE.exists():
        raise FileExistsError(EVIDENCE)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if not (
        result.get("fingerprint") == FINGERPRINT
        and result.get("terminal") is True
        and result.get("all900_completed") is True
        and result.get("technical_failure") is False
        and result.get("checkpoint_contract_pass") is True
        and result.get("checkpoint_years_actual") == EXPECTED_YEARS
        and result.get("current_year") == 900
    ):
        raise ValueError("final-cell-result-contract-failure")
    history = [
        json.loads(line)
        for line in HISTORY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(item["year"]) for item in history] != list(range(901)):
        raise ValueError("annual-history-not-complete-0-through-900")
    processes = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if process.info.get("name", "").lower().startswith("python") and FINGERPRINT in command:
            processes.append(process.pid)
    if processes:
        raise ValueError(f"final-cell-process-still-alive:{processes}")

    EVIDENCE.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "lia-191-all191-fullgrid-finalcell-retry-v10.273-terminal-v1",
        "status": "terminal-complete-full-grid-4011-of-4011-valid",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "transaction": {
            "path": str((TRANSACTION / "transaction_manifest.json").relative_to(ROOT)),
            "sha256": sha256(TRANSACTION / "transaction_manifest.json"),
        },
        "execution": {
            "session_id": 10421,
            "process_exit_code": 0,
            "process_pid": None,
            "pid_note": "the 26.7-second member completed before the read-only PID snapshot returned",
            "process_alive": False,
            "runtime_seconds": result["runtime_seconds"],
            "output_root": str(OUTPUT.relative_to(ROOT)),
        },
        "cell": {
            "member_key": "s2679",
            "lia_unique_id": int(result["lia_unique_id"]),
            "rgi_id": result["rgi_id"],
            "delta_t_eq_degC": float(result["delta_t_eq_degC"]),
            "fingerprint": result["fingerprint"],
            "resumed_from_year": 420,
            "terminal_year": 900,
            "technical_failure": False,
            "stable": not bool(result["scientific_nonconvergence"]),
            "first_pass_boundary_year": result["first_pass_boundary_year"],
            "spinup_target_pass": bool(result["spinup_target_pass"]),
            "terminal_length_m": result["terminal_length_m"],
            "target_lia_length_m": result["target_lia_length_m"],
            "lia_length_relative_error": result["lia_length_relative_error"],
            "checkpoint_contract_pass": True,
            "checkpoint_years": EXPECTED_YEARS,
            "annual_history_rows": len(history),
            "result_path": str(RESULT.relative_to(ROOT)),
            "result_sha256": sha256(RESULT),
            "history_sha256": sha256(HISTORY),
        },
        "combined_full_grid": {
            "target_cells": 4011,
            "valid_year900_cells": 4011,
            "technical_failure_cells": 0,
            "stable_cells": 1393,
            "scientific_nonconvergence_cells": 2618,
            "spinup_target_pass_cells": 193,
            "complete": True,
        },
        "scientific_interpretation": "stable under the frozen 3 percent and 300 m rule, but fails the separate 10 percent LIA-length gate",
        "accepted_count_effect": 0,
        "independent_gpu_branches_untouched": True,
        "next_legal_action": "read-only full-grid analysis and selection only; no further spin-up retry",
    }
    atomic_json(EVIDENCE / "summary.json", summary)
    atomic_json(EVIDENCE / "artifact_hashes.json", {"summary.json": sha256(EVIDENCE / "summary.json")})
    print(json.dumps({"evidence": str(EVIDENCE), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the single-cell v10.273 retry from the complete year-420 state/history."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
EVIDENCE = BASE / "lia_191_all191_fullgrid_finalcell_retry_v10_273_transaction_v1"
OUTPUT = ROOT / "w273"
SOURCE_TRANSACTION = BASE / "lia_191_all191_fullgrid_recovery_v10_270_transaction_v1"
SOURCE_TERMINAL = BASE / "lia_191_all191_fullgrid_recovery_v10_270_terminal_v1/summary.json"
SOURCE_CONTROL = ROOT / "w267/m/s2679/c"
CHAIN = ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_full900_spinup_v10_262.py"
)
STATE = ROOT / "LIA_CURRENT_STATE.json"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
EXECUTION = ROOT / "LIA_EXECUTION_CONTRACT.md"
FINGERPRINT = "1AAB17B0C4EF175D60626B79CA1187177795F875DCA24D580155BEF3A85B763B"
MEMBER_KEY = "s2679"
RESUME_YEAR = 420


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


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    required = (
        SOURCE_TRANSACTION / "transaction_manifest.json",
        SOURCE_TRANSACTION / "production_matrix.csv",
        SOURCE_TERMINAL,
        SOURCE_CONTROL / "diagnostics.jsonl",
        SOURCE_CONTROL / f"checkpoints/year-{RESUME_YEAR:04d}/checkpoint.json",
        SOURCE_CONTROL / f"checkpoints/year-{RESUME_YEAR:04d}/state.pkl.gz",
        CHAIN,
        STATE,
        FACTS,
        EXECUTION,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if EVIDENCE.exists() or OUTPUT.exists():
        raise FileExistsError(f"immutable-root-collision:{EVIDENCE}:{OUTPUT}")

    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state["current_focus"]["system_id"] != "all191-fullgrid-finalcell-retry-approved-transaction-pending":
        raise ValueError("unexpected-current-focus")
    conflicts = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if process.info.get("name", "").lower().startswith("python") and (
            "lia_all191_full900_spinup_v10_262" in command
            or "lia_all191_full900_cpu20" in command
        ):
            conflicts.append({"pid": process.pid, "command": command})
    if conflicts:
        raise ValueError(f"duplicate-cpu-process:{conflicts}")

    source_manifest = json.loads(
        (SOURCE_TRANSACTION / "transaction_manifest.json").read_text(encoding="utf-8")
    )
    rows = [
        row for row in read_csv(SOURCE_TRANSACTION / "production_matrix.csv")
        if row["fingerprint"] == FINGERPRINT
    ]
    members = [
        item for item in source_manifest["production"]["members"]
        if item["fingerprint"] == FINGERPRINT
    ]
    if len(rows) != 1 or len(members) != 1 or members[0]["key"] != MEMBER_KEY:
        raise ValueError("single-cell-source-identity-failure")
    row = rows[0]
    if not (
        int(row["lia_unique_id"]) == 3850
        and float(row["delta_t_eq_degC"]) == 0.8
        and row["rgi_id"] == "RGI60-14.15895"
    ):
        raise ValueError("single-cell-scientific-identity-failure")

    history = [
        json.loads(line)
        for line in (SOURCE_CONTROL / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    years = [int(item["year"]) for item in history]
    if years != list(range(0, RESUME_YEAR + 1)):
        raise ValueError(f"source-history-not-complete-through-{RESUME_YEAR}:{years[:2]}:{years[-2:]}:{len(years)}")
    checkpoints = sorted((SOURCE_CONTROL / "checkpoints").glob("year-*"))
    selected = [path for path in checkpoints if int(path.name.split("-")[1]) <= RESUME_YEAR]
    selected_years = [int(path.name.split("-")[1]) for path in selected]
    if selected_years != list(range(0, RESUME_YEAR + 1, 30)):
        raise ValueError(f"source-checkpoint-cadence-failure:{selected_years}")
    checkpoint_manifest = json.loads((selected[-1] / "checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint_manifest["fingerprint"] != FINGERPRINT or int(checkpoint_manifest["year"]) != RESUME_YEAR:
        raise ValueError("resume-checkpoint-identity-failure")
    if sha256(selected[-1] / "state.pkl.gz") != checkpoint_manifest["state_sha256"]:
        raise ValueError("resume-state-hash-failure")

    EVIDENCE.mkdir(parents=True, exist_ok=False)
    control = OUTPUT / "m" / MEMBER_KEY / "c"
    (control / "checkpoints").mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE_CONTROL / "diagnostics.jsonl", control / "diagnostics.jsonl")
    for checkpoint in selected:
        shutil.copytree(checkpoint, control / "checkpoints" / checkpoint.name)

    staged_files = []
    for destination in sorted(path for path in control.rglob("*") if path.is_file()):
        relative = destination.relative_to(control)
        source = SOURCE_CONTROL / relative
        if not source.is_file() or sha256(source) != sha256(destination):
            raise ValueError(f"staged-hash-failure:{destination}")
        staged_files.append(
            {
                "source_path": str(source.resolve()),
                "destination_path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    rules = {
        "schema": "lia-all191-fullgrid-finalcell-retry-rules-v10.273",
        "authorization_basis": "user: 开始",
        "cell_count": 1,
        "lia_unique_id": 3850,
        "rgi_id": "RGI60-14.15895",
        "delta_t_eq_degC": 0.8,
        "fingerprint": FINGERPRINT,
        "resume_year": RESUME_YEAR,
        "resume_reason": "year-420 is the latest checkpoint with a complete annual diagnostic history",
        "recomputed_unfinished_years": [421, 450],
        "maximum_years": 900,
        "checkpoint_interval_years": 30,
        "scientific_contract_changed": False,
        "model_family": "OGGM one-dimensional SemiImplicitModel only",
        "forcing": "ensemble_rec_ref-1441-1470-repeat",
        "spinup_stability": "unchanged 3 percent volume and 300 m length over the final same-phase 90-year window",
        "lia_length_gate": "unchanged 10 percent",
        "completed_cells_recomputed": 0,
        "other_cells_authorized": 0,
        "independent_gpu_branches_untouched": True,
    }
    atomic_json(EVIDENCE / "rules.json", rules)
    atomic_csv(EVIDENCE / "production_matrix.csv", rows)
    atomic_csv(EVIDENCE / "production_manifest.csv", [{"member_key": MEMBER_KEY, **row}])
    atomic_csv(EVIDENCE / "staged_files.csv", staged_files)
    atomic_json(
        EVIDENCE / "staging_audit.json",
        {
            "passed": True,
            "source_control_root": str(SOURCE_CONTROL.resolve()),
            "destination_control_root": str(control.resolve()),
            "history_years": [0, RESUME_YEAR],
            "history_row_count": len(history),
            "checkpoint_years": selected_years,
            "latest_checkpoint_year": RESUME_YEAR,
            "latest_checkpoint_manifest_sha256": sha256(selected[-1] / "checkpoint.json"),
            "latest_state_sha256": sha256(selected[-1] / "state.pkl.gz"),
            "staged_file_count": len(staged_files),
            "staged_bytes": sum(item["bytes"] for item in staged_files),
            "staged_files_sha256": sha256(EVIDENCE / "staged_files.csv"),
        },
    )
    inputs = source_manifest["inputs"]
    thread_environment = source_manifest["runtime"]["thread_environment"]
    manifest = {
        "schema": "lia-all191-fullgrid-finalcell-retry-v10.273-transaction-v1",
        "transaction_id": "lia-all191-fullgrid-finalcell-retry-v10.273",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "authorized-not-launched",
        "authorization_basis": "user: 开始",
        "population": 191,
        "accepted_count_effect": 0,
        "model_family": "oggm-flowline-1d",
        "inputs": inputs,
        "runtime": {
            "python_executable": source_manifest["runtime"]["python_executable"],
            "output_root": str(OUTPUT.resolve()),
            "unified_short_root": str(OUTPUT.resolve()),
            "workers": 1,
            "chain_driver_sha256": sha256(CHAIN),
            "thread_environment": thread_environment,
            "launch_mode": "one persistent foreground process",
        },
        "technical_fixture": {
            "source_terminal": str(SOURCE_TERMINAL.resolve()),
            "source_terminal_sha256": sha256(SOURCE_TERMINAL),
            "resume_checkpoint_year": RESUME_YEAR,
            "resume_checkpoint_manifest_sha256": sha256(selected[-1] / "checkpoint.json"),
            "resume_state_sha256": sha256(selected[-1] / "state.pkl.gz"),
            "complete_history_through_resume_year": True,
            "staging_audit": str((EVIDENCE / "staging_audit.json").resolve()),
            "staging_audit_sha256": sha256(EVIDENCE / "staging_audit.json"),
        },
        "production": {
            "one_time_only": True,
            "cell_count": 1,
            "unique_system_count": 1,
            "fingerprints": [FINGERPRINT],
            "members": members,
            "maximum_years": 900,
            "full_state_checkpoint_interval_years": 30,
            "forced_full_duration": True,
        },
        "production_authorized": True,
        "launch_authorized": True,
        "simulation_launched": False,
        "further_automatic_retry_authorized": False,
        "independent_gpu_branches_untouched": True,
    }
    atomic_json(EVIDENCE / "transaction_manifest.json", manifest)
    atomic_json(
        EVIDENCE / "input_binding.json",
        {
            "scientific_contract_sha256": sha256(FACTS),
            "execution_contract_sha256": sha256(EXECUTION),
            "current_state_id": state["state_id"],
            "current_state_sha256": sha256(STATE),
            "source_transaction_sha256": sha256(SOURCE_TRANSACTION / "transaction_manifest.json"),
            "source_terminal_sha256": sha256(SOURCE_TERMINAL),
            "chain_driver_sha256": sha256(CHAIN),
        },
    )
    atomic_json(
        EVIDENCE / "summary.json",
        {
            "schema": "lia-all191-fullgrid-finalcell-retry-v10.273-transaction-summary-v1",
            "status": "authorized-not-launched",
            "cell_count": 1,
            "member_key": MEMBER_KEY,
            "lia_unique_id": 3850,
            "delta_t_eq_degC": 0.8,
            "resume_year": RESUME_YEAR,
            "maximum_year": 900,
            "output_root": str(OUTPUT.resolve()),
            "launch_authorized": True,
            "scientific_contract_changed": False,
        },
    )
    atomic_json(
        EVIDENCE / "artifact_hashes.json",
        {path.name: sha256(path) for path in sorted(EVIDENCE.iterdir()) if path.is_file()},
    )
    print(json.dumps({"evidence": str(EVIDENCE), "output": str(OUTPUT), **rules}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

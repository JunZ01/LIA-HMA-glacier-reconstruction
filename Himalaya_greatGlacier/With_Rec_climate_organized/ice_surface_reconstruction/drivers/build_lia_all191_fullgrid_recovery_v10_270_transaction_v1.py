"""Build and stage the exact 884-cell v10.267 recovery transaction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
EVIDENCE = BASE / "lia_191_all191_fullgrid_recovery_v10_270_transaction_v1"
OUTPUT_ROOT = PROJECT_ROOT / "w270"
SOURCE_TRANSACTION = BASE / "lia_191_all191_fullgrid_completion_v10_267_transaction_v1"
SOURCE_TERMINAL = BASE / "lia_191_all191_fullgrid_completion_v10_267_terminal_v1"
SOURCE_OUTPUT = PROJECT_ROOT / "w267"
CHAIN_DRIVER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_full900_spinup_v10_262.py"
)
BASE_CONTROLLER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_full900_cpu20_controller_v10_267.py"
)
RECOVERY_WRAPPER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_full900_cpu20_recovery_controller_v10_270.py"
)
CURRENT_STATE = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
FACTS = PROJECT_ROOT / "LIA_PROJECT_FACTS.md"
EXECUTION_CONTRACT = PROJECT_ROOT / "LIA_EXECUTION_CONTRACT.md"
DECISION_LOG = PROJECT_ROOT / "LIA_DECISION_LOG.md"
EXPECTED_CHECKPOINTS = list(range(0, 901, 30))


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
    if not rows:
        raise ValueError(f"empty-csv:{path}")
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


def assert_no_duplicate_cpu_process() -> None:
    needles = (
        "lia_all191_full900_cpu20_controller_v10_267",
        "lia_all191_full900_cpu20_recovery_controller_v10_270",
        "lia_all191_full900_spinup_v10_262",
    )
    conflicts = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if process.info.get("name", "").lower().startswith("python") and any(
            needle in command for needle in needles
        ):
            conflicts.append({"pid": process.pid, "command": command})
    if conflicts:
        raise ValueError(f"duplicate-cpu-process:{conflicts}")


def main() -> int:
    required = (
        SOURCE_TRANSACTION / "transaction_manifest.json",
        SOURCE_TRANSACTION / "production_matrix.csv",
        SOURCE_TERMINAL / "summary.json",
        SOURCE_TERMINAL / "recovery_queue.csv",
        CHAIN_DRIVER,
        BASE_CONTROLLER,
        RECOVERY_WRAPPER,
        CURRENT_STATE,
        FACTS,
        EXECUTION_CONTRACT,
        DECISION_LOG,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if EVIDENCE.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError(f"immutable-root-collision:{EVIDENCE}:{OUTPUT_ROOT}")
    assert_no_duplicate_cpu_process()

    state = json.loads(CURRENT_STATE.read_text(encoding="utf-8"))
    focus = state["current_focus"]
    if focus["system_id"] != "all191-fullgrid-completion-v10.267-terminal-recovery-decision-required":
        raise ValueError("unexpected-current-focus")
    if int(focus["unfinished_cells"]) != 884:
        raise ValueError("unexpected-unfinished-count")
    if int(state["counts"]["active"]) != 0 or int(state["counts"]["queued"]) != 0:
        raise ValueError("cpu-controller-ledger-not-idle")
    if bool(focus.get("launch_authorized")):
        raise ValueError("stale-launch-authority-present")

    source_manifest = json.loads(
        (SOURCE_TRANSACTION / "transaction_manifest.json").read_text(encoding="utf-8")
    )
    source_terminal = json.loads(
        (SOURCE_TERMINAL / "summary.json").read_text(encoding="utf-8")
    )
    source_rows = read_csv(SOURCE_TRANSACTION / "production_matrix.csv")
    source_by_fingerprint = {row["fingerprint"]: row for row in source_rows}
    source_members = {
        item["fingerprint"]: item for item in source_manifest["production"]["members"]
    }
    queue = read_csv(SOURCE_TERMINAL / "recovery_queue.csv")
    if len(queue) != 884 or len({row["fingerprint"] for row in queue}) != 884:
        raise ValueError("recovery-queue-cardinality-failure")
    if set(row["fingerprint"] for row in queue) - set(source_by_fingerprint):
        raise ValueError("recovery-queue-not-subset-of-source-matrix")
    actions = {}
    for row in queue:
        actions[row["recovery_action"]] = actions.get(row["recovery_action"], 0) + 1
    if actions != {"resume-validated-checkpoint": 20, "start-from-year0": 864}:
        raise ValueError(f"recovery-action-count-failure:{actions}")

    completed_fingerprints = set()
    for result_path in SOURCE_OUTPUT.glob("m/*/c/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("terminal") is True
            and result.get("all900_completed") is True
            and result.get("checkpoint_contract_pass") is True
        ):
            completed_fingerprints.add(result["fingerprint"])
    if len(completed_fingerprints) != 2663:
        raise ValueError(f"source-completed-count-failure:{len(completed_fingerprints)}")
    queue_fingerprints = {row["fingerprint"] for row in queue}
    if completed_fingerprints & queue_fingerprints:
        raise ValueError("completed-result-present-in-recovery-queue")
    if len(completed_fingerprints | queue_fingerprints) != 3547:
        raise ValueError("source-partition-coverage-failure")

    matrix = [source_by_fingerprint[row["fingerprint"]] for row in queue]
    members = [source_members[row["fingerprint"]] for row in queue]
    if len({item["key"] for item in members}) != 884:
        raise ValueError("recovery-member-key-collision")

    EVIDENCE.mkdir(parents=True, exist_ok=False)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    member_root = OUTPUT_ROOT / "m"
    member_root.mkdir(parents=True, exist_ok=False)

    staged_files: list[dict] = []
    staged_members: list[dict] = []
    for row in queue:
        if row["recovery_action"] != "resume-validated-checkpoint":
            continue
        member = source_members[row["fingerprint"]]
        source_control = SOURCE_OUTPUT / "m" / member["key"] / "c"
        destination_control = member_root / member["key"] / "c"
        if (source_control / "result.json").exists():
            raise ValueError(f"partial-source-has-terminal-result:{member['key']}")
        checkpoints = sorted((source_control / "checkpoints").glob("year-*"))
        if len(checkpoints) != int(row["checkpoint_count"]):
            raise ValueError(f"partial-checkpoint-count-mismatch:{member['key']}")
        latest = checkpoints[-1]
        latest_manifest = latest / "checkpoint.json"
        latest_state = latest / "state.pkl.gz"
        if int(latest.name.split("-")[1]) != int(row["latest_checkpoint_year"]):
            raise ValueError(f"partial-latest-year-mismatch:{member['key']}")
        if sha256(latest_manifest) != row["latest_checkpoint_manifest_sha256"]:
            raise ValueError(f"partial-manifest-hash-mismatch:{member['key']}")
        if sha256(latest_state) != row["latest_state_sha256"]:
            raise ValueError(f"partial-state-hash-mismatch:{member['key']}")
        shutil.copytree(source_control, destination_control)
        copied = 0
        for destination in sorted(path for path in destination_control.rglob("*") if path.is_file()):
            relative = destination.relative_to(destination_control)
            source = source_control / relative
            source_hash = sha256(source)
            destination_hash = sha256(destination)
            if source_hash != destination_hash:
                raise ValueError(f"recovery-copy-hash-mismatch:{destination}")
            staged_files.append(
                {
                    "member_key": member["key"],
                    "fingerprint": row["fingerprint"],
                    "source_path": str(source.resolve()),
                    "destination_path": str(destination.resolve()),
                    "bytes": destination.stat().st_size,
                    "sha256": destination_hash,
                }
            )
            copied += 1
        staged_members.append(
            {
                "member_key": member["key"],
                "fingerprint": row["fingerprint"],
                "lia_unique_id": int(row["lia_unique_id"]),
                "delta_t_eq_degC": float(row["delta_t_eq_degC"]),
                "checkpoint_count": int(row["checkpoint_count"]),
                "latest_checkpoint_year": int(row["latest_checkpoint_year"]),
                "copied_files": copied,
            }
        )
    if len(staged_members) != 20:
        raise ValueError("staged-member-count-failure")

    rules = {
        "schema": "lia-all191-fullgrid-recovery-rules-v10.270",
        "authorization_basis": "user: 允许恢复剩余884个",
        "source_transaction": "lia-all191-fullgrid-completion-v10.267",
        "cells": 884,
        "resume_validated_checkpoint_cells": 20,
        "start_from_year0_cells": 864,
        "completed_cells_recomputed": 0,
        "model_family": "OGGM one-dimensional SemiImplicitModel only",
        "forcing": "ensemble_rec_ref-1441-1470-repeat",
        "delta_t_eq_values_degC": [round(-1.0 + 0.1 * index, 1) for index in range(21)],
        "maximum_years": 900,
        "forced_full_duration": True,
        "annual_scalar_diagnostics": True,
        "full_state_checkpoint_interval_years": 30,
        "checkpoint_years": EXPECTED_CHECKPOINTS,
        "spinup_stability": {
            "window_years": 90,
            "cycle_years": 30,
            "each_30y_volume_relative_max": 0.03,
            "total_90y_volume_relative_max": 0.03,
            "each_30y_length_change_max_m": 300.0,
            "total_90y_length_change_max_m": 300.0,
        },
        "lia_length_relative_error_max": 0.10,
        "scientific_failure_action": "record and continue; never stop batch",
        "technical_member_failure_action": "isolate member and continue",
        "workers": 20,
        "internal_threads_per_worker": 1,
        "startup_stagger_seconds": 0.5,
        "gpu_branch_action": "preserve and do not control",
        "prohibited": ["2-D/IGM", "manual glacier tuning", "scientific threshold changes", "recompute completed cells"],
    }
    thread_environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(PROJECT_ROOT.anchor))

    atomic_json(EVIDENCE / "rules.json", rules)
    atomic_csv(EVIDENCE / "production_matrix.csv", matrix)
    member_by_fp = {item["fingerprint"]: item for item in members}
    atomic_csv(
        EVIDENCE / "production_manifest.csv",
        [{"member_key": member_by_fp[row["fingerprint"]]["key"], **row} for row in matrix],
    )
    atomic_csv(EVIDENCE / "recovery_queue.csv", queue)
    atomic_csv(EVIDENCE / "recovery_staging_files.csv", staged_files)
    staging_inventory_hash = sha256(EVIDENCE / "recovery_staging_files.csv")
    atomic_json(
        EVIDENCE / "recovery_staging_audit.json",
        {
            "schema": "lia-all191-fullgrid-recovery-staging-v10.270",
            "passed": True,
            "source_output_root": str(SOURCE_OUTPUT.resolve()),
            "destination_output_root": str(OUTPUT_ROOT.resolve()),
            "staged_members": 20,
            "member_keys": sorted(item["member_key"] for item in staged_members),
            "staged_files": len(staged_files),
            "staged_bytes": sum(int(item["bytes"]) for item in staged_files),
            "file_inventory_sha256": staging_inventory_hash,
            "members": staged_members,
        },
    )
    atomic_json(
        EVIDENCE / "coverage_audit.json",
        {
            "schema": "lia-all191-fullgrid-recovery-v10.270-coverage-audit-v1",
            "full_grid_target_cells": 4011,
            "prior_terminal_cells": 3127,
            "recovery_cells": 884,
            "union_cells": 4011,
            "recovery_unique_fingerprints": len(queue_fingerprints),
            "source_completed_recomputed": 0,
            "resume_cells": 20,
            "year0_cells": 864,
            "passed": True,
        },
    )
    input_binding = {
        "source_transaction": str((SOURCE_TRANSACTION / "transaction_manifest.json").resolve()),
        "source_transaction_sha256": sha256(SOURCE_TRANSACTION / "transaction_manifest.json"),
        "source_terminal": str((SOURCE_TERMINAL / "summary.json").resolve()),
        "source_terminal_sha256": sha256(SOURCE_TERMINAL / "summary.json"),
        "source_recovery_queue": str((SOURCE_TERMINAL / "recovery_queue.csv").resolve()),
        "source_recovery_queue_sha256": sha256(SOURCE_TERMINAL / "recovery_queue.csv"),
        "source_matrix": str((SOURCE_TRANSACTION / "production_matrix.csv").resolve()),
        "source_matrix_sha256": sha256(SOURCE_TRANSACTION / "production_matrix.csv"),
        "scientific_contract": str(FACTS.resolve()),
        "scientific_contract_sha256": sha256(FACTS),
        "execution_contract": str(EXECUTION_CONTRACT.resolve()),
        "execution_contract_sha256": sha256(EXECUTION_CONTRACT),
        "current_state_at_transaction": str(CURRENT_STATE.resolve()),
        "current_state_sha256_at_transaction": sha256(CURRENT_STATE),
        "current_state_id_at_transaction": state["state_id"],
        "chain_driver": str(CHAIN_DRIVER.resolve()),
        "chain_driver_sha256": sha256(CHAIN_DRIVER),
        "base_controller": str(BASE_CONTROLLER.resolve()),
        "base_controller_sha256": sha256(BASE_CONTROLLER),
        "recovery_wrapper": str(RECOVERY_WRAPPER.resolve()),
        "recovery_wrapper_sha256": sha256(RECOVERY_WRAPPER),
    }
    atomic_json(EVIDENCE / "input_binding.json", input_binding)
    longest = max(
        str(OUTPUT_ROOT / "m" / item["key"] / "c/checkpoints/.y0900.tmp.99999/checkpoint.json")
        for item in members
    )
    atomic_json(
        EVIDENCE / "path_length_audit.json",
        {
            "effective_windows_path_limit": 260,
            "longest_projected_path": longest,
            "maximum_projected_path_length": len(longest),
            "minimum_margin_characters": 260 - len(longest),
            "passed_required_margin_80": 260 - len(longest) >= 80,
        },
    )
    atomic_json(
        EVIDENCE / "resource_process_stop_contract.json",
        {
            "workers": 20,
            "each_worker_internal_threads": 1,
            "thread_environment": thread_environment,
            "persistent_foreground_exec_session_required": True,
            "detached_start_process_prohibited": True,
            "status_write_failure_action": "retry-and-fallback; never stop workers",
            "member_failure_action": "isolate and continue",
            "scientific_nonconvergence_action": "record and continue",
            "memory_used_fraction_ceiling": 0.80,
            "technical_invalid_rate_rule": {"minimum_completed_cells": 120, "rate_over": 0.15, "action": "pause new dispatch; running workers continue; user decides"},
            "memory_snapshot": {"used_fraction": vm.percent / 100.0, "available_gib": vm.available / 2**30},
            "disk_snapshot": {"free_gib": disk.free / 2**30},
            "duplicate_cpu_processes": 0,
        },
    )
    manifest = {
        "schema": "lia-all191-fullgrid-recovery-v10.270-transaction-v1",
        "transaction_id": "lia-all191-fullgrid-recovery-v10.270",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authorized-not-launched",
        "authorization_basis": "user: 允许恢复剩余884个",
        "population": 191,
        "accepted_count_effect": 0,
        "model_family": "oggm-flowline-1d",
        "inputs": source_manifest["inputs"],
        "runtime": {
            "python_executable": source_manifest["runtime"]["python_executable"],
            "output_root": str(OUTPUT_ROOT.resolve()),
            "unified_short_root": str(OUTPUT_ROOT.resolve()),
            "workers": 20,
            "chain_driver_sha256": sha256(CHAIN_DRIVER),
            "controller_sha256": sha256(BASE_CONTROLLER),
            "recovery_wrapper_sha256": sha256(RECOVERY_WRAPPER),
            "thread_environment": thread_environment,
            "launch_mode": "persistent foreground exec session; detached Start-Process prohibited",
            "startup_stagger_seconds": 0.5,
        },
        "technical_fixture": {
            "same_contract_terminal_checkpoint_passes": 3127,
            "validated_resume_checkpoint_cells": 20,
            "staging_audit": str((EVIDENCE / "recovery_staging_audit.json").resolve()),
            "staging_audit_sha256": sha256(EVIDENCE / "recovery_staging_audit.json"),
        },
        "production": {
            "one_time_only": True,
            "full_grid_target_cells": 4011,
            "prior_terminal_cells": 3127,
            "cell_count": 884,
            "unique_system_count": len({int(row["lia_unique_id"]) for row in matrix}),
            "resume_validated_checkpoint_cells": 20,
            "start_from_year0_cells": 864,
            "fingerprints": [row["fingerprint"] for row in matrix],
            "members": members,
            "maximum_years": 900,
            "full_state_checkpoint_interval_years": 30,
            "forced_full_duration": True,
        },
        "production_authorized": True,
        "launch_authorized": True,
        "simulation_launched": False,
        "independent_gpu_branches_untouched": True,
        "further_automatic_retry_authorized": False,
    }
    atomic_json(EVIDENCE / "transaction_manifest.json", manifest)
    summary = {
        "schema": "lia-all191-fullgrid-recovery-v10.270-transaction-summary-v1",
        "status": "authorized-not-launched",
        "population": 191,
        "prior_terminal_cells": 3127,
        "scheduled_cells": 884,
        "resume_cells": 20,
        "start_from_year0_cells": 864,
        "completed_cells_recomputed": 0,
        "maximum_years": 900,
        "checkpoint_interval_years": 30,
        "workers": 20,
        "output_root": str(OUTPUT_ROOT.resolve()),
        "launch_authorized": True,
        "accepted_count_effect": 0,
    }
    atomic_json(EVIDENCE / "summary.json", summary)
    atomic_json(
        EVIDENCE / "artifact_hashes.json",
        {path.name: sha256(path) for path in sorted(EVIDENCE.iterdir()) if path.is_file()},
    )
    print(json.dumps({"evidence": str(EVIDENCE), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

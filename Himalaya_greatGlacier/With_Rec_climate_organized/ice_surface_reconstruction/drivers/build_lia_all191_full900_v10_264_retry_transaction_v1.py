"""Build the one authorized v10.264 persistent-session retry transaction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = BASE / "lia_191_all191_full900_spinup_v10_262_transaction_v1"
SOURCE_TERMINAL = (
    BASE / "lia_191_all191_full900_spinup_v10_262_terminal_v1/summary.json"
)
EVIDENCE = BASE / "lia_191_all191_full900_spinup_v10_264_retry_transaction_v1"
OUTPUT_ROOT = PROJECT_ROOT / "w264"
CURRENT_STATE = PROJECT_ROOT / "LIA_CURRENT_STATE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    required = (
        SOURCE / "transaction_manifest.json",
        SOURCE / "production_matrix.csv",
        SOURCE / "production_manifest.csv",
        SOURCE / "rules.json",
        SOURCE / "coverage_audit.json",
        SOURCE / "preserved_v10_260_preflight.json",
        SOURCE / "input_binding.json",
        SOURCE_TERMINAL,
        CURRENT_STATE,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if EVIDENCE.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError(f"immutable-root-collision:{EVIDENCE}:{OUTPUT_ROOT}")

    state = json.loads(CURRENT_STATE.read_text(encoding="utf-8"))
    if state["state_id"] != (
        "2026-09-05-v10.263-full900-launcher-failure-decision-required"
    ):
        raise ValueError("v10.264-requires-v10.263-terminal-state")
    if state["current_focus"]["state"] != "decision-required":
        raise ValueError("v10.264-requires-decision-required")
    if int(state["counts"]["active"]) != 0 or int(state["counts"]["queued"]) != 0:
        raise ValueError("v10.264-requires-zero-active-and-queued")
    terminal = json.loads(SOURCE_TERMINAL.read_text(encoding="utf-8"))
    if (
        terminal["status"]
        != "terminal-technical-launcher-lifetime-failure-before-model-state"
        or int(terminal["member_result_files"]) != 0
        or int(terminal["checkpoint_manifests"]) != 0
        or int(terminal["model_year_reached"]) != 0
    ):
        raise ValueError("source-terminal-not-zero-model-time")

    source_manifest = json.loads(
        (SOURCE / "transaction_manifest.json").read_text(encoding="utf-8")
    )
    matrix = read_csv(SOURCE / "production_matrix.csv")
    if (
        len(matrix) != 482
        or len({row["fingerprint"] for row in matrix}) != 482
        or len({int(row["lia_unique_id"]) for row in matrix}) != 162
    ):
        raise ValueError("source-matrix-cardinality-failure")
    groups: dict[int, int] = {}
    for row in matrix:
        system = int(row["lia_unique_id"])
        groups[system] = groups.get(system, 0) + 1
    if sorted(groups.values()).count(1) != 146 or sorted(groups.values()).count(21) != 16:
        raise ValueError("source-146-plus-16x21-partition-failure")

    EVIDENCE.mkdir(parents=True, exist_ok=False)
    for name in (
        "production_matrix.csv",
        "production_manifest.csv",
        "rules.json",
        "coverage_audit.json",
        "preserved_v10_260_preflight.json",
    ):
        shutil.copy2(SOURCE / name, EVIDENCE / name)

    original_fingerprints = source_manifest["production"]["fingerprints"]
    if set(original_fingerprints) != {row["fingerprint"] for row in matrix}:
        raise ValueError("source-manifest-fingerprint-set-failure")
    manifest = deepcopy(source_manifest)
    manifest["schema"] = "lia-all191-full900-spinup-v10.264-retry-transaction-v1"
    manifest["transaction_id"] = "lia-all191-full900-spinup-v10.264-retry"
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "authorized-not-launched-persistent-session-technical-retry"
    manifest["authorization_basis"] = [
        "user: 允许",
        "scope: one technical relaunch after confirmed detached-launcher lifetime failure",
    ]
    manifest["runtime"]["output_root"] = str(OUTPUT_ROOT)
    manifest["runtime"]["unified_short_root"] = str(OUTPUT_ROOT)
    manifest["runtime"]["launch_mode"] = (
        "persistent foreground exec session; detached Start-Process prohibited"
    )
    manifest["technical_retry"] = {
        "retry_of": str(SOURCE / "transaction_manifest.json"),
        "retry_of_sha256": sha256(SOURCE / "transaction_manifest.json"),
        "source_terminal": str(SOURCE_TERMINAL),
        "source_terminal_sha256": sha256(SOURCE_TERMINAL),
        "identified_cause": "detached launcher lifetime coupling",
        "only_change": "new output root and persistent foreground process ownership",
        "scientific_contract_changed": False,
        "matrix_changed": False,
        "fingerprints_changed": False,
        "retry_ordinal": 1,
        "further_automatic_retry_authorized": False,
    }
    manifest["production_authorized"] = True
    manifest["launch_authorized"] = True
    manifest["simulation_launched"] = False
    manifest["independent_gpu_cancelled_and_untouched"] = True
    atomic_json(EVIDENCE / "transaction_manifest.json", manifest)

    source_input = json.loads(
        (SOURCE / "input_binding.json").read_text(encoding="utf-8")
    )
    input_binding = {
        "source_v10_262_input_binding": str(SOURCE / "input_binding.json"),
        "source_v10_262_input_binding_sha256": sha256(SOURCE / "input_binding.json"),
        "source_v10_262_transaction": str(SOURCE / "transaction_manifest.json"),
        "source_v10_262_transaction_sha256": sha256(
            SOURCE / "transaction_manifest.json"
        ),
        "source_v10_262_terminal": str(SOURCE_TERMINAL),
        "source_v10_262_terminal_sha256": sha256(SOURCE_TERMINAL),
        "unchanged_bound_inputs": source_input,
        "production_matrix_sha256": sha256(EVIDENCE / "production_matrix.csv"),
        "rules_sha256": sha256(EVIDENCE / "rules.json"),
    }
    atomic_json(EVIDENCE / "input_binding.json", input_binding)

    longest = max(
        str(
            OUTPUT_ROOT
            / "m"
            / item["key"]
            / "c"
            / "checkpoints"
            / ".y0900.tmp.99999"
            / "checkpoint.json"
        )
        for item in manifest["production"]["members"]
    )
    path_audit = {
        "schema": "lia-all191-full900-v10.264-path-audit-v1",
        "effective_windows_path_limit": 260,
        "maximum_projected_atomic_checkpoint_path_length": len(longest),
        "minimum_projected_margin_characters": 260 - len(longest),
        "longest_projected_path": longest,
        "passed_required_margin_80": 260 - len(longest) >= 80,
    }
    if not path_audit["passed_required_margin_80"]:
        raise ValueError("path-length-margin-failure")
    atomic_json(EVIDENCE / "path_length_audit.json", path_audit)

    resource_contract = {
        "workers": 20,
        "each_worker_internal_threads": 1,
        "thread_environment": manifest["runtime"]["thread_environment"],
        "output_root": str(OUTPUT_ROOT),
        "persistent_foreground_exec_session_required": True,
        "detached_start_process_prohibited": True,
        "full_state_checkpoint_years": list(range(0, 901, 30)),
        "annual_scalar_diagnostics": True,
        "scientific_nonconvergence_action": "record and continue; no batch stop",
        "status_write_failure_action": "retry-and-fallback; never stop workers",
        "member_checkpoint_or_result_failure_action": "isolate member; continue",
        "memory_used_fraction_ceiling": 0.80,
        "memory_ceiling_action": (
            "pause new dispatch; running workers continue; user decides"
        ),
        "technical_invalid_rate_rule": {
            "minimum_completed_cells": 120,
            "rate_over": 0.15,
            "action": "pause new dispatch; running workers continue; user decides",
        },
        "no_automatic_retry": True,
        "prohibitions": [
            "GPU",
            "2-D/IGM",
            "manual glacier tuning",
            "early scientific termination",
            "detached launcher",
        ],
    }
    atomic_json(EVIDENCE / "resource_process_stop_contract.json", resource_contract)
    summary = {
        "schema": "lia-all191-full900-spinup-v10.264-retry-summary-v1",
        "status": "authorized-not-launched",
        "technical_retry_ordinal": 1,
        "scientific_contract_changed": False,
        "matrix_changed": False,
        "fingerprints_changed": False,
        "new_unique_systems": 162,
        "new_cells": 482,
        "targeted_systems_and_cells": [146, 146],
        "full_grid_systems_values_and_cells": [16, 21, 336],
        "maximum_years": 900,
        "checkpoint_interval_years": 30,
        "workers": 20,
        "output_root": str(OUTPUT_ROOT),
        "launch_mode": "persistent-foreground-exec-session",
        "launch_authorized": True,
        "accepted_count_effect": 0,
    }
    atomic_json(EVIDENCE / "summary.json", summary)
    atomic_json(
        EVIDENCE / "artifact_hashes.json",
        {
            path.name: sha256(path)
            for path in sorted(EVIDENCE.iterdir())
            if path.is_file()
        },
    )
    print(json.dumps({"evidence": str(EVIDENCE), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

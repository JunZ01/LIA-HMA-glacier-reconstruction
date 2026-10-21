"""Build the immutable v10.267 completion transaction for the 191 x 21 grid."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
EVIDENCE = BASE / "lia_191_all191_fullgrid_completion_v10_267_transaction_v1"
OUTPUT_ROOT = PROJECT_ROOT / "w267"
PRIOR_TRANSACTION = BASE / "lia_191_all191_full900_spinup_v10_264_retry_transaction_v1"
PRIOR_TERMINAL = BASE / "lia_191_all191_full900_spinup_v10_264_retry_terminal_v1/summary.json"
PRIOR_OUTPUT = PROJECT_ROOT / "w264"
SOURCE_MATRIX = BASE / "lia_191_all191_oggm_flowline_1d_v10_203_transaction_v1/level1_matrix.csv"
TARGETS = BASE / "lia_elevation_band_core191_length_corrected_v1/elevation_band_pilot_results.csv"
SOURCE_INPUT_HASHES = BASE / "lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1/input_hashes.json"
CURVE_STANDARD = BASE / "lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json"
CHAIN_DRIVER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_full900_spinup_v10_262.py"
)
CONTROLLER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_full900_cpu20_controller_v10_267.py"
)
CURRENT_STATE = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
FACTS = PROJECT_ROOT / "LIA_PROJECT_FACTS.md"
GRID = [round(-1.0 + 0.1 * index, 1) for index in range(21)]
EXPECTED_CHECKPOINTS = list(range(0, 901, 30))


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


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty-csv:{path}")
    temp = path.with_suffix(path.suffix + ".tmp")
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


def key(system: int | str, delta: float | str) -> tuple[int, int]:
    return int(system), int(round(float(delta) * 10))


def number(value: str | int | float) -> float:
    return float(value)


def scientific_cell(source: dict, target: float, delta: float) -> dict:
    base = number(source["base_temp_bias_degC"])
    return {
        "lia_unique_id": int(source["lia_unique_id"]),
        "rgi_id": source["rgi_id"],
        "model_family": "oggm-flowline-1d",
        "geometry_hash": source["geometry_hash"],
        "forcing_hash": source["forcing_hash"],
        "delta_t_eq_degC": float(delta),
        "base_temp_bias_degC": base,
        "effective_temp_bias_degC": base + float(delta),
        "melt_f": number(source["melt_f"]),
        "prcp_fac": number(source["prcp_fac"]),
        "glen_a_pa_3_s_1": number(source["glen_a_pa_3_s_1"]),
        "fs": number(source["fs"]),
        "temp_melt_degC": number(source["temp_melt_degC"]),
        "target_lia_length_m": float(target),
        "spinup_forcing": source["spinup_forcing"],
        "block_years": 300,
        "maximum_years": 900,
        "checkpoint_interval_years": 30,
        "forced_full_duration": True,
        "search_stage": "group-1-delta-t-eq-all191-full-grid-completion-full900",
        "search_round": 1,
        "proposal_delta_t_continuous_degC": float(delta),
        "melt_f_multiplier": 1.0,
        "glen_a_multiplier": 1.0,
        "source_routing": "all191-user-mandated-21-grid-completion",
    }


def main() -> int:
    required = (
        PRIOR_TRANSACTION / "transaction_manifest.json",
        PRIOR_TRANSACTION / "production_matrix.csv",
        PRIOR_TERMINAL,
        PRIOR_OUTPUT / "r/summary.json",
        SOURCE_MATRIX,
        TARGETS,
        SOURCE_INPUT_HASHES,
        CURVE_STANDARD,
        CHAIN_DRIVER,
        CONTROLLER,
        CURRENT_STATE,
        FACTS,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if EVIDENCE.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError(f"immutable-root-collision:{EVIDENCE}:{OUTPUT_ROOT}")

    state = json.loads(CURRENT_STATE.read_text(encoding="utf-8"))
    if int(state["counts"]["active"]) != 0 or int(state["counts"]["queued"]) != 0:
        raise ValueError("completion-requires-zero-active-and-queued")
    if state["current_focus"]["system_id"] != "all191-full900-spinup-v10.264-terminal":
        raise ValueError("completion-requires-v10.264-terminal-focus")

    prior_manifest = json.loads(
        (PRIOR_TRANSACTION / "transaction_manifest.json").read_text(encoding="utf-8")
    )
    prior_matrix = read_csv(PRIOR_TRANSACTION / "production_matrix.csv")
    prior_by_fingerprint = {row["fingerprint"]: row for row in prior_matrix}
    prior_by_key = {key(row["lia_unique_id"], row["delta_t_eq_degC"]): row for row in prior_matrix}
    if len(prior_by_fingerprint) != 482 or len(prior_by_key) != 482:
        raise ValueError("prior-matrix-identity-failure")
    prior_member = {
        item["fingerprint"]: item for item in prior_manifest["production"]["members"]
    }
    prior_summary = json.loads((PRIOR_OUTPUT / "r/summary.json").read_text(encoding="utf-8"))
    if prior_summary["status"] != "completed" or int(prior_summary["completed_cells"]) != 482:
        raise ValueError("prior-controller-not-terminal-complete")

    reusable: dict[tuple[int, int], dict] = {}
    reuse_rows: list[dict] = []
    for record in prior_summary["records"]:
        if bool(record["technical_failure"]) or not bool(record["checkpoint_contract_pass"]):
            continue
        fingerprint = record["fingerprint"]
        row = prior_by_fingerprint[fingerprint]
        member = prior_member[fingerprint]
        result_path = PRIOR_OUTPUT / "m" / member["key"] / "c/result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not (
            result.get("terminal") is True
            and result.get("all900_completed") is True
            and result.get("checkpoint_contract_pass") is True
            and result.get("current_year") == 900
            and result.get("checkpoint_years_actual") == EXPECTED_CHECKPOINTS
            and result.get("fingerprint") == fingerprint
        ):
            raise ValueError(f"nonreusable-prior-result:{fingerprint}")
        combo = key(row["lia_unique_id"], row["delta_t_eq_degC"])
        if combo in reusable:
            raise ValueError(f"duplicate-reusable-combination:{combo}")
        reusable[combo] = row
        reuse_rows.append(
            {
                "lia_unique_id": int(row["lia_unique_id"]),
                "rgi_id": row["rgi_id"],
                "delta_t_eq_degC": float(row["delta_t_eq_degC"]),
                "source_fingerprint": fingerprint,
                "source_member_key": member["key"],
                "result_path": str(result_path),
                "result_sha256": sha256(result_path),
                "checkpoint_count": len(EXPECTED_CHECKPOINTS),
                "reuse_reason": "terminal-full900-checkpoint-contract-pass",
            }
        )
    if len(reusable) != 464:
        raise ValueError(f"reusable-cardinality:{len(reusable)}")

    source_rows = read_csv(SOURCE_MATRIX)
    source_by_system: dict[int, dict] = {}
    for row in source_rows:
        source_by_system.setdefault(int(row["lia_unique_id"]), row)
    target_by_system = {
        int(row["lia_unique_id"]): number(row["target_main_flowline_length_m"])
        for row in read_csv(TARGETS)
    }
    if len(source_by_system) != 191 or set(source_by_system) != set(target_by_system):
        raise ValueError("all191-source-target-coverage-failure")

    rules = {
        "schema": "lia-all191-fullgrid-completion-rules-v10.267",
        "authorization_basis": [
            "user correction: the intended matrix is 191 x 21",
            "user authorization: 跑完",
        ],
        "population": 191,
        "delta_t_eq_values_degC": GRID,
        "full_grid_cells": 4011,
        "strictly_reused_terminal_cells": 464,
        "new_or_technical_retry_cells": 3547,
        "full_duration_years": 900,
        "early_scientific_stop": False,
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
        "controller_startup_stagger_seconds": 0.5,
        "startup_stagger_scope": "technical reliability only; no scientific change",
        "no_manual_glacier_tuning": True,
        "model_family": "OGGM one-dimensional SemiImplicitModel only",
        "gpu_or_2d_prohibited": True,
    }
    rules_hash = hashlib.sha256(
        json.dumps(rules, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    contract = {
        "transaction_id": "lia-all191-fullgrid-completion-v10.267",
        "rules_sha256": rules_hash,
        "source_matrix_sha256": sha256(SOURCE_MATRIX),
        "target_table_sha256": sha256(TARGETS),
        "chain_driver_sha256": sha256(CHAIN_DRIVER),
        "controller_sha256": sha256(CONTROLLER),
        "solver": "SemiImplicitModel-fixed_dt-1.25days",
        "forcing": "ensemble_rec_ref-1441-1470-repeat",
        "maximum_years": 900,
        "checkpoint_interval_years": 30,
    }

    missing_rows: list[dict] = []
    all_keys = {(system, int(round(delta * 10))) for system in source_by_system for delta in GRID}
    missing_keys = sorted(all_keys - set(reusable))
    for ordinal, combo in enumerate(missing_keys, 1):
        system, delta_tenth = combo
        delta = delta_tenth / 10.0
        if combo in prior_by_key:
            # Preserve the original fingerprint for the 18 isolated technical retries.
            row = dict(prior_by_key[combo])
            row["candidate_ordinal"] = ordinal
            row["status"] = "authorized-technical-retry-full900"
            row["launch_authorized"] = True
        else:
            scientific = scientific_cell(
                source_by_system[system], target_by_system[system], delta
            )
            fingerprint = hashlib.sha256(
                json.dumps(
                    {"contract": contract, "cell": scientific},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest().upper()
            row = {
                "candidate_ordinal": ordinal,
                **scientific,
                "fingerprint": fingerprint,
                "status": "authorized-fullgrid-completion-full900",
                "launch_authorized": True,
            }
        missing_rows.append(row)

    fingerprints = {row["fingerprint"] for row in missing_rows}
    if len(all_keys) != 4011 or len(missing_rows) != 3547 or len(fingerprints) != 3547:
        raise ValueError("fullgrid-or-missing-cardinality-failure")
    if set(reusable) | {key(r["lia_unique_id"], r["delta_t_eq_degC"]) for r in missing_rows} != all_keys:
        raise ValueError("fullgrid-union-coverage-failure")
    retry_rows = [row for row in missing_rows if row["fingerprint"] in prior_by_fingerprint]
    if len(retry_rows) != 18:
        raise ValueError(f"technical-retry-cardinality:{len(retry_rows)}")
    scheduled_counts: dict[int, int] = {}
    for row in missing_rows:
        system = int(row["lia_unique_id"])
        scheduled_counts[system] = scheduled_counts.get(system, 0) + 1
    if sorted(scheduled_counts.values()).count(20) != 128:
        raise ValueError("twenty-missing-system-count-failure")
    if sorted(scheduled_counts.values()).count(21) != 47:
        raise ValueError("twentyone-missing-system-count-failure")

    members = [
        {
            "key": f"s{index:04d}",
            "lia_unique_id": int(row["lia_unique_id"]),
            "rgi_id": row["rgi_id"],
            "delta_t_eq_degC": float(row["delta_t_eq_degC"]),
            "fingerprint": row["fingerprint"],
        }
        for index, row in enumerate(missing_rows)
    ]
    by_fingerprint = {item["fingerprint"]: item for item in members}

    EVIDENCE.mkdir(parents=True, exist_ok=False)
    atomic_json(EVIDENCE / "rules.json", rules)
    atomic_csv(EVIDENCE / "production_matrix.csv", missing_rows)
    atomic_csv(
        EVIDENCE / "production_manifest.csv",
        [
            {"member_key": by_fingerprint[row["fingerprint"]]["key"], **row}
            for row in missing_rows
        ],
    )
    atomic_csv(EVIDENCE / "reused_terminal_results.csv", sorted(reuse_rows, key=lambda r: key(r["lia_unique_id"], r["delta_t_eq_degC"])))
    atomic_json(
        EVIDENCE / "coverage_audit.json",
        {
            "schema": "lia-all191-fullgrid-completion-v10.267-coverage-audit-v1",
            "target_systems": 191,
            "values_per_system": 21,
            "target_cells": 4011,
            "reused_valid_cells": 464,
            "reused_unique_systems": len({combo[0] for combo in reusable}),
            "scheduled_cells": 3547,
            "scheduled_unique_systems": len(scheduled_counts),
            "scheduled_systems_with_20_cells": 128,
            "scheduled_systems_with_21_cells": 47,
            "prior_technical_retry_cells": 18,
            "union_cells": len(set(reusable) | {key(r["lia_unique_id"], r["delta_t_eq_degC"]) for r in missing_rows}),
            "missing_cells": 0,
            "duplicate_physical_keys": 0,
            "passed": True,
        },
    )

    prior_inputs = prior_manifest["inputs"]
    input_binding = {
        "source_matrix": str(SOURCE_MATRIX),
        "source_matrix_sha256": sha256(SOURCE_MATRIX),
        "target_table": str(TARGETS),
        "target_table_sha256": sha256(TARGETS),
        "source_input_hashes": str(SOURCE_INPUT_HASHES),
        "source_input_hashes_sha256": sha256(SOURCE_INPUT_HASHES),
        "curve_standard": str(CURVE_STANDARD),
        "curve_standard_sha256": sha256(CURVE_STANDARD),
        "prior_transaction": str(PRIOR_TRANSACTION / "transaction_manifest.json"),
        "prior_transaction_sha256": sha256(PRIOR_TRANSACTION / "transaction_manifest.json"),
        "prior_terminal": str(PRIOR_TERMINAL),
        "prior_terminal_sha256": sha256(PRIOR_TERMINAL),
        "prior_controller_summary": str(PRIOR_OUTPUT / "r/summary.json"),
        "prior_controller_summary_sha256": sha256(PRIOR_OUTPUT / "r/summary.json"),
        "scientific_contract": str(FACTS),
        "scientific_contract_sha256": sha256(FACTS),
        "chain_driver": str(CHAIN_DRIVER),
        "chain_driver_sha256": sha256(CHAIN_DRIVER),
        "controller": str(CONTROLLER),
        "controller_sha256": sha256(CONTROLLER),
    }
    atomic_json(EVIDENCE / "input_binding.json", input_binding)

    longest = max(
        str(OUTPUT_ROOT / "m" / item["key"] / "c/checkpoints/.y0900.tmp.99999/checkpoint.json")
        for item in members
    )
    path_audit = {
        "schema": "lia-all191-fullgrid-completion-v10.267-path-audit-v1",
        "effective_windows_path_limit": 260,
        "maximum_projected_atomic_checkpoint_path_length": len(longest),
        "minimum_projected_margin_characters": 260 - len(longest),
        "longest_projected_path": longest,
        "passed_required_margin_80": 260 - len(longest) >= 80,
    }
    if not path_audit["passed_required_margin_80"]:
        raise ValueError("path-length-margin-failure")
    atomic_json(EVIDENCE / "path_length_audit.json", path_audit)

    thread_environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    atomic_json(
        EVIDENCE / "resource_process_stop_contract.json",
        {
            "workers": 20,
            "each_worker_internal_threads": 1,
            "thread_environment": thread_environment,
            "output_root": str(OUTPUT_ROOT),
            "persistent_foreground_exec_session_required": True,
            "detached_start_process_prohibited": True,
            "controller_startup_stagger_seconds": 0.5,
            "full_state_checkpoint_years": EXPECTED_CHECKPOINTS,
            "annual_scalar_diagnostics": True,
            "scientific_nonconvergence_action": "record and continue; no batch stop",
            "status_write_failure_action": "retry-and-fallback; never stop workers",
            "member_checkpoint_or_result_failure_action": "isolate member; continue",
            "memory_used_fraction_ceiling": 0.80,
            "memory_ceiling_action": "pause new dispatch; running workers continue; user decides",
            "technical_invalid_rate_rule": {
                "minimum_completed_cells": 120,
                "rate_over": 0.15,
                "action": "pause new dispatch; running workers continue; user decides",
            },
            "no_automatic_retry": True,
            "projected_output_gib": 17.3,
            "free_disk_gib_at_design": 2554.7,
            "prohibitions": ["GPU", "2-D/IGM", "manual glacier tuning", "early scientific termination", "detached launcher"],
        },
    )

    manifest = {
        "schema": "lia-all191-fullgrid-completion-v10.267-transaction-v1",
        "transaction_id": "lia-all191-fullgrid-completion-v10.267",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authorized-not-launched",
        "authorization_basis": rules["authorization_basis"],
        "population": 191,
        "accepted_count_effect": 0,
        "model_family": "oggm-flowline-1d",
        "inputs": {
            "source_gdir_root": prior_inputs["source_gdir_root"],
            "elevation_band_root": prior_inputs["elevation_band_root"],
            "climate_pixels": prior_inputs["climate_pixels"],
            "reconstruction_climate": prior_inputs["reconstruction_climate"],
            "input_binding": str(EVIDENCE / "input_binding.json"),
        },
        "runtime": {
            "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
            "output_root": str(OUTPUT_ROOT),
            "unified_short_root": str(OUTPUT_ROOT),
            "workers": 20,
            "chain_driver_sha256": sha256(CHAIN_DRIVER),
            "controller_sha256": sha256(CONTROLLER),
            "thread_environment": thread_environment,
            "launch_mode": "persistent foreground exec session; detached Start-Process prohibited",
            "startup_stagger_seconds": 0.5,
        },
        "technical_fixture": {
            "new_nested_preflight": False,
            "same_contract_terminal_checkpoint_passes": 464,
            "source_terminal": str(PRIOR_TERMINAL),
            "source_terminal_sha256": sha256(PRIOR_TERMINAL),
        },
        "production": {
            "one_time_only": True,
            "full_grid_target_cells": 4011,
            "reused_terminal_cells": 464,
            "cell_count": len(missing_rows),
            "unique_system_count": len(scheduled_counts),
            "values_per_system": 21,
            "prior_technical_retry_cells": 18,
            "new_grid_cells": 3529,
            "fingerprints": [row["fingerprint"] for row in missing_rows],
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
        "schema": "lia-all191-fullgrid-completion-v10.267-transaction-summary-v1",
        "status": "authorized-not-launched",
        "population": 191,
        "full_grid_target_cells": 4011,
        "reused_terminal_cells": 464,
        "scheduled_cells": 3547,
        "scheduled_unique_systems": len(scheduled_counts),
        "previous_technical_failures_retried": 18,
        "new_grid_cells": 3529,
        "maximum_years": 900,
        "checkpoint_interval_years": 30,
        "workers": 20,
        "startup_stagger_seconds": 0.5,
        "estimated_runtime_hours": 5.7,
        "output_root": str(OUTPUT_ROOT),
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

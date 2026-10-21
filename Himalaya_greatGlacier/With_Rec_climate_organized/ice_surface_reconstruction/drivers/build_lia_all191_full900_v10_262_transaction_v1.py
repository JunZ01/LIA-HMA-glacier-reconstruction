"""Build the immutable v10.262 CPU20 full-900-year spin-up transaction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
EVIDENCE = BASE / "lia_191_all191_full900_spinup_v10_262_transaction_v1"
OUTPUT_ROOT = PROJECT_ROOT / "w262"
V260 = BASE / "lia_191_all191_adaptive_spinup_v10_260_transaction_v1"
V260_TERMINAL = (
    BASE / "lia_191_all191_adaptive_spinup_v10_260_preflight_terminal_v2/summary.json"
)
SOURCE_MATRIX = (
    BASE / "lia_191_all191_oggm_flowline_1d_v10_203_transaction_v1/level1_matrix.csv"
)
TARGETS = (
    BASE / "lia_elevation_band_core191_length_corrected_v1/elevation_band_pilot_results.csv"
)
SOURCE_INPUT_HASHES = (
    BASE
    / "lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1/input_hashes.json"
)
CURVE_STANDARD = (
    BASE / "lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json"
)
THIRTY_YEAR_FIXTURE = (
    BASE
    / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_transaction_v1/summary.json"
)
CHAIN_DRIVER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_full900_spinup_v10_262.py"
)
CONTROLLER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_full900_cpu20_controller_v10_262.py"
)
CURRENT_STATE = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
SOURCE_GDIR_ROOT = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/"
    "019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/"
    "oggm_workdir/per_glacier"
)
ELEVATION_ROOT = BASE / "lia_elevation_band_core191_length_corrected_v1"
CLIMATE_PIXELS = (
    BASE / "lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"
)
RECONSTRUCTION_CLIMATE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\oggm_climate\Ensemble"
    r"\climate_oggm_ensemble_rec_ref.nc"
)
DEFERRED_GRID = [round(-1.0 + 0.1 * index, 1) for index in range(21)]
PREVIOUS_PREFLIGHT_SYSTEMS = {79, 2692, 2790}


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


def numeric(value: str | float | int) -> float:
    return float(value)


def base_scientific(
    source: dict,
    target_length_m: float,
    delta_t_eq: float,
    search_stage: str,
    source_routing: str,
    proposal: float,
) -> dict:
    base_bias = numeric(source["base_temp_bias_degC"])
    return {
        "lia_unique_id": int(source["lia_unique_id"]),
        "rgi_id": source["rgi_id"],
        "model_family": "oggm-flowline-1d",
        "geometry_hash": source["geometry_hash"],
        "forcing_hash": source["forcing_hash"],
        "delta_t_eq_degC": float(delta_t_eq),
        "base_temp_bias_degC": base_bias,
        "effective_temp_bias_degC": base_bias + float(delta_t_eq),
        "melt_f": numeric(source["melt_f"]),
        "prcp_fac": numeric(source["prcp_fac"]),
        "glen_a_pa_3_s_1": numeric(source["glen_a_pa_3_s_1"]),
        "fs": numeric(source["fs"]),
        "temp_melt_degC": numeric(source["temp_melt_degC"]),
        "target_lia_length_m": float(target_length_m),
        "spinup_forcing": source["spinup_forcing"],
        "block_years": 300,
        "maximum_years": 900,
        "checkpoint_interval_years": 30,
        "forced_full_duration": True,
        "search_stage": search_stage,
        "search_round": 1,
        "proposal_delta_t_continuous_degC": float(proposal),
        "melt_f_multiplier": 1.0,
        "glen_a_multiplier": 1.0,
        "source_routing": source_routing,
    }


def main() -> int:
    required = (
        V260 / "production_manifest.csv",
        V260 / "deferred_after_group1_seed.csv",
        V260 / "transaction_manifest.json",
        V260_TERMINAL,
        SOURCE_MATRIX,
        TARGETS,
        SOURCE_INPUT_HASHES,
        CURVE_STANDARD,
        THIRTY_YEAR_FIXTURE,
        CHAIN_DRIVER,
        CONTROLLER,
        CURRENT_STATE,
        CLIMATE_PIXELS,
        RECONSTRUCTION_CLIMATE,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if EVIDENCE.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError(f"immutable-root-collision:{EVIDENCE}:{OUTPUT_ROOT}")

    current = json.loads(CURRENT_STATE.read_text(encoding="utf-8"))
    if current["state_id"] != "2026-09-04-v10.261-adaptive-preflight-gate-decision-required":
        raise ValueError("v10.262-requires-v10.261-decision-state")
    if current["current_focus"]["state"] != "decision-required":
        raise ValueError("v10.262-requires-decision-required")
    if int(current["counts"]["active"]) != 0 or int(current["counts"]["queued"]) != 0:
        raise ValueError("v10.262-requires-zero-active-and-queued")

    v260_manifest = json.loads(
        (V260 / "transaction_manifest.json").read_text(encoding="utf-8")
    )
    preflight_fingerprints = set(v260_manifest["preflight"]["fingerprints"])
    old_rows = read_csv(V260 / "production_manifest.csv")
    remaining_rows = [
        row for row in old_rows if row["fingerprint"] not in preflight_fingerprints
    ]
    deferred_rows = read_csv(V260 / "deferred_after_group1_seed.csv")
    source_rows = read_csv(SOURCE_MATRIX)
    target_rows = read_csv(TARGETS)
    if len(remaining_rows) != 146 or len(deferred_rows) != 16:
        raise ValueError(
            f"user-authorized-coverage-cardinality:{len(remaining_rows)}:"
            f"{len(deferred_rows)}"
        )
    if {int(row["lia_unique_id"]) for row in old_rows if row["fingerprint"] in preflight_fingerprints} != PREVIOUS_PREFLIGHT_SYSTEMS:
        raise ValueError("preserved-preflight-system-set-mismatch")

    source_by_system: dict[int, dict] = {}
    for row in source_rows:
        source_by_system.setdefault(int(row["lia_unique_id"]), row)
    target_by_system = {
        int(row["lia_unique_id"]): numeric(row["target_main_flowline_length_m"])
        for row in target_rows
    }
    deferred_systems = {int(row["lia_unique_id"]) for row in deferred_rows}
    remaining_systems = {int(row["lia_unique_id"]) for row in remaining_rows}
    if len(deferred_systems) != 16 or len(remaining_systems) != 146:
        raise ValueError("unique-system-cardinality-failure")
    if deferred_systems & remaining_systems:
        raise ValueError("deferred-and-remaining-system-overlap")
    if deferred_systems | remaining_systems | PREVIOUS_PREFLIGHT_SYSTEMS | {
        int(row["lia_unique_id"])
        for row in read_csv(V260 / "reused_group1_passes.csv")
    } != set(source_by_system):
        raise ValueError("all191-partition-failure")

    rules = {
        "schema": "lia-all191-full900-spinup-rules-v10.262",
        "authorization_basis": [
            "user: 先把这146+16个跑完 一条都不能漏",
            "user: 可以 那么20个核同时启动",
            "user: 16条各自把[-1,1]摄氏度的21个温度全部跑完",
        ],
        "new_systems": 162,
        "remaining_targeted_systems": 146,
        "remaining_targeted_cells": 146,
        "full_grid_systems": 16,
        "full_grid_values_degC": DEFERRED_GRID,
        "full_grid_cells": 336,
        "total_new_cells": 482,
        "previous_preflight_cells_preserved_not_rerun": 3,
        "full_duration_years": 900,
        "early_scientific_stop": False,
        "year900_classification_only": True,
        "annual_scalar_diagnostics": True,
        "full_state_checkpoint_interval_years": 30,
        "checkpoint_years": list(range(0, 901, 30)),
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
        "batch_pause_only_after_user_decision_trigger": {
            "memory_used_fraction_over": 0.80,
            "technical_invalid_rate_over": 0.15,
            "technical_invalid_minimum_completed_cells": 120,
        },
        "no_manual_glacier_tuning": True,
        "model_family": "OGGM one-dimensional SemiImplicitModel only",
        "gpu_or_2d_prohibited": True,
    }
    rules_hash = hashlib.sha256(
        json.dumps(rules, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    contract = {
        "transaction_id": "lia-all191-full900-spinup-v10.262",
        "rules_sha256": rules_hash,
        "source_matrix_sha256": sha256(SOURCE_MATRIX),
        "target_table_sha256": sha256(TARGETS),
        "v10.260_manifest_sha256": sha256(V260 / "transaction_manifest.json"),
        "v10.260_terminal_sha256": sha256(V260_TERMINAL),
        "chain_driver_sha256": sha256(CHAIN_DRIVER),
        "controller_sha256": sha256(CONTROLLER),
        "solver": "SemiImplicitModel-fixed_dt-1.25days",
        "forcing": "ensemble_rec_ref-1441-1470-repeat",
        "maximum_years": 900,
        "checkpoint_interval_years": 30,
    }

    matrix: list[dict] = []
    for old in sorted(remaining_rows, key=lambda row: int(row["lia_unique_id"])):
        system = int(old["lia_unique_id"])
        scientific = base_scientific(
            old,
            target_by_system[system],
            numeric(old["delta_t_eq_degC"]),
            "group-1-delta-t-eq-targeted-full900",
            "v10.260-remaining-round1",
            numeric(old["proposal_delta_t_continuous_degC"]),
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {"contract": contract, "cell": scientific},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest().upper()
        matrix.append(
            {
                "candidate_ordinal": len(matrix) + 1,
                **scientific,
                "fingerprint": fingerprint,
                "status": "authorized-full900",
                "launch_authorized": True,
            }
        )

    for system in sorted(deferred_systems):
        source = source_by_system[system]
        for delta in DEFERRED_GRID:
            scientific = base_scientific(
                source,
                target_by_system[system],
                delta,
                "group-1-delta-t-eq-user-mandated-full-grid-full900",
                "deferred16-user-mandated-21-grid",
                delta,
            )
            fingerprint = hashlib.sha256(
                json.dumps(
                    {"contract": contract, "cell": scientific},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest().upper()
            matrix.append(
                {
                    "candidate_ordinal": len(matrix) + 1,
                    **scientific,
                    "fingerprint": fingerprint,
                    "status": "authorized-full900",
                    "launch_authorized": True,
                }
            )

    if len(matrix) != 482 or len({row["fingerprint"] for row in matrix}) != 482:
        raise ValueError("v10.262-matrix-cardinality-or-fingerprint-failure")
    counts_by_system: dict[int, int] = {}
    for row in matrix:
        system = int(row["lia_unique_id"])
        counts_by_system[system] = counts_by_system.get(system, 0) + 1
    if any(counts_by_system[system] != 1 for system in remaining_systems):
        raise ValueError("remaining146-one-cell-each-failure")
    if any(counts_by_system[system] != 21 for system in deferred_systems):
        raise ValueError("deferred16-21-cells-each-failure")

    members = [
        {
            "key": f"s{index:04d}",
            "lia_unique_id": int(row["lia_unique_id"]),
            "rgi_id": row["rgi_id"],
            "delta_t_eq_degC": float(row["delta_t_eq_degC"]),
            "fingerprint": row["fingerprint"],
        }
        for index, row in enumerate(matrix)
    ]
    member_by_fingerprint = {
        member["fingerprint"]: member for member in members
    }

    EVIDENCE.mkdir(parents=True, exist_ok=False)
    atomic_json(EVIDENCE / "rules.json", rules)
    atomic_csv(EVIDENCE / "production_matrix.csv", matrix)
    atomic_csv(
        EVIDENCE / "production_manifest.csv",
        [
            {
                "member_key": member_by_fingerprint[row["fingerprint"]]["key"],
                **row,
            }
            for row in matrix
        ],
    )
    atomic_json(
        EVIDENCE / "coverage_audit.json",
        {
            "schema": "lia-all191-full900-v10.262-coverage-audit-v1",
            "all191_partition": {
                "reused_prior_passes": 26,
                "preserved_v10.260_preflight": 3,
                "new_targeted_systems": 146,
                "new_full_grid_systems": 16,
                "sum": 191,
            },
            "new_batch": {
                "unique_systems": len(counts_by_system),
                "targeted_systems": len(remaining_systems),
                "targeted_cells": 146,
                "full_grid_systems": len(deferred_systems),
                "full_grid_values_per_system": 21,
                "full_grid_cells": 336,
                "total_cells": len(matrix),
            },
            "full_grid_system_ids": sorted(deferred_systems),
            "missing_systems": [],
            "duplicate_fingerprints": 0,
            "passed": True,
        },
    )
    atomic_json(
        EVIDENCE / "preserved_v10_260_preflight.json",
        {
            "source": str(V260_TERMINAL),
            "source_sha256": sha256(V260_TERMINAL),
            "systems": sorted(PREVIOUS_PREFLIGHT_SYSTEMS),
            "completed_cells": 3,
            "technical_failures": 0,
            "checkpoint_contract_passes": 3,
            "rerun": False,
        },
    )

    longest = max(
        str(
            OUTPUT_ROOT
            / "m"
            / member["key"]
            / "c"
            / "checkpoints"
            / ".y0900.tmp.99999"
            / "checkpoint.json"
        )
        for member in members
    )
    path_audit = {
        "schema": "lia-all191-full900-v10.262-path-audit-v1",
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
    input_binding = {
        "source_matrix": str(SOURCE_MATRIX),
        "source_matrix_sha256": sha256(SOURCE_MATRIX),
        "target_table": str(TARGETS),
        "target_table_sha256": sha256(TARGETS),
        "source_input_hashes": str(SOURCE_INPUT_HASHES),
        "source_input_hashes_sha256": sha256(SOURCE_INPUT_HASHES),
        "curve_standard": str(CURVE_STANDARD),
        "curve_standard_sha256": sha256(CURVE_STANDARD),
        "v10.260_transaction_manifest": str(V260 / "transaction_manifest.json"),
        "v10.260_transaction_manifest_sha256": sha256(
            V260 / "transaction_manifest.json"
        ),
        "v10.260_terminal": str(V260_TERMINAL),
        "v10.260_terminal_sha256": sha256(V260_TERMINAL),
        "thirty_year_fixture": str(THIRTY_YEAR_FIXTURE),
        "thirty_year_fixture_sha256": sha256(THIRTY_YEAR_FIXTURE),
        "chain_driver": str(CHAIN_DRIVER),
        "chain_driver_sha256": sha256(CHAIN_DRIVER),
        "controller": str(CONTROLLER),
        "controller_sha256": sha256(CONTROLLER),
    }
    atomic_json(EVIDENCE / "input_binding.json", input_binding)
    atomic_json(
        EVIDENCE / "resource_process_stop_contract.json",
        {
            "workers": 20,
            "each_worker_internal_threads": 1,
            "thread_environment": thread_environment,
            "output_root": str(OUTPUT_ROOT),
            "member_results": "w262/m/s####/c/result.json",
            "full_state_checkpoint_years": list(range(0, 901, 30)),
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
            "prohibitions": [
                "GPU",
                "2-D/IGM",
                "manual glacier tuning",
                "early scientific termination",
                "whole-batch stop from one scientific failure",
            ],
        },
    )

    manifest = {
        "schema": "lia-all191-full900-spinup-v10.262-transaction-v1",
        "transaction_id": "lia-all191-full900-spinup-v10.262",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authorized-not-launched",
        "authorization_basis": rules["authorization_basis"],
        "population": 191,
        "accepted_count_effect": 0,
        "model_family": "oggm-flowline-1d",
        "inputs": {
            "source_gdir_root": str(SOURCE_GDIR_ROOT),
            "elevation_band_root": str(ELEVATION_ROOT),
            "climate_pixels": str(CLIMATE_PIXELS),
            "reconstruction_climate": str(RECONSTRUCTION_CLIMATE),
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
        },
        "technical_fixture": {
            "new_nested_preflight": False,
            "v10.260_same_model_checkpoint_restore_passes": 3,
            "v10.260_terminal": str(V260_TERMINAL),
            "v10.260_terminal_sha256": sha256(V260_TERMINAL),
            "thirty_year_full_state_fixture": str(THIRTY_YEAR_FIXTURE),
            "thirty_year_full_state_fixture_sha256": sha256(THIRTY_YEAR_FIXTURE),
        },
        "production": {
            "one_time_only": True,
            "cell_count": len(matrix),
            "unique_system_count": len(counts_by_system),
            "targeted_system_count": 146,
            "targeted_cell_count": 146,
            "full_grid_system_count": 16,
            "full_grid_values_per_system": 21,
            "full_grid_cell_count": 336,
            "fingerprints": [row["fingerprint"] for row in matrix],
            "members": members,
            "maximum_years": 900,
            "full_state_checkpoint_interval_years": 30,
            "forced_full_duration": True,
        },
        "production_authorized": True,
        "launch_authorized": True,
        "simulation_launched": False,
        "independent_gpu_cancelled_and_untouched": True,
    }
    atomic_json(EVIDENCE / "transaction_manifest.json", manifest)
    summary = {
        "schema": "lia-all191-full900-spinup-v10.262-transaction-summary-v1",
        "status": "authorized-not-launched",
        "population": 191,
        "new_unique_systems": len(counts_by_system),
        "new_cells": len(matrix),
        "targeted_systems_and_cells": [146, 146],
        "full_grid_systems_values_and_cells": [16, 21, 336],
        "preserved_preflight_cells_not_rerun": 3,
        "maximum_years": 900,
        "checkpoint_interval_years": 30,
        "workers": 20,
        "output_root": str(OUTPUT_ROOT),
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
    print(json.dumps({"evidence": str(EVIDENCE), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

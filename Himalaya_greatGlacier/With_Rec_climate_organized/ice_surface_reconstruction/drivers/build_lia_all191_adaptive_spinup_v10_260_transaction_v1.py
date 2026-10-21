"""Build the immutable v10.260 adaptive all-191 CPU20 round-1 transaction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
EVIDENCE = BASE / "lia_191_all191_adaptive_spinup_v10_260_transaction_v1"
OUTPUT_ROOT = PROJECT_ROOT / "w260a"
SOURCE_MATRIX = BASE / "lia_191_all191_oggm_flowline_1d_v10_203_transaction_v1/level1_matrix.csv"
YEAR900 = BASE / "lia_191_year900_parameter_diagnostics_v2/year900_all_cells_with_parameters.csv"
FORMAL_STABLE = BASE / "lia_191_year900_forward_candidate_manifest_v3/forward_candidate_manifest.csv"
TARGETS = BASE / "lia_elevation_band_core191_length_corrected_v1/elevation_band_pilot_results.csv"
SOURCE_INPUT_HASHES = BASE / "lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1/input_hashes.json"
CURVE_STANDARD = BASE / "lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json"
CHAIN_DRIVER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_adaptive_spinup_v10_260.py"
)
CONTROLLER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_adaptive_cpu20_controller_v10_260.py"
)
CURRENT_STATE = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
SOURCE_GDIR_ROOT = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/"
    "019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"
)
ELEVATION_ROOT = BASE / "lia_elevation_band_core191_length_corrected_v1"
CLIMATE_PIXELS = BASE / "lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"
RECONSTRUCTION_CLIMATE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\oggm_climate\Ensemble\climate_oggm_ensemble_rec_ref.nc"
)


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


def relative_error(value: float, target: float) -> float:
    return abs(value - target) / max(abs(target), 1e-300)


def choose_unused_tenth(prediction: float, used: set[float]) -> float:
    pool = [round(-1.0 + 0.1 * index, 1) for index in range(21)]
    pool = [value for value in pool if value not in used]
    if not pool:
        raise ValueError("delta-t-grid-exhausted")
    return min(pool, key=lambda value: (abs(value - prediction), -value))


def main() -> int:
    required = (
        SOURCE_MATRIX, YEAR900, FORMAL_STABLE, TARGETS, SOURCE_INPUT_HASHES,
        CURVE_STANDARD, CHAIN_DRIVER, CONTROLLER, CURRENT_STATE, CLIMATE_PIXELS,
        RECONSTRUCTION_CLIMATE,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if EVIDENCE.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError(f"immutable-root-collision:{EVIDENCE}:{OUTPUT_ROOT}")

    current = json.loads(CURRENT_STATE.read_text(encoding="utf-8"))
    if current["current_focus"]["state"] != "decision-required":
        raise ValueError("v10.260-requires-forward-endpoint-decision-state")
    if int(current["counts"]["active"]) != 0 or int(current["counts"]["queued"]) != 0:
        raise ValueError("v10.260-requires-zero-cpu-mainline-active-and-queued")

    source_rows = read_csv(SOURCE_MATRIX)
    diagnostics = read_csv(YEAR900)
    stable_rows = read_csv(FORMAL_STABLE)
    target_rows = read_csv(TARGETS)
    if len(source_rows) != 1337 or len(diagnostics) != 1337 or len(target_rows) != 191:
        raise ValueError("source-cardinality-failure")

    base_by_system: dict[int, dict] = {}
    all_by_system: dict[int, list[dict]] = {}
    stable_by_system: dict[int, list[dict]] = {}
    target_by_system = {int(row["lia_unique_id"]): row for row in target_rows}
    for row in source_rows:
        base_by_system.setdefault(int(row["lia_unique_id"]), row)
    for row in diagnostics:
        all_by_system.setdefault(int(row["lia_unique_id"]), []).append(row)
    for row in stable_rows:
        stable_by_system.setdefault(int(row["lia_unique_id"]), []).append(row)
    systems = sorted(base_by_system)
    if systems != sorted(target_by_system) or len(systems) != 191:
        raise ValueError("population-identity-failure")

    rules = {
        "schema": "lia-all191-adaptive-four-group-rules-v1",
        "population": 191,
        "no_manual_glacier_tuning": True,
        "failure_only_after_all_four_groups": True,
        "beam_width": 2,
        "group_1_spinup_delta_t_eq": {
            "role": "spinup-only; never carried into forward forcing",
            "allowed_grid_degC": [round(-1.0 + 0.1 * i, 1) for i in range(21)],
            "existing_v10_203_v10_216_results_reused_without_rerun": True,
            "proposal": "piecewise-linear target crossing followed by nearest unused 0.1 degC; warmer value wins an exact tie",
            "maximum_new_real_oggm_runs_per_system": 5,
            "pass": "confirmed numerical stability by year 900 and LIA total-length relative error <= 10%",
        },
        "group_2_forward_temperature": {
            "role": "forward-only endpoint correction; zero at 1470 and linear to delta_t_forward_end at 2000",
            "range_degC": [-0.9, 0.9],
            "proposal": "two-sided response probes then safeguarded secant/bisection",
            "maximum_real_oggm_runs_per_retained_state": 5,
            "disclosure": "new calibrated reconstruction control; not the prior fixed-climate validation route",
        },
        "group_3_melt_f": {
            "multipliers": [0.7, 1.0, 1.3],
            "optional_single_midpoint_insertion": True,
            "maximum_values_per_parent": 5,
            "physics_change_requires_group_1_re-equilibration": True,
        },
        "group_4_glen_a": {
            "multipliers": [0.5, 1.0, 2.0],
            "optional_single_log_midpoint_insertion": True,
            "maximum_values_per_parent": 5,
            "physics_change_requires_group_1_re-equilibration": True,
        },
        "spinup_stability": {
            "check_boundaries_years": [300, 600, 900],
            "same_phase_cycle_years": 30,
            "rolling_window_years": 90,
            "each_30y_volume_relative_max": 0.03,
            "total_90y_volume_relative_max": 0.03,
            "each_30y_length_change_max_m": 300.0,
            "total_90y_length_change_max_m": 300.0,
            "one_additional_30y_confirmation": True,
            "maximum_years": 900,
        },
        "forward_screen": {
            "lia_peak_total_length_error_max": 0.10,
            "rgi_2000_total_length_error_max": 0.10,
            "separation_normalized_20pct_gate": "diagnostic-only",
            "direction_sign_required_when_target_separation_at_least_m": 300.0,
            "modeled_length_amplitude_min_fraction_of_target_separation": 0.50,
            "retain_v3_lag_multiphase_terminal_drop_and_high_frequency_checks": True,
            "formal_acceptance_requires_independent_reproduction": True,
        },
    }
    rules_hash = hashlib.sha256(
        json.dumps(rules, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    contract = {
        "transaction_id": "lia-all191-adaptive-spinup-v10.260-round1",
        "model_family": "oggm-flowline-1d",
        "solver": "SemiImplicitModel-fixed_dt-1.25days",
        "rules_sha256": rules_hash,
        "source_matrix_sha256": sha256(SOURCE_MATRIX),
        "year900_diagnostics_sha256": sha256(YEAR900),
        "formal_stable_manifest_sha256": sha256(FORMAL_STABLE),
        "target_table_sha256": sha256(TARGETS),
        "chain_driver_sha256": sha256(CHAIN_DRIVER),
        "controller_sha256": sha256(CONTROLLER),
    }

    reused: list[dict] = []
    deferred: list[dict] = []
    matrix: list[dict] = []
    for system in systems:
        target = float(target_by_system[system]["target_main_flowline_length_m"])
        formal = stable_by_system.get(system, [])
        best = None
        if formal:
            best = min(
                formal,
                key=lambda row: abs(float(row["year900_diagnostic_length_ratio_to_year0"]) - 1.0),
            )
        if best is not None and abs(float(best["year900_diagnostic_length_ratio_to_year0"]) - 1.0) <= 0.10:
            reused.append({
                "lia_unique_id": system,
                "rgi_id": best["rgi_id"],
                "delta_t_eq_degC": float(best["delta_t_eq_degC"]),
                "lia_length_relative_error_at_year900": abs(float(best["year900_diagnostic_length_ratio_to_year0"]) - 1.0),
                "result_path": best["result_path"],
                "result_sha256": best["result_sha256"],
                "checkpoint_path": best["confirmation_checkpoint_state_path"],
                "checkpoint_sha256": best["confirmation_checkpoint_state_sha256"],
                "routing": "group-1-pass-reused-no-rerun",
            })
            continue

        observed = sorted(all_by_system[system], key=lambda row: float(row["delta_t_eq_degC"]))
        bracket = None
        for left, right in zip(observed, observed[1:]):
            left_ratio = float(left["length_ratio_to_year0"])
            right_ratio = float(right["length_ratio_to_year0"])
            if (left_ratio - 1.0) * (right_ratio - 1.0) <= 0:
                bracket = (left, right)
                break
        if bracket is None:
            deferred.append({
                "lia_unique_id": system, "rgi_id": base_by_system[system]["rgi_id"],
                "reason": "no-year900-length-target-bracket-in-existing-seven-points",
                "next_group": "group-3-melt-f-because-forward-cannot-repair-spinup",
            })
            continue
        left, right = bracket
        d0, d1 = float(left["delta_t_eq_degC"]), float(right["delta_t_eq_degC"])
        r0, r1 = float(left["length_ratio_to_year0"]), float(right["length_ratio_to_year0"])
        prediction = (d0 + d1) / 2.0 if abs(r1 - r0) <= 1e-15 else d0 + (1.0 - r0) * (d1 - d0) / (r1 - r0)
        if prediction < -1.0 or prediction > 1.0:
            deferred.append({
                "lia_unique_id": system, "rgi_id": base_by_system[system]["rgi_id"],
                "reason": "predicted-target-crossing-outside-user-approved-minus1-plus1-range",
                "continuous_prediction_degC": prediction,
                "next_group": "group-3-melt-f-because-forward-cannot-repair-spinup",
            })
            continue
        used = {round(float(row["delta_t_eq_degC"]), 1) for row in observed}
        candidate = choose_unused_tenth(prediction, used)
        source = base_by_system[system]
        scientific = {
            "lia_unique_id": system,
            "rgi_id": source["rgi_id"],
            "model_family": "oggm-flowline-1d",
            "geometry_hash": source["geometry_hash"],
            "forcing_hash": source["forcing_hash"],
            "delta_t_eq_degC": candidate,
            "base_temp_bias_degC": float(source["base_temp_bias_degC"]),
            "effective_temp_bias_degC": float(source["base_temp_bias_degC"]) + candidate,
            "melt_f": float(source["melt_f"]),
            "prcp_fac": float(source["prcp_fac"]),
            "glen_a_pa_3_s_1": float(source["glen_a_pa_3_s_1"]),
            "fs": float(source["fs"]),
            "temp_melt_degC": float(source["temp_melt_degC"]),
            "target_lia_length_m": target,
            "spinup_forcing": source["spinup_forcing"],
            "block_years": 300,
            "maximum_years": 900,
            "search_stage": "group-1-delta-t-eq",
            "search_round": 1,
            "proposal_delta_t_continuous_degC": prediction,
            "proposal_bracket_cold_delta_degC": d0,
            "proposal_bracket_warm_delta_degC": d1,
            "proposal_bracket_cold_length_ratio": r0,
            "proposal_bracket_warm_length_ratio": r1,
            "melt_f_multiplier": 1.0,
            "glen_a_multiplier": 1.0,
        }
        fingerprint = hashlib.sha256(
            json.dumps({"contract": contract, "cell": scientific}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest().upper()
        matrix.append({
            "candidate_ordinal": len(matrix) + 1,
            **scientific,
            "fingerprint": fingerprint,
            "status": "authorized-round1",
            "launch_authorized": True,
        })

    if len(reused) != 26 or len(matrix) + len(reused) + len(deferred) != 191:
        raise ValueError(f"adaptive-routing-cardinality:{len(reused)}:{len(matrix)}:{len(deferred)}")
    if len({row["fingerprint"] for row in matrix}) != len(matrix):
        raise ValueError("adaptive-fingerprint-collision")
    target_meta = {int(row["lia_unique_id"]): row for row in target_rows}
    candidates_by_nodes = sorted(matrix, key=lambda row: int(target_meta[int(row["lia_unique_id"])]["ice_node_count"]))
    known = next((row for row in matrix if int(row["lia_unique_id"]) == 79), None)
    selected = [candidates_by_nodes[0], known, candidates_by_nodes[-1]]
    if any(row is None for row in selected):
        raise ValueError("required-preflight-system-79-not-in-round1")
    preflight_fingerprints = [row["fingerprint"] for row in selected]
    if len(set(preflight_fingerprints)) != 3:
        raise ValueError("preflight-selection-collision")

    members = [
        {"key": f"a{index:04d}", "lia_unique_id": int(row["lia_unique_id"]), "rgi_id": row["rgi_id"], "fingerprint": row["fingerprint"]}
        for index, row in enumerate(matrix)
    ]
    member_by_fp = {row["fingerprint"]: member for row, member in zip(matrix, members)}
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    atomic_json(EVIDENCE / "adaptive_rules.json", rules)
    atomic_csv(EVIDENCE / "adaptive_round1_matrix.csv", matrix)
    atomic_csv(EVIDENCE / "reused_group1_passes.csv", reused)
    atomic_csv(EVIDENCE / "deferred_after_group1_seed.csv", deferred)
    atomic_csv(EVIDENCE / "production_manifest.csv", [{"member_key": member_by_fp[row["fingerprint"]]["key"], **row} for row in matrix])

    longest = max(
        str(OUTPUT_ROOT / "m" / member["key"] / "w" / "per_glacier" / row["rgi_id"][:8] / row["rgi_id"][:11] / row["rgi_id"] / "inversion_output_f19_fs0_full_v0.pkl")
        for row, member in zip(matrix, members)
    )
    path_audit = {
        "schema": "lia-all191-adaptive-v10.260-path-audit-v1",
        "effective_windows_path_limit": 260,
        "maximum_projected_path_length": len(longest),
        "minimum_projected_margin_characters": 260 - len(longest),
        "longest_projected_path": longest,
        "passed_required_margin_80": 260 - len(longest) >= 80,
    }
    if not path_audit["passed_required_margin_80"]:
        raise ValueError("path-length-margin-failure")
    atomic_json(EVIDENCE / "path_length_audit.json", path_audit)

    thread_environment = {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
    }
    input_binding = {
        "source_matrix": str(SOURCE_MATRIX), "source_matrix_sha256": sha256(SOURCE_MATRIX),
        "year900_diagnostics": str(YEAR900), "year900_diagnostics_sha256": sha256(YEAR900),
        "formal_stable_manifest": str(FORMAL_STABLE), "formal_stable_manifest_sha256": sha256(FORMAL_STABLE),
        "target_table": str(TARGETS), "target_table_sha256": sha256(TARGETS),
        "source_input_hashes": str(SOURCE_INPUT_HASHES), "source_input_hashes_sha256": sha256(SOURCE_INPUT_HASHES),
        "curve_standard": str(CURVE_STANDARD), "curve_standard_sha256": sha256(CURVE_STANDARD),
        "chain_driver": str(CHAIN_DRIVER), "chain_driver_sha256": sha256(CHAIN_DRIVER),
        "controller": str(CONTROLLER), "controller_sha256": sha256(CONTROLLER),
    }
    atomic_json(EVIDENCE / "input_binding.json", input_binding)
    resource_contract = {
        "workers": 20, "each_worker_internal_threads": 1,
        "thread_environment": thread_environment,
        "output_root": str(OUTPUT_ROOT), "member_results": "w260a/m/a####/c/result.json",
        "atomic_full_state_checkpoint_years": [300, 600, 900],
        "confirmation_checkpoint_possible_years": [330, 630],
        "status_write_failure_action": "retry-and-fallback; never stop workers",
        "member_checkpoint_or_result_failure_action": "isolate member; continue batch",
        "memory_used_fraction_ceiling": 0.80,
        "memory_ceiling_action": "pause new dispatch and request user decision; running members continue",
        "technical_invalid_rate_over_15pct_after_120": "advisory and user decision; no automatic batch termination",
        "no_automatic_retry": True,
        "prohibitions": ["GPU control", "2-D/IGM", "manual glacier tuning", "duplicate existing delta-t run"],
    }
    atomic_json(EVIDENCE / "resource_process_stop_contract.json", resource_contract)

    manifest = {
        "schema": "lia-all191-adaptive-spinup-v10.260-transaction-v1",
        "transaction_id": "lia-all191-adaptive-spinup-v10.260-round1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authorized-round1-preflight-then-production",
        "authorization_basis": "user: 可以 那么建立规则并开始",
        "population": 191,
        "accepted_count_effect": 0,
        "model_family": "oggm-flowline-1d",
        "adaptive_rules": {"path": str(EVIDENCE / "adaptive_rules.json"), "sha256": sha256(EVIDENCE / "adaptive_rules.json")},
        "inputs": {
            "source_gdir_root": str(SOURCE_GDIR_ROOT),
            "elevation_band_root": str(ELEVATION_ROOT),
            "climate_pixels": str(CLIMATE_PIXELS),
            "reconstruction_climate": str(RECONSTRUCTION_CLIMATE),
            "input_binding": str(EVIDENCE / "input_binding.json"),
        },
        "runtime": {
            "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
            "output_root": str(OUTPUT_ROOT), "unified_short_root": str(OUTPUT_ROOT),
            "workers": 20, "chain_driver_sha256": sha256(CHAIN_DRIVER),
            "controller_sha256": sha256(CONTROLLER), "thread_environment": thread_environment,
        },
        "preflight": {
            "cell_count": 3, "fingerprints": preflight_fingerprints,
            "members": [member_by_fp[value] for value in preflight_fingerprints],
            "pass": "all three terminal with validated restore and no technical failure; at least one spinup target pass",
            "automatic_production_after_pass": True,
        },
        "production": {
            "one_time_only": True, "cell_count": len(matrix),
            "fingerprints": [row["fingerprint"] for row in matrix], "members": members,
            "reused_group1_passes": len(reused), "deferred_without_round1_candidate": len(deferred),
            "memory_used_fraction_ceiling": 0.80,
        },
        "production_authorized": True,
        "launch_authorized": True,
        "simulation_launched": False,
        "independent_gpu_v10_248_preserved": True,
    }
    atomic_json(EVIDENCE / "transaction_manifest.json", manifest)
    summary = {
        "schema": "lia-all191-adaptive-spinup-v10.260-transaction-summary-v1",
        "status": "authorized-not-launched",
        "population": 191, "reused_group1_passes": len(reused),
        "round1_new_cells": len(matrix), "deferred_to_later_physics_group": len(deferred),
        "preflight_systems": [int(row["lia_unique_id"]) for row in selected],
        "output_root": str(OUTPUT_ROOT), "workers": 20,
        "estimated_round1_wall_minutes_after_preflight": [20, 60],
        "launch_authorized": True, "accepted_count_effect": 0,
    }
    atomic_json(EVIDENCE / "summary.json", summary)
    atomic_json(EVIDENCE / "artifact_hashes.json", {
        path.name: sha256(path) for path in sorted(EVIDENCE.iterdir()) if path.is_file()
    })
    print(json.dumps({"evidence": str(EVIDENCE), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

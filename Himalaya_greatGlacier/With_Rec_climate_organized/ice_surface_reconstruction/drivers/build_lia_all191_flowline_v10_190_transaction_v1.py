"""Create the immutable, launchable staged v10.190 CPU12 transaction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVIDENCE_REL = Path(
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_190_transaction_v1"
)
OUTPUT_REL = Path(
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_190"
)
SOURCE_MATRIX = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1/level1_matrix.csv"
)
SOURCE_INPUT_HASHES = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1/input_hashes.json"
)
CHAIN_DRIVER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_flowline_spinup_v10_190.py"
)
CONTROLLER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_flowline_cpu12_controller_v10_190.py"
)
PREFLIGHT_SYSTEMS = (214, 252, 313, 7, 84, 130, 512, 705, 709, 14, 79, 334)


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
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    evidence = PROJECT_ROOT / EVIDENCE_REL
    output_root = PROJECT_ROOT / OUTPUT_REL
    if evidence.exists() or output_root.exists():
        raise FileExistsError(f"transaction-or-output-collision:{evidence}:{output_root}")
    for path in (SOURCE_MATRIX, SOURCE_INPUT_HASHES, CHAIN_DRIVER, CONTROLLER):
        if not path.is_file():
            raise FileNotFoundError(path)

    chain_hash = sha256(CHAIN_DRIVER)
    controller_hash = sha256(CONTROLLER)
    source_hashes_hash = sha256(SOURCE_INPUT_HASHES)
    curve_standard = PROJECT_ROOT / (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json"
    )
    threshold_bundle = {
        "checkpoint_boundary_years": 300,
        "maximum_spinup_years": 3000,
        "same_phase_cycle_years": 30,
        "rolling_window_years": 90,
        "volume_relative_each_cycle_max": 0.003,
        "volume_relative_90y_drift_max": 0.009,
        "length_same_phase_max_model_grid_cells": 1.0,
        "first_pass_confirmation": (
            "append exactly one 30-year climate cycle; re-evaluate rolling final 90 years; "
            "confirm on second pass, otherwise continue to next 300-year boundary"
        ),
        "second_full_300y_checkpoint_required": False,
        "checkpoint_file_integrity": "sha256-exact",
        "checkpoint_member_parameter_topology_match": "exact",
        "restored_float_state_rtol": 1e-7,
        "resume_30y_volume_area_relative_max": 1e-6,
        "resume_30y_thickness_max_absolute_difference_m": 1e-5,
        "resume_30y_length_max_model_grid_cells": 1.0,
        "downstream_boundary_margin_min_m": 5000.0,
        "technical_invalid_rate_min_completed_cells": 120,
        "technical_invalid_rate_stop_above": 0.15,
        "scientific_nonconvergence_counts_as_technical_invalid": False,
    }
    contract_binding = {
        "transaction_id": "lia-all191-oggm-flowline-1d-v10.190",
        "model_family": "oggm-flowline-1d",
        "oggm_version": "1.6.3.dev3+g865475d",
        "solver": "SemiImplicitModel-fixed_dt-1.25days",
        "threshold_bundle": threshold_bundle,
        "input_hashes_sha256": source_hashes_hash,
        "curve_gate_sha256": sha256(curve_standard),
        "chain_driver_sha256": chain_hash,
        "controller_sha256": controller_hash,
    }

    with SOURCE_MATRIX.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 1337:
        raise ValueError(f"source-matrix-cardinality:{len(source_rows)}")
    matrix: list[dict] = []
    for source in source_rows:
        scientific = {
            key: source[key]
            for key in (
                "lia_unique_id", "rgi_id", "model_family", "geometry_hash", "forcing_hash",
                "delta_t_eq_degC", "base_temp_bias_degC", "melt_f", "prcp_fac",
                "glen_a_pa_3_s_1", "fs", "temp_melt_degC", "spinup_forcing",
                "block_years", "maximum_years", "effective_temp_bias_degC",
            )
        }
        payload = {"contract": contract_binding, "cell": scientific}
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest().upper()
        matrix.append({
            "cell_ordinal": len(matrix) + 1, **scientific, "fingerprint": fingerprint,
            "status": "authorized-staged-preflight-then-production", "launch_authorized": True,
        })
    if len({row["fingerprint"] for row in matrix}) != 1337:
        raise ValueError("duplicate-v10.190-fingerprint")

    preflight_rows = [
        row for system_id in PREFLIGHT_SYSTEMS
        for row in matrix
        if int(row["lia_unique_id"]) == system_id and float(row["delta_t_eq_degC"]) == 0.0
    ]
    if len(preflight_rows) != 12:
        raise ValueError(f"preflight-cardinality:{len(preflight_rows)}")

    evidence.mkdir(parents=True, exist_ok=False)
    atomic_csv(evidence / "level1_matrix.csv", matrix)
    atomic_csv(evidence / "preflight_manifest.csv", preflight_rows)
    input_binding = {
        "schema": "lia-all191-flowline-v10.190-input-binding-v1",
        "v10_189_input_hashes": str(SOURCE_INPUT_HASHES),
        "v10_189_input_hashes_sha256": source_hashes_hash,
        "v10_189_level1_matrix_sha256": sha256(SOURCE_MATRIX),
        "curve_standard": str(curve_standard),
        "curve_standard_sha256": sha256(curve_standard),
        "chain_driver": str(CHAIN_DRIVER), "chain_driver_sha256": chain_hash,
        "controller": str(CONTROLLER), "controller_sha256": controller_hash,
    }
    atomic_json(evidence / "input_binding.json", input_binding)
    resource_contract = {
        "schema": "lia-all191-flowline-v10.190-cpu12-contract-v1",
        "worker_count": 12,
        "thread_environment": {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        },
        "independent_output_roots": "chains/<fingerprint>",
        "atomic_checkpoints": "chains/<fingerprint>/checkpoints/year-<year>",
        "checkpoint_cadence_years": 300,
        "single_worker_failure_isolation": True,
        "no_automatic_retry": True,
        "memory_used_fraction_ceiling": 0.80,
        "pid_root": str(OUTPUT_REL / "runtime/<mode>"),
        "log_root": str(OUTPUT_REL / "logs/<mode>"),
        "prohibitions": ["planform-2d", "GPU", "per-glacier manual tuning", "joint higher-level expansion"],
    }
    atomic_json(evidence / "resource_process_stop_contract.json", resource_contract)
    manifest = {
        "schema": "lia-all191-oggm-flowline-1d-v10.190-transaction-v1",
        "transaction_id": "lia-all191-oggm-flowline-1d-v10.190",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authorized-staged-preflight-then-production",
        "population": 191, "level1_cells": 1337, "legacy_pass_inherited": 0,
        "accepted_count_effect": 0, "model_family": "oggm-flowline-1d",
        "threshold_bundle": threshold_bundle,
        "spinup_forward_matrices_separate": True,
        "equilibrium_length_role": "output-state-coordinate-only",
        "forward_curve_gate": {"path": str(curve_standard), "sha256": sha256(curve_standard)},
        "inputs": {
            "source_gdir_root": str(PROJECT_ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"),
            "elevation_band_root": str(PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_elevation_band_core191_length_corrected_v1"),
            "climate_pixels": str(PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"),
            "reconstruction_climate": r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\oggm_climate\Ensemble\climate_oggm_ensemble_rec_ref.nc",
            "input_binding": str(evidence / "input_binding.json"),
        },
        "runtime": {
            "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
            "output_root": str(output_root),
            "workers": 12, "chain_driver_sha256": chain_hash,
            "controller_sha256": controller_hash,
            "thread_environment": resource_contract["thread_environment"],
        },
        "preflight": {
            "one_time_only": True, "mode": "first-300y-boundary-plus-30y-resume-reproduction",
            "systems": list(PREFLIGHT_SYSTEMS),
            "fingerprints": [row["fingerprint"] for row in preflight_rows],
            "requires_all_12_technical_pass": True, "memory_used_fraction_ceiling": 0.80,
        },
        "production_release_condition": "preflight summary exists and preflight_passed=true",
        "invalid_rate_stop": {"minimum_completed_cells": 120, "stop_above": 0.15},
        "launch_authorized": True,
        "simulation_launched": False,
    }
    atomic_json(evidence / "transaction_manifest.json", manifest)
    summary = {
        "schema": "lia-all191-flowline-v10.190-transaction-summary-v1",
        "status": "authorized-preflight-not-yet-launched",
        "population": 191, "level1_cells": 1337, "preflight_cells": 12,
        "threshold_bundle_complete": True, "input_binding_complete": True,
        "output_root": str(output_root), "output_root_created": False,
        "launch_authorized": True, "preflight_launched": False,
        "production_launched": False, "accepted_count_effect": 0,
    }
    atomic_json(evidence / "summary.json", summary)
    artifact_hashes = {
        path.name: sha256(path) for path in sorted(evidence.iterdir()) if path.is_file()
    }
    atomic_json(evidence / "artifact_hashes.json", artifact_hashes)
    print(json.dumps({"evidence": str(evidence), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

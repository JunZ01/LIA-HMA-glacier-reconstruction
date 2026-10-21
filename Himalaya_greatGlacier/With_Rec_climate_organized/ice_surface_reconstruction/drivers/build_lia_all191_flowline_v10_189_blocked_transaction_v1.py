"""Build the immutable, non-launchable v10.189 all-191 transaction evidence.

This utility enumerates the Level-1 matrix and audits immutable inputs.  It
deliberately leaves every unresolved numerical threshold as null and records
one blocking item.  It never creates the reserved simulation output root and
never starts OGGM.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVIDENCE_REL = Path(
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1"
)
RESERVED_OUTPUT_REL = Path(
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_189"
)

INPUTS = {
    "population_selection": Path(
        "validation_data/results/paleo_surface_validation_v0/"
        "core191_production_candidate_v1/core191_selection_v1.csv"
    ),
    "length_references": Path(
        "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/"
        "core191_length_references.csv"
    ),
    "climate_pixels": Path(
        "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/"
        "core191_climate_pixels_ensemble_rec_ref.csv"
    ),
    "master_status": Path(
        "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/"
        "lia_191_master_status.csv"
    ),
    "stage2_members": Path(
        "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/"
        "019f730c-ee32-74d0-8b77-fdb7832f6a06/stage2_freeze_stage3_routing_v0/"
        "stage2_member_ensemble.csv"
    ),
    "geometry_audit": Path(
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_core191_length_corrected_v1/audit/"
        "core191_elevation_band_audit.csv"
    ),
    "controller_config": Path(
        "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/config/lia_191_controller_v1.json"
    ),
    "forward_curve_standard_v3": Path(
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_retained_32_curve_analysis_v3_essential_standard/"
        "executable_standard_v3.json"
    ),
}

RECONSTRUCTION_CLIMATE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\oggm_climate\Ensemble\climate_oggm_ensemble_rec_ref.nc"
)
SOURCE_GDIR_ROOT = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/"
    "019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/"
    "oggm_workdir/per_glacier"
)
FLOWLINE_ROOT = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_elevation_band_core191_length_corrected_v1"
)
DELTA_T_EQ_VALUES = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def keyed(rows: list[dict[str, str]], key: str) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        item = int(row[key])
        if item in out:
            raise ValueError(f"duplicate {key}={item}")
        out[item] = row
    return out


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def gdir_for(rgi_id: str) -> Path:
    return SOURCE_GDIR_ROOT / rgi_id[:8] / rgi_id[:11] / rgi_id


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def main() -> int:
    evidence_dir = PROJECT_ROOT / EVIDENCE_REL
    reserved_output = PROJECT_ROOT / RESERVED_OUTPUT_REL
    if evidence_dir.exists():
        raise FileExistsError(f"immutable evidence already exists: {evidence_dir}")
    if reserved_output.exists():
        raise FileExistsError(f"reserved simulation output collides: {reserved_output}")

    absolute_inputs = {name: PROJECT_ROOT / rel for name, rel in INPUTS.items()}
    absolute_inputs["reconstruction_climate"] = RECONSTRUCTION_CLIMATE
    missing_global = [name for name, path in absolute_inputs.items() if not path.is_file()]
    if missing_global:
        raise FileNotFoundError(f"missing global inputs: {missing_global}")

    selection = read_csv(absolute_inputs["population_selection"])
    references = keyed(read_csv(absolute_inputs["length_references"]), "lia_unique_id")
    pixels = keyed(read_csv(absolute_inputs["climate_pixels"]), "lia_unique_id")
    master = keyed(read_csv(absolute_inputs["master_status"]), "lia_unique_id")
    geometry = keyed(read_csv(absolute_inputs["geometry_audit"]), "lia_unique_id")
    if len(selection) != 191 or len({int(r["lia_unique_id"]) for r in selection}) != 191:
        raise ValueError("population selection is not exactly 191 unique systems")

    stage2_vectors: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for row in read_csv(absolute_inputs["stage2_members"]):
        if row["ensemble_role"] == "mid-turnover":
            stage2_vectors[int(row["lia_unique_id"])].add(
                (
                    row["rgi_id"], row["melt_f"], row["prcp_fac"],
                    row["glen_a_pa_3_s_1"], row["temp_bias"], row["fs"],
                )
            )

    climate_hash = sha256(RECONSTRUCTION_CLIMATE)
    member_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    per_member_hashes: dict[str, dict[str, str]] = {}
    incomplete: list[int] = []

    for ordinal, selected in enumerate(selection, start=1):
        system_id = int(selected["lia_unique_id"])
        rgi_id = selected["main_rgi_id"]
        ref = references.get(system_id)
        pixel = pixels.get(system_id)
        status = master.get(system_id)
        geo = geometry.get(system_id)
        mid_vectors = {
            vector for vector in stage2_vectors.get(system_id, set())
            if vector[0] == rgi_id
        }
        gdir = gdir_for(rgi_id)
        member_files = {
            "source_mb_calib": gdir / "mb_calib.json",
            "source_inversion_flowlines": gdir / "inversion_flowlines.pkl",
            "source_inversion_input": gdir / "inversion_input.pkl",
            "source_inversion_output": gdir / "inversion_output_f19_fs0_full_v0.pkl",
            "source_climate_historical": gdir / "climate_historical.nc",
            "corrected_flowline": FLOWLINE_ROOT / str(system_id) / "elevation_band_flowline_lia.pkl.gz",
            "corrected_flowline_audit": FLOWLINE_ROOT / str(system_id) / "elevation_band_audit.json",
        }
        files_exist = all(path.is_file() for path in member_files.values())
        ids_match = bool(
            ref and pixel and status and geo
            and ref["rgi_id"] == rgi_id
            and pixel["rgi_id"] == rgi_id
            and status["rgi_id"] == rgi_id
            and geo["rgi_id"] == rgi_id
        )
        params_finite = bool(
            status
            and all(
                finite(status[name])
                for name in ("stage2_melt_f", "stage2_prcp_fac", "stage2_glen_a", "stage2_temp_bias")
            )
        )
        unique_mid_vector = len(mid_vectors) == 1
        baseline_matches = False
        temp_melt = None
        if status and unique_mid_vector:
            vector = next(iter(mid_vectors))
            baseline_matches = (
                vector[0] == rgi_id
                and all(
                    math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=0.0)
                    for left, right in zip(
                        vector[1:5],
                        (
                            status["stage2_melt_f"], status["stage2_prcp_fac"],
                            status["stage2_glen_a"], status["stage2_temp_bias"],
                        ),
                    )
                )
                and float(vector[5]) == 0.0
            )
        if member_files["source_mb_calib"].is_file():
            with member_files["source_mb_calib"].open("r", encoding="utf-8") as handle:
                temp_melt = json.load(handle).get("mb_global_params", {}).get("temp_melt")
        temp_melt_valid = temp_melt is not None and math.isfinite(float(temp_melt))
        static_geometry = bool(
            geo and geo["length_contract_pass"] == "True" and geo["static_geometry_pass"] == "True"
        )
        climate_valid = bool(
            pixel and pixel["within_maximum_distance"] == "True"
            and pixel["pixel_validity"] == "finite-complete-temp-and-prcp"
        )
        complete = all(
            (ids_match, params_finite, unique_mid_vector, baseline_matches,
             temp_melt_valid, static_geometry, climate_valid, files_exist)
        )
        if not complete:
            incomplete.append(system_id)

        hashes = {name: sha256(path) for name, path in member_files.items() if path.is_file()}
        per_member_hashes[str(system_id)] = hashes
        legacy_state = status["preflight_state"] if status else "missing"
        member_rows.append(
            {
                "ledger_ordinal": ordinal,
                "lia_unique_id": system_id,
                "rgi_id": rgi_id,
                "input_complete": complete,
                "legacy_preflight_state": legacy_state,
                "legacy_status_inherited": False,
                "new_route_status": "transaction-held-not-launched",
                "mid_turnover_baseline_unique": unique_mid_vector,
                "baseline_matches_master": baseline_matches,
                "static_geometry_input_pass": static_geometry,
                "climate_pixel_input_pass": climate_valid,
                "source_files_complete": files_exist,
                "temp_melt": temp_melt,
            }
        )
        if complete:
            for delta in DELTA_T_EQ_VALUES:
                scientific_key = {
                    "lia_unique_id": system_id,
                    "rgi_id": rgi_id,
                    "model_family": "oggm-flowline-1d",
                    "geometry_hash": hashes["corrected_flowline"],
                    "forcing_hash": climate_hash,
                    "delta_t_eq_degC": delta,
                    "base_temp_bias_degC": float(status["stage2_temp_bias"]),
                    "melt_f": float(status["stage2_melt_f"]),
                    "prcp_fac": float(status["stage2_prcp_fac"]),
                    "glen_a_pa_3_s_1": float(status["stage2_glen_a"]),
                    "fs": 0.0,
                    "temp_melt_degC": float(temp_melt),
                    "spinup_forcing": "ensemble_rec_ref-1441-1470-repeat",
                    "block_years": 300,
                    "maximum_years": 3000,
                }
                encoded = json.dumps(scientific_key, sort_keys=True, separators=(",", ":")).encode()
                matrix_rows.append(
                    {
                        "cell_ordinal": len(matrix_rows) + 1,
                        **scientific_key,
                        "effective_temp_bias_degC": float(status["stage2_temp_bias"]) + delta,
                        "fingerprint": hashlib.sha256(encoded).hexdigest().upper(),
                        "status": "held-missing-spinup-numerical-threshold-bundle",
                        "launch_authorized": False,
                    }
                )

    if incomplete:
        raise ValueError(f"member inputs incomplete for systems: {incomplete}")
    if len(matrix_rows) != 191 * len(DELTA_T_EQ_VALUES):
        raise ValueError("Level-1 matrix cardinality mismatch")

    global_hashes = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in absolute_inputs.items()
    }
    input_hashes = {
        "schema": "lia-all191-flowline-input-hashes-v1",
        "global_inputs": global_hashes,
        "member_file_hashes": per_member_hashes,
    }

    evidence_dir.mkdir(parents=True, exist_ok=False)
    atomic_csv(evidence_dir / "member_input_audit.csv", list(member_rows[0]), member_rows)
    atomic_csv(evidence_dir / "level1_matrix.csv", list(matrix_rows[0]), matrix_rows)
    atomic_json(evidence_dir / "input_hashes.json", input_hashes)

    resource_contract = {
        "schema": "lia-all191-flowline-cpu12-process-contract-v1",
        "status": "frozen-but-not-launchable",
        "workers": 12,
        "independent_worker_processes": True,
        "thread_environment": {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        },
        "failure_isolation": "single-chain-only",
        "worker_output_layout": str(RESERVED_OUTPUT_REL / "workers/worker-{00..11}"),
        "atomic_checkpoint_layout": str(RESERVED_OUTPUT_REL / "chains/<fingerprint>/checkpoints/year-<0300..3000>"),
        "pid_contract": {
            "parent": str(RESERVED_OUTPUT_REL / "runtime/parent.pid.json"),
            "workers": str(RESERVED_OUTPUT_REL / "runtime/workers.jsonl"),
            "duplicate_check": "exact transaction-id, output-root, driver and matrix path before launch",
        },
        "log_contract": {
            "parent_stdout": str(RESERVED_OUTPUT_REL / "logs/parent.stdout.log"),
            "parent_stderr": str(RESERVED_OUTPUT_REL / "logs/parent.stderr.log"),
            "worker_pattern": str(RESERVED_OUTPUT_REL / "logs/worker-{00..11}.{stdout,stderr}.log"),
            "append_only": True,
        },
        "checkpoint_cadence_years": [300 * i for i in range(1, 11)],
        "preflight": {
            "required": True,
            "exact_contract_concurrency": 12,
            "status": "not-run-blocked-before-preflight",
            "memory_gate": "all 12 workers plus OS remain below 80% committed RAM",
        },
        "stop_conditions": [
            "validator failure or launch_authorized false",
            "missing or hash-mismatched input",
            "duplicate process, fingerprint or output-root collision",
            "any unresolved numerical threshold",
            "non-finite or negative model state",
            "domain or boundary gate failure once its numerical threshold is frozen",
            "checkpoint hash or exact-contract resume reproduction failure",
            "source input mutation",
            "member-local failure stops only that chain and permits no automatic retry",
            "3000 years reached without registered convergence",
        ],
    }
    atomic_json(evidence_dir / "resource_process_stop_contract.json", resource_contract)

    manifest = {
        "schema": "lia-all191-oggm-flowline-1d-v10.189-transaction-v1",
        "transaction_id": "lia-all191-oggm-flowline-1d-v10.189-blocked-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "transaction-complete-launch-blocked",
        "model_family": "oggm-flowline-1d",
        "population": 191,
        "legacy_accepted_inherited": 0,
        "level1_cells": 1337,
        "delta_t_eq_degC": list(DELTA_T_EQ_VALUES),
        "other_parameters": "per-system mid-turnover Stage-2 baseline; temp_melt from source mb_calib",
        "production_climate": {
            "path": str(RECONSTRUCTION_CLIMATE), "sha256": climate_hash,
            "spinup_period": "1441-1470 repeated", "precipitation_anomaly_mode": "off",
        },
        "spinup": {
            "atomic_block_years": 300, "maximum_years": 3000,
            "checkpoints": [300 * i for i in range(1, 11)],
            "comparison_window_years": 90, "same_phase_cycles": [30, 30, 30],
            "equilibrium_length_role": "output-state-coordinate-only",
            "length_target_matching": False,
            "convergence_thresholds": None,
            "checkpoint_reproduction_tolerance": None,
            "boundary_margin_threshold": None,
            "invalid_rate_ceiling": None,
        },
        "forward": {
            "separate_from_spinup_matrix": True,
            "curve_gate_standard": str(INPUTS["forward_curve_standard_v3"]),
            "curve_gate_sha256": global_hashes["forward_curve_standard_v3"]["sha256"],
            "standard_status": "adopted-for-this-route-by-explicit-user-instruction",
            "eligible_states": "unique fingerprint converged final states only",
        },
        "prohibitions": [
            "no planform-2d", "no GPU", "no simultaneous glen_a/MB/temp_melt Cartesian expansion",
            "no glacier-specific hand tuning", "no inherited pass from the legacy 58",
        ],
        "reserved_simulation_output_root": str(RESERVED_OUTPUT_REL),
        "reserved_output_root_created": False,
        "runtime_driver": None,
        "preflight_launched": False,
        "simulation_launched": False,
        "launch_authorized": False,
        "only_blocker": {
            "id": "BLOCK-V10.189-01",
            "description": (
                "Freeze the exact same-phase spin-up convergence thresholds, confirmation count, "
                "checkpoint reproduction tolerance, boundary margin threshold, and invalid-rate ceiling."
            ),
        },
        "host_inventory": {
            "platform": platform.platform(), "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "artifacts": {
            "member_input_audit": "member_input_audit.csv",
            "level1_matrix": "level1_matrix.csv",
            "input_hashes": "input_hashes.json",
            "resource_process_stop_contract": "resource_process_stop_contract.json",
        },
    }
    atomic_json(evidence_dir / "transaction_manifest.json", manifest)
    summary = {
        "schema": "lia-all191-oggm-flowline-1d-v10.189-transaction-summary-v1",
        "status": "decision-required-no-launch",
        "population": 191,
        "member_inputs_complete": 191,
        "level1_cells_enumerated": 1337,
        "curve_standard_adopted": True,
        "spinup_threshold_bundle_complete": False,
        "only_blocker": manifest["only_blocker"],
        "preflight_launched": False,
        "simulation_launched": False,
        "parent_pid": None,
        "worker_pids": [],
        "reserved_output_root_created": False,
        "launch_authorized": False,
    }
    atomic_json(evidence_dir / "summary.json", summary)

    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(evidence_dir.iterdir())
        if path.is_file()
    }
    atomic_json(evidence_dir / "artifact_hashes.json", artifact_hashes)
    print(json.dumps({"evidence": str(evidence_dir), "status": "created-blocked", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

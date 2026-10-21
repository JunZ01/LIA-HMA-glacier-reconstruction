"""Create the immutable, launchable staged v10.196 CPU12 transaction."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVIDENCE_REL = Path(
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_196_transaction_v1"
)
OUTPUT_REL = Path(
    "w196"
)
SOURCE_MATRIX = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_192_transaction_v1/level1_matrix.csv"
)
SOURCE_INPUT_HASHES = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1/input_hashes.json"
)
CHAIN_DRIVER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_flowline_spinup_v10_196.py"
)
CONTROLLER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_flowline_cpu12_controller_v10_196.py"
)
BASE_CHAIN_DRIVER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_flowline_spinup_v10_192.py"
)
PREFLIGHT_SYSTEMS = (214, 252, 313, 7, 84, 130, 512, 705, 709, 14, 79, 334)
SHORT_WORK_ROOT = PROJECT_ROOT / "w196"
CURRENT_STATE = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
SOURCE_GDIR_ROOT = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/"
    "019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"
)
WINDOWS_PATH_LIMIT = 260
REQUIRED_PATH_MARGIN = 100


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


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:8] / rgi_id[:11] / rgi_id


def scientific_equivalence_audit() -> dict:
    function_names = (
        "topology_signature", "snapshot", "relative_difference", "convergence_metrics",
        "make_model", "state_restore_check", "continuation_check",
    )
    constant_names = (
        "BLOCK_YEARS", "MAX_YEARS", "CYCLE_YEARS", "VOLUME_CYCLE_TOL",
        "VOLUME_90_TOL", "LENGTH_GRID_TOL", "STATE_RTOL",
        "CONTINUATION_REL_TOL", "THICKNESS_ABS_TOL_M", "BOUNDARY_MARGIN_M",
    )

    def inspect(path: Path) -> tuple[dict, dict]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = {
            node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest().upper()
            for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names
        }
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in constant_names:
                    constants[name] = ast.literal_eval(node.value)
        return functions, constants

    base_functions, base_constants = inspect(BASE_CHAIN_DRIVER)
    new_functions, new_constants = inspect(CHAIN_DRIVER)
    if base_functions != new_functions or base_constants != new_constants:
        raise ValueError("v10.196-scientific-code-or-threshold-drift")
    return {
        "schema": "lia-all191-flowline-v10.196-scientific-equivalence-v1",
        "base_driver": str(BASE_CHAIN_DRIVER),
        "base_driver_sha256": sha256(BASE_CHAIN_DRIVER),
        "unchanged_function_ast_sha256": new_functions,
        "unchanged_threshold_constants": new_constants,
        "scientific_code_and_thresholds_equal": True,
        "permitted_code_difference": "runtime path resolution, compact member keys, schemas, and version binding only",
    }


def build_path_audit(rows: list[dict], members: list[dict]) -> dict:
    candidates: list[dict] = []
    member_by_fingerprint = {item["fingerprint"]: item for item in members}
    for row in rows:
        member = member_by_fingerprint[row["fingerprint"]]
        key = member["key"]
        rgi_id = row["rgi_id"]
        source = gdir_path(SOURCE_GDIR_ROOT, rgi_id)
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = gdir_path(SHORT_WORK_ROOT / "m" / key / "w" / "per_glacier", rgi_id)
        for source_file in source.rglob("*"):
            if source_file.is_file() and source_file.suffix != ".properties":
                projected = target / source_file.relative_to(source)
                candidates.append({
                    "kind": "gdir-clone", "member_key": key,
                    "lia_unique_id": int(row["lia_unique_id"]),
                    "path": str(projected), "length": len(str(projected)),
                })
        chain = SHORT_WORK_ROOT / "m" / key / "c"
        synthetic = (
            chain / "checkpoints" / ".year-3000.tmp.4294967295" /
            "checkpoint.json.tmp.4294967295"
        )
        candidates.append({
            "kind": "checkpoint-atomic-temp-max-pid", "member_key": key,
            "lia_unique_id": int(row["lia_unique_id"]),
            "path": str(synthetic), "length": len(str(synthetic)),
        })
    longest = max(candidates, key=lambda item: item["length"])
    margin = WINDOWS_PATH_LIMIT - int(longest["length"])
    if margin < REQUIRED_PATH_MARGIN:
        raise ValueError(f"insufficient-windows-path-margin:{margin}<{REQUIRED_PATH_MARGIN}:{longest}")
    return {
        "schema": "lia-all191-flowline-v10.196-path-audit-v1",
        "effective_windows_path_limit": WINDOWS_PATH_LIMIT,
        "required_margin_characters": REQUIRED_PATH_MARGIN,
        "maximum_projected_path_length": int(longest["length"]),
        "minimum_projected_margin_characters": margin,
        "candidate_path_count": len(candidates),
        "longest_projected_path": longest,
        "all_runtime_artifacts_under_unified_short_root": True,
        "unified_short_root": str(SHORT_WORK_ROOT),
        "member_keys": [item["key"] for item in members],
    }


def main() -> int:
    evidence = PROJECT_ROOT / EVIDENCE_REL
    output_root = PROJECT_ROOT / OUTPUT_REL
    if evidence.exists() or output_root.exists():
        raise FileExistsError(f"transaction-output-or-short-root-collision:{evidence}:{output_root}:{SHORT_WORK_ROOT}")
    for path in (SOURCE_MATRIX, SOURCE_INPUT_HASHES, CHAIN_DRIVER, CONTROLLER, BASE_CHAIN_DRIVER, CURRENT_STATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    current = json.loads(CURRENT_STATE.read_text(encoding="utf-8"))
    if current["authoritative_route"]["version"] != "v10.195":
        raise ValueError("v10.196-requires-current-route-v10.195")
    if int(current["counts"]["active"]) != 0 or int(current["counts"]["queued"]) != 0:
        raise ValueError("v10.196-requires-zero-active-and-queued")
    preserved = current.get("production_mainline_preserved", {})
    if preserved.get("route") != "v10.193" or preserved.get("system_id") != "all191-oggm-flowline-1d-cpu12":
        raise ValueError("v10.196-requires-preserved-v10.193-cpu12-mainline")

    chain_hash = sha256(CHAIN_DRIVER)
    controller_hash = sha256(CONTROLLER)
    source_hashes_hash = sha256(SOURCE_INPUT_HASHES)
    scientific_equivalence = scientific_equivalence_audit()
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
        "transaction_id": "lia-all191-oggm-flowline-1d-v10.196",
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
            "status": "frozen-background-not-authorized", "launch_authorized": False,
        })
    if len({row["fingerprint"] for row in matrix}) != 1337:
        raise ValueError("duplicate-v10.196-fingerprint")

    preflight_rows = [
        row for system_id in PREFLIGHT_SYSTEMS
        for row in matrix
        if int(row["lia_unique_id"]) == system_id and float(row["delta_t_eq_degC"]) == 0.0
    ]
    if len(preflight_rows) != 12:
        raise ValueError(f"preflight-cardinality:{len(preflight_rows)}")
    for row in preflight_rows:
        row["status"] = "authorized-one-technical-retry"
        row["launch_authorized"] = True

    members = [
        {
            "key": f"m{index:02d}", "lia_unique_id": int(row["lia_unique_id"]),
            "rgi_id": row["rgi_id"], "fingerprint": row["fingerprint"],
        }
        for index, row in enumerate(preflight_rows)
    ]
    if len({item["key"] for item in members}) != 12:
        raise ValueError("short-member-key-collision")
    path_audit = build_path_audit(preflight_rows, members)
    preflight_export = [
        {"member_key": member["key"], **row}
        for member, row in zip(members, preflight_rows)
    ]

    evidence.mkdir(parents=True, exist_ok=False)
    atomic_csv(evidence / "level1_matrix.csv", matrix)
    atomic_csv(evidence / "preflight_manifest.csv", preflight_export)
    input_binding = {
        "schema": "lia-all191-flowline-v10.196-input-binding-v1",
        "v10_189_input_hashes": str(SOURCE_INPUT_HASHES),
        "v10_189_input_hashes_sha256": source_hashes_hash,
        "v10_192_level1_matrix_sha256": sha256(SOURCE_MATRIX),
        "curve_standard": str(curve_standard),
        "curve_standard_sha256": sha256(curve_standard),
        "chain_driver": str(CHAIN_DRIVER), "chain_driver_sha256": chain_hash,
        "controller": str(CONTROLLER), "controller_sha256": controller_hash,
    }
    atomic_json(evidence / "input_binding.json", input_binding)
    atomic_json(evidence / "scientific_equivalence.json", scientific_equivalence)
    atomic_json(evidence / "path_length_audit.json", path_audit)
    resource_contract = {
        "schema": "lia-all191-flowline-v10.196-cpu12-contract-v1",
        "worker_count": 12,
        "thread_environment": {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        },
        "unified_short_root": str(SHORT_WORK_ROOT),
        "independent_output_roots": "w196/m/<m00..m11>/c",
        "member_work_roots": "w196/m/<m00..m11>/w",
        "atomic_checkpoints": "w196/m/<m00..m11>/c/checkpoints/year-<year>",
        "checkpoint_cadence_years": 300,
        "single_worker_failure_isolation": True,
        "no_automatic_retry": True,
        "memory_used_fraction_ceiling": 0.80,
        "pid_root": str(OUTPUT_REL / "r"),
        "short_work_root": str(SHORT_WORK_ROOT),
        "log_root": str(OUTPUT_REL / "l"),
        "path_audit": path_audit,
        "prohibitions": ["planform-2d", "GPU", "per-glacier manual tuning", "joint higher-level expansion"],
    }
    atomic_json(evidence / "resource_process_stop_contract.json", resource_contract)
    manifest = {
        "schema": "lia-all191-oggm-flowline-1d-v10.196-transaction-v1",
        "transaction_id": "lia-all191-oggm-flowline-1d-v10.196",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authorized-one-end-to-end-short-root-technical-preflight",
        "population": 191, "level1_cells": 1337, "legacy_pass_inherited": 0,
        "technical_retry_of": "v10.192 atomic checkpoint temporary-file Windows path-length failure",
        "authorization_basis": "explicit user approval of a superseding end-to-end short-path CPU12 preflight",
        "authorized_phase": "preflight-only",
        "accepted_count_effect": 0, "model_family": "oggm-flowline-1d",
        "threshold_bundle": threshold_bundle,
        "spinup_forward_matrices_separate": True,
        "equilibrium_length_role": "output-state-coordinate-only",
        "forward_curve_gate": {"path": str(curve_standard), "sha256": sha256(curve_standard)},
        "inputs": {
            "source_gdir_root": str(SOURCE_GDIR_ROOT),
            "elevation_band_root": str(PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_elevation_band_core191_length_corrected_v1"),
            "climate_pixels": str(PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"),
            "reconstruction_climate": r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\oggm_climate\Ensemble\climate_oggm_ensemble_rec_ref.nc",
            "input_binding": str(evidence / "input_binding.json"),
        },
        "runtime": {
            "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
            "output_root": str(output_root),
            "unified_short_root": str(SHORT_WORK_ROOT),
            "short_work_root": str(SHORT_WORK_ROOT),
            "workers": 12, "chain_driver_sha256": chain_hash,
            "controller_sha256": controller_hash,
            "thread_environment": resource_contract["thread_environment"],
        },
        "preflight": {
            "one_time_only": True, "mode": "first-300y-boundary-plus-30y-resume-reproduction",
            "systems": list(PREFLIGHT_SYSTEMS),
            "fingerprints": [row["fingerprint"] for row in preflight_rows],
            "members": members,
            "requires_all_12_technical_pass": True, "memory_used_fraction_ceiling": 0.80,
        },
        "production_release_condition": "production is not authorized by v10.196",
        "production_authorized": False,
        "invalid_rate_stop": {"minimum_completed_cells": 120, "stop_above": 0.15},
        "launch_authorized": True,
        "simulation_launched": False,
    }
    atomic_json(evidence / "transaction_manifest.json", manifest)
    summary = {
        "schema": "lia-all191-flowline-v10.196-transaction-summary-v1",
        "status": "authorized-one-end-to-end-short-root-technical-preflight-not-yet-launched",
        "population": 191, "level1_cells": 1337, "preflight_cells": 12,
        "threshold_bundle_complete": True, "input_binding_complete": True,
        "output_root": str(output_root), "output_root_created": False,
        "path_length_audit": path_audit,
        "launch_authorized": True, "preflight_launched": False,
        "production_authorized": False, "production_launched": False, "accepted_count_effect": 0,
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

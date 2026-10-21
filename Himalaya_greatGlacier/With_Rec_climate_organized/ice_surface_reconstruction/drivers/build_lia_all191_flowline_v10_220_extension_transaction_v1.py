"""Freeze a CPU8 snapshot of completed v10.216 year-3000 nonconverged cells."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from . import lia_all191_flowline_spinup_v10_216 as base


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "w216"
OUTPUT_ROOT = PROJECT_ROOT / "w220"
EVIDENCE = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_220_extension_transaction_v1"
)
SOURCE_TRANSACTION = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_all191_oggm_flowline_1d_v10_216_recovery_transaction_v1"
)
DRIVER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_flowline_extension_v10_220.py"
)
CONTROLLER = PROJECT_ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/lia_all191_flowline_cpu8_extension_controller_v10_220.py"
)
CURRENT_STATE = PROJECT_ROOT / "LIA_CURRENT_STATE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    if EVIDENCE.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError(f"v10.220-collision:{EVIDENCE}:{OUTPUT_ROOT}")
    for path in (
        DRIVER, CONTROLLER, CURRENT_STATE,
        SOURCE_TRANSACTION / "transaction_manifest.json",
        SOURCE_TRANSACTION / "production_manifest.csv",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    current = json.loads(CURRENT_STATE.read_text(encoding="utf-8"))
    if current["authoritative_route"]["version"] != "v10.219":
        raise ValueError("v10.220-requires-current-route-v10.219")
    if current["current_focus"].get("production_parent_pid") != 54172:
        raise ValueError("v10.220-requires-active-v10.216-parent")

    source_manifest_path = SOURCE_TRANSACTION / "transaction_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    with (SOURCE_TRANSACTION / "production_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    by_key = {row["member_key"]: row for row in source_rows}
    if len(by_key) != 1302:
        raise ValueError("source-production-manifest-cardinality")

    eligible = []
    for result_path in sorted(SOURCE_ROOT.glob("m/*/c/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "max-duration-nonconverged":
            continue
        source_key = result_path.parents[1].name
        row = by_key.get(source_key)
        if row is None or result.get("fingerprint") != row["fingerprint"]:
            raise ValueError(f"source-result-member-mismatch:{source_key}")
        checkpoint = result_path.parent / "checkpoints" / "year-3000"
        checkpoint_manifest_path = checkpoint / "checkpoint.json"
        checkpoint_state_path = checkpoint / "state.pkl.gz"
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        if not (
            int(checkpoint_manifest["year"]) == 3000
            and checkpoint_manifest["fingerprint"] == row["fingerprint"]
            and sha256(checkpoint_state_path) == checkpoint_manifest["state_sha256"]
        ):
            raise ValueError(f"source-checkpoint-validation-failure:{source_key}")
        eligible.append((source_key, row, result_path, checkpoint, checkpoint_manifest_path))
    if not eligible:
        raise ValueError("no-completed-year3000-nonconverged-cells")

    driver_hash = sha256(DRIVER)
    controller_hash = sha256(CONTROLLER)
    contract = {
        "transaction_id": "lia-all191-oggm-flowline-1d-v10.220-extension",
        "source_transaction_sha256": sha256(source_manifest_path),
        "model_family": "oggm-flowline-1d",
        "solver": "SemiImplicitModel-fixed_dt-1.25days",
        "start_year": 3000,
        "maximum_boundary_year": 6000,
        "possible_confirmation_year": 6030,
        "block_years": 300,
        "cycle_years": 30,
        "volume_cycle_tolerance": 0.003,
        "volume_90y_tolerance": 0.009,
        "length_grid_tolerance": 1.0,
        "restart_volume_area_relative_tolerance": 0.001,
        "restart_thickness_absolute_tolerance_m": 1.0,
        "driver_sha256": driver_hash,
        "controller_sha256": controller_hash,
    }
    matrix = []
    members = []
    source_bindings = []
    scientific_keys = (
        "lia_unique_id", "rgi_id", "model_family", "geometry_hash", "forcing_hash",
        "delta_t_eq_degC", "base_temp_bias_degC", "melt_f", "prcp_fac",
        "glen_a_pa_3_s_1", "fs", "temp_melt_degC", "spinup_forcing",
        "effective_temp_bias_degC",
    )
    for index, (source_key, source, result_path, checkpoint, checkpoint_manifest_path) in enumerate(eligible):
        scientific = {key: source[key] for key in scientific_keys}
        binding = {
            "source_fingerprint": source["fingerprint"],
            "source_result_sha256": sha256(result_path),
            "source_checkpoint_manifest_sha256": sha256(checkpoint_manifest_path),
            "source_checkpoint_state_sha256": sha256(checkpoint / "state.pkl.gz"),
        }
        fingerprint = hashlib.sha256(json.dumps(
            {"contract": contract, "cell": scientific, "source": binding},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest().upper()
        key = f"e{index:04d}"
        matrix.append({
            "cell_ordinal": index + 1,
            **scientific,
            "source_member_key": source_key,
            "source_fingerprint": source["fingerprint"],
            "source_result_path": str(result_path.resolve()),
            "source_result_sha256": binding["source_result_sha256"],
            "source_checkpoint_path": str(checkpoint.resolve()),
            "source_checkpoint_manifest_sha256": binding["source_checkpoint_manifest_sha256"],
            "source_checkpoint_state_sha256": binding["source_checkpoint_state_sha256"],
            "fingerprint": fingerprint,
            "status": "authorized-extension",
            "launch_authorized": True,
        })
        members.append({
            "key": key,
            "fingerprint": fingerprint,
            "source_member_key": source_key,
            "lia_unique_id": int(source["lia_unique_id"]),
            "rgi_id": source["rgi_id"],
        })
        source_bindings.append({
            "extension_member_key": key,
            "lia_unique_id": int(source["lia_unique_id"]),
            "delta_t_eq_degC": float(source["delta_t_eq_degC"]),
            **binding,
            "source_result_path": str(result_path.resolve()),
            "source_checkpoint_path": str(checkpoint.resolve()),
        })
    if len({row["fingerprint"] for row in matrix}) != len(matrix):
        raise ValueError("extension-fingerprint-collision")

    max_path = max(
        len(str(OUTPUT_ROOT / "m" / member["key"] / "c" / "checkpoints" /
                ".year-6030.tmp.4294967295" / "checkpoint.json.tmp.4294967295"))
        for member in members
    )
    if 260 - max_path < 100:
        raise ValueError(f"extension-path-margin-failure:{260-max_path}")

    EVIDENCE.mkdir(parents=True, exist_ok=False)
    atomic_csv(EVIDENCE / "extension_matrix.csv", matrix)
    atomic_csv(EVIDENCE / "source_checkpoint_bindings.csv", source_bindings)
    resource = {
        "schema": "lia-v10.220-cpu8-extension-resource-contract-v1",
        "workers": 8,
        "main_cpu12_parent_pid": 54172,
        "combined_worker_ceiling": 20,
        "machine_logical_processors": 28,
        "memory_pause_threshold_fraction": 0.80,
        "thread_environment": {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        },
        "separate_output_root": str(OUTPUT_ROOT),
        "maximum_projected_path_length": max_path,
        "minimum_windows_path_margin": 260 - max_path,
        "technical_invalid_rate_role": "advisory-only",
        "automatic_batch_termination_allowed": False,
        "user_confirmation_required_for_batch_termination": True,
        "extreme_resource_action": "pause-new-extension-dispatch-and-await-user-decision",
    }
    atomic_json(EVIDENCE / "resource_contract.json", resource)
    manifest = {
        "schema": "lia-all191-flowline-v10.220-extension-transaction-v1",
        "transaction_id": contract["transaction_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authorized-fixed-snapshot-extension-not-launched",
        "authorization_basis": (
            "user requested concurrent extension of completed year-3000 nonconverged cells "
            "and explicitly identified capacity for eight additional CPU workers"
        ),
        "source_snapshot_cell_count": len(matrix),
        "contract": contract,
        "inputs": source_manifest["inputs"],
        "runtime": {
            "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
            "output_root": str(OUTPUT_ROOT),
            "workers": 8,
            "driver_sha256": driver_hash,
            "controller_sha256": controller_hash,
            "thread_environment": resource["thread_environment"],
        },
        "extension": {
            "cell_count": len(matrix),
            "fingerprints": [row["fingerprint"] for row in matrix],
            "members": members,
            "source_selection": "immutable completed v10.216 results with status max-duration-nonconverged at snapshot time",
            "future_v10.216_results_included": False,
            "source_recomputation_authorized": False,
        },
        "extension_authorized": True,
        "launch_authorized": True,
        "simulation_launched": False,
        "automatic_batch_termination_allowed": False,
        "user_confirmation_required_for_batch_termination": True,
        "gpu_or_2d_authorized": False,
    }
    atomic_json(EVIDENCE / "transaction_manifest.json", manifest)
    summary = {
        "schema": "lia-all191-flowline-v10.220-extension-transaction-summary-v1",
        "status": "authorized-fixed-snapshot-extension-not-launched",
        "source_snapshot_cell_count": len(matrix),
        "unique_systems": len({int(row["lia_unique_id"]) for row in matrix}),
        "start_year": 3000,
        "maximum_boundary_year": 6000,
        "possible_confirmation_year": 6030,
        "workers": 8,
        "output_root": str(OUTPUT_ROOT),
        "output_root_created": False,
        "source_main_run_unchanged": True,
        "launch_authorized": True,
        "accepted_count_effect": 0,
    }
    atomic_json(EVIDENCE / "summary.json", summary)
    atomic_json(EVIDENCE / "artifact_hashes.json", {
        path.name: sha256(path) for path in sorted(EVIDENCE.iterdir()) if path.is_file()
    })
    print(json.dumps({"evidence": str(EVIDENCE), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

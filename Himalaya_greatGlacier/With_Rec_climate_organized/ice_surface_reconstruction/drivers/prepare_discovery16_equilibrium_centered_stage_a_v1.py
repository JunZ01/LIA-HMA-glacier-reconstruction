"""Materialize the frozen discovery16 static adapters and centered climates.

This is Stage A only. It runs no ice dynamics and changes no source input.
All members use the same registered equilibrium operator and common gates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

from .export_system2005_historical_monthly_ti_bridge_v1 import run as export_climate
from .generic_planform2d_adapter_contract_v1 import array_sha256
from .prepare_igm_planform2d_system2005_v1 import run as prepare_static


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE183 = BASE / "lia_191_current_route_v10_183/summary.json"
AUTH = BASE / "lia_191_discovery16_equilibrium_centered_staged_steady_authorization_v1"
MEMBERS = AUTH / "discovery16_member_manifest.csv"
AUTH_SUMMARY = AUTH / "summary.json"
AUDIT = BASE / "lia_191_population93_static_climate_state_equilibrium_audit_v4/summary.json"
COHORTS = BASE / "lia_191_population_extension_cohorts_v1/population_extension_cohort_assignment.csv"
STAGE2 = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06/stage2_freeze_stage3_routing_v0"
    / "stage2_member_ensemble.csv"
)
PIXELS = BASE / "lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"
GRIDS = BASE / "core191_anchor_reroute_f19_v3"
NETWORKS = BASE / "lia_gdir_anchor_reroute_core191_v3"
GDIRS = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0"
    / "oggm_workdir/per_glacier"
)
RECONSTRUCTION = Path(
    "E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/Ensemble/"
    "climate_oggm_ensemble_rec_ref.nc"
)
DEM = Path("D:/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif")
OUT = BASE / "lia_191_discovery16_equilibrium_centered_stage_a_v1"
RUNTIME_ROOT = ROOT / "validation_data/lia_runtime/discovery16_stage_a_v1"

EXPECTED = [988, 2086, 3185, 801, 4959, 2534, 2570, 705, 2727, 4491, 3016, 3660, 2177, 4588, 5770, 2559]
RUN_SYSTEMS = EXPECTED
RUN_VERSION = "v1"
EXPECTED_STATE_ID = "2026-09-01-v10.183-equilibrium-operator-discovery16-staged-steady-preparation"
EXPECTED_NEXT_ACTION = "prepare-discovery16-equilibrium-centered-planform2d-batch-v1"
OPERATOR = "fixed-parent-climate-area-integrated-zero-net-mb-bisection-v1"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"json-root-not-object:{path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def gdir_path(rgi_id: str) -> Path:
    return GDIRS / rgi_id[:8] / rgi_id[:-3] / rgi_id


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def source_shape(path: Path) -> list[int]:
    with xr.open_dataset(path) as dataset:
        return [int(value) for value in dataset["thickness_m"].isel(member=0).shape]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty-csv:{path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stage2_payload(row: pd.Series, equilibrium_delta: float) -> dict[str, Any]:
    return {
        "ensemble_role": "mid-turnover",
        "melt_f": float(row["melt_f"]),
        "prcp_fac": float(row["prcp_fac"]),
        "glen_a": float(row["glen_a_pa_3_s_1"]),
        "fs": float(row["fs"]),
        "temp_melt_c": -1.0,
        "base_temp_bias_c": float(row["temp_bias"]),
        "delta_t_c": float(equilibrium_delta),
        "prcp_anomaly_mode": "off",
        "equilibrium_operator": OPERATOR,
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    state = load(STATE)
    if state.get("state_id") != EXPECTED_STATE_ID:
        raise RuntimeError(f"unexpected-state:{state.get('state_id')}")
    if sha256(FACTS) != str(state["scientific_contract"]["sha256"]).upper():
        raise RuntimeError("facts-hash-mismatch")
    control_route = ROOT / Path(state["authoritative_route"]["path"])
    if sha256(control_route) != str(state["authoritative_route"]["sha256"]).upper():
        raise RuntimeError("control-route-hash-mismatch")
    if state["next_action"]["type"] != EXPECTED_NEXT_ACTION:
        raise RuntimeError("stage-a-not-current-next-action")

    auth_summary = load(AUTH_SUMMARY)
    audit = load(AUDIT)
    if auth_summary.get("status") != "operator-registered-discovery16-staged-steady-contract-authorized-stage-a-released":
        raise RuntimeError("stage-a-not-authorized")
    if audit.get("operator", {}).get("name") != OPERATOR:
        raise RuntimeError("operator-drift")

    members = pd.read_csv(MEMBERS).sort_values("cohort_order")
    if members["system_id"].astype(int).tolist() != EXPECTED:
        raise RuntimeError("discovery16-membership-drift")
    cohorts = pd.read_csv(COHORTS).set_index("system_id")
    stage2 = pd.read_csv(STAGE2)
    pixels = pd.read_csv(PIXELS)

    common_sources = {
        "control_route": file_record(control_route),
        "authorization": file_record(AUTH_SUMMARY),
        "member_manifest": file_record(MEMBERS),
        "equilibrium_audit": file_record(AUDIT),
        "cohort_assignment": file_record(COHORTS),
        "stage2_table": file_record(STAGE2),
        "climate_pixels": file_record(PIXELS),
        "reconstruction": file_record(RECONSTRUCTION),
        "regional_dem": file_record(DEM),
    }
    source_before: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    prepared_inputs: list[dict[str, Any]] = []
    for member in members.itertuples(index=False):
        sid = int(member.system_id)
        if sid not in RUN_SYSTEMS:
            continue
        rgi_id = str(member.rgi_id)
        grid = GRIDS / f"lia_{sid}_{rgi_id}_system_grid_v1.nc"
        network = NETWORKS / str(sid) / "component_0/lia_centerline_network.pkl.gz"
        gdir = gdir_path(rgi_id)
        baseline = gdir / "climate_historical.nc"
        stage2_rows = stage2[
            stage2["lia_unique_id"].astype(int).eq(sid)
            & stage2["ensemble_role"].astype(str).eq("mid-turnover")
            & stage2["rgi_id"].astype(str).eq(rgi_id)
        ]
        pixel_rows = pixels[
            pixels["rgi_id"].astype(str).eq(rgi_id)
            & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
        ]
        paths_ok = all(path.exists() for path in (grid, network, gdir, baseline))
        inventory.append(
            {
                "system_id": sid,
                "rgi_id": rgi_id,
                "cohort_order": int(member.cohort_order),
                "wave": int(member.wave),
                "grid_exists": grid.is_file(),
                "network_exists": network.is_file(),
                "source_gdir_exists": gdir.is_dir(),
                "baseline_climate_exists": baseline.is_file(),
                "stage2_row_count": len(stage2_rows),
                "climate_pixel_row_count": len(pixel_rows),
                "preflight_pass": paths_ok and len(stage2_rows) == 1 and len(pixel_rows) == 1,
            }
        )
        if not inventory[-1]["preflight_pass"]:
            raise RuntimeError(f"member-input-preflight-failed:{sid}:{inventory[-1]}")
        for label, path in (("grid", grid), ("network", network), ("baseline", baseline)):
            source_before[f"{sid}:{label}"] = file_record(path)
        prepared_inputs.append(
            {
                "member": member,
                "grid": grid,
                "network": network,
                "gdir": gdir,
                "baseline": baseline,
                "stage2": stage2_rows.iloc[0],
            }
        )

    OUT.mkdir(parents=True)
    configs = OUT / "configs"
    members_root = OUT / "members"
    configs.mkdir()
    members_root.mkdir()
    write_csv(OUT / "input_inventory.csv", inventory)

    results: list[dict[str, Any]] = []
    batch_started = time.perf_counter()
    for item in prepared_inputs:
        member = item["member"]
        sid = int(member.system_id)
        rgi_id = str(member.rgi_id)
        print(json.dumps({"stage": "member-start", "system_id": sid}), flush=True)
        member_started = time.perf_counter()
        member_root = members_root / f"system_{sid}"
        static_root = member_root / "static"
        climate_root = member_root / "parent_climate_1441_1470"
        result: dict[str, Any] = {
            "system_id": sid,
            "rgi_id": rgi_id,
            "cohort_order": int(member.cohort_order),
            "wave": int(member.wave),
            "static_status": "not-run",
            "climate_status": "not-run",
            "stage_a_status": "fail",
            "error": "",
        }
        try:
            frozen = stage2_payload(
                item["stage2"], float(member.equilibrium_delta_t_c_from_stage2_base)
            )
            if frozen["fs"] != 0.0:
                raise RuntimeError(f"unsupported-fs:{sid}:{frozen['fs']}")
            static_config = {
                "schema": "lia-191-generic-planform2d-static-config-v1",
                "experiment": f"discovery16-equilibrium-centered-stage-a-{RUN_VERSION}",
                "stage": "static-adapter-preparation-only",
                "system_id": sid,
                "main_rgi_id": rgi_id,
                "synthetic_rgi_id": rgi_id,
                "paired_grid": relative(item["grid"]),
                "lia_network": relative(item["network"]),
                "regional_dem": relative(DEM),
                "output_root": relative(static_root),
                "expected_source_shape": source_shape(item["grid"]),
                "padding_m": 5000.0,
                "registered_lia_length_m": float(cohorts.loc[sid, "lia_length_km"]) * 1000.0,
                "adapter_inputs": {
                    "paired_grid": source_before[f"{sid}:grid"],
                    "lia_network": source_before[f"{sid}:network"],
                    "equilibrium_audit": common_sources["equilibrium_audit"],
                    "control_route": common_sources["control_route"],
                    "stage2_table": common_sources["stage2_table"],
                },
                "stage2": frozen,
                "gates": {
                    "area_relative_error_max": 0.005,
                    "volume_relative_error_max": 0.005,
                    "hypsometry_cdf_error_max": 0.02,
                    "length_relative_error_max": 0.10,
                    "length_ice_threshold_m": 10.0,
                    "minimum_ice_to_grid_edge_m": 5000.0,
                    "bed_surface_closure_m_max": 0.001,
                    "numerical_negative_thickness_tolerance_m": 1.0e-6,
                },
                "authorization": {
                    "dynamics": False,
                    "zero_mb": False,
                    "steady_50y": False,
                    "historical": False,
                    "promotion": False,
                },
            }
            static_config_path = configs / f"system_{sid}_static.json"
            write_json(static_config_path, static_config)
            static_summary = prepare_static(static_config_path, ROOT)
            result["static_status"] = static_summary["status"]
            result["static_summary"] = relative(static_root / "static_summary.json")
            result["static_summary_sha256"] = sha256(static_root / "static_summary.json")
            result["geology"] = relative(static_root / "geology.nc")
            result["geology_sha256"] = sha256(static_root / "geology.nc")
            result["source_shape"] = static_summary["source_shape"]
            result["adapter_shape"] = static_summary["padded_shape"]
            result["source_area_m2"] = static_summary["source_area_m2"]
            result["source_volume_m3"] = static_summary["source_volume_m3"]
            result["area_relative_error"] = static_summary["area_relative_error"]
            result["volume_relative_error"] = static_summary["volume_relative_error"]
            result["minimum_ice_to_grid_edge_m"] = static_summary["minimum_ice_to_grid_edge_m"]
            result["registered_length_relative_error"] = static_summary["registered_length_relative_error"]
            result["connected_components"] = static_summary["initial_connected_component_count_8_neighbor"]
            result["source_thickness_sha256"] = static_summary["adapter_contract"]["source_thickness_sha256"]
            with Dataset(static_root / "geology.nc") as dataset:
                result["geology_initial_thickness_sha256"] = array_sha256(
                    np.asarray(dataset.variables["thk"][:], dtype=np.float64)
                )

            climate_config = {
                "schema": "lia-191-generic-historical-climate-export-config-v1",
                "system_id": sid,
                "main_rgi_id": rgi_id,
                "runtime_rgi_id": rgi_id,
                "synthetic_rgi_id": rgi_id,
                "climate_id": "ensemble_rec_ref",
                "start_year": 1441,
                "end_year": 1470,
                "source_gdir": relative(item["gdir"]),
                "reconstruction_climate": relative(RECONSTRUCTION),
                "reconstruction_climate_sha256": common_sources["reconstruction"]["sha256"],
                "climate_pixels": relative(PIXELS),
                "climate_pixels_sha256": common_sources["climate_pixels"]["sha256"],
                "baseline_climate": relative(item["baseline"]),
                "baseline_climate_sha256": source_before[f"{sid}:baseline"]["sha256"],
                "runtime_gdir_base": relative(RUNTIME_ROOT / f"system_{sid}"),
                "geology": relative(static_root / "geology.nc"),
                "geology_sha256": result["geology_sha256"],
                "adapter_inputs": {
                    "static_summary": {
                        "path": result["static_summary"],
                        "sha256": result["static_summary_sha256"],
                    },
                    "equilibrium_audit": common_sources["equilibrium_audit"],
                    "control_route": common_sources["control_route"],
                    "stage2_table": common_sources["stage2_table"],
                },
                "output_root": relative(climate_root),
                "stage2": frozen,
                "production_source_temperature_file_modified": False,
                "initial_thickness_modification_allowed": False,
                "model_launch": False,
            }
            climate_config_path = configs / f"system_{sid}_parent_climate.json"
            write_json(climate_config_path, climate_config)
            climate_summary = export_climate(climate_config_path)
            result["climate_status"] = climate_summary["status"]
            result["climate_summary"] = relative(climate_root / "summary.json")
            result["climate_summary_sha256"] = sha256(climate_root / "summary.json")
            result["climate_package"] = relative(
                climate_root / "monthly_ti_historical_1441_1470.npz"
            )
            result["climate_package_sha256"] = climate_summary["package_sha256"]
            result["sequence_month_count"] = climate_summary["sequence_month_count"]
            result["formula_absolute_error_m_ice_s"] = climate_summary[
                "maximum_formula_absolute_error_m_ice_s"
            ]
            result["formula_relative_error"] = climate_summary[
                "maximum_formula_relative_error"
            ]
            result["effective_centered_temp_bias_c"] = (
                frozen["base_temp_bias_c"] + frozen["delta_t_c"]
            )
            result["equilibrium_delta_t_c"] = frozen["delta_t_c"]
            result["stage_a_status"] = (
                "pass"
                if static_summary["status"] == "static-pass"
                and climate_summary["status"] == "historical-climate-bridge-pass"
                else "fail"
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}:{error}"
            result["traceback"] = traceback.format_exc()
        result["wall_seconds"] = time.perf_counter() - member_started
        results.append(result)
        write_json(member_root / "stage_a_member_result.json", result)
        print(
            json.dumps(
                {
                    "stage": "member-complete",
                    "system_id": sid,
                    "status": result["stage_a_status"],
                    "wall_seconds": result["wall_seconds"],
                    "error": result["error"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    source_after = {
        label: file_record(ROOT / record["path"] if not Path(record["path"]).is_absolute() else Path(record["path"]))
        for label, record in source_before.items()
    }
    source_immutable = all(
        source_before[label]["sha256"] == source_after[label]["sha256"]
        for label in source_before
    )
    common_after = {
        label: file_record(ROOT / record["path"] if not Path(record["path"]).is_absolute() else Path(record["path"]))
        for label, record in common_sources.items()
    }
    common_immutable = all(
        common_sources[label]["sha256"] == common_after[label]["sha256"]
        for label in common_sources
    )
    passed = sum(row["stage_a_status"] == "pass" for row in results)
    summary = {
        "schema": f"lia-191-discovery16-equilibrium-centered-stage-a-{RUN_VERSION}",
        "status": (
            f"stage-a-{len(RUN_SYSTEMS)}-of-{len(RUN_SYSTEMS)}-pass"
            if passed == len(RUN_SYSTEMS) and source_immutable and common_immutable
            else "stage-a-fail"
        ),
        "recorded_at": now(),
        "operator": OPERATOR,
        "systems": RUN_SYSTEMS,
        "member_count": len(RUN_SYSTEMS),
        "passed_count": passed,
        "failed_count": len(RUN_SYSTEMS) - passed,
        "source_files_immutable": source_immutable and common_immutable,
        "static_adapters_created": sum(row["static_status"] == "static-pass" for row in results),
        "centered_climate_packages_created": sum(
            row["climate_status"] == "historical-climate-bridge-pass" for row in results
        ),
        "dynamics_run": False,
        "model_launch_performed": False,
        "initial_thickness_modified": False,
        "old_output_modified": False,
        "production_source_climate_files_modified": False,
        "accepted_count_effect": 0,
        "batch_wall_seconds": time.perf_counter() - batch_started,
        "member_results_file": relative(OUT / "member_preparation_results.csv"),
        "source_manifest_file": relative(OUT / "source_manifest.json"),
        "next_recommended_action": (
            "adjudicate Stage A and authorize the frozen discovery16 short qualification waves"
            if passed == len(RUN_SYSTEMS) and source_immutable and common_immutable
            else "classify Stage A failures without individual parameter rescue"
        ),
    }
    write_csv(OUT / "member_preparation_results.csv", results)
    write_json(
        OUT / "source_manifest.json",
        {
            "created_at": now(),
            "driver": relative(DRIVER),
            "driver_sha256": sha256(DRIVER),
            "common_sources_before": common_sources,
            "common_sources_after": common_after,
            "member_sources_before": source_before,
            "member_sources_after": source_after,
        },
    )
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if summary["status"] == "stage-a-16-of-16-pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

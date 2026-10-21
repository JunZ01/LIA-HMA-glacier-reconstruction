"""Build inversion-ready modern OGGM directories for a system pilot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import xarray as xr
from shapely.validation import make_valid

from oggm import cfg, tasks, workflow


REQUIRED_GEOMETRY_FILES = (
    "dem",
    "gridded_data",
    "centerlines",
    "inversion_flowlines",
    "downstream_line",
    "flowline_catchments",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--phase", choices=("preflight", "geometry"), default="preflight"
    )
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def ensure_non_c_output(path: Path) -> None:
    if path.drive.upper() == "C:":
        raise ValueError(f"Project output is forbidden on C drive: {path}")


def load_selection(
    manifest_path: Path, geometry_path: Path, system_ids: list[int]
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    selected_systems = manifest[manifest["lia_unique_id"].isin(system_ids)].copy()
    if set(selected_systems["lia_unique_id"].astype(int)) != set(system_ids):
        raise ValueError("At least one requested LIA system is absent from the manifest")

    member_to_system: dict[str, int] = {}
    member_flags: dict[str, str] = {}
    for row in selected_systems.itertuples(index=False):
        candidate_ids = {
            value for value in str(row.candidate_member_rgi_ids).split(";") if value
        }
        for rgi_id in str(row.download_member_rgi_ids).split(";"):
            if not rgi_id:
                continue
            if rgi_id in member_to_system:
                raise ValueError(f"Pilot member occurs in two systems: {rgi_id}")
            member_to_system[rgi_id] = int(row.lia_unique_id)
            member_flags[rgi_id] = (
                "membership-provisional" if rgi_id in candidate_ids else "confirmed"
            )

    geometry = gpd.read_file(geometry_path)
    selected = geometry[geometry["RGIId"].isin(member_to_system)].copy()
    missing = sorted(set(member_to_system) - set(selected["RGIId"]))
    if missing:
        raise ValueError(f"Modern geometry missing {len(missing)} members: {missing[:5]}")

    selected["lia_unique_id"] = selected["RGIId"].map(member_to_system)
    selected["membership_flag"] = selected["RGIId"].map(member_flags)
    selected["geometry_repaired"] = ~selected.geometry.is_valid
    selected["geometry"] = selected.geometry.map(
        lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
    )
    if selected.geometry.isna().any() or selected.geometry.is_empty.any():
        raise ValueError("Pilot contains empty modern geometry after derived repair")
    if (~selected.geometry.is_valid).any():
        raise ValueError("Pilot contains invalid modern geometry after derived repair")

    return selected.to_crs("EPSG:4326"), selected_systems


def resolve_system_ids(manifest_path: Path, configured_ids) -> list[int]:
    if configured_ids == "all":
        manifest = pd.read_csv(manifest_path, usecols=["lia_unique_id"])
        return sorted(manifest["lia_unique_id"].astype(int).unique().tolist())
    system_ids = [int(value) for value in configured_ids]
    if not system_ids or len(system_ids) != len(set(system_ids)):
        raise ValueError("lia_system_ids must be a non-empty unique list or 'all'")
    return system_ids


def build_preflight(
    selected: gpd.GeoDataFrame,
    selected_systems: pd.DataFrame,
    dem_path: Path,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    with rasterio.open(dem_path) as dem:
        dem_bounds = dem.bounds
        dem_metadata = {
            "path": str(dem_path),
            "crs": str(dem.crs),
            "width": dem.width,
            "height": dem.height,
            "resolution": list(dem.res),
            "bounds": list(dem_bounds),
            "dtype": dem.dtypes[0],
            "nodata": dem.nodata,
        }

    bounds = selected.total_bounds
    dem_covers_pilot = bool(
        bounds[0] >= dem_bounds.left
        and bounds[1] >= dem_bounds.bottom
        and bounds[2] <= dem_bounds.right
        and bounds[3] <= dem_bounds.top
    )
    if not dem_covers_pilot:
        raise ValueError("Surface DEM does not cover the complete pilot geometry")

    selection = selected[
        [
            "lia_unique_id",
            "RGIId",
            "O1Region",
            "O2Region",
            "Area",
            "membership_flag",
            "geometry_repaired",
        ]
    ].sort_values(["lia_unique_id", "RGIId"])

    systems = []
    for row in selected_systems.sort_values("lia_unique_id").itertuples(index=False):
        systems.append(
            {
                "lia_unique_id": int(row.lia_unique_id),
                "main_rgi_id": row.main_rgi_id,
                "download_member_count": int(row.download_member_count),
                "candidate_member_count": int(row.candidate_member_count),
                "manifest_status": row.overall_manifest_status,
                "qc_flag": row.blocking_reasons or "none",
                "included_in_pilot": True,
            }
        )

    summary = {
        "pilot_id": config["pilot_id"],
        "phase": "preflight",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "system_count": len(systems),
        "member_count": len(selection),
        "unique_member_count": int(selection["RGIId"].nunique()),
        "membership_flag_counts": selection["membership_flag"].value_counts().to_dict(),
        "derived_modern_geometry_repairs": int(selection["geometry_repaired"].sum()),
        "pilot_geometry_bounds_epsg4326": bounds.tolist(),
        "dem_covers_pilot": dem_covers_pilot,
        "surface_dem": dem_metadata,
        "systems": systems,
        "policy": {
            "review_systems_are_included": bool(config["include_review_systems"]),
            "system_2692": config["membership_policy_2692"],
            "source_geometry_is_read_only": True,
            "project_output_on_c_drive": False,
        },
    }
    return selection, summary


def configure_oggm(config: dict, output_dir: Path, dem_path: Path) -> Path:
    workdir = Path(config.get("oggm_workdir_path", output_dir / config["oggm_workdir_name"]))
    ensure_non_c_output(workdir)
    for directory in (
        workdir,
        output_dir / "oggm_cache",
        output_dir / "oggm_tmp",
        output_dir / "oggm_rgi",
        output_dir / "oggm_tests",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    cfg.initialize(logging_level="WORKFLOW")
    cfg.PATHS["working_dir"] = str(workdir)
    cfg.PATHS["dem_file"] = str(dem_path)
    cfg.PATHS["dl_cache_dir"] = str(output_dir / "oggm_cache")
    cfg.PATHS["tmp_dir"] = str(output_dir / "oggm_tmp")
    cfg.PATHS["rgi_dir"] = str(output_dir / "oggm_rgi")
    cfg.PATHS["test_dir"] = str(output_dir / "oggm_tests")
    cfg.PARAMS["border"] = int(config["border"])
    cfg.PARAMS["use_intersects"] = False
    cfg.PARAMS["use_rgi_area"] = False
    cfg.PARAMS["continue_on_error"] = True
    # Existing and newly-built workdirs use OGGM's tarred shapefile layout.
    # Keeping this explicit makes a run re-entrant across Python processes.
    cfg.PARAMS["use_tar_shapefiles"] = True
    cfg.PARAMS["auto_skip_task"] = True
    cfg.PARAMS["use_multiprocessing"] = bool(config["use_multiprocessing"])
    cfg.PARAMS["mp_processes"] = int(config["mp_processes"])
    return workdir


def run_geometry(
    selected: gpd.GeoDataFrame,
    config: dict,
    output_dir: Path,
    dem_path: Path,
    reset: bool,
) -> tuple[pd.DataFrame, dict]:
    started = time.time()
    workdir = configure_oggm(config, output_dir, dem_path)
    gdirs = workflow.init_glacier_directories(
        selected, reset=reset, force=reset
    )

    task_chain = (
        (tasks.define_glacier_region, {"source": config["dem_source"]}),
        (tasks.glacier_masks, {}),
        (tasks.compute_centerlines, {}),
        (tasks.initialize_flowlines, {}),
        (tasks.compute_downstream_line, {}),
        (tasks.compute_downstream_bedshape, {}),
        (tasks.catchment_area, {}),
        (tasks.catchment_intersections, {}),
        (tasks.catchment_width_geom, {}),
        (tasks.catchment_width_correction, {}),
    )
    for task, kwargs in task_chain:
        workflow.execute_entity_task(task, gdirs, **kwargs)

    system_by_rgi = selected.set_index("RGIId")["lia_unique_id"].to_dict()
    membership_by_rgi = selected.set_index("RGIId")["membership_flag"].to_dict()
    rows = []
    for gdir in gdirs:
        files = {name: bool(gdir.has_file(name)) for name in REQUIRED_GEOMETRY_FILES}
        has_glacier_mask = False
        geometry_error = ""
        if files["gridded_data"]:
            try:
                with xr.open_dataset(gdir.get_filepath("gridded_data")) as dataset:
                    has_glacier_mask = bool(
                        "glacier_mask" in dataset
                        and dataset["glacier_mask"].count().item() > 0
                        and dataset["glacier_mask"].max().item() > 0
                    )
            except Exception as exc:
                geometry_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "lia_unique_id": int(system_by_rgi[gdir.rgi_id]),
                "rgi_id": gdir.rgi_id,
                "membership_flag": membership_by_rgi[gdir.rgi_id],
                **{f"has_{name}": value for name, value in files.items()},
                "has_glacier_mask_variable": has_glacier_mask,
                "geometry_ready": all(files.values()) and has_glacier_mask,
                "geometry_error": geometry_error,
                "gdir_path": gdir.dir,
            }
        )
    status = pd.DataFrame(rows).sort_values(["lia_unique_id", "rgi_id"])
    summary = {
        "pilot_id": config["pilot_id"],
        "phase": "geometry",
        "reset": reset,
        "system_count": int(status["lia_unique_id"].nunique()),
        "member_count": len(status),
        "geometry_ready_count": int(status["geometry_ready"].sum()),
        "geometry_failed_count": int((~status["geometry_ready"]).sum()),
        "ready_by_system": {
            str(system_id): {
                "members": int(len(group)),
                "ready": int(group["geometry_ready"].sum()),
            }
            for system_id, group in status.groupby("lia_unique_id")
        },
        "required_files": list(REQUIRED_GEOMETRY_FILES),
        "workdir": str(workdir),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return status, summary


def write_outputs(output_dir: Path, table: pd.DataFrame, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = summary["phase"]
    table.to_csv(output_dir / f"{phase}_member_status.csv", index=False)
    with open(output_dir / f"{phase}_summary.json", "w", encoding="utf-8") as target:
        json.dump(summary, target, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    repo_root = config_path.parent.parent
    with open(config_path, encoding="utf-8") as source:
        config = json.load(source)

    manifest_path = resolve_path(repo_root, config["manifest_path"])
    geometry_path = resolve_path(repo_root, config["modern_geometry_path"])
    dem_path = resolve_path(repo_root, config["surface_dem_path"])
    output_dir = resolve_path(repo_root, config["output_dir"])
    ensure_non_c_output(output_dir)

    for path in (manifest_path, geometry_path, dem_path):
        if not path.exists():
            raise FileNotFoundError(path)

    system_ids = resolve_system_ids(manifest_path, config["lia_system_ids"])
    selected, selected_systems = load_selection(
        manifest_path, geometry_path, system_ids
    )
    preflight_table, preflight_summary = build_preflight(
        selected, selected_systems, dem_path, config
    )
    write_outputs(output_dir, preflight_table, preflight_summary)
    if args.phase == "preflight":
        print(json.dumps(preflight_summary, indent=2, ensure_ascii=False))
        return

    status, summary = run_geometry(
        selected, config, output_dir, dem_path, reset=args.reset
    )
    write_outputs(output_dir, status, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

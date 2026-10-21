"""Build immutable geometry and provenance inputs for the Stage 3 pilot."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio.mask import mask
from shapely.geometry import Point

from common import (
    REPO_ROOT,
    read_json,
    resolve_path,
    select_regional_pilot,
    sha256,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def read_oggm_pickle(path: Path):
    try:
        with gzip.open(path, "rb") as source:
            return pickle.load(source)
    except gzip.BadGzipFile:
        with path.open("rb") as source:
            return pickle.load(source)


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def shapefile_components(path: Path) -> list[Path]:
    components = []
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        candidate = path.with_suffix(suffix)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        components.append(candidate)
    return components


def source_gdir_hashes(
    source_root: Path,
    member_rows: pd.DataFrame,
    inversion_filesuffix: str,
) -> dict:
    names = (
        "glacier_grid.json",
        "inversion_flowlines.pkl",
        "downstream_line.pkl",
        f"inversion_output{inversion_filesuffix}.pkl",
        "climate_historical.nc",
        "mb_calib.json",
    )
    result = {}
    for rgi_id in sorted(member_rows["rgi_id"].astype(str).unique()):
        directory = gdir_path(source_root, rgi_id)
        missing = [name for name in names if not (directory / name).exists()]
        if missing:
            raise FileNotFoundError(f"{rgi_id} missing Stage 0 files: {missing}")
        result[rgi_id] = {
            name: sha256(directory / name) for name in names
        }
    return result


def register_climates(
    configured: list[dict], expected: int, centroids: pd.DataFrame
) -> pd.DataFrame:
    if len(configured) != expected:
        raise ValueError(
            f"Expected {expected} climate members, found {len(configured)}"
        )
    rows = []
    seen = set()
    for item in configured:
        climate_id = str(item["climate_id"])
        if climate_id in seen:
            raise ValueError(f"Duplicate climate ID: {climate_id}")
        seen.add(climate_id)
        path = resolve_path(item["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path, use_cftime=True) as dataset:
            if set(dataset.data_vars) != {"temp", "prcp"}:
                raise ValueError(
                    f"{path.name} must contain only temp and prcp data variables"
                )
            years = dataset["time"].dt.year.values
            if int(years.min()) != 1000 or int(years.max()) != 2000:
                raise ValueError(f"Unexpected climate coverage in {path.name}")
            if len(years) != 12012:
                raise ValueError(f"Unexpected monthly count in {path.name}")
            lon_min = float(dataset["lon"].min())
            lon_max = float(dataset["lon"].max())
            lat_min = float(dataset["lat"].min())
            lat_max = float(dataset["lat"].max())
            covers = bool(
                centroids["centroid_lon"].between(lon_min, lon_max).all()
                and centroids["centroid_lat"].between(lat_min, lat_max).all()
            )
            rows.append(
                {
                    "climate_id": climate_id,
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "time_start_year": int(years.min()),
                    "time_end_year": int(years.max()),
                    "monthly_count": int(len(years)),
                    "temperature_units": dataset["temp"].attrs.get("units", ""),
                    "precipitation_units": dataset["prcp"].attrs.get("units", ""),
                    "longitude_min": lon_min,
                    "longitude_max": lon_max,
                    "latitude_min": lat_min,
                    "latitude_max": lat_max,
                    "covers_pilot_centroids": covers,
                    "working_individual_status": (
                        "registered-independent-working-individual"
                    ),
                }
            )
    registry = pd.DataFrame(rows).sort_values("climate_id")
    if not registry["covers_pilot_centroids"].all():
        raise ValueError("At least one climate file does not cover the pilot")
    return registry


def nearest_valid_pixel(
    dataset: xr.Dataset,
    longitude: float,
    latitude: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[int, int, float]:
    valid = valid_mask
    if valid is None:
        valid = (
            np.isfinite(dataset["temp"]).all(dim="time")
            & np.isfinite(dataset["prcp"]).all(dim="time")
        ).values
    latitudes = np.asarray(dataset["lat"].values, dtype=float)
    longitudes = np.asarray(dataset["lon"].values, dtype=float)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    lat1 = np.deg2rad(float(latitude))
    lat2 = np.deg2rad(lat_grid)
    delta_lat = lat2 - lat1
    delta_lon = np.deg2rad(lon_grid - float(longitude))
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    distance = 2 * 6_371_000.0 * np.arcsin(np.sqrt(haversine))
    distance[~valid] = np.inf
    flat_index = int(np.argmin(distance))
    if not np.isfinite(distance.flat[flat_index]):
        raise ValueError("Climate file has no finite temp/prcp grid cell")
    lat_index, lon_index = np.unravel_index(flat_index, distance.shape)
    return int(lat_index), int(lon_index), float(distance[lat_index, lon_index])


def register_member_climate_pixels(
    configured_climates: list[dict],
    member_rows: pd.DataFrame,
    modern_geometry_path: Path,
    maximum_distance_m: float,
) -> pd.DataFrame:
    member_ids = sorted(member_rows["rgi_id"].astype(str).unique())
    modern = gpd.read_file(modern_geometry_path)
    modern = modern[modern["RGIId"].isin(member_ids)].to_crs("EPSG:4326")
    missing = sorted(set(member_ids) - set(modern["RGIId"].astype(str)))
    if missing:
        raise ValueError(f"Modern geometry misses pilot members: {missing[:5]}")
    points = modern.geometry.representative_point()
    locations = {
        str(rgi_id): (float(point.x), float(point.y))
        for rgi_id, point in zip(modern["RGIId"], points)
    }
    rows = []
    for item in configured_climates:
        climate_id = str(item["climate_id"])
        path = resolve_path(item["path"])
        with xr.open_dataset(path, use_cftime=True) as dataset:
            valid_mask = (
                np.isfinite(dataset["temp"]).all(dim="time")
                & np.isfinite(dataset["prcp"]).all(dim="time")
            ).values
            for rgi_id in member_ids:
                glacier_lon, glacier_lat = locations[rgi_id]
                lat_index, lon_index, distance_m = nearest_valid_pixel(
                    dataset, glacier_lon, glacier_lat, valid_mask
                )
                rows.append(
                    {
                        "climate_id": climate_id,
                        "rgi_id": rgi_id,
                        "glacier_longitude": glacier_lon,
                        "glacier_latitude": glacier_lat,
                        "climate_lat_index": lat_index,
                        "climate_lon_index": lon_index,
                        "climate_pixel_longitude": float(
                            dataset["lon"].values[lon_index]
                        ),
                        "climate_pixel_latitude": float(
                            dataset["lat"].values[lat_index]
                        ),
                        "climate_pixel_distance_m": distance_m,
                        "within_maximum_distance": bool(
                            distance_m <= maximum_distance_m
                        ),
                        "pixel_validity": "finite-complete-temp-and-prcp",
                    }
                )
    registry = pd.DataFrame(rows).sort_values(["climate_id", "rgi_id"])
    if len(registry) != len(member_ids) * len(configured_climates):
        raise ValueError("Climate pixel registry is incomplete")
    if not registry["within_maximum_distance"].all():
        failed = registry.loc[
            ~registry["within_maximum_distance"],
            ["climate_id", "rgi_id", "climate_pixel_distance_m"],
        ]
        raise ValueError(
            "Nearest finite reconstruction pixel exceeds the distance limit: "
            + failed.head().to_json(orient="records")
        )
    return registry


def sample_lia_hypsometry(
    dem: rasterio.io.DatasetReader,
    geometry,
    geometry_crs,
    quantiles: np.ndarray,
    minimum_pixels: int,
) -> dict:
    projected = gpd.GeoSeries([geometry], crs=geometry_crs).to_crs(dem.crs).iloc[0]
    data, _ = mask(dem, [projected], crop=True, filled=False, all_touched=False)
    values = data[0].compressed().astype(float)
    values = values[np.isfinite(values)]
    if len(values) < minimum_pixels:
        return {
            "hypsometry_status": "unavailable-insufficient-valid-dem-pixels",
            "hypsometry_valid_pixel_count": int(len(values)),
            **{f"elevation_q{int(q * 100):02d}_m": math.nan for q in quantiles},
        }
    sampled = np.quantile(values, quantiles)
    result = {
        "hypsometry_status": "available",
        "hypsometry_valid_pixel_count": int(len(values)),
        "elevation_min_m": float(np.min(values)),
        "elevation_max_m": float(np.max(values)),
        "elevation_mean_m": float(np.mean(values)),
    }
    result.update(
        {
            f"elevation_q{int(q * 100):02d}_m": float(value)
            for q, value in zip(quantiles, sampled)
        }
    )
    return result


def sample_main_flowline_target(
    source_root: Path,
    main_rgi_id: str,
    lia_geometry,
    lia_crs,
    polygon_buffer_grid_cells: float,
) -> dict:
    directory = gdir_path(source_root, main_rgi_id)
    grid = read_json(directory / "glacier_grid.json")
    inversion_flowlines = read_oggm_pickle(directory / "inversion_flowlines.pkl")
    downstream = read_oggm_pickle(directory / "downstream_line.pkl")
    main = inversion_flowlines[-1]
    map_dx = abs(float(grid["dxdy"][0]))
    dx_meter = float(main.dx) * map_dx
    x0, y0 = map(float, grid["x0y0"])
    dx, dy = map(float, grid["dxdy"])
    coordinates = np.asarray(downstream["full_line"].coords, dtype=float)
    projected_coordinates = np.column_stack(
        (x0 + coordinates[:, 0] * dx, y0 + coordinates[:, 1] * dy)
    )
    grid_crs = CRS.from_user_input(grid["proj"])
    polygon = gpd.GeoSeries([lia_geometry], crs=lia_crs).to_crs(grid_crs).iloc[0]
    polygon = polygon.buffer(polygon_buffer_grid_cells * map_dx)
    inside = np.array(
        [polygon.covers(Point(x, y)) for x, y in projected_coordinates],
        dtype=bool,
    )
    if not inside.any():
        return {
            "length_status": "unavailable-main-flowline-misses-lia-polygon",
            "target_main_flowline_length_m": math.nan,
            "main_flowline_domain_length_m": float(len(coordinates) * dx_meter),
            "main_flowline_dx_m": dx_meter,
            "target_near_domain_boundary": False,
        }
    final_index = int(np.where(inside)[0].max())
    target_length = float((final_index + 1) * dx_meter)
    domain_length = float(len(coordinates) * dx_meter)
    return {
        "length_status": "available",
        "target_main_flowline_length_m": target_length,
        "main_flowline_domain_length_m": domain_length,
        "main_flowline_dx_m": dx_meter,
        "target_near_domain_boundary": bool(
            domain_length - target_length <= 3 * dx_meter
        ),
    }


def build_geometry_targets(
    selection: pd.DataFrame,
    lia: gpd.GeoDataFrame,
    dem_path: Path,
    source_root: Path,
    quantiles: np.ndarray,
    minimum_pixels: int,
    polygon_buffer_grid_cells: float,
) -> pd.DataFrame:
    lia_by_id = lia.set_index("UniqueID")
    rows = []
    with rasterio.open(dem_path) as dem:
        for selected in selection.itertuples(index=False):
            system_id = int(selected.lia_unique_id)
            if system_id not in lia_by_id.index:
                raise ValueError(f"LIA geometry missing system {system_id}")
            source = lia_by_id.loc[system_id]
            geometry = source.geometry
            if geometry is None or geometry.is_empty:
                raise ValueError(f"Empty LIA geometry for system {system_id}")
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            area_km2 = float(
                gpd.GeoSeries([geometry], crs=lia.crs).area.iloc[0] / 1e6
            )
            row = {
                "lia_unique_id": system_id,
                "main_rgi_id": selected.main_rgi_id,
                "region_code": int(selected.region_code),
                "target_lia_area_km2": area_km2,
                "manifest_lia_area_km2": float(selected.lia_area_km2),
                "area_manifest_relative_difference": abs(
                    area_km2 / float(selected.lia_area_km2) - 1
                ),
                "area_status": "available",
            }
            row.update(
                sample_main_flowline_target(
                    source_root,
                    str(selected.main_rgi_id),
                    geometry,
                    lia.crs,
                    polygon_buffer_grid_cells,
                )
            )
            row.update(
                sample_lia_hypsometry(
                    dem, geometry, lia.crs, quantiles, minimum_pixels
                )
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values("lia_unique_id")


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    readiness_dir = resolve_path(config["readiness_dir"])
    manifest_path = resolve_path(config["manifest_path"])
    timing_path = resolve_path(config["timing_registry_path"])
    modern_geometry_path = resolve_path(config["modern_geometry_path"])
    lia_path = resolve_path(config["lia_geometry_path"])
    dem_path = resolve_path(config["surface_dem_path"])
    stage0_dir = resolve_path(config["stage0_source_dir"])
    source_root = stage0_dir / "oggm_workdir" / "per_glacier"
    paths = {
        "routes": readiness_dir / "stage3_routing_245.csv",
        "system_ensemble": readiness_dir / "stage2_system_ensemble.csv",
        "member_ensemble": readiness_dir / "stage2_member_ensemble.csv",
        "manifest": manifest_path,
        "timing_registry": timing_path,
        "modern_geometry": modern_geometry_path,
    }
    for path in [*paths.values(), lia_path, dem_path, source_root]:
        if not path.exists():
            raise FileNotFoundError(path)

    routes = pd.read_csv(paths["routes"])
    system_ensemble = pd.read_csv(paths["system_ensemble"])
    member_ensemble = pd.read_csv(paths["member_ensemble"])
    manifest = pd.read_csv(manifest_path)
    timing = pd.read_csv(timing_path)
    lia = gpd.read_file(lia_path)
    if len(routes) != int(config["expected_readiness_systems"]):
        raise ValueError("Readiness route count changed")
    core_flag = routes["farinotti_core_191"].astype(str).str.lower().eq("true")
    if int(core_flag.sum()) != int(config["expected_core_systems"]):
        raise ValueError("Readiness core count changed")
    if len(system_ensemble) != int(config["expected_system_ensemble_rows"]):
        raise ValueError("Stage 2 system ensemble row count changed")
    if len(member_ensemble) != int(config["expected_member_ensemble_rows"]):
        raise ValueError("Stage 2 member ensemble row count changed")
    if not system_ensemble["velocity_selection_weight"].eq(0).all():
        raise ValueError("Velocity entered the frozen Stage 2 ensemble")

    selection = select_regional_pilot(
        routes,
        manifest,
        int(config["pilot_per_region"]),
        [float(value) for value in config["pilot_area_quantiles"]],
        [int(value) for value in config["required_review_system_ids"]],
    )
    expected_pilot = int(config["pilot_system_count"])
    if len(selection) != expected_pilot:
        raise ValueError(f"Expected {expected_pilot} pilot systems")
    if selection["region_code"].nunique() != int(config["pilot_region_count"]):
        raise ValueError("Pilot does not cover all four regions")
    if not set(config["required_review_system_ids"]).issubset(
        set(selection["lia_unique_id"])
    ):
        raise ValueError("Pilot lost a permanent review system")

    timing_fields = [
        "lia_unique_id",
        "core_start_ce",
        "core_end_ce",
        "outer_start_ce",
        "outer_end_ce",
        "timing_weight_model",
        "timing_status",
    ]
    selection = selection.merge(
        timing[timing_fields],
        on="lia_unique_id",
        how="left",
        validate="one_to_one",
    )
    if selection[timing_fields[1:]].isna().any().any():
        raise ValueError("Pilot timing registry is incomplete")
    selection["operator_smoke"] = False
    for region_code, group in selection.groupby("region_code"):
        smoke_id = int(
            group.sort_values(
                ["declared_member_count", "lia_area_km2", "lia_unique_id"]
            ).iloc[0]["lia_unique_id"]
        )
        selection.loc[
            selection["lia_unique_id"] == smoke_id, "operator_smoke"
        ] = True

    selected_ids = set(selection["lia_unique_id"].astype(int))
    selected_system_ensemble = system_ensemble[
        system_ensemble["lia_unique_id"].isin(selected_ids)
    ].copy()
    selected_member_ensemble = member_ensemble[
        member_ensemble["lia_unique_id"].isin(selected_ids)
    ].copy()
    if len(selected_system_ensemble) != expected_pilot * 3:
        raise ValueError("Pilot lacks three Stage 2 roles per system")
    role_counts = selected_system_ensemble.groupby("lia_unique_id")[
        "ensemble_role"
    ].nunique()
    if not role_counts.eq(3).all():
        raise ValueError("Pilot Stage 2 roles are incomplete")

    climate_registry = register_climates(
        config["climate_members"],
        int(config["expected_climate_members"]),
        selection,
    )
    climate_pixel_registry = register_member_climate_pixels(
        config["climate_members"],
        selected_member_ensemble,
        modern_geometry_path,
        float(config["maximum_reconstruction_pixel_distance_m"]),
    )
    if config["operator_smoke_climate_id"] not in set(
        climate_registry["climate_id"]
    ):
        raise ValueError("Operator smoke climate is not registered")

    geometry_targets = build_geometry_targets(
        selection,
        lia,
        dem_path,
        source_root,
        np.asarray(config["hypsometry_quantiles"], dtype=float),
        int(config["minimum_hypsometry_pixels"]),
        float(config["length_polygon_buffer_grid_cells"]),
    )
    if geometry_targets["target_lia_area_km2"].isna().any():
        raise ValueError("Pilot has an unavailable LIA area target")
    if (geometry_targets["area_manifest_relative_difference"] > 1e-6).any():
        raise ValueError("LIA source area differs from the frozen manifest")

    source_hashes = source_gdir_hashes(
        source_root,
        selected_member_ensemble,
        config["stage0_inversion_filesuffix"],
    )
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    input_hashes.update(
        {
            f"lia_{path.suffix.lstrip('.')}": sha256(path)
            for path in shapefile_components(lia_path)
        }
    )
    dem_fingerprint = {
        "path": str(dem_path),
        "bytes": dem_path.stat().st_size,
        "modified_at_utc": datetime.fromtimestamp(
            dem_path.stat().st_mtime, timezone.utc
        ).isoformat(),
    }
    with rasterio.open(dem_path) as dem:
        dem_fingerprint.update(
            {
                "crs": str(dem.crs),
                "width": dem.width,
                "height": dem.height,
                "resolution": list(dem.res),
                "bounds": list(dem.bounds),
                "nodata": dem.nodata,
            }
        )

    selection.to_csv(output_dir / "pilot_selection_12.csv", index=False)
    selected_system_ensemble.to_csv(
        output_dir / "pilot_stage2_system_ensemble.csv", index=False
    )
    selected_member_ensemble.to_csv(
        output_dir / "pilot_stage2_member_ensemble.csv", index=False
    )
    climate_registry.to_csv(output_dir / "climate_registry_6.csv", index=False)
    climate_pixel_registry.to_csv(
        output_dir / "climate_pixel_registry_330.csv", index=False
    )
    geometry_targets.to_csv(
        output_dir / "lia_geometry_targets_12.csv", index=False
    )
    pd.DataFrame(
        columns=["lia_unique_id", "stage", "error_type", "error"]
    ).to_csv(output_dir / "failures.csv", index=False)

    summary = {
        "run_id": config["run_id"],
        "status": "stage3-pilot-preflight-passed-no-dynamic-run-started",
        "pilot_system_count": int(len(selection)),
        "pilot_region_count": int(selection["region_code"].nunique()),
        "operator_smoke_system_count": int(selection["operator_smoke"].sum()),
        "required_review_system_ids": sorted(
            int(value) for value in config["required_review_system_ids"]
        ),
        "climate_member_count": int(len(climate_registry)),
        "system_ensemble_row_count": int(len(selected_system_ensemble)),
        "member_ensemble_row_count": int(len(selected_member_ensemble)),
        "length_target_available_count": int(
            geometry_targets["length_status"].eq("available").sum()
        ),
        "hypsometry_target_available_count": int(
            geometry_targets["hypsometry_status"].eq("available").sum()
        ),
        "target_near_domain_boundary_count": int(
            geometry_targets["target_near_domain_boundary"].sum()
        ),
        "source_gdir_member_count": int(len(source_hashes)),
        "climate_pixel_registry_row_count": int(len(climate_pixel_registry)),
        "maximum_climate_pixel_distance_m": float(
            climate_pixel_registry["climate_pixel_distance_m"].max()
        ),
        "failure_count": 0,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    run_manifest = {
        "run_id": config["run_id"],
        "method_version": config["method_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": input_hashes,
        "climate_hashes": climate_registry.set_index("climate_id")[
            "sha256"
        ].to_dict(),
        "dem_fingerprint": dem_fingerprint,
        "source_gdir_key_hashes_before": source_hashes,
        "preflight_artifact_hashes": {
            name: sha256(output_dir / name)
            for name in (
                "pilot_selection_12.csv",
                "pilot_stage2_system_ensemble.csv",
                "pilot_stage2_member_ensemble.csv",
                "climate_registry_6.csv",
                "climate_pixel_registry_330.csv",
                "lia_geometry_targets_12.csv",
                "failures.csv",
            )
        },
        "frozen_stage2_files_are_read_only": True,
        "source_stage0_gdirs_are_read_only": True,
        "forbidden_search_inputs": [
            "ITS_LIVE likelihood",
            "modern Stage 4 terminal residual",
            "parameter rewrites",
        ],
    }
    write_json_atomic(output_dir / "preflight_summary.json", summary)
    write_json_atomic(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

"""Build the Section-181 persistent synthetic LIA geometry for system 3850.

This stage intentionally stops before apparent MB, inversion, and dynamics.
It uses a real OGGM GlacierDirectory, a fixed local grid, the paired LIA
surface inside the outline, and the frozen regional DEM outside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from rasterio.enums import Resampling
from scipy.ndimage import binary_erosion

from ..paths import resolve
from .lia_centerline_pilot import _prepare_oggm_lia_geometry
from .lia_dynamic_matrix_pilot import (
    fixed_grid_for_geometry,
    load_source_state,
    read_dem_on_grid,
    reproject_source_array,
)
from .lia_geometry_preflight import _gdir_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_oggm(output_root: Path, flowline_dx: float) -> str:
    import oggm
    from oggm import cfg

    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = str(output_root / "oggm_work")
    cfg.PATHS["dl_cache_dir"] = str(output_root / "oggm_cache")
    cfg.PATHS["tmp_dir"] = str(output_root / "oggm_tmp")
    cfg.PARAMS["use_intersects"] = False
    cfg.PARAMS["use_multiple_flowlines"] = True
    cfg.PARAMS["filter_min_slope"] = True
    cfg.PARAMS["flowline_dx"] = float(flowline_dx)
    cfg.PARAMS["use_multiprocessing"] = False
    cfg.PARAMS["auto_skip_task"] = False
    cfg.PARAMS["continue_on_error"] = False
    return str(oggm.__version__)


def _source_entity(source_gdir: Path, output_root: Path) -> gpd.GeoDataFrame:
    archive = source_gdir / "outlines.tar.gz"
    extract_root = output_root / "source_outline_metadata"
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(extract_root)
    return gpd.read_file(extract_root / "outlines.shp")


def _synthetic_entity(
    source_entity: gpd.GeoDataFrame,
    lia_geometry,
    lia_crs,
    synthetic_rgi_id: str,
) -> gpd.GeoDataFrame:
    entity = source_entity.copy().to_crs(4326)
    lia_wgs84 = gpd.GeoSeries([lia_geometry], crs=lia_crs).to_crs(4326).iloc[0]
    lia_equal_area = gpd.GeoSeries([lia_wgs84], crs=4326).to_crs(6933).iloc[0]
    representative = lia_wgs84.representative_point()
    entity = entity.iloc[[0]].copy()
    entity.loc[entity.index[0], "geometry"] = lia_wgs84
    entity.loc[entity.index[0], "RGIId"] = synthetic_rgi_id
    entity.loc[entity.index[0], "Area"] = float(lia_equal_area.area / 1e6)
    entity.loc[entity.index[0], "CenLon"] = float(representative.x)
    entity.loc[entity.index[0], "CenLat"] = float(representative.y)
    if "Name" in entity:
        entity.loc[entity.index[0], "Name"] = "Synthetic LIA system 3850"
    if "BgnDate" in entity:
        entity.loc[entity.index[0], "BgnDate"] = "14700101"
    return entity


def _prepare_gridded_data(gdir, grid, lia_geometry, surface_source, regional_dem):
    pixel_geometry, glacier_mask, exterior = _prepare_oggm_lia_geometry(
        lia_geometry, grid
    )
    if not np.isfinite(regional_dem).all():
        raise ValueError("regional-dem-has-nonfinite-cells")
    surface = np.asarray(surface_source, dtype=float)
    topo = np.asarray(regional_dem, dtype=float).copy()
    missing_surface = glacier_mask & ~np.isfinite(surface)
    if missing_surface.any():
        raise ValueError(
            f"paired-lia-surface-missing-inside-mask:{int(missing_surface.sum())}"
        )
    topo[glacier_mask] = surface[glacier_mask]
    exterior = glacier_mask & ~binary_erosion(
        glacier_mask, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    dataset = xr.Dataset(
        data_vars={
            "glacier_mask": (("y", "x"), glacier_mask.astype(np.int16)),
            "glacier_ext": (("y", "x"), exterior.astype(np.int16)),
            "topo": (("y", "x"), topo.astype(np.float32)),
            "topo_smoothed": (("y", "x"), topo.astype(np.float32)),
        },
        coords={"x": np.arange(grid.nx), "y": np.arange(grid.ny)},
    )
    dataset.to_netcdf(gdir.get_filepath("gridded_data"), engine="netcdf4")
    gdir.write_pickle(
        {
            "polygon_pix": pixel_geometry,
            "polygon_area": float(lia_geometry.area),
        },
        "geometries",
    )
    return glacier_mask, topo


def _topology_metrics(flowlines, grid_dx: float, tolerance: float) -> dict:
    roots = [flowline for flowline in flowlines if flowline.flows_to is None]
    if len(roots) != 1:
        return {"root_count": len(roots), "topology_valid": False}
    by_identity = {id(flowline): index for index, flowline in enumerate(flowlines)}
    within_reverse = 0
    junction_reverse = 0
    for flowline in flowlines:
        surface = np.asarray(flowline.surface_h, dtype=float)
        within_reverse += int(np.sum(np.diff(surface) > tolerance))
        seen = set()
        cursor = flowline
        while cursor.flows_to is not None:
            if id(cursor) in seen:
                return {
                    "root_count": 1,
                    "topology_valid": False,
                    "cycle_detected": True,
                }
            seen.add(id(cursor))
            cursor = cursor.flows_to
        if flowline.flows_to is not None:
            receiver_index = int(flowline.flows_to_indice)
            if float(flowline.flows_to.surface_h[receiver_index]) > (
                float(flowline.surface_h[-1]) + tolerance
            ):
                junction_reverse += 1
    root = roots[0]
    return {
        "root_count": 1,
        "root_index": int(by_identity[id(root)]),
        "branch_count": int(len(flowlines)),
        "topology_valid": True,
        "cycle_detected": False,
        "within_branch_reverse_slope_count": within_reverse,
        "junction_reverse_slope_count": junction_reverse,
        "root_spatial_length_m": float(root.line.length * grid_dx),
    }


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    oggm_version = _configure_oggm(output_root, float(config["flowline_dx"]))

    outline_path = resolve(config["lia_outline"])
    paired_path = resolve(config["paired_grid"])
    source_gdir_root = resolve(config["source_gdir_root"])
    regional_dem_path = Path(config["regional_dem"])
    source_gdir = _gdir_path(source_gdir_root, str(config["main_rgi_id"]))
    source_files = [outline_path, paired_path, regional_dem_path, source_gdir / "outlines.tar.gz"]
    before = {str(path): _sha256(path) for path in source_files}

    source = load_source_state(paired_path, "farinotti2019")
    outlines = gpd.read_file(outline_path)
    row = outlines.loc[outlines["UniqueID"].astype(int).eq(int(config["system_id"]))]
    if len(row) != 1:
        raise ValueError(f"lia-outline-row-count:{len(row)}")
    lia_geometry = row.to_crs(source.crs).iloc[0].geometry
    if not lia_geometry.is_valid or lia_geometry.geom_type != "Polygon":
        raise ValueError("lia-outline-not-one-valid-polygon")

    grid = fixed_grid_for_geometry(
        lia_geometry,
        source.crs,
        float(config["map_dx_m"]),
        float(config["buffer_m"]),
    )
    regional_dem = read_dem_on_grid(regional_dem_path, grid)
    paired_surface = reproject_source_array(
        source,
        source.surface_h_m,
        grid,
        resampling=Resampling.bilinear,
    )

    source_entity = _source_entity(source_gdir, output_root)
    entity = _synthetic_entity(
        source_entity,
        row.iloc[0].geometry,
        outlines.crs,
        str(config["synthetic_rgi_id"]),
    )

    from oggm import tasks, utils

    gdir_base = output_root / "synthetic_gdir"
    gdir = utils.GlacierDirectory(entity.iloc[0], base_dir=gdir_base, reset=True)
    grid.to_json(gdir.get_filepath("glacier_grid"))
    glacier_mask, _ = _prepare_gridded_data(
        gdir, grid, lia_geometry, paired_surface, regional_dem
    )

    tasks.compute_centerlines(gdir)
    tasks.initialize_flowlines(gdir)
    tasks.catchment_area(gdir)
    tasks.catchment_intersections(gdir)
    tasks.catchment_width_geom(gdir)
    tasks.catchment_width_correction(gdir)
    tasks.compute_downstream_line(gdir)
    tasks.compute_downstream_bedshape(gdir)

    flowlines = gdir.read_pickle("inversion_flowlines")
    topology = _topology_metrics(
        flowlines,
        abs(float(grid.dx)),
        float(config["gates"]["reverse_slope_tolerance_m"]),
    )
    geometries = gdir.read_pickle("geometries")
    catchments = geometries.get("catchment_indices", [])
    covered = np.zeros_like(glacier_mask, dtype=bool)
    for cells in catchments:
        values = np.asarray(cells, dtype=int)
        covered[tuple(values.T)] = True
    missing_catchment = int(np.sum(glacier_mask & ~covered))

    downstream = gdir.read_pickle("downstream_line")
    full_line = downstream.get("full_line", downstream.get("downstream_line"))
    downstream_line = downstream.get("downstream_line", full_line)
    downstream_length_m = float(downstream_line.length * abs(float(grid.dx)))
    raster_area_m2 = float(glacier_mask.sum() * abs(float(grid.dx * grid.dy)))
    outline_area_m2 = float(lia_geometry.area)
    registered_length = float(config["registered_lia_length_m"])
    root_length = float(topology.get("root_spatial_length_m", np.nan))
    root_length_error = abs(root_length - registered_length) / registered_length
    area_error = abs(raster_area_m2 - outline_area_m2) / outline_area_m2

    flags = []
    if not topology.get("topology_valid", False):
        flags.append("invalid-topology")
    if int(topology.get("within_branch_reverse_slope_count", 1)) > 0:
        flags.append("within-branch-reverse-slope")
    if int(topology.get("junction_reverse_slope_count", 1)) > 0:
        flags.append("junction-reverse-slope")
    if missing_catchment:
        flags.append("catchment-coverage-hole")
    if root_length_error > float(config["gates"]["root_length_relative_error_max"]):
        flags.append("registered-root-length-mismatch")
    if area_error > float(config["gates"]["outline_area_relative_error_max"]):
        flags.append("outline-raster-area-mismatch")
    if downstream_length_m < float(config["gates"]["minimum_free_downstream_m"]):
        flags.append("downstream-domain-too-short")

    after = {str(path): _sha256(path) for path in source_files}
    source_unchanged = before == after
    if not source_unchanged:
        flags.append("source-mutation")
    summary = {
        "status": "geometry-pass" if not flags else "geometry-fail",
        "experiment": config["experiment"],
        "system_id": int(config["system_id"]),
        "oggm_version": oggm_version,
        "synthetic_rgi_id": str(config["synthetic_rgi_id"]),
        "synthetic_gdir": str(Path(gdir.dir)),
        "outline_hole_count": int(len(lia_geometry.interiors)),
        "outline_area_m2": outline_area_m2,
        "raster_area_m2": raster_area_m2,
        "outline_area_relative_error": area_error,
        "registered_lia_length_m": registered_length,
        "root_length_relative_error": root_length_error,
        "downstream_free_length_m": downstream_length_m,
        "glacier_mask_cell_count": int(glacier_mask.sum()),
        "catchment_count": int(len(catchments)),
        "catchment_missing_ice_cell_count": missing_catchment,
        "source_products_unchanged": source_unchanged,
        **topology,
        "review_flags": flags,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
    }
    _write_json(output_root / "config_snapshot.json", config)
    _write_json(output_root / "geometry_summary.json", summary)
    _write_json(
        output_root / "provenance.json",
        {"source_sha256_before": before, "source_sha256_after": after},
    )
    pd.DataFrame(
        [
            {
                "branch_index": index,
                "flows_to_index": (
                    -1
                    if flowline.flows_to is None
                    else next(
                        i for i, target in enumerate(flowlines)
                        if target is flowline.flows_to
                    )
                ),
                "node_count": int(flowline.nx),
                "spatial_length_m": float(flowline.line.length * abs(float(grid.dx))),
                "surface_min_m": float(np.min(flowline.surface_h)),
                "surface_max_m": float(np.max(flowline.surface_h)),
            }
            for index, flowline in enumerate(flowlines)
        ]
    ).to_csv(output_root / "flowline_inventory.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

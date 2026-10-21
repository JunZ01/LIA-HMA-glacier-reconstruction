"""Materialize the source-native 24 m planform state for LIA system 14.

This is the formal static gate for a new two-dimensional representation.  It
keeps every authoritative paired V3/F19 positive-ice cell on its native grid,
pads that grid with the frozen regional DEM, and instantiates (but does not
advance) OGGM's :class:`Upstream2D`.  No climate, steady, or historical model
is run here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from rasterio.warp import Resampling, reproject
from scipy.interpolate import RegularGridInterpolator

from .v3_flowline_geometry_audit import maximum_cumulative_hypsometry_error_fraction


POSITIVE_ICE_M = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def _control_lengths(coordinates: np.ndarray) -> np.ndarray:
    segments = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    result = np.empty(len(coordinates), dtype=float)
    result[0] = segments[0] / 2.0
    result[-1] = segments[-1] / 2.0
    result[1:-1] = (segments[:-1] + segments[1:]) / 2.0
    return result


def _minimum_ice_to_edge_m(mask: np.ndarray, dx: float) -> float:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0.0
    cell_distance = np.minimum.reduce(
        [rows, mask.shape[0] - 1 - rows, cols, mask.shape[1] - 1 - cols]
    )
    return float(cell_distance.min() * dx)


def run(config_path: Path, project: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = _resolve(project, config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    source = _resolve(project, config["paired_grid"])
    network_path = _resolve(project, config["lia_network"])
    topology_path = _resolve(project, config["topology_audit"])
    feasibility_path = _resolve(project, config["source_native_feasibility"])
    dem_path = _resolve(project, config["regional_dem"])
    inputs = [source, network_path, topology_path, feasibility_path, dem_path]
    before = {str(path): _sha256(path) for path in inputs}

    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if not topology.get("topology_repair_pass"):
        raise RuntimeError("topology-repair-not-passed")
    if feasibility.get("status") != "static-feasibility-pass":
        raise RuntimeError("source-native-feasibility-not-passed")

    with gzip.open(network_path, "rb") as stream:
        network = pickle.load(stream)
    root = network["branches"][int(network["main_branch_index"])]
    root_xy = np.asarray(root["coords_projected_m"], dtype=float)

    with xr.open_dataset(source) as ds:
        thickness = np.asarray(ds["thickness_m"].isel(member=0), dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].isel(member=0), dtype=float)
        bed = np.asarray(ds["bed_elevation_m"].isel(member=0), dtype=float)
        valid = np.asarray(ds["valid_mask"].isel(member=0), dtype=bool)
        lia = np.asarray(ds["lia_mask"], dtype=bool)
        x = np.asarray(ds["x"], dtype=float)
        y = np.asarray(ds["y"], dtype=float)
        source_crs = str(ds["spatial_ref"].attrs["crs_wkt"])
        source_system = int(ds.attrs["lia_unique_id"])
    if source_system != int(config["system_id"]):
        raise RuntimeError("paired-grid-system-mismatch")

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if dx <= 0 or dy >= 0 or not np.isclose(dx, abs(dy)):
        raise RuntimeError("source-grid-not-square-or-oriented")
    positive = (
        valid
        & lia
        & np.isfinite(thickness)
        & np.isfinite(surface)
        & np.isfinite(bed)
        & (thickness > POSITIVE_ICE_M)
    )
    if not positive.any():
        raise RuntimeError("no-positive-source-ice")

    padding_cells = int(math.ceil(float(config["padding_m"]) / dx))
    padded_x = x[0] - padding_cells * dx + np.arange(
        len(x) + 2 * padding_cells, dtype=float
    ) * dx
    padded_y = y[0] + padding_cells * dx - np.arange(
        len(y) + 2 * padding_cells, dtype=float
    ) * dx
    dst_transform = Affine(
        dx, 0.0, padded_x[0] - dx / 2.0, 0.0, -dx, padded_y[0] + dx / 2.0
    )
    padded_bed = np.full((len(padded_y), len(padded_x)), np.nan, dtype=np.float64)
    with rasterio.open(dem_path) as dem:
        reproject(
            source=rasterio.band(dem, 1),
            destination=padded_bed,
            src_transform=dem.transform,
            src_crs=dem.crs,
            src_nodata=dem.nodata,
            dst_transform=dst_transform,
            dst_crs=source_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        dem_crs = str(dem.crs)
        dem_nodata = dem.nodata

    row0 = padding_cells
    col0 = padding_cells
    source_slice = np.s_[row0 : row0 + len(y), col0 : col0 + len(x)]
    source_bed_finite = np.isfinite(bed)
    bed_window = padded_bed[source_slice]
    bed_window[source_bed_finite] = bed[source_bed_finite]
    padded_bed[source_slice] = bed_window
    padded_thickness = np.zeros_like(padded_bed)
    thickness_window = padded_thickness[source_slice]
    thickness_window[positive] = thickness[positive]
    padded_thickness[source_slice] = thickness_window

    from oggm import cfg
    from oggm.core.sia2d import Upstream2D

    cfg.initialize(logging_level="WARNING")
    model = Upstream2D(
        padded_bed,
        init_ice_thick=padded_thickness,
        dx=dx,
        mb_model=None,
        y0=0.0,
        glen_a=float(config["stage2"]["glen_a"]),
        ice_thick_filter=None,
    )

    cell_area = dx * dx
    source_area = float(positive.sum() * cell_area)
    source_volume = float(np.sum(thickness[positive]) * cell_area)
    model_positive = padded_thickness > POSITIVE_ICE_M
    model_area = float(model_positive.sum() * cell_area)
    model_volume = float(np.sum(padded_thickness[model_positive]) * cell_area)
    model_surface = padded_bed + padded_thickness
    closure = float(
        np.max(
            np.abs(
                model_surface[model_positive]
                - padded_bed[model_positive]
                - padded_thickness[model_positive]
            )
        )
    )
    hypsometry = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=surface[positive],
        source_area_m2=np.full(int(positive.sum()), cell_area),
        node_surface_m=model_surface[model_positive],
        model_area_m2=np.full(int(model_positive.sum()), cell_area),
        bin_size_m=100.0,
    )

    interpolator = RegularGridInterpolator(
        (padded_y[::-1], padded_x),
        padded_thickness[::-1],
        bounds_error=False,
        fill_value=np.nan,
    )
    sampled = interpolator(np.column_stack([root_xy[:, 1], root_xy[:, 0]]))
    active = np.isfinite(sampled) & (
        sampled > float(config["gates"]["length_ice_threshold_m"])
    )
    indices = np.flatnonzero(active)
    if len(indices):
        first, last = int(indices[0]), int(indices[-1])
        gaps = np.flatnonzero(~active[first : last + 1]) + first
        observed_length = float(_control_lengths(root_xy)[active].sum())
    else:
        first, last = -1, -1
        gaps = np.array([], dtype=int)
        observed_length = 0.0
    registered = float(config["registered_lia_length_m"])
    length_error = abs(observed_length - registered) / registered
    area_error = abs(model_area - source_area) / source_area
    volume_error = abs(model_volume - source_volume) / source_volume
    edge_distance = _minimum_ice_to_edge_m(model_positive, dx)

    gates = config["gates"]
    flags = {
        "topology_support_unchanged": bool(
            topology.get("raster_support_changed_cell_count") == 0
        ),
        "source_native_feasibility": True,
        "regional_dem_finite": bool(np.isfinite(padded_bed).all()),
        "finite_nonnegative_state": bool(
            np.isfinite(padded_thickness).all() and np.all(padded_thickness >= 0)
        ),
        "surface_bed_thickness_closure": closure
        <= float(gates["bed_surface_closure_m_max"]),
        "area_conservation": area_error <= float(gates["area_relative_error_max"]),
        "volume_conservation": volume_error
        <= float(gates["volume_relative_error_max"]),
        "hypsometry": hypsometry <= float(gates["hypsometry_cdf_error_max"]),
        "registered_length": length_error
        <= float(gates["length_relative_error_max"]),
        "observation_path_continuous": len(gaps) == 0,
        "ice_free_safety_boundary": edge_distance
        >= float(gates["minimum_ice_to_grid_edge_m"]),
        "fs_supported": float(config["stage2"]["fs"]) == 0.0,
    }

    after = {str(path): _sha256(path) for path in inputs}
    flags["source_immutable"] = before == after
    passed = bool(all(flags.values()))
    np.savez_compressed(
        output / "source_native_padded_initial_state.npz",
        x_m=padded_x,
        y_m=padded_y,
        bed_elevation_m=padded_bed.astype(np.float32),
        thickness_m=padded_thickness.astype(np.float32),
        source_row_offset=np.int32(row0),
        source_col_offset=np.int32(col0),
        dx_m=np.float64(dx),
    )
    pd.DataFrame(
        {
            "node": np.arange(len(root_xy), dtype=int),
            "x_m": root_xy[:, 0],
            "y_m": root_xy[:, 1],
            "sampled_thickness_m": sampled,
            "active_above_threshold": active,
        }
    ).to_csv(output / "observation_path_static.csv", index=False)
    summary = {
        "schema": "lia-191-system14-source-native-padded-upstream2d-static-v1",
        "status": "static-pass" if passed else "static-fail",
        "system_id": int(config["system_id"]),
        "mechanism": "source-native-24m-planform-upstream2d-with-real-dem-padding",
        "upstream2d_instantiated": True,
        "upstream2d_advanced": False,
        "zero_mb_run": False,
        "steady_run": False,
        "historical_run": False,
        "source_shape": list(thickness.shape),
        "padded_shape": list(padded_thickness.shape),
        "source_dx_m": dx,
        "padding_cells_each_side": padding_cells,
        "requested_padding_m": float(config["padding_m"]),
        "positive_cell_count": int(positive.sum()),
        "source_area_m2": source_area,
        "model_area_m2": model_area,
        "area_relative_error": area_error,
        "source_volume_m3": source_volume,
        "model_volume_m3": model_volume,
        "volume_relative_error": volume_error,
        "hypsometry_cdf_error_100m": hypsometry,
        "surface_bed_thickness_closure_m": closure,
        "minimum_ice_to_grid_edge_m": edge_distance,
        "registered_lia_length_m": registered,
        "observed_active_control_volume_length_m": observed_length,
        "registered_length_relative_error": length_error,
        "root_first_active_node": first,
        "root_last_active_node": last,
        "root_internal_gap_indices": gaps.astype(int).tolist(),
        "regional_dem_crs": dem_crs,
        "regional_dem_nodata": dem_nodata,
        "stage2_frozen": config["stage2"],
        "gate_flags": flags,
        "source_hashes_before": before,
        "source_hashes_after": after,
        "next_authorized_stage": "zero-mb-0p1y" if passed else None,
    }
    _write_json(output / "static_summary.json", summary)
    _write_json(output / "config_snapshot.json", config)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.config.resolve(), args.project_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

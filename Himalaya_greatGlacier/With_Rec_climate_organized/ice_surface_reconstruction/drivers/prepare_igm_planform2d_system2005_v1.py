"""Prepare an authoritative LIA state for an isolated planform-2D pilot.

This driver is static only.  It preserves every positive V3/F19 source cell
on the native 32.5 m grid, pads the domain with the frozen regional DEM, and
writes IGM-standard ``topg``, ``thk`` and ``usurf`` fields.  It never advances
ice dynamics or climate.
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
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from .v3_flowline_geometry_audit import maximum_cumulative_hypsometry_error_fraction
from .generic_planform2d_adapter_contract_v1 import (
    array_sha256,
    static_schema_names,
    validate_hashed_paths,
)


POSITIVE_ICE_M = 1.0e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
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


def _component_table(mask: np.ndarray, thickness: np.ndarray, cell_area: float) -> list[dict]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    rows = []
    for label in range(1, count + 1):
        selected = labels == label
        rows.append(
            {
                "component": label,
                "cell_count": int(selected.sum()),
                "area_m2": float(selected.sum() * cell_area),
                "volume_m3": float(np.sum(thickness[selected]) * cell_area),
            }
        )
    rows.sort(key=lambda item: item["volume_m3"], reverse=True)
    total_area = sum(item["area_m2"] for item in rows)
    total_volume = sum(item["volume_m3"] for item in rows)
    for item in rows:
        item["area_fraction"] = item["area_m2"] / total_area
        item["volume_fraction"] = item["volume_m3"] / total_volume
    return rows


def run(config_path: Path, project: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    geology_schema, summary_schema, generic_mode = static_schema_names(config)
    output = _resolve(project, config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    source = _resolve(project, config["paired_grid"])
    network_path = _resolve(project, config["lia_network"])
    dem_path = _resolve(project, config["regional_dem"])
    adapter_inputs = validate_hashed_paths(project, config.get("adapter_inputs", {}))
    source_hashes_before = {
        str(source): _sha256(source),
        str(network_path): _sha256(network_path),
        str(dem_path): _sha256(dem_path),
    }

    with gzip.open(network_path, "rb") as stream:
        network = pickle.load(stream)
    root = network["branches"][int(network["main_branch_index"])]
    root_xy = np.asarray(root["coords_projected_m"], dtype=float)

    with xr.open_dataset(source) as dataset:
        thickness_raw = np.asarray(dataset["thickness_m"].isel(member=0), dtype=float)
        surface = np.asarray(dataset["surface_elevation_m"].isel(member=0), dtype=float)
        bed = np.asarray(dataset["bed_elevation_m"].isel(member=0), dtype=float)
        valid = np.asarray(dataset["valid_mask"].isel(member=0), dtype=bool)
        lia = np.asarray(dataset["lia_mask"], dtype=bool)
        x = np.asarray(dataset["x"], dtype=float)
        y = np.asarray(dataset["y"], dtype=float)
        source_crs = str(dataset["spatial_ref"].attrs["crs_wkt"])
        source_system = int(dataset.attrs["lia_unique_id"])

    source_thickness_sha256 = array_sha256(thickness_raw)
    expected_shape = config.get("expected_source_shape")
    if expected_shape is not None and list(thickness_raw.shape) != [int(value) for value in expected_shape]:
        raise RuntimeError(
            f"source-shape-mismatch:{list(thickness_raw.shape)}:{list(expected_shape)}"
        )
    if source_system != int(config["system_id"]):
        raise RuntimeError("paired-grid-system-mismatch")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if dx <= 0 or dy >= 0 or not np.isclose(dx, abs(dy)):
        raise RuntimeError("source-grid-not-square-or-oriented")

    numerical_negative = valid & lia & np.isfinite(thickness_raw) & (thickness_raw < 0)
    most_negative = float(np.min(thickness_raw[numerical_negative])) if numerical_negative.any() else 0.0
    if most_negative < -float(config["gates"]["numerical_negative_thickness_tolerance_m"]):
        raise RuntimeError("source-negative-thickness-beyond-numerical-tolerance")
    thickness = np.maximum(thickness_raw, 0.0)
    positive = valid & lia & np.isfinite(surface) & np.isfinite(bed) & (thickness > POSITIVE_ICE_M)
    if not positive.any():
        raise RuntimeError("no-positive-source-ice")

    padding_cells = int(math.ceil(float(config["padding_m"]) / dx))
    padded_x = x[0] - padding_cells * dx + np.arange(len(x) + 2 * padding_cells) * dx
    padded_y = y[0] + padding_cells * dx - np.arange(len(y) + 2 * padding_cells) * dx
    transform = Affine(dx, 0.0, padded_x[0] - dx / 2.0, 0.0, -dx, padded_y[0] + dx / 2.0)
    padded_bed = np.full((len(padded_y), len(padded_x)), np.nan, dtype=np.float64)
    with rasterio.open(dem_path) as dem:
        reproject(
            source=rasterio.band(dem, 1),
            destination=padded_bed,
            src_transform=dem.transform,
            src_crs=dem.crs,
            src_nodata=dem.nodata,
            dst_transform=transform,
            dst_crs=source_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        dem_crs = str(dem.crs)

    row0 = padding_cells
    col0 = padding_cells
    source_slice = np.s_[row0 : row0 + len(y), col0 : col0 + len(x)]
    bed_window = padded_bed[source_slice]
    bed_window[np.isfinite(bed)] = bed[np.isfinite(bed)]
    padded_bed[source_slice] = bed_window
    padded_thickness = np.zeros_like(padded_bed)
    thickness_window = padded_thickness[source_slice]
    thickness_window[positive] = thickness[positive]
    padded_thickness[source_slice] = thickness_window
    padded_surface = padded_bed + padded_thickness
    source_mask_padded = np.zeros_like(padded_thickness, dtype=np.int8)
    source_mask_window = source_mask_padded[source_slice]
    source_mask_window[lia] = 1
    source_mask_padded[source_slice] = source_mask_window

    cell_area = dx * dx
    source_area = float(positive.sum() * cell_area)
    source_volume = float(np.sum(thickness[positive]) * cell_area)
    model_positive = padded_thickness > POSITIVE_ICE_M
    model_area = float(model_positive.sum() * cell_area)
    model_volume = float(np.sum(padded_thickness[model_positive]) * cell_area)
    area_error = abs(model_area - source_area) / source_area
    volume_error = abs(model_volume - source_volume) / source_volume
    closure = float(np.max(np.abs(padded_surface - padded_bed - padded_thickness)))
    hypsometry = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=surface[positive],
        source_area_m2=np.full(int(positive.sum()), cell_area),
        node_surface_m=padded_surface[model_positive],
        model_area_m2=np.full(int(model_positive.sum()), cell_area),
        bin_size_m=100.0,
    )
    components = _component_table(model_positive, padded_thickness, cell_area)
    rows, cols = np.nonzero(model_positive)
    edge_distance = float(
        np.minimum.reduce(
            [rows, model_positive.shape[0] - 1 - rows, cols, model_positive.shape[1] - 1 - cols]
        ).min()
        * dx
    )

    interpolator = RegularGridInterpolator(
        (padded_y[::-1], padded_x), padded_thickness[::-1], bounds_error=False, fill_value=np.nan
    )
    sampled = interpolator(np.column_stack([root_xy[:, 1], root_xy[:, 0]]))
    active = np.isfinite(sampled) & (sampled > float(config["gates"]["length_ice_threshold_m"]))
    active_indices = np.flatnonzero(active)
    if len(active_indices):
        first, last = int(active_indices[0]), int(active_indices[-1])
        gaps = (np.flatnonzero(~active[first : last + 1]) + first).astype(int)
        observed_length = float(_control_lengths(root_xy)[active].sum())
    else:
        first, last = -1, -1
        gaps = np.array([], dtype=int)
        observed_length = 0.0
    registered = float(config["registered_lia_length_m"])
    length_error = abs(observed_length - registered) / registered

    geology = xr.Dataset(
        data_vars={
            "topg": (("y", "x"), padded_bed.astype(np.float32)),
            "thk": (("y", "x"), padded_thickness.astype(np.float32)),
            "usurf": (("y", "x"), padded_surface.astype(np.float32)),
            "icemask": (("y", "x"), np.ones_like(source_mask_padded, dtype=np.int8)),
            "lia_source_mask": (("y", "x"), source_mask_padded),
        },
        coords={"x": padded_x.astype(np.float64), "y": padded_y.astype(np.float64)},
        attrs={
            "schema": geology_schema,
            "system_id": int(config["system_id"]),
            "source_paired_grid": str(source),
            "crs_wkt": source_crs,
            "dx_m": dx,
        },
    )
    for variable in ("topg", "thk", "usurf", "icemask", "lia_source_mask"):
        geology[variable].attrs["grid_mapping"] = "spatial_ref"
    geology["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": source_crs, "spatial_ref": source_crs})
    geology_path = output / "geology.nc"
    geology.to_netcdf(geology_path, engine="netcdf4")

    gates = config["gates"]
    flags = {
        "regional_dem_finite": bool(np.isfinite(padded_bed).all()),
        "finite_nonnegative_state": bool(np.isfinite(padded_thickness).all() and np.all(padded_thickness >= 0)),
        "surface_bed_thickness_closure": closure <= float(gates["bed_surface_closure_m_max"]),
        "area_conservation": area_error <= float(gates["area_relative_error_max"]),
        "volume_conservation": volume_error <= float(gates["volume_relative_error_max"]),
        "hypsometry": hypsometry <= float(gates["hypsometry_cdf_error_max"]),
        "registered_length": length_error <= float(gates["length_relative_error_max"]),
        "ice_free_safety_boundary": edge_distance >= float(gates["minimum_ice_to_grid_edge_m"]),
        "source_ice_connected_2d": len(components) == 1,
        "fs_supported": float(config["stage2"]["fs"]) == 0.0,
    }
    source_hashes_after = {
        str(source): _sha256(source),
        str(network_path): _sha256(network_path),
        str(dem_path): _sha256(dem_path),
    }
    flags["source_immutable"] = source_hashes_before == source_hashes_after
    passed = bool(all(flags.values()))

    pd.DataFrame(
        {
            "node": np.arange(len(root_xy), dtype=int),
            "x_m": root_xy[:, 0],
            "y_m": root_xy[:, 1],
            "sampled_thickness_m": sampled,
            "active_above_threshold": active,
        }
    ).to_csv(output / "observation_path_static.csv", index=False)
    pd.DataFrame(components).to_csv(output / "initial_components.csv", index=False)
    summary = {
        "schema": summary_schema,
        "status": "static-pass" if passed else "static-fail",
        "system_id": int(config["system_id"]),
        "mechanism": "independent-igm-3p2-planform2d-native-grid-real-dem-padding",
        "dynamics_run": False,
        "source_shape": list(thickness.shape),
        "padded_shape": list(padded_thickness.shape),
        "dx_m": dx,
        "padding_cells_each_side": padding_cells,
        "source_positive_cell_count": int(positive.sum()),
        "numerical_negative_source_cell_count": int(numerical_negative.sum()),
        "most_negative_source_thickness_m": most_negative,
        "source_area_m2": source_area,
        "model_area_m2": model_area,
        "area_relative_error": area_error,
        "source_volume_m3": source_volume,
        "model_volume_m3": model_volume,
        "volume_relative_error": volume_error,
        "hypsometry_cdf_error_100m": hypsometry,
        "surface_bed_thickness_closure_m": closure,
        "minimum_ice_to_grid_edge_m": edge_distance,
        "initial_connected_component_count_8_neighbor": len(components),
        "initial_components": components,
        "registered_lia_length_m": registered,
        "observed_active_control_volume_length_m": observed_length,
        "registered_length_relative_error": length_error,
        "root_first_active_node": first,
        "root_last_active_node": last,
        "root_internal_gap_indices_diagnostic_only": gaps.tolist(),
        "root_gap_policy": "classify-and-report; not a numerical failure for the 2D state",
        "regional_dem_crs": dem_crs,
        "stage2_frozen": config["stage2"],
        "gate_flags": flags,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "geology_nc": str(geology_path),
        "geology_nc_sha256": _sha256(geology_path),
        "next_authorized_stage": "igm-load-and-gpu-smoke-test" if passed else None,
    }
    if generic_mode:
        summary["adapter_contract"] = {
            "generic_system_id": True,
            "source_system_id_matches": source_system == int(config["system_id"]),
            "source_thickness_sha256": source_thickness_sha256,
            "initial_thickness_modified": False,
            "old_output_modified": False,
            "validated_inputs": adapter_inputs,
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

"""Static preflight for a conservative planform 2-D SIA representation.

This is a new geometry/dynamics family.  It does not alter climate, Stage-2
physics, the registered length targets, or any accepted result.  The paired
V3/F19 raster is remapped conservatively onto an existing immutable 25 m
synthetic-gdir DEM domain and only a static :class:`oggm.core.sia2d.Upstream2D`
model is instantiated.  No zero-MB, climate-steady, or historical run occurs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .v3_flowline_geometry_audit import (
    maximum_cumulative_hypsometry_error_fraction,
)


SCHEMA_VERSION = 1


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(safe(payload), indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_gdir_grid(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    item = json.loads(path.read_text(encoding="utf-8"))
    nx, ny = [int(v) for v in item["nxny"]]
    x0, y0 = [float(v) for v in item["x0y0"]]
    dx, dy = [float(v) for v in item["dxdy"]]
    if dx <= 0 or dy >= 0 or not np.isclose(abs(dx), abs(dy)):
        raise ValueError("target-grid-must-be-square-x-ascending-y-descending")
    x = x0 + np.arange(nx, dtype=float) * dx
    y = y0 + np.arange(ny, dtype=float) * dy
    return x, y, dx, dy


def _conservative_axis_aligned_remap(
    *,
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_thickness: np.ndarray,
    source_bed: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Conservatively remap positive-ice source cells to a square target grid."""
    sx = np.asarray(source_x, dtype=float)
    sy = np.asarray(source_y, dtype=float)
    tx = np.asarray(target_x, dtype=float)
    ty = np.asarray(target_y, dtype=float)
    thick = np.asarray(source_thickness, dtype=float)
    bed = np.asarray(source_bed, dtype=float)
    sdx = float(np.median(np.diff(sx)))
    sdy = float(np.median(np.diff(sy)))
    tdx = float(np.median(np.diff(tx)))
    tdy = float(np.median(np.diff(ty)))
    if sdx <= 0 or sdy >= 0 or tdx <= 0 or tdy >= 0:
        raise ValueError("unsupported-grid-orientation")
    if not np.isclose(abs(sdx), abs(sdy)) or not np.isclose(abs(tdx), abs(tdy)):
        raise ValueError("source-and-target-cells-must-be-square")

    ny, nx = len(ty), len(tx)
    target_cell_area = abs(tdx * tdy)
    volume = np.zeros((ny, nx), dtype=np.float64)
    ice_area = np.zeros((ny, nx), dtype=np.float64)
    bed_area = np.zeros((ny, nx), dtype=np.float64)
    source_positive = np.isfinite(thick) & np.isfinite(bed) & (thick > 1e-6)
    source_rows, source_cols = np.nonzero(source_positive)
    target_left = tx[0] - 0.5 * tdx
    target_top = ty[0] + 0.5 * abs(tdy)
    source_half = 0.5 * abs(sdx)

    for row, col in zip(source_rows.tolist(), source_cols.tolist()):
        left = sx[col] - source_half
        right = sx[col] + source_half
        bottom = sy[row] - source_half
        top = sy[row] + source_half
        i0 = int(np.floor((left - target_left) / tdx))
        i1 = int(np.floor((np.nextafter(right, -np.inf) - target_left) / tdx))
        j0 = int(np.floor((target_top - top) / abs(tdy)))
        j1 = int(np.floor((target_top - np.nextafter(bottom, np.inf)) / abs(tdy)))
        if i0 < 0 or j0 < 0 or i1 >= nx or j1 >= ny:
            raise RuntimeError(f"source-positive-cell-outside-target:{row}:{col}")
        for j in range(j0, j1 + 1):
            target_bottom = ty[j] - 0.5 * abs(tdy)
            target_upper = ty[j] + 0.5 * abs(tdy)
            overlap_y = min(top, target_upper) - max(bottom, target_bottom)
            if overlap_y <= 0:
                continue
            for i in range(i0, i1 + 1):
                target_l = tx[i] - 0.5 * tdx
                target_r = tx[i] + 0.5 * tdx
                overlap_x = min(right, target_r) - max(left, target_l)
                if overlap_x <= 0:
                    continue
                overlap = overlap_x * overlap_y
                ice_area[j, i] += overlap
                volume[j, i] += thick[row, col] * overlap
                bed_area[j, i] += bed[row, col] * overlap

    target_thickness = volume / target_cell_area
    target_bed_ice = np.full_like(volume, np.nan)
    mapped = ice_area > 0
    target_bed_ice[mapped] = bed_area[mapped] / ice_area[mapped]
    source_cell_area = abs(sdx * sdy)
    return {
        "thickness": target_thickness,
        "bed_on_ice": target_bed_ice,
        "ice_fraction": ice_area / target_cell_area,
        "source_positive_mask": source_positive,
        "source_cell_area_m2": source_cell_area,
        "target_cell_area_m2": target_cell_area,
        "overlap_area_m2": float(ice_area.sum()),
        "mapped_volume_m3": float(volume.sum()),
    }


def _load_root_path(
    flowline_path: Path,
    grid_json: Path,
) -> tuple[np.ndarray, float, int]:
    with gzip.open(flowline_path, "rb") as stream:
        flowlines = pickle.load(stream)
    roots = [(index, fl) for index, fl in enumerate(flowlines) if fl.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"root-count-not-one:{len(roots)}")
    root_index, root = roots[0]
    grid = json.loads(grid_json.read_text(encoding="utf-8"))
    x0, y0 = [float(v) for v in grid["x0y0"]]
    dx, dy = [float(v) for v in grid["dxdy"]]
    coordinates = np.asarray(root.line.coords, dtype=float)
    projected = np.column_stack([
        x0 + coordinates[:, 0] * dx,
        y0 + coordinates[:, 1] * dy,
    ])
    spacing = float(root.dx * root.map_dx)
    return projected, spacing, int(root_index)


def _sample_path(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    cols = np.rint((coordinates[:, 0] - x[0]) / dx).astype(int)
    rows = np.rint((coordinates[:, 1] - y[0]) / dy).astype(int)
    if np.any(cols < 0) or np.any(cols >= len(x)) or np.any(rows < 0) or np.any(rows >= len(y)):
        raise RuntimeError("observation-path-outside-target-grid")
    return np.asarray(values[rows, cols], float), rows, cols


def _minimum_ice_to_edge_distance_m(mask: np.ndarray, dx_m: float) -> float:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0.0
    cells = np.minimum.reduce([
        rows,
        mask.shape[0] - 1 - rows,
        cols,
        mask.shape[1] - 1 - cols,
    ])
    return float(cells.min() * dx_m)


def run(config_path: Path, project_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = _resolve(project_root, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    paired_grid = _resolve(project_root, config["paired_grid"])
    target_gridded = _resolve(project_root, config["target_gridded_data"])
    target_grid_json = _resolve(project_root, config["target_grid_json"])
    observation_flowlines = _resolve(project_root, config["observation_flowlines"])
    inputs = [paired_grid, target_gridded, target_grid_json, observation_flowlines]
    before_hash = {str(path): _sha256(path) for path in inputs}

    with xr.open_dataset(paired_grid) as source_ds:
        source_x = np.asarray(source_ds.x.values, float)
        source_y = np.asarray(source_ds.y.values, float)
        source_thickness = np.asarray(source_ds.thickness_m.values[0], float)
        source_bed = np.asarray(source_ds.bed_elevation_m.values[0], float)
        source_surface = np.asarray(source_ds.surface_elevation_m.values[0], float)
        source_system = int(source_ds.attrs["lia_unique_id"])
    if source_system != int(config["system_id"]):
        raise RuntimeError("paired-grid-system-id-mismatch")

    target_x, target_y, target_dx, _ = _load_gdir_grid(target_grid_json)
    with xr.open_dataset(target_gridded) as target_ds:
        target_dem = np.asarray(target_ds["topo"].values, float)
    expected_shape = (len(target_y), len(target_x))
    if target_dem.shape != expected_shape or not np.isfinite(target_dem).all():
        raise RuntimeError("target-dem-shape-or-finite-gate-failed")

    remap = _conservative_axis_aligned_remap(
        source_x=source_x,
        source_y=source_y,
        source_thickness=source_thickness,
        source_bed=source_bed,
        target_x=target_x,
        target_y=target_y,
    )
    target_thickness = np.asarray(remap["thickness"], float)
    target_bed = target_dem.copy()
    mapped = np.asarray(remap["ice_fraction"], float) > 0
    target_bed[mapped] = np.asarray(remap["bed_on_ice"], float)[mapped]
    target_surface = target_bed + target_thickness

    from oggm import cfg
    from oggm.core.sia2d import Upstream2D

    cfg.initialize(logging_level="WARNING")
    model = Upstream2D(
        target_bed,
        init_ice_thick=target_thickness,
        dx=target_dx,
        mb_model=None,
        y0=0.0,
        glen_a=float(config["stage2"]["glen_a"]),
        ice_thick_filter=None,
    )

    source_positive = np.asarray(remap["source_positive_mask"], bool)
    source_cell_area = float(remap["source_cell_area_m2"])
    source_area = float(source_positive.sum() * source_cell_area)
    source_volume = float(np.sum(source_thickness[source_positive]) * source_cell_area)
    model_area = float(model.area_m2)
    model_volume = float(model.volume_m3)
    source_surface_positive = source_surface[source_positive]
    model_positive = target_thickness > 1e-6
    hypsometry_error = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=source_surface_positive,
        source_area_m2=np.full(source_positive.sum(), source_cell_area),
        node_surface_m=target_surface[model_positive],
        model_area_m2=np.full(model_positive.sum(), target_dx * target_dx),
        bin_size_m=100.0,
    )

    path_xy, path_spacing, root_index = _load_root_path(
        observation_flowlines, target_grid_json
    )
    path_thickness, path_rows, path_cols = _sample_path(
        target_thickness, target_x, target_y, path_xy
    )
    threshold = float(config["gates"]["length_ice_threshold_m"])
    active = path_thickness > threshold
    active_indices = np.flatnonzero(active)
    initial_gap_count = 0
    if len(active_indices):
        last_active = int(active_indices[-1])
        initial_gap_count = int((~active[: last_active + 1]).sum())
        observed_length = float((last_active + 1) * path_spacing)
        free_path = float((len(path_thickness) - last_active - 1) * path_spacing)
    else:
        last_active = -1
        observed_length = 0.0
        free_path = float(len(path_thickness) * path_spacing)

    registered_length = float(config["registered_lia_length_m"])
    length_error = abs(observed_length - registered_length) / registered_length
    volume_error = abs(model_volume - source_volume) / source_volume
    area_error = abs(model_area - source_area) / source_area
    overlap_error = abs(float(remap["overlap_area_m2"]) - source_area) / source_area
    min_edge_distance = _minimum_ice_to_edge_distance_m(model_positive, target_dx)
    bed_surface_closure = float(
        np.max(np.abs(target_surface[model_positive] - target_bed[model_positive]
                      - target_thickness[model_positive]))
    )

    gates = config["gates"]
    flags = {
        "source_coverage": overlap_error <= float(gates["source_coverage_relative_error_max"]),
        "volume_conservation": volume_error <= float(gates["volume_relative_error_max"]),
        "area_representation": area_error <= float(gates["area_relative_error_max"]),
        "hypsometry": hypsometry_error <= float(gates["hypsometry_cdf_error_max"]),
        "finite_legal_state": bool(
            np.isfinite(target_bed).all()
            and np.isfinite(target_thickness).all()
            and np.all(target_thickness >= 0)
            and bed_surface_closure <= float(gates["bed_surface_closure_m_max"])
        ),
        "registered_length": length_error <= float(gates["length_relative_error_max"]),
        "path_connected": initial_gap_count == 0,
        "downstream_path": free_path >= float(gates["minimum_free_downstream_m"]),
        "two_dimensional_boundary": min_edge_distance >= float(gates["minimum_ice_to_grid_edge_m"]),
        "fs_supported": float(config["stage2"]["fs"]) == 0.0,
    }

    path_table = pd.DataFrame({
        "node": np.arange(len(path_thickness), dtype=int),
        "x_m": path_xy[:, 0],
        "y_m": path_xy[:, 1],
        "row": path_rows,
        "col": path_cols,
        "thickness_m": path_thickness,
        "active_above_threshold": active,
    })
    path_table.to_csv(output_root / "observation_path_static.csv", index=False)
    np.savez_compressed(
        output_root / "planform_2d_initial_state.npz",
        x_m=target_x,
        y_m=target_y,
        bed_elevation_m=target_bed.astype(np.float32),
        thickness_m=target_thickness.astype(np.float32),
        ice_fraction=np.asarray(remap["ice_fraction"], np.float32),
    )

    after_hash = {str(path): _sha256(path) for path in inputs}
    source_unchanged = before_hash == after_hash
    flags["source_immutable"] = source_unchanged
    passed = bool(all(flags.values()))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "static-pass" if passed else "static-fail",
        "experiment": "planform-2d-sia-v1",
        "stage": "static-preflight",
        "system_id": int(config["system_id"]),
        "representation": "conservative V3/F19 planform raster on padded 25 m DEM",
        "prospective_solver": "oggm.core.sia2d.Upstream2D",
        "oggm_run": False,
        "upstream2d_instantiated": True,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "source_grid_shape": list(source_thickness.shape),
        "target_grid_shape": list(target_thickness.shape),
        "source_dx_m": float(np.median(np.diff(source_x))),
        "target_dx_m": target_dx,
        "source_positive_cell_count": int(source_positive.sum()),
        "target_positive_cell_count": int(model_positive.sum()),
        "source_area_m2": source_area,
        "model_area_m2": model_area,
        "area_relative_error": area_error,
        "source_volume_m3": source_volume,
        "model_volume_m3": model_volume,
        "volume_relative_error": volume_error,
        "source_overlap_relative_error": overlap_error,
        "hypsometry_cdf_error_100m": hypsometry_error,
        "bed_surface_closure_m": bed_surface_closure,
        "minimum_ice_to_grid_edge_m": min_edge_distance,
        "observation_root_index": root_index,
        "observation_path_node_spacing_m": path_spacing,
        "observation_path_node_count": len(path_thickness),
        "last_active_path_node": last_active,
        "path_internal_gap_count": initial_gap_count,
        "initial_observed_length_m": observed_length,
        "registered_lia_length_m": registered_length,
        "registered_length_relative_error": length_error,
        "free_downstream_path_m": free_path,
        "stage2_frozen": config["stage2"],
        "gate_flags": flags,
        "source_hashes_before": before_hash,
        "source_hashes_after": after_hash,
        "source_unchanged": source_unchanged,
        "next_authorized_stage": "zero-mb" if passed else None,
    }
    _write_json(output_root / "static_summary.json", summary)
    _write_json(output_root / "config_snapshot.json", config)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.config.resolve(), args.project_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()

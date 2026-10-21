"""Preflight LIA system geometry before 3-D surface reconstruction.

This driver does not solve a LIA surface. It checks whether the selected LIA
polygon is representable on each modern gdir grid and whether both bed
scenarios cover the polygon. Modern flowlines are reported as provenance only;
they are not silently promoted to LIA centerlines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from scipy.ndimage import binary_erosion
from shapely.geometry import box

from ..paths import resolve
from .gridded_bed import load_gridded_bed
from .oggm_flowline_io import load_flowline_inputs


def _gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def _modern_margin_bed_jumps(
    bed_m: np.ndarray, modern_mask: np.ndarray, lia_mask: np.ndarray
) -> dict[str, float | int]:
    """Absolute bed jumps across modern-mask edges inside the LIA polygon."""
    values: list[np.ndarray] = []
    horizontal = modern_mask[:, 1:] != modern_mask[:, :-1]
    horizontal &= lia_mask[:, 1:] & lia_mask[:, :-1]
    if horizontal.any():
        values.append(np.abs(bed_m[:, 1:] - bed_m[:, :-1])[horizontal])
    vertical = modern_mask[1:, :] != modern_mask[:-1, :]
    vertical &= lia_mask[1:, :] & lia_mask[:-1, :]
    if vertical.any():
        values.append(np.abs(bed_m[1:, :] - bed_m[:-1, :])[vertical])
    if not values:
        return {
            "modern_margin_edge_count": 0,
            "modern_margin_bed_jump_median_m": np.nan,
            "modern_margin_bed_jump_p95_m": np.nan,
            "modern_margin_bed_jump_max_m": np.nan,
        }
    jumps = np.concatenate(values)
    jumps = jumps[np.isfinite(jumps)]
    if not jumps.size:
        return {
            "modern_margin_edge_count": 0,
            "modern_margin_bed_jump_median_m": np.nan,
            "modern_margin_bed_jump_p95_m": np.nan,
            "modern_margin_bed_jump_max_m": np.nan,
        }
    return {
        "modern_margin_edge_count": int(jumps.size),
        "modern_margin_bed_jump_median_m": float(np.median(jumps)),
        "modern_margin_bed_jump_p95_m": float(np.percentile(jumps, 95)),
        "modern_margin_bed_jump_max_m": float(np.max(jumps)),
    }


def _modern_margin_thickness(
    topo_m: np.ndarray,
    bed_m: np.ndarray,
    modern_mask: np.ndarray,
    lia_mask: np.ndarray,
) -> dict[str, float | int]:
    """Residual inversion thickness on the modern-mask boundary."""
    boundary = modern_mask & ~binary_erosion(modern_mask, border_value=0)
    boundary &= lia_mask
    values = np.maximum(topo_m[boundary] - bed_m[boundary], 0.0)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "modern_margin_cell_count": 0,
            "modern_margin_thickness_median_m": np.nan,
            "modern_margin_thickness_p95_m": np.nan,
            "modern_margin_thickness_max_m": np.nan,
        }
    return {
        "modern_margin_cell_count": int(values.size),
        "modern_margin_thickness_median_m": float(np.median(values)),
        "modern_margin_thickness_p95_m": float(np.percentile(values, 95)),
        "modern_margin_thickness_max_m": float(np.max(values)),
    }


def preflight_one(
    rgi_id: str,
    lia_unique_id: int,
    lia_geometry,
    lia_crs,
    gdir_dir: Path,
    bed_scenarios: list[str],
) -> dict:
    row: dict = {
        "rgi_id": rgi_id,
        "lia_unique_id": int(lia_unique_id),
        "status": "ok",
        "polygon_area_km2": float(
            gpd.GeoSeries([lia_geometry], crs=lia_crs).to_crs("ESRI:102025").area.iloc[0]
            / 1e6
        ),
        "grid_crs": "",
        "flowline_count": np.nan,
        "lia_centerline_status": "not-built-modern-flowlines-provenance-only",
        "boundary_status": "unknown-boundary-review",
    }
    try:
        inputs, _ = load_flowline_inputs(gdir_dir)
        row["flowline_count"] = len(inputs)
    except (FileNotFoundError, ValueError) as exc:
        row.update(status="missing-flowline-input", failure_reason=str(exc))
        return row

    try:
        grid = json.loads((gdir_dir / "glacier_grid.json").read_text(encoding="utf-8"))
        grid_crs = CRS.from_user_input(grid["proj"])
        row["grid_crs"] = str(grid_crs)
        x0, y0 = map(float, grid["x0y0"])
        dx, dy = map(float, grid["dxdy"])
        lia_grid = gpd.GeoSeries([lia_geometry], crs=lia_crs).to_crs(grid_crs).iloc[0]
    except (FileNotFoundError, ValueError) as exc:
        row.update(status="missing-grid", failure_reason=str(exc))
        return row

    with xr.open_dataset(gdir_dir / "gridded_data.nc") as dataset:
        topo_m = np.asarray(dataset["topo"].values, dtype=float)

    masks: list[np.ndarray] = []
    polygon_masks: list[np.ndarray] = []
    for scenario in bed_scenarios:
        try:
            bed = load_gridded_bed(gdir_dir, scenario)
        except (FileNotFoundError, ValueError) as exc:
            row[f"{scenario}_status"] = "missing-bed-scenario"
            row[f"{scenario}_failure_reason"] = str(exc)
            row["status"] = "missing-bed-scenario"
            continue
        nrow, ncol = bed.shape
        transform = Affine(dx, 0.0, x0 - 0.5 * dx, 0.0, dy, y0 - 0.5 * dy)
        corners = [
            transform * (0, 0),
            transform * (ncol, 0),
            transform * (ncol, nrow),
            transform * (0, nrow),
        ]
        xs, ys = zip(*corners)
        grid_footprint = box(min(xs), min(ys), max(xs), max(ys))
        polygon_grid_fraction = float(
            lia_grid.intersection(grid_footprint).area / lia_grid.area
        ) if lia_grid.area > 0 else 0.0
        inside = geometry_mask(
            [lia_grid.__geo_interface__],
            out_shape=(nrow, ncol),
            transform=transform,
            invert=True,
            all_touched=True,
        )
        finite = np.isfinite(bed.bed_m)
        covered = inside & finite
        fraction = float(covered.sum() / inside.sum()) if inside.any() else 0.0
        masks.append(covered)
        polygon_masks.append(inside)
        row[f"{scenario}_grid_shape"] = f"{nrow}x{ncol}"
        row[f"{scenario}_polygon_grid_coverage_fraction"] = polygon_grid_fraction
        row[f"{scenario}_lia_pixels"] = int(inside.sum())
        row[f"{scenario}_finite_bed_pixels"] = int(covered.sum())
        row[f"{scenario}_coverage_fraction"] = fraction
        row[f"{scenario}_lia_overlaps_modern_mask_fraction"] = float(
            (inside & bed.glacier_mask).sum() / inside.sum()
        ) if inside.any() else np.nan
        row.update(
            {
                f"{scenario}_{key}": value
                for key, value in _modern_margin_bed_jumps(
                    bed.bed_m, bed.glacier_mask, inside
                ).items()
            }
        )
        row.update(
            {
                f"{scenario}_{key}": value
                for key, value in _modern_margin_thickness(
                    topo_m, bed.bed_m, bed.glacier_mask, inside
                ).items()
            }
        )
        if fraction < 1.0 or polygon_grid_fraction < 0.999:
            row["status"] = "bed-coverage-review"

    if len(masks) == len(bed_scenarios) and masks:
        expected = np.logical_or.reduce(polygon_masks)
        both = np.logical_and.reduce(masks)
        row["both_bed_coverage_fraction"] = float(
            both.sum() / expected.sum()
        ) if expected.any() else 0.0
        row["bed_scenario_agreement_fraction"] = float(
            np.logical_and.reduce(masks).sum() / np.logical_or.reduce(masks).sum()
        ) if np.logical_or.reduce(masks).any() else np.nan
    return row


def run(config: dict) -> pd.DataFrame:
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    root = resolve(config["stage0_gdir_root"])
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    if "UniqueID" not in lia.columns:
        raise ValueError("LIA geometry must contain UniqueID")
    lia_by_id = lia.set_index("UniqueID")
    rows: list[dict] = []
    for item in selection.itertuples(index=False):
        uid = int(item.lia_unique_id)
        if uid not in lia_by_id.index:
            rows.append({"rgi_id": item.main_rgi_id, "lia_unique_id": uid, "status": "missing-lia-polygon"})
            continue
        rows.append(
            preflight_one(
                str(item.main_rgi_id), uid, lia_by_id.loc[uid].geometry,
                lia.crs, _gdir_path(root, str(item.main_rgi_id)), config["bed_scenarios"]
            )
        )
    frame = pd.DataFrame(rows)
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "lia_geometry_preflight_v0.csv", index=False)
    (output_dir / "lia_geometry_preflight_summary.json").write_text(
        json.dumps({
            "systems": int(len(frame)),
            "status_counts": frame["status"].value_counts(dropna=False).to_dict(),
            "bed_scenarios": config["bed_scenarios"],
            "centerline_policy": "modern-flowlines-provenance-only; LIA centerlines not built",
            "boundary_policy": config.get("boundary_policy", {}),
        }, indent=2), encoding="utf-8"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    frame = run(json.loads(args.config.read_text(encoding="utf-8")))
    print(frame["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

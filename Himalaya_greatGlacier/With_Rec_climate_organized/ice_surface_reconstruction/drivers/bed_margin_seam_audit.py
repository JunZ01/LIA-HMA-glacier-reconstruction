"""Audit the bed seam where a modern inversion meets exposed terrain.

This driver is diagnostic only. It never edits ``gridded_data.nc``. The raw
bed is decomposed edge by edge as::

    bed_out - bed_in = (topo_out - topo_in) + thickness_in

Candidate tapers are evaluated as registered structural scenarios. They taper
the inverted thickness to zero *inside* the modern outline because the exposed
terrain outside the outline is observed bed, not an unknown continuation of
the inversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt

from ..paths import resolve
from .gridded_bed import load_gridded_bed
from .lia_geometry_preflight import _gdir_path


def _paired_values(
    values: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    return ((values[:, :-1], values[:, 1:]), (values[:-1, :], values[1:, :]))


def margin_edge_components(
    topo_m: np.ndarray,
    thickness_m: np.ndarray,
    modern_mask: np.ndarray,
    domain_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return signed, outward-oriented components for modern-margin edges."""
    terrain_parts: list[np.ndarray] = []
    thickness_parts: list[np.ndarray] = []
    bed_parts: list[np.ndarray] = []

    topo_pairs = _paired_values(topo_m)
    thickness_pairs = _paired_values(thickness_m)
    mask_pairs = _paired_values(modern_mask)
    domain_pairs = _paired_values(domain_mask)
    for (topo_a, topo_b), (h_a, h_b), (mask_a, mask_b), (dom_a, dom_b) in zip(
        topo_pairs, thickness_pairs, mask_pairs, domain_pairs
    ):
        crossing = (mask_a != mask_b) & dom_a & dom_b
        finite = (
            np.isfinite(topo_a) & np.isfinite(topo_b)
            & np.isfinite(h_a) & np.isfinite(h_b)
        )
        crossing &= finite
        if not crossing.any():
            continue
        a_inside = mask_a[crossing]
        topo_in = np.where(a_inside, topo_a[crossing], topo_b[crossing])
        topo_out = np.where(a_inside, topo_b[crossing], topo_a[crossing])
        h_in = np.where(a_inside, h_a[crossing], h_b[crossing])
        terrain = topo_out - topo_in
        terrain_parts.append(terrain)
        thickness_parts.append(h_in)
        bed_parts.append(terrain + h_in)

    if not terrain_parts:
        empty = np.asarray([], dtype=float)
        return {
            "terrain_step_outward_m": empty,
            "thickness_cutoff_m": empty,
            "bed_step_outward_m": empty,
        }
    return {
        "terrain_step_outward_m": np.concatenate(terrain_parts),
        "thickness_cutoff_m": np.concatenate(thickness_parts),
        "bed_step_outward_m": np.concatenate(bed_parts),
    }


def taper_thickness_inside_margin(
    thickness_m: np.ndarray,
    modern_mask: np.ndarray,
    map_dx_m: float,
    taper_width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Taper thickness from zero at the first interior cell to full inland."""
    if taper_width_m <= 0:
        weight = modern_mask.astype(float)
        return np.where(modern_mask, thickness_m, 0.0), weight
    distance_cells = distance_transform_edt(modern_mask)
    width_cells = float(taper_width_m) / float(map_dx_m)
    weight = np.clip((distance_cells - 1.0) / width_cells, 0.0, 1.0)
    weight = np.where(modern_mask, weight, 0.0)
    return np.where(modern_mask, thickness_m * weight, 0.0), weight


def _percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else np.nan


def _edge_differences(
    values: np.ndarray, domain_mask: np.ndarray, selected_cells: np.ndarray
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for (a, b), (dom_a, dom_b), (sel_a, sel_b) in zip(
        _paired_values(values),
        _paired_values(domain_mask),
        _paired_values(selected_cells),
    ):
        use = dom_a & dom_b & (sel_a | sel_b) & np.isfinite(a) & np.isfinite(b)
        if use.any():
            parts.append(np.abs(b[use] - a[use]))
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def audit_scenario(
    topo_m: np.ndarray,
    bed_m: np.ndarray,
    modern_mask: np.ndarray,
    lia_mask: np.ndarray,
    map_dx_m: float,
    taper_widths_m: list[float],
) -> list[dict[str, float | int | str]]:
    """Evaluate raw and candidate-taper seam behavior for one bed scenario."""
    raw_h = np.where(modern_mask, np.maximum(topo_m - bed_m, 0.0), 0.0)
    components = margin_edge_components(topo_m, raw_h, modern_mask, lia_mask)
    terrain = components["terrain_step_outward_m"]
    cutoff = components["thickness_cutoff_m"]
    raw_bed_step = components["bed_step_outward_m"]
    raw_volume = float(np.sum(raw_h[modern_mask & lia_mask]))
    closure = raw_bed_step - terrain - cutoff

    common: dict[str, float | int] = {
        "boundary_edge_count": int(raw_bed_step.size),
        "terrain_step_abs_median_m": _percentile(np.abs(terrain), 50),
        "terrain_step_abs_p95_m": _percentile(np.abs(terrain), 95),
        "thickness_cutoff_median_m": _percentile(cutoff, 50),
        "thickness_cutoff_p95_m": _percentile(cutoff, 95),
        "raw_bed_step_abs_median_m": _percentile(np.abs(raw_bed_step), 50),
        "raw_bed_step_abs_p95_m": _percentile(np.abs(raw_bed_step), 95),
        "thickness_dominates_terrain_fraction": (
            float(np.mean(cutoff > np.abs(terrain))) if cutoff.size else np.nan
        ),
        "raw_bed_steps_up_outward_fraction": (
            float(np.mean(raw_bed_step > 0.0)) if raw_bed_step.size else np.nan
        ),
        "decomposition_closure_max_abs_m": (
            float(np.max(np.abs(closure))) if closure.size else np.nan
        ),
    }

    rows: list[dict[str, float | int | str]] = []
    distance_cells = distance_transform_edt(modern_mask)
    for taper_width_m in [0.0, *taper_widths_m]:
        tapered_h, weight = taper_thickness_inside_margin(
            raw_h, modern_mask, map_dx_m, taper_width_m
        )
        if taper_width_m <= 0:
            selected = modern_mask & (distance_cells <= 1.0)
            label = "raw-hard-splice"
        else:
            width_cells = taper_width_m / map_dx_m
            selected = modern_mask & (distance_cells <= 1.0 + width_cells)
            label = f"inward-taper-{taper_width_m:g}m"
        raw_bed = topo_m - raw_h
        tapered_bed = topo_m - tapered_h
        raw_transition = _edge_differences(raw_bed, lia_mask, selected)
        tapered_transition = _edge_differences(tapered_bed, lia_mask, selected)
        raw_h_transition = _edge_differences(raw_h, lia_mask, selected)
        tapered_h_transition = _edge_differences(tapered_h, lia_mask, selected)
        tapered_components = margin_edge_components(
            topo_m, tapered_h, modern_mask, lia_mask
        )
        tapered_boundary_step = tapered_components["bed_step_outward_m"]
        tapered_volume = float(np.sum(tapered_h[modern_mask & lia_mask]))
        row: dict[str, float | int | str] = {
            **common,
            "scenario": label,
            "taper_width_m": float(taper_width_m),
            "taper_width_pixels": float(taper_width_m / map_dx_m),
            "transition_edge_count": int(tapered_transition.size),
            "raw_transition_bed_step_abs_p95_m": _percentile(raw_transition, 95),
            "candidate_transition_bed_step_abs_p95_m": _percentile(
                tapered_transition, 95
            ),
            "raw_transition_thickness_step_abs_p95_m": _percentile(
                raw_h_transition, 95
            ),
            "candidate_transition_thickness_step_abs_p95_m": _percentile(
                tapered_h_transition, 95
            ),
            "candidate_boundary_bed_step_abs_p95_m": _percentile(
                np.abs(tapered_boundary_step), 95
            ),
            "retained_lia_modern_volume_fraction": (
                tapered_volume / raw_volume if raw_volume > 0 else np.nan
            ),
            "modified_lia_modern_cell_fraction": float(
                np.mean(weight[modern_mask & lia_mask] < 1.0)
            ) if np.any(modern_mask & lia_mask) else np.nan,
            "max_bed_adjustment_m": float(np.max(raw_h - tapered_h)),
        }
        rows.append(row)
    return rows


def run(config: dict) -> pd.DataFrame:
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    gdir_root = resolve(config["stage0_gdir_root"])
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    lia_by_id = lia.set_index("UniqueID")
    audit_config = config.get("bed_margin_seam_audit", {})
    widths = [float(value) for value in audit_config.get("taper_widths_m", [])]
    rows: list[dict] = []

    for item in selection.itertuples(index=False):
        rgi_id = str(item.main_rgi_id)
        uid = int(item.lia_unique_id)
        gdir_dir = _gdir_path(gdir_root, rgi_id)
        grid = json.loads((gdir_dir / "glacier_grid.json").read_text())
        grid_crs = CRS.from_user_input(grid["proj"])
        x0, y0 = map(float, grid["x0y0"])
        dx, dy = map(float, grid["dxdy"])
        with xr.open_dataset(gdir_dir / "gridded_data.nc") as dataset:
            topo_m = np.asarray(dataset["topo"].values, dtype=float)
        transform = Affine(dx, 0.0, x0 - 0.5 * dx, 0.0, dy, y0 - 0.5 * dy)
        lia_geometry = gpd.GeoSeries(
            [lia_by_id.loc[uid].geometry], crs=lia.crs
        ).to_crs(grid_crs).iloc[0]
        lia_mask = geometry_mask(
            [lia_geometry.__geo_interface__],
            out_shape=topo_m.shape,
            transform=transform,
            invert=True,
            all_touched=True,
        )
        for bed_scenario in config["bed_scenarios"]:
            try:
                bed = load_gridded_bed(gdir_dir, bed_scenario)
            except (FileNotFoundError, ValueError) as exc:
                rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "bed_scenario": bed_scenario,
                    "status": "missing-bed-scenario",
                    "failure_reason": str(exc),
                })
                continue
            for result in audit_scenario(
                topo_m, bed.bed_m, bed.glacier_mask, lia_mask,
                bed.map_dx_m, widths,
            ):
                rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "bed_scenario": bed_scenario,
                    "status": "ok",
                    **result,
                })

    frame = pd.DataFrame(rows)
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "bed_margin_seam_audit_v0.csv", index=False)
    valid = frame[frame["status"] == "ok"].copy()
    aggregate_columns = [
        "terrain_step_abs_p95_m",
        "thickness_cutoff_p95_m",
        "raw_bed_step_abs_p95_m",
        "candidate_boundary_bed_step_abs_p95_m",
        "candidate_transition_bed_step_abs_p95_m",
        "retained_lia_modern_volume_fraction",
        "modified_lia_modern_cell_fraction",
    ]
    aggregates: list[dict] = []
    for (bed_scenario, scenario), group in valid.groupby(
        ["bed_scenario", "scenario"], sort=True
    ):
        record: dict[str, object] = {
            "bed_scenario": bed_scenario,
            "scenario": scenario,
            "systems": int(group["rgi_id"].nunique()),
        }
        for column in aggregate_columns:
            record[f"median_{column}"] = float(group[column].median())
            record[f"max_{column}"] = float(group[column].max())
        aggregates.append(record)
    summary = {
        "status": "diagnostic-only-unfrozen",
        "systems_requested": int(len(selection)),
        "bed_scenarios": list(config["bed_scenarios"]),
        "taper_widths_m": widths,
        "decomposition": "bed_step_outward = topo_step_outward + inside_thickness",
        "candidate_policy": (
            "inward thickness taper; no gdir or source raster is modified"
        ),
        "status_counts": frame["status"].value_counts(dropna=False).to_dict(),
        "aggregates": aggregates,
    }
    (output_dir / "bed_margin_seam_audit_summary_v0.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frame = run(config)
    print(frame["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

"""Auditable v1 follow-up for the calibrated LIA failure dossier.

This driver is deliberately separate from the v0 diagnostic output.  It adds
the minimum evidence needed before routing the pilot members toward production,
input review, or model-domain review.  It never edits a gdir, clips a negative
thickness, or promotes a counterfactual tau scan to production.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LightSource
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.stats import linregress, t as student_t
from shapely.geometry import Point

from ..paths import resolve
from .gridded_bed import load_gridded_bed
from .lia_3d_zero_boundary_pilot import _gdir_path
from .lia_centerline_pilot import _configure_oggm
from .diagnose_lia_failures import (
    _build_context,
    _failure_grid_position,
    _scenario,
    replay_branch,
)
from ..core import solve_flowline_network


TAU_THRESHOLDS = (1.2, 1.3, 1.5)
SEAM_ZONE_PIXELS = 1.0
SEAM_NEARBY_PIXELS = 2.0
SEAM_THICKNESS_DOMINANCE = 0.70
SEAM_CLOSURE_TOLERANCE_M = 1.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260723
MODERN_VOLUME_DENOMINATOR_FLOOR = 1e-3

THICKNESS_VAR = {
    "farinotti2019": "distributed_thickness_f19_fs0_full_v0",
    "millan2022": "distributed_thickness_m22_fs0_full_v0",
}


def _gdir_arrays(config: dict, rgi_id: str, bed_scenario: str) -> dict:
    gdir = _gdir_path(resolve(config["stage0_gdir_root"]), rgi_id)
    grid = json.loads((gdir / "glacier_grid.json").read_text())
    with xr.open_dataset(gdir / "gridded_data.nc") as ds:
        topo = np.asarray(ds["topo"].values, dtype=float)
        mask = np.asarray(ds["glacier_mask"].values, dtype=bool)
        var = THICKNESS_VAR[bed_scenario]
        thickness = np.asarray(ds[var].values, dtype=float)
    return {
        "gdir": gdir,
        "grid": grid,
        "topo_m": topo,
        "modern_mask": mask,
        "thickness_m": thickness,
        "map_dx_m": abs(float(grid["dxdy"][0])),
    }


def _signed_distance_outside_positive(mask: np.ndarray, row: int, col: int, dx: float) -> float:
    """Signed pixel-center distance: inside negative, outside positive."""
    if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
        return float("nan")
    if bool(mask[row, col]):
        return -float(distance_transform_edt(mask)[row, col] * dx)
    return float(distance_transform_edt(~mask)[row, col] * dx)


def _nearest_margin_edge(
    topo: np.ndarray,
    thickness: np.ndarray,
    mask: np.ndarray,
    row: int,
    col: int,
    radius: int = 4,
) -> dict:
    """Find the nearest raster mask edge and decompose its bed step."""
    candidates: list[tuple[float, dict]] = []
    r0 = max(0, row - radius)
    r1 = min(mask.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(mask.shape[1], col + radius + 1)
    for r in range(r0, r1):
        for c in range(c0, c1 - 1):
            if bool(mask[r, c]) == bool(mask[r, c + 1]):
                continue
            inside = (r, c) if mask[r, c] else (r, c + 1)
            outside = (r, c + 1) if mask[r, c] else (r, c)
            topo_in = float(topo[inside])
            topo_out = float(topo[outside])
            h_in = float(thickness[inside])
            terrain = topo_out - topo_in
            bed_step = terrain + h_in
            distance = math.hypot((r + 0.0) - row, (c + 0.5) - col)
            candidates.append((distance, {
                "edge_row": r,
                "edge_col": c,
                "terrain_step_m": terrain,
                "edge_thickness_m": h_in,
                "spliced_bed_step_m": bed_step,
                "decomposition_closure_m": bed_step - terrain - h_in,
                "edge_orientation": "vertical",
            }))
    for r in range(r0, r1 - 1):
        for c in range(c0, c1):
            if bool(mask[r, c]) == bool(mask[r + 1, c]):
                continue
            inside = (r, c) if mask[r, c] else (r + 1, c)
            outside = (r + 1, c) if mask[r, c] else (r, c)
            topo_in = float(topo[inside])
            topo_out = float(topo[outside])
            h_in = float(thickness[inside])
            terrain = topo_out - topo_in
            bed_step = terrain + h_in
            distance = math.hypot((r + 0.5) - row, c - col)
            candidates.append((distance, {
                "edge_row": r,
                "edge_col": c,
                "terrain_step_m": terrain,
                "edge_thickness_m": h_in,
                "spliced_bed_step_m": bed_step,
                "decomposition_closure_m": bed_step - terrain - h_in,
                "edge_orientation": "horizontal",
            }))
    if not candidates:
        return {
            "seam_edge_found": False,
            "seam_edge_distance_pixels": np.nan,
        }
    distance, payload = min(candidates, key=lambda item: item[0])
    return {
        "seam_edge_found": True,
        "seam_edge_distance_pixels": float(distance),
        **payload,
        "edge_thickness_dominance_fraction": abs(float(payload["edge_thickness_m"]))
        / max(abs(float(payload["spliced_bed_step_m"])), 1.0),
    }


def _task1_refinement(config: dict, v0: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    tau_bounds = config.get("modern_tau_calibration", {}).get("tau_bounds_kpa", [40.0, 180.0])
    for item in v0.itertuples(index=False):
        data = _gdir_arrays(config, str(item.rgi_id), str(item.bed_scenario))
        distance = _signed_distance_outside_positive(
            data["modern_mask"], int(item.grid_row), int(item.grid_col), data["map_dx_m"]
        )
        seam_zone = abs(distance) <= SEAM_ZONE_PIXELS * data["map_dx_m"]
        cal = calibration[
            (calibration["rgi_id"].astype(str) == str(item.rgi_id))
            & (calibration["bed_scenario"].astype(str) == str(item.bed_scenario))
        ]
        cal_row = cal.iloc[0] if not cal.empty else None
        tau_min = float(item.feasible_tau_min_kpa) if pd.notna(item.feasible_tau_min_kpa) else np.nan
        tau_upper = float(item.modern_near_optimal_tau_max_kpa) if pd.notna(item.modern_near_optimal_tau_max_kpa) else np.nan
        ratio = tau_min / tau_upper if np.isfinite(tau_min) and np.isfinite(tau_upper) and tau_upper > 0 else np.nan
        pixel_center_in_modern_mask = bool(
            data["modern_mask"][int(item.grid_row), int(item.grid_col)]
        )
        interval_censored = False
        if cal_row is not None:
            near_min = float(cal_row.near_optimal_tau_min_kpa)
            near_max = float(cal_row.near_optimal_tau_max_kpa)
            interval_censored = (
                abs(near_min - float(tau_bounds[0])) < 1e-6
                or abs(near_max - float(tau_bounds[1])) < 1e-6
            )
        base = {
            "member_id": f"{item.rgi_id}__{item.bed_scenario}",
            "rgi_id": str(item.rgi_id),
            "lia_unique_id": int(item.lia_unique_id),
            "bed_scenario": str(item.bed_scenario),
            "original_diagnostic_class": str(item.diagnostic_class),
            "failed_branch_id": str(item.failed_branch_id),
            "native_branch_failure": bool(item.native_branch_failure),
            "pixel_center_in_modern_mask": pixel_center_in_modern_mask,
            "fail_in_modern_mask": pixel_center_in_modern_mask,
            "dist_to_modern_margin_m": float(distance),
            "seam_zone": bool(seam_zone),
            "dist_to_lia_boundary_m": float(item.distance_to_lia_boundary_m),
            "scan_min_feasible_tau_kpa": tau_min,
            "modern_near_optimal_tau_upper_kpa": tau_upper,
            "tau_excess_ratio": ratio,
            "calibration_interval_censored": interval_censored,
            "tau_feasibility_status": "feasible-in-scan" if np.isfinite(tau_min) else "no-feasible-in-scan",
        }
        for threshold in TAU_THRESHOLDS:
            if not bool(item.native_branch_failure):
                label = "connector-not-native-tau-diagnostic"
            elif seam_zone:
                label = "seam-suspect"
            elif not np.isfinite(ratio):
                label = "no-feasible-in-scan"
            elif interval_censored:
                label = "calibration-boundary-review"
            elif ratio <= threshold and base["fail_in_modern_mask"]:
                label = "tau-transfer-suspect"
            elif ratio <= threshold:
                label = "ambiguous-tau-or-bed"
            elif not base["fail_in_modern_mask"]:
                label = "bed-input-suspect"
            else:
                label = "geometry-suspect"
            base[f"tau_subclass_at_{threshold:g}"] = label
        rows.append(base)
    return pd.DataFrame(rows)


def _load_failure_branch(config: dict, row, output_dir: Path, cache: dict):
    key = (str(row.rgi_id), int(row.lia_unique_id), str(row.bed_scenario))
    if key not in cache:
        temp = output_dir / "_centerline_tmp"
        temp.mkdir(parents=True, exist_ok=True)
        cache[key] = _build_context(config, *key, temp)
    return cache[key]


def _branch_coords_xy(branch_id: str, build, inputs) -> np.ndarray:
    if branch_id.startswith("conn:"):
        audit = next(item for item in build.connector_audit if item.connector_branch_id == branch_id)
        return np.asarray([(col, row) for row, col in audit.pixels_rc], dtype=float)
    provenance = next(item for item in build.provenance if item.branch_id == branch_id)
    source = next(item for item in inputs if item.flowline_index == provenance.source_flowline_index)
    indices = np.arange(
        provenance.native_index_end,
        provenance.native_index_start - 1,
        -1,
        dtype=int,
    )
    return np.asarray(source.coords_xy[indices], dtype=float)


def _boundary_crossings(mask: np.ndarray, coords_xy: np.ndarray, distance_m: np.ndarray) -> list[float]:
    sampled = map_coordinates(
        mask.astype(float),
        np.vstack([coords_xy[:, 1], coords_xy[:, 0]]),
        order=0,
        mode="nearest",
    ) >= 0.5
    indices = np.flatnonzero(sampled[1:] != sampled[:-1])
    return [float(0.5 * (distance_m[i] + distance_m[i + 1])) for i in indices]


def _make_profile_review(config: dict, row, output_dir: Path, cache: dict) -> dict:
    """Create a reproducible four-panel human-review bundle."""
    context = _load_failure_branch(config, row, output_dir, cache)
    gdir, grid, grid_crs, geometry, flowlines, lia_mask, bed, inputs, build = context
    solution = solve_flowline_network(build.branches, _scenario(float(row.tau_kpa)), outlet_node_ids=build.outlet_node_ids)
    branch_id = str(solution.diagnostics.get("failed_branch_id", row.failed_branch_id))
    branch = next(item for item in build.branches if item.branch_id == branch_id)
    branch_coords = _branch_coords_xy(branch_id, build, inputs)
    failed_index = int(row.failed_index)
    inherited = solution.node_surface_m.get(branch.downstream_node_id, float(branch.bed_elevation_m[0]))
    replay, _, _ = replay_branch(branch, inherited, _scenario(float(row.tau_kpa)))
    x = np.asarray(branch.distance_m, dtype=float)
    z = np.asarray(branch.bed_elevation_m, dtype=float)
    center = float(x[failed_index])
    segment_slope_deg = np.degrees(np.arctan2(np.diff(z), np.diff(x)))
    slope_before = float(segment_slope_deg[failed_index - 2]) if failed_index >= 2 else np.nan
    slope_at = float(segment_slope_deg[failed_index - 1]) if failed_index >= 1 else np.nan
    slope_after = float(segment_slope_deg[failed_index]) if failed_index < len(segment_slope_deg) else np.nan
    local_turning_point = bool(
        np.isfinite(slope_before)
        and np.isfinite(slope_at)
        and np.isfinite(slope_after)
        and abs(slope_at) >= abs(slope_before) + 5.0
        and abs(slope_at) > abs(slope_after)
    )
    window = (x >= center - 2000.0) & (x <= center + 2000.0)
    xw, zw = x[window], z[window]
    slope_deg = np.degrees(np.arctan(np.gradient(zw, xw))) if len(xw) > 2 else np.full(len(xw), np.nan)
    coords = np.asarray([(int(row.grid_col), int(row.grid_row))], dtype=float)
    topo = _gdir_arrays(config, str(row.rgi_id), str(row.bed_scenario))["topo_m"]
    dx = _gdir_arrays(config, str(row.rgi_id), str(row.bed_scenario))["map_dx_m"]
    grid_row, grid_col = int(row.grid_row), int(row.grid_col)
    tangent = np.array([1.0, 0.0])
    if failed_index > 0 and failed_index + 1 < len(branch_coords):
        tangent = branch_coords[failed_index + 1] - branch_coords[failed_index - 1]
        norm = np.linalg.norm(tangent)
        if norm > 0:
            tangent = tangent / norm
    normal = np.array([-tangent[1], tangent[0]])
    offsets_m = np.linspace(-500.0, 500.0, 101)
    sample_xy = np.column_stack([
        float(grid_col) + normal[0] * offsets_m / dx,
        float(grid_row) + normal[1] * offsets_m / dx,
    ])
    cross = map_coordinates(topo, np.vstack([sample_xy[:, 1], sample_xy[:, 0]]), order=1, mode="nearest")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(xw - center, zw, color="#704214", label="bed")
    replay_ok = replay[
        (replay["distance_m"] >= center - 2000.0)
        & (replay["distance_m"] <= center)
    ]
    axes[0, 0].plot(replay_ok["distance_m"] - center, replay_ok["surface_m"], color="#1769aa", label="PP surface")
    axes[0, 0].axvline(0, color="#b00020", linestyle="--", label="failure")
    modern_mask = _gdir_arrays(config, str(row.rgi_id), str(row.bed_scenario))["modern_mask"]
    for value in _boundary_crossings(modern_mask, branch_coords, x):
        if center - 2000.0 <= value <= center + 2000.0:
            axes[0, 0].axvline(value - center, color="#38a169", linestyle=":", alpha=0.8, label="modern boundary")
    for value in _boundary_crossings(lia_mask, branch_coords, x):
        if center - 2000.0 <= value <= center + 2000.0:
            axes[0, 0].axvline(value - center, color="#e6a700", linestyle=":", alpha=0.8, label="LIA boundary")
    axes[0, 0].set(title="Longitudinal bed / PP replay", xlabel="distance from failure (m)", ylabel="elevation (m)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(xw - center, slope_deg, color="#333333")
    axes[0, 1].axvline(0, color="#b00020", linestyle="--")
    axes[0, 1].set(title="Local bed slope", xlabel="distance from failure (m)", ylabel="slope (deg)")
    axes[1, 0].plot(offsets_m, cross, color="#704214")
    axes[1, 0].axvline(0, color="#b00020", linestyle="--")
    axes[1, 0].set(title="Cross-section at failure", xlabel="normal offset (m)", ylabel="DEM elevation (m)")
    half = int(math.ceil(2000.0 / dx))
    r0, r1 = max(0, grid_row - half), min(topo.shape[0], grid_row + half + 1)
    c0, c1 = max(0, grid_col - half), min(topo.shape[1], grid_col + half + 1)
    local = topo[r0:r1, c0:c1]
    rgb = LightSource(azdeg=315, altdeg=45).shade(local, cmap=plt.get_cmap("gray"), blend_mode="overlay")
    axes[1, 1].imshow(rgb, origin="upper")
    axes[1, 1].scatter([grid_col - c0], [grid_row - r0], color="#b00020", s=25)
    axes[1, 1].contour(lia_mask[r0:r1, c0:c1], levels=[0.5], colors=["#e6a700"], linewidths=1.2)
    axes[1, 1].contour(modern_mask[r0:r1, c0:c1], levels=[0.5], colors=["#38a169"], linewidths=1.0)
    local_branch = (x >= center - 2000.0) & (x <= center + 2000.0)
    axes[1, 1].plot(branch_coords[local_branch, 0] - c0, branch_coords[local_branch, 1] - r0, color="#1769aa", linewidth=1.0)
    axes[1, 1].set(title="2 km DEM hillshade / masks", xlabel="grid col", ylabel="grid row")
    fig.suptitle(f"{row.rgi_id} / {row.bed_scenario} / {branch_id}")
    fig.tight_layout()
    stem = f"{row.rgi_id}_{row.bed_scenario}"
    png = output_dir / "moraine_profile_review_v0" / f"{stem}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    review_path = png.parent / f"{stem}.json"
    prior = {}
    if review_path.exists():
        prior = json.loads(review_path.read_text(encoding="utf-8"))
    review = {
        "member_id": stem,
        "figure": png.relative_to(output_dir).as_posix(),
        "review_status": prior.get("review_status", "pending-human-review"),
        "fixed_options": ["moraine-confirmed", "bedrock-step", "ambiguous-terrain", "not-topographic"],
        "automatic_evidence": {
            "distance_to_lia_boundary_m": float(row.distance_to_lia_boundary_m),
            "inside_modern_mask": bool(row.inside_modern_mask),
            "profile_bed_prominence_m": float(row.profile_bed_prominence_m),
            "profile_prominent_width_m": float(row.profile_prominent_width_m),
            "failed_bed_step_m": float(row.failed_bed_step_m),
            "segment_slope_before_deg": slope_before,
            "segment_slope_at_failure_deg": slope_at,
            "segment_slope_after_deg": slope_after,
            "local_bed_turning_point": local_turning_point,
        },
        "selected_option": prior.get("selected_option"),
        "selected_features": prior.get("selected_features", []),
    }
    if "review_note" in prior:
        review["review_note"] = prior["review_note"]
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    return review


def _lia_scale_covariates(config: dict, calibration: pd.DataFrame) -> pd.DataFrame:
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    lia = gpd.read_file(resolve(config["lia_geometry_path"])).set_index("UniqueID")
    centerlines = pd.read_csv(resolve(config["output_dir"]) / "lia_centerline_scenarios_v0.csv")
    rows = []
    for item in selection.itertuples(index=False):
        rgi_id, uid = str(item.main_rgi_id), int(item.lia_unique_id)
        data = _gdir_arrays(config, rgi_id, "farinotti2019")
        geometry = gpd.GeoSeries([lia.loc[uid].geometry], crs=lia.crs).to_crs(data["grid"]["proj"]).iloc[0]
        area = float(geometry.area) / 1e6
        mask = data["modern_mask"]
        topo = data["topo_m"]
        grid = data["grid"]
        from rasterio.features import geometry_mask
        from rasterio.transform import Affine
        dx, dy = map(float, grid["dxdy"])
        x0, y0 = map(float, grid["x0y0"])
        transform = Affine(dx, 0.0, x0 - 0.5 * dx, 0.0, dy, y0 - 0.5 * dy)
        lia_mask = geometry_mask([geometry.__geo_interface__], out_shape=topo.shape, transform=transform, invert=True, all_touched=True)
        vals = topo[lia_mask & np.isfinite(topo)]
        span = float(np.max(vals) - np.min(vals)) if vals.size else np.nan
        cl = centerlines[
            (centerlines.rgi_id.astype(str) == rgi_id)
            & (centerlines.lia_unique_id == uid)
            & (centerlines.elevation_field == "raw-bed-farinotti2019")
            & (centerlines.status == "ok")
        ]
        length = float(cl.iloc[0].total_length_km) if not cl.empty else np.nan
        rows.append({"rgi_id": rgi_id, "lia_unique_id": uid, "lia_area_km2": area, "lia_elevation_span_m": span, "lia_centerline_length_km": length})
    return pd.DataFrame(rows)


def _bootstrap_slope(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2:
            continue
        slopes.append(float(linregress(x[idx], y[idx]).slope))
    return (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))) if slopes else (np.nan, np.nan)


def _run_tau_scale(config: dict, calibration: pd.DataFrame, task1: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    cov = _lia_scale_covariates(config, calibration)
    cov.to_csv(output_dir / "lia_scale_covariates_v1.csv", index=False)
    data = calibration[calibration.status.astype(str).eq("identified-interior")].merge(cov, on="rgi_id", how="left")
    records = []
    for bed, group in data.groupby("bed_scenario", sort=True):
        for predictor, column in [("modern_area", "modern_area_km2"), ("modern_elevation_span", "modern_elevation_span_m")]:
            group = group[np.isfinite(group[column]) & np.isfinite(group.kappa_kpa)]
            x = np.log10(group[column].to_numpy(float)) if predictor == "modern_area" else group[column].to_numpy(float) / 1000.0
            y = group.kappa_kpa.to_numpy(float)
            fit = linregress(x, y)
            lo, hi = _bootstrap_slope(x, y, BOOTSTRAP_SEED + len(records))
            records.append({"bed_scenario": bed, "predictor": predictor, "n": len(group), "slope": fit.slope, "intercept": fit.intercept, "r2": fit.rvalue ** 2, "p_value": fit.pvalue, "slope_bootstrap_low": lo, "slope_bootstrap_high": hi, "interpretation": "indicative-only-n-small"})
    regression = pd.DataFrame(records)
    regression.to_csv(output_dir / "tau_scale_regression_v1.csv", index=False)
    member_checks = []
    shape_factor = float(config["modern_tau_calibration"]["shape_factor"])
    task_members = task1.merge(cov, on=["rgi_id", "lia_unique_id"], how="left")
    for bed, group in data.groupby("bed_scenario", sort=True):
        group = group[np.isfinite(group.modern_area_km2) & np.isfinite(group.kappa_kpa)]
        x = np.log10(group.modern_area_km2.to_numpy(float))
        y = group.kappa_kpa.to_numpy(float)
        fit = linregress(x, y)
        residual = y - (fit.intercept + fit.slope * x)
        residual_se = float(np.sqrt(np.sum(residual ** 2) / max(len(x) - 2, 1)))
        sxx = float(np.sum((x - np.mean(x)) ** 2))
        critical = float(student_t.ppf(0.975, max(len(x) - 2, 1)))
        members = task_members[
            (task_members.bed_scenario == bed)
            & np.isfinite(task_members.scan_min_feasible_tau_kpa)
            & np.isfinite(task_members.lia_area_km2)
        ]
        for item in members.itertuples(index=False):
            x0 = math.log10(float(item.lia_area_km2))
            prediction = float(fit.intercept + fit.slope * x0)
            half = critical * residual_se * math.sqrt(
                1.0 + 1.0 / len(x) + ((x0 - float(np.mean(x))) ** 2 / max(sxx, 1e-12))
            )
            feasible_kappa = float(item.scan_min_feasible_tau_kpa) / shape_factor
            in_prediction = prediction - half <= feasible_kappa <= prediction + half
            member_checks.append({
                "member_id": item.member_id,
                "rgi_id": item.rgi_id,
                "bed_scenario": bed,
                "lia_area_km2": float(item.lia_area_km2),
                "scan_min_feasible_kappa_kpa": feasible_kappa,
                "predicted_lia_scale_kappa_kpa": prediction,
                "prediction_low_kpa": prediction - half,
                "prediction_high_kpa": prediction + half,
                "feasible_kappa_inside_prediction_interval": bool(in_prediction),
                "positive_significant_area_slope": bool(fit.slope > 0 and fit.pvalue < 0.05),
                "supports_scale_transfer": bool(fit.slope > 0 and fit.pvalue < 0.05 and in_prediction),
            })
    checks = pd.DataFrame(member_checks)
    checks.to_csv(output_dir / "tau_scale_transfer_member_check_v1.csv", index=False)
    summary = {
        "status": "exploratory-not-transfer-correction",
        "records": records,
        "sample_warning": "n is approximately 11 per bed product; no scaling law is frozen.",
        "scale_transfer_supported": bool(not checks.empty and checks.supports_scale_transfer.any()),
        "supporting_member_count": int(checks.supports_scale_transfer.sum()) if not checks.empty else 0,
    }
    (output_dir / "tau_scale_regression_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return regression, summary


def _modern_volume(data: dict) -> tuple[float, int, float]:
    valid = data["modern_mask"] & np.isfinite(data["thickness_m"])
    thickness = data["thickness_m"]
    area = data["map_dx_m"] ** 2
    return float(np.sum(thickness[valid]) * area / 1e9), int(np.sum(valid)), float(np.nanmin(thickness[valid])) if np.any(valid) else np.nan


def _run_cross_bed_volume(config: dict, pilot: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    refined = (
        "flowline_integration_refinement_factor" in pilot.columns
        and pd.to_numeric(
            pilot["flowline_integration_refinement_factor"], errors="coerce"
        ).max() > 1
    )
    primary = (
        pilot[np.isfinite(pilot.head_exclusion_distance_m)].copy()
        if refined
        else pilot[pilot.head_exclusion_distance_m == 100.0].copy()
    )
    rows = []
    for rgi_id, group in primary.groupby("rgi_id"):
        statuses = group.set_index("bed_scenario").status.astype(str).to_dict()
        if statuses.get("farinotti2019") != "success" or statuses.get("millan2022") != "success":
            continue
        lia_vol = group.set_index("bed_scenario").volume_km3.astype(float)
        modern = {}
        for bed in ("farinotti2019", "millan2022"):
            modern[bed] = _modern_volume(_gdir_arrays(config, str(rgi_id), bed))[0]
        d_lia = abs(float(lia_vol["farinotti2019"]) - float(lia_vol["millan2022"])) / max(float(np.mean([lia_vol["farinotti2019"], lia_vol["millan2022"]])), 1e-12)
        d_modern = abs(modern["farinotti2019"] - modern["millan2022"]) / max(float(np.mean([modern["farinotti2019"], modern["millan2022"]])), 1e-12)
        rows.append({"rgi_id": str(rgi_id), "lia_volume_f19_km3": float(lia_vol["farinotti2019"]), "lia_volume_m22_km3": float(lia_vol["millan2022"]), "modern_volume_f19_km3": modern["farinotti2019"], "modern_volume_m22_km3": modern["millan2022"], "delta_v_lia_fraction": d_lia, "delta_v_modern_fraction": d_modern, "compensation_ratio": d_lia / d_modern if d_modern > MODERN_VOLUME_DENOMINATOR_FLOOR else np.nan, "compensation_ratio_status": "defined" if d_modern > MODERN_VOLUME_DENOMINATOR_FLOOR else "undefined-small-modern-denominator"})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "cross_bed_volume_comparison_v1.csv", index=False)
    return frame


def _build_final_routing(config: dict, pilot: pd.DataFrame, task1: pd.DataFrame, seam: pd.DataFrame, scale_summary: dict, output_dir: Path) -> pd.DataFrame:
    rows = []
    refined = (
        "flowline_integration_refinement_factor" in pilot.columns
        and pd.to_numeric(
            pilot["flowline_integration_refinement_factor"], errors="coerce"
        ).max() > 1
    )
    primary = (
        pilot[np.isfinite(pilot.head_exclusion_distance_m)].copy()
        if refined
        else pilot[pilot.head_exclusion_distance_m == 100.0].copy()
    )
    scale_support = bool(scale_summary.get("scale_transfer_supported", False))
    for item in primary.itertuples(index=False):
        member = str(getattr(item, "member_id", f"{item.rgi_id}__{item.bed_scenario}"))
        status = str(item.status)
        if status == "success":
            category = "production-pass"
            reason = "v0 1-D and 2-D status success at primary 100 m setting"
            evidence = "../lia_3d_calibrated_head_exclusion_pilot_v0/lia_3d_calibrated_results_v0.csv"
        elif status == "centerline-structure-review":
            category = "centerline-structure-review"
            reason = "centerline structure gate unresolved"
            evidence = "../lia_centerline_scenarios_v0.csv"
        elif str(item.bed_scenario) == "millan2022" and status == "tau-calibration-review":
            category = "input-missing"
            reason = "Millan thickness input/calibration unavailable"
            evidence = "../lia_geometry_preflight_v0.csv"
        else:
            q = task1[(task1.rgi_id == str(item.rgi_id)) & (task1.bed_scenario == str(item.bed_scenario))]
            s = seam[(seam.rgi_id == str(item.rgi_id)) & (seam.bed_scenario == str(item.bed_scenario))]
            qrow = q.iloc[0] if not q.empty else None
            srow = s.iloc[0] if not s.empty else None
            if qrow is not None and not bool(qrow.native_branch_failure):
                category, reason = "connector-inherited-surface-review", "connector failure retained for separate interface diagnosis"
                evidence = "lia_failure_diagnostics_v0/lia_failure_diagnostics_v0.csv"
            elif srow is not None and str(srow.seam_assessment) == "seam-artifact-supported":
                category, reason = "known-input-issue", "mask-edge bed splice is supported by component decomposition"
                evidence = "lia_failure_diagnostics_v1/seam_check_v1.csv"
            elif qrow is not None and str(qrow["tau_subclass_at_1.3"]) == "tau-transfer-suspect" and scale_support:
                category, reason = "tau-transfer-pending", "tau excess and exploratory scale analysis support a v1 transfer check"
                evidence = "lia_failure_diagnostics_v1/tau_scale_regression_v1.csv"
            else:
                category, reason = "terrain-review-pending", "human terrain classification or seam exclusion still required"
                evidence = "lia_failure_diagnostics_v1/moraine_profile_review_v0/"
        rows.append({"member_id": member, "rgi_id": str(item.rgi_id), "bed_scenario": str(item.bed_scenario), "final_category": category, "reason": reason, "evidence_pointer": evidence, "decision_date": "2026-07-23"})
    present = {(row["rgi_id"], row["bed_scenario"]) for row in rows}
    for item in pilot.itertuples(index=False):
        rgi_id = str(item.rgi_id)
        bed_scenario = str(item.bed_scenario)
        if str(item.status) == "centerline-structure-review" and (rgi_id, bed_scenario) not in present:
            rows.append({
                "member_id": f"{rgi_id}__centerline",
                "rgi_id": rgi_id,
                "bed_scenario": "structure",
                "final_category": "centerline-structure-review",
                "reason": "centerline structure gate unresolved",
                "evidence_pointer": "../lia_centerline_scenarios_v0.csv",
                "decision_date": "2026-07-23",
            })
            present.add((rgi_id, bed_scenario))
        if str(item.status) == "tau-calibration-review" and bed_scenario == "millan2022" and (rgi_id, bed_scenario) not in present:
            rows.append({
                "member_id": f"{rgi_id}__{bed_scenario}",
                "rgi_id": rgi_id,
                "bed_scenario": bed_scenario,
                "final_category": "input-missing",
                "reason": "Millan thickness input/calibration unavailable",
                "evidence_pointer": "../lia_geometry_preflight_v0.csv",
                "decision_date": "2026-07-23",
            })
            present.add((rgi_id, bed_scenario))
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "lia_failure_routing_v1.csv", index=False)
    summary = {"status": "pilot-routing-pending-human-terrain-review", "category_counts": frame.final_category.value_counts().to_dict(), "core_191_gate": "held-until-all-review-pending-members-are-resolved"}
    (output_dir / "lia_failure_routing_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frame


def run(config: dict) -> dict:
    _configure_oggm(config)
    root = resolve(config["output_dir"])
    v0_dir = root / "lia_failure_diagnostics_v0"
    output_dir = root / "lia_failure_diagnostics_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    v0 = pd.read_csv(v0_dir / "lia_failure_diagnostics_v0.csv")
    calibration = pd.read_csv(root / str(config["modern_tau_calibration"]["output_subdir"]) / "modern_tau_calibration_v0.csv")
    task1 = _task1_refinement(config, v0, calibration)
    task1.to_csv(output_dir / "tau_class_refinement_v1.csv", index=False)

    seam_rows = []
    for item in v0.itertuples(index=False):
        data = _gdir_arrays(config, str(item.rgi_id), str(item.bed_scenario))
        signed = -float(item.signed_distance_to_modern_mask_m)
        edge = _nearest_margin_edge(data["topo_m"], data["thickness_m"], data["modern_mask"], int(item.grid_row), int(item.grid_col))
        seam_distance = abs(signed)
        dominance = float(edge.get("edge_thickness_dominance_fraction", np.nan))
        supported = seam_distance <= SEAM_NEARBY_PIXELS * data["map_dx_m"] and np.isfinite(dominance) and dominance >= SEAM_THICKNESS_DOMINANCE and abs(float(edge.get("decomposition_closure_m", np.nan))) <= SEAM_CLOSURE_TOLERANCE_M
        assessment = "seam-artifact-supported" if supported else ("seam-nearby-not-dominant" if seam_distance <= SEAM_NEARBY_PIXELS * data["map_dx_m"] else "not-seam-related")
        seam_rows.append({"member_id": f"{item.rgi_id}__{item.bed_scenario}", "rgi_id": str(item.rgi_id), "bed_scenario": str(item.bed_scenario), "failed_grid_row": int(item.grid_row), "failed_grid_col": int(item.grid_col), "dist_jump_to_seam_m": seam_distance, "seam_zone": seam_distance <= SEAM_ZONE_PIXELS * data["map_dx_m"], "seam_assessment": assessment, **edge})
    seam = pd.DataFrame(seam_rows)
    seam.to_csv(output_dir / "seam_check_v1.csv", index=False)

    profile_cache: dict = {}
    candidate = task1[(task1["original_diagnostic_class"] == "modern-exposed-topography-suspect") | task1[["tau_subclass_at_1.2", "tau_subclass_at_1.3", "tau_subclass_at_1.5"]].isin(["ambiguous-tau-or-bed", "bed-input-suspect"]).any(axis=1)]
    reviews = []
    for item in candidate.itertuples(index=False):
        source_row = v0[
            (v0["rgi_id"].astype(str) == str(item.rgi_id))
            & (v0["bed_scenario"].astype(str) == str(item.bed_scenario))
        ].iloc[0]
        reviews.append(_make_profile_review(config, source_row, output_dir, profile_cache))
    (output_dir / "moraine_profile_review_v0" / "review_manifest.json").write_text(json.dumps(reviews, indent=2), encoding="utf-8")

    regression, regression_summary = _run_tau_scale(config, calibration, task1, output_dir)
    frozen = config.get("lia_3d_refined_flowline_pilot_v1", {})
    refined_path = (
        root / str(frozen.get("output_subdir", "lia_3d_refined_flowline_pilot_v1"))
        / "lia_3d_calibrated_results_v1.csv"
    )
    pilot_path = (
        refined_path
        if refined_path.exists()
        else root / str(config["lia_3d_zero_boundary_pilot"]["output_subdir"])
        / "lia_3d_calibrated_results_v0.csv"
    )
    pilot = pd.read_csv(pilot_path)
    volume = _run_cross_bed_volume(config, pilot, output_dir)
    routing = _build_final_routing(config, pilot, task1, seam, regression_summary, output_dir)
    summary = {"status": "completed-v1-diagnostic-only", "failure_members": int(len(v0)), "task1_rows": int(len(task1)), "seam_rows": int(len(seam)), "human_profile_review_count": int(len(reviews)), "routing_category_counts": routing.final_category.value_counts().to_dict(), "production_fields_unchanged": True, "no_negative_thickness_clipping": True, "regression": regression_summary, "cross_bed_volume_rows": int(len(volume))}
    (output_dir / "lia_failure_diagnostics_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()

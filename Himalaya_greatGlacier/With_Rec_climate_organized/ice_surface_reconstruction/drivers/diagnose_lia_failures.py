"""Failure-dossier driver for the calibrated LIA 3-D pilot.

This is diagnostic only. It replays the first failing perfect-plasticity step,
scans a small tau multiplier bracket, and (for Millan failures) swaps the bed
product while holding the route fixed. No production field is repaired or
clipped.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
from shapely.geometry import Point

from ..core import SurfaceScenario, solve_flowline_network
from ..paths import resolve
from .gridded_bed import load_gridded_bed
from .junction_connector import ConnectorCostWeights
from .lia_3d_zero_boundary_pilot import (
    _gdir_path,
    flowline_objects_to_inputs,
    lia_gridded_bed,
)
from .lia_centerline_pilot import _configure_oggm, compute_centerline_tree
from .oggm_network_adapter import EXPLICIT_CONNECTOR, build_network


TAU_SCAN_FACTORS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
BED_EQUALITY_TOLERANCE_M = 1.0
PROFILE_OBSTACLE_PROMINENCE_M = 20.0
PROFILE_OBSTACLE_STEP_M = 100.0


def replay_branch(branch, inherited_surface_m: float, scenario: SurfaceScenario) -> tuple[pd.DataFrame, str, int | None]:
    """Replay the core recurrence while retaining the last valid state."""
    distance = np.asarray(branch.distance_m, dtype=float)
    bed = np.asarray(branch.bed_elevation_m, dtype=float)
    surface = np.full(len(bed), np.nan, dtype=float)
    thickness = np.full(len(bed), np.nan, dtype=float)
    surface[0] = inherited_surface_m
    thickness[0] = surface[0] - bed[0]
    rows = [{
        "index": 0,
        "distance_m": float(distance[0]),
        "bed_m": float(bed[0]),
        "surface_m": float(surface[0]),
        "thickness_m": float(thickness[0]),
        "bed_step_m": np.nan,
        "discriminant": np.nan,
        "status": "start",
    }]
    if thickness[0] < -1e-8:
        return pd.DataFrame(rows), "negative-start-thickness", 0
    for i in range(1, len(bed)):
        dx = distance[i] - distance[i - 1]
        zsum = bed[i - 1] + bed[i]
        flux_term = 2.0 * dx * scenario.basal_shear_stress_pa / (
            scenario.ice_density_kg_m3 * scenario.gravity_m_s2 * scenario.shape_factor
        )
        constant = surface[i - 1] * (zsum - surface[i - 1]) - flux_term
        discriminant = zsum * zsum - 4.0 * constant
        row = {
            "index": i,
            "distance_m": float(distance[i]),
            "bed_m": float(bed[i]),
            "surface_m": np.nan,
            "thickness_m": np.nan,
            "bed_step_m": float(bed[i] - bed[i - 1]),
            "discriminant": float(discriminant),
            "status": "step",
        }
        scale = max(abs(zsum * zsum), 1.0)
        if discriminant < -1e-8 * scale:
            row["status"] = "negative-discriminant"
            rows.append(row)
            return pd.DataFrame(rows), "negative-discriminant", i
        root = max(discriminant, 0.0) ** 0.5
        surface[i] = 0.5 * (zsum + root)
        thickness[i] = surface[i] - bed[i]
        row["surface_m"] = float(surface[i])
        row["thickness_m"] = float(thickness[i])
        if thickness[i] < -1e-8:
            row["status"] = "negative-thickness"
            rows.append(row)
            return pd.DataFrame(rows), "negative-thickness", i
        rows.append(row)
    return pd.DataFrame(rows), "success", None


def _scenario(tau_kpa: float) -> SurfaceScenario:
    return SurfaceScenario("failure-diagnostic", float(tau_kpa) * 1000.0, 0.8)


def _build_context(config: dict, rgi_id: str, uid: int, bed_scenario: str, temp_root: Path):
    gdir = _gdir_path(resolve(config["stage0_gdir_root"]), rgi_id)
    grid_json = json.loads((gdir / "glacier_grid.json").read_text())
    import salem
    grid = salem.Grid.from_json(gdir / "glacier_grid.json")
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    geometry = gpd.GeoSeries(
        [lia.set_index("UniqueID").loc[uid].geometry], crs=lia.crs
    ).to_crs(grid_json["proj"]).iloc[0]
    route_bed = load_gridded_bed(gdir, "farinotti2019")
    work_dir = Path(tempfile.mkdtemp(prefix=f"{rgi_id}_{bed_scenario}_", dir=temp_root))
    flowlines, lia_mask, _ = compute_centerline_tree(
        rgi_id, geometry, grid, route_bed.bed_m, work_dir
    )
    source = load_gridded_bed(gdir, bed_scenario)
    bed = lia_gridded_bed(source, lia_mask)
    inputs = flowline_objects_to_inputs(flowlines, bed)
    build = build_network(
        inputs,
        junction_transfer_policy=EXPLICIT_CONNECTOR,
        gridded_bed=bed,
        connector_weights=ConnectorCostWeights(alpha=0.0, beta=0.0),
    )
    return gdir, grid, grid_json["proj"], geometry, flowlines, lia_mask, bed, inputs, build


def _failure_grid_position(build, inputs, failed_branch_id: str, failed_index: int) -> tuple[int, int, int | None, int | None]:
    if failed_branch_id.startswith("conn:"):
        audit = next(item for item in build.connector_audit if item.connector_branch_id == failed_branch_id)
        row, col = audit.pixels_rc[int(failed_index)]
        return int(row), int(col), None, None
    provenance = next(item for item in build.provenance if item.branch_id == failed_branch_id)
    native_index = int(provenance.native_index_end - int(failed_index))
    source = next(item for item in inputs if item.flowline_index == provenance.source_flowline_index)
    col, row = source.coords_xy[native_index]
    return int(round(row)), int(round(col)), native_index, int(provenance.source_flowline_index)


def _signed_mask_distance(mask: np.ndarray, row: int, col: int, map_dx_m: float) -> float:
    inside = bool(mask[row, col])
    if inside:
        return float(distance_transform_edt(mask)[row, col] * map_dx_m)
    return -float(distance_transform_edt(~mask)[row, col] * map_dx_m)


def _profile_metrics(branch, failed_index: int | None) -> dict:
    """Summarize the local bed obstacle around the first failing point."""
    if failed_index is None:
        return {}
    distance = np.asarray(branch.distance_m, dtype=float)
    bed = np.asarray(branch.bed_elevation_m, dtype=float)
    center = float(distance[int(failed_index)])
    window = (distance >= center - 2000.0) & (distance <= center + 2000.0)
    indices = np.flatnonzero(window)
    if indices.size < 3:
        return {"profile_window_point_count": int(indices.size)}
    x = distance[indices]
    z = bed[indices]
    baseline = np.linspace(float(z[0]), float(z[-1]), len(z))
    residual = z - baseline
    prominence = float(np.max(residual))
    half_level = max(prominence * 0.5, 1.0)
    ridge = x[residual >= half_level]
    return {
        "profile_window_point_count": int(indices.size),
        "profile_window_length_m": float(x[-1] - x[0]),
        "profile_bed_min_m": float(np.min(z)),
        "profile_bed_max_m": float(np.max(z)),
        "profile_bed_prominence_m": prominence,
        "profile_prominent_width_m": float(ridge[-1] - ridge[0]) if ridge.size > 1 else 0.0,
        "profile_max_positive_step_m": float(np.max(np.diff(z))),
        "profile_max_negative_step_m": float(np.min(np.diff(z))),
    }


def _diagnose_one(config, row, calibration: pd.DataFrame, output_dir: Path, context_cache: dict) -> tuple[dict, list[dict], list[dict]]:
    rgi_id = str(row.rgi_id)
    uid = int(row.lia_unique_id)
    bed_scenario = str(row.bed_scenario)
    cache_key = (rgi_id, uid, bed_scenario)
    if cache_key not in context_cache:
        temp_root = output_dir / "_centerline_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        context_cache[cache_key] = _build_context(config, rgi_id, uid, bed_scenario, temp_root)
    gdir, grid, grid_crs, lia_geometry, flowlines, lia_mask, bed, inputs, build = context_cache[cache_key]
    tau = float(row.basal_shear_stress_kpa)
    scenario = _scenario(tau)
    solution = solve_flowline_network(build.branches, scenario, outlet_node_ids=build.outlet_node_ids)
    failed_branch_id = str(solution.diagnostics.get("failed_branch_id", row.failed_branch_id))
    branch = next(branch for branch in build.branches if branch.branch_id == failed_branch_id)
    inherited = solution.node_surface_m.get(branch.downstream_node_id, float(branch.bed_elevation_m[0]))
    replay, replay_status, failed_index = replay_branch(branch, inherited, scenario)
    fail_row = replay.iloc[-1]
    profile_metrics = _profile_metrics(branch, failed_index)
    grid_row, grid_col, native_index, source_flowline_index = _failure_grid_position(
        build, inputs, failed_branch_id, int(failed_index or 0)
    )
    map_x, map_y = grid.ij_to_crs([grid_col], [grid_row])
    lon_lat = gpd.GeoSeries(
        [Point(float(map_x[0]), float(map_y[0]))], crs=grid_crs
    ).to_crs("EPSG:4326").iloc[0]
    modern_mask = np.asarray(load_gridded_bed(gdir, "farinotti2019").glacier_mask, dtype=bool)
    f19_bed_grid = load_gridded_bed(gdir, "farinotti2019").bed_m
    try:
        m22_bed_grid = load_gridded_bed(gdir, "millan2022").bed_m
        f19_m22_bed_difference_m = float(
            m22_bed_grid[grid_row, grid_col] - f19_bed_grid[grid_row, grid_col]
        )
    except (FileNotFoundError, ValueError):
        f19_m22_bed_difference_m = np.nan
    native_branch = not failed_branch_id.startswith("conn:")
    local_slope = np.nan
    if failed_index is not None and failed_index > 0:
        local_slope = float(abs(fail_row.bed_step_m) / max(branch.distance_m[1] - branch.distance_m[0], 1.0))
    profile_path = output_dir / f"failure_{rgi_id}_{bed_scenario}.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(replay.distance_m, replay.bed_m, label="bed", color="#7b3f00")
    ax.plot(replay.distance_m, replay.surface_m, label="replayed surface", color="#1769aa")
    if failed_index is not None:
        ax.scatter([float(fail_row.distance_m)], [float(fail_row.bed_m)], color="#b00020", zorder=4, label=f"failure index {failed_index}")
    ax.set_xlabel("distance from downstream node (m)")
    ax.set_ylabel("elevation (m)")
    ax.set_title(f"{rgi_id} / {bed_scenario} / {failed_branch_id} / tau={tau:.1f} kPa")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(profile_path, dpi=140)
    plt.close(fig)
    primary = {
        "rgi_id": rgi_id,
        "lia_unique_id": uid,
        "bed_scenario": bed_scenario,
        "tau_kpa": tau,
        "failed_branch_id": failed_branch_id,
        "native_branch_failure": native_branch,
        "downstream_node_id": branch.downstream_node_id,
        "upstream_node_id": branch.upstream_node_id,
        "inherited_surface_m": float(inherited),
        "failed_index": failed_index,
        "failure_status_replayed": replay_status,
        "failed_distance_m": float(fail_row.distance_m),
        "grid_row": grid_row,
        "grid_col": grid_col,
        "map_x_m": float(map_x[0]),
        "map_y_m": float(map_y[0]),
        "longitude": float(lon_lat.x),
        "latitude": float(lon_lat.y),
        "source_flowline_index": source_flowline_index,
        "source_native_index": native_index,
        "inside_modern_mask": bool(modern_mask[grid_row, grid_col]),
        "inside_lia_mask": bool(lia_mask[grid_row, grid_col]),
        "signed_distance_to_modern_mask_m": _signed_mask_distance(
            modern_mask, grid_row, grid_col, bed.map_dx_m
        ),
        "distance_to_lia_boundary_m": float(
            lia_geometry.boundary.distance(Point(float(map_x[0]), float(map_y[0])))
        ),
        "f19_m22_bed_difference_m": f19_m22_bed_difference_m,
        "failed_bed_m": float(fail_row.bed_m),
        "failed_surface_m": float(fail_row.surface_m) if np.isfinite(fail_row.surface_m) else np.nan,
        "failed_thickness_m": float(fail_row.thickness_m) if np.isfinite(fail_row.thickness_m) else np.nan,
        "failed_bed_step_m": float(fail_row.bed_step_m) if np.isfinite(fail_row.bed_step_m) else np.nan,
        "failed_discriminant": float(fail_row.discriminant) if np.isfinite(fail_row.discriminant) else np.nan,
        "local_bed_slope_abs": local_slope,
        "profile_png": profile_path.name,
        "route_bed_note": "F19-route-on-M22-bed-mismatch-suspect" if bed_scenario == "millan2022" else "route-and-bed-both-F19",
        **profile_metrics,
    }
    tau_rows = []
    for factor in TAU_SCAN_FACTORS:
        trial = solve_flowline_network(build.branches, _scenario(tau * factor), outlet_node_ids=build.outlet_node_ids)
        tau_rows.append({
            "rgi_id": rgi_id,
            "bed_scenario": bed_scenario,
            "base_tau_kpa": tau,
            "tau_factor": factor,
            "trial_tau_kpa": tau * factor,
            "status": trial.status,
            "failed_branch_id": trial.diagnostics.get("failed_branch_id", ""),
        })
    bed_rows = []
    if bed_scenario == "millan2022":
        f19 = context_cache.get((rgi_id, uid, "farinotti2019"))
        if f19 is None:
            temp_root = output_dir / "_centerline_tmp"
            f19 = _build_context(config, rgi_id, uid, "farinotti2019", temp_root)
            context_cache[(rgi_id, uid, "farinotti2019")] = f19
        _, _, _, _, _, _, f19_bed, _, f19_build = f19
        same_tau = solve_flowline_network(f19_build.branches, _scenario(tau), outlet_node_ids=f19_build.outlet_node_ids)
        f19_tau_row = calibration[(calibration.rgi_id == rgi_id) & (calibration.bed_scenario == "farinotti2019")]
        f19_tau = float(f19_tau_row.iloc[0].tau_kpa) if not f19_tau_row.empty and np.isfinite(f19_tau_row.iloc[0].tau_kpa) else np.nan
        tau_swap = solve_flowline_network(build.branches, _scenario(f19_tau), outlet_node_ids=build.outlet_node_ids) if np.isfinite(f19_tau) else None
        bed_rows.extend([
            {"rgi_id": rgi_id, "bed_scenario_observed": "millan2022", "counterfactual": "F19-bed-same-M22-tau", "tau_kpa": tau, "status": same_tau.status, "failed_branch_id": same_tau.diagnostics.get("failed_branch_id", "")},
            {"rgi_id": rgi_id, "bed_scenario_observed": "millan2022", "counterfactual": "M22-bed-F19-tau", "tau_kpa": f19_tau, "status": tau_swap.status if tau_swap is not None else "no-f19-tau", "failed_branch_id": tau_swap.diagnostics.get("failed_branch_id", "") if tau_swap is not None else ""},
        ])
    return primary, tau_rows, bed_rows


def _attach_failure_classification(
    primary_df: pd.DataFrame,
    tau_df: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Attach modern calibration support, network feasibility, and cause class."""
    cal_cols = [
        "rgi_id", "bed_scenario", "tau_kpa", "kappa_kpa", "status",
        "near_optimal_tau_min_kpa", "near_optimal_tau_max_kpa",
    ]
    cal_join = calibration[[c for c in cal_cols if c in calibration.columns]].copy()
    cal_join = cal_join.rename(
        columns={
            "tau_kpa": "modern_calibrated_tau_kpa",
            "kappa_kpa": "modern_calibrated_kappa_kpa",
            "status": "modern_tau_calibration_status",
            "near_optimal_tau_min_kpa": "modern_near_optimal_tau_min_kpa",
            "near_optimal_tau_max_kpa": "modern_near_optimal_tau_max_kpa",
        }
    )
    result = primary_df.merge(cal_join, on=["rgi_id", "bed_scenario"], how="left")

    feasible = tau_df[tau_df["status"].astype(str).eq("success")].copy()
    if not feasible.empty:
        feasible = feasible.sort_values(["rgi_id", "bed_scenario", "trial_tau_kpa"])
        first = feasible.groupby(["rgi_id", "bed_scenario"], as_index=False).first()
        first = first[["rgi_id", "bed_scenario", "tau_factor", "trial_tau_kpa"]].rename(
            columns={
                "tau_factor": "first_feasible_tau_factor",
                "trial_tau_kpa": "first_feasible_tau_kpa",
            }
        )
        result = result.merge(first, on=["rgi_id", "bed_scenario"], how="left")
        feasible_stats = feasible.groupby(["rgi_id", "bed_scenario"], as_index=False).agg(
            feasible_tau_count=("trial_tau_kpa", "size"),
            feasible_tau_min_kpa=("trial_tau_kpa", "min"),
            feasible_tau_max_kpa=("trial_tau_kpa", "max"),
        )
        result = result.merge(feasible_stats, on=["rgi_id", "bed_scenario"], how="left")
    else:
        result["first_feasible_tau_factor"] = np.nan
        result["first_feasible_tau_kpa"] = np.nan
        result["feasible_tau_count"] = 0
        result["feasible_tau_min_kpa"] = np.nan
        result["feasible_tau_max_kpa"] = np.nan

    result["first_feasible_tau_relation"] = "none-in-scan"
    has_first = result["first_feasible_tau_kpa"].notna()
    in_near = (
        has_first
        & result["modern_near_optimal_tau_min_kpa"].notna()
        & result["modern_near_optimal_tau_max_kpa"].notna()
        & (result["first_feasible_tau_kpa"] >= result["modern_near_optimal_tau_min_kpa"])
        & (result["first_feasible_tau_kpa"] <= result["modern_near_optimal_tau_max_kpa"])
    )
    result.loc[has_first & in_near, "first_feasible_tau_relation"] = "within-modern-near-optimal-support"
    result.loc[has_first & ~in_near, "first_feasible_tau_relation"] = "outside-modern-near-optimal-support"

    result["bed_products_equal_at_failure"] = (
        result["f19_m22_bed_difference_m"].abs() <= BED_EQUALITY_TOLERANCE_M
    )
    result["profile_obstacle_evidence"] = (
        (result["profile_bed_prominence_m"].fillna(0.0) >= PROFILE_OBSTACLE_PROMINENCE_M)
        | (result["profile_max_positive_step_m"].fillna(0.0).abs() >= PROFILE_OBSTACLE_STEP_M)
    )

    native = result["native_branch_failure"].astype(bool)
    connector = ~native
    outside_common_bed = (
        native
        & ~result["inside_modern_mask"].astype(bool)
        & result["bed_products_equal_at_failure"].fillna(False)
    )
    route_mismatch = (
        native
        & result["inside_modern_mask"].astype(bool)
        & result["bed_scenario"].eq("millan2022")
        & result["m22_same_tau_on_f19_status"].astype(str).eq("success")
    )
    parameter_edge = (
        native
        & result["first_feasible_tau_kpa"].notna()
        & result["first_feasible_tau_relation"].ne("within-modern-near-optimal-support")
    )
    geometry = native & result["first_feasible_tau_kpa"].isna() & result["profile_obstacle_evidence"]
    model_domain = native & result["first_feasible_tau_kpa"].isna() & ~result["profile_obstacle_evidence"]

    result["diagnostic_class"] = "model-domain-infeasible"
    result.loc[geometry, "diagnostic_class"] = "geometry-dominated-failure"
    result.loc[parameter_edge, "diagnostic_class"] = "parameter-edge-feasible"
    result.loc[route_mismatch, "diagnostic_class"] = "route-bed-mismatch-suspect"
    result.loc[outside_common_bed, "diagnostic_class"] = "modern-exposed-topography-suspect"
    result.loc[connector, "diagnostic_class"] = "connector-inherited-surface-review"
    return result


def _build_production_routing(pilot: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Collapse primary-member outcomes to one auditable route per system."""
    all_pilot = pilot.copy()
    structure_ids = set(
        all_pilot.loc[
            all_pilot["status"].astype(str).eq("centerline-structure-review"), "rgi_id"
        ].astype(str)
    )
    pilot = pilot[pilot["head_exclusion_distance_m"] == 100.0].copy()
    rgi_ids = sorted(set(pilot["rgi_id"].astype(str)) | structure_ids)
    rows = []
    for rgi_id in rgi_ids:
        group = pilot[pilot["rgi_id"].astype(str).eq(rgi_id)]
        if rgi_id in structure_ids:
            route = "centerline-structure-review"
            reason = "centerline structure gate precedes bed routing"
        else:
            statuses = group.set_index("bed_scenario")["status"].astype(str).to_dict()
            f19 = statuses.get("farinotti2019", "missing")
            m22 = statuses.get("millan2022", "missing")
            f19_ok, m22_ok = f19 == "success", m22 == "success"
            if f19_ok and m22_ok:
                route, reason = "dual-bed-pass", "both bed members passed"
            elif f19_ok:
                route, reason = "primary-bed-pass", f"F19 passed; M22={m22}"
            elif m22_ok:
                route, reason = "alternate-bed-only", f"M22 passed; F19={f19}"
            elif "missing" in (f19, m22) or "tau-calibration-review" in (f19, m22):
                route, reason = "input-or-calibration-review", f"F19={f19}; M22={m22}"
            else:
                route = "documented-exclusion"
                reason = f"both bed members failed: F19={f19}; M22={m22}"
        diag = diagnostics[diagnostics["rgi_id"] == rgi_id]
        rows.append({
            "rgi_id": rgi_id,
            "production_route": route,
            "routing_reason": reason,
            "farinotti_status": str(group.loc[group["bed_scenario"] == "farinotti2019", "status"].iloc[0]) if (group["bed_scenario"] == "farinotti2019").any() else "missing",
            "millan_status": str(group.loc[group["bed_scenario"] == "millan2022", "status"].iloc[0]) if (group["bed_scenario"] == "millan2022").any() else "missing",
            "failure_diagnostic_classes": ";".join(sorted(set(diag["diagnostic_class"].astype(str)))) if not diag.empty else "",
        })
    return pd.DataFrame(rows)


def run(config: dict) -> dict:
    _configure_oggm(config)
    root = resolve(config["output_dir"])
    output_dir = root / "lia_failure_diagnostics_v0"
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_path = root / str(config["lia_3d_zero_boundary_pilot"]["output_subdir"]) / "lia_3d_calibrated_results_v0.csv"
    failures = pd.read_csv(primary_path)
    failures = failures[(failures["head_exclusion_distance_m"] == 100.0) & failures["status"].astype(str).str.startswith("network-failed:")]
    calibration = pd.read_csv(root / str(config["modern_tau_calibration"]["output_subdir"]) / "modern_tau_calibration_v0.csv")
    context_cache: dict = {}
    primary_rows: list[dict] = []
    tau_rows: list[dict] = []
    bed_rows: list[dict] = []
    for row in failures.itertuples(index=False):
        primary, tau, bed = _diagnose_one(config, row, calibration, output_dir, context_cache)
        primary_rows.append(primary)
        tau_rows.extend(tau)
        bed_rows.extend(bed)
    primary_df = pd.DataFrame(primary_rows)
    tau_df = pd.DataFrame(tau_rows)
    bed_df = pd.DataFrame(bed_rows)
    if not bed_df.empty:
        bed_status = bed_df[bed_df["counterfactual"] == "F19-bed-same-M22-tau"].set_index("rgi_id")["status"].to_dict()
        primary_df["m22_same_tau_on_f19_status"] = primary_df["rgi_id"].map(bed_status).fillna("")
    else:
        primary_df["m22_same_tau_on_f19_status"] = ""
    primary_df = _attach_failure_classification(primary_df, tau_df, calibration)
    pilot = pd.read_csv(primary_path)
    routing_df = _build_production_routing(pilot, primary_df)
    primary_df.to_csv(output_dir / "lia_failure_diagnostics_v0.csv", index=False)
    tau_df.to_csv(output_dir / "lia_failure_tau_scan_v0.csv", index=False)
    bed_df.to_csv(output_dir / "lia_failure_bed_tau_counterfactual_v0.csv", index=False)
    routing_df.to_csv(output_dir / "lia_failure_production_routing_v0.csv", index=False)
    summary = {
        "status": "completed-diagnostic-only",
        "failure_member_count": int(len(failures)),
        "diagnostic_class_counts": primary_df["diagnostic_class"].value_counts(dropna=False).to_dict() if not primary_df.empty else {},
        "tau_scan_factors": list(TAU_SCAN_FACTORS),
        "routing_table": "lia_failure_production_routing_v0.csv",
        "production_fields_unchanged": True,
        "interpretation": "A counterfactual success is evidence about cause, not permission to repair the production member.",
    }
    (output_dir / "lia_failure_diagnostics_summary_v0.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()

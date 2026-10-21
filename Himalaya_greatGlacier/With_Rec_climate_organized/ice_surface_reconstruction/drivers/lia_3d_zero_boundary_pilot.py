"""True 3-D LIA reconstruction smoke on systems with zero-boundary eligibility.

Only systems with no candidate ice-divide segment and stable pilot centerline
routing are admitted. Centerline geometry is fixed from the Farinotti raw-bed
route for every bed scenario so the bed-product axis does not silently change
the network topology. This driver never writes to a modern gdir.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_erosion, map_coordinates
from shapely.geometry import Point

from ..core import (
    SurfaceScenario,
    harmonic_extend_thickness,
    refine_flowline_network,
    sample_network_solution_at_original_nodes,
    solve_flowline_network,
    surface_quality_metrics,
)
from ..paths import resolve
from .gridded_bed import GriddedBed, load_gridded_bed
from .junction_connector import ConnectorCostWeights
from .lia_centerline_pilot import (
    _configure_oggm,
    compute_centerline_tree,
)
from .lia_geometry_preflight import _gdir_path
from .oggm_network_adapter import EXPLICIT_CONNECTOR, FlowlineInput, build_network


def _sample_grid(values: np.ndarray, coords_xy: np.ndarray) -> np.ndarray:
    return map_coordinates(
        values,
        np.vstack([coords_xy[:, 1], coords_xy[:, 0]]),
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def flowline_objects_to_inputs(
    flowlines: list,
    bed: GriddedBed,
) -> list[FlowlineInput]:
    """Convert in-memory OGGM LIA flowlines and sample a scenario bed."""
    index_by_id = {id(flowline): i for i, flowline in enumerate(flowlines)}
    coords_by_index = {
        i: np.asarray(flowline.line.coords, dtype=float)
        for i, flowline in enumerate(flowlines)
    }
    inputs: list[FlowlineInput] = []
    for index, flowline in enumerate(flowlines):
        parent_index = (
            None if flowline.flows_to is None else index_by_id[id(flowline.flows_to)]
        )
        parent_native_index = None
        if parent_index is not None:
            point = np.asarray(
                flowline.__dict__["flows_to_point"].coords[0], dtype=float
            )
            parent_coords = coords_by_index[parent_index]
            parent_native_index = int(np.argmin(np.hypot(
                parent_coords[:, 0] - point[0],
                parent_coords[:, 1] - point[1],
            )))
        coords = coords_by_index[index]
        sampled_bed = _sample_grid(bed.bed_m, coords)
        reference = np.asarray(flowline.surface_h, dtype=float)
        if not np.isfinite(sampled_bed).all():
            raise ValueError(f"flowline {index} samples non-finite bed")
        inputs.append(FlowlineInput(
            flowline_index=index,
            flows_to=parent_index,
            flows_to_indice=parent_native_index,
            bed_elevation_m=sampled_bed,
            reference_surface_m=reference,
            dx_meter=float(flowline.dx_meter),
            map_dx_m=float(bed.map_dx_m),
            coords_xy=coords,
        ))
    return inputs


def lia_gridded_bed(source: GriddedBed, lia_mask: np.ndarray) -> GriddedBed:
    """Use the continuous scenario bed with the LIA footprint as path mask."""
    gy, gx = np.gradient(source.bed_m, source.map_dx_m, source.map_dx_m)
    slope = np.arctan(np.hypot(gx, gy))
    return replace(
        source,
        glacier_mask=np.asarray(lia_mask, dtype=bool),
        slope=slope,
    )


def collect_network_constraints(
    flowline_inputs: list[FlowlineInput],
    build,
    solution,
    duplicate_surface_tolerance_m: float = 1e-3,
    cross_branch_surface_tolerance_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Map solved branches to one representative constraint per raster cell.

    OGGM flowlines are sampled at metre scale while the 2-D grid is tens to
    hundreds of metres. Repeated points from the *same* branch therefore do
    not constitute a physical conflict; the point nearest the cell centre is
    retained. Representatives from distinct branches remain auditable and are
    rejected when their surface values disagree beyond the registered
    cross-branch tolerance.  Accepted collisions retain the representative
    nearest the raster-cell centre; surfaces are never averaged.
    """
    cross_branch_tolerance_m = (
        float(duplicate_surface_tolerance_m)
        if cross_branch_surface_tolerance_m is None
        else float(cross_branch_surface_tolerance_m)
    )
    if cross_branch_tolerance_m < 0:
        raise ValueError("cross_branch_surface_tolerance_m must be non-negative")
    by_index = {item.flowline_index: item for item in flowline_inputs}
    candidates: dict[tuple[int, int], list[tuple[str, float, float, float]]] = {}

    def add_points(source_id: str, coords, values_h, values_s) -> None:
        for coord, h_value, s_value in zip(coords, values_h, values_s):
            col = int(round(float(coord[0])))
            row = int(round(float(coord[1])))
            distance_to_cell_center = (
                (float(coord[0]) - col) ** 2 + (float(coord[1]) - row) ** 2
            )
            candidates.setdefault((row, col), []).append(
                (source_id, distance_to_cell_center, float(h_value), float(s_value))
            )

    for provenance in build.provenance:
        branch_solution = solution.branch_solutions[provenance.branch_id]
        native_indices = np.arange(
            provenance.native_index_end,
            provenance.native_index_start - 1,
            -1,
            dtype=int,
        )
        coords = by_index[provenance.source_flowline_index].coords_xy[native_indices]
        if len(coords) != len(branch_solution.thickness_m):
            raise ValueError(f"constraint provenance length mismatch: {provenance.branch_id}")
        add_points(
            provenance.branch_id,
            coords,
            branch_solution.thickness_m,
            branch_solution.surface_elevation_m,
        )
    for connector in build.connector_audit:
        branch_solution = solution.branch_solutions[connector.connector_branch_id]
        if len(connector.pixels_rc) != len(branch_solution.thickness_m):
            raise ValueError(
                f"connector provenance length mismatch: {connector.connector_branch_id}"
            )
        coords = np.asarray(
            [(col, row) for row, col in connector.pixels_rc], dtype=float
        )
        add_points(
            connector.connector_branch_id,
            coords,
            branch_solution.thickness_m,
            branch_solution.surface_elevation_m,
        )
    rows: list[int] = []
    cols: list[int] = []
    thickness: list[float] = []
    surface: list[float] = []
    cross_branch_conflicts: list[dict] = []
    cross_branch_shared_cell_count = 0
    accepted_cross_branch_collision_count = 0
    maximum_accepted_cross_branch_surface_spread_m = 0.0
    for (row, col), values in sorted(candidates.items()):
        by_source: dict[str, tuple[str, float, float, float]] = {}
        for value in values:
            source_id = value[0]
            if source_id not in by_source or value[1] < by_source[source_id][1]:
                by_source[source_id] = value
        representatives = list(by_source.values())
        spread = float(np.ptp([value[3] for value in representatives]))
        if len(representatives) > 1:
            cross_branch_shared_cell_count += 1
            if spread > cross_branch_tolerance_m:
                cross_branch_conflicts.append({
                    "row": row,
                    "col": col,
                    "source_ids": sorted(by_source),
                    "surface_spread_m": spread,
                })
            else:
                accepted_cross_branch_collision_count += 1
                maximum_accepted_cross_branch_surface_spread_m = max(
                    maximum_accepted_cross_branch_surface_spread_m, spread
                )
        selected = min(representatives, key=lambda value: value[1])
        rows.append(row)
        cols.append(col)
        thickness.append(selected[2])
        surface.append(selected[3])
    if cross_branch_conflicts:
        first = cross_branch_conflicts[0]
        raise ValueError(
            "rasterized-cross-branch-cell-conflict: "
            f"row={first['row']} col={first['col']} "
            f"spread_m={first['surface_spread_m']:.3f} "
            f"sources={','.join(first['source_ids'])}"
        )
    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(thickness, dtype=float),
        np.asarray(surface, dtype=float),
        {
            "input_constraint_count": int(sum(len(value) for value in candidates.values())),
            "raster_constraint_count": int(len(rows)),
            "same_branch_repeated_cell_count": int(
                sum(len(value) - len({item[0] for item in value}) for value in candidates.values())
            ),
            "cross_branch_shared_cell_count": int(cross_branch_shared_cell_count),
            "accepted_cross_branch_collision_count": int(
                accepted_cross_branch_collision_count
            ),
            "maximum_accepted_cross_branch_surface_spread_m": float(
                maximum_accepted_cross_branch_surface_spread_m
            ),
            "cross_branch_surface_tolerance_m": float(cross_branch_tolerance_m),
        },
    )


def filter_constraints_by_boundary_distance(
    rows: np.ndarray,
    cols: np.ndarray,
    thickness_m: np.ndarray,
    surface_m: np.ndarray,
    *,
    grid,
    boundary_geometry,
    domain_mask: np.ndarray,
    exclusion_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Drop flowline constraints inside a metric buffer of the LIA boundary.

    Distances are measured from raster-cell centres in the grid map CRS to the
    original vector polygon boundary.  The rule is therefore invariant to
    raster resolution and does not rely on a one-cell morphological boundary.
    """
    if exclusion_distance_m < 0:
        raise ValueError("exclusion_distance_m must be non-negative")
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    thickness = np.asarray(thickness_m, dtype=float)
    surface = np.asarray(surface_m, dtype=float)
    if not (rows.shape == cols.shape == thickness.shape == surface.shape):
        raise ValueError("constraint arrays must have matching shapes")
    x, y = grid.ij_to_crs(cols.astype(float), rows.astype(float))
    distances = np.asarray(
        [boundary_geometry.distance(Point(float(px), float(py))) for px, py in zip(x, y)],
        dtype=float,
    )
    inside = np.asarray(domain_mask, dtype=bool)
    if inside.ndim != 2:
        raise ValueError("domain_mask must be a 2-D array")
    raster_boundary = inside & ~binary_erosion(inside, border_value=0)
    on_raster_boundary = raster_boundary[rows, cols]
    keep_by_distance = distances >= float(exclusion_distance_m)
    keep = keep_by_distance & ~on_raster_boundary
    removed_h = thickness[~keep]
    retained_h = thickness[keep]
    retained_d = distances[keep]
    implied_angles = np.degrees(
        np.arctan2(retained_h, np.maximum(retained_d, 1e-9))
    )
    metadata = {
        "head_exclusion_distance_m": float(exclusion_distance_m),
        "excluded_constraint_count": int((~keep).sum()),
        "metric_excluded_constraint_count": int((~keep_by_distance).sum()),
        "raster_boundary_excluded_constraint_count": int(
            (keep_by_distance & on_raster_boundary).sum()
        ),
        "retained_constraint_count": int(keep.sum()),
        "maximum_excluded_thickness_m": (
            float(np.max(removed_h)) if removed_h.size else 0.0
        ),
        "minimum_retained_boundary_distance_m": (
            float(np.min(retained_d)) if retained_d.size else np.nan
        ),
        "maximum_inner_edge_implied_slope_deg": (
            float(np.max(implied_angles)) if implied_angles.size else np.nan
        ),
    }
    return rows[keep], cols[keep], thickness[keep], surface[keep], metadata


def classify_flowline_heads(
    flowlines: list,
    *,
    grid,
    boundary_segments: gpd.GeoDataFrame,
    exclusion_candidates_m: list[float],
) -> list[dict]:
    """Assign each OGGM head to its nearest automatic boundary segment class."""
    rows: list[dict] = []
    for index, flowline in enumerate(flowlines):
        col, row = np.asarray(flowline.line.coords[0], dtype=float)
        x, y = grid.ij_to_crs([col], [row])
        point = Point(float(x[0]), float(y[0]))
        distances = boundary_segments.geometry.distance(point)
        if distances.empty:
            nearest_class = "unknown-boundary-review"
            distance_m = np.nan
        else:
            nearest_index = distances.idxmin()
            nearest_class = str(boundary_segments.loc[nearest_index, "boundary_class"])
            distance_m = float(distances.loc[nearest_index])
        output = {
            "flowline_index": int(index),
            "head_col": float(col),
            "head_row": float(row),
            "head_x_m": float(x[0]),
            "head_y_m": float(y[0]),
            "head_distance_to_boundary_m": distance_m,
            "nearest_boundary_class": nearest_class,
            "candidate_divide": nearest_class == "ice-divide-candidate-review",
            "classification_basis": "nearest-automatic-LIA-boundary-segment",
        }
        for value in exclusion_candidates_m:
            output[f"excluded_at_{float(value):g}m"] = bool(
                np.isfinite(distance_m) and distance_m < float(value)
            )
        rows.append(output)
    return rows


def _calibrated_surface_scenario(config: dict, calibration_row) -> SurfaceScenario:
    shape = float(config["modern_tau_calibration"]["shape_factor"])
    return SurfaceScenario(
        "modern-tau-calibrated",
        float(calibration_row.tau_kpa) * 1000.0,
        shape,
    )


def _eligible_systems(config: dict) -> pd.DataFrame:
    output_dir = resolve(config["output_dir"])
    boundary = pd.read_csv(output_dir / "lia_boundary_preflight_v0.csv")
    routing = pd.read_csv(output_dir / "lia_centerline_routing_decisions_v0.csv")
    eligible = boundary.merge(
        routing,
        on="rgi_id",
        how="inner",
    )
    return eligible


def run(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    _configure_oggm(config)
    pilot_config = config["lia_3d_zero_boundary_pilot"]
    result_version = str(pilot_config.get("result_version", "v0"))
    integration_factor = int(
        pilot_config.get("flowline_integration_refinement_factor", 1)
    )
    if integration_factor < 1:
        raise ValueError("flowline_integration_refinement_factor must be positive")
    output_root = resolve(config["output_dir"])
    output_dir = output_root / str(pilot_config["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    gdir_root = resolve(config["stage0_gdir_root"])
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    lia_by_id = lia.set_index("UniqueID")
    boundary_segments = gpd.read_file(output_root / "lia_boundary_segments_v0.geojson")
    eligible = _eligible_systems(config)
    calibration_dir = (
        output_root / str(config["modern_tau_calibration"]["output_subdir"])
    )
    calibration = pd.read_csv(calibration_dir / "modern_tau_calibration_v0.csv")
    rows: list[dict] = []
    head_rows: list[dict] = []

    for system in eligible.itertuples(index=False):
        rgi_id = str(system.rgi_id)
        uid = int(system.lia_unique_id)
        if str(system.centerline_route_status) != "stable-pilot":
            rows.append({
                "rgi_id": rgi_id,
                "lia_unique_id": uid,
                "status": "centerline-structure-review",
                "failure_reason": "LIA centerline route did not pass the pilot stability gate",
                "boundary_preflight_status": str(system.status),
            })
            continue
        gdir_dir = _gdir_path(gdir_root, rgi_id)
        grid_json = json.loads((gdir_dir / "glacier_grid.json").read_text())
        import salem
        grid = salem.Grid.from_json(gdir_dir / "glacier_grid.json")
        grid_crs = grid_json["proj"]
        lia_geometry = gpd.GeoSeries(
            [lia_by_id.loc[uid].geometry], crs=lia.crs
        ).to_crs(grid_crs).iloc[0]
        try:
            route_bed = load_gridded_bed(gdir_dir, "farinotti2019")
            with tempfile.TemporaryDirectory(
                prefix=f"lia3d_{uid}_", dir=output_dir
            ) as temp_dir:
                flowlines, lia_mask, _ = compute_centerline_tree(
                    rgi_id,
                    lia_geometry,
                    grid,
                    route_bed.bed_m,
                    Path(temp_dir),
                )
            system_segments = boundary_segments[
                boundary_segments["lia_unique_id"].astype(int) == uid
            ].to_crs(grid_crs)
            head_rows.extend({
                "rgi_id": rgi_id,
                "lia_unique_id": uid,
                **item,
            } for item in classify_flowline_heads(
                flowlines,
                grid=grid,
                boundary_segments=system_segments,
                exclusion_candidates_m=[
                    float(value)
                    for value in pilot_config["head_exclusion_candidates_m"]
                ],
            ))
        except Exception as exc:
            rows.append({
                "rgi_id": rgi_id,
                "lia_unique_id": uid,
                "status": "centerline-build-review",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        successful_arrays: list[np.ndarray] = []
        successful_surfaces: list[np.ndarray] = []
        successful_beds: list[np.ndarray] = []
        successful_member_ids: list[str] = []
        for bed_scenario in config["bed_scenarios"]:
            tau_rows = calibration[
                (calibration["rgi_id"] == rgi_id)
                & (calibration["bed_scenario"] == bed_scenario)
            ]
            if tau_rows.empty or not np.isfinite(tau_rows.iloc[0].get("tau_kpa", np.nan)):
                rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "bed_scenario": bed_scenario,
                    "status": "tau-calibration-review",
                    "failure_reason": (
                        "missing or non-finite modern tau calibration"
                        if tau_rows.empty
                        else str(tau_rows.iloc[0].get("status", ""))
                    ),
                    "boundary_preflight_status": str(system.status),
                })
                continue
            tau_row = next(tau_rows.itertuples(index=False))
            try:
                source_bed = load_gridded_bed(gdir_dir, bed_scenario)
                bed = lia_gridded_bed(source_bed, lia_mask)
                inputs = flowline_objects_to_inputs(flowlines, bed)
            except (FileNotFoundError, ValueError) as exc:
                rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "bed_scenario": bed_scenario,
                    "status": "bed-input-review",
                    "failure_reason": str(exc),
                })
                continue
            for alpha in [float(value) for value in pilot_config["connector_alpha"]]:
                try:
                    build = build_network(
                        inputs,
                        junction_transfer_policy=EXPLICIT_CONNECTOR,
                        gridded_bed=bed,
                        connector_weights=ConnectorCostWeights(
                            alpha=alpha,
                            beta=float(pilot_config["connector_beta"]),
                        ),
                    )
                except ValueError as exc:
                    rows.append({
                        "rgi_id": rgi_id,
                        "lia_unique_id": uid,
                        "bed_scenario": bed_scenario,
                        "connector_alpha": alpha,
                        "status": "network-adapter-review",
                        "failure_reason": str(exc),
                    })
                    continue
                grid_multiplier = pilot_config.get(
                    "head_exclusion_grid_multiplier"
                )
                if grid_multiplier is None:
                    head_exclusion_values = [
                        float(value)
                        for value in pilot_config["head_exclusion_candidates_m"]
                    ]
                else:
                    head_exclusion_values = [max(
                        float(pilot_config["head_exclusion_primary_m"]),
                        float(grid_multiplier) * float(bed.map_dx_m),
                    )]
                for head_exclusion_m in head_exclusion_values:
                    scenario = _calibrated_surface_scenario(config, tau_row)
                    member_id = (
                        f"{bed_scenario}__alpha{alpha:g}__tau{scenario.basal_shear_stress_pa / 1000.0:.2f}"
                        f"__head{head_exclusion_m:g}m"
                        + (f"__step{integration_factor}x" if integration_factor != 1 else "")
                    )
                    refined_branches = refine_flowline_network(
                        build.branches, integration_factor
                    )
                    network_solution = solve_flowline_network(
                        refined_branches,
                        scenario,
                        outlet_node_ids=build.outlet_node_ids,
                    )
                    base_row = {
                        "rgi_id": rgi_id,
                        "lia_unique_id": uid,
                        "bed_scenario": bed_scenario,
                        "connector_alpha": alpha,
                        "connector_beta": float(pilot_config["connector_beta"]),
                        "stress_scenario": scenario.scenario_id,
                        "tau_calibration_status": str(tau_row.status),
                        "basal_shear_stress_kpa": scenario.basal_shear_stress_pa / 1000.0,
                        "shape_factor": scenario.shape_factor,
                        "head_exclusion_distance_m": head_exclusion_m,
                        "boundary_preflight_status": str(system.status),
                        "candidate_divide_v0_zero_boundary": str(system.status) != "provisional-zero-boundaries",
                        "route_bed_compatibility_note": (
                            "F19-route-on-M22-bed-mismatch-suspect"
                            if bed_scenario == "millan2022"
                            else "route-and-bed-both-F19"
                        ),
                        "member_id": member_id,
                        "branch_count": len(build.branches),
                        "connector_count": len(build.connector_audit),
                        "flowline_integration_refinement_factor": integration_factor,
                        "flowline_integration_policy": "piecewise-linear-bed-no-smoothing",
                        "head_exclusion_grid_multiplier": (
                            float(grid_multiplier)
                            if grid_multiplier is not None else np.nan
                        ),
                        "map_dx_m": float(bed.map_dx_m),
                    }
                    if not network_solution.success:
                        failed_branch_id = network_solution.diagnostics.get(
                            "failed_branch_id", ""
                        )
                        failed_solution = network_solution.branch_solutions.get(
                            failed_branch_id
                        )
                        failed_index = (
                            failed_solution.diagnostics.get("failed_index")
                            if failed_solution is not None else None
                        )
                        failed_bed_m = np.nan
                        if failed_solution is not None and failed_index is not None:
                            failed_bed_m = float(
                                failed_solution.bed_elevation_m[int(failed_index)]
                            )
                        rows.append({
                            **base_row,
                            "status": f"network-failed:{network_solution.status}",
                            "failure_reason": failed_branch_id,
                            "failed_branch_id": failed_branch_id,
                            "failed_branch_is_connector": str(failed_branch_id).startswith(
                                "conn:"
                            ),
                            "failed_index": failed_index,
                            "failed_bed_elevation_m": failed_bed_m,
                        })
                        continue
                    if integration_factor != 1:
                        network_solution = sample_network_solution_at_original_nodes(
                            build.branches,
                            network_solution,
                            integration_factor,
                            scenario,
                        )
                    try:
                        (
                            constraint_rows,
                            constraint_cols,
                            constraint_h,
                            constraint_s,
                            constraint_meta,
                        ) = collect_network_constraints(
                            inputs,
                            build,
                            network_solution,
                            duplicate_surface_tolerance_m=float(
                                pilot_config["duplicate_surface_tolerance_m"]
                            ),
                        )
                    except ValueError as exc:
                        rows.append({
                            **base_row,
                            "status": "raster-constraint-review",
                            "failure_reason": str(exc),
                        })
                        continue
                    (
                        constraint_rows,
                        constraint_cols,
                        constraint_h,
                        constraint_s,
                        head_meta,
                    ) = filter_constraints_by_boundary_distance(
                        constraint_rows,
                        constraint_cols,
                        constraint_h,
                        constraint_s,
                        grid=grid,
                        boundary_geometry=lia_geometry.boundary,
                        domain_mask=lia_mask,
                        exclusion_distance_m=head_exclusion_m,
                    )
                    if len(constraint_h) == 0:
                        rows.append({
                            **base_row,
                            **head_meta,
                            "status": "head-exclusion-review",
                            "failure_reason": "all rasterized network constraints were excluded",
                        })
                        continue
                    thickness_2d, extension = harmonic_extend_thickness(
                        lia_mask,
                        constraint_rows,
                        constraint_cols,
                        constraint_h,
                        centerline_surface_m=constraint_s,
                        duplicate_tolerance_m=float(
                            pilot_config["duplicate_surface_tolerance_m"]
                        ),
                        boundary_tolerance_m=float(
                            pilot_config["boundary_thickness_tolerance_m"]
                        ),
                        max_iterations=int(pilot_config["harmonic_max_iterations"]),
                        tolerance_m=float(pilot_config["harmonic_tolerance_m"]),
                    )
                    if extension["status"] != "converged":
                        rows.append({
                            **base_row,
                            **head_meta,
                            "status": f"extension-failed:{extension['status']}",
                            "failure_reason": extension.get("failure_reason", ""),
                            "harmonic_iterations": extension.get("iterations"),
                            "conflict_cell_count": extension.get("conflict_cell_count"),
                            "conflict_row": extension.get("conflict_row"),
                            "conflict_col": extension.get("conflict_col"),
                            "maximum_duplicate_surface_spread_m": extension.get(
                                "maximum_duplicate_surface_spread_m"
                            ),
                            "conflict_constraint_thickness_m": extension.get(
                                "conflict_constraint_thickness_m"
                            ),
                            "maximum_boundary_constraint_thickness_m": extension.get(
                                "maximum_boundary_constraint_thickness_m"
                            ),
                            "input_centerline_constraint_count": len(constraint_h),
                            "post_exclusion_raster_constraint_count": len(constraint_h),
                            "same_branch_repeated_cell_count": constraint_meta[
                                "same_branch_repeated_cell_count"
                            ],
                        })
                        continue
                    quality = surface_quality_metrics(
                        lia_mask,
                        bed.bed_m,
                        thickness_2d,
                        centerline_rows=constraint_rows,
                        centerline_cols=constraint_cols,
                        centerline_reference_m=constraint_h,
                    )
                    volume_m3 = float(
                        np.sum(thickness_2d[lia_mask]) * bed.map_dx_m**2
                    )
                    rows.append({
                        **base_row,
                        **head_meta,
                        "status": "success",
                        "volume_km3": volume_m3 / 1e9,
                        "mean_thickness_m": float(np.mean(thickness_2d[lia_mask])),
                        "maximum_thickness_m": float(np.max(thickness_2d[lia_mask])),
                        "harmonic_iterations": extension["iterations"],
                        "harmonic_maximum_update_m": extension["maximum_update_m"],
                        "centerline_rmse_m": quality["centerline_rmse_m"],
                        "boundary_max_thickness_m": quality["boundary_max_thickness_m"],
                        "finite_fraction_inside": quality["finite_fraction_inside"],
                        "input_centerline_constraint_count": constraint_meta[
                            "input_constraint_count"
                        ],
                        "pre_exclusion_raster_constraint_count": constraint_meta[
                            "raster_constraint_count"
                        ],
                        "same_branch_repeated_cell_count": constraint_meta[
                            "same_branch_repeated_cell_count"
                        ],
                    })
                    successful_arrays.append(thickness_2d.astype(np.float32))
                    successful_surfaces.append(
                        (bed.bed_m + thickness_2d).astype(np.float32)
                    )
                    successful_beds.append(bed.bed_m.astype(np.float32))
                    successful_member_ids.append(member_id)

        if successful_arrays:
            dataset = xr.Dataset(
                data_vars={
                    "thickness_m": (
                        ("member", "y", "x"), np.stack(successful_arrays)
                    ),
                    "surface_elevation_m": (
                        ("member", "y", "x"), np.stack(successful_surfaces)
                    ),
                    "bed_elevation_m": (
                        ("member", "y", "x"), np.stack(successful_beds)
                    ),
                    "lia_mask": (("y", "x"), lia_mask.astype(np.int8)),
                },
                coords={
                    "member": np.arange(len(successful_member_ids)),
                    "member_id": ("member", successful_member_ids),
                },
                attrs={
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "centerline_field": "raw-bed-farinotti2019-fixed-across-members",
                    "boundary_policy": "v0-zero-thickness-including-candidate-divides",
                    "tau_policy": "modern-product-specific-effective-stress-calibration",
                    "flowline_integration_refinement_factor": integration_factor,
                    "flowline_integration_policy": "piecewise-linear-bed-no-smoothing",
                },
            )
            encoding = {
                name: {"zlib": True, "complevel": 4}
                for name in ("thickness_m", "surface_elevation_m", "bed_elevation_m")
            }
            dataset.to_netcdf(
                output_dir / f"lia_{uid}_{rgi_id}_3d_{result_version}.nc",
                engine="netcdf4",
                encoding=encoding,
            )

    results = pd.DataFrame(rows)
    results.to_csv(
        output_dir / f"lia_3d_calibrated_results_{result_version}.csv", index=False
    )
    pd.DataFrame(head_rows).to_csv(
        output_dir / f"lia_head_boundary_classification_{result_version}.csv", index=False
    )
    successful = results[results["status"] == "success"].copy()
    sensitivity_rows: list[dict] = []
    if not successful.empty:
        for keys, group in successful.groupby(
            ["rgi_id", "lia_unique_id", "bed_scenario", "connector_alpha"]
        ):
            by_distance = group.set_index("head_exclusion_distance_m")["volume_km3"]
            reference_distance = float(min(by_distance.index))
            reference_volume = float(by_distance.loc[reference_distance])
            for distance_m, volume_km3 in by_distance.items():
                sensitivity_rows.append({
                    "rgi_id": keys[0],
                    "lia_unique_id": keys[1],
                    "bed_scenario": keys[2],
                    "connector_alpha": keys[3],
                    "head_exclusion_distance_m": float(distance_m),
                    "reference_distance_m": reference_distance,
                    "volume_km3": float(volume_km3),
                    "relative_volume_change_vs_min_distance": (
                        (float(volume_km3) - reference_volume) / reference_volume
                        if reference_volume > 0 else np.nan
                    ),
                })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        output_dir / f"lia_3d_head_exclusion_sensitivity_{result_version}.csv", index=False
    )
    summary = {
        "status": "completed-diagnostic-pilot",
        "pilot_systems": int(len(eligible)),
        "result_status_counts": results["status"].value_counts(dropna=False).to_dict(),
        "successful_systems": int(successful["rgi_id"].nunique()) if not successful.empty else 0,
        "successful_members": int(len(successful)),
        "flowline_integration_refinement_factor": integration_factor,
        "flowline_integration_policy": "piecewise-linear-bed-no-smoothing",
        "scientific_use": "calibrated-tau metric-head-exclusion pilot; not HMA production",
    }
    (output_dir / f"lia_3d_calibrated_summary_{result_version}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return results, sensitivity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    results, _ = run(json.loads(args.config.read_text(encoding="utf-8")))
    print(results["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

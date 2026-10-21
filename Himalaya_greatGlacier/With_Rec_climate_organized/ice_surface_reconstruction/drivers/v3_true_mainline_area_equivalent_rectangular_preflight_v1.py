"""Cycle-3 system-3850 true-mainline static and zero-MB preflight.

This isolated driver implements the Section-168 mechanism only.  It pairs the
anchor-reroute F19/V3 grid with the anchor-reroute stored LIA network, reduces
the geometry to the unique stored main branch, and never launches a historical
climate chain.  Every positive source cell is assigned once to a main-line
control volume.  Rectangular widths are area-equivalent.  The paired V3 bed is
sampled along the main line and thickness is obtained from a constrained
least-squares projection with exact total volume and a non-increasing surface.

The zero-MB gate is run only after every static gate passes, using the parent
single-flowline SemiImplicitModel and its recorded glen_a.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from .lia_dynamic_matrix_pilot import load_source_state
from .v3_flowline_geometry_audit import (
    maximum_cumulative_hypsometry_error_fraction,
    polygon_transverse_widths_m,
)


SUPPORTED_SYSTEM_ID = 3850
SCHEMA_VERSION = 1
MECHANISM = "true-mainline-area-equivalent-width-conservative-rectangular-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _load_network(path: Path) -> tuple[dict, dict, LineString]:
    with gzip.open(path, "rb") as stream:
        network = pickle.load(stream)
    if not isinstance(network, dict):
        raise ValueError("paired-network-is-not-a-dictionary")
    branches = network.get("branches")
    main_index = network.get("main_branch_index")
    if not isinstance(branches, list) or not branches:
        raise ValueError("paired-network-has-no-branches")
    if not isinstance(main_index, (int, np.integer)):
        raise ValueError("paired-network-main-branch-index-is-missing")
    matches = [
        branch for branch in branches
        if int(branch.get("branch_index", -1)) == int(main_index)
    ]
    if len(matches) != 1:
        raise ValueError(f"paired-network-main-branch-count:{len(matches)}")
    main = matches[0]
    if main.get("parent_index") is not None:
        raise ValueError("paired-network-stored-main-is-not-unique-root")
    other_roots = [
        branch for branch in branches
        if branch.get("parent_index") is None
        and int(branch.get("branch_index", -1)) != int(main_index)
    ]
    if other_roots:
        raise ValueError(f"paired-network-root-count:{1 + len(other_roots)}")
    coordinates = np.asarray(main["coords_projected_m"], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("paired-network-main-coordinates-invalid")
    if not np.isfinite(coordinates).all():
        raise ValueError("paired-network-main-coordinates-nonfinite")
    line = LineString(coordinates)
    if line.length <= 0:
        raise ValueError("paired-network-main-line-has-zero-length")
    return network, main, line


def _sample_nearest_source(source, nodes_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(source.valid_mask)
    tree = cKDTree(np.column_stack([source.x[cols], source.y[rows]]))
    distances, indices = tree.query(nodes_xy, k=1)
    rr = rows[np.asarray(indices, dtype=int)]
    cc = cols[np.asarray(indices, dtype=int)]
    return rr, cc, np.asarray(distances, dtype=float)


def _source_positive_cells(source, thickness_tolerance_m: float) -> dict:
    mask = source.valid_mask & (source.thickness_m > float(thickness_tolerance_m))
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("paired-source-has-no-positive-ice-cells")
    return {
        "mask": mask,
        "rows": rows,
        "cols": cols,
        "x": source.x[cols],
        "y": source.y[rows],
        "area_m2": np.full(len(rows), source.cell_area_m2, dtype=float),
        "thickness_m": source.thickness_m[rows, cols].astype(float),
        "surface_h_m": source.surface_h_m[rows, cols].astype(float),
        "bed_h_m": source.bed_h_m[rows, cols].astype(float),
    }


def _control_nodes(line: LineString, registered_length_m: float, target_dx_m: float) -> tuple[np.ndarray, float]:
    if abs(float(line.length) - registered_length_m) > 1e-6:
        raise ValueError(
            "stored-mainline-registered-length-conflict:"
            f"stored={line.length}:registered={registered_length_m}"
        )
    node_count = max(2, int(round(registered_length_m / target_dx_m)))
    dx_m = registered_length_m / node_count
    stations = (np.arange(node_count, dtype=float) + 0.5) * dx_m
    coordinates = np.array(
        [[line.interpolate(float(station)).x, line.interpolate(float(station)).y]
         for station in stations],
        dtype=float,
    )
    return coordinates, float(dx_m)


def _solve_thickness(
    *,
    bed_h_m: np.ndarray,
    sampled_thickness_m: np.ndarray,
    assigned_area_m2: np.ndarray,
    target_volume_m3: float,
    options: dict,
) -> tuple[np.ndarray, dict]:
    """Solve surface S with S>=bed, S[i]>=S[i+1], and exact A.H=V."""
    decimals = int(options["sample_rounding_decimals"])
    bed = np.round(np.asarray(bed_h_m, dtype=float), decimals)
    sampled_h = np.maximum(
        np.round(np.asarray(sampled_thickness_m, dtype=float), decimals), 0.0
    )
    area = np.asarray(assigned_area_m2, dtype=float)
    if np.any(area <= 0):
        raise ValueError("mainline-control-volume-has-nonpositive-area")
    if not np.isfinite(bed).all() or not np.isfinite(sampled_h).all():
        raise ValueError("sampled-bed-or-thickness-is-nonfinite")

    # Lowest surface satisfying both lower-bed and downstream-monotonic gates.
    minimum_surface = np.maximum.accumulate(bed[::-1])[::-1]
    minimum_volume = float(np.dot(area, minimum_surface - bed))
    feasibility_margin = float(target_volume_m3 - minimum_volume)
    if feasibility_margin < -float(options["volume_constraint_tolerance_fraction"]) * target_volume_m3:
        raise ValueError(
            "monotonic-bed-minimum-volume-exceeds-source-volume:"
            f"minimum={minimum_volume}:source={target_volume_m3}"
        )
    feasibility_margin = max(feasibility_margin, 0.0)
    feasible_surface = minimum_surface + feasibility_margin / float(area.sum())

    target_surface = bed + sampled_h
    weights = area / float(np.median(area))
    difference = np.zeros((len(area) - 1, len(area)), dtype=float)
    index = np.arange(len(area) - 1)
    difference[index, index] = 1.0
    difference[index, index + 1] = -1.0
    target_surface_sum = target_volume_m3 + float(np.dot(area, bed))
    constraints = [
        LinearConstraint(difference, 0.0, np.inf),
        LinearConstraint(area[np.newaxis, :], target_surface_sum, target_surface_sum),
    ]
    result = minimize(
        lambda candidate: 0.5 * float(np.sum(weights * np.square(candidate - target_surface))),
        feasible_surface,
        jac=lambda candidate: weights * (candidate - target_surface),
        bounds=[(float(value), None) for value in bed],
        constraints=constraints,
        method=str(options["method"]),
        options={
            "maxiter": int(options["maximum_iterations"]),
            "ftol": float(options["function_tolerance"]),
            "disp": False,
        },
    )
    surface = np.asarray(result.x, dtype=float)
    thickness = surface - bed
    reverse = np.diff(surface) > float(options["constraint_tolerance_m"])
    volume_residual = float(np.dot(area, thickness) - target_volume_m3)
    if not result.success:
        raise RuntimeError(f"thickness-optimization-failed:{result.status}:{result.message}")
    if np.any(reverse):
        raise RuntimeError(f"thickness-optimization-reverse-slope-count:{int(reverse.sum())}")
    if float(np.min(thickness)) < -float(options["constraint_tolerance_m"]):
        raise RuntimeError(f"thickness-optimization-negative-thickness:{float(np.min(thickness))}")
    if abs(volume_residual) > float(options["volume_constraint_tolerance_fraction"]) * target_volume_m3:
        raise RuntimeError(f"thickness-optimization-volume-residual:{volume_residual}")
    return np.maximum(thickness, 0.0), {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iteration_count": int(result.nit),
        "objective_value": float(result.fun),
        "minimum_feasible_volume_m3": minimum_volume,
        "source_minus_minimum_feasible_volume_m3": float(target_volume_m3 - minimum_volume),
        "volume_constraint_residual_m3": volume_residual,
        "minimum_surface_constraint_m": float(np.min(surface - bed)),
        "minimum_monotonic_constraint_m": float(np.min(surface[:-1] - surface[1:])),
        "sample_rounding_decimals": decimals,
    }


def _build_static(config: dict, config_path: Path) -> tuple[dict, pd.DataFrame, dict]:
    system_id = int(config["system_id"])
    if system_id != SUPPORTED_SYSTEM_ID:
        raise ValueError(f"v1-supports-only-system-{SUPPORTED_SYSTEM_ID}")
    source_path = _path(config["paired_source_grid"])
    network_path = _path(config["paired_network"])
    outline_path = _path(config["lia_outline"])
    reference_path = _path(config["length_references"])
    parent_path = _path(config["parent_chain_summary"])
    forbidden = "core191_production_candidate_v2"
    if forbidden in str(source_path) or forbidden in str(network_path):
        raise ValueError("forbidden-production-candidate-input-mix")
    if "core191_anchor_reroute_f19_v3" not in str(source_path):
        raise ValueError("source-is-not-anchor-reroute-f19-v3")
    if "lia_gdir_anchor_reroute_core191_v3" not in str(network_path):
        raise ValueError("network-is-not-anchor-reroute-core191-v3")

    source = load_source_state(source_path, str(config["bed_scenario"]))
    if int(source.lia_unique_id) != system_id:
        raise ValueError("paired-source-system-id-mismatch")
    network, main, line = _load_network(network_path)
    network_crs = CRS.from_user_input(network["crs"])
    if network_crs != source.crs:
        raise ValueError("paired-source-network-crs-mismatch")
    refs = pd.read_csv(reference_path)
    selected = refs.loc[refs["lia_unique_id"].astype(int).eq(system_id)]
    if len(selected) != 1 or str(selected.iloc[0]["reference_status"]) != "ok":
        raise ValueError(f"registered-reference-row-count-or-status:{len(selected)}")
    reference = selected.iloc[0]
    if str(reference["rgi_id"]) != str(source.rgi_id):
        raise ValueError("registered-reference-rgi-id-mismatch")
    registered_length_m = float(reference["lia_registered_length_m"])
    nodes_xy, node_dx_m = _control_nodes(
        line,
        registered_length_m,
        float(config["target_node_spacing_m"]),
    )

    positive = _source_positive_cells(
        source, float(config["thickness_positive_tolerance_m"])
    )
    node_tree = cKDTree(nodes_xy)
    assignment_distance, assignments = node_tree.query(
        np.column_stack([positive["x"], positive["y"]]), k=1
    )
    assignments = np.asarray(assignments, dtype=int)
    counts = np.bincount(assignments, minlength=len(nodes_xy))
    assigned_area = np.bincount(
        assignments, weights=positive["area_m2"], minlength=len(nodes_xy)
    ).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"empty-mainline-control-volume-count:{int(np.sum(counts == 0))}")
    sampled_rows, sampled_cols, sampled_distance = _sample_nearest_source(source, nodes_xy)
    sampled_bed = source.bed_h_m[sampled_rows, sampled_cols].astype(float)
    sampled_thickness = source.thickness_m[sampled_rows, sampled_cols].astype(float)
    target_volume_m3 = float(np.dot(positive["area_m2"], positive["thickness_m"]))
    solved_thickness, solver = _solve_thickness(
        bed_h_m=sampled_bed,
        sampled_thickness_m=sampled_thickness,
        assigned_area_m2=assigned_area,
        target_volume_m3=target_volume_m3,
        options=config["optimization"],
    )
    decimals = int(config["optimization"]["sample_rounding_decimals"])
    solved_bed = np.round(sampled_bed, decimals)
    solved_surface = solved_bed + solved_thickness
    width = assigned_area / node_dx_m
    section = width * solved_thickness

    outlines = gpd.read_file(outline_path)
    id_field = "UniqueID" if "UniqueID" in outlines.columns else "lia_unique_id"
    outline_rows = outlines.loc[outlines[id_field].astype(int).eq(system_id)]
    if len(outline_rows) != 1:
        raise ValueError(f"lia-outline-row-count:{len(outline_rows)}")
    outline = gpd.GeoSeries([outline_rows.iloc[0].geometry], crs=outlines.crs).to_crs(source.crs).iloc[0]
    outline_width = polygon_transverse_widths_m(
        outline, nodes_xy[:, 0], nodes_xy[:, 1]
    )

    node_frame = pd.DataFrame({
        "node_index": np.arange(len(nodes_xy), dtype=int),
        "station_center_m": (np.arange(len(nodes_xy), dtype=float) + 0.5) * node_dx_m,
        "node_dx_m": node_dx_m,
        "x_m": nodes_xy[:, 0],
        "y_m": nodes_xy[:, 1],
        "assigned_positive_cell_count": counts,
        "assigned_area_m2": assigned_area,
        "area_equivalent_width_m": width,
        "outline_transverse_width_m": outline_width,
        "sampled_source_row": sampled_rows,
        "sampled_source_col": sampled_cols,
        "sample_distance_m": sampled_distance,
        "sampled_bed_h_m_unrounded": sampled_bed,
        "sampled_thickness_m": sampled_thickness,
        "optimized_bed_h_m": solved_bed,
        "optimized_thickness_m": solved_thickness,
        "optimized_surface_h_m": solved_surface,
        "section_m2": section,
        "surface_adjustment_from_sample_m": solved_surface - (sampled_bed + sampled_thickness),
        "downstream_surface_drop_m": np.r_[solved_surface[:-1] - solved_surface[1:], np.nan],
    })

    source_area_m2 = float(np.sum(positive["area_m2"]))
    represented_area_m2 = float(np.dot(width, np.full(len(width), node_dx_m)))
    represented_volume_m3 = float(np.dot(section, np.full(len(section), node_dx_m)))
    hypsometry_error = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=positive["surface_h_m"],
        source_area_m2=positive["area_m2"],
        node_surface_m=solved_surface,
        model_area_m2=assigned_area,
        bin_size_m=100.0,
    )
    length_error = (len(nodes_xy) * node_dx_m - registered_length_m) / registered_length_m
    area_error = (represented_area_m2 - source_area_m2) / source_area_m2
    volume_error = (represented_volume_m3 - target_volume_m3) / target_volume_m3
    reverse_count = int(np.sum(np.diff(solved_surface) > float(config["optimization"]["constraint_tolerance_m"])))
    negative_thickness_count = int(np.sum(solved_thickness < -float(config["optimization"]["constraint_tolerance_m"])))
    nonfinite_count = int(np.sum(~np.isfinite(node_frame.select_dtypes(include=[np.number]).drop(columns=["downstream_surface_drop_m"]).to_numpy())))
    nonpositive_width_count = int(np.sum(width <= 0))
    nonpositive_section_count = int(np.sum(section <= 0))
    source_cell_count = int(len(assignments))
    assigned_cell_count = int(counts.sum())
    omitted_count = max(source_cell_count - assigned_cell_count, 0)
    duplicated_count = max(assigned_cell_count - source_cell_count, 0)

    gates = config["static_gates"]
    flags = []
    checks = {
        "paired_provenance": True,
        "unique_root_main_crosswalk": True,
        "registered_length_closure": abs(length_error) <= float(gates["length_relative_error_max"]),
        "no_positive_cell_omission": omitted_count <= int(gates["omitted_positive_cell_count_max"]),
        "no_positive_cell_duplication": duplicated_count <= int(gates["duplicated_positive_cell_count_max"]),
        "area_conservation": abs(area_error) <= float(gates["area_relative_error_max"]),
        "volume_conservation": abs(volume_error) <= float(gates["volume_relative_error_max"]),
        "hypsometry_100m": hypsometry_error <= float(gates["hypsometry_100m_cumulative_error_max"]),
        "zero_reverse_surface_slopes": reverse_count <= int(gates["reverse_surface_slope_count_max"]),
        "finite_positive_widths_sections": nonfinite_count == 0 and nonpositive_width_count == 0 and nonpositive_section_count == 0,
        "nonnegative_thickness": negative_thickness_count <= int(gates["negative_thickness_count_max"]),
    }
    flags.extend(name for name, passed in checks.items() if not passed)
    adjustments = node_frame["surface_adjustment_from_sample_m"].to_numpy(dtype=float)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_parameters = parent["parameters"]
    parent_glen_a = float(parent_parameters["glen_a_pa_3_s_1"])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": MECHANISM,
        "status": "static-pass" if not flags else "static-fail",
        "review_flags": flags,
        "system_id": system_id,
        "rgi_id": str(source.rgi_id),
        "paired_source_grid": str(source_path.resolve()),
        "paired_network": str(network_path.resolve()),
        "network_version": str(network.get("network_version")),
        "stored_main_branch_index": int(network["main_branch_index"]),
        "stored_branch_count": int(len(network["branches"])),
        "stored_main_coordinate_count": int(len(main["coords_projected_m"])),
        "stored_mainline_length_m": float(line.length),
        "registered_lia_length_m": registered_length_m,
        "control_node_count": int(len(nodes_xy)),
        "control_node_dx_m": node_dx_m,
        "represented_length_m": float(len(nodes_xy) * node_dx_m),
        "length_relative_error": float(length_error),
        "source_positive_cell_count": source_cell_count,
        "assigned_positive_cell_count": assigned_cell_count,
        "omitted_positive_cell_count": omitted_count,
        "duplicated_positive_cell_count": duplicated_count,
        "minimum_cells_per_control_volume": int(counts.min()),
        "maximum_cells_per_control_volume": int(counts.max()),
        "maximum_cell_assignment_distance_m": float(np.max(assignment_distance)),
        "source_area_m2": source_area_m2,
        "represented_area_m2": represented_area_m2,
        "area_relative_error": float(area_error),
        "source_volume_m3": target_volume_m3,
        "represented_volume_m3": represented_volume_m3,
        "volume_relative_error": float(volume_error),
        "hypsometry_bin_size_m": 100.0,
        "hypsometry_maximum_cumulative_error_fraction": float(hypsometry_error),
        "reverse_surface_slope_count": reverse_count,
        "negative_thickness_count": negative_thickness_count,
        "nonfinite_numeric_count": nonfinite_count,
        "nonpositive_width_count": nonpositive_width_count,
        "nonpositive_section_count": nonpositive_section_count,
        "maximum_absolute_surface_adjustment_m": float(np.max(np.abs(adjustments))),
        "area_weighted_mean_absolute_surface_adjustment_m": float(np.average(np.abs(adjustments), weights=assigned_area)),
        "maximum_sample_distance_m": float(np.max(sampled_distance)),
        "solver": solver,
        "checks": checks,
        "gates": gates,
        "parent_solver": "SemiImplicitModel",
        "parent_glen_a_pa_3_s_1": parent_glen_a,
        "parent_fs": float(config["zero_mb"]["fs"]),
        "source_products_mutated": False,
        "historical_chain_authorized": False,
        "next_gate": "semiimplicit-zero-mb-0.1-and-5y" if not flags else "stop-cycle3-static-failed",
    }
    assignment_payload = {
        "source_row": positive["rows"].astype(np.int32),
        "source_col": positive["cols"].astype(np.int32),
        "control_node_index": assignments.astype(np.int32),
        "assignment_distance_m": np.asarray(assignment_distance, dtype=np.float64),
    }
    context = {
        "source": source,
        "node_frame": node_frame,
        "parent_glen_a": parent_glen_a,
        "assignment_payload": assignment_payload,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": MECHANISM,
        "system_id": system_id,
        "production_candidate_mixed": False,
        "inputs": [_input_record(path) for path in [config_path, source_path, network_path, outline_path, reference_path, parent_path]],
    }
    return summary, node_frame, {"context": context, "manifest": manifest}


def _flowline_from_nodes(node_frame: pd.DataFrame, config: dict):
    from oggm.core.flowline import RectangularBedFlowline

    dx_m = float(node_frame["node_dx_m"].iloc[0])
    ice_nodes = len(node_frame)
    buffer_nodes = int(np.ceil(float(config["downstream_safety_buffer_m"]) / dx_m))
    bed = node_frame["optimized_bed_h_m"].to_numpy(dtype=float)
    surface = node_frame["optimized_surface_h_m"].to_numpy(dtype=float)
    width = node_frame["area_equivalent_width_m"].to_numpy(dtype=float)
    extension_distance = np.arange(1, buffer_nodes + 1, dtype=float) * dx_m
    extension_bed = bed[-1] - float(config["downstream_bed_slope"]) * extension_distance
    extension_width = np.full(buffer_nodes, width[-1], dtype=float)
    flowline = RectangularBedFlowline(
        line=None,
        dx=dx_m,
        map_dx=1.0,
        surface_h=np.concatenate([surface, extension_bed]),
        bed_h=np.concatenate([bed, extension_bed]),
        widths=np.concatenate([width, extension_width]),
        rgi_id=f"LIA-{int(config['system_id'])}-TRUE-MAINLINE-V1",
    )
    return flowline, ice_nodes, buffer_nodes


def _run_zero_mb(node_frame: pd.DataFrame, config: dict, parent_glen_a: float) -> tuple[dict, pd.DataFrame, object]:
    import oggm
    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import ScalarMassBalance

    cfg.initialize(logging_level="WARNING")
    flowline, ice_nodes, buffer_nodes = _flowline_from_nodes(node_frame, config)
    initial_volume = float(flowline.volume_m3)
    initial_area = float(flowline.area_m2)
    initial_sections = np.asarray(flowline.section, dtype=float).copy()
    model = SemiImplicitModel(
        [flowline],
        mb_model=ScalarMassBalance(mb=0.0),
        y0=0.0,
        glen_a=float(parent_glen_a),
        fs=float(config["zero_mb"]["fs"]),
        check_for_boundaries=bool(config["zero_mb"]["check_for_boundaries"]),
    )
    rows = []
    failure_flags = []
    started = time.perf_counter()
    for year in [float(value) for value in config["zero_mb"]["checkpoints_years"]]:
        model.run_until(year)
        fl = model.fls[0]
        section = np.asarray(fl.section, dtype=float)
        thickness = np.asarray(fl.thick, dtype=float)
        volume = float(fl.volume_m3)
        area = float(fl.area_m2)
        drift = (volume - initial_volume) / initial_volume
        nonfinite = int(np.sum(~np.isfinite(section)) + np.sum(~np.isfinite(thickness)))
        negative_section = int(np.sum(section < -1e-12))
        terminal_thickness = float(thickness[-1])
        observation_terminal_thickness = float(thickness[ice_nodes - 1])
        row = {
            "year": year,
            "volume_m3": volume,
            "volume_drift_fraction": drift,
            "area_m2": area,
            "area_change_fraction": (area - initial_area) / initial_area,
            "terminal_thickness_m": terminal_thickness,
            "observation_prefix_terminal_thickness_m": observation_terminal_thickness,
            "minimum_section_m2": float(np.min(section)),
            "maximum_section_m2": float(np.max(section)),
            "nonfinite_state_count": nonfinite,
            "negative_section_count": negative_section,
            "maximum_absolute_section_change_m2": float(np.max(np.abs(section - initial_sections))),
        }
        rows.append(row)
        if abs(drift) > float(config["zero_mb"]["volume_drift_absolute_fraction_max"]):
            failure_flags.append(f"volume-drift-over-gate-at-{year:g}y")
        if terminal_thickness > float(config["zero_mb"]["terminal_thickness_m_max"]):
            failure_flags.append(f"terminal-thickness-over-gate-at-{year:g}y")
        if nonfinite:
            failure_flags.append(f"nonfinite-state-at-{year:g}y")
        if negative_section:
            failure_flags.append(f"negative-section-at-{year:g}y")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": MECHANISM,
        "status": "zero-mb-pass" if not failure_flags else "zero-mb-fail",
        "review_flags": list(dict.fromkeys(failure_flags)),
        "system_id": int(config["system_id"]),
        "solver": "SemiImplicitModel",
        "solver_change": False,
        "mass_balance_model": "ScalarMassBalance(mb=0.0)",
        "oggm_version": str(oggm.__version__),
        "python_executable": str(Path(sys.executable).resolve()),
        "glen_a_pa_3_s_1": float(parent_glen_a),
        "fs": float(config["zero_mb"]["fs"]),
        "initial_volume_m3": initial_volume,
        "initial_area_m2": initial_area,
        "ice_control_node_count": int(ice_nodes),
        "downstream_buffer_node_count": int(buffer_nodes),
        "downstream_buffer_m": float(buffer_nodes * float(node_frame["node_dx_m"].iloc[0])),
        "checkpoints": rows,
        "gates": config["zero_mb"],
        "wall_time_seconds": float(time.perf_counter() - started),
        "technical_failure": False,
        "source_products_mutated": False,
        "historical_chain_authorized": False,
        "next_gate": "representative-selection-separate-decision" if not failure_flags else "stop-cycle3-q1-mechanism-zero-mb-failed",
    }
    return summary, pd.DataFrame(rows), model.fls[0]


def run(config_path: Path) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = _path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"independent-output-already-exists:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        static_summary, node_frame, bundle = _build_static(config, config_path)
        node_frame.to_csv(output_root / "true_mainline_nodes.csv", index=False)
        np.savez_compressed(output_root / "positive_cell_assignment.npz", **bundle["context"]["assignment_payload"])
        _write_json(output_root / "input_manifest.json", bundle["manifest"])
        _write_json(output_root / "static_preflight_summary.json", static_summary)
        if static_summary["status"] != "static-pass":
            decision = {
                "status": "closed-static-fail",
                "system_id": int(config["system_id"]),
                "review_flags": static_summary["review_flags"],
                "zero_mb_run": False,
                "historical_chain_run": False,
                "historical_chain_authorized": False,
            }
            _write_json(output_root / "decision.json", decision)
            return decision

        zero_summary, checkpoints, final_flowline = _run_zero_mb(
            node_frame, config, bundle["context"]["parent_glen_a"]
        )
        checkpoints.to_csv(output_root / "zero_mb_checkpoints.csv", index=False)
        with gzip.open(output_root / "true_mainline_flowline_zero_mb_final.pkl.gz", "wb") as stream:
            pickle.dump(final_flowline, stream)
        _write_json(output_root / "zero_mb_summary.json", zero_summary)
        decision = {
            "status": "preflight-pass" if zero_summary["status"] == "zero-mb-pass" else "closed-zero-mb-fail",
            "system_id": int(config["system_id"]),
            "static_status": static_summary["status"],
            "zero_mb_status": zero_summary["status"],
            "review_flags": zero_summary["review_flags"],
            "historical_chain_run": False,
            "historical_chain_authorized": False,
        }
        _write_json(output_root / "decision.json", decision)
        return decision
    except Exception as exc:
        failure = {
            "status": "technical-fail",
            "system_id": int(config.get("system_id", SUPPORTED_SYSTEM_ID)),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(limit=20),
            "historical_chain_run": False,
            "historical_chain_authorized": False,
        }
        _write_json(output_root / "decision.json", _json_safe(failure))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()

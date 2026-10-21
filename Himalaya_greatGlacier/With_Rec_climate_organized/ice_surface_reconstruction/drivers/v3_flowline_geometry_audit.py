"""Audit conservative 2-D to v3 flowline cross-section mapping for core191."""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import os
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree
from shapely import wkb
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import transform as transform_geometry

from ..paths import resolve
from .lia_dynamic_flowlines import derive_fixed_bed_trapezoid_geometry
from .lia_dynamic_matrix_pilot import load_source_state
from .lia_system_grid import normalize_lia_components


POSITIVE_ICE_TOLERANCE_M = 1e-6
MINIMUM_MAPPED_CENTERLINE_THICKNESS_M = 1.0
BOUNDARY_NODE_EXCLUSION_M = 100.0
VOLUME_GATE_FRACTION = 0.005
AREA_GATE_FRACTION = 0.02
HYPSOMETRY_GATE_FRACTION = 0.02


def flowline_control_lengths_m(coordinates_xy: np.ndarray) -> np.ndarray:
    """Return half-segment control lengths for one physical polyline."""
    coordinates = np.asarray(coordinates_xy, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("flowline-coordinates-must-be-n-by-2-with-n-at-least-2")
    segments = np.hypot(
        np.diff(coordinates[:, 0]), np.diff(coordinates[:, 1])
    )
    if not np.isfinite(segments).all() or np.any(segments <= 0):
        raise ValueError("flowline-has-nonpositive-or-nonfinite-segment")
    lengths = np.empty(len(coordinates), dtype=float)
    lengths[0] = 0.5 * segments[0]
    lengths[-1] = 0.5 * segments[-1]
    if len(coordinates) > 2:
        lengths[1:-1] = 0.5 * (segments[:-1] + segments[1:])
    return lengths


def polygon_transverse_widths_m(
    coordinates_xy: np.ndarray,
    component_geometry,
    *,
    search_half_length_m: float | None = None,
) -> np.ndarray:
    """Measure the local SHP width normal to a flowline at every node."""
    coordinates = np.asarray(coordinates_xy, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("transverse-width-coordinates-must-be-n-by-2")
    if search_half_length_m is None:
        minx, miny, maxx, maxy = component_geometry.bounds
        search_half_length_m = float(np.hypot(maxx - minx, maxy - miny))
    if search_half_length_m <= 0:
        raise ValueError("transverse-width-search-length-must-be-positive")

    def line_parts(geometry):
        if isinstance(geometry, LineString):
            return [geometry]
        if isinstance(geometry, (MultiLineString, GeometryCollection)):
            output = []
            for part in geometry.geoms:
                output.extend(line_parts(part))
            return output
        return []

    widths = np.zeros(len(coordinates), dtype=float)
    for index, point_xy in enumerate(coordinates):
        if index == 0:
            tangent = coordinates[1] - coordinates[0]
        elif index == len(coordinates) - 1:
            tangent = coordinates[-1] - coordinates[-2]
        else:
            tangent = coordinates[index + 1] - coordinates[index - 1]
        norm = float(np.linalg.norm(tangent))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"transverse-width-invalid-tangent:{index}")
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float) / norm
        transect = LineString([
            point_xy - normal * search_half_length_m,
            point_xy + normal * search_half_length_m,
        ])
        parts = line_parts(transect.intersection(component_geometry))
        if not parts:
            continue
        point = Point(point_xy)
        selected = min(parts, key=lambda part: (part.distance(point), -part.length))
        widths[index] = float(selected.length)
    return widths


def conservative_node_assignment(
    *,
    cell_xy_m: np.ndarray,
    cell_thickness_m: np.ndarray,
    cell_surface_m: np.ndarray,
    node_xy_m: np.ndarray,
    node_thickness_m: np.ndarray,
    node_control_length_m: np.ndarray,
    cell_area_m2: float,
    maximum_node_to_ice_distance_m: float,
    eligible_node_mask: np.ndarray | None = None,
) -> dict:
    """Assign every positive-ice cell once to a nearby positive node."""
    cells = np.asarray(cell_xy_m, dtype=float)
    cell_h = np.asarray(cell_thickness_m, dtype=float)
    cell_s = np.asarray(cell_surface_m, dtype=float)
    nodes = np.asarray(node_xy_m, dtype=float)
    node_h = np.asarray(node_thickness_m, dtype=float)
    control_length = np.asarray(node_control_length_m, dtype=float)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cell-coordinates-must-be-n-by-2")
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("node-coordinates-must-be-n-by-2")
    if not (len(cells) == len(cell_h) == len(cell_s)):
        raise ValueError("cell-mapping-array-length-mismatch")
    if not (len(nodes) == len(node_h) == len(control_length)):
        raise ValueError("node-mapping-array-length-mismatch")
    if cell_area_m2 <= 0 or maximum_node_to_ice_distance_m <= 0:
        raise ValueError("mapping-spatial-scales-must-be-positive")
    if (
        not np.isfinite(cells).all()
        or not np.isfinite(cell_h).all()
        or not np.isfinite(cell_s).all()
    ):
        raise ValueError("mapping-cells-contain-nonfinite-values")
    if not np.isfinite(nodes).all() or not np.isfinite(node_h).all():
        raise ValueError("mapping-nodes-contain-nonfinite-values")
    if np.any(control_length <= 0):
        raise ValueError("node-control-length-must-be-positive")

    positive_cells = cell_h > POSITIVE_ICE_TOLERANCE_M
    if not positive_cells.any():
        raise ValueError("component-has-no-positive-ice-cells")
    cells = cells[positive_cells]
    cell_h = cell_h[positive_cells]
    cell_s = cell_s[positive_cells]
    positive_nodes = node_h > POSITIVE_ICE_TOLERANCE_M
    if not positive_nodes.any():
        raise ValueError("component-has-no-positive-centerline-nodes")

    cell_tree = cKDTree(cells)
    distance_to_ice, _ = cell_tree.query(nodes, k=1)
    eligible = positive_nodes & (
        distance_to_ice <= maximum_node_to_ice_distance_m
    )
    if eligible_node_mask is not None:
        requested = np.asarray(eligible_node_mask, dtype=bool)
        if requested.shape != eligible.shape:
            raise ValueError("eligible-node-mask-shape-mismatch")
        eligible &= requested
    if not eligible.any():
        raise ValueError("component-has-no-eligible-centerline-nodes")
    eligible_indices = np.flatnonzero(eligible)
    node_tree = cKDTree(nodes[eligible])
    _, local_assignment = node_tree.query(cells, k=1)
    assignment = eligible_indices[np.asarray(local_assignment, dtype=int)]

    cell_count = np.bincount(assignment, minlength=len(nodes)).astype(int)
    node_area = cell_count.astype(float) * float(cell_area_m2)
    node_volume = np.bincount(
        assignment,
        weights=cell_h * float(cell_area_m2),
        minlength=len(nodes),
    )
    node_surface_moment = np.bincount(
        assignment,
        weights=cell_s * float(cell_area_m2),
        minlength=len(nodes),
    )
    node_surface_squared_moment = np.bincount(
        assignment,
        weights=np.square(cell_s) * float(cell_area_m2),
        minlength=len(nodes),
    )
    node_surface_minimum = np.full(len(nodes), np.inf, dtype=float)
    node_surface_maximum = np.full(len(nodes), -np.inf, dtype=float)
    np.minimum.at(node_surface_minimum, assignment, cell_s)
    np.maximum.at(node_surface_maximum, assignment, cell_s)
    mapped = cell_count > 0
    node_surface_minimum[~mapped] = np.nan
    node_surface_maximum[~mapped] = np.nan
    section = np.zeros(len(nodes), dtype=float)
    area_equivalent_width = np.zeros(len(nodes), dtype=float)
    section[mapped] = node_volume[mapped] / control_length[mapped]
    area_equivalent_width[mapped] = node_area[mapped] / control_length[mapped]
    return {
        "positive_cell_xy_m": cells,
        "positive_cell_thickness_m": cell_h,
        "positive_cell_surface_m": cell_s,
        "assignment": assignment,
        "eligible_node": eligible,
        "mapped_node": mapped,
        "node_cell_count": cell_count,
        "node_area_m2": node_area,
        "node_volume_m3": node_volume,
        "node_surface_moment_m3": node_surface_moment,
        "node_surface_squared_moment_m4": node_surface_squared_moment,
        "node_surface_minimum_m": node_surface_minimum,
        "node_surface_maximum_m": node_surface_maximum,
        "section_m2": section,
        "area_equivalent_width_m": area_equivalent_width,
        "excluded_positive_node_count": int((positive_nodes & ~eligible).sum()),
        "unrepresented_positive_node_count": int((eligible & ~mapped).sum()),
    }


def maximum_cumulative_hypsometry_error_fraction(
    *,
    source_surface_m: np.ndarray,
    source_area_m2: np.ndarray,
    node_surface_m: np.ndarray,
    model_area_m2: np.ndarray,
    bin_size_m: float = 100.0,
) -> float:
    source_z = np.asarray(source_surface_m, dtype=float)
    source_a = np.asarray(source_area_m2, dtype=float)
    node_z = np.asarray(node_surface_m, dtype=float)
    model_a = np.asarray(model_area_m2, dtype=float)
    if bin_size_m <= 0 or source_a.sum() <= 0:
        raise ValueError("hypsometry-scales-must-be-positive")
    lower = np.floor(min(source_z.min(), node_z.min()) / bin_size_m) * bin_size_m
    upper = np.ceil(max(source_z.max(), node_z.max()) / bin_size_m) * bin_size_m
    edges = np.arange(lower, upper + 1.5 * bin_size_m, bin_size_m)
    source_hist, _ = np.histogram(source_z, bins=edges, weights=source_a)
    model_hist, _ = np.histogram(node_z, bins=edges, weights=model_a)
    source_cumulative = np.cumsum(source_hist[::-1])
    model_cumulative = np.cumsum(model_hist[::-1])
    return float(
        np.max(np.abs(model_cumulative - source_cumulative)) / source_a.sum()
    )


def _transform_coordinates(
    coordinates: np.ndarray, source_crs: CRS, target_crs: CRS
) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if source_crs == target_crs:
        return values
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(values[:, 0], values[:, 1])
    return np.column_stack([x, y])


def _transform_shape(geometry, source_crs: CRS, target_crs: CRS):
    if source_crs == target_crs:
        return geometry
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transform_geometry(transformer.transform, geometry)


def _sample_nearest(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(valid & np.isfinite(values))
    if not len(rows):
        raise ValueError("nearest-sampling-source-is-empty")
    points = np.column_stack([x[cols], y[rows]])
    distance, index = cKDTree(points).query(coordinates, k=1)
    return np.asarray(values[rows[index], cols[index]], float), np.asarray(distance, float)


def _junction_audit(branches: list[dict], source_crs: CRS, target_crs: CRS, dx_m: float):
    by_index = {int(branch["branch_index"]): branch for branch in branches}
    rows = []
    for branch in branches:
        parent_index = branch.get("parent_index")
        if parent_index is None:
            continue
        coordinates = _transform_coordinates(
            branch["coords_projected_m"], source_crs, target_crs
        )
        parent = by_index[int(parent_index)]
        parent_coordinates = _transform_coordinates(
            parent["coords_projected_m"], source_crs, target_crs
        )
        junction = np.asarray(branch.get("flows_to_point_projected_m"), float)[None, :]
        junction = _transform_coordinates(junction, source_crs, target_crs)[0]
        declared_index = int(np.argmin(np.hypot(
            parent_coordinates[:, 0] - junction[0],
            parent_coordinates[:, 1] - junction[1],
        )))
        terminal = coordinates[-1]
        contact_distances = np.hypot(
            parent_coordinates[:, 0] - terminal[0],
            parent_coordinates[:, 1] - terminal[1],
        )
        contact_index = int(np.argmin(contact_distances))
        override = float(contact_distances[contact_index]) <= dx_m
        effective_index = contact_index if override else declared_index
        rows.append({
            "branch_index": int(branch["branch_index"]),
            "parent_index": int(parent_index),
            "declared_parent_node_index": declared_index,
            "effective_parent_node_index": effective_index,
            "geometric_contact_override": bool(override and contact_index != declared_index),
            "terminal_to_parent_contact_distance_m": float(contact_distances[contact_index]),
            "terminal_to_effective_junction_distance_m": float(np.linalg.norm(
                terminal - parent_coordinates[effective_index]
            )),
        })
    return rows


def _component_mapping(
    *,
    source,
    component_geometry,
    network_path: Path,
    component_index: int,
) -> tuple[list[dict], dict, list[dict]]:
    component_mask = geometry_mask(
        [component_geometry.__geo_interface__],
        out_shape=source.shape,
        transform=source.transform,
        invert=True,
        all_touched=True,
    )
    valid = source.valid_mask & component_mask
    positive = valid & (source.thickness_m > POSITIVE_ICE_TOLERANCE_M)
    if not positive.any():
        raise ValueError("component-has-no-positive-source-ice")
    with gzip.open(network_path, "rb") as stream:
        network = pickle.load(stream)
    network_crs = CRS.from_user_input(network["crs"])
    branches = list(network["branches"])
    main_index = int(network.get("main_branch_index", len(branches) - 1))

    node_rows = []
    coordinates = []
    control_lengths = []
    transverse_widths = []
    for branch in branches:
        branch_coordinates = _transform_coordinates(
            branch["coords_projected_m"], network_crs, source.crs
        )
        lengths = flowline_control_lengths_m(branch_coordinates)
        widths = polygon_transverse_widths_m(
            branch_coordinates, component_geometry
        )
        for node_index, (point, control_length) in enumerate(
            zip(branch_coordinates, lengths)
        ):
            node_rows.append({
                "component_index": int(component_index),
                "branch_index": int(branch["branch_index"]),
                "node_index": int(node_index),
                "is_main_branch": int(branch["branch_index"]) == main_index,
                "parent_branch_index": (
                    None if branch.get("parent_index") is None
                    else int(branch["parent_index"])
                ),
                "x_m": float(point[0]),
                "y_m": float(point[1]),
                "control_length_m": float(control_length),
            })
            coordinates.append(point)
            control_lengths.append(control_length)
            transverse_widths.append(widths[node_index])
    coordinates = np.asarray(coordinates, dtype=float)
    control_lengths = np.asarray(control_lengths, dtype=float)
    transverse_widths = np.asarray(transverse_widths, dtype=float)
    sampled_h, node_distance = _sample_nearest(
        source.x, source.y, source.thickness_m, valid, coordinates
    )
    sampled_s, _ = _sample_nearest(
        source.x, source.y, source.surface_h_m, valid, coordinates
    )
    sampled_b, _ = _sample_nearest(
        source.x, source.y, source.bed_h_m, valid, coordinates
    )
    sampled_h = sampled_s - sampled_b
    boundary_distance = np.asarray([
        component_geometry.boundary.distance(Point(point))
        for point in coordinates
    ], dtype=float)
    eligible_by_boundary = (
        boundary_distance >= BOUNDARY_NODE_EXCLUSION_M
    ) & (sampled_h >= MINIMUM_MAPPED_CENTERLINE_THICKNESS_M)

    cell_rows, cell_cols = np.nonzero(positive)
    cell_xy = np.column_stack([source.x[cell_cols], source.y[cell_rows]])
    assignment = conservative_node_assignment(
        cell_xy_m=cell_xy,
        cell_thickness_m=source.thickness_m[cell_rows, cell_cols],
        cell_surface_m=source.surface_h_m[cell_rows, cell_cols],
        node_xy_m=coordinates,
        node_thickness_m=sampled_h,
        node_control_length_m=control_lengths,
        cell_area_m2=source.cell_area_m2,
        maximum_node_to_ice_distance_m=1.5 * np.sqrt(source.cell_area_m2),
        eligible_node_mask=eligible_by_boundary,
    )
    mapped = assignment["mapped_node"]
    invalid_width = mapped & (~np.isfinite(transverse_widths) | (transverse_widths <= 0))
    if invalid_width.any():
        bad = np.flatnonzero(invalid_width)
        raise ValueError(
            "mapped-node-has-no-positive-polygon-transverse-width:"
            + ",".join(map(str, bad[:20].tolist()))
        )
    geometry = derive_fixed_bed_trapezoid_geometry(
        sampled_h[mapped],
        assignment["section_m2"][mapped],
        assignment["area_equivalent_width_m"][mapped],
    )
    strip_mean_surface = np.full(len(coordinates), np.nan, dtype=float)
    strip_surface_standard_deviation = np.full(len(coordinates), np.nan, dtype=float)
    strip_mean_surface[mapped] = (
        assignment["node_surface_moment_m3"][mapped]
        / assignment["node_area_m2"][mapped]
    )
    strip_surface_variance = (
        assignment["node_surface_squared_moment_m4"][mapped]
        / assignment["node_area_m2"][mapped]
        - np.square(strip_mean_surface[mapped])
    )
    strip_surface_standard_deviation[mapped] = np.sqrt(
        np.maximum(strip_surface_variance, 0.0)
    )
    mapped_indices = np.flatnonzero(mapped)
    geometry_by_node = {index: local for local, index in enumerate(mapped_indices)}
    for index, row in enumerate(node_rows):
        local = geometry_by_node.get(index)
        row.update({
            "sample_distance_to_ice_m": float(node_distance[index]),
            "source_surface_m": float(sampled_s[index]),
            "source_bed_m": float(sampled_b[index]),
            "source_centerline_thickness_m": float(sampled_h[index]),
            "distance_to_lia_boundary_m": float(boundary_distance[index]),
            "excluded_by_boundary_or_minimum_thickness": bool(
                not eligible_by_boundary[index]
            ),
            "eligible_positive_node": bool(assignment["eligible_node"][index]),
            "mapping_active": bool(mapped[index]),
            "assigned_cell_count": int(assignment["node_cell_count"][index]),
            "assigned_area_m2": float(assignment["node_area_m2"][index]),
            "assigned_volume_m3": float(assignment["node_volume_m3"][index]),
            "strip_mean_surface_m": float(strip_mean_surface[index]),
            "node_minus_strip_mean_surface_m": float(
                sampled_s[index] - strip_mean_surface[index]
            ),
            "strip_surface_standard_deviation_m": float(
                strip_surface_standard_deviation[index]
            ),
            "strip_surface_minimum_m": float(
                assignment["node_surface_minimum_m"][index]
            ),
            "strip_surface_maximum_m": float(
                assignment["node_surface_maximum_m"][index]
            ),
            "strip_surface_range_m": float(
                assignment["node_surface_maximum_m"][index]
                - assignment["node_surface_minimum_m"][index]
            ),
            "section_m2": float(assignment["section_m2"][index]),
            "area_equivalent_surface_width_m": float(
                assignment["area_equivalent_width_m"][index]
            ),
            "polygon_transverse_width_m": float(transverse_widths[index]),
            "area_shape_factor": (
                float(geometry.area_shape_factor[local]) if local is not None else np.nan
            ),
            "lambda": float(geometry.lambdas[local]) if local is not None else np.nan,
            "bottom_width_m": (
                float(geometry.bottom_width_m[local]) if local is not None else np.nan
            ),
            "effective_surface_width_m": (
                float(geometry.effective_surface_width_m[local])
                if local is not None else np.nan
            ),
            "width_error_fraction": (
                float(geometry.width_error_fraction[local])
                if local is not None else np.nan
            ),
            "regime_code": int(geometry.regime_code[local]) if local is not None else 0,
        })

    source_area_m2 = float(positive.sum() * source.cell_area_m2)
    source_volume_m3 = float(np.sum(source.thickness_m[positive]) * source.cell_area_m2)
    observed_area_m2 = float(np.sum(
        assignment["area_equivalent_width_m"][mapped] * control_lengths[mapped]
    ))
    polygon_transverse_area_m2 = float(np.sum(
        transverse_widths[mapped] * control_lengths[mapped]
    ))
    mapped_volume_m3 = float(np.sum(
        assignment["section_m2"][mapped] * control_lengths[mapped]
    ))
    effective_node_area = geometry.effective_surface_width_m * control_lengths[mapped]
    effective_area_m2 = float(effective_node_area.sum())
    observed_node_area = assignment["node_area_m2"][mapped]
    source_surface = source.surface_h_m[positive]
    cell_area = np.full(source_surface.shape, source.cell_area_m2, dtype=float)
    observed_hypsometry_error = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=source_surface,
        source_area_m2=cell_area,
        node_surface_m=sampled_s[mapped],
        model_area_m2=observed_node_area,
    )
    strip_mean_hypsometry_error = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=source_surface,
        source_area_m2=cell_area,
        node_surface_m=strip_mean_surface[mapped],
        model_area_m2=observed_node_area,
    )
    effective_hypsometry_error = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=source_surface,
        source_area_m2=cell_area,
        node_surface_m=sampled_s[mapped],
        model_area_m2=effective_node_area,
    )
    source_moment = float(np.sum(source_surface * source.cell_area_m2))
    effective_moment = float(np.sum(sampled_s[mapped] * effective_node_area))
    summary = {
        "component_index": int(component_index),
        "vector_component_area_km2": float(component_geometry.area / 1e6),
        "branch_count": int(len(branches)),
        "node_count": int(len(node_rows)),
        "mapped_node_count": int(mapped.sum()),
        "unrepresented_positive_node_count": int(
            assignment["unrepresented_positive_node_count"]
        ),
        "excluded_positive_node_count": int(
            assignment["excluded_positive_node_count"]
        ),
        "source_positive_area_km2": source_area_m2 / 1e6,
        "source_volume_km3": source_volume_m3 / 1e9,
        "observed_flowline_area_km2": observed_area_m2 / 1e6,
        "polygon_transverse_flowline_area_km2": polygon_transverse_area_m2 / 1e6,
        "effective_flowline_area_km2": effective_area_m2 / 1e6,
        "mapped_volume_km3": mapped_volume_m3 / 1e9,
        "observed_area_error_fraction": (
            observed_area_m2 - source_area_m2
        ) / source_area_m2,
        "polygon_transverse_area_error_fraction": (
            polygon_transverse_area_m2 - source_area_m2
        ) / source_area_m2,
        "effective_area_error_fraction": (
            effective_area_m2 - source_area_m2
        ) / source_area_m2,
        "volume_error_fraction": (
            mapped_volume_m3 - source_volume_m3
        ) / source_volume_m3,
        "exact_width_regime_fraction": float(np.mean(geometry.regime_code == 2)),
        "low_shape_factor_regime_fraction": float(np.mean(geometry.regime_code == 1)),
        "high_shape_factor_regime_fraction": float(np.mean(geometry.regime_code == 3)),
        "shape_factor_minimum": float(np.min(geometry.area_shape_factor)),
        "shape_factor_median": float(np.median(geometry.area_shape_factor)),
        "shape_factor_maximum": float(np.max(geometry.area_shape_factor)),
        "lambda_minimum": float(np.min(geometry.lambdas)),
        "lambda_median": float(np.median(geometry.lambdas)),
        "lambda_maximum": float(np.max(geometry.lambdas)),
        "bottom_width_minimum_m": float(np.min(geometry.bottom_width_m)),
        "width_error_weighted_mean_fraction": float(np.average(
            geometry.width_error_fraction,
            weights=observed_node_area,
        )),
        "width_error_weighted_absolute_mean_fraction": float(np.average(
            np.abs(geometry.width_error_fraction),
            weights=observed_node_area,
        )),
        "area_equivalent_to_polygon_width_weighted_mean_ratio": float(
            np.average(
                assignment["area_equivalent_width_m"][mapped]
                / transverse_widths[mapped],
                weights=observed_node_area,
            )
        ),
        "observed_hypsometry_maximum_cumulative_error_fraction": float(
            observed_hypsometry_error
        ),
        "strip_mean_hypsometry_maximum_cumulative_error_fraction": float(
            strip_mean_hypsometry_error
        ),
        "effective_hypsometry_maximum_cumulative_error_fraction": float(
            effective_hypsometry_error
        ),
        "width_zero_order_area_bias_fraction": (
            effective_area_m2 - source_area_m2
        ) / source_area_m2,
        "width_first_order_elevation_moment_bias_m": (
            effective_moment - source_moment
        ) / source_area_m2,
        "source_surface_area_weighted_mean_m": source_moment / source_area_m2,
        "effective_surface_area_weighted_mean_m": (
            effective_moment / effective_area_m2
        ),
    }
    return node_rows, summary, _junction_audit(
        branches, network_crs, source.crs, np.sqrt(source.cell_area_m2)
    )


def _run_system(payload: dict) -> dict:
    source = load_source_state(Path(payload["netcdf_path"]), "farinotti2019")
    geometry = wkb.loads(bytes.fromhex(payload["geometry_wkb_hex"]))
    geometry = _transform_shape(
        geometry,
        CRS.from_user_input(payload["geometry_crs"]),
        source.crs,
    )
    components = normalize_lia_components(geometry)
    excluded = set(payload["excluded_component_indices"])
    node_rows = []
    component_rows = []
    junction_rows = []
    for component_index, component in enumerate(components):
        if component_index in excluded:
            continue
        network_path = (
            Path(payload["network_root"])
            / str(source.lia_unique_id)
            / f"component_{component_index}"
            / "lia_centerline_network.pkl.gz"
        )
        if not network_path.exists():
            raise FileNotFoundError(f"v3-network-missing:{network_path}")
        nodes, component_summary, junctions = _component_mapping(
            source=source,
            component_geometry=component,
            network_path=network_path,
            component_index=component_index,
        )
        node_rows.extend(nodes)
        component_rows.append(component_summary)
        junction_rows.extend({"component_index": component_index, **row} for row in junctions)

    source_area = sum(row["source_positive_area_km2"] for row in component_rows)
    vector_area = sum(row["vector_component_area_km2"] for row in component_rows)
    source_volume = sum(row["source_volume_km3"] for row in component_rows)
    observed_area = sum(row["observed_flowline_area_km2"] for row in component_rows)
    polygon_transverse_area = sum(
        row["polygon_transverse_flowline_area_km2"] for row in component_rows
    )
    effective_area = sum(row["effective_flowline_area_km2"] for row in component_rows)
    mapped_volume = sum(row["mapped_volume_km3"] for row in component_rows)
    mapped_nodes = [row for row in node_rows if row["mapping_active"]]
    mapped_area = np.asarray([row["assigned_area_m2"] for row in mapped_nodes])
    flags = []
    volume_error = (mapped_volume - source_volume) / source_volume
    observed_area_error = (observed_area - source_area) / source_area
    effective_area_error = (effective_area - source_area) / source_area
    effective_hypsometry_error = max(
        row["effective_hypsometry_maximum_cumulative_error_fraction"]
        for row in component_rows
    )
    observed_hypsometry_error = max(
        row["observed_hypsometry_maximum_cumulative_error_fraction"]
        for row in component_rows
    )
    unrepresented = sum(
        row["unrepresented_positive_node_count"] for row in component_rows
    )
    excluded_positive_nodes = sum(
        row["excluded_positive_node_count"] for row in component_rows
    )
    if abs(volume_error) > VOLUME_GATE_FRACTION:
        flags.append("volume-error-over-0.5pct")
    if abs(effective_area_error) > AREA_GATE_FRACTION:
        flags.append("effective-area-error-over-2pct")
    if effective_hypsometry_error > HYPSOMETRY_GATE_FRACTION:
        flags.append("effective-hypsometry-error-over-2pct")
    if observed_hypsometry_error > HYPSOMETRY_GATE_FRACTION:
        flags.append("centerline-hypsometry-error-over-2pct")
    if unrepresented:
        flags.append("positive-centerline-node-without-control-cell")
    status = "mapping-pass" if not flags else "mapping-review"
    system_row = {
        "rgi_id": source.rgi_id,
        "lia_unique_id": int(source.lia_unique_id),
        "status": status,
        "component_count": int(len(component_rows)),
        "excluded_component_count": int(len(excluded)),
        "branch_count": int(sum(row["branch_count"] for row in component_rows)),
        "node_count": int(len(node_rows)),
        "mapped_node_count": int(len(mapped_nodes)),
        "unrepresented_positive_node_count": int(unrepresented),
        "excluded_positive_node_count": int(excluded_positive_nodes),
        "source_positive_area_km2": float(source_area),
        "vector_active_area_km2": float(vector_area),
        "source_positive_minus_vector_area_fraction": float(
            (source_area - vector_area) / vector_area
        ),
        "observed_flowline_area_km2": float(observed_area),
        "polygon_transverse_flowline_area_km2": float(polygon_transverse_area),
        "polygon_transverse_area_error_fraction": float(
            (polygon_transverse_area - source_area) / source_area
        ),
        "effective_flowline_area_km2": float(effective_area),
        "observed_area_error_fraction": float(observed_area_error),
        "effective_area_error_fraction": float(effective_area_error),
        "source_volume_km3": float(source_volume),
        "mapped_volume_km3": float(mapped_volume),
        "volume_error_fraction": float(volume_error),
        "exact_width_regime_fraction": float(np.average(
            [row["exact_width_regime_fraction"] for row in component_rows],
            weights=[row["mapped_node_count"] for row in component_rows],
        )),
        "low_shape_factor_regime_fraction": float(np.average(
            [row["low_shape_factor_regime_fraction"] for row in component_rows],
            weights=[row["mapped_node_count"] for row in component_rows],
        )),
        "high_shape_factor_regime_fraction": float(np.average(
            [row["high_shape_factor_regime_fraction"] for row in component_rows],
            weights=[row["mapped_node_count"] for row in component_rows],
        )),
        "shape_factor_minimum": float(min(row["shape_factor_minimum"] for row in component_rows)),
        "shape_factor_median_node_weighted": float(np.median([
            row["area_shape_factor"] for row in mapped_nodes
        ])),
        "shape_factor_maximum": float(max(row["shape_factor_maximum"] for row in component_rows)),
        "lambda_maximum": float(max(row["lambda_maximum"] for row in component_rows)),
        "bottom_width_minimum_m": float(min(row["bottom_width_minimum_m"] for row in component_rows)),
        "width_error_area_weighted_mean_fraction": float(np.average(
            [row["width_error_fraction"] for row in mapped_nodes], weights=mapped_area
        )),
        "width_error_area_weighted_absolute_mean_fraction": float(np.average(
            [abs(row["width_error_fraction"]) for row in mapped_nodes], weights=mapped_area
        )),
        "observed_hypsometry_maximum_cumulative_error_fraction": float(max(
            row["observed_hypsometry_maximum_cumulative_error_fraction"]
            for row in component_rows
        )),
        "effective_hypsometry_maximum_cumulative_error_fraction": float(
            effective_hypsometry_error
        ),
        "geometric_contact_override_count": int(sum(
            row["geometric_contact_override"] for row in junction_rows
        )),
        "junction_count": int(len(junction_rows)),
        "review_flag_count": int(len(flags)),
        "review_flags": ";".join(flags),
    }
    for row in node_rows:
        row.update({"rgi_id": source.rgi_id, "lia_unique_id": source.lia_unique_id})
    for row in component_rows:
        row.update({"rgi_id": source.rgi_id, "lia_unique_id": source.lia_unique_id})
    for row in junction_rows:
        row.update({"rgi_id": source.rgi_id, "lia_unique_id": source.lia_unique_id})
    return {
        "system": system_row,
        "nodes": node_rows,
        "components": component_rows,
        "junctions": junction_rows,
    }


def run(
    *,
    config_path: Path,
    result_root: Path,
    network_root: Path,
    output_root: Path,
    workers: int,
    selected_system_ids: set[int] | None = None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outlines = gpd.read_file(resolve(config["lia_geometry_path"]))
    outlines["UniqueID"] = outlines["UniqueID"].astype(int)
    geometry_by_id = outlines.set_index("UniqueID").geometry
    components = pd.read_csv(result_root / "lia_system_component_results_v1.csv")
    payloads = []
    for path in sorted(result_root.glob("lia_*_*_system_grid_v1.nc")):
        unique_id = int(path.name.split("_")[1])
        if selected_system_ids and unique_id not in selected_system_ids:
            continue
        excluded = components[
            components["lia_unique_id"].astype(int).eq(unique_id)
            & components["status"].astype(str).eq(
                "documented-small-component-exclusion"
            )
        ]
        payloads.append({
            "netcdf_path": str(path),
            "network_root": str(network_root),
            "geometry_wkb_hex": geometry_by_id.loc[unique_id].wkb_hex,
            "geometry_crs": outlines.crs.to_string(),
            "excluded_component_indices": excluded["component_index"].astype(int).tolist(),
        })
    expected = len(selected_system_ids) if selected_system_ids else 191
    if len(payloads) != expected:
        raise ValueError(f"geometry-audit-source-count:{len(payloads)}:expected={expected}")

    system_rows = []
    node_rows = []
    component_rows = []
    junction_rows = []
    failures = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_run_system, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                result = future.result()
                system_rows.append(result["system"])
                node_rows.extend(result["nodes"])
                component_rows.extend(result["components"])
                junction_rows.extend(result["junctions"])
            except Exception as exc:
                failures.append({
                    "netcdf_path": payload["netcdf_path"],
                    "status": "mapping-failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
    output_root.mkdir(parents=True, exist_ok=True)
    system_frame = pd.DataFrame(system_rows).sort_values(
        ["status", "review_flag_count", "lia_unique_id"],
        ascending=[True, False, True],
    ) if system_rows else pd.DataFrame()
    if not system_frame.empty:
        system_frame["maximum_gate_ratio"] = pd.concat([
            system_frame["volume_error_fraction"].abs() / VOLUME_GATE_FRACTION,
            system_frame["effective_area_error_fraction"].abs() / AREA_GATE_FRACTION,
            system_frame[
                "effective_hypsometry_maximum_cumulative_error_fraction"
            ] / HYPSOMETRY_GATE_FRACTION,
        ], axis=1).max(axis=1)
    pd.DataFrame(node_rows).to_csv(output_root / "v3_flowline_node_geometry.csv", index=False)
    pd.DataFrame(component_rows).to_csv(output_root / "v3_flowline_component_geometry.csv", index=False)
    pd.DataFrame(junction_rows).to_csv(output_root / "v3_flowline_junction_audit.csv", index=False)
    system_frame.to_csv(output_root / "v3_flowline_system_geometry.csv", index=False)
    if not system_frame.empty:
        system_frame.nsmallest(20, "maximum_gate_ratio").to_csv(
            output_root / "v3_flowline_nearest20.csv", index=False
        )
        system_frame.nlargest(20, "maximum_gate_ratio").to_csv(
            output_root / "v3_flowline_worst20.csv", index=False
        )
    pd.DataFrame(failures).to_csv(output_root / "v3_flowline_geometry_failures.csv", index=False)
    def quantiles(column: str, *, absolute: bool = False) -> dict:
        values = system_frame[column].dropna()
        if absolute:
            values = values.abs()
        return {
            label: float(values.quantile(value))
            for label, value in (
                ("minimum", 0.0),
                ("p10", 0.1),
                ("p25", 0.25),
                ("median", 0.5),
                ("p75", 0.75),
                ("p90", 0.9),
                ("maximum", 1.0),
            )
        }

    exploded_flags = (
        system_frame["review_flags"].fillna("").str.split(";").explode()
        if not system_frame.empty else pd.Series(dtype=str)
    )
    exploded_flags = exploded_flags[exploded_flags.ne("")]
    summary = {
        "status": "completed" if not failures else "completed-with-failures",
        "scientific_scope": "pure-geometry-mapping-no-oggm-dynamics",
        "system_count": int(expected),
        "mapped_system_count": int(len(system_frame)),
        "failure_count": int(len(failures)),
        "mapping_status_counts": (
            system_frame["status"].value_counts().to_dict()
            if not system_frame.empty else {}
        ),
        "review_flag_counts": exploded_flags.value_counts().to_dict(),
        "individual_gate_pass_counts": {
            "volume": int(system_frame["volume_error_fraction"].abs().le(
                VOLUME_GATE_FRACTION
            ).sum()) if not system_frame.empty else 0,
            "effective_area": int(system_frame[
                "effective_area_error_fraction"
            ].abs().le(AREA_GATE_FRACTION).sum()) if not system_frame.empty else 0,
            "centerline_hypsometry": int(system_frame[
                "observed_hypsometry_maximum_cumulative_error_fraction"
            ].le(HYPSOMETRY_GATE_FRACTION).sum()) if not system_frame.empty else 0,
            "effective_hypsometry": int(system_frame[
                "effective_hypsometry_maximum_cumulative_error_fraction"
            ].le(HYPSOMETRY_GATE_FRACTION).sum()) if not system_frame.empty else 0,
            "all_positive_nodes_represented": int(system_frame[
                "unrepresented_positive_node_count"
            ].eq(0).sum()) if not system_frame.empty else 0,
        },
        "maximum_absolute_volume_error_fraction": (
            float(system_frame["volume_error_fraction"].abs().max())
            if not system_frame.empty else None
        ),
        "maximum_absolute_observed_area_error_fraction": (
            float(system_frame["observed_area_error_fraction"].abs().max())
            if not system_frame.empty else None
        ),
        "maximum_absolute_effective_area_error_fraction": (
            float(system_frame["effective_area_error_fraction"].abs().max())
            if not system_frame.empty else None
        ),
        "median_exact_width_regime_fraction": (
            float(system_frame["exact_width_regime_fraction"].median())
            if not system_frame.empty else None
        ),
        "median_width_error_area_weighted_absolute_mean_fraction": (
            float(system_frame["width_error_area_weighted_absolute_mean_fraction"].median())
            if not system_frame.empty else None
        ),
        "effective_area_absolute_error_quantiles": (
            quantiles("effective_area_error_fraction", absolute=True)
            if not system_frame.empty else {}
        ),
        "centerline_hypsometry_error_quantiles": (
            quantiles("observed_hypsometry_maximum_cumulative_error_fraction")
            if not system_frame.empty else {}
        ),
        "effective_hypsometry_error_quantiles": (
            quantiles("effective_hypsometry_maximum_cumulative_error_fraction")
            if not system_frame.empty else {}
        ),
        "weighted_absolute_width_error_quantiles": (
            quantiles("width_error_area_weighted_absolute_mean_fraction")
            if not system_frame.empty else {}
        ),
        "node_count": int(len(node_rows)),
        "mapped_node_count": int(sum(row["mapping_active"] for row in node_rows)),
        "excluded_positive_node_count": int(
            system_frame["excluded_positive_node_count"].sum()
        ) if not system_frame.empty else 0,
        "unrepresented_positive_node_count": int(
            system_frame["unrepresented_positive_node_count"].sum()
        ) if not system_frame.empty else 0,
        "geometric_contact_override_count": (
            int(system_frame["geometric_contact_override_count"].sum())
            if not system_frame.empty else 0
        ),
        "source_products_mutated": False,
        "gates": {
            "volume_error_fraction": VOLUME_GATE_FRACTION,
            "effective_area_error_fraction": AREA_GATE_FRACTION,
            "effective_hypsometry_error_fraction": HYPSOMETRY_GATE_FRACTION,
        },
    }
    (output_root / "v3_flowline_geometry_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--network-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--system-id", action="append", type=int)
    args = parser.parse_args()
    print(json.dumps(run(
        config_path=args.config,
        result_root=args.result_root,
        network_root=args.network_root,
        output_root=args.output_root,
        workers=args.workers,
        selected_system_ids=set(args.system_id) if args.system_id else None,
    ), indent=2))


if __name__ == "__main__":
    main()

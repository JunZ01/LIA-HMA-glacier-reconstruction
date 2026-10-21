"""Build and audit LIA-specific OGGM centerline networks.

The pilot uses the original vector LIA outline and the frozen v0.1 LIA surface.
It is geometry-only: no cross-section remapping or ice dynamics are performed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry import LineString, Point

from ..paths import resolve
from .lia_dynamic_matrix_pilot import (
    fixed_grid_for_geometry,
    grid_transform,
    load_source_state,
    prepare_centerline_gdir,
    read_dem_on_grid,
    reproject_source_array,
    write_json,
)
from .lia_system_grid import normalize_lia_components


@dataclass
class _GeometryFlowline:
    line: LineString
    section: np.ndarray
    dx_meter: float
    flows_to: "_GeometryFlowline | None" = None


def _grid_line_to_projected(line, grid) -> LineString:
    coordinates = np.asarray(line.coords, dtype=float)
    x = float(grid.x0) + coordinates[:, 0] * float(grid.dx)
    y = float(grid.y0) + coordinates[:, 1] * float(grid.dy)
    return LineString(np.column_stack([x, y]))


def _line_prefix(line: LineString, values: np.ndarray, station: float) -> tuple[LineString, np.ndarray]:
    """Cut a sampled line at an exact station and interpolate the final value."""
    coordinates = np.asarray(line.coords, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(coordinates) != len(values):
        raise ValueError(
            f"line-value-length-mismatch:{len(coordinates)}:{len(values)}"
        )
    segment_lengths = np.hypot(
        np.diff(coordinates[:, 0]), np.diff(coordinates[:, 1])
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    station = float(np.clip(station, 0.0, cumulative[-1]))
    exact = np.flatnonzero(np.isclose(cumulative, station, atol=1e-7, rtol=0.0))
    if exact.size:
        end = int(exact[-1])
        return LineString(coordinates[: end + 1]), values[: end + 1].copy()
    left = int(np.searchsorted(cumulative, station, side="right") - 1)
    fraction = (station - cumulative[left]) / segment_lengths[left]
    endpoint = coordinates[left] + fraction * (coordinates[left + 1] - coordinates[left])
    endpoint_value = values[left] + fraction * (values[left + 1] - values[left])
    cut_coordinates = np.vstack([coordinates[: left + 1], endpoint])
    cut_values = np.append(values[: left + 1], endpoint_value)
    return LineString(cut_coordinates), cut_values


def _line_suffix(line: LineString, station: float) -> LineString:
    """Return the part of a line after an exact station."""
    coordinates = np.asarray(line.coords, dtype=float)
    dummy = np.arange(len(coordinates), dtype=float)
    prefix, _ = _line_prefix(line, dummy, station)
    endpoint = np.asarray(prefix.coords[-1], dtype=float)
    segment_lengths = np.hypot(
        np.diff(coordinates[:, 0]), np.diff(coordinates[:, 1])
    )
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    right = int(np.searchsorted(cumulative, station, side="right"))
    suffix = np.vstack([endpoint, coordinates[right:]])
    if len(suffix) < 2:
        suffix = np.vstack([endpoint, endpoint])
    return LineString(suffix)


def _centerline_network_payload(
    centerlines: list, downstream_line: dict, component_geometry, grid, crs
) -> dict:
    """Serialize the routed network, extending its root to the true LIA outlet."""
    index_by_id = {id(flowline): index for index, flowline in enumerate(centerlines)}
    root_indices = [
        index for index, flowline in enumerate(centerlines) if flowline.flows_to is None
    ]
    if len(root_indices) != 1:
        raise ValueError(f"lia-network-root-count:{len(root_indices)}")
    main_index = root_indices[0]
    main = centerlines[main_index]
    projected_main = _grid_line_to_projected(main.line, grid)
    projected_full = _grid_line_to_projected(downstream_line["full_line"], grid)
    anchor_index = downstream_line.get("anchor_index")
    main_reference = np.asarray(main.surface_h, dtype=float)
    if anchor_index is not None:
        main_reference = main_reference[: int(anchor_index) + 1]
    full_reference = np.append(
        main_reference,
        np.asarray(downstream_line["surface_h"], dtype=float),
    )
    if len(full_reference) != len(projected_full.coords):
        raise ValueError(
            "downstream-reference-length-mismatch:"
            f"{len(projected_full.coords)}:{len(full_reference)}"
        )
    join_point = (
        _grid_line_to_projected(
            LineString(
                [
                    downstream_line["anchor_point_grid"],
                    downstream_line["anchor_point_grid"],
                ]
            ),
            grid,
        ).boundary.centroid
        if anchor_index is not None
        else Point(projected_main.coords[-1])
    )
    join_station = float(projected_full.project(join_point))
    crossings = _outward_exterior_crossings(projected_full, component_geometry)
    if not crossings:
        raise ValueError("lia-main-line-has-no-exterior-outlet")
    outlet_station, outlet = crossings[0]
    saved_main, saved_reference = _line_prefix(
        projected_full, full_reference, outlet_station
    )
    join_gap = float(
        Point(projected_main.coords[-1]).distance(
            projected_full.interpolate(join_station)
        )
    )
    branches = []
    for index, flowline in enumerate(centerlines):
        if index == main_index:
            projected = saved_main
            reference = saved_reference
        else:
            projected = _grid_line_to_projected(flowline.line, grid)
            reference = np.asarray(flowline.surface_h, dtype=float)
        parent_index = (
            None
            if flowline.flows_to is None
            else index_by_id[id(flowline.flows_to)]
        )
        junction = flowline.__dict__.get("flows_to_point")
        projected_junction = None
        if junction is not None:
            projected_junction = list(
                _grid_line_to_projected(
                    LineString([junction.coords[0], junction.coords[0]]), grid
                ).coords[0]
            )
        branches.append({
            "branch_index": index,
            "parent_index": parent_index,
            "coords_projected_m": np.asarray(projected.coords, dtype=float),
            "reference_surface_m": reference,
            "dx_meter": float(flowline.dx_meter),
            "flows_to_point_projected_m": projected_junction,
        })
    return {
        "format": "lia-projected-centerline-network-v1",
        "crs": str(crs),
        "branches": branches,
        "main_branch_index": main_index,
        "main_outlet_projected_m": [float(outlet.x), float(outlet.y)],
        "main_outlet_station_m": float(outlet_station),
        "legacy_main_endpoint_station_m": float(join_station),
        "main_extension_length_m": float(outlet_station - join_station),
        "main_extension_join_gap_m": join_gap,
        "anchor_rerouted": bool(downstream_line.get("anchor_rerouted", False)),
        "anchor_back_nodes": (
            None
            if anchor_index is None
            else int(downstream_line["anchor_back_nodes"])
        ),
        "anchor_join_turn_deg": (
            None
            if anchor_index is None
            else float(downstream_line["anchor_join_turn_deg"])
        ),
        "anchor_point_projected_m": (
            None
            if anchor_index is None
            else [
                float(join_point.x),
                float(join_point.y),
            ]
        ),
        "legacy_main_endpoint_projected_m": [
            float(projected_main.coords[-1][0]),
            float(projected_main.coords[-1][1]),
        ],
    }


def _load_pickle(path: Path):
    with path.open("rb") as stream:
        magic = stream.read(2)
    context = gzip.open(path, "rb") if magic == b"\x1f\x8b" else path.open("rb")
    with context as stream:
        return pickle.load(stream)


def _modern_gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:8] / rgi_id[:11] / rgi_id


def _load_old_modern_network(
    root: Path, rgi_id: str, target_crs: CRS
) -> list[LineString]:
    directory = _modern_gdir_path(root, rgi_id)
    grid_spec = json.loads((directory / "glacier_grid.json").read_text())
    x0, y0 = (float(value) for value in grid_spec["x0y0"])
    dx, dy = (float(value) for value in grid_spec["dxdy"])
    source_crs = CRS.from_user_input(grid_spec["proj"])
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lines = []
    for flowline in _load_pickle(directory / "inversion_flowlines.pkl"):
        coordinates = np.asarray(flowline.line.coords, dtype=float)
        x = x0 + coordinates[:, 0] * dx
        y = y0 + coordinates[:, 1] * dy
        tx, ty = transformer.transform(x, y)
        lines.append(LineString(np.column_stack([tx, ty])))
    return lines


def _point_parts(geometry) -> list[Point]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type == "MultiPoint":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        points = []
        for part in geometry.geoms:
            points.extend(_point_parts(part))
        return points
    if geometry.geom_type in {"LineString", "LinearRing"}:
        coordinates = list(geometry.coords)
        return [Point(coordinates[0]), Point(coordinates[-1])]
    if geometry.geom_type.startswith("Multi"):
        points = []
        for part in geometry.geoms:
            points.extend(_point_parts(part))
        return points
    return []


def _boundary_crossings(line: LineString, polygon) -> list[Point]:
    """Return intersections with the true outer outline, excluding hole rings."""
    points = _point_parts(line.intersection(polygon.exterior))
    stations = sorted((float(line.project(point)), point) for point in points)
    unique = []
    for station, point in stations:
        if not unique or abs(station - unique[-1][0]) > 1e-6:
            unique.append((station, point))
    return [point for _, point in unique]


def _outward_exterior_crossings(
    line: LineString, polygon, *, minimum_station: float = 0.0
) -> list[tuple[float, Point]]:
    """Return ordered outer-ring crossings where the routed line leaves the LIA polygon."""
    tolerance = max(1e-4, min(0.25, 1e-5 * float(line.length)))
    candidates = []
    for point in _boundary_crossings(line, polygon):
        station = float(line.project(point))
        if station + 1e-7 < float(minimum_station):
            continue
        before = line.interpolate(max(0.0, station - tolerance))
        after = line.interpolate(min(float(line.length), station + tolerance))
        before_inside = bool(polygon.covers(before))
        after_inside = bool(polygon.covers(after))
        terminal_boundary = bool(
            abs(station - float(line.length)) <= tolerance
            and point.distance(polygon.exterior) <= 1e-5
        )
        if before_inside and (not after_inside or terminal_boundary):
            candidates.append((station, point))
    return sorted(candidates, key=lambda item: item[0])


def _parent_indices(flowlines: list) -> list[int | None]:
    by_id = {id(flowline): index for index, flowline in enumerate(flowlines)}
    return [
        None if flowline.flows_to is None else by_id.get(id(flowline.flows_to), -1)
        for flowline in flowlines
    ]


def _build_geometry_flowlines(
    centerlines: list,
    downstream_line: dict,
    sections: list[np.ndarray],
    map_dx_m: float,
    main_model_section: np.ndarray | None = None,
) -> list[_GeometryFlowline]:
    receiver_indices = [
        None if flowline.flows_to is None else centerlines.index(flowline.flows_to)
        for flowline in centerlines
    ]
    created = []
    for centerline, section, receiver_index in zip(
        centerlines, sections, receiver_indices
    ):
        line = centerline.line
        values = np.asarray(section, dtype=float)
        if receiver_index is None:
            line = downstream_line["full_line"]
            values = (
                np.asarray(main_model_section, dtype=float)
                if main_model_section is not None
                else np.append(
                    values,
                    np.zeros(len(downstream_line["surface_h"]), dtype=float),
                )
            )
            if len(values) != len(line.coords):
                raise ValueError(
                    f"main-model-section-length-mismatch:{len(line.coords)}:{len(values)}"
                )
        created.append(
            _GeometryFlowline(
                line=line,
                section=values,
                dx_meter=float(centerline.dx) * float(map_dx_m),
            )
        )
    for flowline, receiver_index in zip(created, receiver_indices):
        if receiver_index is not None:
            flowline.flows_to = created[receiver_index]
    return created


def _reroute_main_from_upstream_anchor(
    *,
    centerlines: list,
    downstream_line: dict,
    component_geometry,
    grid,
    routing_topography_m: np.ndarray,
    glacier_mask: np.ndarray,
    max_back_nodes: int = 8,
    maximum_join_turn_deg: float = 45.0,
    outlet_projected_m: tuple[float, float] | list[float] | None = None,
    minimum_anchor_index: int | None = None,
) -> dict:
    """Reroute the root line from a nearby upstream node to the same outlet.

    OGGM's downstream route starts at the inversion-line endpoint.  On a coarse
    raster that endpoint can create a sharp join.  This helper searches a small
    upstream window, keeps the original LIA outlet, and accepts the closest
    candidate whose aggregated join tangent is smooth enough.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.graph import route_through_array

    root_indices = [
        index for index, flowline in enumerate(centerlines) if flowline.flows_to is None
    ]
    if len(root_indices) != 1:
        raise ValueError(f"lia-network-root-count:{len(root_indices)}")
    main_index = root_indices[0]
    main = centerlines[main_index]
    current_full_grid = downstream_line["full_line"]
    current_full = _grid_line_to_projected(current_full_grid, grid)
    if outlet_projected_m is None:
        crossings = _outward_exterior_crossings(current_full, component_geometry)
        if not crossings:
            raise ValueError("anchor-reroute-no-lia-exterior-outlet")
        outlet_station, outlet = crossings[0]
    else:
        outlet = Point(float(outlet_projected_m[0]), float(outlet_projected_m[1]))
        outlet_station = float(current_full.project(outlet))
    outlet_grid = np.asarray([
        (float(outlet.x) - float(grid.x0)) / float(grid.dx),
        (float(outlet.y) - float(grid.y0)) / float(grid.dy),
    ])
    target = (
        int(np.clip(np.rint(outlet_grid[1]), 0, grid.ny - 1)),
        int(np.clip(np.rint(outlet_grid[0]), 0, grid.nx - 1)),
    )
    finite = np.isfinite(routing_topography_m)
    if not np.any(finite[glacier_mask]):
        raise ValueError("anchor-reroute-no-finite-routing-surface")
    fill = float(np.nanmax(routing_topography_m[finite])) + 500.0
    cost_grid = np.where(finite, routing_topography_m, fill)
    cost_grid = cost_grid**8 + distance_transform_edt(~glacier_mask)

    coordinates = np.asarray(main.line.coords, dtype=float)
    candidates = []
    upper = min(int(max_back_nodes), len(coordinates) - 3)
    for back in range(1, upper + 1):
        anchor_index = len(coordinates) - 1 - back
        if minimum_anchor_index is not None and anchor_index < int(
            minimum_anchor_index
        ):
            continue
        anchor = coordinates[anchor_index]
        start = (
            int(np.clip(np.rint(anchor[1]), 0, grid.ny - 1)),
            int(np.clip(np.rint(anchor[0]), 0, grid.nx - 1)),
        )
        try:
            route_indices, route_cost = route_through_array(
                cost_grid, start, target, geometric=True
            )
        except Exception:
            continue
        route_xy = np.asarray(route_indices, dtype=float)[:, [1, 0]]
        if len(route_xy) < 2:
            continue
        incoming = anchor - coordinates[anchor_index - 1]
        lookahead = route_xy[min(8, len(route_xy) - 1)] - route_xy[0]
        incoming_norm = float(np.linalg.norm(incoming))
        lookahead_norm = float(np.linalg.norm(lookahead))
        if incoming_norm <= 0 or lookahead_norm <= 0:
            continue
        turn = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(incoming, lookahead)
                        / (incoming_norm * lookahead_norm),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        if turn > float(maximum_join_turn_deg):
            continue
        prefix = LineString(coordinates[: anchor_index + 1])
        route_coordinates = route_xy
        if np.linalg.norm(route_coordinates[0] - anchor) > 1e-8:
            route_coordinates = np.vstack([anchor, route_coordinates])
        else:
            route_coordinates = route_coordinates.copy()
            route_coordinates[0] = anchor
        route_line = LineString(route_coordinates)
        spacing = float(main.dx)
        stations = np.arange(spacing, route_line.length, spacing)
        extension_coordinates = [anchor]
        extension_coordinates.extend(
            np.asarray(route_line.interpolate(float(station)).coords[0])
            for station in stations
        )
        if np.linalg.norm(
            np.asarray(extension_coordinates[-1]) - route_coordinates[-1]
        ) > 1e-8:
            extension_coordinates.append(route_coordinates[-1])
        extension_coordinates = np.asarray(extension_coordinates, dtype=float)
        rerouted_extension = LineString(extension_coordinates)
        rerouted_full = LineString(
            np.vstack([
                np.asarray(prefix.coords, dtype=float),
                extension_coordinates[1:],
            ])
        )
        outlet_station_grid = float(
            current_full_grid.project(
                Point(float(outlet_grid[0]), float(outlet_grid[1]))
            )
        )
        original_suffix = _line_suffix(current_full_grid, outlet_station_grid)
        rerouted_coords = np.asarray(rerouted_full.coords, dtype=float)
        suffix_coords = np.asarray(original_suffix.coords, dtype=float)
        if np.linalg.norm(rerouted_coords[-1] - suffix_coords[0]) <= 1.5:
            combined_coords = np.vstack([rerouted_coords, suffix_coords[1:]])
        else:
            combined_coords = np.vstack([rerouted_coords, suffix_coords])
        keep = np.ones(len(combined_coords), dtype=bool)
        if len(combined_coords) > 1:
            keep[1:] = np.hypot(
                np.diff(combined_coords[:, 0]),
                np.diff(combined_coords[:, 1]),
            ) > 1e-8
        combined_coords = combined_coords[keep]
        if len(combined_coords) < 2:
            continue
        combined = LineString(combined_coords)
        if not combined.is_simple:
            continue
        # The candidate must stay in the LIA until the retained outlet.
        combined_projected = _grid_line_to_projected(combined, grid)
        if outlet_projected_m is None:
            first_crossings = _outward_exterior_crossings(
                combined_projected, component_geometry
            )
            if not first_crossings:
                continue
        elif Point(combined_projected.coords[-1]).distance(outlet) > 2.0 * abs(
            float(grid.dx)
        ):
            continue
        candidates.append(
            {
                # Prefer the smallest upstream change; use turn only to break
                # ties among candidates at the same distance.
                "score": (back, turn),
                "turn": turn,
                "back": back,
                "anchor_index": anchor_index,
                "anchor_grid": anchor.tolist(),
                "line": combined,
                "extension": rerouted_extension,
                "route_cost": float(route_cost),
            }
        )
    if not candidates:
        raise ValueError("anchor-reroute-no-smooth-candidate")
    selected = min(candidates, key=lambda item: item["score"])
    full_line = selected["line"]
    coordinates_full = np.asarray(full_line.coords, dtype=float)
    # Surface values are sampled by the caller after this geometry is selected.
    downstream_line = dict(downstream_line)
    downstream_line["full_line"] = full_line
    downstream_line["anchor_index"] = int(selected["anchor_index"])
    downstream_line["anchor_point_grid"] = list(selected["anchor_grid"])
    downstream_line["anchor_back_nodes"] = int(selected["back"])
    downstream_line["anchor_join_turn_deg"] = float(selected["turn"])
    downstream_line["anchor_route_cost"] = float(selected["route_cost"])
    downstream_line["anchor_rerouted"] = True
    return downstream_line


def _network_topology(flowlines: list, lines: list[LineString], tolerance_m: float):
    parents = _parent_indices(flowlines)
    duplicate_count = 0
    overlap_count = 0
    crossing_count = 0
    for left in range(len(lines)):
        for right in range(left + 1, len(lines)):
            first = lines[left]
            second = lines[right]
            if first.equals(second):
                duplicate_count += 1
                continue
            intersection = first.intersection(second)
            lineal_length = sum(
                part.length
                for part in getattr(intersection, "geoms", [intersection])
                if part.geom_type in {"LineString", "MultiLineString"}
            )
            if lineal_length > 1e-6:
                overlap_count += 1
            points = _point_parts(intersection)
            if not points:
                continue
            child = left if parents[left] == right else right if parents[right] == left else None
            expected = False
            if child is not None:
                child_tail = Point(lines[child].coords[-1])
                expected = all(point.distance(child_tail) <= tolerance_m for point in points)
            if not expected:
                crossing_count += 1
    return {
        "branch_count": len(lines),
        "root_count": int(sum(parent is None for parent in parents)),
        "missing_parent_count": int(sum(parent == -1 for parent in parents)),
        "self_intersection_count": int(sum(not line.is_simple for line in lines)),
        "exact_duplicate_branch_pair_count": duplicate_count,
        "overlapping_branch_pair_count": overlap_count,
        "nonjunction_crossing_pair_count": crossing_count,
    }


def _grid_boundary_polygon(grid):
    transform = grid_transform(grid)
    left = float(transform.c)
    top = float(transform.f)
    right = left + float(grid.nx) * float(transform.a)
    bottom = top + float(grid.ny) * float(transform.e)
    from shapely.geometry import box

    return box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))


def audit_outlet_geometry(
    *,
    centerlines: list,
    model_flowlines: list,
    network_payload: dict,
    component_geometry,
    grid,
) -> tuple[dict, dict]:
    legacy_interior_lines = [
        _grid_line_to_projected(flowline.line, grid) for flowline in centerlines
    ]
    interior_lines = [
        LineString(np.asarray(branch["coords_projected_m"], dtype=float))
        for branch in sorted(
            network_payload["branches"], key=lambda item: item["branch_index"]
        )
    ]
    model_lines = [_grid_line_to_projected(flowline.line, grid) for flowline in model_flowlines]
    main_index = int(network_payload["main_branch_index"])
    if main_index != len(interior_lines) - 1:
        raise ValueError(f"main-flowline-not-last:{main_index}:{len(interior_lines)}")
    main_interior = interior_lines[-1]
    legacy_main_interior = legacy_interior_lines[-1]
    main_model = model_lines[-1]
    main_flowline = model_flowlines[-1]
    legacy_endpoint_station = float(
        main_model.project(Point(legacy_main_interior.coords[-1]))
    )
    crossings = _boundary_crossings(main_model, component_geometry)
    outlet_candidates = _outward_exterior_crossings(main_model, component_geometry)
    outlet_station, outlet = (
        min(outlet_candidates, key=lambda item: item[0])
        if outlet_candidates
        else (math.nan, None)
    )
    positive = np.flatnonzero(np.asarray(main_flowline.section) > 1e-8)
    last_positive_index = int(positive[-1]) if positive.size else -1
    last_positive = (
        Point(main_model.coords[last_positive_index]) if last_positive_index >= 0 else None
    )
    last_positive_station = (
        float(main_model.project(last_positive)) if last_positive is not None else math.nan
    )
    node_dx_m = float(main_flowline.dx_meter)
    grid_boundary = _grid_boundary_polygon(grid).boundary
    positive_node_distances = [
        Point(main_model.coords[int(index)]).distance(component_geometry)
        for index in positive
    ]
    positive_outside_maximum = max(positive_node_distances, default=math.nan)
    topology = _network_topology(
        centerlines, interior_lines, tolerance_m=max(1.0, 0.1 * float(grid.dx))
    )
    outlet_count = len(outlet_candidates)
    outlet_error = (
        abs(last_positive_station - outlet_station)
        if outlet is not None and last_positive is not None
        else math.nan
    )
    downstream_buffer = (
        float(main_model.length - outlet_station) if outlet is not None else math.nan
    )
    endpoint_grid_distance = Point(main_model.coords[-1]).distance(grid_boundary)
    saved_endpoint = Point(main_interior.coords[-1])
    saved_endpoint_to_outlet = (
        float(saved_endpoint.distance(outlet)) if outlet is not None else math.nan
    )
    saved_outside_length = float(
        main_interior.difference(component_geometry.buffer(1e-6)).length
    )
    join_gap = float(network_payload.get("main_extension_join_gap_m", math.inf))
    maximum_saved_step = float(
        np.max(
            np.hypot(
                np.diff(np.asarray(main_interior.coords)[:, 0]),
                np.diff(np.asarray(main_interior.coords)[:, 1]),
            )
        )
    )
    gates = {
        "single_outlet": outlet_count == 1,
        "anchor_join_turn_within_limit": bool(
            network_payload.get("anchor_join_turn_deg") is None
            or float(network_payload["anchor_join_turn_deg"]) <= 45.0
        ),
        "saved_main_reaches_exact_outlet": bool(
            np.isfinite(saved_endpoint_to_outlet) and saved_endpoint_to_outlet <= 1e-5
        ),
        "saved_main_inside_lia_outline": saved_outside_length <= 1e-5,
        "saved_main_has_no_join_gap": join_gap <= 1e-5,
        "saved_main_step_is_continuous": maximum_saved_step <= 2.0 * node_dx_m + 1e-6,
        "positive_terminus_within_one_node": bool(
            np.isfinite(outlet_error) and outlet_error <= node_dx_m + 1e-6
        ),
        "positive_ice_inside_lia_tolerance": bool(
            np.isfinite(positive_outside_maximum)
            and positive_outside_maximum <= node_dx_m + 1e-6
        ),
        "zero_thickness_downstream_buffer": bool(
            np.isfinite(downstream_buffer) and downstream_buffer >= 2.0 * node_dx_m
        ),
        # OGGM routes to a boundary pixel centre; smoothing can leave the final
        # vertex up to roughly two map pixels from the mathematical grid edge.
        "model_line_reaches_grid_boundary": endpoint_grid_distance <= 2.0 * float(grid.dx) + 1e-6,
        "topology_clean": all(
            topology[key] == 0
            for key in (
                "missing_parent_count",
                "self_intersection_count",
                "exact_duplicate_branch_pair_count",
                "overlapping_branch_pair_count",
                "nonjunction_crossing_pair_count",
            )
        )
        and topology["root_count"] == 1,
    }
    audit = {
        **topology,
        "node_spacing_m": node_dx_m,
        "main_centerline_inside_length_m": float(main_interior.length),
        "legacy_main_centerline_length_m": float(legacy_main_interior.length),
        "main_outlet_extension_length_m": float(
            network_payload["main_extension_length_m"]
        ),
        "main_extension_join_gap_m": join_gap,
        "anchor_rerouted": bool(network_payload.get("anchor_rerouted", False)),
        "anchor_back_nodes": network_payload.get("anchor_back_nodes"),
        "anchor_join_turn_deg": network_payload.get("anchor_join_turn_deg"),
        "saved_main_endpoint_to_outlet_m": (
            None
            if not np.isfinite(saved_endpoint_to_outlet)
            else saved_endpoint_to_outlet
        ),
        "saved_main_outside_lia_length_m": saved_outside_length,
        "saved_main_maximum_node_step_m": maximum_saved_step,
        "main_model_line_length_m": float(main_model.length),
        "main_centerline_endpoint_to_lia_boundary_m": float(
            Point(main_interior.coords[-1]).distance(component_geometry.boundary)
        ),
        "lia_exterior_intersection_count": len(crossings),
        "outlet_candidate_count": outlet_count,
        "outlet_station_m": None if outlet is None else float(outlet_station),
        "last_positive_node_index": last_positive_index,
        "last_positive_station_m": (
            None if last_positive is None else float(last_positive_station)
        ),
        "last_positive_to_outlet_distance_m": (
            None if not np.isfinite(outlet_error) else float(outlet_error)
        ),
        "outlet_to_model_endpoint_buffer_m": (
            None if not np.isfinite(downstream_buffer) else float(downstream_buffer)
        ),
        "maximum_positive_node_outside_lia_distance_m": (
            None
            if not np.isfinite(positive_outside_maximum)
            else float(positive_outside_maximum)
        ),
        "model_endpoint_to_grid_boundary_m": float(endpoint_grid_distance),
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }
    markers = {
        "outlet": outlet,
        "legacy_centerline_endpoint": Point(legacy_main_interior.coords[-1]),
        "last_positive": last_positive,
        "model_endpoint": Point(main_model.coords[-1]),
    }
    return audit, {
        "interior_lines": interior_lines,
        "model_lines": model_lines,
        "markers": markers,
    }


def _write_vector_outputs(case_dir: Path, crs: CRS, geometry, spatial: dict) -> None:
    gpd.GeoDataFrame(
        [{"role": "lia_outline", "geometry": geometry}], crs=crs
    ).to_file(case_dir / "lia_outline.shp")
    interior = spatial["interior_lines"]
    model = spatial["model_lines"]
    gpd.GeoDataFrame(
        [
            {"branch": index, "main": index == len(interior) - 1, "geometry": line}
            for index, line in enumerate(interior)
        ],
        crs=crs,
    ).to_file(case_dir / "lia_centerlines.shp")
    gpd.GeoDataFrame(
        [
            {"branch": index, "main": index == len(model) - 1, "geometry": line}
            for index, line in enumerate(model)
        ],
        crs=crs,
    ).to_file(case_dir / "lia_model_lines.shp")
    marker_rows = [
        {"kind": kind, "geometry": point}
        for kind, point in spatial["markers"].items()
        if point is not None
    ]
    gpd.GeoDataFrame(marker_rows, crs=crs).to_file(case_dir / "outlet_markers.shp")


def _plot_case(
    *, case_dir: Path, source, geometry, old_lines: list[LineString], spatial: dict, audit: dict
) -> None:
    valid = source.valid_mask & geometry_mask(
        [geometry.__geo_interface__],
        out_shape=source.shape,
        transform=source.transform,
        invert=True,
        all_touched=True,
    )
    surface = np.where(valid, source.surface_h_m, np.nan)
    bounds = geometry.bounds
    pad = 0.06 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)
    extent = [
        float(source.x[0] - 0.5 * source.transform.a),
        float(source.x[-1] + 0.5 * source.transform.a),
        float(source.y[-1] + 0.5 * source.transform.e),
        float(source.y[0] - 0.5 * source.transform.e),
    ]
    finite = surface[np.isfinite(surface)]
    low, high = np.percentile(finite, [2, 98])
    for axis, title in zip(axes, ["Old modern-derived network", "New LIA-specific network"]):
        axis.imshow(
            surface,
            extent=extent,
            origin="upper",
            cmap="terrain",
            vmin=low,
            vmax=high,
            alpha=0.72,
        )
        x, y = geometry.exterior.xy
        axis.plot(x, y, color="black", linewidth=1.4, label="LIA SHP outline")
        for interior in geometry.interiors:
            ix, iy = interior.xy
            axis.plot(ix, iy, color="black", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlim(bounds[0] - pad, bounds[2] + pad)
        axis.set_ylim(bounds[1] - pad, bounds[3] + pad)
        axis.set_aspect("equal")
        axis.tick_params(labelsize=8)
        axis.set_xlabel("Projected x (m)")
        axis.set_ylabel("Projected y (m)")
    for index, line in enumerate(old_lines):
        x, y = line.xy
        axes[0].plot(
            x,
            y,
            color="#5b6770" if index < len(old_lines) - 1 else "#a23b3b",
            linewidth=1.0 if index < len(old_lines) - 1 else 2.2,
            alpha=0.9,
        )
    interior_lines = spatial["interior_lines"]
    model_lines = spatial["model_lines"]
    for index, line in enumerate(interior_lines):
        x, y = line.xy
        axes[1].plot(
            x,
            y,
            color="#2563a6" if index < len(interior_lines) - 1 else "#b42318",
            linewidth=1.1 if index < len(interior_lines) - 1 else 2.3,
            alpha=0.95,
        )
    main_model = model_lines[-1]
    outlet_station = audit.get("outlet_station_m")
    if outlet_station is not None:
        downstream_buffer = _line_suffix(main_model, float(outlet_station))
        x, y = downstream_buffer.xy
        axes[1].plot(x, y, color="#b42318", linewidth=1.8, linestyle="--")
    marker_style = {
        "outlet": ("X", "black", 75),
        "legacy_centerline_endpoint": ("o", "#d97706", 45),
        "last_positive": ("D", "#16803a", 38),
        "model_endpoint": ("s", "#2563a6", 40),
    }
    for kind, point in spatial["markers"].items():
        if point is None:
            continue
        marker, color, size = marker_style[kind]
        axes[1].scatter([point.x], [point.y], marker=marker, c=color, s=size, zorder=5)
    old_main = old_lines[-1]
    old_endpoint_distance = Point(old_main.coords[-1]).distance(geometry.boundary)
    axes[0].text(
        0.02,
        0.02,
        f"Main endpoint to LIA boundary: {old_endpoint_distance / 1000:.2f} km",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    axes[1].text(
        0.02,
        0.02,
        "Last positive to outlet: "
        f"{audit['last_positive_to_outlet_distance_m']:.0f} m\n"
        "Added LIA outlet segment: "
        f"{audit['main_outlet_extension_length_m']:.0f} m\n"
        "Zero-thickness downstream buffer: "
        f"{audit['outlet_to_model_endpoint_buffer_m'] / 1000:.2f} km\n"
        f"Outlet gates: {'PASS' if audit['all_gates_pass'] else 'REVIEW'}",
        transform=axes[1].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    fig.suptitle(
        f"LIA system {source.lia_unique_id} ({source.rgi_id}) | Farinotti | "
        f"{audit['map_dx_m']:g} m geometry",
        fontsize=13,
    )
    fig.savefig(case_dir / "old_vs_lia_centerlines.png", dpi=220)
    plt.close(fig)


def _configure_oggm(config: dict, flowline_dx: float) -> str:
    import oggm
    from oggm import cfg

    cfg.initialize(logging_level="WARNING")
    for key, value in config.get("lia_centerlines", {}).get("oggm_params", {}).items():
        if key in cfg.PARAMS:
            cfg.PARAMS[key] = value
    cfg.PARAMS["flowline_dx"] = float(flowline_dx)
    cfg.PARAMS["downstream_line_shape"] = "trapezoidal"
    cfg.PARAMS["use_kcalving_for_run"] = False
    return str(oggm.__version__)


def run(
    config: dict,
    *,
    system_ids: list[int],
    map_dx_m: float,
    flowline_dx: float,
    buffer_m: float,
    output_subdir: str,
    source_subdir: str | None = None,
    compact_output: bool = False,
    write_summary_files: bool = True,
    anchor_reroute: bool = False,
    anchor_max_back_nodes: int = 8,
) -> tuple[pd.DataFrame, dict]:
    oggm_version = _configure_oggm(config, flowline_dx)
    source_dir = resolve(config["output_dir"]) / str(
        source_subdir
        or config["lia_3d_system_grid_pilot_v1"]["output_subdir"]
    )
    output_dir = resolve(config["output_dir"]) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    modern_root = resolve(
        config["lia_dynamic_fixed_bed_audit"]["modern_mb_gdir_root"]
    )
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    lia_by_id = lia.set_index("UniqueID")
    dem_path = resolve(config["surface_dem_path"])
    rows = []
    for unique_id in system_ids:
        started = time.perf_counter()
        try:
            candidates = sorted(source_dir.glob(f"lia_{unique_id}_*_system_grid_v1.nc"))
            if len(candidates) != 1:
                raise ValueError(f"system-grid-file-count:{unique_id}:{len(candidates)}")
            source = load_source_state(candidates[0], "farinotti2019")
            geometry = gpd.GeoSeries(
                [lia_by_id.loc[unique_id].geometry], crs=lia.crs
            ).to_crs(source.crs).iloc[0]
            components = normalize_lia_components(geometry)
            for component_index, component in enumerate(components):
                case_dir = output_dir / str(unique_id) / f"component_{component_index}"
                case_dir.mkdir(parents=True, exist_ok=True)
                grid = fixed_grid_for_geometry(component, source.crs, map_dx_m, buffer_m)
                dem = read_dem_on_grid(dem_path, grid)
                lia_surface = reproject_source_array(
                    source, source.surface_h_m, grid, resampling=Resampling.bilinear
                )
                lia_thickness = reproject_source_array(
                    source, source.thickness_m, grid, resampling=Resampling.bilinear
                )
                mask = geometry_mask(
                    [component.__geo_interface__],
                    out_shape=(grid.ny, grid.nx),
                    transform=grid_transform(grid),
                    invert=True,
                    all_touched=True,
                )
                routing = np.where(mask & np.isfinite(lia_surface), lia_surface, dem)
                with tempfile.TemporaryDirectory(prefix="lia_gdir_", dir=case_dir) as temp:
                    gdir, _ = prepare_centerline_gdir(
                        directory=Path(temp),
                        rgi_id=f"LIA-{unique_id}-C{component_index}",
                        component_geometry=component,
                        grid=grid,
                        routing_topography_m=routing,
                        raster_island_policy=config.get(
                            "small_component_exclusion"
                        ),
                    )
                    raster_cleanup = dict(gdir.raster_component_cleanup)
                    centerlines = gdir.read_pickle("inversion_flowlines")
                    downstream = gdir.read_pickle("downstream_line")
                    if anchor_reroute:
                        downstream = _reroute_main_from_upstream_anchor(
                            centerlines=centerlines,
                            downstream_line=downstream,
                            component_geometry=component,
                            grid=grid,
                            routing_topography_m=routing,
                            glacier_mask=mask,
                            max_back_nodes=int(anchor_max_back_nodes),
                        )
                        full_coordinates = np.asarray(
                            downstream["full_line"].coords, dtype=float
                        )
                        full_columns = np.clip(
                            np.rint(full_coordinates[:, 0]).astype(int),
                            0,
                            grid.nx - 1,
                        )
                        full_rows = np.clip(
                            np.rint(full_coordinates[:, 1]).astype(int),
                            0,
                            grid.ny - 1,
                        )
                        sampled_surface = np.asarray(
                            routing[full_rows, full_columns], dtype=float
                        )
                        anchor_length = int(downstream["anchor_index"]) + 1
                        downstream["surface_h"] = sampled_surface[anchor_length:]
                    network_payload = _centerline_network_payload(
                        centerlines,
                        downstream,
                        component,
                        grid,
                        source.crs,
                    )
                    # Geometry-only model lines retain the OGGM downstream extension.
                    sections = []
                    for flowline in centerlines:
                        coordinates = np.asarray(flowline.line.coords, dtype=float)
                        columns = np.clip(
                            np.rint(coordinates[:, 0]).astype(int), 0, grid.nx - 1
                        )
                        rows_on_grid = np.clip(
                            np.rint(coordinates[:, 1]).astype(int), 0, grid.ny - 1
                        )
                        thickness = np.asarray(
                            lia_thickness[rows_on_grid, columns], dtype=float
                        )
                        thickness = np.where(
                            np.isfinite(thickness) & (thickness > 0.1), thickness, 0.0
                        )
                        sections.append(thickness)
                    main_model_coordinates = np.asarray(
                        downstream["full_line"].coords, dtype=float
                    )
                    main_columns = np.clip(
                        np.rint(main_model_coordinates[:, 0]).astype(int),
                        0,
                        grid.nx - 1,
                    )
                    main_rows = np.clip(
                        np.rint(main_model_coordinates[:, 1]).astype(int),
                        0,
                        grid.ny - 1,
                    )
                    main_model_section = np.asarray(
                        lia_thickness[main_rows, main_columns], dtype=float
                    )
                    projected_main_model = _grid_line_to_projected(
                        downstream["full_line"], grid
                    )
                    inside_lia = np.asarray(
                        [
                            component.covers(Point(coordinate))
                            for coordinate in projected_main_model.coords
                        ],
                        dtype=bool,
                    )
                    main_model_section = np.where(
                        inside_lia
                        & np.isfinite(main_model_section)
                        & (main_model_section > 0.1),
                        main_model_section,
                        0.0,
                    )
                    model_flowlines = _build_geometry_flowlines(
                        centerlines,
                        downstream,
                        sections,
                        map_dx_m,
                        main_model_section=main_model_section,
                    )
                with gzip.open(
                    case_dir / "lia_centerline_network.pkl.gz", "wb"
                ) as stream:
                    pickle.dump(network_payload, stream)
                audit, spatial = audit_outlet_geometry(
                    centerlines=centerlines,
                    model_flowlines=model_flowlines,
                    network_payload=network_payload,
                    component_geometry=component,
                    grid=grid,
                )
                old_lines = _load_old_modern_network(modern_root, source.rgi_id, source.crs)
                old_endpoint_distance = float(
                    Point(old_lines[-1].coords[-1]).distance(component.boundary)
                )
                audit.update(
                    {
                        "lia_unique_id": unique_id,
                        "rgi_id": source.rgi_id,
                        "component_index": component_index,
                        "map_dx_m": map_dx_m,
                        "flowline_dx": flowline_dx,
                        "buffer_m": buffer_m,
                        "routing_surface": "lia-surface-inside_bare-dem-outside",
                        **raster_cleanup,
                        "old_modern_branch_count": len(old_lines),
                        "old_main_length_m": float(old_lines[-1].length),
                        "old_main_endpoint_to_lia_boundary_m": old_endpoint_distance,
                        "wall_time_seconds": float(time.perf_counter() - started),
                    }
                )
                write_json(case_dir / "outlet_audit.json", audit)
                if not compact_output:
                    _write_vector_outputs(case_dir, source.crs, component, spatial)
                _plot_case(
                    case_dir=case_dir,
                    source=source,
                    geometry=component,
                    old_lines=old_lines,
                    spatial=spatial,
                    audit=audit,
                )
                rows.append({**audit, "status": "ok"})
        except Exception as exc:
            rows.append(
                {
                    "lia_unique_id": unique_id,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(limit=16),
                    "wall_time_seconds": float(time.perf_counter() - started),
                }
            )
        if write_summary_files:
            pd.DataFrame(rows).drop(columns=["gates"], errors="ignore").to_csv(
                output_dir / "lia_gdir_outlet_pilot_results.csv", index=False
            )
    frame = pd.DataFrame(rows)
    summary = {
        "status": "completed" if frame["status"].eq("ok").all() else "completed-with-failures",
        "oggm_version": oggm_version,
        "system_ids": system_ids,
        "case_count": len(frame),
        "ok_count": int(frame["status"].eq("ok").sum()),
        "all_gate_pass_count": int(frame.get("all_gates_pass", pd.Series(dtype=bool)).fillna(False).sum()),
        "map_dx_m": map_dx_m,
        "flowline_dx": flowline_dx,
        "buffer_m": buffer_m,
        "source_products_mutable": False,
    }
    if write_summary_files:
        write_json(output_dir / "lia_gdir_outlet_pilot_summary.json", summary)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system-id", action="append", type=int)
    parser.add_argument("--map-dx", type=float, default=75.0)
    parser.add_argument("--flowline-dx", type=float, default=1.0)
    parser.add_argument("--buffer-m", type=float, default=5000.0)
    parser.add_argument("--output-subdir", default="lia_gdir_outlet_pilot_v2")
    parser.add_argument("--source-subdir")
    parser.add_argument("--compact-output", action="store_true")
    parser.add_argument("--anchor-reroute", action="store_true")
    parser.add_argument("--anchor-max-back-nodes", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, summary = run(
        config,
        system_ids=args.system_id or [7, 553, 1133],
        map_dx_m=args.map_dx,
        flowline_dx=args.flowline_dx,
        buffer_m=args.buffer_m,
        output_subdir=args.output_subdir,
        source_subdir=args.source_subdir,
        compact_output=args.compact_output,
        anchor_reroute=args.anchor_reroute,
        anchor_max_back_nodes=args.anchor_max_back_nodes,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

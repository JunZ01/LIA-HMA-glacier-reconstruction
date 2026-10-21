"""Audit vector topology of the core-191 LIA flowline networks.

The audit is read-only. It checks the archived Farinotti member networks and
the v0.1 default production initial states against the source LIA shapefile.
All geometry is transformed from OGGM grid coordinates to the source LIA CRS
before intersection or containment tests are evaluated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import warnings
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely import make_valid
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import unary_union
from shapely.validation import explain_validity


def read_pickle(path: Path) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open(path, "rb") as source:
            return pickle.load(source)


def read_grid(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def polygonal_part(geometry):
    valid = make_valid(geometry)
    if valid.geom_type in {"Polygon", "MultiPolygon"}:
        return valid
    if isinstance(valid, GeometryCollection):
        polygons = [item for item in valid.geoms if item.geom_type in {"Polygon", "MultiPolygon"}]
        if polygons:
            return unary_union(polygons)
    raise ValueError(f"make-valid-produced-no-polygon:{valid.geom_type}")


def lineal_part(geometry):
    if geometry.is_empty:
        return geometry
    if geometry.geom_type in {"LineString", "MultiLineString"}:
        return geometry
    if isinstance(geometry, GeometryCollection):
        lines = [item for item in geometry.geoms if item.geom_type in {"LineString", "MultiLineString"}]
        return unary_union(lines) if lines else LineString()
    return LineString()


def point_parts(geometry) -> list[Point]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type == "MultiPoint":
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        points: list[Point] = []
        for item in geometry.geoms:
            points.extend(point_parts(item))
        return points
    return []


def grid_line_to_target(line, grid: dict, target_crs: CRS) -> LineString:
    coords = np.asarray(line.coords, dtype=float)
    x0, y0 = map(float, grid["x0y0"])
    dx, dy = map(float, grid["dxdy"])
    x = x0 + coords[:, 0] * dx
    y = y0 + coords[:, 1] * dy
    source_crs = CRS.from_user_input(grid["proj"])
    if not source_crs.equals(target_crs):
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        x, y = transformer.transform(x, y)
    return LineString(np.column_stack((x, y)))


def positive_ice_line(flowline, projected: LineString):
    coords = np.asarray(projected.coords, dtype=float)
    thickness = np.asarray(getattr(flowline, "thick", np.ones(len(coords))), dtype=float)
    if thickness.shape != (len(coords),):
        return projected
    positive_segments = (thickness[:-1] > 1e-6) & (thickness[1:] > 1e-6)
    if not np.any(positive_segments):
        return LineString()
    segments = [
        LineString((coords[index], coords[index + 1]))
        for index in np.flatnonzero(positive_segments)
    ]
    return unary_union(segments)


def line_coordinates(geometry) -> np.ndarray:
    if geometry.is_empty:
        return np.empty((0, 2), dtype=float)
    if geometry.geom_type == "LineString":
        return np.asarray(geometry.coords, dtype=float)
    if geometry.geom_type == "MultiLineString":
        return np.vstack([np.asarray(item.coords, dtype=float) for item in geometry.geoms])
    return np.empty((0, 2), dtype=float)


def parent_indices(flowlines: list) -> list[int | None]:
    by_identity = {id(flowline): index for index, flowline in enumerate(flowlines)}
    output: list[int | None] = []
    for flowline in flowlines:
        parent = getattr(flowline, "flows_to", None)
        output.append(None if parent is None else by_identity.get(id(parent), -1))
    return output


def has_cycle(parents: list[int | None]) -> bool:
    for start in range(len(parents)):
        seen: set[int] = set()
        current: int | None = start
        while current is not None and current >= 0:
            if current in seen:
                return True
            seen.add(current)
            current = parents[current]
    return False


def append_event(events: list[dict], *, geometry=None, **values) -> None:
    row = dict(values)
    row["geometry"] = geometry
    events.append(row)


def audit_network(
    *,
    system_id: int,
    rgi_id: str,
    role: str,
    flowlines: list,
    lines: list[LineString],
    ice_lines: list[LineString],
    polygon,
    map_dx_m: float,
    events: list[dict],
) -> dict:
    parents = parent_indices(flowlines)
    missing_parent_count = sum(parent == -1 for parent in parents)
    root_indices = [index for index, parent in enumerate(parents) if parent is None]
    cycle = has_cycle(parents) if not missing_parent_count else False
    zero_segments = 0
    self_intersections = 0
    boundary_lines = 0
    outside_length_m = 0.0
    hard_boundary_lines = 0
    hard_outside_length_m = 0.0
    maximum_outside_distance_m = 0.0
    boundary_tolerance = max(float(map_dx_m), 1.0)

    for index, (flowline, line, ice_line) in enumerate(zip(flowlines, lines, ice_lines)):
        coords = np.asarray(line.coords, dtype=float)
        segment_lengths = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
        repeated = int(np.count_nonzero(segment_lengths <= 1e-6))
        zero_segments += repeated
        if repeated:
            append_event(
                events,
                geometry=line,
                lia_unique_id=system_id,
                rgi_id=rgi_id,
                network_role=role,
                event_type="zero-length-segment",
                branch_index=index,
                other_rgi_id="",
                other_branch_index="",
                magnitude_m=float(repeated),
                detail=f"count={repeated}",
            )
        if not line.is_simple:
            self_intersections += 1
            append_event(
                events,
                geometry=line,
                lia_unique_id=system_id,
                rgi_id=rgi_id,
                network_role=role,
                event_type="self-intersection",
                branch_index=index,
                other_rgi_id="",
                other_branch_index="",
                magnitude_m=math.nan,
                detail="LineString.is_simple=False",
            )
        if ice_line.is_empty:
            continue
        outside = lineal_part(ice_line.difference(polygon))
        outside_tolerant = lineal_part(ice_line.difference(polygon.buffer(boundary_tolerance)))
        raw_length = float(outside.length)
        tolerant_length = float(outside_tolerant.length)
        if raw_length > 1e-6:
            boundary_lines += 1
            outside_length_m += raw_length
            if tolerant_length > 1e-6:
                hard_boundary_lines += 1
                hard_outside_length_m += tolerant_length
            coords_ice = line_coordinates(ice_line)
            max_distance = max(Point(xy).distance(polygon) for xy in coords_ice)
            maximum_outside_distance_m = max(maximum_outside_distance_m, float(max_distance))
            append_event(
                events,
                geometry=outside,
                lia_unique_id=system_id,
                rgi_id=rgi_id,
                network_role=role,
                event_type=("outside-lia" if tolerant_length > 1e-6 else "boundary-touch-tolerance"),
                branch_index=index,
                other_rgi_id="",
                other_branch_index="",
                magnitude_m=raw_length,
                detail=(
                    f"outside_after_{boundary_tolerance:.3f}m_buffer={tolerant_length:.3f};"
                    f"max_node_distance_m={max_distance:.3f}"
                ),
            )

    duplicates = 0
    overlaps = 0
    nonjunction_crossings = 0
    junction_tolerance = max(1.0, 0.1 * float(map_dx_m))
    for left in range(len(lines)):
        for right in range(left + 1, len(lines)):
            line_left = lines[left]
            line_right = lines[right]
            if line_left.equals(line_right):
                duplicates += 1
                append_event(
                    events,
                    geometry=line_left,
                    lia_unique_id=system_id,
                    rgi_id=rgi_id,
                    network_role=role,
                    event_type="exact-duplicate-branch",
                    branch_index=left,
                    other_rgi_id=rgi_id,
                    other_branch_index=right,
                    magnitude_m=float(line_left.length),
                    detail="topological-equality",
                )
                continue
            intersection = line_left.intersection(line_right)
            overlap = lineal_part(intersection)
            if not overlap.is_empty and overlap.length > 1e-6:
                overlaps += 1
                append_event(
                    events,
                    geometry=overlap,
                    lia_unique_id=system_id,
                    rgi_id=rgi_id,
                    network_role=role,
                    event_type="overlapping-branches",
                    branch_index=left,
                    other_rgi_id=rgi_id,
                    other_branch_index=right,
                    magnitude_m=float(overlap.length),
                    detail="same-member-network",
                )
            points = point_parts(intersection)
            if not points:
                continue
            child = None
            if parents[left] == right:
                child = left
            elif parents[right] == left:
                child = right
            expected = False
            if child is not None:
                child_end = Point(lines[child].coords[-1])
                expected = all(point.distance(child_end) <= junction_tolerance for point in points)
            if not expected:
                nonjunction_crossings += 1
                append_event(
                    events,
                    geometry=unary_union(points),
                    lia_unique_id=system_id,
                    rgi_id=rgi_id,
                    network_role=role,
                    event_type="nonjunction-crossing",
                    branch_index=left,
                    other_rgi_id=rgi_id,
                    other_branch_index=right,
                    magnitude_m=0.0,
                    detail=f"parent_child={child is not None}",
                )

    if missing_parent_count:
        append_event(
            events,
            lia_unique_id=system_id,
            rgi_id=rgi_id,
            network_role=role,
            event_type="missing-parent-reference",
            branch_index="",
            other_rgi_id="",
            other_branch_index="",
            magnitude_m=float(missing_parent_count),
            detail=f"count={missing_parent_count}",
        )
    if cycle:
        append_event(
            events,
            lia_unique_id=system_id,
            rgi_id=rgi_id,
            network_role=role,
            event_type="parent-cycle",
            branch_index="",
            other_rgi_id="",
            other_branch_index="",
            magnitude_m=math.nan,
            detail="cycle-detected",
        )
    if len(root_indices) != 1:
        append_event(
            events,
            lia_unique_id=system_id,
            rgi_id=rgi_id,
            network_role=role,
            event_type="root-count",
            branch_index="",
            other_rgi_id="",
            other_branch_index="",
            magnitude_m=float(len(root_indices)),
            detail=f"root_indices={root_indices}",
        )
    return {
        "lia_unique_id": system_id,
        "rgi_id": rgi_id,
        "network_role": role,
        "branch_count": len(lines),
        "root_count": len(root_indices),
        "missing_parent_count": missing_parent_count,
        "has_parent_cycle": cycle,
        "zero_length_segment_count": zero_segments,
        "self_intersection_count": self_intersections,
        "exact_duplicate_branch_pair_count": duplicates,
        "overlapping_branch_pair_count": overlaps,
        "nonjunction_crossing_pair_count": nonjunction_crossings,
        "positive_ice_outside_lia_branch_count": boundary_lines,
        "positive_ice_outside_lia_length_m": outside_length_m,
        "positive_ice_outside_lia_beyond_tolerance_branch_count": hard_boundary_lines,
        "positive_ice_outside_lia_beyond_tolerance_length_m": hard_outside_length_m,
        "maximum_positive_ice_outside_lia_distance_m": maximum_outside_distance_m,
        "map_dx_m": float(map_dx_m),
    }


def cross_member_audit(
    *,
    system_id: int,
    networks: list[dict],
    events: list[dict],
) -> dict:
    overlap_pairs = 0
    crossing_pairs = 0
    main_overlap_pairs = 0
    main_crossing_pairs = 0
    for left in range(len(networks)):
        for right in range(left + 1, len(networks)):
            first = networks[left]
            second = networks[right]
            for first_index, first_line in enumerate(first["lines"]):
                for second_index, second_line in enumerate(second["lines"]):
                    intersection = first_line.intersection(second_line)
                    if intersection.is_empty:
                        continue
                    main_pair = first_index in first["roots"] and second_index in second["roots"]
                    overlap = lineal_part(intersection)
                    if not overlap.is_empty and overlap.length > 1e-6:
                        overlap_pairs += 1
                        main_overlap_pairs += int(main_pair)
                        append_event(
                            events,
                            geometry=overlap,
                            lia_unique_id=system_id,
                            rgi_id=first["rgi_id"],
                            network_role="cross-member-source",
                            event_type=("cross-member-main-overlap" if main_pair else "cross-member-overlap"),
                            branch_index=first_index,
                            other_rgi_id=second["rgi_id"],
                            other_branch_index=second_index,
                            magnitude_m=float(overlap.length),
                            detail="independent-modern-member-networks",
                        )
                    points = point_parts(intersection)
                    if points:
                        crossing_pairs += 1
                        main_crossing_pairs += int(main_pair)
                        append_event(
                            events,
                            geometry=unary_union(points),
                            lia_unique_id=system_id,
                            rgi_id=first["rgi_id"],
                            network_role="cross-member-source",
                            event_type=("cross-member-main-crossing" if main_pair else "cross-member-crossing"),
                            branch_index=first_index,
                            other_rgi_id=second["rgi_id"],
                            other_branch_index=second_index,
                            magnitude_m=0.0,
                            detail="independent-modern-member-networks",
                        )
    return {
        "cross_member_overlap_pair_count": overlap_pairs,
        "cross_member_crossing_pair_count": crossing_pairs,
        "cross_member_main_overlap_pair_count": main_overlap_pairs,
        "cross_member_main_crossing_pair_count": main_crossing_pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lia-shp", required=True)
    parser.add_argument("--routing-csv", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--geometry-status-csv", required=True)
    parser.add_argument("--production-geometry-status-csv", required=True)
    parser.add_argument("--release-coverage-csv", required=True)
    parser.add_argument("--release-routes-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lia = gpd.read_file(args.lia_shp)
    lia_crs = CRS.from_user_input(lia.crs)
    routing = pd.read_csv(args.routing_csv)
    core_ids = set(
        routing.loc[routing["farinotti_core_191"].astype(bool), "lia_unique_id"].astype(int)
    )
    manifest = pd.read_csv(args.manifest_csv)
    manifest = manifest.loc[manifest["lia_unique_id"].astype(int).isin(core_ids)].copy()
    geometry_status = pd.read_csv(args.geometry_status_csv)
    geometry_status = geometry_status.loc[
        geometry_status["lia_unique_id"].astype(int).isin(core_ids)
        & geometry_status["geometry_ready"].astype(bool)
    ].copy()
    production_geometry_status = pd.read_csv(args.production_geometry_status_csv)
    production_geometry_status = production_geometry_status.loc[
        production_geometry_status["lia_unique_id"].astype(int).isin(core_ids)
        & production_geometry_status["geometry_ready"].astype(bool)
    ].copy()
    coverage = pd.read_csv(args.release_coverage_csv)
    state_by_system = {
        int(row.lia_unique_id): (Path(str(row.default_input_pickle)), "production-default")
        for row in coverage.itertuples()
        if isinstance(row.default_input_pickle, str) and row.default_input_pickle
    }
    routes = pd.read_csv(args.release_routes_csv)
    for system_id in sorted(core_ids - set(state_by_system)):
        candidates = routes.loc[
            routes["lia_unique_id"].astype(int).eq(system_id)
            & routes["input_pickle"].notna()
            & routes["input_pickle"].astype(str).ne("")
        ].copy()
        if candidates.empty:
            continue
        candidates["scenario_rank"] = candidates["bed_scenario"].astype(str).ne("farinotti2019").astype(int)
        candidate = candidates.sort_values(["scenario_rank", "bed_scenario"]).iloc[0]
        state_by_system[system_id] = (Path(str(candidate["input_pickle"])), "review-candidate")
    lia_by_id = {int(row.UniqueID): row.geometry for row in lia.itertuples()}
    manifest_by_id = {int(row.lia_unique_id): row for row in manifest.itertuples()}

    events: list[dict] = []
    member_rows: list[dict] = []
    system_rows: list[dict] = []
    line_features: list[dict] = []

    for system_id in sorted(core_ids):
        source_polygon = lia_by_id[system_id]
        source_valid = bool(source_polygon.is_valid)
        validity_reason = "Valid Geometry" if source_valid else explain_validity(source_polygon)
        polygon = polygonal_part(source_polygon)
        area_delta_fraction = (
            (float(polygon.area) - float(source_polygon.area)) / float(source_polygon.area)
            if source_polygon.area else math.nan
        )
        system_members = geometry_status.loc[
            geometry_status["lia_unique_id"].astype(int).eq(system_id)
        ]
        networks: list[dict] = []
        load_failures = 0
        for member in system_members.itertuples():
            rgi_id = str(member.rgi_id)
            gdir = Path(str(member.gdir_path))
            try:
                flowlines = read_pickle(gdir / "inversion_flowlines.pkl")
                grid = read_grid(gdir / "glacier_grid.json")
                lines = [grid_line_to_target(flowline.line, grid, lia_crs) for flowline in flowlines]
                ice_lines = list(lines)
                parents = parent_indices(flowlines)
                roots = [index for index, parent in enumerate(parents) if parent is None]
                row = audit_network(
                    system_id=system_id,
                    rgi_id=rgi_id,
                    role="member-source",
                    flowlines=flowlines,
                    lines=lines,
                    ice_lines=ice_lines,
                    polygon=polygon,
                    map_dx_m=abs(float(grid["dxdy"][0])),
                    events=events,
                )
                member_rows.append(row)
                networks.append({"rgi_id": rgi_id, "lines": lines, "roots": roots})
                for index, line in enumerate(lines):
                    line_features.append(
                        {
                            "lia_unique_id": system_id,
                            "main_rgi_id": str(manifest_by_id[system_id].main_rgi_id),
                            "rgi_id": rgi_id,
                            "branch_index": index,
                            "is_main_branch": index in roots,
                            "parent_branch_index": parents[index],
                            "geometry": line,
                        }
                    )
            except Exception as exc:
                load_failures += 1
                append_event(
                    events,
                    lia_unique_id=system_id,
                    rgi_id=rgi_id,
                    network_role="member-source",
                    event_type="member-load-failure",
                    branch_index="",
                    other_rgi_id="",
                    other_branch_index="",
                    magnitude_m=math.nan,
                    detail=f"{type(exc).__name__}: {exc}",
                )

        cross = cross_member_audit(system_id=system_id, networks=networks, events=events)
        production_status = "missing"
        production_row = None
        state_selection = state_by_system.get(system_id)
        default_pickle = None if state_selection is None else state_selection[0]
        state_role = "production-default" if state_selection is None else state_selection[1]
        if default_pickle is not None and default_pickle.exists():
            try:
                flowlines = read_pickle(default_pickle)
                rgi_id = default_pickle.parent.name
                source_member = production_geometry_status.loc[
                    production_geometry_status["rgi_id"].astype(str).eq(rgi_id)
                ]
                if source_member.empty:
                    raise KeyError(f"no-gdir-for-production-rgi:{rgi_id}")
                gdir = Path(str(source_member.iloc[0]["gdir_path"]))
                grid = read_grid(gdir / "glacier_grid.json")
                lines = [grid_line_to_target(flowline.line, grid, lia_crs) for flowline in flowlines]
                ice_lines = [positive_ice_line(flowline, line) for flowline, line in zip(flowlines, lines)]
                production_row = audit_network(
                    system_id=system_id,
                    rgi_id=rgi_id,
                    role=state_role,
                    flowlines=flowlines,
                    lines=lines,
                    ice_lines=ice_lines,
                    polygon=polygon,
                    map_dx_m=abs(float(grid["dxdy"][0])),
                    events=events,
                )
                member_rows.append(production_row)
                production_status = "audited" if state_role == "production-default" else "audited-review-candidate"
            except Exception as exc:
                production_status = "failed"
                append_event(
                    events,
                    lia_unique_id=system_id,
                    rgi_id=str(manifest_by_id[system_id].main_rgi_id),
                    network_role=state_role,
                    event_type="production-load-failure",
                    branch_index="",
                    other_rgi_id="",
                    other_branch_index="",
                    magnitude_m=math.nan,
                    detail=f"{type(exc).__name__}: {exc}",
                )

        source_rows = [row for row in member_rows if row["lia_unique_id"] == system_id and row["network_role"] == "member-source"]
        system_rows.append(
            {
                "lia_unique_id": system_id,
                "main_rgi_id": str(manifest_by_id[system_id].main_rgi_id),
                "declared_member_count": int(manifest_by_id[system_id].declared_member_count),
                "audited_member_count": len(source_rows),
                "member_load_failure_count": load_failures,
                "source_lia_valid": source_valid,
                "source_lia_validity_reason": validity_reason,
                "make_valid_area_delta_fraction": area_delta_fraction,
                "source_branch_count": sum(row["branch_count"] for row in source_rows),
                "source_root_count_anomaly_count": sum(row["root_count"] != 1 for row in source_rows),
                "source_missing_parent_count": sum(row["missing_parent_count"] for row in source_rows),
                "source_parent_cycle_count": sum(bool(row["has_parent_cycle"]) for row in source_rows),
                "source_zero_length_segment_count": sum(row["zero_length_segment_count"] for row in source_rows),
                "source_self_intersection_count": sum(row["self_intersection_count"] for row in source_rows),
                "source_exact_duplicate_branch_pair_count": sum(row["exact_duplicate_branch_pair_count"] for row in source_rows),
                "source_overlapping_branch_pair_count": sum(row["overlapping_branch_pair_count"] for row in source_rows),
                "source_nonjunction_crossing_pair_count": sum(row["nonjunction_crossing_pair_count"] for row in source_rows),
                "source_outside_lia_branch_count": sum(row["positive_ice_outside_lia_branch_count"] for row in source_rows),
                "source_outside_lia_length_m": sum(row["positive_ice_outside_lia_length_m"] for row in source_rows),
                "source_outside_lia_beyond_tolerance_branch_count": sum(
                    row["positive_ice_outside_lia_beyond_tolerance_branch_count"] for row in source_rows
                ),
                "source_outside_lia_beyond_tolerance_length_m": sum(
                    row["positive_ice_outside_lia_beyond_tolerance_length_m"] for row in source_rows
                ),
                **cross,
                "production_default_status": production_status,
                "production_branch_count": 0 if production_row is None else production_row["branch_count"],
                "production_root_count_anomaly_count": 0 if production_row is None else int(production_row["root_count"] != 1),
                "production_missing_parent_count": 0 if production_row is None else production_row["missing_parent_count"],
                "production_parent_cycle_count": 0 if production_row is None else int(bool(production_row["has_parent_cycle"])),
                "production_zero_length_segment_count": 0 if production_row is None else production_row["zero_length_segment_count"],
                "production_self_intersection_count": 0 if production_row is None else production_row["self_intersection_count"],
                "production_exact_duplicate_branch_pair_count": 0 if production_row is None else production_row["exact_duplicate_branch_pair_count"],
                "production_overlapping_branch_pair_count": 0 if production_row is None else production_row["overlapping_branch_pair_count"],
                "production_nonjunction_crossing_pair_count": 0 if production_row is None else production_row["nonjunction_crossing_pair_count"],
                "production_positive_ice_outside_lia_branch_count": 0 if production_row is None else production_row["positive_ice_outside_lia_branch_count"],
                "production_positive_ice_outside_lia_length_m": 0.0 if production_row is None else production_row["positive_ice_outside_lia_length_m"],
                "production_positive_ice_outside_lia_beyond_tolerance_branch_count": 0 if production_row is None else production_row["positive_ice_outside_lia_beyond_tolerance_branch_count"],
                "production_positive_ice_outside_lia_beyond_tolerance_length_m": 0.0 if production_row is None else production_row["positive_ice_outside_lia_beyond_tolerance_length_m"],
            }
        )

    systems = pd.DataFrame(system_rows)
    members = pd.DataFrame(member_rows)
    event_frame = pd.DataFrame([{key: value for key, value in row.items() if key != "geometry"} for row in events])
    systems.to_csv(output_dir / "core191_flowline_topology_systems_v1.csv", index=False)
    members.to_csv(output_dir / "core191_flowline_topology_networks_v1.csv", index=False)
    event_frame.to_csv(output_dir / "core191_flowline_topology_events_v1.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    gpd.GeoDataFrame(line_features, geometry="geometry", crs=lia.crs).to_file(
        output_dir / "core191_member_flowlines_v1.geojson", driver="GeoJSON"
    )
    issue_features = [row for row in events if row.get("geometry") is not None and not row["geometry"].is_empty]
    if issue_features:
        gpd.GeoDataFrame(issue_features, geometry="geometry", crs=lia.crs).to_file(
            output_dir / "core191_flowline_topology_issues_v1.geojson", driver="GeoJSON"
        )

    counts = Counter(row["event_type"] for row in events)
    missing_state_ids = systems.loc[
        systems["production_default_status"].eq("missing"), "lia_unique_id"
    ].astype(int).tolist()
    summary = {
        "status": "complete" if len(systems) == 191 else "incomplete",
        "system_count": int(len(systems)),
        "source_member_network_count": int((members["network_role"] == "member-source").sum()),
        "production_default_network_count": int((members["network_role"] == "production-default").sum()),
        "review_candidate_network_count": int((members["network_role"] == "review-candidate").sum()),
        "state_candidate_network_count": int(
            members["network_role"].isin(["production-default", "review-candidate"]).sum()
        ),
        "missing_state_package_count": len(missing_state_ids),
        "missing_state_package_system_ids": missing_state_ids,
        "source_branch_count": int(systems["source_branch_count"].sum()),
        "production_branch_count": int(systems["production_branch_count"].sum()),
        "source_invalid_lia_polygon_count": int((~systems["source_lia_valid"]).sum()),
        "systems_with_source_self_intersection": int((systems["source_self_intersection_count"] > 0).sum()),
        "systems_with_source_graph_topology_issue": int(
            (
                (systems["source_root_count_anomaly_count"] > 0)
                | (systems["source_missing_parent_count"] > 0)
                | (systems["source_parent_cycle_count"] > 0)
                | (systems["source_zero_length_segment_count"] > 0)
            ).sum()
        ),
        "systems_with_source_duplicate_branches": int((systems["source_exact_duplicate_branch_pair_count"] > 0).sum()),
        "systems_with_source_overlapping_branches": int((systems["source_overlapping_branch_pair_count"] > 0).sum()),
        "systems_with_source_nonjunction_crossings": int((systems["source_nonjunction_crossing_pair_count"] > 0).sum()),
        "systems_with_cross_member_main_overlap": int((systems["cross_member_main_overlap_pair_count"] > 0).sum()),
        "systems_with_cross_member_main_crossing": int((systems["cross_member_main_crossing_pair_count"] > 0).sum()),
        "systems_with_source_outside_lia": int((systems["source_outside_lia_branch_count"] > 0).sum()),
        "systems_with_source_outside_lia_beyond_tolerance": int(
            (systems["source_outside_lia_beyond_tolerance_branch_count"] > 0).sum()
        ),
        "systems_with_production_topology_issue": int(
            (
                (systems["production_root_count_anomaly_count"] > 0)
                | (systems["production_missing_parent_count"] > 0)
                | (systems["production_parent_cycle_count"] > 0)
                | (systems["production_zero_length_segment_count"] > 0)
                | (systems["production_self_intersection_count"] > 0)
                | (systems["production_exact_duplicate_branch_pair_count"] > 0)
                | (systems["production_overlapping_branch_pair_count"] > 0)
                | (systems["production_nonjunction_crossing_pair_count"] > 0)
            ).sum()
        ),
        "systems_with_production_positive_ice_outside_lia": int(
            (systems["production_positive_ice_outside_lia_branch_count"] > 0).sum()
        ),
        "systems_with_production_positive_ice_outside_lia_beyond_tolerance": int(
            (systems["production_positive_ice_outside_lia_beyond_tolerance_branch_count"] > 0).sum()
        ),
        "event_counts": dict(sorted(counts.items())),
        "scope_note": (
            "Cross-member intersections are integration risks between independent modern-member networks; "
            "they are not treated as defects in each isolated OGGM member network."
        ),
    }
    (output_dir / "core191_flowline_topology_summary_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

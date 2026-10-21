"""Generate compact upstream-anchor reroute previews for the core191 networks."""

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
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from ..paths import resolve
from .lia_dynamic_matrix_pilot import (
    fixed_grid_for_geometry,
    grid_transform,
    load_source_state,
    reproject_source_array,
    write_json,
)
from .lia_gdir_outlet_pilot import (
    _grid_line_to_projected,
    _reroute_main_from_upstream_anchor,
)
from .lia_system_grid import normalize_lia_components


def _load_pickle(path: Path) -> dict:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def _projected_to_grid(coordinates: np.ndarray, grid) -> np.ndarray:
    return np.column_stack([
        (coordinates[:, 0] - float(grid.x0)) / float(grid.dx),
        (coordinates[:, 1] - float(grid.y0)) / float(grid.dy),
    ])


def _maximum_turn_deg(coordinates: np.ndarray, start_index: int = 1) -> float:
    turns = []
    for index in range(max(1, start_index), len(coordinates) - 1):
        incoming = coordinates[index] - coordinates[index - 1]
        outgoing = coordinates[index + 1] - coordinates[index]
        denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if denominator > 0:
            turns.append(float(np.degrees(np.arccos(np.clip(
                np.dot(incoming, outgoing) / denominator, -1.0, 1.0
            )))))
    return max(turns, default=0.0)


def _smooth_original_tail(
    *,
    current_projected: np.ndarray,
    legacy_index: int,
    minimum_anchor_index: int,
    component,
    map_dx_m: float,
    junctions: list,
) -> dict:
    """Smooth only the terminal tail while preserving outlet and topology."""
    candidates = []
    maximum_back = min(20, legacy_index - int(minimum_anchor_index))
    for back in range(1, maximum_back + 1):
        anchor_index = legacy_index - back
        raw_tail = np.asarray(current_projected[anchor_index:], dtype=float)
        if len(raw_tail) < 3:
            continue
        smoothed = raw_tail
        for iterations in range(1, 5):
            left = 0.75 * smoothed[:-1] + 0.25 * smoothed[1:]
            right = 0.25 * smoothed[:-1] + 0.75 * smoothed[1:]
            refined = [smoothed[0]]
            for first, second in zip(left, right):
                refined.extend([first, second])
            refined.append(smoothed[-1])
            smoothed = np.asarray(refined, dtype=float)
            tail_line = LineString(smoothed)
            stations = np.arange(float(map_dx_m), tail_line.length, float(map_dx_m))
            sampled = [smoothed[0]]
            sampled.extend(
                np.asarray(tail_line.interpolate(float(station)).coords[0])
                for station in stations
            )
            if np.linalg.norm(np.asarray(sampled[-1]) - smoothed[-1]) > 1e-8:
                sampled.append(smoothed[-1])
            combined = np.vstack([
                current_projected[: anchor_index + 1],
                np.asarray(sampled, dtype=float)[1:],
            ])
            line = LineString(combined)
            if not line.is_simple:
                continue
            if line.difference(component.buffer(0.5 * float(map_dx_m))).length > 1e-6:
                continue
            maximum_turn = _maximum_turn_deg(
                combined, max(1, anchor_index - 1)
            )
            if maximum_turn > 45.0 + 1e-8:
                continue
            junction_distance = max(
                (float(line.distance(Point(point))) for point in junctions),
                default=0.0,
            )
            if junction_distance > 2.0 * float(map_dx_m) + 1e-6:
                continue
            candidates.append({
                "coordinates": combined,
                "anchor_index": anchor_index,
                "anchor_back_nodes": back,
                "anchor_join_turn_deg": maximum_turn,
                "iterations": iterations,
            })
            break
    if not candidates:
        raise ValueError("anchor-tail-smoothing-no-valid-candidate")
    return min(
        candidates,
        key=lambda item: (
            item["anchor_back_nodes"],
            item["anchor_join_turn_deg"],
            item["iterations"],
        ),
    )


def _bezier_original_tail(
    *,
    current_projected: np.ndarray,
    legacy_index: int,
    minimum_anchor_index: int,
    component,
    map_dx_m: float,
    junctions: list,
) -> dict:
    """Build a tangent-controlled cubic tail when routed smoothing is insufficient."""
    candidates = []
    start_fractions = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
    end_fractions = (0.1, 0.2, 0.3, 0.4, 0.5)
    for anchor_index in range(legacy_index - 1, minimum_anchor_index - 1, -1):
        start = current_projected[anchor_index]
        outlet = current_projected[-1]
        chord = float(np.linalg.norm(outlet - start))
        if chord <= 0:
            continue
        start_direction = start - current_projected[anchor_index - 1]
        end_direction = current_projected[-1] - current_projected[-2]
        start_direction /= np.linalg.norm(start_direction)
        end_direction /= np.linalg.norm(end_direction)
        for start_fraction in start_fractions:
            for end_fraction in end_fractions:
                control_start = start + start_direction * chord * start_fraction
                control_end = outlet - end_direction * chord * end_fraction
                parameter = np.linspace(0.0, 1.0, 200)[:, None]
                curve = (
                    (1.0 - parameter) ** 3 * start
                    + 3.0 * (1.0 - parameter) ** 2 * parameter * control_start
                    + 3.0 * (1.0 - parameter) * parameter**2 * control_end
                    + parameter**3 * outlet
                )
                curve_line = LineString(curve)
                stations = np.arange(
                    float(map_dx_m), curve_line.length, float(map_dx_m)
                )
                sampled = [start]
                sampled.extend(
                    np.asarray(curve_line.interpolate(float(station)).coords[0])
                    for station in stations
                )
                if np.linalg.norm(np.asarray(sampled[-1]) - outlet) > 1e-8:
                    sampled.append(outlet)
                combined = np.vstack([
                    current_projected[: anchor_index + 1],
                    np.asarray(sampled, dtype=float)[1:],
                ])
                line = LineString(combined)
                if not line.is_simple:
                    continue
                if line.difference(
                    component.buffer(0.5 * float(map_dx_m))
                ).length > 1e-6:
                    continue
                maximum_turn = _maximum_turn_deg(
                    combined, max(1, anchor_index - 1)
                )
                if maximum_turn > 45.0 + 1e-8:
                    continue
                junction_distance = max(
                    (float(line.distance(Point(point))) for point in junctions),
                    default=0.0,
                )
                if junction_distance > 2.0 * float(map_dx_m) + 1e-6:
                    continue
                candidates.append({
                    "coordinates": combined,
                    "anchor_index": anchor_index,
                    "anchor_back_nodes": legacy_index - anchor_index,
                    "anchor_join_turn_deg": maximum_turn,
                    "start_fraction": start_fraction,
                    "end_fraction": end_fraction,
                })
    if not candidates:
        raise ValueError("anchor-tail-bezier-no-valid-candidate")
    return min(
        candidates,
        key=lambda item: (
            item["anchor_back_nodes"],
            item["anchor_join_turn_deg"],
            item["start_fraction"],
            item["end_fraction"],
        ),
    )


def _component_paths(output_root: Path, unique_id: int, component_index: int):
    case = output_root / str(unique_id) / f"component_{component_index}"
    image = (
        output_root
        / "images"
        / f"system_{unique_id}_component_{component_index}.png"
    )
    return case, image, case / "anchor_reroute_preview.json"


def _run_component(payload: tuple) -> dict:
    (
        config,
        unique_id,
        component_index,
        map_dx_m,
        output_subdir,
        include_tributaries,
        network_output_subdir,
    ) = payload
    output_root = resolve(config["output_dir"]) / output_subdir
    case_dir, image_path, audit_path = _component_paths(
        output_root, unique_id, component_index
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    network_path = (
        None
        if network_output_subdir is None
        else resolve(config["output_dir"])
        / network_output_subdir
        / str(unique_id)
        / f"component_{component_index}"
        / "lia_centerline_network.pkl.gz"
    )
    if (
        image_path.exists()
        and audit_path.exists()
        and (network_path is None or network_path.exists())
    ):
        return json.loads(audit_path.read_text(encoding="utf-8"))
    try:
        source_root = (
            resolve(config["output_dir"])
            / config["lia_3d_system_grid_pilot_v1"]["output_subdir"]
        )
        source_paths = sorted(source_root.glob(f"lia_{unique_id}_*_system_grid_v1.nc"))
        if len(source_paths) != 1:
            raise ValueError(f"source-grid-count:{len(source_paths)}")
        source = load_source_state(source_paths[0], "farinotti2019")
        outlines = gpd.read_file(resolve(config["lia_geometry_path"]))
        geometry = gpd.GeoSeries(
            [outlines.set_index("UniqueID").loc[unique_id].geometry],
            crs=outlines.crs,
        ).to_crs(source.crs).iloc[0]
        components = normalize_lia_components(geometry)
        component = components[component_index]
        grid = fixed_grid_for_geometry(
            component, source.crs, float(map_dx_m), 5000.0
        )
        surface = reproject_source_array(
            source, source.surface_h_m, grid, resampling=Resampling.bilinear
        )
        mask = geometry_mask(
            [component.__geo_interface__],
            out_shape=(grid.ny, grid.nx),
            transform=grid_transform(grid),
            invert=True,
            all_touched=True,
        )
        routing = np.where(mask & np.isfinite(surface), surface, np.nan)
        routing = np.where(
            np.isfinite(routing), routing, float(np.nanmax(routing[mask])) + 500.0
        )

        result_root = resolve(config["output_dir"])
        v1 = _load_pickle(
            result_root
            / "lia_gdir_outlet_core191_v1"
            / str(unique_id)
            / f"component_{component_index}"
            / "lia_centerline_network.pkl.gz"
        )
        v2 = _load_pickle(
            result_root
            / "lia_gdir_outlet_core191_v2"
            / str(unique_id)
            / f"component_{component_index}"
            / "lia_centerline_network.pkl.gz"
        )
        legacy_projected = np.asarray(v1["branches"][-1]["coords_projected_m"], float)
        current_projected = np.asarray(v2["branches"][-1]["coords_projected_m"], float)
        legacy_grid = _projected_to_grid(legacy_projected, grid)
        current_grid = _projected_to_grid(current_projected, grid)
        main_index = int(v2.get("main_branch_index", len(v2["branches"]) - 1))
        initial_main_line = LineString(current_projected)
        initial_junction_distances = []
        for branch in v2["branches"]:
            if branch.get("parent_index") != main_index:
                continue
            junction = branch.get("flows_to_point_projected_m")
            if junction is not None:
                initial_junction_distances.append(
                    float(initial_main_line.distance(Point(junction)))
                )
        restored_legacy_terminus = False
        if max(initial_junction_distances, default=0.0) > 2.0 * float(
            map_dx_m
        ) + 1e-6:
            _, true_terminus = nearest_points(
                Point(legacy_projected[-1]), component.exterior
            )
            terminus = np.asarray(true_terminus.coords[0], dtype=float)
            current_projected = legacy_projected.copy()
            if np.linalg.norm(current_projected[-1] - terminus) > 1e-8:
                current_projected = np.vstack([current_projected, terminus])
            current_grid = _projected_to_grid(current_projected, grid)
            restored_legacy_terminus = True
        effective_outlet = (
            np.asarray(current_projected[-1], dtype=float)
            if restored_legacy_terminus
            else np.asarray(v2["main_outlet_projected_m"], dtype=float)
        )
        minimum_anchor_index = 1
        for branch in v2["branches"]:
            if branch.get("parent_index") != main_index:
                continue
            junction = branch.get("flows_to_point_projected_m")
            if junction is None:
                continue
            junction_grid = _projected_to_grid(
                np.asarray([junction], dtype=float), grid
            )[0]
            nearest = int(np.argmin(np.hypot(
                legacy_grid[:, 0] - junction_grid[0],
                legacy_grid[:, 1] - junction_grid[1],
            )))
            minimum_anchor_index = max(minimum_anchor_index, nearest + 1)
        main = SimpleNamespace(
            line=LineString(legacy_grid),
            surface_h=np.ones(len(legacy_grid)),
            flows_to=None,
            dx=1.0,
            dx_meter=float(map_dx_m),
        )
        downstream = {
            "full_line": LineString(current_grid),
            "surface_h": np.zeros(max(1, len(current_grid) - len(legacy_grid))),
        }
        current_line_for_gate = LineString(current_projected)
        legacy_station = float(
            current_line_for_gate.project(Point(legacy_projected[-1]))
        )
        segment_lengths = np.hypot(
            np.diff(current_projected[:, 0]), np.diff(current_projected[:, 1])
        )
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        legacy_index = max(1, int(np.searchsorted(cumulative, legacy_station) - 1))
        original_turns = []
        for index in range(max(1, legacy_index - 2), len(current_projected) - 1):
            incoming = current_projected[index] - current_projected[index - 1]
            outgoing = current_projected[index + 1] - current_projected[index]
            denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
            if denominator > 0:
                original_turns.append(float(np.degrees(np.arccos(np.clip(
                    np.dot(incoming, outgoing) / denominator, -1.0, 1.0
                )))))
        original_max_turn = max(original_turns, default=0.0)
        original_junction_distances = []
        for branch in v2["branches"]:
            if branch.get("parent_index") != main_index:
                continue
            junction = branch.get("flows_to_point_projected_m")
            if junction is not None:
                original_junction_distances.append(
                    float(current_line_for_gate.distance(Point(junction)))
                )
        original_route_acceptable = bool(
            current_line_for_gate.is_simple
            and original_max_turn <= 45.0
            and max(original_junction_distances, default=0.0)
            <= 2.0 * float(map_dx_m) + 1e-6
        )
        def retain_original_route() -> dict:
            retained = dict(downstream)
            retained.update({
                "anchor_index": len(legacy_grid) - 1,
                "anchor_point_grid": legacy_grid[-1].tolist(),
                "anchor_back_nodes": 0,
                "anchor_join_turn_deg": float(original_max_turn),
                "anchor_route_cost": None,
                "anchor_rerouted": False,
            })
            return retained
        tail_smoothed = False
        tail_bezier = False
        try:
            rerouted = _reroute_main_from_upstream_anchor(
                centerlines=[main],
                downstream_line=downstream,
                component_geometry=component,
                grid=grid,
                routing_topography_m=routing,
                glacier_mask=mask,
                max_back_nodes=8,
                outlet_projected_m=effective_outlet,
                minimum_anchor_index=minimum_anchor_index,
            )
        except ValueError:
            if original_route_acceptable:
                rerouted = retain_original_route()
            else:
                junctions = [
                    branch["flows_to_point_projected_m"]
                    for branch in v2["branches"]
                    if branch.get("parent_index") == main_index
                    and branch.get("flows_to_point_projected_m") is not None
                ]
                try:
                    smoothed = _smooth_original_tail(
                        current_projected=current_projected,
                        legacy_index=legacy_index,
                        minimum_anchor_index=minimum_anchor_index,
                        component=component,
                        map_dx_m=float(map_dx_m),
                        junctions=junctions,
                    )
                except ValueError:
                    smoothed = _bezier_original_tail(
                        current_projected=current_projected,
                        legacy_index=legacy_index,
                        minimum_anchor_index=minimum_anchor_index,
                        component=component,
                        map_dx_m=float(map_dx_m),
                        junctions=junctions,
                    )
                    tail_bezier = True
                smoothed_grid = _projected_to_grid(
                    smoothed["coordinates"], grid
                )
                rerouted = dict(downstream)
                rerouted.update({
                    "full_line": LineString(smoothed_grid),
                    "anchor_index": int(smoothed["anchor_index"]),
                    "anchor_point_grid": smoothed_grid[
                        int(smoothed["anchor_index"])
                    ].tolist(),
                    "anchor_back_nodes": int(smoothed["anchor_back_nodes"]),
                    "anchor_join_turn_deg": float(
                        smoothed["anchor_join_turn_deg"]
                    ),
                    "anchor_route_cost": None,
                    "anchor_rerouted": True,
                })
                tail_smoothed = True
        rerouted_projected = _grid_line_to_projected(rerouted["full_line"], grid)
        candidate_line = LineString(rerouted_projected.coords)
        candidate_junction_distances = []
        for branch in v2["branches"]:
            if branch.get("parent_index") != main_index:
                continue
            junction = branch.get("flows_to_point_projected_m")
            if junction is not None:
                candidate_junction_distances.append(
                    float(candidate_line.distance(Point(junction)))
                )
        if (
            original_route_acceptable
            and (
                not candidate_line.is_simple
                or max(candidate_junction_distances, default=0.0)
                > 2.0 * float(map_dx_m) + 1e-6
            )
        ):
            rerouted = retain_original_route()
            rerouted_projected = _grid_line_to_projected(
                rerouted["full_line"], grid
            )
        anchor_projected = np.asarray(
            _grid_line_to_projected(
                LineString([
                    rerouted["anchor_point_grid"],
                    rerouted["anchor_point_grid"],
                ]),
                grid,
            ).coords[0]
        )
        outlet = np.asarray(effective_outlet, float)
        network_written = False
        maximum_main_junction_distance_m = 0.0
        if network_path is not None:
            network_payload = pickle.loads(pickle.dumps(v2))
            main_index = int(network_payload.get(
                "main_branch_index", len(network_payload["branches"]) - 1
            ))
            rerouted_coordinates = np.asarray(rerouted_projected.coords, dtype=float)
            rerouted_grid = _projected_to_grid(rerouted_coordinates, grid)
            columns = np.clip(
                np.rint(rerouted_grid[:, 0]).astype(int), 0, grid.nx - 1
            )
            rows = np.clip(
                np.rint(rerouted_grid[:, 1]).astype(int), 0, grid.ny - 1
            )
            reference_surface = np.asarray(routing[rows, columns], dtype=float)
            if reference_surface.shape != (len(rerouted_coordinates),):
                raise ValueError("rerouted-reference-surface-shape-mismatch")
            if not np.isfinite(reference_surface).all():
                raise ValueError("rerouted-reference-surface-nonfinite")
            network_payload["branches"][main_index]["coords_projected_m"] = (
                rerouted_coordinates
            )
            network_payload["branches"][main_index]["reference_surface_m"] = (
                reference_surface
            )
            network_payload["main_outlet_projected_m"] = outlet.tolist()
            network_payload["anchor_rerouted"] = bool(
                rerouted.get("anchor_rerouted", False)
            )
            network_payload["anchor_back_nodes"] = int(
                rerouted["anchor_back_nodes"]
            )
            network_payload["anchor_join_turn_deg"] = float(
                rerouted["anchor_join_turn_deg"]
            )
            network_payload["anchor_point_projected_m"] = anchor_projected.tolist()
            network_payload["network_version"] = "lia-anchor-reroute-v3"
            rerouted_line = LineString(rerouted_coordinates)
            for branch in network_payload["branches"]:
                if int(branch["branch_index"]) == main_index:
                    continue
                if branch.get("parent_index") != main_index:
                    continue
                junction = branch.get("flows_to_point_projected_m")
                if junction is None:
                    raise ValueError("main-child-missing-junction-point")
                distance = float(rerouted_line.distance(Point(junction)))
                maximum_main_junction_distance_m = max(
                    maximum_main_junction_distance_m, distance
                )
            if maximum_main_junction_distance_m > 2.0 * float(map_dx_m) + 1e-6:
                raise ValueError(
                    "rerouted-main-junction-distance:"
                    f"{maximum_main_junction_distance_m:.3f}"
                )
            if not rerouted_line.is_simple:
                raise ValueError("rerouted-main-self-intersection")
            network_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(network_path, "wb") as stream:
                pickle.dump(network_payload, stream)
            network_written = True
        bounds = component.bounds
        pad = 0.05 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        fig, axis = plt.subplots(figsize=(8, 8))
        shown = np.where(mask, surface, np.nan)
        extent = [
            float(grid.x0),
            float(grid.x0 + grid.nx * grid.dx),
            float(grid.y0 + grid.ny * grid.dy),
            float(grid.y0),
        ]
        finite_surface = shown[np.isfinite(shown)]
        if finite_surface.size:
            low, high = np.percentile(finite_surface, [2, 98])
            axis.imshow(
                shown,
                extent=extent,
                origin="upper",
                cmap="terrain",
                vmin=low,
                vmax=high,
                alpha=0.65,
            )
        x, y = component.exterior.xy
        axis.plot(x, y, color="black", linewidth=1.4, label="LIA SHP exterior")
        for interior in component.interiors:
            ix, iy = interior.xy
            axis.plot(ix, iy, color="black", linewidth=0.8)
        current_line = LineString(current_projected)
        tributary_count = max(0, len(v2["branches"]) - 1)
        if include_tributaries:
            for branch_index, branch in enumerate(v2["branches"][:-1]):
                tributary = LineString(
                    np.asarray(branch["coords_projected_m"], dtype=float)
                )
                axis.plot(
                    *tributary.xy,
                    color="#477da8",
                    linewidth=1.0,
                    alpha=0.9,
                    label="tributary flowlines" if branch_index == 0 else None,
                )
        axis.plot(
            *current_line.xy,
            color="#b42318",
            linewidth=1.4,
            linestyle="--",
            label="current outlet route",
        )
        axis.plot(
            *rerouted_projected.xy,
            color="#2563a6",
            linewidth=2.3,
            label="upstream-anchor reroute",
        )
        axis.scatter(
            [legacy_projected[-1, 0]],
            [legacy_projected[-1, 1]],
            color="#d97706",
            s=45,
            label="legacy endpoint",
            zorder=5,
        )
        axis.scatter(
            [anchor_projected[0]],
            [anchor_projected[1]],
            color="#16803a",
            s=45,
            label="selected anchor",
            zorder=5,
        )
        axis.scatter(
            [outlet[0]], [outlet[1]], color="black", marker="X", s=65,
            label="fixed LIA outlet", zorder=6
        )
        axis.set_xlim(bounds[0] - pad, bounds[2] + pad)
        axis.set_ylim(bounds[1] - pad, bounds[3] + pad)
        axis.set_aspect("equal")
        axis.set_title(f"LIA system {unique_id}, component {component_index}")
        axis.set_xlabel("Projected x (m)")
        axis.set_ylabel("Projected y (m)")
        axis.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(image_path, dpi=180)
        plt.close(fig)
        audit = {
            "status": "ok",
            "lia_unique_id": int(unique_id),
            "component_index": int(component_index),
            "map_dx_m": float(map_dx_m),
            "anchor_back_nodes": int(rerouted["anchor_back_nodes"]),
            "anchor_join_turn_deg": float(rerouted["anchor_join_turn_deg"]),
            "anchor_rerouted": bool(rerouted.get("anchor_rerouted", False)),
            "route_policy": (
                "restored-legacy-main-to-nearest-terminal-exterior"
                if restored_legacy_terminus
                else "anchor-tail-cubic-bezier"
                if tail_bezier
                else "anchor-tail-chaikin-smoothing"
                if tail_smoothed
                else "upstream-anchor-reroute"
                if rerouted.get("anchor_rerouted", False)
                else "retained-original-route-passed-turn-gate"
            ),
            "legacy_endpoint_projected_m": legacy_projected[-1].tolist(),
            "selected_anchor_projected_m": anchor_projected.tolist(),
            "fixed_outlet_projected_m": outlet.tolist(),
            "preview_only": True,
            "production_network_written": bool(network_written),
            "network_path": (
                None if network_path is None else str(network_path.resolve())
            ),
            "maximum_main_junction_distance_m": float(
                maximum_main_junction_distance_m
            ),
            "include_tributaries": bool(include_tributaries),
            "tributary_count": int(tributary_count),
        }
        write_json(audit_path, audit)
        return audit
    except Exception as exc:
        if "component" in locals() and "current_projected" in locals():
            fig, axis = plt.subplots(figsize=(8, 8))
            x, y = component.exterior.xy
            axis.plot(x, y, color="black", linewidth=1.4, label="LIA SHP exterior")
            current_line = LineString(current_projected)
            if include_tributaries and "v2" in locals():
                for branch_index, branch in enumerate(v2["branches"][:-1]):
                    tributary = LineString(
                        np.asarray(branch["coords_projected_m"], dtype=float)
                    )
                    axis.plot(
                        *tributary.xy,
                        color="#477da8",
                        linewidth=1.0,
                        alpha=0.9,
                        label="tributary flowlines" if branch_index == 0 else None,
                    )
            axis.plot(
                *current_line.xy,
                color="#b42318",
                linewidth=1.8,
                linestyle="--",
                label="current outlet route",
            )
            axis.text(
                0.02,
                0.02,
                "No upstream anchor satisfies the 45 deg turn gate",
                transform=axis.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            bounds = component.bounds
            pad = 0.05 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            axis.set_xlim(bounds[0] - pad, bounds[2] + pad)
            axis.set_ylim(bounds[1] - pad, bounds[3] + pad)
            axis.set_aspect("equal")
            axis.set_title(
                f"LIA system {unique_id}, component {component_index}: review required"
            )
            axis.set_xlabel("Projected x (m)")
            axis.set_ylabel("Projected y (m)")
            axis.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(image_path, dpi=180)
            plt.close(fig)
        audit = {
            "status": "failed",
            "lia_unique_id": int(unique_id),
            "component_index": int(component_index),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(limit=12),
            "preview_only": True,
            "production_network_written": False,
            "include_tributaries": bool(include_tributaries),
        }
        write_json(audit_path, audit)
        return audit


def run(
    config: dict,
    *,
    workers: int,
    output_subdir: str,
    include_tributaries: bool = False,
    network_output_subdir: str | None = None,
) -> dict:
    result_root = resolve(config["output_dir"])
    v2_root = result_root / "lia_gdir_outlet_core191_v2"
    payloads = []
    for audit_path in sorted(v2_root.glob("*/component_*/outlet_audit.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        payloads.append((
            config,
            int(audit["lia_unique_id"]),
            int(audit["component_index"]),
            float(audit["map_dx_m"]),
            output_subdir,
            include_tributaries,
            network_output_subdir,
        ))
    records = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [executor.submit(_run_component, payload) for payload in payloads]
        for future in as_completed(futures):
            records.append(future.result())
    frame = pd.DataFrame(records).sort_values(
        ["lia_unique_id", "component_index"]
    )
    output_root = result_root / output_subdir
    frame.drop(columns=["failure_traceback"], errors="ignore").to_csv(
        output_root / "anchor_reroute_preview_results.csv", index=False
    )
    ok = frame[frame["status"].eq("ok")]
    summary = {
        "status": "completed" if len(ok) == len(frame) else "completed-with-failures",
        "component_count": int(len(frame)),
        "successful_component_count": int(len(ok)),
        "failed_component_count": int(len(frame) - len(ok)),
        "worker_count": int(workers),
        "preview_only": True,
        "production_network_written": bool(network_output_subdir),
        "network_output_directory": (
            None
            if network_output_subdir is None
            else str((result_root / network_output_subdir).resolve())
        ),
        "include_tributaries": bool(include_tributaries),
        "output_directory": str(output_root.resolve()),
        "image_directory": str((output_root / "images").resolve()),
    }
    write_json(output_root / "anchor_reroute_preview_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument(
        "--output-subdir", default="anchor_reroute_preview_core191_v1"
    )
    parser.add_argument("--include-tributaries", action="store_true")
    parser.add_argument("--network-output-subdir")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(
        run(
            config,
            workers=args.workers,
            output_subdir=args.output_subdir,
            include_tributaries=args.include_tributaries,
            network_output_subdir=args.network_output_subdir,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()

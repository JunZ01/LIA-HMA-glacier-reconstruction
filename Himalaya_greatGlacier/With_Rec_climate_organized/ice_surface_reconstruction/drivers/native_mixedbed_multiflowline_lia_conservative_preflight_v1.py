"""Section 176/177 native-MixedBed multi-flowline preflight for system 3850.

The driver is deliberately self-contained and versioned.  It clones every
source glacier directory, registers the exact F19 inversion in the clone,
initialises OGGM-native MixedBedFlowlines, and then maps the paired LIA grid
conservatively by spatial proximity.  It never mutates source glacier
directories and never runs historical climate.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from .flux_mass_budget import budget_rows_as_dicts, run_zero_mb_budget
from .initial_state_adapter import gdir_path, read_json
from .initial_state_merged_network_pilot import _project_flowlines_to_main_grid
from .v3_flowline_geometry_audit import (
    maximum_cumulative_hypsometry_error_fraction,
)


SCHEMA_VERSION = 1
SUPPORTED_SYSTEM_ID = 3850


def _write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(safe(payload), indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _relative_gdir_path(rgi_id: str) -> Path:
    return Path(rgi_id[:8]) / rgi_id[:11] / rgi_id


def _source_records(source_root: Path, member_ids: list[str], suffix: str) -> list[dict]:
    records = []
    for rgi_id in member_ids:
        directory = gdir_path(source_root, rgi_id)
        required = {
            "inversion_flowlines": directory / "inversion_flowlines.pkl",
            "f19_inversion": directory / f"inversion_output{suffix}.pkl",
            "unsuffixed_inversion": directory / "inversion_output.pkl",
            "downstream_line": directory / "downstream_line.pkl",
            "glacier_grid": directory / "glacier_grid.json",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing-source-gdir-input:" + ";".join(missing))
        records.append({
            "rgi_id": rgi_id,
            "source_directory": str(directory.resolve()),
            "files": {
                name: {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for name, path in required.items()
            },
        })
    return records


def _verify_sources_unchanged(records: list[dict]) -> list[dict]:
    checks = []
    for record in records:
        for name, item in record["files"].items():
            path = Path(item["path"])
            current = _sha256(path)
            checks.append({
                "rgi_id": record["rgi_id"],
                "file_role": name,
                "path": str(path),
                "before_sha256": item["sha256"],
                "after_sha256": current,
                "unchanged": current == item["sha256"],
            })
    return checks


def _clone_and_initialize(
    *,
    source_root: Path,
    clone_root: Path,
    member_ids: list[str],
    suffix: str,
) -> tuple[dict[str, list], list[dict]]:
    from oggm import GlacierDirectory, tasks

    initialized: dict[str, list] = {}
    audits = []
    for rgi_id in member_ids:
        source_dir = gdir_path(source_root, rgi_id)
        clone_dir = clone_root / _relative_gdir_path(rgi_id)
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, clone_dir)
        f19 = clone_dir / f"inversion_output{suffix}.pkl"
        unsuffixed = clone_dir / "inversion_output.pkl"
        shutil.copy2(f19, unsuffixed)
        if _sha256(f19) != _sha256(unsuffixed):
            raise RuntimeError(f"f19-registration-hash-mismatch:{rgi_id}")
        gdir = GlacierDirectory(rgi_id, base_dir=str(clone_root))
        inversion_flowlines = gdir.read_pickle("inversion_flowlines")
        tasks.init_present_time_glacier(gdir)
        model_flowlines = gdir.read_pickle("model_flowlines")
        if len(model_flowlines) != len(inversion_flowlines):
            raise RuntimeError(f"native-flowline-count-changed:{rgi_id}")
        for index, flowline in enumerate(model_flowlines):
            flowline._section176_member_id = rgi_id
            flowline._section176_local_index = int(index)
            flowline._section176_native_nx = int(len(inversion_flowlines[index].surface_h))
        initialized[rgi_id] = model_flowlines
        audits.append({
            "rgi_id": rgi_id,
            "clone_directory": str(clone_dir.resolve()),
            "f19_registered_sha256": _sha256(f19),
            "unsuffixed_registered_sha256": _sha256(unsuffixed),
            "native_flowline_count": len(model_flowlines),
            "native_root_count": int(sum(fl.flows_to is None for fl in model_flowlines)),
            "native_node_counts": [int(fl.nx) for fl in model_flowlines],
            "native_inversion_node_counts": [
                int(len(fl.surface_h)) for fl in inversion_flowlines
            ],
        })
    return initialized, audits


def _combine_native_network(
    initialized: dict[str, list],
    *,
    main_rgi_id: str,
    member_ids: list[str],
    source_root: Path,
) -> tuple[list, list[dict], dict]:
    main_grid = read_json(gdir_path(source_root, main_rgi_id) / "glacier_grid.json")
    by_member: dict[str, list] = {}
    for rgi_id in member_ids:
        if rgi_id == main_rgi_id:
            flowlines = copy.deepcopy(initialized[rgi_id])
        else:
            flowlines = _project_flowlines_to_main_grid(
                initialized[rgi_id],
                read_json(gdir_path(source_root, rgi_id) / "glacier_grid.json"),
                main_grid,
            )
        for index, flowline in enumerate(flowlines):
            flowline._section176_member_id = rgi_id
            flowline._section176_local_index = int(index)
            flowline._section176_native_nx = int(
                getattr(initialized[rgi_id][index], "_section176_native_nx")
            )
        by_member[rgi_id] = flowlines

    main_flowlines = by_member[main_rgi_id]
    main_roots = [fl for fl in main_flowlines if fl.flows_to is None]
    if len(main_roots) != 1:
        raise RuntimeError(f"main-member-root-count:{len(main_roots)}")
    connections = []
    for rgi_id in member_ids:
        if rgi_id == main_rgi_id:
            continue
        roots = [fl for fl in by_member[rgi_id] if fl.flows_to is None]
        if len(roots) != 1:
            raise RuntimeError(f"member-root-count:{rgi_id}:{len(roots)}")
        root = roots[0]
        tail = Point(root.line.coords[-1])
        target = min(main_flowlines, key=lambda fl: fl.line.distance(tail))
        distance_pixels = float(target.line.distance(tail))
        root.set_flows_to(target)
        connections.append({
            "source_member_rgi_id": rgi_id,
            "source_local_index": int(root._section176_local_index),
            "target_member_rgi_id": main_rgi_id,
            "target_local_index": int(target._section176_local_index),
            "terminal_to_target_distance_pixels": distance_pixels,
            "target_node_index": int(root.flows_to_indice),
            "relation": "spatial-nearest-native-control-line",
        })

    combined = [fl for rgi_id in member_ids for fl in by_member[rgi_id]]
    roots = [fl for fl in combined if fl.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"combined-root-count:{len(roots)}")
    audit = {
        "member_count": len(by_member),
        "flowline_count": len(combined),
        "combined_root_count": len(roots),
        "combined_root_member_rgi_id": roots[0]._section176_member_id,
        "combined_root_local_index": int(roots[0]._section176_local_index),
        "main_grid": main_grid,
    }
    return combined, connections, audit


def _flowline_coordinates_in_crs(flowlines: list, main_grid: dict, target_crs: CRS):
    source_crs = CRS.from_user_input(main_grid["proj"])
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x0, y0 = map(float, main_grid["x0y0"])
    dx, dy = map(float, main_grid["dxdy"])
    coordinates = []
    rows = []
    for global_index, flowline in enumerate(flowlines):
        pixels = np.asarray(flowline.line.coords, dtype=float)
        x = x0 + pixels[:, 0] * dx
        y = y0 + pixels[:, 1] * dy
        x, y = transformer.transform(x, y)
        xy = np.column_stack([x, y])
        coordinates.append(xy)
        for node_index, point in enumerate(xy):
            rows.append({
                "global_flowline_index": int(global_index),
                "member_rgi_id": flowline._section176_member_id,
                "local_flowline_index": int(flowline._section176_local_index),
                "node_index": int(node_index),
                "x_m": float(point[0]),
                "y_m": float(point[1]),
            })
    return coordinates, rows


def _load_paired_source(path: Path, tolerance_m: float) -> dict:
    with xr.open_dataset(path) as dataset:
        ids = [str(value) for value in dataset["member_id"].values]
        member_index = ids.index("farinotti2019")
        thickness = np.asarray(
            dataset["thickness_m"].isel(member=member_index).values, dtype=float
        )
        surface = np.asarray(
            dataset["surface_elevation_m"].isel(member=member_index).values,
            dtype=float,
        )
        bed = np.asarray(
            dataset["bed_elevation_m"].isel(member=member_index).values,
            dtype=float,
        )
        valid = (
            np.asarray(dataset["valid_mask"].isel(member=member_index).values) > 0
        ) & (np.asarray(dataset["lia_mask"].values) > 0)
        valid &= np.isfinite(thickness) & np.isfinite(surface) & np.isfinite(bed)
        positive = valid & (thickness > tolerance_m)
        x = np.asarray(dataset["x"].values, dtype=float)
        y = np.asarray(dataset["y"].values, dtype=float)
        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))
        crs = CRS.from_wkt(str(dataset["spatial_ref"].attrs["crs_wkt"]))
        attrs = dict(dataset.attrs)
    rows, cols = np.nonzero(positive)
    cell_area = abs(dx * dy)
    return {
        "path": path,
        "sha256": _sha256(path),
        "crs": crs,
        "x": x,
        "y": y,
        "rows": rows,
        "cols": cols,
        "cell_xy": np.column_stack([x[cols], y[rows]]),
        "cell_thickness": thickness[rows, cols],
        "cell_surface": surface[rows, cols],
        "cell_area_m2": cell_area,
        "source_area_m2": float(len(rows) * cell_area),
        "source_volume_m3": float(np.sum(thickness[rows, cols]) * cell_area),
        "attrs": attrs,
    }


def _spatial_crosswalk(
    *,
    network_path: Path,
    native_coordinates: list[np.ndarray],
    flowlines: list,
    target_crs: CRS,
) -> tuple[pd.DataFrame, dict]:
    with gzip.open(network_path, "rb") as stream:
        network = pickle.load(stream)
    branches = list(network["branches"])
    transformer = Transformer.from_crs(
        CRS.from_user_input(network["crs"]), target_crs, always_xy=True
    )
    lia_lines = []
    for branch in branches:
        xy = np.asarray(branch["coords_projected_m"], dtype=float)
        x, y = transformer.transform(xy[:, 0], xy[:, 1])
        lia_lines.append(LineString(np.column_stack([x, y])))
    native_lines = [LineString(xy) for xy in native_coordinates]
    rows = []
    for branch_index, line in enumerate(lia_lines):
        distances = np.asarray([line.distance(other) for other in native_lines])
        nearest = int(np.argmin(distances))
        rows.append({
            "direction": "lia-branch-to-native-flowline",
            "lia_branch_index": int(branch_index),
            "lia_is_main": int(branch_index) == int(network["main_branch_index"]),
            "native_global_index": nearest,
            "native_member_rgi_id": flowlines[nearest]._section176_member_id,
            "native_local_index": int(flowlines[nearest]._section176_local_index),
            "distance_m": float(distances[nearest]),
        })
    for native_index, line in enumerate(native_lines):
        distances = np.asarray([line.distance(other) for other in lia_lines])
        nearest = int(np.argmin(distances))
        rows.append({
            "direction": "native-flowline-to-lia-branch",
            "lia_branch_index": nearest,
            "lia_is_main": nearest == int(network["main_branch_index"]),
            "native_global_index": int(native_index),
            "native_member_rgi_id": flowlines[native_index]._section176_member_id,
            "native_local_index": int(flowlines[native_index]._section176_local_index),
            "distance_m": float(distances[nearest]),
        })
    audit = {
        "lia_branch_count": len(branches),
        "lia_main_branch_index": int(network["main_branch_index"]),
        "native_flowline_count": len(native_lines),
        "bijection_claimed": False,
        "relationship_contract": "spatial-many-to-one-and-one-to-many-permitted",
    }
    return pd.DataFrame(rows), audit


def _rebuild_mapped_state(
    *,
    baseline: list,
    native_coordinates: list[np.ndarray],
    source: dict,
    registered_length_m: float,
) -> tuple[list, pd.DataFrame, dict]:
    from oggm.core.flowline import MixedBedFlowline

    root_indices = [i for i, fl in enumerate(baseline) if fl.flows_to is None]
    if len(root_indices) != 1:
        raise RuntimeError(f"mapped-input-root-count:{len(root_indices)}")
    root_index = root_indices[0]
    native_root = baseline[root_index]
    root_active_count = int(round(registered_length_m / native_root.dx_meter))
    if root_active_count < 1 or root_active_count > native_root.nx:
        raise RuntimeError("registered-length-outside-native-root-downstream-domain")
    root_dx_meter = registered_length_m / root_active_count

    eligible_rows = []
    eligible_xy = []
    for global_index, (flowline, xy) in enumerate(zip(baseline, native_coordinates)):
        active_count = (
            root_active_count
            if global_index == root_index
            else int(flowline._section176_native_nx)
        )
        if active_count > flowline.nx:
            raise RuntimeError(f"native-prefix-exceeds-flowline:{global_index}")
        for node_index in range(active_count):
            eligible_rows.append((global_index, node_index))
            eligible_xy.append(xy[node_index])
    eligible_xy = np.asarray(eligible_xy, dtype=float)
    tree = cKDTree(eligible_xy)
    distances, assignment = tree.query(source["cell_xy"], k=1)
    if len(assignment) != len(source["rows"]):
        raise RuntimeError("cell-assignment-count-mismatch")
    counts = np.bincount(assignment, minlength=len(eligible_rows))
    if int(counts.sum()) != len(source["rows"]):
        raise RuntimeError("cell-assignment-omission-or-duplication")

    area = counts.astype(float) * source["cell_area_m2"]
    volume = np.bincount(
        assignment,
        weights=source["cell_thickness"] * source["cell_area_m2"],
        minlength=len(eligible_rows),
    )
    surface_moment = np.bincount(
        assignment,
        weights=source["cell_surface"] * source["cell_area_m2"],
        minlength=len(eligible_rows),
    )
    lookup = {key: index for index, key in enumerate(eligible_rows)}
    new_flowlines = []
    node_rows = []
    local_rectangular_fallbacks = 0
    for global_index, original in enumerate(baseline):
        dx_meter = root_dx_meter if global_index == root_index else original.dx_meter
        dx = dx_meter / original.map_dx
        section = np.zeros(original.nx, dtype=float)
        bed_h = np.asarray(original.bed_h, dtype=float).copy()
        surface_h = bed_h.copy()
        bed_shape = np.asarray(original.bed_shape, dtype=float).copy()
        lambdas = np.asarray(original.lambdas, dtype=float).copy()
        is_trapezoid = np.asarray(original.is_trapezoid, dtype=bool).copy()
        widths_m = np.asarray(original.widths_m, dtype=float).copy()
        for node_index in range(original.nx):
            local = lookup.get((global_index, node_index))
            assigned_area = float(area[local]) if local is not None else 0.0
            assigned_volume = float(volume[local]) if local is not None else 0.0
            mapped = assigned_area > 0
            fallback = False
            if mapped:
                width = assigned_area / dx_meter
                sec = assigned_volume / dx_meter
                source_surface = float(surface_moment[local] / assigned_area)
                if is_trapezoid[node_index]:
                    lam = float(lambdas[node_index])
                    discriminant = width * width - 2.0 * lam * sec
                    if lam > 0 and discriminant >= 0:
                        thick = (width - math.sqrt(discriminant)) / lam
                        w0 = width - lam * thick
                        legal = np.isfinite(thick) and thick > 0 and w0 > 0
                    elif lam == 0:
                        thick = sec / width
                        w0 = width
                        legal = np.isfinite(thick) and thick > 0 and w0 > 0
                    else:
                        legal = False
                    if not legal:
                        lam = 0.0
                        thick = sec / width
                        w0 = width
                        lambdas[node_index] = 0.0
                        fallback = True
                        local_rectangular_fallbacks += 1
                    widths_m[node_index] = width
                    bed_shape[node_index] = np.nan
                else:
                    thick = 1.5 * assigned_volume / assigned_area
                    bed_shape[node_index] = 4.0 * thick / (width * width)
                    widths_m[node_index] = width
                    w0 = np.nan
                    lam = np.nan
                section[node_index] = sec
                bed_h[node_index] = source_surface - thick
                surface_h[node_index] = source_surface
            else:
                thick = 0.0
                width = float(widths_m[node_index])
                w0 = np.nan
                lam = float(lambdas[node_index]) if is_trapezoid[node_index] else np.nan
            node_rows.append({
                "global_flowline_index": int(global_index),
                "member_rgi_id": original._section176_member_id,
                "local_flowline_index": int(original._section176_local_index),
                "node_index": int(node_index),
                "eligible_control_volume": local is not None,
                "mapping_active": bool(mapped),
                "assigned_cell_count": int(counts[local]) if local is not None else 0,
                "assigned_area_m2": assigned_area,
                "assigned_volume_m3": assigned_volume,
                "node_dx_m": float(dx_meter),
                "surface_h_m": float(surface_h[node_index]),
                "bed_h_m": float(bed_h[node_index]),
                "thickness_m": float(thick),
                "section_m2": float(section[node_index]),
                "surface_width_m": float(width),
                "is_trapezoid": bool(is_trapezoid[node_index]),
                "lambda": float(lam) if np.isfinite(lam) else np.nan,
                "bottom_width_m": float(w0) if np.isfinite(w0) else np.nan,
                "rectangular_fallback": bool(fallback),
                "is_system_root": global_index == root_index,
                "is_safety_tail": global_index == root_index and node_index >= root_active_count,
            })
        new = MixedBedFlowline(
            line=copy.deepcopy(original.line),
            dx=float(dx),
            map_dx=float(original.map_dx),
            surface_h=surface_h,
            bed_h=bed_h,
            section=section,
            bed_shape=bed_shape,
            is_trapezoid=is_trapezoid,
            lambdas=lambdas,
            widths_m=widths_m,
            rgi_id=original.rgi_id,
            gdir=None,
        )
        new._section176_member_id = original._section176_member_id
        new._section176_local_index = int(original._section176_local_index)
        new._section176_native_nx = int(original._section176_native_nx)
        new_flowlines.append(new)

    index_by_identity = {id(fl): i for i, fl in enumerate(baseline)}
    for old, new in zip(baseline, new_flowlines):
        if old.flows_to is not None:
            new.set_flows_to(new_flowlines[index_by_identity[id(old.flows_to)]])

    audit = {
        "source_cell_count": int(len(source["rows"])),
        "assigned_cell_count": int(counts.sum()),
        "unassigned_cell_count": int(len(source["rows"]) - counts.sum()),
        "duplicate_assignment_count": 0,
        "eligible_control_volume_count": int(len(eligible_rows)),
        "mapped_control_volume_count": int(np.count_nonzero(counts)),
        "maximum_cell_to_control_distance_m": float(np.max(distances)),
        "area_weighted_mean_cell_to_control_distance_m": float(np.mean(distances)),
        "system_root_global_index": int(root_index),
        "root_active_node_count": int(root_active_count),
        "root_dx_m": float(root_dx_meter),
        "registered_length_m": float(registered_length_m),
        "represented_root_length_m": float(root_active_count * root_dx_meter),
        "local_rectangular_fallback_count": int(local_rectangular_fallbacks),
    }
    return new_flowlines, pd.DataFrame(node_rows), audit


def _static_audit(flowlines: list, nodes: pd.DataFrame, source: dict, build: dict, gates: dict):
    active = nodes["mapping_active"].astype(bool).to_numpy()
    active_nodes = nodes.loc[active].copy()
    model_area = float(np.sum(active_nodes["surface_width_m"] * active_nodes["node_dx_m"]))
    model_volume = float(np.sum(nodes["section_m2"] * nodes["node_dx_m"]))
    area_error = (model_area - source["source_area_m2"]) / source["source_area_m2"]
    volume_error = (model_volume - source["source_volume_m3"]) / source["source_volume_m3"]
    hypsometry = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=source["cell_surface"],
        source_area_m2=np.full(len(source["cell_surface"]), source["cell_area_m2"]),
        node_surface_m=active_nodes["surface_h_m"].to_numpy(dtype=float),
        model_area_m2=active_nodes["assigned_area_m2"].to_numpy(dtype=float),
        bin_size_m=100.0,
    )
    reverse_rows = []
    for global_index, branch in nodes.groupby("global_flowline_index", sort=True):
        branch = branch.sort_values("node_index")
        for (_, up), (_, down) in zip(branch.iloc[:-1].iterrows(), branch.iloc[1:].iterrows()):
            if bool(up.mapping_active) and bool(down.mapping_active):
                step = float(down.surface_h_m - up.surface_h_m)
                reverse_rows.append({
                    "global_flowline_index": int(global_index),
                    "upstream_node_index": int(up.node_index),
                    "downstream_node_index": int(down.node_index),
                    "surface_step_m": step,
                    "reverse": step > float(gates["reverse_slope_tolerance_m"]),
                })
    junction_rows = []
    for index, flowline in enumerate(flowlines):
        if flowline.flows_to is None:
            continue
        target_index = flowlines.index(flowline.flows_to)
        receiver_index = int(flowline.flows_to_indice)
        terminal = float(flowline.surface_h[-1])
        receiver = float(flowline.flows_to.surface_h[receiver_index])
        drop = terminal - receiver
        junction_rows.append({
            "tributary_global_index": int(index),
            "receiver_global_index": int(target_index),
            "receiver_node_index": receiver_index,
            "tributary_terminal_surface_m": terminal,
            "receiver_surface_m": receiver,
            "surface_drop_m": drop,
            "reverse": drop < -float(gates["reverse_slope_tolerance_m"]),
        })
    reverse_frame = pd.DataFrame(reverse_rows)
    junction_frame = pd.DataFrame(junction_rows)
    active_widths = active_nodes["surface_width_m"].to_numpy(dtype=float)
    active_sections = active_nodes["section_m2"].to_numpy(dtype=float)
    finite_legal = bool(
        np.isfinite(active_widths).all()
        and np.isfinite(active_sections).all()
        and np.all(active_widths > 0)
        and np.all(active_sections > 0)
        and np.isfinite(nodes["thickness_m"]).all()
        and np.isfinite(nodes["section_m2"]).all()
        and np.all(nodes["thickness_m"] >= 0)
        and np.all(nodes["section_m2"] >= 0)
    )
    trap = active_nodes["is_trapezoid"].astype(bool)
    legal_trap = bool(
        np.isfinite(active_nodes.loc[trap, "lambda"]).all()
        and np.all(active_nodes.loc[trap, "lambda"] >= 0)
        and np.isfinite(active_nodes.loc[trap, "bottom_width_m"]).all()
        and np.all(active_nodes.loc[trap, "bottom_width_m"] > 0)
    )
    tail = nodes["is_safety_tail"].astype(bool)
    maximum_tail_thickness = float(nodes.loc[tail, "thickness_m"].max()) if tail.any() else 0.0
    branch_reverse_count = int(reverse_frame["reverse"].sum()) if not reverse_frame.empty else 0
    junction_reverse_count = int(junction_frame["reverse"].sum()) if not junction_frame.empty else 0
    length_error = (
        build["represented_root_length_m"] - build["registered_length_m"]
    ) / build["registered_length_m"]
    flags = []
    if build["unassigned_cell_count"] or build["duplicate_assignment_count"]:
        flags.append("cell-assignment-not-conservative")
    if abs(area_error) > float(gates["area_relative_error_max"]):
        flags.append("area-error-over-gate")
    if abs(volume_error) > float(gates["volume_relative_error_max"]):
        flags.append("volume-error-over-gate")
    if hypsometry > float(gates["hypsometry_100m_cumulative_error_max"]):
        flags.append("hypsometry-error-over-gate")
    if abs(length_error) > float(gates["length_relative_error_max"]):
        flags.append("registered-length-error-over-gate")
    if branch_reverse_count:
        flags.append("within-branch-reverse-slope")
    if junction_reverse_count:
        flags.append("junction-reverse-slope")
    if not finite_legal or not legal_trap:
        flags.append("illegal-mixedbed-state")
    if maximum_tail_thickness > float(gates["terminal_tail_ice_tolerance_m"]):
        flags.append("nonzero-safety-tail")
    summary = {
        "status": "static-pass" if not flags else "static-fail",
        "review_flags": flags,
        "source_area_m2": source["source_area_m2"],
        "mapped_area_m2": model_area,
        "area_error_fraction": float(area_error),
        "source_volume_m3": source["source_volume_m3"],
        "mapped_volume_m3": model_volume,
        "volume_error_fraction": float(volume_error),
        "hypsometry_100m_maximum_cumulative_error_fraction": float(hypsometry),
        "registered_length_error_fraction": float(length_error),
        "within_branch_reverse_slope_count": branch_reverse_count,
        "junction_reverse_slope_count": junction_reverse_count,
        "mixedbed_active_state_legal": finite_legal and legal_trap,
        "maximum_safety_tail_thickness_m": maximum_tail_thickness,
        "local_rectangular_fallback_count": build["local_rectangular_fallback_count"],
    }
    return summary, reverse_frame, junction_frame


def _zero_mb_case(flowlines: list, settings: dict, label: str, output_root: Path) -> dict:
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    model = FluxBasedModel(
        copy.deepcopy(flowlines),
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(settings["glen_a_pa_3_s_1"]),
        fs=float(settings["fs"]),
        smooth_trib_influx=bool(settings["smooth_tributary_influx"]),
        check_for_boundaries=bool(settings["check_for_boundaries"]),
    )
    initial_volume = float(model.volume_m3)
    all_rows = []
    checkpoints = []
    previous = 0.0
    for checkpoint in settings["checkpoints_years"]:
        rows, totals = run_zero_mb_budget(model, float(checkpoint) - previous)
        all_rows.extend(budget_rows_as_dicts(rows))
        roots = [fl for fl in model.fls if fl.flows_to is None]
        terminal = max((float(fl.thick[-1]) for fl in roots), default=np.nan)
        checkpoints.append({
            "year": float(checkpoint),
            "volume_m3": float(model.volume_m3),
            "volume_drift_fraction": float((model.volume_m3 - initial_volume) / initial_volume),
            "area_m2": float(model.area_m2),
            "terminal_thickness_m": terminal,
            "interval_budget": totals,
        })
        previous = float(checkpoint)
    pd.DataFrame(all_rows).to_csv(output_root / f"zero_mb_{label}_steps.csv", index=False)
    creation = float(sum(
        row["negative_tributary_creation_m3"] + row["negative_section_clip_creation_m3"]
        for row in all_rows
    ))
    maximum_drift = max(abs(row["volume_drift_fraction"]) for row in checkpoints)
    maximum_terminal = max(row["terminal_thickness_m"] for row in checkpoints)
    flags = []
    if maximum_drift > float(settings["volume_drift_fraction_max"]):
        flags.append("zero-mb-volume-drift-over-gate")
    if creation > float(settings["material_creation_tolerance_m3"]):
        flags.append("material-negative-section-or-tributary-creation")
    if maximum_terminal > float(settings["terminal_tail_thickness_max_m"]):
        flags.append("terminal-thickness-over-gate")
    result = {
        "label": label,
        "status": "zero-mb-pass" if not flags else "zero-mb-fail",
        "review_flags": flags,
        "solver": "FluxBasedModel",
        "mass_balance": "ScalarMassBalance(0.0)",
        "initial_volume_m3": initial_volume,
        "maximum_absolute_volume_drift_fraction": float(maximum_drift),
        "negative_section_and_tributary_creation_m3": creation,
        "maximum_terminal_thickness_m": float(maximum_terminal),
        "checkpoints": checkpoints,
    }
    _write_json(output_root / f"zero_mb_{label}_summary.json", result)
    return result


def _write_pickle(path: Path, value) -> None:
    with gzip.open(path, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def run(config_path: Path, project_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["system_id"]) != SUPPORTED_SYSTEM_ID:
        raise ValueError("only-system-3850-is-authorized")
    output_root = _resolve(project_root, config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"refuse-to-overwrite-versioned-output:{output_root}")
    output_root.mkdir(parents=True)
    _write_json(output_root / "config_snapshot.json", config)

    from oggm import cfg
    if not cfg.IS_INITIALIZED:
        cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_multiprocessing"] = False
    cfg.PATHS["working_dir"] = str((output_root / "oggm_work").resolve())

    source_root = _resolve(project_root, config["source_gdir_root"])
    paired_path = _resolve(project_root, config["paired_system_grid"])
    network_path = _resolve(project_root, config["paired_lia_network"])
    members = list(config["member_rgi_ids"])
    source_records = _source_records(
        source_root, members, config["f19_inversion_suffix"]
    )
    source = _load_paired_source(paired_path, float(config["positive_ice_tolerance_m"]))
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "experiment": config["experiment"],
        "python_executable": sys.executable,
        "system_id": SUPPORTED_SYSTEM_ID,
        "paired_system_grid": str(paired_path.resolve()),
        "paired_system_grid_sha256": source["sha256"],
        "paired_lia_network": str(network_path.resolve()),
        "paired_lia_network_sha256": _sha256(network_path),
        "positive_ice_condition": "valid_mask & lia_mask & thickness_m > 1e-6 m",
        "source_area_m2": source["source_area_m2"],
        "source_volume_m3": source["source_volume_m3"],
        "source_gdirs": source_records,
    }
    _write_json(output_root / "provenance.json", provenance)

    initialized, clone_audit = _clone_and_initialize(
        source_root=source_root,
        clone_root=output_root / "isolated_gdirs",
        member_ids=members,
        suffix=config["f19_inversion_suffix"],
    )
    baseline, connections, topology = _combine_native_network(
        initialized,
        main_rgi_id=config["main_rgi_id"],
        member_ids=members,
        source_root=source_root,
    )
    topology.update({
        "member_initialization": clone_audit,
        "cross_member_connections": connections,
    })
    main_grid = topology.pop("main_grid")
    coordinates, coordinate_rows = _flowline_coordinates_in_crs(
        baseline, main_grid, source["crs"]
    )
    crosswalk, crosswalk_audit = _spatial_crosswalk(
        network_path=network_path,
        native_coordinates=coordinates,
        flowlines=baseline,
        target_crs=source["crs"],
    )
    topology.update(crosswalk_audit)
    crosswalk.to_csv(output_root / "spatial_crosswalk.csv", index=False)
    pd.DataFrame(coordinate_rows).to_csv(
        output_root / "native_control_coordinates.csv", index=False
    )
    _write_json(output_root / "topology_audit.json", topology)

    expected = config["static_gates"]
    topology_flags = []
    if topology["member_count"] != int(expected["expected_member_count"]):
        topology_flags.append("member-count-mismatch")
    if topology["flowline_count"] != int(expected["expected_native_flowline_count"]):
        topology_flags.append("native-flowline-count-mismatch")
    if topology["lia_branch_count"] != int(expected["expected_lia_branch_count"]):
        topology_flags.append("lia-branch-count-mismatch")
    if topology["combined_root_count"] != 1:
        topology_flags.append("combined-root-count-mismatch")
    if topology_flags:
        decision = {
            "status": "provenance-or-topology-fail",
            "review_flags": topology_flags,
            "zero_mb_run": False,
            "historical_chain_run": False,
        }
        _write_json(output_root / "decision.json", decision)
        return decision

    mapped, nodes, build = _rebuild_mapped_state(
        baseline=baseline,
        native_coordinates=coordinates,
        source=source,
        registered_length_m=float(config["registered_lia_main_length_m"]),
    )
    nodes.to_csv(output_root / "mapped_nodes.csv", index=False)
    _write_json(output_root / "mapping_audit.json", build)
    static, slopes, junctions = _static_audit(
        mapped, nodes, source, build, expected
    )
    slopes.to_csv(output_root / "within_branch_slopes.csv", index=False)
    junctions.to_csv(output_root / "junction_slopes.csv", index=False)
    _write_json(output_root / "static_summary.json", static)
    _write_pickle(output_root / "native_f19_system_baseline.pkl.gz", baseline)
    _write_pickle(output_root / "mapped_lia_system_state.pkl.gz", mapped)

    source_checks = _verify_sources_unchanged(source_records)
    pd.DataFrame(source_checks).to_csv(output_root / "source_immutability.csv", index=False)
    if not all(row["unchanged"] for row in source_checks):
        static["status"] = "static-fail"
        static["review_flags"].append("source-gdir-mutated")
        _write_json(output_root / "static_summary.json", static)

    if static["status"] != "static-pass":
        decision = {
            "status": "closed-at-static-gate",
            "review_flags": static["review_flags"],
            "zero_mb_run": False,
            "historical_chain_run": False,
            "other_representatives_run": False,
        }
        _write_json(output_root / "decision.json", decision)
        return decision

    zero_results = []
    for label, flowlines in (("native_f19_baseline", baseline), ("mapped_lia", mapped)):
        try:
            zero_results.append(
                _zero_mb_case(flowlines, config["zero_mb"], label, output_root)
            )
        except Exception as exc:
            failure = {
                "label": label,
                "status": "zero-mb-fail",
                "review_flags": [f"{type(exc).__name__}: {exc}"],
                "traceback": traceback.format_exc(),
            }
            _write_json(output_root / f"zero_mb_{label}_summary.json", failure)
            zero_results.append(failure)
            break
    zero_pass = len(zero_results) == 2 and all(
        item["status"] == "zero-mb-pass" for item in zero_results
    )
    decision = {
        "status": "preflight-pass" if zero_pass else "closed-at-zero-mb-gate",
        "static_status": static["status"],
        "zero_mb_results": zero_results,
        "historical_chain_run": False,
        "other_representatives_run": False,
    }
    _write_json(output_root / "decision.json", decision)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(Path(args.config).resolve(), Path(args.project_root).resolve())
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()

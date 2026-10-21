"""Inject all stored branches before OGGM flowline initialization for system 2907."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from pyproj import CRS
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from ..paths import resolve
from . import synthetic_lia_system_geometry_preflight_v1 as geometry_v1
from . import synthetic_lia_system_geometry_preflight_v3 as _geometry_v3  # noqa: F401
from . import synthetic_lia_system_geometry_preflight_v4 as _geometry_v4  # noqa: F401
from . import synthetic_lia_anchor_guided_centerline_pilot_v1 as pilot


pilot.run_native_geometry = geometry_v1.run
_TARGET_BRANCHES: list[dict] | None = None
_MAIN_INDEX: int | None = None
_FILTER_AUDIT: dict[str, object] = {}


def _stored_network_filter(lines, heads, k, r):
    del k, r
    if _TARGET_BRANCHES is None or _MAIN_INDEX is None:
        raise RuntimeError("stored-network-target-not-captured")
    head_xy = np.asarray([head.coords[0] for head in heads], dtype=float)
    start_xy = np.asarray([branch["coords"][0] for branch in _TARGET_BRANCHES], dtype=float)
    cost = np.linalg.norm(start_xy[:, None, :] - head_xy[None, :, :], axis=2)
    branch_rows, head_cols = linear_sum_assignment(cost)
    if len(branch_rows) != len(_TARGET_BRANCHES):
        raise RuntimeError("insufficient-unique-native-heads")
    assigned_head = {int(branch): int(head) for branch, head in zip(branch_rows, head_cols)}
    raw_lines: dict[int, LineString] = {}
    for index, branch in enumerate(_TARGET_BRANCHES):
        coords = np.asarray(branch["coords"], dtype=float).copy()
        coords[0] = head_xy[assigned_head[index]]
        raw_lines[index] = LineString(coords)
    terminal_candidates = np.asarray([line.coords[-1] for line in lines], dtype=float)
    root_coords = np.asarray(raw_lines[_MAIN_INDEX].coords, dtype=float)
    root_coords[-1] = terminal_candidates[int(np.argmin(np.linalg.norm(terminal_candidates - root_coords[-1], axis=1)))]
    raw_lines[_MAIN_INDEX] = LineString(root_coords)
    for index, branch in enumerate(_TARGET_BRANCHES):
        parent = branch["parent_index"]
        if parent is None:
            continue
        coords = np.asarray(raw_lines[index].coords, dtype=float)
        parent_line = raw_lines[int(parent)]
        coords[-1] = np.asarray(parent_line.interpolate(parent_line.project(Point(coords[-1]))).coords[0], dtype=float)
        raw_lines[index] = LineString(coords)
    other = [i for i in range(len(_TARGET_BRANCHES)) if i != _MAIN_INDEX]
    other.sort(key=lambda i: raw_lines[i].length, reverse=True)
    order = [_MAIN_INDEX] + other
    output_lines = [raw_lines[i] for i in order]
    output_heads = [heads[assigned_head[i]] for i in order]
    snap_px = [float(cost[i, assigned_head[i]]) for i in range(len(_TARGET_BRANCHES))]
    _FILTER_AUDIT.clear()
    _FILTER_AUDIT.update({
        "stored_branch_count": len(_TARGET_BRANCHES),
        "unique_native_head_count_used": len(set(assigned_head.values())),
        "maximum_branch_head_snap_px": max(snap_px),
        "p95_branch_head_snap_px": float(np.percentile(snap_px, 95)),
        "output_order": order,
    })
    return output_lines, output_heads


def run(config_path: Path) -> dict:
    global _TARGET_BRANCHES, _MAIN_INDEX
    config = json.loads(config_path.read_text(encoding="utf-8"))
    network_path = resolve(config["registered_network"])
    with gzip.open(network_path, "rb") as stream:
        network = pickle.load(stream)
    network_crs = CRS.from_user_input(network["crs"])
    _MAIN_INDEX = int(network["main_branch_index"])
    target_map = [np.asarray(branch["coords_projected_m"], dtype=float) for branch in network["branches"]]
    import oggm.core.centerlines as centerlines
    native_heads = centerlines._get_centerlines_heads
    native_filter = centerlines._filter_lines
    map_dx: dict[str, float] = {}

    def capture_target(gdir, ext_yx, zoutline, single_fl, glacier_mask, topo, geom, poly_pix):
        global _TARGET_BRANCHES
        branches = []
        for branch, coords_map in zip(network["branches"], target_map):
            i, j = gdir.grid.transform(coords_map[:, 0], coords_map[:, 1], crs=network_crs, nearest=False)
            branches.append({"coords": np.column_stack([i, j]), "parent_index": branch["parent_index"]})
        _TARGET_BRANCHES = branches
        map_dx["value"] = float(abs(gdir.grid.dx))
        return native_heads(gdir, ext_yx, zoutline, single_fl, glacier_mask, topo, geom, poly_pix)

    centerlines._get_centerlines_heads = capture_target
    centerlines._filter_lines = _stored_network_filter
    try:
        base = pilot.run(config_path)
    finally:
        centerlines._get_centerlines_heads = native_heads
        centerlines._filter_lines = native_filter

    output_root = resolve(config["output_root"])
    from oggm import cfg, utils
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=output_root / "synthetic_gdir", reset=False)
    generated_flowlines = gdir.read_pickle("inversion_flowlines")
    roots = [fl for fl in generated_flowlines if fl.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"root-count:{len(roots)}")
    root_xy = np.asarray(roots[0].line.coords, dtype=float)
    x, y = gdir.grid.ij_to_crs(root_xy[:, 0], root_xy[:, 1], crs=network_crs)
    generated = np.column_stack([x, y])
    stored_main = target_map[_MAIN_INDEX]
    distance, _ = cKDTree(stored_main).query(generated, k=1, workers=1)
    p95 = float(np.percentile(distance, 95))
    maximum = float(np.max(distance))
    max_head_snap_m = float(_FILTER_AUDIT["maximum_branch_head_snap_px"]) * map_dx["value"]
    flags = list(base.get("review_flags", []))
    if p95 > float(config["gates"]["root_to_stored_main_p95_max_m"]):
        flags.append("generated-root-stored-main-p95-over-limit")
    if max_head_snap_m > float(config["gates"]["maximum_branch_head_snap_m"]):
        flags.append("stored-branch-head-snap-over-limit")
    if len(generated_flowlines) != len(network["branches"]):
        flags.append("stored-branch-count-not-preserved")
    formal = {
        **base,
        "explicit_stored_full_network_injection": True,
        "stored_branch_count": len(network["branches"]),
        "generated_branch_count": len(generated_flowlines),
        "maximum_branch_head_snap_m": max_head_snap_m,
        "generated_root_to_stored_main_p95_m": p95,
        "generated_root_to_stored_main_max_m": maximum,
        "generated_root_to_stored_main_p95_gate_m": float(config["gates"]["root_to_stored_main_p95_max_m"]),
        "review_flags": flags,
        "status": "geometry-pass" if not flags else "geometry-fail",
        "inversion_run": False, "zero_mb_run": False, "historical_run": False,
    }
    (output_root / "geometry_summary_stored_full_network.json").write_text(json.dumps(formal, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _TARGET_BRANCHES = None
    _MAIN_INDEX = None
    return formal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

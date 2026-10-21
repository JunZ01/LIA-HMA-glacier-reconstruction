"""Read-only system-84 global topological hypsometric assignment audit.

This is distinct from the failed nearest-owner and confluence-only audits.  It
keeps every source cell, every node cell-count quota, and the full 17-branch
topology.  Nodes are placed in a deterministic topological order (priority:
descending nearest-owner mean surface), and source cells are assigned in
descending source-surface order.  The audit measures whether exact global
conservation and all monotonic network inequalities can be obtained, and how
nonlocal that representation would be.  It does not materialize geometry.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_84_RGI60-15.01996_system_grid_v1.nc"
GDIR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
FLOWLINES = GDIR / "inversion_flowlines.pkl"
GRID = GDIR / "glacier_grid.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system84_global_topological_hypsometric_assignment_v1"
TOL = 1e-7


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cdf_error_100m(source_surface, node_surface, node_count) -> float:
    lo = 100.0 * np.floor(min(float(np.min(source_surface)), float(np.min(node_surface))) / 100.0)
    hi = 100.0 * np.ceil(max(float(np.max(source_surface)), float(np.max(node_surface))) / 100.0)
    thresholds = np.arange(lo, hi + 100.0, 100.0)
    source_cdf = np.asarray([np.mean(source_surface <= z) for z in thresholds])
    mapped_cdf = np.asarray([np.sum(node_count[node_surface <= z]) / np.sum(node_count) for z in thresholds])
    return float(np.max(np.abs(source_cdf - mapped_cdf)))


def main() -> None:
    grid = json.loads(GRID.read_text(encoding="utf-8"))
    x0, y0 = map(float, grid["x0y0"])
    dx, dy = map(float, grid["dxdy"])
    with xr.open_dataset(SOURCE) as ds:
        h = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
        s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
        valid = np.asarray(ds.valid_mask.isel(member=0), dtype=bool)
        lia = np.asarray(ds.lia_mask, dtype=bool)
        positive = valid & lia & np.isfinite(h) & np.isfinite(s) & (h > 1e-6)
        yy, xx = np.meshgrid(np.asarray(ds.y, dtype=float), np.asarray(ds.x, dtype=float), indexing="ij")
        cell_xy = np.column_stack([xx[positive], yy[positive]])
        cell_h = h[positive]
        cell_s = s[positive]
        cell_area = float(abs(np.diff(np.asarray(ds.x, dtype=float)).mean() * np.diff(np.asarray(ds.y, dtype=float)).mean()))
    with gzip.open(FLOWLINES, "rb") as stream:
        flowlines = pickle.load(stream)
    branch_by_object = {id(fl): i for i, fl in enumerate(flowlines)}
    node_xy_parts, branch_of, node_of, offsets = [], [], [], []
    cursor = 0
    for branch, fl in enumerate(flowlines):
        offsets.append(cursor)
        coords = np.asarray(fl.line.coords, dtype=float)
        projected = np.column_stack([x0 + coords[:, 0] * dx, y0 + coords[:, 1] * dy])
        node_xy_parts.append(projected)
        branch_of.extend([branch] * fl.nx)
        node_of.extend(range(fl.nx))
        cursor += fl.nx
    node_xy = np.vstack(node_xy_parts)
    branch_of = np.asarray(branch_of, dtype=int)
    node_of = np.asarray(node_of, dtype=int)
    tree = cKDTree(node_xy)
    nearest_distance, nearest_owner = tree.query(cell_xy, k=1, workers=1)
    node_count = np.bincount(nearest_owner, minlength=len(node_xy)).astype(int)
    if np.any(node_count == 0):
        raise RuntimeError(f"unmapped-node-count:{np.count_nonzero(node_count == 0)}")
    nearest_mean = np.bincount(nearest_owner, weights=cell_s, minlength=len(node_xy)) / node_count

    outgoing = [[] for _ in range(len(node_xy))]
    indegree = np.zeros(len(node_xy), dtype=int)
    edges = []
    for branch, fl in enumerate(flowlines):
        off = offsets[branch]
        for node in range(fl.nx - 1):
            edges.append((off + node, off + node + 1, "within"))
        if fl.flows_to is not None:
            receiver = branch_by_object[id(fl.flows_to)]
            edges.append((off + fl.nx - 1, offsets[receiver] + int(fl.flows_to_indice), "junction"))
    for u, v, _ in edges:
        outgoing[u].append(v)
        indegree[v] += 1
    heap = []
    for node in np.flatnonzero(indegree == 0):
        heapq.heappush(heap, (-float(nearest_mean[node]), int(node)))
    order = []
    while heap:
        _, node = heapq.heappop(heap)
        order.append(node)
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(heap, (-float(nearest_mean[target]), int(target)))
    if len(order) != len(node_xy):
        raise RuntimeError("node-network-not-dag")

    sorted_cells = np.argsort(-cell_s, kind="stable")
    new_owner = np.empty(len(cell_s), dtype=int)
    node_surface = np.empty(len(node_xy), dtype=float)
    node_volume = np.zeros(len(node_xy), dtype=float)
    cursor = 0
    for node in order:
        count = int(node_count[node])
        chosen = sorted_cells[cursor:cursor + count]
        if len(chosen) != count:
            raise RuntimeError("cell-quota-overrun")
        new_owner[chosen] = node
        node_surface[node] = float(np.mean(cell_s[chosen]))
        node_volume[node] = float(np.sum(cell_h[chosen]) * cell_area)
        cursor += count
    if cursor != len(cell_s):
        raise RuntimeError("cell-quota-underrun")
    reverse = [(u, v, kind, float(node_surface[v] - node_surface[u])) for u, v, kind in edges if node_surface[v] - node_surface[u] > TOL]
    movement = np.linalg.norm(cell_xy - node_xy[new_owner], axis=1)
    adjustment = node_surface - nearest_mean
    cdf = cdf_error_100m(cell_s, node_surface, node_count)
    source_area = float(len(cell_s) * cell_area)
    mapped_area = float(np.sum(node_count) * cell_area)
    source_volume = float(np.sum(cell_h) * cell_area)
    mapped_volume = float(np.sum(node_volume))
    core_pass = (
        not reverse
        and abs(mapped_area - source_area) <= max(1e-6, source_area * 1e-12)
        and abs(mapped_volume - source_volume) <= max(1e-6, source_volume * 1e-12)
        and cdf <= 0.02
    )
    summary = {
        "schema": "lia-191-system84-global-topological-hypsometric-assignment-v1",
        "system_id": 84,
        "status": "mathematical-feasible-locality-not-adjudicated" if core_pass else "mathematical-fail-mechanism-closed",
        "mechanism": "global-topological-linear-extension-with-fixed-node-cell-count-quotas",
        "positive_cell_count": int(len(cell_s)),
        "node_count": int(len(node_xy)),
        "branch_count": int(len(flowlines)),
        "edge_count": int(len(edges)),
        "within_reverse_count": sum(kind == "within" for _, _, kind, _ in reverse),
        "junction_reverse_count": sum(kind == "junction" for _, _, kind, _ in reverse),
        "maximum_reverse_gap_m": max([gap for *_, gap in reverse], default=0.0),
        "source_area_m2": source_area,
        "mapped_area_m2": mapped_area,
        "area_error_fraction": (mapped_area - source_area) / source_area,
        "source_volume_m3": source_volume,
        "mapped_volume_m3": mapped_volume,
        "volume_error_fraction": (mapped_volume - source_volume) / source_volume,
        "hypsometry_cdf_error_100m": cdf,
        "maximum_absolute_node_surface_adjustment_m": float(np.max(np.abs(adjustment))),
        "mean_absolute_node_surface_adjustment_m": float(np.average(np.abs(adjustment), weights=node_count)),
        "cell_movement_mean_m": float(np.mean(movement)),
        "cell_movement_p95_m": float(np.percentile(movement, 95)),
        "cell_movement_max_m": float(np.max(movement)),
        "nearest_owner_movement_p95_m": float(np.percentile(nearest_distance, 95)),
        "nearest_owner_movement_max_m": float(np.max(nearest_distance)),
        "locality_gate_registered": False,
        "materialized_geometry": False,
        "oggm_run": False,
        "zero_mb_run": False,
        "historical_run": False,
        "source_hashes": {"paired_grid": sha256(SOURCE), "inversion_flowlines": sha256(FLOWLINES), "glacier_grid": sha256(GRID)},
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

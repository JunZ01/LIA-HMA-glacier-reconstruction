"""Read-only branch-preserving hypsometric cell assignment for system 84."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_84_RGI60-15.01996_system_grid_v1.nc"
GDIR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
FLOWLINES = GDIR / "inversion_flowlines.pkl"
GRID = GDIR / "glacier_grid.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system84_branchwise_hypsometric_assignment_v1"


def main() -> None:
    grid = json.loads(GRID.read_text(encoding="utf-8")); x0, y0 = map(float, grid["x0y0"]); dx, dy = map(float, grid["dxdy"])
    with xr.open_dataset(SOURCE) as ds:
        h = np.asarray(ds.thickness_m.isel(member=0), dtype=float); s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
        positive = np.asarray(ds.valid_mask.isel(member=0), dtype=bool) & np.asarray(ds.lia_mask, dtype=bool) & np.isfinite(h) & np.isfinite(s) & (h > 1e-6)
        yy, xx = np.meshgrid(np.asarray(ds.y, dtype=float), np.asarray(ds.x, dtype=float), indexing="ij")
        cell_xy = np.column_stack([xx[positive], yy[positive]]); cell_h = h[positive]; cell_s = s[positive]
        cell_area = float(abs(np.diff(np.asarray(ds.x, dtype=float)).mean() * np.diff(np.asarray(ds.y, dtype=float)).mean()))
    with gzip.open(FLOWLINES, "rb") as stream: flowlines = pickle.load(stream)
    branch_by_object = {id(fl): i for i, fl in enumerate(flowlines)}
    node_xy_parts, branch_of, offsets = [], [], []; cursor = 0
    for branch, fl in enumerate(flowlines):
        offsets.append(cursor); coords = np.asarray(fl.line.coords, dtype=float)
        node_xy_parts.append(np.column_stack([x0 + coords[:, 0] * dx, y0 + coords[:, 1] * dy])); branch_of.extend([branch] * fl.nx); cursor += fl.nx
    node_xy = np.vstack(node_xy_parts); branch_of = np.asarray(branch_of, dtype=int)
    distance, nearest = cKDTree(node_xy).query(cell_xy, k=1, workers=1)
    count = np.bincount(nearest, minlength=len(node_xy)).astype(int)
    if np.any(count == 0): raise RuntimeError("unmapped-node")
    old_mean = np.bincount(nearest, weights=cell_s, minlength=len(node_xy)) / count
    cell_branch = branch_of[nearest]
    owner = np.empty(len(cell_s), dtype=int); node_surface = np.empty(len(node_xy)); node_volume = np.zeros(len(node_xy))
    for branch, fl in enumerate(flowlines):
        cells = np.flatnonzero(cell_branch == branch); cells = cells[np.argsort(-cell_s[cells], kind="stable")]
        cursor = 0
        for local_node in range(fl.nx):
            node = offsets[branch] + local_node; quota = int(count[node]); chosen = cells[cursor:cursor + quota]
            if len(chosen) != quota: raise RuntimeError(f"quota:{branch}:{local_node}")
            owner[chosen] = node; node_surface[node] = float(np.mean(cell_s[chosen])); node_volume[node] = float(np.sum(cell_h[chosen]) * cell_area); cursor += quota
        if cursor != len(cells): raise RuntimeError(f"branch-unused-cells:{branch}")
    within, junction = [], []
    for branch, fl in enumerate(flowlines):
        off = offsets[branch]
        for node in range(fl.nx - 1):
            gap = float(node_surface[off + node + 1] - node_surface[off + node])
            if gap > 1e-7: within.append((branch, node, gap))
        if fl.flows_to is not None:
            receiver = branch_by_object[id(fl.flows_to)]; gap = float(node_surface[offsets[receiver] + int(fl.flows_to_indice)] - node_surface[off + fl.nx - 1])
            if gap > 1e-7: junction.append((branch, receiver, int(fl.flows_to_indice), gap))
    movement = np.linalg.norm(cell_xy - node_xy[owner], axis=1); adjustment = node_surface - old_mean
    source_area = float(len(cell_s) * cell_area); source_volume = float(np.sum(cell_h) * cell_area)
    mapped_volume = float(np.sum(node_volume)); cdf = cdf_error_100m(cell_s, node_surface, count)
    summary = {
        "schema": "lia-191-system84-branchwise-hypsometric-assignment-v1", "system_id": 84,
        "status": "branchwise-pass-junction-repair-required" if not within and junction else ("full-network-static-feasible-not-materialized" if not within else "branchwise-fail"),
        "cell_count": int(len(cell_s)), "node_count": int(len(node_xy)), "branch_count": len(flowlines),
        "within_reverse_count": len(within), "junction_reverse_count": len(junction),
        "maximum_junction_reverse_gap_m": max([x[3] for x in junction], default=0.0),
        "junction_reversals": [{"tributary_branch": a, "receiver_branch": b, "receiver_node": c, "gap_m": d} for a,b,c,d in junction],
        "source_area_m2": source_area, "mapped_area_m2": float(np.sum(count) * cell_area), "area_error_fraction": 0.0,
        "source_volume_m3": source_volume, "mapped_volume_m3": mapped_volume, "volume_error_fraction": (mapped_volume-source_volume)/source_volume,
        "hypsometry_cdf_error_100m": cdf,
        "maximum_absolute_node_surface_adjustment_m": float(np.max(np.abs(adjustment))),
        "mean_absolute_node_surface_adjustment_m": float(np.average(np.abs(adjustment), weights=count)),
        "cell_movement_mean_m": float(np.mean(movement)), "cell_movement_p95_m": float(np.percentile(movement,95)), "cell_movement_max_m": float(np.max(movement)),
        "nearest_branch_identity_preserved": True, "per_node_cell_count_preserved": True, "per_branch_area_volume_preserved": True,
        "materialized_geometry": False, "oggm_run": False, "zero_mb_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False); (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()

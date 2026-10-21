"""Deterministic discrete realization of the system-1931 continuous surface target."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_1931_RGI60-15.09870_system_grid_v1.nc"
GDIR = PROJECT / "w400geom1931v1/synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91931"
TARGET = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system1931_native_surface_lp_bounds_v1/continuous_target_v1.npz"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system1931_continuous_target_discrete_assignment_v1"


def main() -> None:
    grid = json.loads((GDIR / "glacier_grid.json").read_text(encoding="utf-8"))
    x0, y0 = map(float, grid["x0y0"])
    dx, dy = map(float, grid["dxdy"])
    with xr.open_dataset(SOURCE) as ds:
        h = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
        s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
        positive = (
            np.asarray(ds.valid_mask.isel(member=0), dtype=bool)
            & np.asarray(ds.lia_mask, dtype=bool)
            & np.isfinite(h)
            & np.isfinite(s)
            & (h > 1e-6)
        )
        yy, xx = np.meshgrid(np.asarray(ds.y, dtype=float), np.asarray(ds.x, dtype=float), indexing="ij")
        cell_xy = np.column_stack([xx[positive], yy[positive]])
        cell_h = h[positive]
        cell_s = s[positive]
        cell_area = float(abs(np.diff(np.asarray(ds.x, dtype=float)).mean() * np.diff(np.asarray(ds.y, dtype=float)).mean()))
    with gzip.open(GDIR / "inversion_flowlines.pkl", "rb") as stream:
        flowlines = pickle.load(stream)
    branch_by_object = {id(fl): i for i, fl in enumerate(flowlines)}
    offsets, node_xy_parts, native = [], [], []
    cursor = 0
    for fl in flowlines:
        offsets.append(cursor)
        coords = np.asarray(fl.line.coords, dtype=float)
        node_xy_parts.append(np.column_stack([x0 + coords[:, 0] * dx, y0 + coords[:, 1] * dy]))
        native.extend(np.asarray(fl.surface_h, dtype=float))
        cursor += fl.nx
    node_xy = np.vstack(node_xy_parts)
    native = np.asarray(native, dtype=float)
    target_data = np.load(TARGET)
    target = np.asarray(target_data["target_surface_m"], dtype=float)
    quota = np.asarray(target_data["node_quota"], dtype=int)
    if len(target) != len(node_xy) or int(np.sum(quota)) != len(cell_s) or np.any(quota <= 0):
        raise RuntimeError("target-quota-schema")
    node_order = np.argsort(-target, kind="stable")
    cell_order = np.argsort(-cell_s, kind="stable")
    owner = np.empty(len(cell_s), dtype=int)
    node_surface = np.empty(len(node_xy), dtype=float)
    node_volume = np.empty(len(node_xy), dtype=float)
    cursor = 0
    for node in node_order:
        count = int(quota[node])
        chosen = cell_order[cursor:cursor + count]
        if len(chosen) != count:
            raise RuntimeError("cell-exhaustion")
        owner[chosen] = node
        node_surface[node] = float(np.mean(cell_s[chosen]))
        node_volume[node] = float(np.sum(cell_h[chosen]) * cell_area)
        cursor += count
    if cursor != len(cell_s) or len(np.unique(owner)) > len(node_xy):
        raise RuntimeError("assignment-incomplete")
    within, junction = [], []
    for branch, fl in enumerate(flowlines):
        off = offsets[branch]
        for node in range(fl.nx - 1):
            gap = float(node_surface[off + node + 1] - node_surface[off + node])
            if gap > 1e-7:
                within.append((branch, node, gap))
        if fl.flows_to is not None:
            receiver = branch_by_object[id(fl.flows_to)]
            receiver_node = int(fl.flows_to_indice)
            gap = float(node_surface[offsets[receiver] + receiver_node] - node_surface[off + fl.nx - 1])
            if gap > 1e-7:
                junction.append((branch, receiver, receiver_node, gap))
    adjustment = node_surface - native
    movement = np.linalg.norm(cell_xy - node_xy[owner], axis=1)
    source_area = float(len(cell_s) * cell_area)
    source_volume = float(np.sum(cell_h) * cell_area)
    mapped_volume = float(np.sum(node_volume))
    cdf = cdf_error_100m(cell_s, node_surface, quota)
    formal_geometry_gates = bool(
        not within and not junction and cdf <= 0.02
        and float(np.max(np.abs(adjustment))) <= 150.0
        and float(np.average(np.abs(adjustment), weights=quota)) <= 20.0
    )
    summary = {
        "schema": "lia-191-system1931-continuous-target-discrete-assignment-v1",
        "system_id": 1931,
        "status": "discrete-static-gates-pass-not-materialized" if formal_geometry_gates else "discrete-static-gates-fail",
        "cell_count": int(len(cell_s)), "node_count": int(len(node_xy)), "branch_count": len(flowlines),
        "within_reverse_count": len(within), "junction_reverse_count": len(junction),
        "area_error_fraction": 0.0, "volume_error_fraction": (mapped_volume - source_volume) / source_volume,
        "source_area_m2": source_area, "source_volume_m3": source_volume, "mapped_volume_m3": mapped_volume,
        "hypsometry_cdf_error_100m": cdf, "hypsometry_gate": 0.02,
        "maximum_absolute_node_surface_adjustment_m": float(np.max(np.abs(adjustment))), "maximum_adjustment_gate_m": 150.0,
        "mean_absolute_node_surface_adjustment_m": float(np.average(np.abs(adjustment), weights=quota)), "mean_adjustment_gate_m": 20.0,
        "cell_movement_mean_m": float(np.mean(movement)), "cell_movement_p95_m": float(np.percentile(movement, 95)), "cell_movement_max_m": float(np.max(movement)),
        "global_elevation_order_assignment": True,
        "spatial_locality_not_a_formal_pass_claim": True,
        "formal_geometry_gates_pass": formal_geometry_gates,
        "materialized_geometry": False, "oggm_run": False, "zero_mb_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(OUTPUT / "assignment_arrays.npz", owner=owner, node_surface_m=node_surface, node_volume_m3=node_volume, node_quota=quota, movement_m=movement)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

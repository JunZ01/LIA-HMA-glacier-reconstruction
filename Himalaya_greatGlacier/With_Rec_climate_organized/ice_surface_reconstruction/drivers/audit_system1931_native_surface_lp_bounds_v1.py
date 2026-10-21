"""Surface-moment-conserving continuous LP bounds for system 1931."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


PROJECT = Path(__file__).resolve().parents[4]
GDIR = PROJECT / "w400geom1931v1/synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91931"
ARRAYS = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system1931_branchwise_hypsometric_assignment_v1/assignment_audit_arrays.npz"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system1931_native_surface_lp_bounds_v1"


def main() -> None:
    data = np.load(ARRAYS)
    weight = np.asarray(data["node_quota"], dtype=float)
    cell_surface = np.asarray(data["cell_surface_m"], dtype=float)
    with gzip.open(GDIR / "inversion_flowlines.pkl", "rb") as stream:
        flowlines = pickle.load(stream)
    by_id = {id(fl): i for i, fl in enumerate(flowlines)}
    offsets, native = [], []
    cursor = 0
    for fl in flowlines:
        offsets.append(cursor)
        native.extend(np.asarray(fl.surface_h, dtype=float))
        cursor += fl.nx
    native = np.asarray(native, dtype=float)
    edges = []
    for branch, fl in enumerate(flowlines):
        for node in range(fl.nx - 1):
            edges.append((offsets[branch] + node, offsets[branch] + node + 1))
        if fl.flows_to is not None:
            receiver = by_id[id(fl.flows_to)]
            edges.append((offsets[branch] + fl.nx - 1, offsets[receiver] + int(fl.flows_to_indice)))
    n = len(native)
    required_delta_sum = float(np.sum(cell_surface) - np.dot(weight, native))

    rows, cols, vals, rhs = [], [], [], []
    row = 0
    for upstream, downstream in edges:
        rows += [row, row]
        cols += [downstream, upstream]
        vals += [1.0, -1.0]
        rhs.append(float(native[upstream] - native[downstream]))
        row += 1
    for i in range(n):
        rows += [row, row]
        cols += [i, n]
        vals += [1.0, -1.0]
        rhs.append(0.0)
        row += 1
        rows += [row, row]
        cols += [i, n]
        vals += [-1.0, -1.0]
        rhs.append(0.0)
        row += 1
    a_ub = coo_matrix((vals, (rows, cols)), shape=(row, n + 1)).tocsr()
    objective = np.zeros(n + 1)
    objective[n] = 1.0
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = weight
    minimax_result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.asarray(rhs),
        A_eq=a_eq,
        b_eq=np.asarray([required_delta_sum]),
        bounds=[(None, None)] * n + [(0.0, None)],
        method="highs",
    )

    rows, cols, vals, rhs = [], [], [], []
    row = 0
    for upstream, downstream in edges:
        rows += [row, row]
        cols += [downstream, upstream]
        vals += [1.0, -1.0]
        rhs.append(float(native[upstream] - native[downstream]))
        row += 1
    for i in range(n):
        rows += [row, row]
        cols += [i, n + i]
        vals += [1.0, -1.0]
        rhs.append(0.0)
        row += 1
        rows += [row, row]
        cols += [i, n + i]
        vals += [-1.0, -1.0]
        rhs.append(0.0)
        row += 1
    a_ub_l1 = coo_matrix((vals, (rows, cols)), shape=(row, 2 * n)).tocsr()
    objective_l1 = np.r_[np.zeros(n), weight / np.sum(weight)]
    a_eq_l1 = np.zeros((1, 2 * n))
    a_eq_l1[0, :n] = weight
    l1_result = linprog(
        objective_l1,
        A_ub=a_ub_l1,
        b_ub=np.asarray(rhs),
        A_eq=a_eq_l1,
        b_eq=np.asarray([required_delta_sum]),
        bounds=[(-150.0, 150.0)] * n + [(0.0, None)] * n,
        method="highs",
    )
    minimax = None if not minimax_result.success else float(minimax_result.x[n])
    mean_l1 = None if not l1_result.success else float(l1_result.fun)
    target_mean = float(np.mean(cell_surface))
    native_weighted_mean = float(np.dot(weight, native) / np.sum(weight))
    passed = bool(minimax_result.success and minimax <= 150.0 and l1_result.success and mean_l1 <= 20.0)
    summary = {
        "schema": "lia-191-system1931-native-surface-lp-bounds-v1",
        "system_id": 1931,
        "status": "surface-moment-conserving-continuous-bounds-pass" if passed else "surface-moment-conserving-continuous-bound-fail-mechanism-closed",
        "node_count": n, "edge_count": len(edges),
        "source_cell_mean_surface_m": target_mean,
        "native_quota_weighted_mean_surface_m": native_weighted_mean,
        "required_quota_weighted_mean_adjustment_m": target_mean - native_weighted_mean,
        "surface_moment_equality_enforced": True,
        "minimax_lp_success": bool(minimax_result.success),
        "minimum_possible_maximum_adjustment_m": minimax,
        "maximum_adjustment_gate_m": 150.0,
        "mean_l1_lp_success_under_max150": bool(l1_result.success),
        "minimum_possible_area_weighted_mean_absolute_adjustment_m": mean_l1,
        "mean_adjustment_gate_m": 20.0,
        "continuous_bounds_pass": passed,
        "discrete_assignment_authorized": passed,
        "geometry_materialized": False, "oggm_run": False, "zero_mb_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if l1_result.success:
        np.savez_compressed(OUTPUT / "continuous_target_v1.npz", delta_m=l1_result.x[:n], target_surface_m=native + l1_result.x[:n], node_quota=weight)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

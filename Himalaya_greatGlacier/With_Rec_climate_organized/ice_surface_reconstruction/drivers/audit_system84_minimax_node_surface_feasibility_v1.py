"""Continuous minimax lower bound for monotone system-84 node surfaces."""

from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


PROJECT = Path(__file__).resolve().parents[4]
OWNER = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_confluence_exact_cell_reassignment_v1/system_84/deterministic_owner_audit_v1/node_owner_summary.csv"
FLOWLINES = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084/inversion_flowlines.pkl"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system84_minimax_node_surface_feasibility_v1"


def main() -> None:
    rows = list(csv.DictReader(OWNER.open("r", encoding="utf-8-sig")))
    surface = np.asarray([float(row["surface_mean_m"]) for row in rows])
    weight = np.asarray([float(row["cell_count"]) for row in rows])
    with gzip.open(FLOWLINES, "rb") as stream:
        flowlines = pickle.load(stream)
    offsets, cursor = [], 0
    for fl in flowlines:
        offsets.append(cursor); cursor += fl.nx
    branch_by_object = {id(fl): i for i, fl in enumerate(flowlines)}
    edges = []
    for branch, fl in enumerate(flowlines):
        for node in range(fl.nx - 1):
            edges.append((offsets[branch] + node, offsets[branch] + node + 1, "within"))
        if fl.flows_to is not None:
            receiver = branch_by_object[id(fl.flows_to)]
            edges.append((offsets[branch] + fl.nx - 1, offsets[receiver] + int(fl.flows_to_indice), "junction"))
    n = len(surface)
    rr, cc, dd, upper = [], [], [], []
    row_index = 0
    for u, v, _ in edges:
        # (s_v + d_v) - (s_u + d_u) <= 0
        rr.extend([row_index, row_index]); cc.extend([v, u]); dd.extend([1.0, -1.0]); upper.append(float(surface[u] - surface[v])); row_index += 1
    for i in range(n):
        # d_i - z <= 0 and -d_i - z <= 0
        rr.extend([row_index, row_index]); cc.extend([i, n]); dd.extend([1.0, -1.0]); upper.append(0.0); row_index += 1
        rr.extend([row_index, row_index]); cc.extend([i, n]); dd.extend([-1.0, -1.0]); upper.append(0.0); row_index += 1
    aub = coo_matrix((dd, (rr, cc)), shape=(row_index, n + 1)).tocsr()
    aeq = np.zeros((1, n + 1)); aeq[0, :n] = weight / np.sum(weight)
    objective = np.zeros(n + 1); objective[n] = 1.0
    result = linprog(objective, A_ub=aub, b_ub=np.asarray(upper), A_eq=aeq, b_eq=np.zeros(1), bounds=[(None, None)] * n + [(0.0, None)], method="highs")
    if not result.success:
        raise RuntimeError(f"linprog:{result.status}:{result.message}")
    delta = result.x[:n]
    adjusted = surface + delta
    reverse = [(u, v, kind, float(adjusted[v] - adjusted[u])) for u, v, kind in edges if adjusted[v] - adjusted[u] > 1e-7]
    summary = {
        "schema": "lia-191-system84-minimax-node-surface-feasibility-v1",
        "system_id": 84,
        "status": "continuous-minimax-pass-discrete-not-authorized" if result.x[n] <= 150.0 and not reverse else "continuous-lower-bound-fail-mechanism-closed",
        "node_count": n,
        "edge_count": len(edges),
        "minimum_possible_maximum_absolute_adjustment_m": float(result.x[n]),
        "registered_maximum_adjustment_gate_m": 150.0,
        "area_weighted_mean_adjustment_m": float(np.average(delta, weights=weight)),
        "actual_maximum_absolute_adjustment_m": float(np.max(np.abs(delta))),
        "within_reverse_count": sum(kind == "within" for _, _, kind, _ in reverse),
        "junction_reverse_count": sum(kind == "junction" for _, _, kind, _ in reverse),
        "continuous_only": True,
        "integer_cell_assignment_run": False,
        "geometry_materialized": False,
        "oggm_run": False,
        "zero_mb_run": False,
        "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

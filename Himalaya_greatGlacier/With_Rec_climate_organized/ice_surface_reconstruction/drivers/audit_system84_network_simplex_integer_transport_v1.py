"""Exact integer min-cost transport realization for the system-84 continuous target."""

from __future__ import annotations

import gzip
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m


PROJECT = Path(__file__).resolve().parents[4]
GDIR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
ARRAYS = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system84_local_topological_pool_reassignment_v2/assignment_audit_arrays.npz"
TARGET = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system84_native_surface_lp_bounds_v2/continuous_target_v2.npz"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system84_network_simplex_integer_transport_v1"
ELEVATION_BAND_M = 25.0
K_NEAREST = 16
MAX_DISTANCE_M = 2500.0
ELEVATION_COST_MULTIPLIER = 10.0


def main() -> None:
    data = np.load(ARRAYS)
    current_owner = np.asarray(data["owner"], dtype=int)
    cell_surface = np.asarray(data["cell_surface_m"], dtype=float)
    cell_xy = np.asarray(data["cell_xy_m"], dtype=float)
    node_xy = np.asarray(data["node_xy_m"], dtype=float)
    quota = np.asarray(data["node_quota"], dtype=int)
    native = np.asarray(data["native_surface_m"], dtype=float)
    target = np.asarray(np.load(TARGET)["target_surface_m"], dtype=float)
    n_nodes = len(quota)
    if len(target) != n_nodes or len(native) != n_nodes or int(np.sum(quota)) != len(cell_surface):
        raise RuntimeError("input-schema")
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    band = np.floor(cell_surface / ELEVATION_BAND_M).astype(int)
    for cell, key in enumerate(zip(current_owner, band)):
        grouped[(int(key[0]), int(key[1]))].append(cell)
    groups = [np.asarray(cells, dtype=int) for _, cells in sorted(grouped.items())]
    tree = cKDTree(node_xy)
    graph = nx.DiGraph()
    for node, demand in enumerate(quota):
        graph.add_node(f"n{node}", demand=int(demand))
    candidate_counts = []
    for group_index, cells in enumerate(groups):
        group_name = f"g{group_index}"
        supply = int(len(cells))
        graph.add_node(group_name, demand=-supply)
        centroid = np.mean(cell_xy[cells], axis=0)
        mean_surface = float(np.mean(cell_surface[cells]))
        _, nearest = tree.query(centroid, k=min(K_NEAREST, n_nodes))
        candidates = {int(current_owner[cells[0]])} | set(np.atleast_1d(nearest).astype(int).tolist())
        accepted = []
        for destination in sorted(candidates):
            distances = np.linalg.norm(cell_xy[cells] - node_xy[destination], axis=1)
            if float(np.max(distances)) > MAX_DISTANCE_M:
                continue
            spatial_cost = float(np.mean(distances))
            elevation_cost = ELEVATION_COST_MULTIPLIER * abs(mean_surface - float(target[destination]))
            weight = int(round(spatial_cost + elevation_cost))
            graph.add_edge(group_name, f"n{destination}", capacity=supply, weight=weight)
            accepted.append(destination)
        source = int(current_owner[cells[0]])
        if source not in accepted:
            distances = np.linalg.norm(cell_xy[cells] - node_xy[source], axis=1)
            if float(np.max(distances)) > MAX_DISTANCE_M:
                raise RuntimeError(f"source-locality-fail:{group_index}")
            weight = int(round(float(np.mean(distances)) + ELEVATION_COST_MULTIPLIER * abs(mean_surface - float(target[source]))))
            graph.add_edge(group_name, f"n{source}", capacity=supply, weight=weight)
            accepted.append(source)
        candidate_counts.append(len(accepted))
    started = time.time()
    objective, flow = nx.network_simplex(graph)
    runtime = time.time() - started
    owner = np.full(len(cell_surface), -1, dtype=int)
    for group_index, cells in enumerate(groups):
        outgoing = [(int(name[1:]), int(amount)) for name, amount in flow[f"g{group_index}"].items() if amount > 0]
        if sum(amount for _, amount in outgoing) != len(cells):
            raise RuntimeError(f"group-flow:{group_index}")
        outgoing.sort(key=lambda item: float(target[item[0]]), reverse=True)
        ordered_cells = cells[np.argsort(-cell_surface[cells], kind="stable")]
        cursor = 0
        for destination, amount in outgoing:
            chosen = ordered_cells[cursor:cursor + amount]
            owner[chosen] = destination
            cursor += amount
    if np.any(owner < 0):
        raise RuntimeError("unassigned-cell")
    realized_quota = np.bincount(owner, minlength=n_nodes)
    if not np.array_equal(realized_quota, quota):
        raise RuntimeError("quota-mismatch")
    node_surface = np.bincount(owner, weights=cell_surface, minlength=n_nodes) / quota
    movement = np.linalg.norm(cell_xy - node_xy[owner], axis=1)
    with gzip.open(GDIR / "inversion_flowlines.pkl", "rb") as stream:
        flowlines = pickle.load(stream)
    by_id = {id(fl): i for i, fl in enumerate(flowlines)}
    offsets = []
    cursor = 0
    for fl in flowlines:
        offsets.append(cursor)
        cursor += fl.nx
    within, junction = [], []
    for branch, fl in enumerate(flowlines):
        off = offsets[branch]
        for node in range(fl.nx - 1):
            gap = float(node_surface[off + node + 1] - node_surface[off + node])
            if gap > 1e-7:
                within.append((branch, node, gap))
        if fl.flows_to is not None:
            receiver = by_id[id(fl.flows_to)]
            receiver_node = int(fl.flows_to_indice)
            gap = float(node_surface[offsets[receiver] + receiver_node] - node_surface[off + fl.nx - 1])
            if gap > 1e-7:
                junction.append((branch, receiver, receiver_node, gap))
    adjustment = node_surface - native
    cdf = cdf_error_100m(cell_surface, node_surface, quota)
    maximum_adjustment = float(np.max(np.abs(adjustment)))
    mean_adjustment = float(np.average(np.abs(adjustment), weights=quota))
    passed = bool(
        not within and not junction and maximum_adjustment <= 150.0 and mean_adjustment <= 20.0
        and cdf <= 0.02 and float(np.max(movement)) <= MAX_DISTANCE_M + 1e-7
    )
    summary = {
        "schema": "lia-191-system84-network-simplex-integer-transport-v1",
        "system_id": 84,
        "status": "integer-transport-static-pass-materialization-authorized" if passed else "integer-transport-static-fail-backend-closed",
        "solver": "networkx-network-simplex-exact-integer-flow",
        "group_definition": "current-owner-node plus 25 m source-surface band",
        "group_count": len(groups), "graph_node_count": graph.number_of_nodes(), "graph_edge_count": graph.number_of_edges(),
        "candidate_nearest_nodes": K_NEAREST, "maximum_distance_m": MAX_DISTANCE_M,
        "elevation_cost_multiplier": ELEVATION_COST_MULTIPLIER,
        "minimum_candidate_count": min(candidate_counts), "maximum_candidate_count": max(candidate_counts),
        "integer_objective": int(objective), "solver_runtime_seconds": runtime,
        "cell_count": len(cell_surface), "node_count": n_nodes,
        "quota_exact": True, "unassigned_cell_count": 0,
        "within_reverse_count": len(within), "junction_reverse_count": len(junction),
        "maximum_reverse_gap_m": max([x[-1] for x in within + junction], default=0.0),
        "maximum_absolute_node_surface_adjustment_m": maximum_adjustment, "maximum_adjustment_gate_m": 150.0,
        "area_weighted_mean_absolute_adjustment_m": mean_adjustment, "mean_adjustment_gate_m": 20.0,
        "hypsometry_cdf_error_100m": cdf, "hypsometry_gate": 0.02,
        "movement_mean_m": float(np.mean(movement)), "movement_p95_m": float(np.percentile(movement, 95)), "movement_max_m": float(np.max(movement)),
        "one_shot_no_weight_or_neighborhood_retry": True,
        "geometry_materialized": False, "oggm_run": False, "zero_mb_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(OUTPUT / "integer_assignment_v1.npz", owner=owner, node_surface_m=node_surface, movement_m=movement, quota=quota)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

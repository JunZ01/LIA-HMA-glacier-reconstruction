"""Read-only shared-confluence surface feasibility audit for system 1856."""

from __future__ import annotations

import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m


PROJECT = Path(__file__).resolve().parents[4]
GDIR = PROJECT / "w447geom1856v1/synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91856"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system1856_shared_confluence_surface_v1"
TOLERANCE = 1e-7


def main() -> None:
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
    candidate = native.copy()
    incoming: dict[tuple[int, int], list[int]] = defaultdict(list)
    for branch, fl in enumerate(flowlines):
        if fl.flows_to is not None:
            receiver = by_id[id(fl.flows_to)]
            incoming[(receiver, int(fl.flows_to_indice))].append(branch)
    failing_groups = []
    for (receiver, receiver_node), tributaries in incoming.items():
        receiver_global = offsets[receiver] + receiver_node
        failing = [branch for branch in tributaries if native[receiver_global] - native[offsets[branch] + flowlines[branch].nx - 1] > TOLERANCE]
        if not failing:
            continue
        group_branches = list(tributaries)
        group_nodes = [receiver_global] + [offsets[branch] + flowlines[branch].nx - 1 for branch in group_branches]
        lower = -np.inf
        upper = np.inf
        if receiver_node > 0:
            upper = min(upper, native[receiver_global - 1])
        if receiver_node < flowlines[receiver].nx - 1:
            lower = max(lower, native[receiver_global + 1])
        for branch in group_branches:
            terminal = offsets[branch] + flowlines[branch].nx - 1
            if flowlines[branch].nx > 1:
                upper = min(upper, native[terminal - 1])
        originals = native[group_nodes]
        unconstrained = float(np.median(originals))
        shared = float(np.clip(unconstrained, lower, upper))
        candidate[group_nodes] = shared
        failing_groups.append({
            "receiver_branch": receiver, "receiver_node": receiver_node,
            "tributary_branches": group_branches, "original_failing_branches": failing,
            "lower_bound_m": float(lower), "upper_bound_m": float(upper),
            "unconstrained_median_m": unconstrained, "shared_surface_m": shared,
            "group_node_count": len(group_nodes),
            "maximum_group_adjustment_m": float(np.max(np.abs(shared - originals))),
        })
    within, junction = [], []
    for branch, fl in enumerate(flowlines):
        off = offsets[branch]
        for node in range(fl.nx - 1):
            gap = float(candidate[off + node + 1] - candidate[off + node])
            if gap > TOLERANCE:
                within.append((branch, node, gap))
        if fl.flows_to is not None:
            receiver = by_id[id(fl.flows_to)]
            receiver_node = int(fl.flows_to_indice)
            gap = float(candidate[offsets[receiver] + receiver_node] - candidate[off + fl.nx - 1])
            if gap > TOLERANCE:
                junction.append((branch, receiver, receiver_node, gap))
    adjustment = candidate - native
    node_weight = np.ones(len(native), dtype=float)
    cdf = cdf_error_100m(native, candidate, node_weight)
    maximum = float(np.max(np.abs(adjustment)))
    mean = float(np.average(np.abs(adjustment), weights=node_weight))
    interval_feasible = all(group["lower_bound_m"] <= group["upper_bound_m"] for group in failing_groups)
    passed = bool(interval_feasible and not within and not junction and maximum <= 150.0 and mean <= 20.0 and cdf <= 0.02)
    summary = {
        "schema": "lia-191-system1856-shared-confluence-surface-v1",
        "system_id": 1856,
        "status": "shared-confluence-static-pass-materialization-authorized" if passed else "shared-confluence-static-fail-mechanism-closed",
        "branch_count": len(flowlines), "node_count": len(native),
        "native_failing_junction_count": sum(len(group["original_failing_branches"]) for group in failing_groups),
        "shared_confluence_group_count": len(failing_groups), "groups": failing_groups,
        "all_group_intervals_feasible": interval_feasible,
        "within_reverse_count": len(within), "junction_reverse_count": len(junction),
        "maximum_remaining_reverse_gap_m": max([item[-1] for item in within + junction], default=0.0),
        "maximum_absolute_adjustment_m": maximum, "maximum_adjustment_gate_m": 150.0,
        "node_weighted_mean_absolute_adjustment_m": mean, "mean_adjustment_gate_m": 20.0,
        "hypsometry_cdf_error_100m": cdf, "hypsometry_gate": 0.02,
        "only_failing_confluence_duplicate_nodes_changed": True,
        "outline_dem_dx_topology_unchanged": True,
        "geometry_materialized": False, "inversion_run": False, "zero_mb_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(OUTPUT / "candidate_surface_v1.npz", native_surface_m=native, candidate_surface_m=candidate, adjustment_m=adjustment)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

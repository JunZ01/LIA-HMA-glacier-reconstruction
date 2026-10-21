"""Shared-confluence feasibility on the raster-support outline geometry for system 1014."""

from __future__ import annotations

import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m


PROJECT = Path(__file__).resolve().parents[4]
GDIR = PROJECT / "Himalaya_greatGlacier/With_Rec_climate_organized/w1014rasteroutlinev1/geometry/synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91014"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system1014_raster_shared_confluence_v1"
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
    groups = []
    for (receiver, receiver_node), tributaries in incoming.items():
        receiver_global = offsets[receiver] + receiver_node
        failing = [branch for branch in tributaries if native[receiver_global] - native[offsets[branch] + flowlines[branch].nx - 1] > TOLERANCE]
        if not failing:
            continue
        group_nodes = [receiver_global] + [offsets[branch] + flowlines[branch].nx - 1 for branch in tributaries]
        lower, upper = -np.inf, np.inf
        if receiver_node > 0:
            upper = min(upper, native[receiver_global - 1])
        if receiver_node < flowlines[receiver].nx - 1:
            lower = max(lower, native[receiver_global + 1])
        for branch in tributaries:
            terminal = offsets[branch] + flowlines[branch].nx - 1
            if flowlines[branch].nx > 1:
                upper = min(upper, native[terminal - 1])
        originals = native[group_nodes]
        median = float(np.median(originals))
        shared = float(np.clip(median, lower, upper))
        candidate[group_nodes] = shared
        groups.append({
            "receiver_branch": receiver, "receiver_node": receiver_node,
            "tributary_branches": list(tributaries), "original_failing_branches": failing,
            "lower_bound_m": float(lower), "upper_bound_m": float(upper),
            "interval_feasible": bool(lower <= upper),
            "shared_surface_m": shared,
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
    weight = np.ones(len(native), dtype=float)
    cdf = cdf_error_100m(native, candidate, weight)
    maximum = float(np.max(np.abs(adjustment)))
    mean = float(np.average(np.abs(adjustment), weights=weight))
    intervals = all(group["interval_feasible"] for group in groups)
    passed = bool(intervals and not within and not junction and maximum <= 150.0 and mean <= 20.0 and cdf <= 0.02)
    summary = {
        "schema": "lia-191-system1014-raster-shared-confluence-v1",
        "system_id": 1014,
        "status": "raster-shared-confluence-static-pass-materialization-authorized" if passed else "raster-shared-confluence-static-fail-mechanism-closed",
        "branch_count": len(flowlines), "node_count": len(native),
        "native_failing_junction_count": sum(len(group["original_failing_branches"]) for group in groups),
        "shared_group_count": len(groups), "groups": groups,
        "all_intervals_feasible": intervals,
        "within_reverse_count": len(within), "junction_reverse_count": len(junction),
        "maximum_remaining_reverse_gap_m": max([item[-1] for item in within + junction], default=0.0),
        "maximum_absolute_adjustment_m": maximum, "maximum_adjustment_gate_m": 150.0,
        "node_weighted_mean_absolute_adjustment_m": mean, "mean_adjustment_gate_m": 20.0,
        "hypsometry_cdf_error_100m": cdf, "hypsometry_gate": 0.02,
        "raster_support_outline_unchanged": True, "topology_dx_dem_unchanged": True,
        "geometry_materialized": False, "inversion_run": False, "zero_mb_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(OUTPUT / "candidate_surface_v1.npz", native_surface_m=native, candidate_surface_m=candidate, adjustment_m=adjustment)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

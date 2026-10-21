"""Lexicographic weighted-L2 refinement of the system-1499 datum LP.

V1 proved that the minimum maximum absolute branch offset is feasible but its
arbitrary degenerate LP solution failed the paired-hypsometry gate.  V2 keeps
that exact minimum-max bound and, within it, minimizes the branch-area-weighted
L2 offset.  This is a single deterministic objective refinement; no scientific
constraint, threshold, input or geometry state is changed.
"""

from __future__ import annotations

import gzip
import json
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize

from . import audit_invalid_outline1499_branch_rigid_vertical_datum_v1 as v1


OUTPUT = v1.PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline1499_branch_rigid_vertical_datum_v2"
V1_SUMMARY = v1.OUTPUT / "summary.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    predecessor = json.loads(V1_SUMMARY.read_text(encoding="utf-8"))
    if predecessor["lp_status"] != 0 or predecessor["maximum_absolute_branch_offset_m"] is None:
        raise RuntimeError("v1-minmax-solution-not-available")
    z_opt = float(predecessor["maximum_absolute_branch_offset_m"])
    offsets_v1 = pd.read_csv(v1.OUTPUT / "branch_offsets.csv")["offset_m"].to_numpy(dtype=float)
    with gzip.open(v1.FLOWLINES, "rb") as stream:
        flowlines = pickle.load(stream)
    branch_count = len(flowlines)
    branch_by_object = {id(flowline): index for index, flowline in enumerate(flowlines)}
    weights = np.asarray([float(np.sum(flowline.widths_m) * flowline.dx_meter) for flowline in flowlines])
    weights_normalized = weights / np.sum(weights)
    junctions = []
    for tributary, flowline in enumerate(flowlines):
        if flowline.flows_to is None:
            continue
        parent = branch_by_object[id(flowline.flows_to)]
        receiver = int(flowline.flows_to_indice)
        gap = float(flowlines[parent].surface_h[receiver] - flowline.surface_h[-1])
        junctions.append((tributary, parent, receiver, gap))

    constraints = [
        {
            "type": "eq",
            "fun": lambda offsets: float(np.dot(weights_normalized, offsets)),
        }
    ]
    for tributary, parent, _, gap in junctions:
        constraints.append({
            "type": "ineq",
            "fun": lambda offsets, t=tributary, p=parent, g=gap: float(offsets[t] - offsets[p] - g),
        })
    objective = lambda offsets: float(0.5 * np.dot(weights_normalized, offsets * offsets))
    result = minimize(
        objective,
        offsets_v1,
        method="SLSQP",
        bounds=[(-z_opt - 1e-8, z_opt + 1e-8)] * branch_count,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
    )
    offsets = np.asarray(result.x, dtype=float)
    candidate_parts = [
        np.asarray(flowline.surface_h, dtype=float) + offsets[index]
        for index, flowline in enumerate(flowlines)
    ]
    candidate_surface = np.concatenate(candidate_parts)
    node_weights = np.concatenate([
        np.asarray(flowline.widths_m, dtype=float) * float(flowline.dx_meter) for flowline in flowlines
    ])
    with xr.open_dataset(v1.PAIRED) as ds:
        thick = np.asarray(ds["thickness_m"].values[0], dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        positive = valid & lia & np.isfinite(thick) & np.isfinite(surface) & (thick > 1e-6)
        source_surface = surface[positive]
    baseline_surface = np.concatenate([np.asarray(flowline.surface_h, dtype=float) for flowline in flowlines])
    baseline_cdf = v1.weighted_cdf_error(source_surface, baseline_surface, node_weights)
    candidate_cdf = v1.weighted_cdf_error(source_surface, candidate_surface, node_weights)
    within_reverse = int(sum(np.count_nonzero(np.diff(part) > v1.TOL) for part in candidate_parts))
    junction_rows = []
    junction_reverse = 0
    maximum_gap = 0.0
    minimum_constraint_margin = np.inf
    for tributary, parent, receiver, original_gap in junctions:
        new_gap = float(candidate_parts[parent][receiver] - candidate_parts[tributary][-1])
        reverse_gap = max(0.0, new_gap)
        margin = float(offsets[tributary] - offsets[parent] - original_gap)
        minimum_constraint_margin = min(minimum_constraint_margin, margin)
        junction_reverse += int(reverse_gap > v1.TOL)
        maximum_gap = max(maximum_gap, reverse_gap)
        junction_rows.append({
            "tributary_branch": tributary,
            "parent_branch": parent,
            "parent_node": receiver,
            "original_reverse_gap_m": max(0.0, original_gap),
            "candidate_reverse_gap_m": reverse_gap,
            "constraint_margin_m": margin,
        })
    weighted_mean = float(np.dot(weights_normalized, offsets))
    maximum_offset = float(np.max(np.abs(offsets)))
    optimization_valid = bool(result.success) and minimum_constraint_margin >= -1e-7 and abs(weighted_mean) <= 1e-8
    gates = {
        "lexicographic_weighted_l2_solve_valid": optimization_valid,
        "minimum_maximum_offset_preserved": maximum_offset <= z_opt + 1e-7,
        "maximum_absolute_offset_within_100m": maximum_offset <= 100.0,
        "area_weighted_mean_offset_closed": abs(weighted_mean) <= 1e-8,
        "within_branch_reverse_zero": within_reverse == 0,
        "junction_reverse_zero": junction_reverse == 0,
        "paired_hypsometry_cdf_within_2pct": candidate_cdf <= 0.02,
        "predecessor_static_length_area_domain_gates_pass": all(
            predecessor["gate_results"][key]
            for key in (
                "predecessor_length_within_10pct",
                "predecessor_area_within_10pct",
                "predecessor_domain_at_least_5km",
            )
        ),
    }
    passed = all(gates.values())
    summary = {
        "schema": "lia-191-invalid-outline1499-branch-rigid-vertical-datum-v2",
        "status": "static-mathematical-feasibility-pass" if passed else "static-mathematical-feasibility-fail",
        "decision": "materialization-authorized-system1499-only" if passed else "branch-rigid-vertical-datum-closed-after-lexicographic-refinement",
        "system_id": v1.SYSTEM_ID,
        "mechanism": "minimum-max-offset-then-area-weighted-L2-branch-rigid-vertical-datum",
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "minimum_maximum_offset_v1_m": z_opt,
        "maximum_absolute_branch_offset_v2_m": maximum_offset,
        "area_weighted_mean_offset_m": weighted_mean,
        "minimum_junction_constraint_margin_m": float(minimum_constraint_margin),
        "baseline_paired_hypsometry_cdf_error_100m": baseline_cdf,
        "candidate_paired_hypsometry_cdf_error_100m": candidate_cdf,
        "candidate_within_reverse_count": within_reverse,
        "candidate_junction_reverse_count": junction_reverse,
        "candidate_maximum_junction_reverse_gap_m": maximum_gap,
        "gate_results": gates,
        "thickness_area_volume_horizontal_topology_changed": False,
        "source_files_modified": False,
        "geometry_materialized": False,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "materialize system1499 only and run unchanged inversion static gates" if passed else "close branch-rigid vertical datum mechanism",
        "provenance": {
            "v1_summary_sha256": v1.sha256(V1_SUMMARY),
            "flowlines_sha256": v1.sha256(v1.FLOWLINES),
            "paired_sha256": v1.sha256(v1.PAIRED),
        },
    }
    OUTPUT.mkdir(parents=True)
    pd.DataFrame({
        "branch_index": np.arange(branch_count),
        "area_weight_m2": weights,
        "offset_v1_m": offsets_v1,
        "offset_v2_m": offsets,
    }).to_csv(OUTPUT / "branch_offsets.csv", index=False)
    pd.DataFrame(junction_rows).to_csv(OUTPUT / "junctions.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

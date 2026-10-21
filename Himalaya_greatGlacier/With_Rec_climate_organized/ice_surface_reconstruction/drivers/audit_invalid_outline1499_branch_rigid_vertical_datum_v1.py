"""Read-only branch-rigid vertical-datum feasibility audit for system 1499.

Each branch receives one constant vertical offset.  The offset is applied to
surface and, after inversion, the corresponding bed datum, so thickness,
section, area, volume, along-branch slopes, horizontal topology and receiver
indices are unchanged.  A deterministic linear program minimizes the maximum
absolute branch offset subject to non-rising junction surfaces and a
flowline-area-weighted zero-mean datum constraint.  No geometry is materialized
and OGGM is not run here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.optimize import linprog


PROJECT = Path(__file__).resolve().parents[4]
SYSTEM_ID = 1499
GEOMETRY = PROJECT / "wcr10invpoly1499v2/system_1499/geometry/geometry_summary.json"
FLOWLINES = PROJECT / "wcr10invpoly1499v2/system_1499/geometry/synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91499/inversion_flowlines.pkl"
PAIRED = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_1499_RGI60-15.10814_system_grid_v1.nc"
PREDECESSOR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline1499_shared_confluence_observed_surface_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline1499_branch_rigid_vertical_datum_v1"
TOL = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_cdf_error(source: np.ndarray, candidate: np.ndarray, weights: np.ndarray) -> float:
    lo = 100.0 * np.floor(min(float(source.min()), float(candidate.min())) / 100.0)
    hi = 100.0 * np.ceil(max(float(source.max()), float(candidate.max())) / 100.0)
    thresholds = np.arange(lo, hi + 100.0, 100.0)
    source_cdf = np.asarray([np.mean(source <= threshold) for threshold in thresholds])
    candidate_cdf = np.asarray([
        np.sum(weights[candidate <= threshold]) / np.sum(weights) for threshold in thresholds
    ])
    return float(np.max(np.abs(source_cdf - candidate_cdf)))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if geometry["within_branch_reverse_slope_count"] != 0 or geometry["junction_reverse_slope_count"] != 2:
        raise RuntimeError("unexpected-system1499-predecessor-geometry")
    if predecessor["status"] != "static-feasibility-fail":
        raise RuntimeError("shared-confluence-predecessor-not-closed")
    with gzip.open(FLOWLINES, "rb") as stream:
        flowlines = pickle.load(stream)
    branch_by_object = {id(flowline): index for index, flowline in enumerate(flowlines)}
    branch_count = len(flowlines)
    branch_weights = np.asarray([
        float(np.sum(flowline.widths_m) * flowline.dx_meter) for flowline in flowlines
    ])
    all_surface = np.concatenate([np.asarray(flowline.surface_h, dtype=float) for flowline in flowlines])
    all_weights = np.concatenate([
        np.asarray(flowline.widths_m, dtype=float) * float(flowline.dx_meter) for flowline in flowlines
    ])

    junctions = []
    for tributary, flowline in enumerate(flowlines):
        if flowline.flows_to is None:
            continue
        parent = branch_by_object[id(flowline.flows_to)]
        receiver = int(flowline.flows_to_indice)
        tributary_surface = float(flowline.surface_h[-1])
        receiver_surface = float(flowlines[parent].surface_h[receiver])
        junctions.append({
            "tributary_branch": tributary,
            "parent_branch": parent,
            "parent_node": receiver,
            "tributary_terminal_surface_m": tributary_surface,
            "receiver_surface_m": receiver_surface,
            "original_reverse_gap_m": max(0.0, receiver_surface - tributary_surface),
        })

    variable_count = branch_count + 1
    z_index = branch_count
    objective = np.zeros(variable_count)
    objective[z_index] = 1.0
    a_ub = []
    b_ub = []
    for branch in range(branch_count):
        row = np.zeros(variable_count)
        row[branch] = 1.0
        row[z_index] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)
        row = np.zeros(variable_count)
        row[branch] = -1.0
        row[z_index] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)
    for junction in junctions:
        row = np.zeros(variable_count)
        row[junction["parent_branch"]] = 1.0
        row[junction["tributary_branch"]] = -1.0
        a_ub.append(row)
        b_ub.append(junction["tributary_terminal_surface_m"] - junction["receiver_surface_m"])
    a_eq = np.zeros((1, variable_count))
    a_eq[0, :branch_count] = branch_weights / np.sum(branch_weights)
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.asarray([0.0]),
        bounds=[(None, None)] * branch_count + [(0.0, None)],
        method="highs",
    )
    feasible = bool(result.success)
    offsets = np.full(branch_count, np.nan)
    maximum_offset = None
    weighted_mean = None
    candidate_cdf = None
    within_reverse = None
    junction_reverse = None
    maximum_junction_gap = None
    candidate_surface = None
    junction_rows = []
    if feasible:
        offsets = np.asarray(result.x[:branch_count], dtype=float)
        maximum_offset = float(np.max(np.abs(offsets)))
        weighted_mean = float(np.sum(branch_weights * offsets) / np.sum(branch_weights))
        candidate_parts = [
            np.asarray(flowline.surface_h, dtype=float) + offsets[index]
            for index, flowline in enumerate(flowlines)
        ]
        candidate_surface = np.concatenate(candidate_parts)
        with xarray_open(PAIRED) as paired_surface:
            candidate_cdf = weighted_cdf_error(paired_surface, candidate_surface, all_weights)
        within_reverse = int(sum(np.count_nonzero(np.diff(part) > TOL) for part in candidate_parts))
        junction_reverse = 0
        maximum_junction_gap = 0.0
        for junction in junctions:
            tributary = junction["tributary_branch"]
            parent = junction["parent_branch"]
            receiver = junction["parent_node"]
            tributary_surface = float(candidate_parts[tributary][-1])
            receiver_surface = float(candidate_parts[parent][receiver])
            gap = max(0.0, receiver_surface - tributary_surface)
            junction_reverse += int(gap > TOL)
            maximum_junction_gap = max(maximum_junction_gap, gap)
            junction_rows.append({
                **junction,
                "tributary_offset_m": float(offsets[tributary]),
                "parent_offset_m": float(offsets[parent]),
                "candidate_reverse_gap_m": gap,
            })

    gates = {
        "linear_program_feasible": feasible,
        "maximum_absolute_offset_within_100m": feasible and maximum_offset <= 100.0,
        "area_weighted_mean_offset_closed": feasible and abs(weighted_mean) <= 1e-8,
        "within_branch_reverse_zero": feasible and within_reverse == 0,
        "junction_reverse_zero": feasible and junction_reverse == 0,
        "paired_hypsometry_cdf_within_2pct": feasible and candidate_cdf <= 0.02,
        "predecessor_length_within_10pct": float(geometry["root_length_relative_error"]) <= 0.10,
        "predecessor_area_within_10pct": float(geometry["outline_area_relative_error"]) <= 0.10,
        "predecessor_domain_at_least_5km": float(geometry["downstream_free_length_m"]) >= 5000.0,
    }
    passed = all(gates.values())
    summary = {
        "schema": "lia-191-invalid-outline1499-branch-rigid-vertical-datum-v1",
        "status": "static-mathematical-feasibility-pass" if passed else "static-mathematical-feasibility-fail",
        "decision": "materialization-authorized-system1499-only" if passed else "branch-rigid-vertical-datum-mechanism-closed-at-mathematical-gate",
        "system_id": SYSTEM_ID,
        "mechanism": "one-constant-surface-and-bed-vertical-offset-per-branch-with-weighted-zero-mean-datum",
        "branch_count": branch_count,
        "junction_count": len(junctions),
        "original_junction_reverse_count": int(sum(row["original_reverse_gap_m"] > TOL for row in junctions)),
        "candidate_within_reverse_count": within_reverse,
        "candidate_junction_reverse_count": junction_reverse,
        "candidate_maximum_junction_reverse_gap_m": maximum_junction_gap,
        "maximum_absolute_branch_offset_m": maximum_offset,
        "area_weighted_mean_offset_m": weighted_mean,
        "paired_hypsometry_cdf_error_100m": candidate_cdf,
        "lp_status": int(result.status),
        "lp_message": str(result.message),
        "gate_results": gates,
        "thickness_changed": False,
        "area_or_volume_changed": False,
        "within_branch_slope_changed": False,
        "horizontal_topology_changed": False,
        "source_files_modified": False,
        "geometry_materialized": False,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "materialize and re-run unchanged inversion for system1499 only" if passed else "close this datum mechanism",
        "provenance": {
            "geometry_sha256": sha256(GEOMETRY),
            "flowlines_sha256": sha256(FLOWLINES),
            "paired_sha256": sha256(PAIRED),
            "predecessor_sha256": sha256(PREDECESSOR),
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "branch_offsets.csv").write_text(
        "branch_index,area_weight_m2,offset_m\n" + "\n".join(
            f"{index},{branch_weights[index]:.17g},{offsets[index]:.17g}" for index in range(branch_count)
        ) + "\n",
        encoding="utf-8",
    )
    import pandas as pd
    pd.DataFrame(junction_rows).to_csv(OUTPUT / "junctions.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def xarray_open(path: Path):
    class _SurfaceContext:
        def __enter__(self):
            self.dataset = xr.open_dataset(path)
            thick = np.asarray(self.dataset["thickness_m"].values[0], dtype=float)
            surface = np.asarray(self.dataset["surface_elevation_m"].values[0], dtype=float)
            valid = np.asarray(self.dataset["valid_mask"].values[0], dtype=bool)
            lia = np.asarray(self.dataset["lia_mask"].values, dtype=bool)
            positive = valid & lia & np.isfinite(thick) & np.isfinite(surface) & (thick > 1e-6)
            return surface[positive]

        def __exit__(self, exc_type, exc, traceback):
            self.dataset.close()
            return False

    return _SurfaceContext()


if __name__ == "__main__":
    main()

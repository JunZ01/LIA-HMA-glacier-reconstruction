"""Read-only branch-rigid vertical-datum feasibility for system 5598."""

from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr

from .audit_branch_rigid_vertical_datum_hold3_v1 import (
    cumulative_hypsometry_error,
    sha256,
    solve_offsets,
)


PROJECT = Path(__file__).resolve().parents[4]
SYSTEM_ID = 5598
RGI_ID = "RGI60-14.26411"
GEOMETRY = PROJECT / "wcr10invpoly5598v1/system_5598/geometry/geometry_summary.json"
GRID = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/"
    "lia_5598_RGI60-14.26411_system_grid_v1.nc"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_invalid_outline5598_branch_rigid_vertical_datum_v1"
)
MAX_OFFSET_M = 100.0
MAX_CDF_ERROR = 0.02


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    if int(geometry["within_branch_reverse_slope_count"]) != 0:
        raise RuntimeError("precondition-fail:within-branch-reversal-present")
    gdir = Path(geometry["synthetic_gdir"])
    flow_path = gdir / "inversion_flowlines.pkl"
    before = {str(flow_path.relative_to(PROJECT)): sha256(flow_path), str(GRID.relative_to(PROJECT)): sha256(GRID)}
    with gzip.open(flow_path, "rb") as stream:
        flowlines = pickle.load(stream)
    index = {id(flowline): i for i, flowline in enumerate(flowlines)}
    surfaces = [np.asarray(flowline.surface_h, dtype=float) for flowline in flowlines]
    weights = [np.asarray(flowline.widths_m, dtype=float) * float(flowline.dx_meter) for flowline in flowlines]
    branch_areas = np.asarray([float(np.sum(item)) for item in weights])
    receivers = []
    for branch, flowline in enumerate(flowlines):
        if flowline.flows_to is not None:
            receivers.append((branch, index[id(flowline.flows_to)], int(flowline.flows_to_indice)))
    offsets, solver = solve_offsets(surfaces, receivers, branch_areas)
    if offsets is None:
        result = {"system_id": SYSTEM_ID, "lp_feasible": False, **solver}
        branch_rows = []
    else:
        shifted = [surface + float(offsets[i]) for i, surface in enumerate(surfaces)]
        remaining = []
        for branch, receiver, receiver_index in receivers:
            gap = float(shifted[receiver][receiver_index] - shifted[branch][-1])
            if gap > 1e-7:
                remaining.append(gap)
        node_values = np.concatenate(surfaces)
        shifted_values = np.concatenate(shifted)
        node_weights = np.concatenate(weights)
        with xr.open_dataset(GRID) as ds:
            h = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
            s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
            positive = (
                np.asarray(ds.valid_mask.isel(member=0), dtype=bool)
                & np.asarray(ds.lia_mask, dtype=bool)
                & np.isfinite(h)
                & np.isfinite(s)
                & (h > 1e-6)
            )
            source_values = s[positive]
        original_cdf = cumulative_hypsometry_error(node_values, node_weights, source_values, np.ones_like(source_values))
        shifted_cdf = cumulative_hypsometry_error(shifted_values, node_weights, source_values, np.ones_like(source_values))
        weighted_mean = float(np.dot(branch_areas, offsets) / np.sum(branch_areas))
        max_offset = float(np.max(np.abs(offsets)))
        after = {str(flow_path.relative_to(PROJECT)): sha256(flow_path), str(GRID.relative_to(PROJECT)): sha256(GRID)}
        feasible = bool(
            max_offset <= MAX_OFFSET_M
            and abs(weighted_mean) <= 1e-8
            and not remaining
            and shifted_cdf <= MAX_CDF_ERROR
            and before == after
        )
        result = {
            "system_id": SYSTEM_ID,
            "lp_feasible": True,
            "branch_count": len(flowlines),
            "junction_count": len(receivers),
            "minimum_max_offset_m": float(solver["phase1_z_m"]),
            "weighted_l1_offset_m": float(solver["phase2_weighted_l1_m"]),
            "maximum_absolute_offset_m": max_offset,
            "area_weighted_mean_offset_m": weighted_mean,
            "remaining_junction_reverse_count": len(remaining),
            "original_hypsometry_cdf_error_100m": original_cdf,
            "shifted_hypsometry_cdf_error_100m": shifted_cdf,
            "source_immutable": before == after,
            "static_feasible": feasible,
        }
        branch_rows = [
            {"branch_index": i, "branch_area_weight_m2": float(branch_areas[i]), "offset_m": float(offsets[i])}
            for i in range(len(flowlines))
        ]
    OUTPUT.mkdir(parents=True)
    if branch_rows:
        with (OUTPUT / "branch_offsets.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(branch_rows[0]))
            writer.writeheader()
            writer.writerows(branch_rows)
    summary = {
        "schema": "lia-191-invalid-outline5598-branch-rigid-vertical-datum-v1",
        "status": "static-feasibility-pass-materialization-not-run" if result.get("static_feasible") else "mechanism-mathematically-closed",
        "system_id": SYSTEM_ID,
        "mechanism": "per-branch rigid surface offset; area-weighted mean-zero; lexicographic min max-absolute then weighted-L1",
        "gates": {"maximum_absolute_offset_m": MAX_OFFSET_M, "hypsometry_cdf_error_100m": MAX_CDF_ERROR, "remaining_junction_reverse_count": 0},
        "result": result,
        "geometry_materialized": False,
        "inversion_run": False,
        "zero_mb_run": False,
        "historical_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

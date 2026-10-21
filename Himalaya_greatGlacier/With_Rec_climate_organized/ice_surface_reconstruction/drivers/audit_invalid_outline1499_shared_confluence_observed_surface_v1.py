"""Read-only shared observed confluence-surface audit for system 1499.

At each reverse junction, the tributary terminal and its receiver control are
treated as two representations of one physical confluence.  Both are assigned
the unchanged paired F19 surface sampled at the receiver coordinate.  No
optimization, smoothing, bed change, or flowline mutation is performed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .build_negative_flux2342_decoupled_branch_elevation_bands_v1 import cdf_error


PROJECT = Path(__file__).resolve().parents[4]
GEOMETRY = PROJECT / "wcr10invpoly1499v2/system_1499/geometry/geometry_summary.json"
PAIRED = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_1499_RGI60-15.10814_system_grid_v1.nc"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline1499_shared_confluence_observed_surface_v1"
TOLERANCE_M = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    if geometry.get("junction_reverse_slope_count") != 2 or geometry.get("within_branch_reverse_slope_count") != 0:
        raise RuntimeError("system1499 geometry failure signature changed")
    gdir = Path(geometry["synthetic_gdir"])
    flowline_path = gdir / "inversion_flowlines.pkl"
    grid_path = gdir / "glacier_grid.json"
    hashes_before = {str(path.relative_to(PROJECT)): sha256(path) for path in (GEOMETRY, PAIRED, flowline_path, grid_path)}
    with gzip.open(flowline_path, "rb") as stream:
        flowlines = pickle.load(stream)
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    with xr.open_dataset(PAIRED) as ds:
        paired_surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        paired_thick = np.asarray(ds["thickness_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        transform = [float(value) for value in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
    supported = valid & lia & np.isfinite(paired_surface) & np.isfinite(paired_thick) & (paired_thick > 1e-6)
    rr, cc = np.nonzero(supported)
    paired_xy = np.column_stack([
        transform[0] + (cc + 0.5) * transform[1] + (rr + 0.5) * transform[2],
        transform[3] + (cc + 0.5) * transform[4] + (rr + 0.5) * transform[5],
    ])
    paired_values = paired_surface[supported]
    paired_tree = cKDTree(paired_xy)
    cell_diag = float(np.hypot(abs(transform[1]), abs(transform[5])))

    candidate = [np.asarray(flowline.surface_h, dtype=float).copy() for flowline in flowlines]
    source = [array.copy() for array in candidate]
    by_identity = {id(flowline): index for index, flowline in enumerate(flowlines)}
    rows = []
    for branch_index, branch in enumerate(flowlines):
        if branch.flows_to is None:
            continue
        parent_index = by_identity[id(branch.flows_to)]
        receiver = int(branch.flows_to_indice)
        gap = float(source[parent_index][receiver] - source[branch_index][-1])
        if gap <= TOLERANCE_M:
            continue
        receiver_grid_xy = np.asarray(branch.flows_to.line.coords[receiver], dtype=float)
        receiver_xy = np.asarray([
            float(grid["x0y0"][0]) + receiver_grid_xy[0] * float(grid["dxdy"][0]),
            float(grid["x0y0"][1]) + receiver_grid_xy[1] * float(grid["dxdy"][1]),
        ])
        distance, nearest = paired_tree.query(receiver_xy, k=1, workers=1)
        observed = float(paired_values[int(nearest)])
        candidate[branch_index][-1] = observed
        candidate[parent_index][receiver] = observed
        rows.append({
            "branch_index": branch_index,
            "parent_branch_index": parent_index,
            "receiver_node": receiver,
            "original_tributary_terminal_surface_m": float(source[branch_index][-1]),
            "original_receiver_surface_m": float(source[parent_index][receiver]),
            "original_reverse_gap_m": gap,
            "paired_confluence_surface_m": observed,
            "paired_sample_distance_m": float(distance),
            "tributary_adjustment_m": observed - float(source[branch_index][-1]),
            "receiver_adjustment_m": observed - float(source[parent_index][receiver]),
        })

    within_reverse = int(sum(np.count_nonzero(np.diff(array) > TOLERANCE_M) for array in candidate))
    junction_reverse = 0
    for branch_index, branch in enumerate(flowlines):
        if branch.flows_to is None:
            continue
        parent_index = by_identity[id(branch.flows_to)]
        receiver = int(branch.flows_to_indice)
        junction_reverse += int(candidate[parent_index][receiver] > candidate[branch_index][-1] + TOLERANCE_M)
    weights = []
    source_values = []
    candidate_values = []
    adjustments = []
    for flowline, old, new in zip(flowlines, source, candidate):
        weight = np.asarray(flowline.widths, dtype=float) * float(flowline.dx_meter)
        weights.extend(weight.tolist())
        source_values.extend(old.tolist())
        candidate_values.extend(new.tolist())
        adjustments.extend(np.abs(new - old).tolist())
    cdf = cdf_error(np.asarray(source_values), np.asarray(weights), np.asarray(candidate_values), np.asarray(weights))
    max_adjustment = float(np.max(adjustments))
    hashes_after = {str(path.relative_to(PROJECT)): sha256(path) for path in (GEOMETRY, PAIRED, flowline_path, grid_path)}
    gates = {
        "exactly_two_reverse_junctions_addressed": len(rows) == 2,
        "paired_sample_distance_within_1p5_cell_diagonals": all(row["paired_sample_distance_m"] <= 1.5 * cell_diag for row in rows),
        "maximum_surface_adjustment_within_150m": max_adjustment <= 150.0,
        "within_branch_reverse_zero": within_reverse == 0,
        "junction_reverse_zero": junction_reverse == 0,
        "hypsometry_cdf_within_2pct": cdf <= 0.02,
        "sources_immutable": hashes_before == hashes_after,
        "inherited_root_length_area_domain_gates_pass": (
            float(geometry["root_length_relative_error"]) <= 0.10
            and float(geometry["outline_area_relative_error"]) <= 0.10
            and float(geometry["downstream_free_length_m"]) >= 5000.0
        ),
    }
    passed = all(gates.values())
    summary = {
        "schema": "lia-191-invalid-outline1499-shared-confluence-observed-surface-v1",
        "status": "static-feasibility-pass-materialization-not-run" if passed else "static-feasibility-fail",
        "system_id": 1499,
        "reverse_junction_count_before": int(geometry["junction_reverse_slope_count"]),
        "within_branch_reverse_count_after": within_reverse,
        "junction_reverse_count_after": junction_reverse,
        "maximum_absolute_surface_adjustment_m": max_adjustment,
        "hypsometry_cdf_error_100m": cdf,
        "gate_results": gates,
        "surface_values_materialized": False,
        "bed_or_thickness_modified": False,
        "optimization_or_smoothing_used": False,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": (
            "materialize only the four shared confluence surface entries and rerun unchanged static geometry gates"
            if passed else
            "close shared paired-confluence surface mechanism for system1499"
        ),
        "provenance_sha256": hashes_before,
    }
    OUTPUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "junction_observed_surface_audit.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

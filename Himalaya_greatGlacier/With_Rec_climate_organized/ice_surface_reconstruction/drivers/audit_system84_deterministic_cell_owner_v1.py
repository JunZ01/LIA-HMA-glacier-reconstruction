"""Deterministic read-only V3-cell to synthetic-flowline owner audit for LIA 84.

This audit deliberately stops before changing geometry.  It reconstructs a
versioned owner map from the frozen paired V3/F19 grid and the frozen synthetic
inversion-flowline network, then asks whether cell-mean surface elevations are
already monotone within every branch and fail only at confluences.  A failure
of that condition closes the proposed confluence-only reassignment mechanism.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_NC = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "core191_anchor_reroute_f19_v3/"
    "lia_84_RGI60-15.01996_system_grid_v1.nc"
)
GDIR = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_geometry_repair_v1/regularization_pilot_84_v1/geometry/"
    "synthetic_gdir/RGI60-15/RGI60-15.90/RGI60-15.90084"
)
FLOWLINES = GDIR / "inversion_flowlines.pkl"
GRID_JSON = GDIR / "glacier_grid.json"
OUTPUT = PROJECT_ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_confluence_exact_cell_reassignment_v1/"
    "system_84/deterministic_owner_audit_v1"
)
POSITIVE_ICE_M = 1e-6
SLOPE_TOL_M = 1e-7
TIE_TOL_M = 1e-9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_flowlines():
    with gzip.open(FLOWLINES, "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    grid = json.loads(GRID_JSON.read_text(encoding="utf-8"))
    x0, y0 = map(float, grid["x0y0"])
    dx_grid, dy_grid = map(float, grid["dxdy"])

    with xr.open_dataset(SOURCE_NC) as ds:
        thickness = np.asarray(ds["thickness_m"].isel(member=0), dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].isel(member=0), dtype=float)
        valid = np.asarray(ds["valid_mask"].isel(member=0), dtype=bool)
        lia = np.asarray(ds["lia_mask"], dtype=bool)
        positive = valid & lia & np.isfinite(thickness) & np.isfinite(surface) & (thickness > POSITIVE_ICE_M)
        yy, xx = np.meshgrid(np.asarray(ds["y"], dtype=float), np.asarray(ds["x"], dtype=float), indexing="ij")
        cell_xy = np.column_stack([xx[positive], yy[positive]])
        cell_h = thickness[positive]
        cell_s = surface[positive]
        source_dx = float(abs(np.diff(np.asarray(ds["x"], dtype=float)).mean()))
        source_dy = float(abs(np.diff(np.asarray(ds["y"], dtype=float)).mean()))

    flowlines = load_flowlines()
    node_xy_parts = []
    branch_index = []
    node_index = []
    branch_offsets = []
    cursor = 0
    for branch, fl in enumerate(flowlines):
        coords = np.asarray(fl.line.coords, dtype=float)
        projected = np.column_stack([x0 + coords[:, 0] * dx_grid, y0 + coords[:, 1] * dy_grid])
        node_xy_parts.append(projected)
        branch_index.extend([branch] * len(projected))
        node_index.extend(range(len(projected)))
        branch_offsets.append(cursor)
        cursor += len(projected)
    node_xy = np.vstack(node_xy_parts)
    branch_index = np.asarray(branch_index, dtype=int)
    node_index = np.asarray(node_index, dtype=int)

    # Query several neighbours and resolve exact/near ties by the lowest global
    # node index.  This makes the reconstructed owner map reproducible even at
    # coincident confluence nodes.
    tree = cKDTree(node_xy)
    distances, candidates = tree.query(cell_xy, k=min(8, len(node_xy)), workers=1)
    if distances.ndim == 1:
        distances = distances[:, None]
        candidates = candidates[:, None]
    minimum = distances[:, 0]
    tied = distances <= (minimum[:, None] + TIE_TOL_M)
    sentinel = np.iinfo(np.int64).max
    owner = np.min(np.where(tied, candidates, sentinel), axis=1).astype(int)

    cell_count = np.bincount(owner, minlength=len(node_xy)).astype(int)
    area_cell = source_dx * source_dy
    node_area = cell_count.astype(float) * area_cell
    node_volume = np.bincount(owner, weights=cell_h * area_cell, minlength=len(node_xy))
    node_surface_moment = np.bincount(owner, weights=cell_s * area_cell, minlength=len(node_xy))
    mapped = cell_count > 0
    node_surface = np.full(len(node_xy), np.nan, dtype=float)
    node_surface[mapped] = node_surface_moment[mapped] / node_area[mapped]

    within = []
    branch_rows = []
    for branch, fl in enumerate(flowlines):
        offset = branch_offsets[branch]
        globals_b = offset + np.arange(len(fl.surface_h), dtype=int)
        mapped_g = globals_b[mapped[globals_b]]
        reversals = 0
        max_gap = 0.0
        for upstream, downstream in zip(mapped_g[:-1], mapped_g[1:]):
            gap = float(node_surface[downstream] - node_surface[upstream])
            if gap > SLOPE_TOL_M:
                reversals += 1
                max_gap = max(max_gap, gap)
                within.append({
                    "branch_index": branch,
                    "upstream_node_index": int(node_index[upstream]),
                    "downstream_node_index": int(node_index[downstream]),
                    "upstream_surface_m": float(node_surface[upstream]),
                    "downstream_surface_m": float(node_surface[downstream]),
                    "reverse_gap_m": gap,
                })
        branch_rows.append({
            "branch_index": branch,
            "node_count": int(len(globals_b)),
            "mapped_node_count": int(len(mapped_g)),
            "assigned_cell_count": int(cell_count[globals_b].sum()),
            "within_reverse_count": reversals,
            "maximum_within_reverse_gap_m": max_gap,
        })

    junctions = []
    for branch, fl in enumerate(flowlines):
        if fl.flows_to is None:
            continue
        receiver = flowlines.index(fl.flows_to)
        tributary_globals = branch_offsets[branch] + np.arange(len(fl.surface_h), dtype=int)
        mapped_tributary = tributary_globals[mapped[tributary_globals]]
        receiver_target = branch_offsets[receiver] + int(fl.flows_to_indice)
        if not len(mapped_tributary):
            status = "tributary-has-no-mapped-node"
            tributary_global = -1
            receiver_global = -1
            gap = float("nan")
        else:
            tributary_global = int(mapped_tributary[-1])
            receiver_globals = branch_offsets[receiver] + np.arange(len(flowlines[receiver].surface_h), dtype=int)
            mapped_receiver = receiver_globals[mapped[receiver_globals]]
            if not len(mapped_receiver):
                status = "receiver-has-no-mapped-node"
                receiver_global = -1
                gap = float("nan")
            else:
                receiver_global = int(mapped_receiver[np.argmin(np.abs(mapped_receiver - receiver_target))])
                gap = float(node_surface[receiver_global] - node_surface[tributary_global])
                status = "reverse" if gap > SLOPE_TOL_M else "pass"
        junctions.append({
            "tributary_branch_index": branch,
            "receiver_branch_index": receiver,
            "registered_receiver_node_index": int(fl.flows_to_indice),
            "used_tributary_node_index": -1 if tributary_global < 0 else int(node_index[tributary_global]),
            "used_receiver_node_index": -1 if receiver_global < 0 else int(node_index[receiver_global]),
            "receiver_minus_tributary_surface_m": gap,
            "status": status,
        })

    within_count = len(within)
    junction_reverse_count = sum(row["status"] == "reverse" for row in junctions)
    unmapped_nodes = int((~mapped).sum())
    maximum_projection = float(np.max(minimum))
    p95_projection = float(np.percentile(minimum, 95))
    disposition = (
        "eligible-for-local-exact-cell-reassignment-milp"
        if within_count == 0 and junction_reverse_count > 0
        else "close-confluence-only-owner-reassignment-not-junction-only"
    )

    summary = {
        "schema": "lia-191-system84-deterministic-cell-owner-audit-v1",
        "system_id": 84,
        "status": "audit-pass" if within_count == 0 else "audit-fail",
        "disposition": disposition,
        "positive_cell_count": int(len(cell_xy)),
        "cell_area_m2": area_cell,
        "source_area_m2": float(len(cell_xy) * area_cell),
        "assigned_area_m2": float(node_area.sum()),
        "source_volume_m3": float(np.sum(cell_h * area_cell)),
        "assigned_volume_m3": float(node_volume.sum()),
        "node_count": int(len(node_xy)),
        "mapped_node_count": int(mapped.sum()),
        "unmapped_node_count": unmapped_nodes,
        "within_branch_reverse_count": within_count,
        "junction_reverse_count": junction_reverse_count,
        "junction_count": len(junctions),
        "cell_to_owner_distance_p95_m": p95_projection,
        "cell_to_owner_distance_max_m": maximum_projection,
        "tie_policy": "nearest-node; distances within 1e-9 m resolve to lowest global node index",
        "source_hashes": {
            "paired_v3_f19_nc": sha256(SOURCE_NC),
            "inversion_flowlines": sha256(FLOWLINES),
            "glacier_grid_json": sha256(GRID_JSON),
        },
        "geometry_changed": False,
        "oggm_run": False,
        "zero_mb_run": False,
        "historical_run": False,
    }

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(OUTPUT / "branch_summary.csv", branch_rows)
    write_csv(OUTPUT / "within_branch_reversals.csv", within)
    write_csv(OUTPUT / "junction_audit.csv", junctions)
    with (OUTPUT / "node_owner_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["global_node_index", "branch_index", "node_index", "x_m", "y_m", "cell_count", "area_m2", "volume_m3", "surface_mean_m"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for g in range(len(node_xy)):
            writer.writerow({
                "global_node_index": g,
                "branch_index": int(branch_index[g]),
                "node_index": int(node_index[g]),
                "x_m": float(node_xy[g, 0]),
                "y_m": float(node_xy[g, 1]),
                "cell_count": int(cell_count[g]),
                "area_m2": float(node_area[g]),
                "volume_m3": float(node_volume[g]),
                "surface_mean_m": "" if not mapped[g] else float(node_surface[g]),
            })
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

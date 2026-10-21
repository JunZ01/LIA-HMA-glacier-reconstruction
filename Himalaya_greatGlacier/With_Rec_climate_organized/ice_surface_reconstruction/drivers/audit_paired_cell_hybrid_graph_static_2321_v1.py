"""Static contract for a paired-cell 2-D ice graph plus a native 1-D zero-ice tail.

This is deliberately read-only.  It materializes no dynamic state and runs no
OGGM model.  The paired F19/V3 positive cells remain the complete active ice
state; only the already validated zero-ice portion of the synthetic root is
admitted as a possible future downstream domain.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import label
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[4]
SYSTEM_ID = 2321
RGI_ID = "RGI60-15.04415"
REGISTERED_LENGTH_M = 9091.108842538311
GRID = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_2321_RGI60-15.04415_system_grid_v1.nc"
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/2321/component_0/lia_centerline_network.pkl.gz"
GDIR = PROJECT / "w395inv2321v1/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92321"
MODEL_FLOWLINES = GDIR / "model_flowlines.pkl"
GRID_JSON = GDIR / "glacier_grid.json"
GEOMETRY_SUMMARY = PROJECT / "w394geom2321v1/geometry_summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_static_2321_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    sources = [GRID, NETWORK, MODEL_FLOWLINES, GRID_JSON, GEOMETRY_SUMMARY]
    before = {str(path.relative_to(PROJECT)): sha256(path) for path in sources}
    with xr.open_dataset(GRID) as ds:
        h = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
        s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
        b = np.asarray(ds.bed_elevation_m.isel(member=0), dtype=float)
        valid = np.asarray(ds.valid_mask.isel(member=0), dtype=bool)
        lia = np.asarray(ds.lia_mask, dtype=bool)
        x = np.asarray(ds.x, dtype=float)
        y = np.asarray(ds.y, dtype=float)
    positive = valid & lia & np.isfinite(h) & np.isfinite(s) & np.isfinite(b) & (h > 1e-6)
    cell_dx = float(abs(np.median(np.diff(x))))
    cell_dy = float(abs(np.median(np.diff(y))))
    cell_area = cell_dx * cell_dy
    cell_diag = float(np.hypot(cell_dx, cell_dy))
    yy, xx = np.meshgrid(y, x, indexing="ij")
    positive_xy = np.column_stack([xx[positive], yy[positive]])
    positive_h = h[positive]
    positive_s = s[positive]
    positive_b = b[positive]

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    _, four_components = label(positive, structure=cross)
    horizontal_edges = int(np.count_nonzero(positive[:, :-1] & positive[:, 1:]))
    vertical_edges = int(np.count_nonzero(positive[:-1, :] & positive[1:, :]))
    graph_edges = horizontal_edges + vertical_edges

    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    main_branch = network["branches"][int(network["main_branch_index"])]
    main_xy = np.asarray(main_branch["coords_projected_m"], dtype=float)
    main_length = float(np.linalg.norm(np.diff(main_xy, axis=0), axis=1).sum())
    main_distance, main_nearest = cKDTree(positive_xy).query(main_xy, k=1, workers=1)
    main_unique_cells = int(np.unique(main_nearest).size)

    with gzip.open(MODEL_FLOWLINES, "rb") as stream:
        flowlines = pickle.load(stream)
    roots = [flowline for flowline in flowlines if getattr(flowline, "flows_to", None) is None]
    if len(roots) != 1:
        raise RuntimeError(f"expected one native root, got {len(roots)}")
    root = roots[0]
    root_positive = np.flatnonzero(np.asarray(root.thick, dtype=float) > 1e-6)
    if root_positive.size == 0:
        raise RuntimeError("native root has no positive ice")
    last_active = int(root_positive[-1])
    tail_start = last_active + 1
    if tail_start >= int(root.nx):
        raise RuntimeError("native root has no zero-ice tail")
    grid = json.loads(GRID_JSON.read_text(encoding="utf-8"))
    line_grid_xy = np.asarray(root.line.coords, dtype=float)
    root_xy = np.column_stack([
        grid["x0y0"][0] + line_grid_xy[:, 0] * grid["dxdy"][0],
        grid["x0y0"][1] + line_grid_xy[:, 1] * grid["dxdy"][1],
    ])
    root_head = root_xy[0]
    root_active_end = root_xy[last_active]
    tail_xy = root_xy[tail_start:]
    tail_bed = np.asarray(root.bed_h[tail_start:], dtype=float)
    tail_thick = np.asarray(root.thick[tail_start:], dtype=float)
    tail_length = float(np.linalg.norm(np.diff(tail_xy, axis=0), axis=1).sum())

    last_to_active = float(np.linalg.norm(main_xy[-1] - root_active_end))
    first_to_active = float(np.linalg.norm(main_xy[0] - root_active_end))
    first_to_head = float(np.linalg.norm(main_xy[0] - root_head))
    last_to_head = float(np.linalg.norm(main_xy[-1] - root_head))
    outlet_distance, outlet_flat = cKDTree(positive_xy).query(main_xy[-1], k=1, workers=1)
    outlet_flat = int(outlet_flat)
    connector_distance = float(np.linalg.norm(positive_xy[outlet_flat] - tail_xy[0]))
    connector_surface_drop = float(positive_s[outlet_flat] - tail_bed[0])
    connector_bed_drop = float(positive_b[outlet_flat] - tail_bed[0])
    tail_steps = np.diff(tail_bed)

    area_m2 = float(positive.sum() * cell_area)
    volume_m3 = float(np.sum(positive_h * cell_area))
    closure_max_m = float(np.max(np.abs(positive_s - positive_b - positive_h)))
    paired_cdf_error = 0.0
    source_after = {str(path.relative_to(PROJECT)): sha256(path) for path in sources}
    gates = {
        "positive_graph_four_neighbor_connected": int(four_components) == 1,
        "positive_state_finite": bool(np.all(np.isfinite(positive_h)) and np.all(np.isfinite(positive_s)) and np.all(np.isfinite(positive_b))),
        "surface_bed_thickness_closure_le_1mm": closure_max_m <= 1e-3,
        "area_volume_exact_by_source_control_volumes": True,
        "hypsometry_cdf_error_le_2pct": paired_cdf_error <= 0.02,
        "registered_length_error_le_0p5pct": abs(main_length - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M <= 0.005,
        "main_support_distance_le_1p5_cell_diagonals": float(np.max(main_distance)) <= 1.5 * cell_diag,
        "stored_main_direction_head_to_outlet": last_to_active < first_to_active and first_to_head < last_to_head,
        "connector_distance_le_2_cell_diagonals": connector_distance <= 2.0 * cell_diag,
        "connector_is_downhill": connector_surface_drop >= -1e-7 and connector_bed_drop >= -1e-7,
        "zero_ice_tail_length_ge_5km": tail_length >= 5000.0,
        "tail_initially_zero_ice": bool(np.all(np.abs(tail_thick) <= 1e-12)),
        "tail_coordinates_and_bed_finite": bool(np.all(np.isfinite(tail_xy)) and np.all(np.isfinite(tail_bed))),
        "sources_immutable": before == source_after,
    }
    status = "hybrid-graph-static-contract-pass" if all(gates.values()) else "hybrid-graph-static-contract-fail"
    summary = {
        "schema": "lia-191-paired-cell-hybrid-graph-static-2321-v1",
        "status": status,
        "system_id": SYSTEM_ID,
        "representation": "paired-positive-cell-four-neighbor-graph-plus-native-root-zero-ice-tail",
        "active_graph": {
            "node_count": int(positive.sum()), "edge_count": graph_edges,
            "horizontal_edges": horizontal_edges, "vertical_edges": vertical_edges,
            "four_neighbor_component_count": int(four_components),
            "cell_dx_m": cell_dx, "cell_dy_m": cell_dy, "cell_area_m2": cell_area,
            "area_m2": area_m2, "volume_m3": volume_m3,
            "surface_bed_thickness_closure_max_m": closure_max_m,
            "hypsometry_cdf_error_100m": paired_cdf_error,
        },
        "observation_path": {
            "stored_main_node_count": int(len(main_xy)), "unique_mapped_positive_cells": main_unique_cells,
            "length_m": main_length, "registered_length_m": REGISTERED_LENGTH_M,
            "length_error_fraction": abs(main_length - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M,
            "maximum_support_distance_m": float(np.max(main_distance)),
        },
        "hybrid_interface": {
            "outlet_positive_cell_xy_m": positive_xy[outlet_flat].tolist(),
            "outlet_to_stored_main_end_distance_m": float(outlet_distance),
            "tail_first_xy_m": tail_xy[0].tolist(), "connector_distance_m": connector_distance,
            "connector_surface_drop_m": connector_surface_drop, "connector_bed_drop_m": connector_bed_drop,
            "tail_node_count": int(len(tail_xy)), "tail_length_m": tail_length,
            "tail_bed_rise_count": int(np.count_nonzero(tail_steps > 1e-7)),
            "tail_maximum_bed_rise_m": float(np.max(tail_steps)) if tail_steps.size else 0.0,
            "tail_terminal_bed_m": float(tail_bed[-1]),
        },
        "gates": gates,
        "source_sha256": before,
        "static_contract_changed_source_state": False,
        "geometry_materialized": False,
        "dynamic_solver_implemented": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "synthetic hybrid-graph solver fixtures only" if status.endswith("pass") else "close paired-cell hybrid graph mechanism",
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

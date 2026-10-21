"""Read-only stored-main skeleton feasibility audit for system 2907."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import label
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[4]
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/2907/component_0/lia_centerline_network.pkl.gz"
PAIRED = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_2907_RGI60-13.26239_system_grid_v1.nc"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2907_stored_main_skeleton_v1"
REGISTERED_LENGTH_M = 20609.06074730425


def main() -> None:
    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    main_index = int(network["main_branch_index"])
    branches = network["branches"]
    main = branches[main_index]
    coords = np.asarray(main["coords_projected_m"], dtype=float)
    reference_surface = np.asarray(main["reference_surface_m"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(reference_surface):
        raise RuntimeError("main-schema")
    segment = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    length = float(np.sum(segment))
    with xr.open_dataset(PAIRED) as ds:
        h = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
        s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
        valid = np.asarray(ds.valid_mask.isel(member=0), dtype=bool)
        lia = np.asarray(ds.lia_mask, dtype=bool)
        positive = valid & lia & np.isfinite(h) & np.isfinite(s) & (h > 1e-6)
        x = np.asarray(ds.x, dtype=float)
        y = np.asarray(ds.y, dtype=float)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        cell_xy = np.column_stack([xx[positive], yy[positive]])
        cell_surface = s[positive]
        cell_diag = float(np.hypot(np.diff(x).mean(), np.diff(y).mean()))
    cell_tree = cKDTree(cell_xy)
    main_to_cell_distance, main_cell = cell_tree.query(coords, k=1, workers=1)
    sampled_surface = cell_surface[main_cell]
    all_branch_xy = np.vstack([np.asarray(branch["coords_projected_m"], dtype=float) for branch in branches])
    positive_to_network_distance, _ = cKDTree(all_branch_xy).query(cell_xy, k=1, workers=1)
    positive_to_main_distance, _ = cKDTree(coords).query(cell_xy, k=1, workers=1)
    ref_reverse = np.diff(reference_surface) > 1e-7
    sampled_reverse = np.diff(sampled_surface) > 1e-7
    _, components = label(positive, structure=np.ones((3, 3), dtype=int))
    length_error = abs(length - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M
    direction_pass = bool(reference_surface[0] > reference_surface[-1])
    pass_static = bool(
        length_error <= 0.005
        and direction_pass
        and np.all(np.isfinite(coords))
        and np.all(np.isfinite(reference_surface))
        and float(np.max(main_to_cell_distance)) <= cell_diag
        and int(np.count_nonzero(ref_reverse)) == 0
        and components == 1
    )
    summary = {
        "schema": "lia-191-system2907-stored-main-skeleton-v1",
        "system_id": 2907,
        "status": "stored-main-skeleton-static-pass-builder-not-yet-materialized" if pass_static else "stored-main-skeleton-static-fail-mechanism-closed",
        "network_version": network.get("network_version"), "network_crs": network.get("crs"),
        "branch_count": len(branches), "main_branch_index": main_index, "main_node_count": len(coords),
        "declared_dx_meter": float(main["dx_meter"]),
        "main_polyline_length_m": length, "registered_length_m": REGISTERED_LENGTH_M,
        "registered_length_error_fraction": length_error, "registered_length_gate": 0.005,
        "main_outlet_station_m": float(network["main_outlet_station_m"]),
        "main_extension_length_m": float(network["main_extension_length_m"]),
        "main_extension_join_gap_m": float(network["main_extension_join_gap_m"]),
        "anchor_rerouted": bool(network["anchor_rerouted"]),
        "head_reference_surface_m": float(reference_surface[0]), "outlet_reference_surface_m": float(reference_surface[-1]),
        "head_to_outlet_direction_pass": direction_pass,
        "reference_surface_reverse_count": int(np.count_nonzero(ref_reverse)),
        "paired_nearest_sample_reverse_count_diagnostic": int(np.count_nonzero(sampled_reverse)),
        "main_node_to_positive_cell_max_distance_m": float(np.max(main_to_cell_distance)),
        "paired_cell_diagonal_m": cell_diag,
        "positive_mask_component_count": int(components),
        "positive_cell_to_full_network_p95_m": float(np.percentile(positive_to_network_distance, 95)),
        "positive_cell_to_full_network_max_m": float(np.max(positive_to_network_distance)),
        "positive_cell_to_main_p95_m": float(np.percentile(positive_to_main_distance, 95)),
        "positive_cell_to_main_max_m": float(np.max(positive_to_main_distance)),
        "source_network_immutable": True,
        "forced_root_builder_authorized": pass_static,
        "geometry_materialized": False, "inversion_run": False, "zero_mb_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Read-only paired-cell graph readiness screen for eight negative-flux hold systems."""

from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import label
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[4]
ROUTE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_12/remaining_137_current_route.csv"
GRID_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3"
NETWORK_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux8_paired_cell_graph_readiness_v1"
SYSTEMS = [890, 988, 1026, 1743, 2321, 2342, 2689, 2793]


def count_internal_false_runs(mask: np.ndarray) -> int:
    true = np.flatnonzero(mask)
    if len(true) == 0:
        return -1
    core = mask[true[0]:true[-1] + 1]
    false = ~core
    return int(np.count_nonzero(false & np.r_[True, ~false[:-1]]))


def main() -> None:
    route = {int(row["lia_unique_id"]): row for row in csv.DictReader(ROUTE.open("r", encoding="utf-8-sig"))}
    rows = []
    for system_id in SYSTEMS:
        row = route[system_id]
        rgi = row["rgi_id"]
        registered = float(row["lia_registered_length_m"])
        grid_path = GRID_ROOT / f"lia_{system_id}_{rgi}_system_grid_v1.nc"
        network_path = NETWORK_ROOT / str(system_id) / "component_0/lia_centerline_network.pkl.gz"
        with gzip.open(network_path, "rb") as stream:
            network = pickle.load(stream)
        main = network["branches"][int(network["main_branch_index"])]
        main_xy = np.asarray(main["coords_projected_m"], dtype=float)
        main_length = float(np.sum(np.linalg.norm(np.diff(main_xy, axis=0), axis=1)))
        with xr.open_dataset(grid_path) as ds:
            h = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
            s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
            positive = np.asarray(ds.valid_mask.isel(member=0), dtype=bool) & np.asarray(ds.lia_mask, dtype=bool) & np.isfinite(h) & np.isfinite(s) & (h > 1e-6)
            x = np.asarray(ds.x, dtype=float)
            y = np.asarray(ds.y, dtype=float)
            yy, xx = np.meshgrid(y, x, indexing="ij")
            positive_xy = np.column_stack([xx[positive], yy[positive]])
            positive_h = h[positive]
            cell_area = float(abs(np.diff(x).mean() * np.diff(y).mean()))
            cell_diag = float(np.hypot(np.diff(x).mean(), np.diff(y).mean()))
        _, component_count = label(positive, structure=np.ones((3, 3), dtype=int))
        distance, nearest = cKDTree(positive_xy).query(main_xy, k=1, workers=1)
        sampled_h = positive_h[nearest]
        thick10 = sampled_h > 10.0
        gaps = count_internal_false_runs(thick10)
        eligible = bool(
            component_count == 1 and abs(main_length - registered) / registered <= 0.005
            and float(np.max(distance)) <= 1.5 * cell_diag and gaps == 0
        )
        rows.append({
            "system_id": system_id, "rgi_id": rgi,
            "positive_cell_count": int(np.count_nonzero(positive)), "positive_area_m2": float(np.count_nonzero(positive) * cell_area),
            "positive_component_count": int(component_count),
            "stored_main_node_count": len(main_xy), "stored_main_length_m": main_length,
            "registered_length_m": registered, "registered_length_error_fraction": abs(main_length - registered) / registered,
            "main_node_to_positive_cell_max_m": float(np.max(distance)), "cell_diagonal_m": cell_diag,
            "sampled_thickness_min_m": float(np.min(sampled_h)), "sampled_thickness_p05_m": float(np.percentile(sampled_h, 5)),
            "internal_thickness_over10_gap_count": gaps,
            "thickness_over10_fraction_on_main": float(np.mean(thick10)),
            "paired_cell_graph_static_ready": eligible,
        })
    eligible_rows = [row for row in rows if row["paired_cell_graph_static_ready"]]
    selected = min(eligible_rows, key=lambda row: row["positive_cell_count"])["system_id"] if eligible_rows else None
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    with (OUTPUT / "readiness.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "schema": "lia-191-negative-flux8-paired-cell-graph-readiness-v1",
        "status": "representative-selected" if selected is not None else "class-static-readiness-fail-no-pilot",
        "system_count": len(rows), "eligible_count": len(eligible_rows),
        "eligible_system_ids": [row["system_id"] for row in eligible_rows],
        "selection_rule": "minimum positive-cell count among systems passing all frozen readiness gates",
        "selected_representative": selected,
        "readiness_gates": {"positive_components": 1, "stored_main_length_error_max": 0.005, "main_node_support_distance_max_cell_diagonals": 1.5, "internal_thickness_over10_gap_count": 0},
        "oggm_run": False, "geometry_materialized": False, "zero_mb_run": False, "historical_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

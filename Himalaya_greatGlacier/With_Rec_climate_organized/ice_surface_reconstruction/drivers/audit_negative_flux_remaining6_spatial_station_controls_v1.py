"""Read-only raw spatial-station control audit for the six negative-flux holds.

This generalises the closed system-2342 audit without changing its mechanism:
each paired F19 positive cell stays on its nearest stored LIA branch, cells are
ordered by nearest branch station, and contiguous groups form approximately
50 m controls.  There is no elevation sorting, smoothing, bed correction, or
OGGM run.  The purpose is to determine whether 2342's failure is class-wide or
whether a geometrically distinct subset is already feasible in raw space.
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
PAIRED_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3"
NETWORK_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3"
ROUTE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_19/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux_remaining6_spatial_station_controls_v1"
SYSTEM_IDS = (890, 988, 1026, 1743, 2689, 2793)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_file(pattern: str) -> Path:
    matches = sorted(PAIRED_ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected-one-paired-file:{pattern}:{matches}")
    return matches[0]


def audit_system(system_id: int, registered_length_m: float, rgi_id: str) -> dict:
    paired = unique_file(f"lia_{system_id}_*_system_grid_v1.nc")
    network_path = NETWORK_ROOT / str(system_id) / "component_0" / "lia_centerline_network.pkl.gz"
    with gzip.open(network_path, "rb") as stream:
        network = pickle.load(stream)
    branches = network["branches"]
    main_index = int(network["main_branch_index"])

    with xr.open_dataset(paired) as ds:
        thick = np.asarray(ds["thickness_m"].values[0], dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        bed = np.asarray(ds["bed_elevation_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        transform = [float(value) for value in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
    dx_cell = abs(transform[1])
    dy_cell = abs(transform[5])
    cell_area = dx_cell * dy_cell
    positive = valid & lia & np.isfinite(thick) & np.isfinite(surface) & np.isfinite(bed) & (thick > 1e-6)
    rr, cc = np.nonzero(positive)
    xy = np.column_stack([
        transform[0] + (cc + 0.5) * transform[1] + (rr + 0.5) * transform[2],
        transform[3] + (cc + 0.5) * transform[4] + (rr + 0.5) * transform[5],
    ])

    all_nodes = []
    all_branch = []
    all_station = []
    branch_lengths = {}
    for branch in branches:
        idx = int(branch["branch_index"])
        coords = np.asarray(branch["coords_projected_m"], dtype=float)
        station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        branch_lengths[idx] = float(station[-1])
        all_nodes.append(coords)
        all_branch.extend([idx] * len(coords))
        all_station.extend(station.tolist())
    all_nodes = np.vstack(all_nodes)
    all_branch = np.asarray(all_branch, dtype=int)
    all_station = np.asarray(all_station, dtype=float)
    _, nearest = cKDTree(all_nodes).query(xy, k=1, workers=1)
    assigned_branch = all_branch[nearest]
    nearest_station = all_station[nearest]

    source_surface = surface[positive]
    source_thick = thick[positive]
    total_area = 0.0
    total_volume = 0.0
    model_surface = []
    model_area = []
    movements = []
    reverse_total = 0
    branch_rows = []
    failures = []
    for branch in branches:
        idx = int(branch["branch_index"])
        selected = np.flatnonzero(assigned_branch == idx)
        target = registered_length_m if idx == main_index else branch_lengths[idx]
        controls = max(2, int(round(target / 50.0)))
        exact_dx = target / controls
        if len(selected) < controls:
            failures.append({
                "system_id": system_id,
                "branch_index": idx,
                "reason": "fewer-cells-than-controls",
                "cells": len(selected),
                "controls": controls,
            })
            continue
        order = selected[np.argsort(nearest_station[selected], kind="stable")]
        groups = np.array_split(order, controls)
        branch_surfaces = []
        branch_area = 0.0
        branch_volume = 0.0
        branch_movement = []
        for node_index, group in enumerate(groups):
            area = float(len(group) * cell_area)
            volume = float(np.sum(source_thick[group]) * cell_area)
            mean_surface = float(np.mean(source_surface[group]))
            original_scaled = (
                nearest_station[group] / branch_lengths[idx] * target
                if branch_lengths[idx] > 0 else np.zeros(len(group))
            )
            destination = (node_index + 0.5) * exact_dx
            moved = np.abs(original_scaled - destination)
            branch_movement.extend(moved.tolist())
            branch_surfaces.append(mean_surface)
            branch_area += area
            branch_volume += volume
            model_surface.append(mean_surface)
            model_area.append(area)
        branch_surfaces = np.asarray(branch_surfaces, dtype=float)
        steps = np.diff(branch_surfaces)
        reverses = int(np.count_nonzero(steps > 1e-7))
        reverse_total += reverses
        total_area += branch_area
        total_volume += branch_volume
        movements.extend(branch_movement)
        branch_rows.append({
            "system_id": system_id,
            "branch_index": idx,
            "is_main": idx == main_index,
            "source_cell_count": len(selected),
            "control_count": controls,
            "target_length_m": target,
            "dx_m": exact_dx,
            "reverse_count": reverses,
            "maximum_reverse_m": float(max(0.0, np.max(steps))) if steps.size else 0.0,
            "movement_p95_m": float(np.quantile(branch_movement, 0.95)),
            "movement_max_m": float(np.max(branch_movement)),
            "area_m2": branch_area,
            "volume_m3": branch_volume,
        })

    source_area = float(len(source_surface) * cell_area)
    source_volume = float(np.sum(source_thick) * cell_area)
    cdf = (
        cdf_error(
            source_surface,
            np.full(len(source_surface), cell_area),
            np.asarray(model_surface),
            np.asarray(model_area),
        )
        if model_surface else float("inf")
    )
    p95 = float(np.quantile(movements, 0.95)) if movements else float("inf")
    maximum = float(np.max(movements)) if movements else float("inf")
    assigned_count = int(sum(row["source_cell_count"] for row in branch_rows))
    gates = {
        "all_cells_assigned_once": assigned_count == len(source_surface),
        "all_branches_retained": len(branch_rows) == len(branches) and not failures,
        "area_conserved": abs(total_area - source_area) <= 1e-5,
        "volume_conserved": abs(total_volume - source_volume) <= 1e-3,
        "hypsometry_within_2pct": cdf <= 0.02,
        "main_length_within_10pct": True,
        "raw_spatial_controls_monotonic": reverse_total == 0,
        "movement_p95_within_1km": p95 <= 1000.0,
        "movement_max_within_2p5km": maximum <= 2500.0,
    }
    passed = all(gates.values())
    result = {
        "system_id": system_id,
        "rgi_id": rgi_id,
        "status": "static-feasibility-pass" if passed else "static-feasibility-fail",
        "branch_count": len(branches),
        "main_branch_index": main_index,
        "source_cell_count": len(source_surface),
        "source_area_m2": source_area,
        "candidate_area_m2": total_area,
        "source_volume_m3": source_volume,
        "candidate_volume_m3": total_volume,
        "hypsometry_cdf_error_100m": cdf,
        "total_reverse_count": reverse_total,
        "movement_p95_m": p95,
        "movement_max_m": maximum,
        "gate_results": gates,
        "failures": failures,
        "paired_path": str(paired.relative_to(PROJECT)),
        "network_path": str(network_path.relative_to(PROJECT)),
        "paired_sha256": sha256(paired),
        "network_sha256": sha256(network_path),
        "branch_rows": branch_rows,
    }
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    route = pd.read_csv(ROUTE)
    results = []
    all_branches = []
    all_failures = []
    for system_id in SYSTEM_IDS:
        row = route.loc[route["lia_unique_id"] == system_id]
        if len(row) != 1:
            raise RuntimeError(f"route-row-not-unique:{system_id}:{len(row)}")
        record = row.iloc[0]
        result = audit_system(
            system_id,
            float(record["lia_registered_length_m"]),
            str(record["rgi_id"]),
        )
        results.append(result)
        all_branches.extend(result.pop("branch_rows"))
        all_failures.extend(result["failures"])
        system_dir = OUTPUT / f"system_{system_id}"
        system_dir.mkdir()
        (system_dir / "summary.json").write_text(
            json.dumps(result, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
            encoding="utf-8",
        )

    table_rows = []
    for result in results:
        table_rows.append({
            "system_id": result["system_id"],
            "rgi_id": result["rgi_id"],
            "status": result["status"],
            "branch_count": result["branch_count"],
            "source_cell_count": result["source_cell_count"],
            "source_area_m2": result["source_area_m2"],
            "source_volume_m3": result["source_volume_m3"],
            "hypsometry_cdf_error_100m": result["hypsometry_cdf_error_100m"],
            "total_reverse_count": result["total_reverse_count"],
            "movement_p95_m": result["movement_p95_m"],
            "movement_max_m": result["movement_max_m"],
            "failed_gates": ";".join(key for key, value in result["gate_results"].items() if not value),
        })
    pd.DataFrame(table_rows).to_csv(OUTPUT / "system_audit.csv", index=False)
    pd.DataFrame(all_branches).to_csv(OUTPUT / "branch_audit.csv", index=False)
    if all_failures:
        pd.DataFrame(all_failures).to_csv(OUTPUT / "failures.csv", index=False)
    passing = [result["system_id"] for result in results if result["status"] == "static-feasibility-pass"]
    aggregate = {
        "schema": "lia-191-negative-flux-remaining6-spatial-station-controls-v1",
        "status": "subclass-found" if passing else "no-feasible-subclass",
        "mechanism": "raw-nearest-branch-spatial-station-contiguous-controls-no-regularization",
        "system_ids": list(SYSTEM_IDS),
        "passing_system_ids": passing,
        "passing_count": len(passing),
        "failing_count": len(SYSTEM_IDS) - len(passing),
        "paired_source": "core191_anchor_reroute_f19_v3",
        "source_state_modified": False,
        "elevation_sorting_used": False,
        "surface_or_bed_regularization_used": False,
        "oggm_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": (
            "materialize the lowest-risk passing representative only"
            if passing else
            "close raw spatial-station controls for the remaining negative-flux holds and choose a new representation"
        ),
        "route_sha256": sha256(ROUTE),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

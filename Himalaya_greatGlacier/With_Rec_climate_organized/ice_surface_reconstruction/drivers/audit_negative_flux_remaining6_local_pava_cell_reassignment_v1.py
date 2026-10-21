"""Read-only local PAVA-block exact-cell reassignment for six negative-flux holds.

The raw nearest-branch spatial-station controls are the immutable starting
point.  Only contiguous node blocks that violate downstream-nonincreasing mean
surface are pooled.  Complete paired F19 cells inside each pooled block are
sorted by their unchanged source surface and repartitioned using the original
per-node cell counts.  This is local exact-cell reassignment, not elevation
regularisation: no cell value, branch identity, area, or volume is changed.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .audit_negative_flux_remaining6_spatial_station_controls_v1 import (
    NETWORK_ROOT,
    OUTPUT as RAW_OUTPUT,
    PAIRED_ROOT,
    PROJECT,
    ROUTE,
    SYSTEM_IDS,
    sha256,
    unique_file,
)
from .build_negative_flux2342_decoupled_branch_elevation_bands_v1 import cdf_error


OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux_remaining6_local_pava_cell_reassignment_v1"


def pava_blocks(means: np.ndarray, weights: np.ndarray) -> list[tuple[int, int]]:
    """Return minimal contiguous blocks whose weighted means are nonincreasing."""
    blocks: list[list[float | int]] = []
    for index, (mean, weight) in enumerate(zip(means, weights)):
        blocks.append([index, index + 1, float(mean * weight), float(weight)])
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left[2]) / float(left[3])
            right_mean = float(right[2]) / float(right[3])
            if left_mean >= right_mean - 1e-12:
                break
            blocks[-2:] = [[left[0], right[1], float(left[2]) + float(right[2]), float(left[3]) + float(right[3])]]
    return [(int(block[0]), int(block[1])) for block in blocks]


def exact_local_reassignment(groups: list[np.ndarray], values: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Iterate minimal PAVA pooling until exact repartition has no reverse step."""
    current = [np.asarray(group, dtype=np.int64) for group in groups]
    all_changed_blocks: list[tuple[int, int]] = []
    for _ in range(len(current) + 1):
        means = np.asarray([float(np.mean(values[group])) for group in current])
        counts = np.asarray([len(group) for group in current], dtype=float)
        blocks = pava_blocks(means, counts)
        changed = [(start, stop) for start, stop in blocks if stop - start > 1]
        if not changed:
            return current, all_changed_blocks
        rebuilt = list(current)
        for start, stop in changed:
            pool = np.concatenate(current[start:stop])
            ordered = pool[np.argsort(-values[pool], kind="stable")]
            cursor = 0
            for node_index in range(start, stop):
                count = len(current[node_index])
                rebuilt[node_index] = ordered[cursor:cursor + count]
                cursor += count
            all_changed_blocks.append((start, stop))
        current = rebuilt
    raise RuntimeError("local-pava-did-not-converge")


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
    source_surface = surface[positive]
    source_thick = thick[positive]

    all_nodes = []
    all_branch = []
    all_station = []
    branch_lengths: dict[int, float] = {}
    branch_coords: dict[int, np.ndarray] = {}
    branch_station: dict[int, np.ndarray] = {}
    for branch in branches:
        idx = int(branch["branch_index"])
        coords = np.asarray(branch["coords_projected_m"], dtype=float)
        station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        branch_lengths[idx] = float(station[-1])
        branch_coords[idx] = coords
        branch_station[idx] = station
        all_nodes.append(coords)
        all_branch.extend([idx] * len(coords))
        all_station.extend(station.tolist())
    all_nodes = np.vstack(all_nodes)
    all_branch = np.asarray(all_branch, dtype=int)
    all_station = np.asarray(all_station, dtype=float)
    _, nearest = cKDTree(all_nodes).query(xy, k=1, workers=1)
    assigned_branch = all_branch[nearest]
    nearest_station = all_station[nearest]

    branch_states = {}
    failures = []
    all_movements = []
    block_rows = []
    control_surfaces = []
    control_areas = []
    total_area = 0.0
    total_volume = 0.0
    raw_reverse_total = 0
    final_reverse_total = 0
    for branch in branches:
        idx = int(branch["branch_index"])
        selected = np.flatnonzero(assigned_branch == idx)
        target = registered_length_m if idx == main_index else branch_lengths[idx]
        controls = max(2, int(round(target / 50.0)))
        exact_dx = target / controls
        if len(selected) < controls:
            failures.append({"system_id": system_id, "branch_index": idx, "reason": "fewer-cells-than-controls"})
            continue
        order = selected[np.argsort(nearest_station[selected], kind="stable")]
        raw_groups = [np.asarray(group, dtype=np.int64) for group in np.array_split(order, controls)]
        raw_means = np.asarray([float(np.mean(source_surface[group])) for group in raw_groups])
        raw_reverse_total += int(np.count_nonzero(np.diff(raw_means) > 1e-7))
        final_groups, changed_blocks = exact_local_reassignment(raw_groups, source_surface)
        final_means = np.asarray([float(np.mean(source_surface[group])) for group in final_groups])
        final_reverse_total += int(np.count_nonzero(np.diff(final_means) > 1e-7))
        destinations = np.empty(len(source_surface), dtype=float)
        destinations.fill(np.nan)
        for node_index, group in enumerate(final_groups):
            destinations[group] = (node_index + 0.5) * exact_dx
            area = float(len(group) * cell_area)
            volume = float(np.sum(source_thick[group]) * cell_area)
            total_area += area
            total_volume += volume
            control_surfaces.append(float(np.mean(source_surface[group])))
            control_areas.append(area)
        original_scaled = nearest_station[selected] / branch_lengths[idx] * target if branch_lengths[idx] > 0 else np.zeros(len(selected))
        movement = np.abs(original_scaled - destinations[selected])
        all_movements.extend(movement.tolist())
        unique_blocks = sorted(set(changed_blocks))
        for start, stop in unique_blocks:
            block_rows.append({
                "system_id": system_id,
                "branch_index": idx,
                "start_node": start,
                "stop_node_exclusive": stop,
                "node_count": stop - start,
                "span_m": (stop - start) * exact_dx,
            })
        branch_states[idx] = {
            "surface": final_means,
            "dx": exact_dx,
            "target": target,
            "groups": final_groups,
            "movement_p95_m": float(np.quantile(movement, 0.95)),
            "movement_max_m": float(np.max(movement)),
            "largest_pool_span_m": max(((stop - start) * exact_dx for start, stop in unique_blocks), default=0.0),
        }

    junction_rows = []
    junction_reverse_count = 0
    for branch in branches:
        idx = int(branch["branch_index"])
        parent = branch.get("parent_index")
        if parent is None or idx not in branch_states or int(parent) not in branch_states:
            continue
        parent = int(parent)
        point = np.asarray(branch["flows_to_point_projected_m"], dtype=float)
        parent_node = int(np.argmin(np.linalg.norm(branch_coords[parent] - point, axis=1)))
        parent_native_station = float(branch_station[parent][parent_node])
        parent_control = min(
            len(branch_states[parent]["surface"]) - 1,
            max(0, int(parent_native_station / branch_lengths[parent] * branch_states[parent]["target"] / branch_states[parent]["dx"])),
        )
        tributary_surface = float(branch_states[idx]["surface"][-1])
        receiver_surface = float(branch_states[parent]["surface"][parent_control])
        reverse = receiver_surface > tributary_surface + 1e-7
        junction_reverse_count += int(reverse)
        junction_rows.append({
            "system_id": system_id,
            "tributary_branch": idx,
            "parent_branch": parent,
            "parent_control": parent_control,
            "tributary_terminal_surface_m": tributary_surface,
            "receiver_surface_m": receiver_surface,
            "reverse_m": max(0.0, receiver_surface - tributary_surface),
            "reverse": reverse,
        })

    source_area = float(len(source_surface) * cell_area)
    source_volume = float(np.sum(source_thick) * cell_area)
    cdf = cdf_error(
        source_surface,
        np.full(len(source_surface), cell_area),
        np.asarray(control_surfaces),
        np.asarray(control_areas),
    )
    p95 = float(np.quantile(all_movements, 0.95)) if all_movements else float("inf")
    maximum = float(np.max(all_movements)) if all_movements else float("inf")
    largest_pool = max((row["span_m"] for row in block_rows if row["system_id"] == system_id), default=0.0)
    gates = {
        "all_branches_retained": len(branch_states) == len(branches) and not failures,
        "area_conserved": abs(total_area - source_area) <= 1e-5,
        "volume_conserved": abs(total_volume - source_volume) <= 1e-3,
        "hypsometry_within_2pct": cdf <= 0.02,
        "within_branch_monotonic": final_reverse_total == 0,
        "junction_monotonic": junction_reverse_count == 0,
        "movement_p95_within_1km": p95 <= 1000.0,
        "movement_max_within_2p5km": maximum <= 2500.0,
    }
    return {
        "system_id": system_id,
        "rgi_id": rgi_id,
        "status": "static-feasibility-pass" if all(gates.values()) else "static-feasibility-fail",
        "branch_count": len(branches),
        "source_cell_count": len(source_surface),
        "source_area_m2": source_area,
        "candidate_area_m2": total_area,
        "source_volume_m3": source_volume,
        "candidate_volume_m3": total_volume,
        "hypsometry_cdf_error_100m": cdf,
        "raw_within_reverse_count": raw_reverse_total,
        "final_within_reverse_count": final_reverse_total,
        "junction_reverse_count": junction_reverse_count,
        "movement_p95_m": p95,
        "movement_max_m": maximum,
        "largest_local_pool_span_m": largest_pool,
        "gate_results": gates,
        "failures": failures,
        "paired_path": str(paired.relative_to(PROJECT)),
        "network_path": str(network_path.relative_to(PROJECT)),
        "paired_sha256": sha256(paired),
        "network_sha256": sha256(network_path),
        "block_rows": block_rows,
        "junction_rows": junction_rows,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    raw = json.loads((RAW_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    if raw["status"] != "no-feasible-subclass":
        raise RuntimeError("raw spatial-station predecessor is not closed")
    route = pd.read_csv(ROUTE)
    OUTPUT.mkdir(parents=True)
    results = []
    block_rows = []
    junction_rows = []
    for system_id in SYSTEM_IDS:
        row = route.loc[route["lia_unique_id"] == system_id]
        if len(row) != 1:
            raise RuntimeError(f"route-row-not-unique:{system_id}:{len(row)}")
        result = audit_system(system_id, float(row.iloc[0]["lia_registered_length_m"]), str(row.iloc[0]["rgi_id"]))
        block_rows.extend(result.pop("block_rows"))
        junction_rows.extend(result.pop("junction_rows"))
        results.append(result)
        system_dir = OUTPUT / f"system_{system_id}"
        system_dir.mkdir()
        (system_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    rows = []
    for result in results:
        rows.append({
            "system_id": result["system_id"],
            "rgi_id": result["rgi_id"],
            "status": result["status"],
            "branch_count": result["branch_count"],
            "hypsometry_cdf_error_100m": result["hypsometry_cdf_error_100m"],
            "raw_within_reverse_count": result["raw_within_reverse_count"],
            "final_within_reverse_count": result["final_within_reverse_count"],
            "junction_reverse_count": result["junction_reverse_count"],
            "movement_p95_m": result["movement_p95_m"],
            "movement_max_m": result["movement_max_m"],
            "largest_local_pool_span_m": result["largest_local_pool_span_m"],
            "failed_gates": ";".join(key for key, value in result["gate_results"].items() if not value),
        })
    pd.DataFrame(rows).to_csv(OUTPUT / "system_audit.csv", index=False)
    pd.DataFrame(block_rows).to_csv(OUTPUT / "pava_blocks.csv", index=False)
    pd.DataFrame(junction_rows).to_csv(OUTPUT / "junction_audit.csv", index=False)
    passing = [result["system_id"] for result in results if result["status"] == "static-feasibility-pass"]
    aggregate = {
        "schema": "lia-191-negative-flux-remaining6-local-pava-cell-reassignment-v1",
        "status": "subclass-found" if passing else "no-feasible-subclass",
        "mechanism": "minimal-contiguous-violating-block-complete-cell-reassignment",
        "system_ids": list(SYSTEM_IDS),
        "passing_system_ids": passing,
        "passing_count": len(passing),
        "failing_count": len(SYSTEM_IDS) - len(passing),
        "cell_values_modified": False,
        "branch_assignments_modified": False,
        "global_elevation_sorting_used": False,
        "surface_or_bed_regularization_used": False,
        "oggm_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": (
            "materialize the lowest-risk passing representative only"
            if passing else
            "close local PAVA cell reassignment and select a representation with explicit junction coupling"
        ),
        "predecessor_summary_sha256": sha256(RAW_OUTPUT / "summary.json"),
        "route_sha256": sha256(ROUTE),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

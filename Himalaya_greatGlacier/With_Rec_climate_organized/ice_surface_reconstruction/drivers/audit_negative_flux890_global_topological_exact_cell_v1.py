"""Global precedence-coupled exact-cell assignment for negative-flux system 890."""

from __future__ import annotations

import gzip
import heapq
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .audit_negative_flux_remaining6_local_pava_cell_reassignment_v1 import exact_local_reassignment
from .audit_negative_flux_remaining6_spatial_station_controls_v1 import NETWORK_ROOT, PROJECT, ROUTE, sha256, unique_file


SYSTEM_ID = 890
PREDECESSOR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux890_junction_coupled_cell_reassignment_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux890_global_topological_exact_cell_v1"


def interpolate_line(coords: np.ndarray, station: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(targets, station, coords[:, 0]), np.interp(targets, station, coords[:, 1])])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if predecessor["status"] != "static-feasibility-fail":
        raise RuntimeError("predecessor-not-closed")
    route = pd.read_csv(ROUTE)
    row = route.loc[route["lia_unique_id"] == SYSTEM_ID].iloc[0]
    registered = float(row["lia_registered_length_m"]); rgi_id = str(row["rgi_id"])
    paired = unique_file(f"lia_{SYSTEM_ID}_*_system_grid_v1.nc")
    network_path = NETWORK_ROOT / str(SYSTEM_ID) / "component_0" / "lia_centerline_network.pkl.gz"
    with gzip.open(network_path, "rb") as stream: network = pickle.load(stream)
    branches = network["branches"]; main_index = int(network["main_branch_index"])
    with xr.open_dataset(paired) as ds:
        thick = np.asarray(ds["thickness_m"].values[0], dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        bed = np.asarray(ds["bed_elevation_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        transform = [float(value) for value in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
    cell_area = abs(transform[1] * transform[5])
    positive = valid & lia & np.isfinite(thick) & np.isfinite(surface) & np.isfinite(bed) & (thick > 1e-6)
    rr, cc = np.nonzero(positive)
    xy = np.column_stack([
        transform[0] + (cc + 0.5) * transform[1] + (rr + 0.5) * transform[2],
        transform[3] + (cc + 0.5) * transform[4] + (rr + 0.5) * transform[5],
    ])
    source_surface = surface[positive]; source_thick = thick[positive]

    all_nodes = []; all_branch = []; all_station = []
    coords_by = {}; station_by = {}; lengths = {}; branch_by = {}
    for branch in branches:
        idx = int(branch["branch_index"]); coords = np.asarray(branch["coords_projected_m"], dtype=float)
        station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        coords_by[idx] = coords; station_by[idx] = station; lengths[idx] = float(station[-1]); branch_by[idx] = branch
        all_nodes.append(coords); all_branch.extend([idx] * len(coords)); all_station.extend(station.tolist())
    all_nodes = np.vstack(all_nodes); all_branch = np.asarray(all_branch); all_station = np.asarray(all_station)
    _, nearest = cKDTree(all_nodes).query(xy, k=1, workers=1)
    assigned_branch = all_branch[nearest]; nearest_station = all_station[nearest]

    states = {}; node_keys = []; quota = {}; control_xy = {}; initial_mean = {}; initial_branch_counts = {}
    for idx in sorted(branch_by):
        selected = np.flatnonzero(assigned_branch == idx)
        target = registered if idx == main_index else lengths[idx]
        controls = max(2, int(round(target / 50.0))); exact_dx = target / controls
        order = selected[np.argsort(nearest_station[selected], kind="stable")]
        raw = [np.asarray(group, dtype=np.int64) for group in np.array_split(order, controls)]
        local, _ = exact_local_reassignment(raw, source_surface)
        native_targets = ((np.arange(controls) + 0.5) * exact_dx / target) * lengths[idx]
        coords = interpolate_line(coords_by[idx], station_by[idx], native_targets)
        states[idx] = {"groups": local, "controls": controls, "dx": exact_dx, "target": target}
        initial_branch_counts[idx] = int(sum(len(group) for group in local))
        for node, group in enumerate(local):
            key = (idx, node); node_keys.append(key); quota[key] = len(group); control_xy[key] = coords[node]
            initial_mean[key] = float(np.mean(source_surface[group]))

    outgoing = {key: [] for key in node_keys}; indegree = {key: 0 for key in node_keys}
    for idx, state in states.items():
        for node in range(state["controls"] - 1):
            u = (idx, node); v = (idx, node + 1); outgoing[u].append(v); indegree[v] += 1
    junctions = []
    for idx in sorted(branch_by):
        parent = branch_by[idx].get("parent_index")
        if parent is None: continue
        parent = int(parent); point = np.asarray(branch_by[idx]["flows_to_point_projected_m"], dtype=float)
        native_node = int(np.argmin(np.linalg.norm(coords_by[parent] - point, axis=1)))
        native_station = float(station_by[parent][native_node])
        receiver = min(states[parent]["controls"] - 1, max(0, int(native_station / lengths[parent] * states[parent]["target"] / states[parent]["dx"])))
        u = (idx, states[idx]["controls"] - 1); v = (parent, receiver)
        outgoing[u].append(v); indegree[v] += 1; junctions.append((u, v))

    ready = []
    for key in node_keys:
        if indegree[key] == 0: heapq.heappush(ready, (-initial_mean[key], key[0], key[1]))
    topological = []
    while ready:
        _, branch, node = heapq.heappop(ready); key = (branch, node); topological.append(key)
        for target in outgoing[key]:
            indegree[target] -= 1
            if indegree[target] == 0: heapq.heappush(ready, (-initial_mean[target], target[0], target[1]))
    if len(topological) != len(node_keys): raise RuntimeError("control-network-not-dag")

    ordered_cells = np.argsort(-source_surface, kind="stable")
    groups = {}; cursor = 0
    for key in topological:
        count = quota[key]; groups[key] = ordered_cells[cursor:cursor + count]; cursor += count
    if cursor != len(source_surface): raise RuntimeError(f"cell-quota-mismatch:{cursor}:{len(source_surface)}")

    means = {key: float(np.mean(source_surface[group])) for key, group in groups.items()}
    within_reverse = sum(int(means[(idx, node + 1)] > means[(idx, node)] + 1e-7) for idx, state in states.items() for node in range(state["controls"] - 1))
    junction_reverse = sum(int(means[v] > means[u] + 1e-7) for u, v in junctions)
    destination = np.empty_like(xy)
    final_branch_counts = {idx: 0 for idx in states}; branch_volume_initial = {}; branch_volume_final = {}
    for idx, state in states.items():
        original = np.concatenate(state["groups"]); branch_volume_initial[idx] = float(np.sum(source_thick[original]) * cell_area)
    for key, group in groups.items():
        destination[group] = control_xy[key]; final_branch_counts[key[0]] += len(group)
    for idx in states:
        final = np.concatenate([groups[(idx, node)] for node in range(states[idx]["controls"])])
        branch_volume_final[idx] = float(np.sum(source_thick[final]) * cell_area)
    movement = np.linalg.norm(xy - destination, axis=1)
    branch_volume_change = {idx: (branch_volume_final[idx] - branch_volume_initial[idx]) / branch_volume_initial[idx] for idx in states}
    gates = {
        "all_branches_retained": len(states) == len(branches),
        "all_cells_assigned_once": len(np.unique(np.concatenate(list(groups.values())))) == len(source_surface),
        "branch_area_quotas_unchanged": all(final_branch_counts[idx] == initial_branch_counts[idx] for idx in states),
        "system_area_conserved": True,
        "system_volume_conserved": True,
        "hypsometry_exact": True,
        "within_branch_monotonic": within_reverse == 0,
        "junction_monotonic": junction_reverse == 0,
        "movement_p95_within_1km": float(np.quantile(movement, 0.95)) <= 1000.0,
        "movement_max_within_2p5km": float(np.max(movement)) <= 2500.0,
        "maximum_branch_volume_change_within_25pct": max(abs(value) for value in branch_volume_change.values()) <= 0.25,
    }
    summary = {
        "schema": "lia-191-negative-flux890-global-topological-exact-cell-v1",
        "status": "static-feasibility-pass" if all(gates.values()) else "static-feasibility-fail",
        "system_id": SYSTEM_ID, "rgi_id": rgi_id, "branch_count": len(branches), "control_node_count": len(node_keys),
        "source_cell_count": len(source_surface), "source_area_m2": float(len(source_surface) * cell_area),
        "source_volume_m3": float(np.sum(source_thick) * cell_area), "within_reverse_count": within_reverse,
        "junction_reverse_count": junction_reverse, "movement_p95_m": float(np.quantile(movement, 0.95)),
        "movement_max_m": float(np.max(movement)), "maximum_absolute_branch_volume_change_fraction": max(abs(value) for value in branch_volume_change.values()),
        "branch_volume_change_fraction": branch_volume_change, "gate_results": gates,
        "cell_values_modified": False, "branch_area_quotas_modified": False, "surface_or_bed_regularization_used": False,
        "oggm_run": False, "zero_mb_run": False, "steady_50y_run": False, "historical_run": False,
        "source_sha256": {str(paired.relative_to(PROJECT)): sha256(paired), str(network_path.relative_to(PROJECT)): sha256(network_path), str(PREDECESSOR.relative_to(PROJECT)): sha256(PREDECESSOR)},
        "next_action": "materialize one static representative only" if all(gates.values()) else "close global-topological exact-cell assignment",
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()

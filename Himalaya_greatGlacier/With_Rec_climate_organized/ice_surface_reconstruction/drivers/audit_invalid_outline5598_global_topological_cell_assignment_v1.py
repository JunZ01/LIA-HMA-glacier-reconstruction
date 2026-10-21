"""Read-only global network/cell assignment feasibility audit for system 5598.

The candidate keeps every paired F19 positive-ice cell and the complete stored
LIA branch topology.  Fixed 50 m-scale control-volume quotas are obtained from
nearest spatial ownership.  Controls are then ordered by the full network DAG
and complete source cells are assigned in descending source-surface order.
This guarantees all within-branch and junction monotonic inequalities if the
fixed quotas are feasible.  The audit does not materialize flowlines or run
OGGM; locality and registered geometry gates remain decisive.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[4]
SYSTEM_ID = 5598
ROUTE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_22/lia_191_current_route.csv"
PAIRED_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3"
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/5598/component_0/lia_centerline_network.pkl.gz"
PREDECESSOR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_lowrisk8_stored_lia_network_surface_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline5598_global_topological_cell_assignment_v1"
TOL = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate_line(coords: np.ndarray, station: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(targets, station, coords[:, 0]),
        np.interp(targets, station, coords[:, 1]),
    ])


def cdf_error_100m(source_surface: np.ndarray, node_surface: np.ndarray, node_count: np.ndarray) -> float:
    lo = 100.0 * np.floor(min(float(source_surface.min()), float(node_surface.min())) / 100.0)
    hi = 100.0 * np.ceil(max(float(source_surface.max()), float(node_surface.max())) / 100.0)
    thresholds = np.arange(lo, hi + 100.0, 100.0)
    source_cdf = np.asarray([np.mean(source_surface <= value) for value in thresholds])
    mapped_cdf = np.asarray([
        np.sum(node_count[node_surface <= value]) / np.sum(node_count) for value in thresholds
    ])
    return float(np.max(np.abs(source_cdf - mapped_cdf)))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if predecessor["status"] != "no-feasible-subclass":
        raise RuntimeError("stored-network predecessor is not closed")
    route = pd.read_csv(ROUTE)
    row = route.loc[route["lia_unique_id"].astype(int).eq(SYSTEM_ID)]
    if len(row) != 1:
        raise RuntimeError(f"route-row-not-unique:{len(row)}")
    registered_length = float(row.iloc[0]["lia_registered_length_m"])
    rgi_id = str(row.iloc[0]["rgi_id"])
    paired_matches = sorted(PAIRED_ROOT.glob(f"lia_{SYSTEM_ID}_*_system_grid_v1.nc"))
    if len(paired_matches) != 1:
        raise RuntimeError(f"paired-not-unique:{paired_matches}")
    paired = paired_matches[0]
    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    branches = network["branches"]
    main_index = int(network["main_branch_index"])
    by_index = {int(branch["branch_index"]): branch for branch in branches}

    with xr.open_dataset(paired) as ds:
        thick = np.asarray(ds["thickness_m"].values[0], dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        bed = np.asarray(ds["bed_elevation_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        transform = [float(value) for value in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
    positive = valid & lia & np.isfinite(thick) & np.isfinite(surface) & np.isfinite(bed) & (thick > 1e-6)
    rr, cc = np.nonzero(positive)
    cell_xy = np.column_stack([
        transform[0] + (cc + 0.5) * transform[1] + (rr + 0.5) * transform[2],
        transform[3] + (cc + 0.5) * transform[4] + (rr + 0.5) * transform[5],
    ])
    cell_surface = surface[positive]
    cell_thick = thick[positive]
    cell_area = abs(transform[1] * transform[5] - transform[2] * transform[4])

    control_xy_parts: list[np.ndarray] = []
    control_branch: list[int] = []
    control_node: list[int] = []
    offsets: dict[int, int] = {}
    control_count: dict[int, int] = {}
    branch_length: dict[int, float] = {}
    branch_station: dict[int, np.ndarray] = {}
    cursor = 0
    for idx in sorted(by_index):
        branch = by_index[idx]
        coords = np.asarray(branch["coords_projected_m"], dtype=float)
        station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        native_length = float(station[-1])
        target_length = registered_length if idx == main_index else native_length
        count = max(2, int(round(target_length / 50.0)))
        exact_dx = target_length / count
        native_targets = ((np.arange(count) + 0.5) * exact_dx / target_length) * native_length
        controls = interpolate_line(coords, station, native_targets)
        offsets[idx] = cursor
        control_count[idx] = count
        branch_length[idx] = native_length
        branch_station[idx] = station
        cursor += count
        control_xy_parts.append(controls)
        control_branch.extend([idx] * count)
        control_node.extend(range(count))
    control_xy = np.vstack(control_xy_parts)
    control_branch_arr = np.asarray(control_branch, dtype=int)
    control_node_arr = np.asarray(control_node, dtype=int)

    nearest_distance, nearest_owner = cKDTree(control_xy).query(cell_xy, k=1, workers=1)
    node_count = np.bincount(nearest_owner, minlength=len(control_xy)).astype(int)
    zero_quota_nodes = int(np.count_nonzero(node_count == 0))
    nearest_sum = np.bincount(nearest_owner, weights=cell_surface, minlength=len(control_xy))
    nearest_mean = np.divide(nearest_sum, node_count, out=np.full(len(control_xy), -np.inf), where=node_count > 0)

    edges: list[tuple[int, int, str]] = []
    for idx in sorted(by_index):
        off = offsets[idx]
        for node in range(control_count[idx] - 1):
            edges.append((off + node, off + node + 1, "within"))
        branch = by_index[idx]
        parent = branch.get("parent_index")
        if parent is not None:
            parent = int(parent)
            point = np.asarray(branch["flows_to_point_projected_m"], dtype=float)
            parent_controls = control_xy[offsets[parent]: offsets[parent] + control_count[parent]]
            receiver = int(np.argmin(np.linalg.norm(parent_controls - point, axis=1)))
            edges.append((off + control_count[idx] - 1, offsets[parent] + receiver, "junction"))

    outgoing: list[list[int]] = [[] for _ in range(len(control_xy))]
    indegree = np.zeros(len(control_xy), dtype=int)
    for upstream, downstream, _ in edges:
        outgoing[upstream].append(downstream)
        indegree[downstream] += 1
    heap: list[tuple[float, int]] = []
    for node in np.flatnonzero(indegree == 0):
        heapq.heappush(heap, (-float(nearest_mean[node]), int(node)))
    order: list[int] = []
    while heap:
        _, node = heapq.heappop(heap)
        order.append(node)
        for downstream in outgoing[node]:
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                heapq.heappush(heap, (-float(nearest_mean[downstream]), int(downstream)))
    dag_valid = len(order) == len(control_xy)

    assigned_owner = np.full(len(cell_surface), -1, dtype=int)
    node_surface = np.full(len(control_xy), np.nan, dtype=float)
    node_volume = np.zeros(len(control_xy), dtype=float)
    quota_complete = zero_quota_nodes == 0 and dag_valid
    if quota_complete:
        sorted_cells = np.argsort(-cell_surface, kind="stable")
        cell_cursor = 0
        for node in order:
            count = int(node_count[node])
            chosen = sorted_cells[cell_cursor:cell_cursor + count]
            if len(chosen) != count:
                quota_complete = False
                break
            assigned_owner[chosen] = node
            node_surface[node] = float(np.mean(cell_surface[chosen]))
            node_volume[node] = float(np.sum(cell_thick[chosen]) * cell_area)
            cell_cursor += count
        quota_complete = quota_complete and cell_cursor == len(cell_surface) and np.all(assigned_owner >= 0)

    if quota_complete:
        reverse_rows = [
            (upstream, downstream, kind, float(node_surface[downstream] - node_surface[upstream]))
            for upstream, downstream, kind in edges
            if node_surface[downstream] - node_surface[upstream] > TOL
        ]
        destination = control_xy[assigned_owner]
        movement = np.linalg.norm(cell_xy - destination, axis=1)
        adjustment = node_surface - nearest_mean
        cdf = cdf_error_100m(cell_surface, node_surface, node_count)
        mapped_volume = float(np.sum(node_volume))
        branch_old_area = np.bincount(control_branch_arr[nearest_owner], minlength=max(by_index) + 1) * cell_area
        branch_new_area = np.bincount(control_branch_arr[assigned_owner], minlength=max(by_index) + 1) * cell_area
        used = np.asarray(sorted(by_index), dtype=int)
        relative_branch_area_change = np.abs(branch_new_area[used] - branch_old_area[used]) / np.maximum(branch_old_area[used], cell_area)
        metrics = {
            "within_reverse_count": int(sum(kind == "within" for *_, kind, _ in reverse_rows)),
            "junction_reverse_count": int(sum(kind == "junction" for *_, kind, _ in reverse_rows)),
            "maximum_reverse_gap_m": max([gap for *_, gap in reverse_rows], default=0.0),
            "hypsometry_cdf_error_100m": cdf,
            "maximum_absolute_node_surface_adjustment_m": float(np.max(np.abs(adjustment))),
            "mean_absolute_node_surface_adjustment_m": float(np.average(np.abs(adjustment), weights=node_count)),
            "cell_movement_mean_m": float(np.mean(movement)),
            "cell_movement_p95_m": float(np.quantile(movement, 0.95)),
            "cell_movement_max_m": float(np.max(movement)),
            "maximum_relative_branch_area_change": float(np.max(relative_branch_area_change)),
            "mapped_volume_m3": mapped_volume,
        }
    else:
        reverse_rows = []
        metrics = {
            "within_reverse_count": None,
            "junction_reverse_count": None,
            "maximum_reverse_gap_m": None,
            "hypsometry_cdf_error_100m": None,
            "maximum_absolute_node_surface_adjustment_m": None,
            "mean_absolute_node_surface_adjustment_m": None,
            "cell_movement_mean_m": None,
            "cell_movement_p95_m": None,
            "cell_movement_max_m": None,
            "maximum_relative_branch_area_change": None,
            "mapped_volume_m3": None,
        }

    source_area = float(len(cell_surface) * cell_area)
    source_volume = float(np.sum(cell_thick) * cell_area)
    mapped_area = float(np.sum(node_count) * cell_area)
    gates = {
        "network_is_dag": dag_valid,
        "all_control_quotas_positive": zero_quota_nodes == 0,
        "all_cells_assigned_once": quota_complete,
        "area_conserved": abs(mapped_area - source_area) <= max(1e-6, source_area * 1e-12),
        "volume_conserved": quota_complete and abs(metrics["mapped_volume_m3"] - source_volume) <= max(1e-6, source_volume * 1e-12),
        "within_branch_monotonic": quota_complete and metrics["within_reverse_count"] == 0,
        "junction_monotonic": quota_complete and metrics["junction_reverse_count"] == 0,
        "hypsometry_within_2pct": quota_complete and metrics["hypsometry_cdf_error_100m"] <= 0.02,
        "maximum_node_adjustment_within_150m": quota_complete and metrics["maximum_absolute_node_surface_adjustment_m"] <= 150.0,
        "euclidean_displacement_p95_within_1km": quota_complete and metrics["cell_movement_p95_m"] <= 1000.0,
        "euclidean_displacement_max_within_2p5km": quota_complete and metrics["cell_movement_max_m"] <= 2500.0,
    }
    passed = all(gates.values())
    summary = {
        "schema": "lia-191-invalid-outline5598-global-topological-cell-assignment-v1",
        "status": "static-feasibility-pass" if passed else "static-feasibility-fail",
        "decision": "materialization-authorized-for-system5598-only" if passed else "global-topological-cell-assignment-closed-at-feasibility-gate",
        "system_id": SYSTEM_ID,
        "rgi_id": rgi_id,
        "mechanism": "global-DAG-linear-extension-with-fixed-nearest-spatial-control-quotas-and-complete-cell-assignment",
        "branch_count": len(branches),
        "control_count": len(control_xy),
        "edge_count": len(edges),
        "positive_cell_count": len(cell_surface),
        "zero_quota_control_count": zero_quota_nodes,
        "source_area_m2": source_area,
        "mapped_area_m2": mapped_area,
        "area_error_fraction": (mapped_area - source_area) / source_area,
        "source_volume_m3": source_volume,
        "volume_error_fraction": None if not quota_complete else (metrics["mapped_volume_m3"] - source_volume) / source_volume,
        **metrics,
        "gate_results": gates,
        "source_cells_modified": False,
        "source_files_modified": False,
        "geometry_materialized": False,
        "oggm_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "materialize system 5598 only" if passed else "close this global assignment mechanism and route to a different state representation",
        "provenance": {
            "paired_path": str(paired.relative_to(PROJECT)),
            "paired_sha256": sha256(paired),
            "network_path": str(NETWORK.relative_to(PROJECT)),
            "network_sha256": sha256(NETWORK),
            "predecessor_sha256": sha256(PREDECESSOR),
            "route_sha256": sha256(ROUTE),
        },
    }
    OUTPUT.mkdir(parents=True)
    pd.DataFrame([
        {
            "upstream_global_control": upstream,
            "downstream_global_control": downstream,
            "edge_kind": kind,
            "reverse_gap_m": gap,
        }
        for upstream, downstream, kind, gap in reverse_rows
    ]).to_csv(OUTPUT / "reverse_edges.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

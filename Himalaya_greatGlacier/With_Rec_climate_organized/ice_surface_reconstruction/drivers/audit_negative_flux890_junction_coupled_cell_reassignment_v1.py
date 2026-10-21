"""Read-only exact-cell junction coupling audit for system 890.

Starting from the already audited branch-local PAVA state, each remaining
reverse junction may pool only a contiguous tributary-terminal suffix and a
contiguous receiver-downstream segment.  Complete source cells are sorted by
their unchanged surface and repartitioned using unchanged node cell counts.
Candidates are accepted only when they remove the selected junction reversal,
keep both branches monotonic, and strictly reduce the whole-network junction
reversal count.  No OGGM model or source file is modified.
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

from .audit_negative_flux_remaining6_local_pava_cell_reassignment_v1 import exact_local_reassignment
from .audit_negative_flux_remaining6_spatial_station_controls_v1 import (
    NETWORK_ROOT,
    PAIRED_ROOT,
    PROJECT,
    ROUTE,
    sha256,
    unique_file,
)
from .build_negative_flux2342_decoupled_branch_elevation_bands_v1 import cdf_error


SYSTEM_ID = 890
PREDECESSOR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux_remaining6_local_pava_cell_reassignment_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux890_junction_coupled_cell_reassignment_v1"


def interpolate_line(coords: np.ndarray, station: np.ndarray, targets: np.ndarray) -> np.ndarray:
    x = np.interp(targets, station, coords[:, 0])
    y = np.interp(targets, station, coords[:, 1])
    return np.column_stack([x, y])


def branch_means(groups: list[np.ndarray], values: np.ndarray) -> np.ndarray:
    return np.asarray([float(np.mean(values[group])) for group in groups])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if predecessor["status"] != "no-feasible-subclass":
        raise RuntimeError("local-PAVA predecessor is not closed")
    route = pd.read_csv(ROUTE)
    row = route.loc[route["lia_unique_id"] == SYSTEM_ID]
    if len(row) != 1:
        raise RuntimeError(f"route-row-not-unique:{len(row)}")
    registered_length = float(row.iloc[0]["lia_registered_length_m"])
    rgi_id = str(row.iloc[0]["rgi_id"])

    paired = unique_file(f"lia_{SYSTEM_ID}_*_system_grid_v1.nc")
    network_path = NETWORK_ROOT / str(SYSTEM_ID) / "component_0" / "lia_centerline_network.pkl.gz"
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
    source_xy = np.column_stack([
        transform[0] + (cc + 0.5) * transform[1] + (rr + 0.5) * transform[2],
        transform[3] + (cc + 0.5) * transform[4] + (rr + 0.5) * transform[5],
    ])
    source_surface = surface[positive]
    source_thick = thick[positive]

    all_nodes = []
    all_branch = []
    all_station = []
    lengths = {}
    coords_by_branch = {}
    station_by_branch = {}
    branch_by_index = {}
    for branch in branches:
        idx = int(branch["branch_index"])
        coords = np.asarray(branch["coords_projected_m"], dtype=float)
        station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        lengths[idx] = float(station[-1])
        coords_by_branch[idx] = coords
        station_by_branch[idx] = station
        branch_by_index[idx] = branch
        all_nodes.append(coords)
        all_branch.extend([idx] * len(coords))
        all_station.extend(station.tolist())
    all_nodes = np.vstack(all_nodes)
    all_branch = np.asarray(all_branch, dtype=int)
    all_station = np.asarray(all_station, dtype=float)
    _, nearest = cKDTree(all_nodes).query(source_xy, k=1, workers=1)
    assigned_branch = all_branch[nearest]
    nearest_station = all_station[nearest]

    states = {}
    raw_reverse = 0
    for idx in sorted(branch_by_index):
        selected = np.flatnonzero(assigned_branch == idx)
        target = registered_length if idx == main_index else lengths[idx]
        controls = max(2, int(round(target / 50.0)))
        exact_dx = target / controls
        if len(selected) < controls:
            raise RuntimeError(f"fewer-cells-than-controls:{idx}:{len(selected)}:{controls}")
        order = selected[np.argsort(nearest_station[selected], kind="stable")]
        raw_groups = [np.asarray(group, dtype=np.int64) for group in np.array_split(order, controls)]
        raw_reverse += int(np.count_nonzero(np.diff(branch_means(raw_groups, source_surface)) > 1e-7))
        groups, _ = exact_local_reassignment(raw_groups, source_surface)
        native_targets = ((np.arange(controls) + 0.5) * exact_dx / target) * lengths[idx]
        control_xy = interpolate_line(coords_by_branch[idx], station_by_branch[idx], native_targets)
        states[idx] = {
            "groups": groups,
            "target": target,
            "dx": exact_dx,
            "control_xy": control_xy,
        }

    junctions = []
    for idx in sorted(branch_by_index):
        branch = branch_by_index[idx]
        parent = branch.get("parent_index")
        if parent is None:
            continue
        parent = int(parent)
        point = np.asarray(branch["flows_to_point_projected_m"], dtype=float)
        native_parent_node = int(np.argmin(np.linalg.norm(coords_by_branch[parent] - point, axis=1)))
        native_parent_station = float(station_by_branch[parent][native_parent_node])
        receiver = min(
            len(states[parent]["groups"]) - 1,
            max(0, int(native_parent_station / lengths[parent] * states[parent]["target"] / states[parent]["dx"])),
        )
        junctions.append((idx, parent, receiver))

    def junction_table() -> list[dict]:
        rows = []
        for tributary, parent, receiver in junctions:
            t_surface = float(branch_means(states[tributary]["groups"], source_surface)[-1])
            p_surface = float(branch_means(states[parent]["groups"], source_surface)[receiver])
            rows.append({
                "tributary_branch": tributary,
                "parent_branch": parent,
                "parent_control": receiver,
                "tributary_terminal_surface_m": t_surface,
                "receiver_surface_m": p_surface,
                "reverse_m": max(0.0, p_surface - t_surface),
                "reverse": p_surface > t_surface + 1e-7,
            })
        return rows

    initial_junctions = junction_table()
    operation_rows = []
    stalled = False
    for iteration in range(len(junctions) + 1):
        current = junction_table()
        reverse_rows = [entry for entry in current if entry["reverse"]]
        if not reverse_rows:
            break
        reverse_rows.sort(key=lambda entry: (-entry["reverse_m"], entry["tributary_branch"]))
        base_reverse_count = len(reverse_rows)
        chosen = None
        for edge in reverse_rows:
            tributary = int(edge["tributary_branch"])
            parent = int(edge["parent_branch"])
            receiver = int(edge["parent_control"])
            t_groups = states[tributary]["groups"]
            p_groups = states[parent]["groups"]
            max_t = min(20, len(t_groups))
            max_p = min(20, len(p_groups) - receiver)
            candidates = [(kt + kp, kt, kp) for kt in range(1, max_t + 1) for kp in range(1, max_p + 1)]
            candidates.sort()
            for _, kt, kp in candidates:
                trial_t = list(t_groups)
                trial_p = list(p_groups)
                pool_groups = trial_t[-kt:] + trial_p[receiver:receiver + kp]
                pool = np.concatenate(pool_groups)
                ordered = pool[np.argsort(-source_surface[pool], kind="stable")]
                counts = [len(group) for group in pool_groups]
                cursor = 0
                rebuilt = []
                for count in counts:
                    rebuilt.append(ordered[cursor:cursor + count])
                    cursor += count
                trial_t[-kt:] = rebuilt[:kt]
                trial_p[receiver:receiver + kp] = rebuilt[kt:]
                t_means = branch_means(trial_t, source_surface)
                p_means = branch_means(trial_p, source_surface)
                if np.any(np.diff(t_means) > 1e-7) or np.any(np.diff(p_means) > 1e-7):
                    continue
                old_t = states[tributary]["groups"]
                old_p = states[parent]["groups"]
                states[tributary]["groups"] = trial_t
                states[parent]["groups"] = trial_p
                new_junctions = junction_table()
                new_reverse_count = sum(int(entry["reverse"]) for entry in new_junctions)
                selected_fixed = not next(entry for entry in new_junctions if entry["tributary_branch"] == tributary)["reverse"]
                states[tributary]["groups"] = old_t
                states[parent]["groups"] = old_p
                if selected_fixed and new_reverse_count < base_reverse_count:
                    chosen = (tributary, parent, receiver, kt, kp, trial_t, trial_p, new_reverse_count, len(pool))
                    break
            if chosen is not None:
                break
        if chosen is None:
            stalled = True
            break
        tributary, parent, receiver, kt, kp, trial_t, trial_p, new_count, pool_cells = chosen
        states[tributary]["groups"] = trial_t
        states[parent]["groups"] = trial_p
        operation_rows.append({
            "iteration": iteration + 1,
            "tributary_branch": tributary,
            "parent_branch": parent,
            "parent_control": receiver,
            "tributary_suffix_nodes": kt,
            "parent_downstream_nodes": kp,
            "pooled_cell_count": pool_cells,
            "junction_reverse_count_before": base_reverse_count,
            "junction_reverse_count_after": new_count,
        })

    final_junctions = junction_table()
    final_within = sum(
        int(np.count_nonzero(np.diff(branch_means(state["groups"], source_surface)) > 1e-7))
        for state in states.values()
    )
    final_junction_reverse = sum(int(entry["reverse"]) for entry in final_junctions)

    destination_xy = np.empty_like(source_xy)
    destination_xy.fill(np.nan)
    control_surfaces = []
    control_areas = []
    total_area = 0.0
    total_volume = 0.0
    for idx, state in states.items():
        for node_index, group in enumerate(state["groups"]):
            destination_xy[group] = state["control_xy"][node_index]
            area = float(len(group) * cell_area)
            volume = float(np.sum(source_thick[group]) * cell_area)
            total_area += area
            total_volume += volume
            control_surfaces.append(float(np.mean(source_surface[group])))
            control_areas.append(area)
    if not np.all(np.isfinite(destination_xy)):
        raise RuntimeError("some-cells-lack-destination")
    displacement = np.linalg.norm(destination_xy - source_xy, axis=1)
    source_area = float(len(source_surface) * cell_area)
    source_volume = float(np.sum(source_thick) * cell_area)
    cdf = cdf_error(
        source_surface,
        np.full(len(source_surface), cell_area),
        np.asarray(control_surfaces),
        np.asarray(control_areas),
    )
    gates = {
        "all_10_branches_retained": len(states) == len(branches),
        "all_cells_assigned_once": int(sum(len(group) for state in states.values() for group in state["groups"])) == len(source_surface),
        "area_conserved": abs(total_area - source_area) <= 1e-5,
        "volume_conserved": abs(total_volume - source_volume) <= 1e-3,
        "hypsometry_within_2pct": cdf <= 0.02,
        "within_branch_monotonic": final_within == 0,
        "junction_monotonic": final_junction_reverse == 0,
        "junction_solver_not_stalled": not stalled,
        "euclidean_displacement_p95_within_1km": float(np.quantile(displacement, 0.95)) <= 1000.0,
        "euclidean_displacement_max_within_2p5km": float(np.max(displacement)) <= 2500.0,
    }
    passed = all(gates.values())
    summary = {
        "schema": "lia-191-negative-flux890-junction-coupled-cell-reassignment-v1",
        "status": "static-feasibility-pass" if passed else "static-feasibility-fail",
        "decision": "materialization-authorized" if passed else "mechanism-closed-at-junction-cell-gate",
        "system_id": SYSTEM_ID,
        "rgi_id": rgi_id,
        "branch_count": len(branches),
        "main_branch_index": main_index,
        "source_cell_count": len(source_surface),
        "source_area_m2": source_area,
        "candidate_area_m2": total_area,
        "source_volume_m3": source_volume,
        "candidate_volume_m3": total_volume,
        "hypsometry_cdf_error_100m": cdf,
        "raw_within_reverse_count": raw_reverse,
        "final_within_reverse_count": final_within,
        "initial_junction_reverse_count": sum(int(entry["reverse"]) for entry in initial_junctions),
        "final_junction_reverse_count": final_junction_reverse,
        "junction_operation_count": len(operation_rows),
        "solver_stalled": stalled,
        "euclidean_displacement_p95_m": float(np.quantile(displacement, 0.95)),
        "euclidean_displacement_max_m": float(np.max(displacement)),
        "gate_results": gates,
        "cell_values_modified": False,
        "source_branch_assignment_can_change_only_inside_accepted_junction_blocks": True,
        "surface_or_bed_regularization_used": False,
        "oggm_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "materialize system 890 only" if passed else "close junction-coupled exact-cell reassignment",
        "provenance": {
            "paired_path": str(paired.relative_to(PROJECT)),
            "paired_sha256": sha256(paired),
            "network_path": str(network_path.relative_to(PROJECT)),
            "network_sha256": sha256(network_path),
            "predecessor_sha256": sha256(PREDECESSOR),
            "route_sha256": sha256(ROUTE),
        },
    }
    OUTPUT.mkdir(parents=True)
    pd.DataFrame(initial_junctions).to_csv(OUTPUT / "junctions_initial.csv", index=False)
    pd.DataFrame(final_junctions).to_csv(OUTPUT / "junctions_final.csv", index=False)
    pd.DataFrame(operation_rows).to_csv(OUTPUT / "junction_operations.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

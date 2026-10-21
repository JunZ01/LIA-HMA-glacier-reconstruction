"""Read-only stored-LIA-network surface/topology audit for eight low-risk holds."""

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


PROJECT = Path(__file__).resolve().parents[4]
RISK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_hold12_paired_network_risk_v1/risk_ranking.csv"
ROUTE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_21/lia_191_current_route.csv"
NETWORK_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3"
PAIRED_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_lowrisk8_stored_lia_network_surface_v1"
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
    risk = pd.read_csv(RISK)
    selected = risk.loc[risk["low_risk_static_candidate"].astype(str).str.lower().eq("true")]
    if len(selected) != 8:
        raise RuntimeError(f"expected-eight-low-risk-systems:{len(selected)}")
    route = pd.read_csv(ROUTE).set_index("lia_unique_id")
    system_rows = []
    branch_rows = []
    junction_rows = []
    for system_id in selected["system_id"].astype(int):
        network_path = NETWORK_ROOT / str(system_id) / "component_0" / "lia_centerline_network.pkl.gz"
        paired_matches = sorted(PAIRED_ROOT.glob(f"lia_{system_id}_*_system_grid_v1.nc"))
        if len(paired_matches) != 1:
            raise RuntimeError(f"paired-not-unique:{system_id}:{paired_matches}")
        paired = paired_matches[0]
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
        positive = valid & lia & np.isfinite(thick) & np.isfinite(surface) & np.isfinite(bed) & (thick > 1e-6)
        rr, cc = np.nonzero(positive)
        positive_xy = np.column_stack([
            transform[0] + (cc + 0.5) * transform[1] + (rr + 0.5) * transform[2],
            transform[3] + (cc + 0.5) * transform[4] + (rr + 0.5) * transform[5],
        ])
        positive_surface = surface[positive]
        cell_diag = float(np.hypot(abs(transform[1]), abs(transform[5])))
        tree = cKDTree(positive_xy)
        by_index = {int(branch["branch_index"]): branch for branch in branches}
        within_reverse = 0
        all_node_distance = []
        all_surface_difference = []
        branch_data = {}
        finite = True
        for branch in branches:
            idx = int(branch["branch_index"])
            coords = np.asarray(branch["coords_projected_m"], dtype=float)
            reference = np.asarray(branch["reference_surface_m"], dtype=float)
            if len(coords) != len(reference):
                raise RuntimeError(f"coordinate-surface-length-mismatch:{system_id}:{idx}")
            distance, nearest = tree.query(coords, k=1, workers=1)
            sampled = positive_surface[nearest]
            steps = np.diff(reference)
            reverse_count = int(np.count_nonzero(steps > TOLERANCE_M))
            within_reverse += reverse_count
            finite = finite and bool(np.all(np.isfinite(coords)) and np.all(np.isfinite(reference)))
            differences = reference - sampled
            all_node_distance.extend(distance.tolist())
            all_surface_difference.extend(np.abs(differences).tolist())
            station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
            branch_data[idx] = {"coords": coords, "surface": reference, "station": station}
            branch_rows.append({
                "system_id": system_id,
                "branch_index": idx,
                "is_main": idx == main_index,
                "node_count": len(coords),
                "length_m": float(station[-1]),
                "within_reverse_count": reverse_count,
                "maximum_within_reverse_m": float(max(0.0, np.max(steps))) if steps.size else 0.0,
                "paired_support_p95_m": float(np.quantile(distance, 0.95)),
                "paired_support_max_m": float(np.max(distance)),
                "reference_minus_paired_abs_p95_m": float(np.quantile(np.abs(differences), 0.95)),
                "reference_minus_paired_abs_max_m": float(np.max(np.abs(differences))),
            })
        junction_reverse = 0
        connector_distances = []
        for branch in branches:
            idx = int(branch["branch_index"])
            parent = branch.get("parent_index")
            if parent is None:
                continue
            parent = int(parent)
            point = np.asarray(branch["flows_to_point_projected_m"], dtype=float)
            parent_coords = branch_data[parent]["coords"]
            receiver = int(np.argmin(np.linalg.norm(parent_coords - point, axis=1)))
            tributary_surface = float(branch_data[idx]["surface"][-1])
            receiver_surface = float(branch_data[parent]["surface"][receiver])
            gap = receiver_surface - tributary_surface
            reverse = gap > TOLERANCE_M
            junction_reverse += int(reverse)
            connector_distance = float(np.linalg.norm(branch_data[idx]["coords"][-1] - parent_coords[receiver]))
            connector_distances.append(connector_distance)
            junction_rows.append({
                "system_id": system_id,
                "tributary_branch": idx,
                "parent_branch": parent,
                "parent_node": receiver,
                "tributary_terminal_surface_m": tributary_surface,
                "receiver_surface_m": receiver_surface,
                "reverse_gap_m": max(0.0, gap),
                "reverse": reverse,
                "connector_distance_m": connector_distance,
            })
        registered = float(route.loc[system_id, "lia_registered_length_m"])
        main_length = float(branch_data[main_index]["station"][-1])
        gates = {
            "all_reference_surfaces_finite": finite,
            "within_branch_reverse_zero": within_reverse == 0,
            "junction_reverse_zero": junction_reverse == 0,
            "main_length_error_within_0p5pct": abs(main_length - registered) / registered <= 0.005,
            "main_path_gap_free_in_predecessor": bool(selected.loc[selected["system_id"].eq(system_id), "main_path_gap_free"].iloc[0]),
            "main_support_within_predecessor_limit": bool(selected.loc[selected["system_id"].eq(system_id), "main_support_within_limit"].iloc[0]),
        }
        feasible = all(gates.values())
        system_rows.append({
            "system_id": system_id,
            "rgi_id": route.loc[system_id, "rgi_id"],
            "status": "stored-network-surface-feasible" if feasible else "stored-network-surface-fail",
            "branch_count": len(branches),
            "main_branch_index": main_index,
            "within_reverse_count": within_reverse,
            "junction_reverse_count": junction_reverse,
            "main_length_m": main_length,
            "registered_length_m": registered,
            "main_length_error_fraction": abs(main_length - registered) / registered,
            "all_branch_node_support_p95_m": float(np.quantile(all_node_distance, 0.95)),
            "all_branch_node_support_max_m": float(np.max(all_node_distance)),
            "reference_paired_abs_difference_p95_m": float(np.quantile(all_surface_difference, 0.95)),
            "reference_paired_abs_difference_max_m": float(np.max(all_surface_difference)),
            "maximum_connector_distance_m": max(connector_distances, default=0.0),
            "gate_results": json.dumps(gates, sort_keys=True),
            "paired_sha256": sha256(paired),
            "network_sha256": sha256(network_path),
        })
    systems = pd.DataFrame(system_rows)
    feasible = systems.loc[systems["status"].eq("stored-network-surface-feasible")].sort_values(["branch_count", "main_length_m"])
    selected_rep = None if feasible.empty else int(feasible.iloc[0]["system_id"])
    summary = {
        "schema": "lia-191-invalid-outline-lowrisk8-stored-lia-network-surface-v1",
        "status": "feasible-subclass-found" if selected_rep is not None else "no-feasible-subclass",
        "system_count": len(systems),
        "feasible_system_ids": feasible["system_id"].astype(int).tolist(),
        "feasible_count": len(feasible),
        "selected_representative": selected_rep,
        "mechanism": "stored-anchor-reroute-LIA-network-reference-surface-and-topology",
        "state_materialized": False,
        "surface_or_bed_modified": False,
        "geometry_run": False,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": (
            "audit conservative current-paired cell mapping onto the selected stored network before materialization"
            if selected_rep is not None else
            "close stored-network reference-surface route for the low-risk invalid-outline subclass"
        ),
        "risk_audit_sha256": sha256(RISK),
        "route_sha256": sha256(ROUTE),
    }
    OUTPUT.mkdir(parents=True)
    systems.to_csv(OUTPUT / "system_summary.csv", index=False)
    pd.DataFrame(branch_rows).to_csv(OUTPUT / "branch_summary.csv", index=False)
    pd.DataFrame(junction_rows).to_csv(OUTPUT / "junction_summary.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

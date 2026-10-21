"""Build a conservative, decoupled branch-elevation-band state for 2342.

Every current paired F19 positive-ice cell is assigned exactly once to its
nearest stored LIA-network branch.  Each branch is then collapsed independently
to a conservative rectangular elevation-band flowline.  Branches do not flow
into one another in this candidate: that is the new structural hypothesis used
to remove negative apparent-flux and junction mass-creation paths.  The stored
main branch alone carries the registered length observation.

This driver performs static construction only.  It writes candidate flowlines
only if all conservation, hypsometry, support-distance, and length gates pass.
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

from .lia_elevation_band import build_elevation_band_state, build_rectangular_flowline


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_decoupled_branch_elevation_bands_v1"
PAIRED = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_2342_RGI60-15.04541_system_grid_v1.nc"
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/2342/component_0/lia_centerline_network.pkl.gz"
FORMAL = PROJECT / "w673inv2342_continuous_v1/inversion_summary_formal.json"
CONFIG = PROJECT / "w673inv2342_continuous_v1/config_snapshot.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cdf_error(source_surface, source_area, model_surface, model_area) -> float:
    lo = 100.0 * np.floor(min(np.min(source_surface), np.min(model_surface)) / 100.0)
    hi = 100.0 * np.ceil(max(np.max(source_surface), np.max(model_surface)) / 100.0)
    edges = np.arange(lo, hi + 100.0, 100.0)
    hs, _ = np.histogram(source_surface, bins=edges, weights=source_area)
    hm, _ = np.histogram(model_surface, bins=edges, weights=model_area)
    return float(np.max(np.abs(np.cumsum(hs) / np.sum(hs) - np.cumsum(hm) / np.sum(hm))))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    branches = network["branches"]
    main_index = int(network["main_branch_index"])
    if sorted(int(branch["branch_index"]) for branch in branches) != list(range(len(branches))):
        raise RuntimeError("branch-indices-not-contiguous")

    with xr.open_dataset(PAIRED) as ds:
        thick = np.asarray(ds["thickness_m"].values[0], dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        bed = np.asarray(ds["bed_elevation_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        transform = [float(value) for value in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
    cell_dx = abs(transform[1])
    cell_dy = abs(transform[5])
    cell_area = cell_dx * cell_dy
    positive = valid & lia & np.isfinite(thick) & np.isfinite(surface) & np.isfinite(bed) & (thick > 1e-6)
    rr, cc = np.nonzero(positive)
    x = transform[0] + (cc + 0.5) * transform[1] + (rr + 0.5) * transform[2]
    y = transform[3] + (cc + 0.5) * transform[4] + (rr + 0.5) * transform[5]
    cell_xy = np.column_stack([x, y])
    node_xy = []
    node_branch = []
    for branch in branches:
        coords = np.asarray(branch["coords_projected_m"], dtype=float)
        node_xy.append(coords)
        node_branch.extend([int(branch["branch_index"])] * len(coords))
    node_xy = np.vstack(node_xy)
    node_branch = np.asarray(node_branch, dtype=int)
    distances, nearest = cKDTree(node_xy).query(cell_xy, k=1)
    assignment = node_branch[nearest]

    source_area = float(np.sum(positive) * cell_area)
    source_volume = float(np.sum(thick[positive]) * cell_area)
    flowlines = []
    states = []
    rows = []
    failures = []
    for branch in branches:
        idx = int(branch["branch_index"])
        branch_valid = np.zeros_like(positive)
        selected_positions = assignment == idx
        branch_valid[rr[selected_positions], cc[selected_positions]] = True
        coords = np.asarray(branch["coords_projected_m"], dtype=float)
        spatial_length = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        target_length = float(config["registered_lia_length_m"]) if idx == main_index else spatial_length
        try:
            state = build_elevation_band_state(
                surface,
                bed,
                thick,
                branch_valid,
                cell_dx_m=cell_dx,
                cell_dy_m=cell_dy,
                elevation_bin_size_m=30.0,
                node_dx_m=50.0,
                target_length_m=target_length,
            )
            flowline = build_rectangular_flowline(
                state,
                rgi_id=f"LIA-2342-B{idx}",
                downstream_buffer_m=5000.0,
            )
            states.append(state)
            flowlines.append(flowline)
            rows.append({
                "branch_index": idx,
                "is_main": idx == main_index,
                "source_cell_count": int(np.sum(selected_positions)),
                "source_area_m2": state.source_area_m2,
                "source_volume_m3": state.source_volume_m3,
                "target_length_m": target_length,
                "initial_length_m": float(np.sum(np.asarray(flowline.thick) > 1e-6) * flowline.dx_meter),
                "node_count_active": int(np.sum(np.asarray(flowline.thick) > 1e-6)),
                "total_node_count": int(flowline.nx),
                "maximum_cell_support_distance_m": float(np.max(distances[selected_positions])),
                "p95_cell_support_distance_m": float(np.quantile(distances[selected_positions], 0.95)),
                "surface_reverse_count": int(np.sum(np.diff(state.node_surface_h_m) > 1e-7)),
            })
        except Exception as exc:
            failures.append({"branch_index": idx, "error": f"{type(exc).__name__}: {exc}"})

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "branch_static_audit.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT / "branch_failures.csv", index=False)

    if states:
        model_area = np.concatenate([state.node_width_m * state.node_dx_m for state in states])
        model_surface = np.concatenate([state.node_surface_h_m for state in states])
        total_area = float(sum(state.source_area_m2 for state in states))
        total_volume = float(sum(state.source_volume_m3 for state in states))
        source_surfaces = surface[positive]
        source_areas = np.full(source_surfaces.size, cell_area)
        cdf = cdf_error(source_surfaces, source_areas, model_surface, model_area)
    else:
        total_area = total_volume = cdf = float("nan")

    main_rows = frame[frame["is_main"]] if not frame.empty else frame
    main_length_error = (
        float(abs(main_rows.iloc[0]["initial_length_m"] - config["registered_lia_length_m"]) / config["registered_lia_length_m"])
        if len(main_rows) == 1 else float("inf")
    )
    assignment_counts = np.bincount(assignment, minlength=len(branches))
    gate_results = {
        "all_source_cells_assigned_once": int(np.sum(assignment_counts)) == int(np.sum(positive)),
        "every_stored_branch_retained": len(states) == len(branches) and np.all(assignment_counts > 0),
        "no_branch_build_failure": len(failures) == 0,
        "area_conserved": abs(total_area - source_area) <= 1e-5,
        "volume_conserved": abs(total_volume - source_volume) <= 1e-3,
        "hypsometry_within_2pct": cdf <= 0.02,
        "main_length_within_10pct": main_length_error <= 0.10,
        "all_branches_monotonic": bool(not frame.empty and int(frame["surface_reverse_count"].sum()) == 0),
        "cell_support_p95_within_2p5km": bool(not frame.empty and float(frame["p95_cell_support_distance_m"].max()) <= 2500.0),
        "cell_support_max_within_5km": bool(not frame.empty and float(frame["maximum_cell_support_distance_m"].max()) <= 5000.0),
        "paired_area_matches_formal": abs(source_area - float(formal["paired_area_m2"])) <= 1e-6,
        "paired_volume_matches_formal": abs(source_volume - float(formal["paired_volume_m3"])) <= 1e-3,
    }
    passed = all(gate_results.values())
    if passed:
        with gzip.open(OUTPUT / "candidate_decoupled_branch_flowlines.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream)
    summary = {
        "schema": "lia-191-negative-flux2342-decoupled-branch-elevation-bands-v1",
        "status": "static-pass" if passed else "static-fail",
        "decision": "zero-mb-authorized" if passed else "mechanism-closed-at-static-gate",
        "system_id": 2342,
        "mechanism": "nearest-LIA-network-branch conservative independent elevation-band ensemble",
        "main_branch_index": main_index,
        "branch_count": len(branches),
        "source_cell_count": int(np.sum(positive)),
        "source_area_m2": source_area,
        "candidate_area_m2": total_area,
        "source_volume_m3": source_volume,
        "candidate_volume_m3": total_volume,
        "hypsometry_cdf_error_100m": cdf,
        "main_length_error_fraction": main_length_error,
        "maximum_support_distance_m": None if frame.empty else float(frame["maximum_cell_support_distance_m"].max()),
        "maximum_branch_p95_support_distance_m": None if frame.empty else float(frame["p95_cell_support_distance_m"].max()),
        "gate_results": gate_results,
        "failures": failures,
        "negative_apparent_flux_used": False,
        "junction_mass_transfer_used": False,
        "climate_or_stage2_modified": False,
        "oggm_dynamics_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "promotion_run": False,
        "provenance": {
            "paired": str(PAIRED.resolve()), "paired_sha256": sha256(PAIRED),
            "network": str(NETWORK.resolve()), "network_sha256": sha256(NETWORK),
            "formal": str(FORMAL.resolve()), "formal_sha256": sha256(FORMAL),
            "config": str(CONFIG.resolve()), "config_sha256": sha256(CONFIG),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Read-only risk stratification for the 12 current invalid-outline holds."""

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
ROUTE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_20/lia_191_current_route.csv"
CLASSIFICATION = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/current_route_v8_invalid_polygon_classification_v1/invalid_polygon_classification.csv"
PAIRED_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3"
NETWORK_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_hold12_paired_network_risk_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def false_runs(mask: np.ndarray) -> int:
    active = np.flatnonzero(mask)
    if not active.size:
        return 0
    core = mask[active[0]:active[-1] + 1]
    false = ~core
    return int(np.count_nonzero(false & np.r_[True, ~false[:-1]]))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    route = pd.read_csv(ROUTE)
    mask = route["controller_state"].eq("hold") & route["outline_state"].eq("invalid-polygon")
    holds = route.loc[mask].copy()
    if len(holds) != 12:
        raise RuntimeError(f"expected-12-invalid-outline-holds:{len(holds)}")
    classification = pd.read_csv(CLASSIFICATION).set_index("system_id")
    rows = []
    for _, record in holds.iterrows():
        system_id = int(record["lia_unique_id"])
        paired_matches = sorted(PAIRED_ROOT.glob(f"lia_{system_id}_*_system_grid_v1.nc"))
        if len(paired_matches) != 1:
            raise RuntimeError(f"paired-not-unique:{system_id}:{paired_matches}")
        paired = paired_matches[0]
        network_path = NETWORK_ROOT / str(system_id) / "component_0" / "lia_centerline_network.pkl.gz"
        with gzip.open(network_path, "rb") as stream:
            network = pickle.load(stream)
        main = network["branches"][int(network["main_branch_index"])]
        main_xy = np.asarray(main["coords_projected_m"], dtype=float)
        main_length = float(np.sum(np.linalg.norm(np.diff(main_xy, axis=0), axis=1)))
        with xr.open_dataset(paired) as ds:
            thick = np.asarray(ds["thickness_m"].values[0], dtype=float)
            surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
            bed = np.asarray(ds["bed_elevation_m"].values[0], dtype=float)
            valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
            lia = np.asarray(ds["lia_mask"].values, dtype=bool)
            transform = [float(value) for value in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
        cell_dx = abs(transform[1])
        cell_dy = abs(transform[5])
        cell_area = cell_dx * cell_dy
        cell_diag = float(np.hypot(cell_dx, cell_dy))
        positive = valid & lia & np.isfinite(thick) & np.isfinite(surface) & np.isfinite(bed) & (thick > 1e-6)
        rr, cc = np.nonzero(positive)
        xy = np.column_stack([
            transform[0] + (cc + 0.5) * transform[1] + (rr + 0.5) * transform[2],
            transform[3] + (cc + 0.5) * transform[4] + (rr + 0.5) * transform[5],
        ])
        distance, nearest = cKDTree(xy).query(main_xy, k=1, workers=1)
        mapped_thick = thick[positive][nearest]
        supported = (distance <= 1.5 * cell_diag) & (mapped_thick > 10.0)
        internal_gaps = false_runs(supported)
        source = classification.loc[system_id]
        rows.append({
            "system_id": system_id,
            "rgi_id": record["rgi_id"],
            "registered_lia_length_m": float(record["lia_registered_length_m"]),
            "stored_main_length_m": main_length,
            "stored_main_length_error_fraction": abs(main_length - float(record["lia_registered_length_m"])) / float(record["lia_registered_length_m"]),
            "branch_count": len(network["branches"]),
            "paired_positive_cell_count": int(np.count_nonzero(positive)),
            "paired_area_m2": float(np.count_nonzero(positive) * cell_area),
            "paired_volume_m3": float(np.sum(thick[positive]) * cell_area),
            "outline_area_m2": float(source["source_area_m2"]),
            "source_holes": int(source["source_holes"]),
            "repaired_holes": int(source["repaired_holes"]),
            "main_support_p95_m": float(np.quantile(distance, 0.95)),
            "main_support_max_m": float(np.max(distance)),
            "main_support_limit_m": 1.5 * cell_diag,
            "main_node_count": len(main_xy),
            "main_supported_thick_gt10_count": int(np.count_nonzero(supported)),
            "main_internal_gap_count": internal_gaps,
            "main_path_gap_free": internal_gaps == 0,
            "main_support_within_limit": float(np.max(distance)) <= 1.5 * cell_diag,
            "paired_sha256": sha256(paired),
            "network_sha256": sha256(network_path),
        })
    table = pd.DataFrame(rows)
    table["low_risk_static_candidate"] = (
        table["main_path_gap_free"]
        & table["main_support_within_limit"]
        & (table["stored_main_length_error_fraction"] <= 0.005)
    )
    ranked = table.sort_values(
        ["low_risk_static_candidate", "main_internal_gap_count", "branch_count", "paired_area_m2"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ranked["risk_rank"] = np.arange(1, len(ranked) + 1)
    passing = ranked.loc[ranked["low_risk_static_candidate"]]
    representative = None if passing.empty else int(passing.iloc[0]["system_id"])
    summary = {
        "schema": "lia-191-invalid-outline-hold12-paired-network-risk-v1",
        "status": "low-risk-subclass-found" if representative is not None else "no-low-risk-subclass",
        "system_count": len(ranked),
        "system_ids": ranked["system_id"].astype(int).tolist(),
        "low_risk_system_ids": passing["system_id"].astype(int).tolist(),
        "low_risk_count": len(passing),
        "selected_representative": representative,
        "selection_rule": "gap-free stored main path; support max<=1.5 cell diagonals; stored main length error<=0.5%; then minimum branch count and area",
        "topology_repair_changed_25m_support": False,
        "geometry_run": False,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "source_modified": False,
        "next_action": (
            "run one unchanged topology-repair geometry/static representative, without terminal condensation"
            if representative is not None else
            "close current paired-network low-risk route and require a new observation-path representation"
        ),
        "route_sha256": sha256(ROUTE),
        "classification_sha256": sha256(CLASSIFICATION),
    }
    OUTPUT.mkdir(parents=True)
    ranked.to_csv(OUTPUT / "risk_ranking.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Static LP gate for an unchanged dynamic root plus passive tributary reservoir."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .audit_negative_flux2342_dynamic_subnetwork_reservoir_v1 import reservoir_lp
from .audit_negative_flux2342_hypsometric_residual_reservoir_v1 import (
    CONFIG, FLOWLINES, FORMAL, PAIRED, PROJECT, sha256,
)


OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_dynamic_trunk_passive_reservoir_v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with gzip.open(FLOWLINES, "rb") as stream:
        flowlines = pickle.load(stream)
    roots = [(i, fl) for i, fl in enumerate(flowlines) if fl.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"root-count:{len(roots)}")
    root_index, root = roots[0]
    active = np.asarray(root.thick) > 1e-6
    root_area_values = np.asarray(root.widths_m)[active] * root.dx_meter
    root_surface = np.asarray(root.surface_h)[active]
    root_area = float(np.sum(root_area_values))
    root_volume = float(np.sum(np.asarray(root.section)[active] * root.dx_meter))

    with xr.open_dataset(PAIRED) as ds:
        h = np.asarray(ds["thickness_m"].values[0], dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        transform = [float(value) for value in str(ds["spatial_ref"].attrs["GeoTransform"]).split()]
    cell_area = abs(transform[1] * transform[5])
    mask = valid & lia & np.isfinite(h) & np.isfinite(surface) & (h > 1e-6)
    paired_h = h[mask]
    paired_surface = surface[mask]
    paired_area = float(mask.sum() * cell_area)
    paired_volume = float(np.sum(paired_h) * cell_area)

    lo = 100 * np.floor(min(np.min(paired_surface), np.min(root_surface)) / 100)
    hi = 100 * np.ceil(max(np.max(paired_surface), np.max(root_surface)) / 100)
    edges = np.arange(lo, hi + 100, 100)
    n = len(edges) - 1
    paired_ids = np.clip(np.digitize(paired_surface, edges) - 1, 0, n - 1)
    root_ids = np.clip(np.digitize(root_surface, edges) - 1, 0, n - 1)
    paired_bins = np.bincount(paired_ids, weights=np.full(len(paired_ids), cell_area), minlength=n)
    root_bins = np.bincount(root_ids, weights=root_area_values, minlength=n)
    hmin = np.zeros(n); hmax = np.zeros(n)
    for j in range(n):
        values = paired_h[paired_ids == j]
        if values.size:
            hmin[j] = float(values.min()); hmax[j] = float(values.max())
    result = reservoir_lp(root_bins, root_area, root_volume, paired_bins, paired_area, paired_volume, hmin, hmax)
    feasible = bool(result is not None and result.success)
    cdf = float(result.x[-1]) if feasible else None
    area_fraction = root_area / paired_area
    volume_fraction = root_volume / paired_volume
    if feasible:
        rarea = result.x[:n]; rvolume = result.x[n:2*n]
        pd.DataFrame({
            "bin_lower_m": edges[:-1], "bin_upper_m": edges[1:],
            "paired_area_m2": paired_bins, "dynamic_root_area_m2": root_bins,
            "reservoir_area_m2": rarea, "reservoir_volume_m3": rvolume,
            "reservoir_mean_thickness_m": np.divide(rvolume, rarea, out=np.zeros_like(rvolume), where=rarea>0),
        }).to_csv(OUTPUT / "reservoir_by_100m_bin.csv", index=False)
        area_error = root_area + float(np.sum(rarea)) - paired_area
        volume_error = root_volume + float(np.sum(rvolume)) - paired_volume
    else:
        area_error = volume_error = None
    gates = {
        "lp_nonnegative_source_supported_reservoir_feasible": feasible,
        "dynamic_root_area_at_least_20pct": area_fraction >= 0.20,
        "dynamic_root_volume_at_least_20pct": volume_fraction >= 0.20,
        "combined_hypsometry_within_2pct": feasible and cdf <= 0.02,
        "area_closure": feasible and abs(area_error) <= 1e-5,
        "volume_closure": feasible and abs(volume_error) <= 1e-3,
        "root_length_within_10pct": abs(float(formal["model_root_length_m"])-float(config["registered_lia_length_m"]))/float(config["registered_lia_length_m"]) <= 0.10,
        "root_terminal_zero": float(root.thick[-1]) <= 10.0,
        "root_active_surface_monotonic": int(np.sum(np.diff(np.asarray(root.surface_h)[np.flatnonzero(active)]) > 1e-7)) == 0,
        "free_tail_at_least_5km": float(formal["model_free_tail_m"]) >= 5000.0,
    }
    passed = all(gates.values())
    if passed:
        with gzip.open(OUTPUT / "candidate_dynamic_root.pkl.gz", "wb") as stream:
            pickle.dump(root, stream)
    summary = {
        "schema": "lia-191-negative-flux2342-dynamic-trunk-passive-reservoir-v1",
        "status": "static-pass" if passed else "static-fail",
        "decision": "zero-mb-authorized" if passed else "mechanism-closed-at-static-gate",
        "system_id": 2342,
        "root_index": root_index,
        "dynamic_root_area_m2": root_area,
        "dynamic_root_volume_m3": root_volume,
        "paired_area_m2": paired_area,
        "paired_volume_m3": paired_volume,
        "dynamic_root_area_fraction": area_fraction,
        "dynamic_root_volume_fraction": volume_fraction,
        "reservoir_area_m2": paired_area-root_area,
        "reservoir_volume_m3": paired_volume-root_volume,
        "optimized_combined_cdf_error_100m": cdf,
        "area_closure_error_m2": area_error,
        "volume_closure_error_m3": volume_error,
        "gate_results": gates,
        "root_geometry_mutated": False,
        "climate_or_stage2_modified": False,
        "oggm_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "promotion_run": False,
        "provenance": {
            "paired_sha256": sha256(PAIRED), "flowlines_sha256": sha256(FLOWLINES),
            "formal_sha256": sha256(FORMAL), "config_sha256": sha256(CONFIG),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Bilinear path-sampling adjudication for system-14 source-native 2-D state."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


PROJECT = Path(__file__).resolve().parents[4]
V1 = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_v1/system_14/summary.json"
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_14_RGI60-15.01152_system_grid_v1.nc"
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/14/component_0/lia_centerline_network.pkl.gz"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_v2/system_14"
THRESHOLD_M = 10.0
LENGTH_GATE = 0.005


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base = json.loads(V1.read_text(encoding="utf-8"))
    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    coords = np.asarray(network["branches"][network["main_branch_index"]]["coords_projected_m"], dtype=float)
    with xr.open_dataset(SOURCE) as ds:
        h = np.asarray(ds["thickness_m"].isel(member=0), dtype=float)
        x = np.asarray(ds["x"], dtype=float)
        y = np.asarray(ds["y"], dtype=float)
    interpolator = RegularGridInterpolator((y[::-1], x), h[::-1], bounds_error=False, fill_value=np.nan)
    sampled = interpolator(np.column_stack([coords[:, 1], coords[:, 0]]))
    active = np.isfinite(sampled) & (sampled > THRESHOLD_M)
    active_indices = np.flatnonzero(active)
    first, last = int(active_indices[0]), int(active_indices[-1])
    internal_gaps = np.flatnonzero(~active[first:last + 1]) + first
    segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    control_lengths = np.empty(len(coords), dtype=float)
    control_lengths[0] = segment_lengths[0] / 2.0
    control_lengths[-1] = segment_lengths[-1] / 2.0
    control_lengths[1:-1] = (segment_lengths[:-1] + segment_lengths[1:]) / 2.0
    active_length = float(control_lengths[active].sum())
    registered = float(base["registered_length_m"])
    error = abs(active_length - registered) / registered
    checks = dict(base["checks"])
    checks["root_observation_path_continuous"] = len(internal_gaps) == 0
    checks["root_active_control_volume_length"] = error <= LENGTH_GATE
    checks["bilinear_sampling_is_native_2d_operator"] = True
    checks.pop("root_length", None)
    status = "static-feasibility-pass" if all(checks.values()) else "static-feasibility-fail"
    summary = {
        **base,
        "schema": "lia-191-invalid-outline-planform2d-source-native-system14-v2",
        "status": status,
        "path_sampling": "bilinear-native-2d-thickness-with-voronoi-control-lengths",
        "root_active_first_index": first,
        "root_active_last_index": last,
        "root_internal_gap_indices": internal_gaps.astype(int).tolist(),
        "root_sampled_thickness_minimum_active_span_m": float(np.nanmin(sampled[first:last + 1])),
        "root_active_control_volume_length_m": active_length,
        "root_active_length_relative_error": error,
        "root_active_length_gate": LENGTH_GATE,
        "checks": checks,
        "v1_nearest_cell_audit": str(V1),
        "padded_grid_materialized": False,
        "upstream2d_run": False,
        "zero_mb_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

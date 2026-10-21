"""Read-only feasibility audit for a passive hypsometric residual reservoir.

System 2342 is the only current negative-flux hold whose standard dynamic core
is below the paired F19 target in both area and volume.  This audit leaves the
core untouched and asks whether a non-negative, non-overlapping *aggregate*
reservoir can exactly supply the missing area and volume while bringing the
combined 100 m surface-elevation CDF within the formal two-percent gate.

The variables are reservoir area and volume per 100 m source-supported bin.
Volume in each bin is bounded by the minimum and maximum paired F19 thickness
observed in that bin.  No OGGM model is run and no dynamic state is written.
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
from scipy.optimize import linprog


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_hypsometric_residual_reservoir_v1"
PAIRED = PROJECT / "validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_2342_RGI60-15.04541_system_grid_v1.nc"
FORMAL = PROJECT / "w673inv2342_continuous_v1/inversion_summary_formal.json"
FLOWLINES = PROJECT / "w673inv2342_continuous_v1/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92342/model_flowlines.pkl"
CONFIG = PROJECT / "w673inv2342_continuous_v1/config_snapshot.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    flowlines = load_pickle(FLOWLINES)

    with xr.open_dataset(PAIRED) as ds:
        h = np.asarray(ds["thickness_m"].values[0], dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        transform = str(ds["spatial_ref"].attrs["GeoTransform"]).split()
        cell_area = abs(float(transform[1]) * float(transform[5]))
    mask = valid & lia & np.isfinite(h) & np.isfinite(surface) & (h > 1e-6)
    paired_h = h[mask]
    paired_surface = surface[mask]
    paired_area = float(mask.sum() * cell_area)
    paired_volume = float(np.sum(paired_h) * cell_area)

    core_area_parts = []
    core_volume_parts = []
    core_surface_parts = []
    core_reverse = 0
    root_count = 0
    for fl in flowlines:
        thick = np.asarray(fl.thick, dtype=float)
        active = thick > 1e-6
        area = np.asarray(fl.widths_m, dtype=float) * float(fl.dx_meter)
        volume = np.asarray(fl.section, dtype=float) * float(fl.dx_meter)
        core_area_parts.append(area[active])
        core_volume_parts.append(volume[active])
        core_surface_parts.append(np.asarray(fl.surface_h, dtype=float)[active])
        ids = np.flatnonzero(active)
        if ids.size > 1:
            consecutive = ids[1:] == ids[:-1] + 1
            delta = np.diff(np.asarray(fl.surface_h, dtype=float)[ids])
            core_reverse += int(np.sum(consecutive & (delta > 1e-7)))
        if getattr(fl, "flows_to", None) is None:
            root_count += 1
    core_area_values = np.concatenate(core_area_parts)
    core_volume_values = np.concatenate(core_volume_parts)
    core_surface_values = np.concatenate(core_surface_parts)
    core_area = float(np.sum(core_area_values))
    core_volume = float(np.sum(core_volume_values))
    residual_area = paired_area - core_area
    residual_volume = paired_volume - core_volume

    lo = 100.0 * np.floor(min(float(np.min(paired_surface)), float(np.min(core_surface_values))) / 100.0)
    hi = 100.0 * np.ceil(max(float(np.max(paired_surface)), float(np.max(core_surface_values))) / 100.0)
    edges = np.arange(lo, hi + 100.0, 100.0)
    if edges.size < 2:
        raise RuntimeError("insufficient-hypsometry-bins")
    n = edges.size - 1
    paired_bin = np.clip(np.digitize(paired_surface, edges) - 1, 0, n - 1)
    core_bin = np.clip(np.digitize(core_surface_values, edges) - 1, 0, n - 1)
    paired_area_bin = np.bincount(paired_bin, weights=np.full(paired_bin.size, cell_area), minlength=n)
    core_area_bin = np.bincount(core_bin, weights=core_area_values, minlength=n)
    h_min = np.zeros(n)
    h_max = np.zeros(n)
    supported = np.zeros(n, dtype=bool)
    for j in range(n):
        values = paired_h[paired_bin == j]
        if values.size:
            supported[j] = True
            h_min[j] = float(np.min(values))
            h_max[j] = float(np.max(values))

    # x = [A_0..A_n-1, V_0..V_n-1, t]
    m = 2 * n + 1
    c = np.zeros(m)
    c[-1] = 1.0
    a_eq = np.zeros((2, m))
    a_eq[0, :n] = 1.0
    a_eq[1, n : 2 * n] = 1.0
    b_eq = np.array([residual_area, residual_volume])
    a_ub = []
    b_ub = []
    for j in range(n):
        # V_j <= hmax_j A_j
        row = np.zeros(m)
        row[n + j] = 1.0
        row[j] = -h_max[j]
        a_ub.append(row)
        b_ub.append(0.0)
        # V_j >= hmin_j A_j
        row = np.zeros(m)
        row[j] = h_min[j]
        row[n + j] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)
    paired_cum = np.cumsum(paired_area_bin)
    core_cum = np.cumsum(core_area_bin)
    for j in range(n):
        row = np.zeros(m)
        row[: j + 1] = 1.0
        row[-1] = -paired_area
        a_ub.append(row)
        b_ub.append(float(paired_cum[j] - core_cum[j]))
        row = np.zeros(m)
        row[: j + 1] = -1.0
        row[-1] = -paired_area
        a_ub.append(row)
        b_ub.append(float(core_cum[j] - paired_cum[j]))
    bounds = []
    for j in range(n):
        bounds.append((0.0, float(paired_area_bin[j]) if supported[j] else 0.0))
    bounds.extend((0.0, None) for _ in range(n))
    bounds.append((0.0, 1.0))

    preconditions = {
        "paired_matches_formal_area": abs(paired_area - float(formal["paired_area_m2"])) <= 1e-6,
        "paired_matches_formal_volume": abs(paired_volume - float(formal["paired_volume_m3"])) <= 1e-3,
        "core_matches_formal_area": abs(core_area - float(formal["model_area_m2"])) <= 1e-5,
        "core_matches_formal_volume": abs(core_volume - float(formal["model_volume_m3"])) <= 1e-3,
        "residual_area_positive": residual_area > 0.0,
        "residual_volume_positive": residual_volume > 0.0,
        "core_active_reverse_count_zero": core_reverse == 0,
        "unique_root": root_count == 1,
    }
    if not all(preconditions.values()):
        result = None
        decision = "closed-at-input-contract"
    else:
        result = linprog(
            c,
            A_ub=np.asarray(a_ub),
            b_ub=np.asarray(b_ub),
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        decision = "static-feasible-materialization-not-yet-run" if result.success else "closed-mathematically-infeasible"

    if result is not None and result.success:
        reservoir_area_bin = result.x[:n]
        reservoir_volume_bin = result.x[n : 2 * n]
        combined_area_bin = core_area_bin + reservoir_area_bin
        cdf = float(np.max(np.abs(np.cumsum(combined_area_bin) / paired_area - paired_cum / paired_area)))
        rows = pd.DataFrame({
            "bin_lower_m": edges[:-1],
            "bin_upper_m": edges[1:],
            "paired_area_m2": paired_area_bin,
            "core_area_m2": core_area_bin,
            "reservoir_area_m2": reservoir_area_bin,
            "reservoir_volume_m3": reservoir_volume_bin,
            "reservoir_mean_thickness_m": np.divide(
                reservoir_volume_bin,
                reservoir_area_bin,
                out=np.zeros_like(reservoir_volume_bin),
                where=reservoir_area_bin > 0,
            ),
            "paired_thickness_min_m": h_min,
            "paired_thickness_max_m": h_max,
            "combined_cdf": np.cumsum(combined_area_bin) / paired_area,
            "paired_cdf": paired_cum / paired_area,
        })
        rows.to_csv(OUTPUT / "reservoir_by_100m_bin.csv", index=False)
        gate_results = {
            **preconditions,
            "linear_program_success": True,
            "area_closure": abs(float(np.sum(reservoir_area_bin)) - residual_area) <= 1e-5,
            "volume_closure": abs(float(np.sum(reservoir_volume_bin)) - residual_volume) <= 1e-3,
            "combined_hypsometry_within_2pct": cdf <= 0.02,
            "root_length_within_10pct": abs(float(formal["model_root_length_m"]) - float(config["registered_lia_length_m"])) / float(config["registered_lia_length_m"]) <= 0.10,
            "free_tail_at_least_5km": float(formal["model_free_tail_m"]) >= 5000.0,
            "root_terminal_zero": float(formal["root_terminal_thickness_m"]) <= 10.0,
        }
        status = "static-feasibility-pass" if all(gate_results.values()) else "static-feasibility-fail"
    else:
        cdf = None
        gate_results = preconditions
        status = "static-feasibility-fail"

    summary = {
        "schema": "lia-191-negative-flux2342-hypsometric-residual-reservoir-v1",
        "status": status,
        "decision": decision if status == "static-feasibility-pass" else "closed-before-materialization",
        "system_id": 2342,
        "mechanism": "unchanged-standard-dynamic-core-plus-passive-hypsometric-residual-reservoir",
        "dynamic_core_mutated": False,
        "reservoir_is_aggregate_not_a_spatial_cell_assignment": True,
        "paired_positive_threshold_m": 1e-6,
        "cell_area_m2": cell_area,
        "paired_area_m2": paired_area,
        "paired_volume_m3": paired_volume,
        "core_area_m2": core_area,
        "core_volume_m3": core_volume,
        "residual_area_m2": residual_area,
        "residual_volume_m3": residual_volume,
        "residual_mean_thickness_m": residual_volume / residual_area,
        "optimized_combined_hypsometry_cdf_error_100m": cdf,
        "gate_results": gate_results,
        "solver_status": None if result is None else int(result.status),
        "solver_message": None if result is None else str(result.message),
        "oggm_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "promotion_run": False,
        "provenance": {
            "paired": str(PAIRED.resolve()),
            "paired_sha256": sha256(PAIRED),
            "formal": str(FORMAL.resolve()),
            "formal_sha256": sha256(FORMAL),
            "flowlines": str(FLOWLINES.resolve()),
            "flowlines_sha256": sha256(FLOWLINES),
            "config": str(CONFIG.resolve()),
            "config_sha256": sha256(CONFIG),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

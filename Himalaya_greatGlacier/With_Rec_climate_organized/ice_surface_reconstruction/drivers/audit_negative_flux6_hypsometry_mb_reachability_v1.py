"""Rigorous initial net-MB reachability bound under the frozen 2% CDF gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from oggm import cfg, utils
from oggm.core.massbalance import MonthlyTIModel

from ..paths import resolve
from .audit_branch_rigid_vertical_datum_hold3_v1 import sha256


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux6_hypsometry_mb_reachability_v1"
ROOTS = {890: "w456inv890v1", 988: "w427inv988v1", 1026: "w337inv1026v1", 1743: "w583inv1743v1", 2689: "w586inv2689v1", 2793: "w555inv2793v1"}
CDF_EPSILON = 0.02


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    cfg.initialize(logging_level="WARNING"); cfg.PARAMS["use_tar_shapefiles"] = False
    rows = []
    for system_id, root_name in ROOTS.items():
        root = PROJECT / root_name; config = json.loads((root / "config_snapshot.json").read_text(encoding="utf-8"))
        paired = resolve(config["paired_grid"])
        with xr.open_dataset(paired) as ds:
            h = np.asarray(ds.thickness_m.isel(member=0), dtype=float)
            s = np.asarray(ds.surface_elevation_m.isel(member=0), dtype=float)
            valid = np.asarray(ds.valid_mask.isel(member=0), dtype=bool); lia = np.asarray(ds.lia_mask, dtype=bool)
            x = np.asarray(ds.x, dtype=float); y = np.asarray(ds.y, dtype=float)
        positive = valid & lia & np.isfinite(h) & np.isfinite(s) & (h > 1e-6)
        surfaces = s[positive]; cell_area = float(abs(np.median(np.diff(x))) * abs(np.median(np.diff(y))))
        area = float(len(surfaces) * cell_area); volume = float(np.sum(h[positive]) * cell_area)
        gdir = utils.GlacierDirectory(config["synthetic_rgi_id"], base_dir=root / "synthetic_gdir", reset=False)
        stage2 = config["stage2"]; cfg.PARAMS["temp_melt"] = float(stage2["temp_melt_c"])
        model = MonthlyTIModel(gdir, filename="gcm_data", input_filesuffix="_batch1_1441_1470_prcp_off",
            melt_f=float(stage2["melt_f"]), prcp_fac=float(stage2["prcp_fac"]),
            temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
            ys=1441, ye=1470, repeat=True, check_calib_params=False)
        annual_depth = np.mean(np.vstack([
            np.asarray(model.get_annual_mb(surfaces, year=float(year)), dtype=float) * cfg.SEC_IN_YEAR
            for year in range(30)
        ]), axis=0)
        order = np.argsort(surfaces, kind="stable"); z = surfaces[order]; g = annual_depth[order]
        unique_z, first, counts = np.unique(z, return_index=True, return_counts=True)
        unique_g = np.add.reduceat(g, first) / counts
        total_variation = float(np.sum(np.abs(np.diff(unique_g))))
        mb_range = float(np.max(unique_g) - np.min(unique_g))
        mean_depth = float(np.mean(annual_depth)); bound = CDF_EPSILON * total_variation
        lower = mean_depth - bound; upper = mean_depth + bound
        excludes_zero = bool(lower > 0 or upper < 0)
        monotone_violations = int(np.count_nonzero(np.diff(unique_g) < -1e-12))
        rows.append({
            "system_id": system_id, "positive_cell_count": len(surfaces), "area_m2": area, "volume_m3": volume,
            "mean_annual_mb_depth_m_ice": mean_depth, "minimum_mb_depth_m_ice": float(np.min(unique_g)),
            "maximum_mb_depth_m_ice": float(np.max(unique_g)), "mb_depth_range_m_ice": mb_range,
            "mb_function_total_variation_m_ice": total_variation, "cdf_epsilon": CDF_EPSILON,
            "maximum_expectation_shift_bound_m_ice": bound, "reachable_mean_mb_lower_m_ice": lower,
            "reachable_mean_mb_upper_m_ice": upper, "zero_excluded_by_bounded_variation_ks_bound": excludes_zero,
            "sorted_mb_monotonicity_violation_count": monotone_violations,
            "mean_annual_mb_m3": mean_depth * area, "maximum_absolute_reachable_shift_m3": bound * area,
            "paired_grid_sha256": sha256(paired),
        })
    table = pd.DataFrame(rows).sort_values("system_id")
    excluded = table.loc[table.zero_excluded_by_bounded_variation_ks_bound, "system_id"].astype(int).tolist()
    summary = {
        "schema": "lia-191-negative-flux6-hypsometry-mb-reachability-v1",
        "status": "all-six-initial-exact-steady-excluded" if len(excluded) == 6 else "mixed-reachability",
        "system_ids": sorted(ROOTS), "cdf_gate": CDF_EPSILON,
        "mathematical_bound": "For equal-area elevation distributions with KS/CDF distance <= epsilon, the change in expectation of any bounded-variation MB(z) is <= epsilon*TV(MB).",
        "zero_excluded_system_ids": excluded,
        "interpretation": "Exclusion proves no exact initial zero-net-MB paired-state representation within the 2% hypsometry gate under frozen climate/Stage2. It does not by itself prove every nonlinear 50-year transient fails.",
        "geometry_materialized": False, "oggm_dynamics_run": False, "historical_run": False,
        "next_action": "If all six exclude zero, close representation-only retries; reopening requires a changed climate/MB contract, relaxed hypsometry gate, or non-steady LIA assumption.",
    }
    OUTPUT.mkdir(parents=True); table.to_csv(OUTPUT / "systems.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()

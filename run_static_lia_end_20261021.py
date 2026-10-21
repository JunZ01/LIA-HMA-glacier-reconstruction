"""Checkpoint 2 of the same-method static route: the LIA end and the volume change.

Runs the PRODUCTION LIA system-grid driver (lia_3d_system_grid_pilot_v1.run)
unchanged except for two inputs, so both ends use the same static method:
  * extension operator: the driver's module-level ``harmonic_extend_thickness``
    is replaced, for this process only, by a signature-compatible wrapper
    around ``thinplate_extend_thickness`` (operator T);
  * tau: the unified-bed profile tau of checkpoint 1 (static_modern_end.csv),
    written as a separate modern_tau_calibration_v0.csv in a NEW subdirectory.
Bed scenario is restricted to farinotti2019 (the only bed calibrated in
checkpoint 1). Output goes to a new output_subdir; production outputs are not
touched. Then the same-method volume change is computed per system:
    dV = V_LIA(T) - V_modern(T),   dV% = dV / V_LIA(T)
with the old static result (legacy LIA harmonic vs modern reference) and the
mean-thickness change reported beside it.

Checkpoint-2 gates are frozen in GATES before running; the human decides
whether to adopt them and whether this step is authorised.

Usage (repo root, oggm_env2; use the SAME config and external flowline root
as the run that produced the legacy LIA netCDFs listed in pilot_systems.csv):
    python run_static_lia_end_20261021.py --config CONFIG.json \
        --pilot-csv .../selection/pilot_systems.csv \
        --modern-end-csv .../modern_end/static_modern_end.csv \
        --pairing-csv .../legacy191_source_pairing_diagnostic.csv \
        --output-subdir static_samemethod_lia_end_20261021_v1 \
        [--external-flowline-root DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized"))
from ice_surface_reconstruction.core import thinplate_extend_thickness  # noqa: E402

BED = "farinotti2019"
GATES = {
    "technical_complete_share_min": 0.8,      # of ALL pilot systems, LIA end complete
    "volume_loss_share_min": 0.8,             # of complete systems, V_LIA(T) > V_modern(T)
}


def thinplate_as_harmonic(mask, centerline_rows, centerline_cols, centerline_thickness_m, *,
                          centerline_surface_m=None, duplicate_tolerance_m=1e-3,
                          boundary_tolerance_m=1e-3, **_ignored):
    """Signature-compatible stand-in for harmonic_extend_thickness."""
    try:
        thickness, meta = thinplate_extend_thickness(
            mask, centerline_rows, centerline_cols, centerline_thickness_m,
            centerline_surface_m=centerline_surface_m,
            duplicate_tolerance_m=duplicate_tolerance_m,
            boundary_tolerance_m=boundary_tolerance_m)
    except ValueError as error:
        failed = np.full(np.shape(mask), np.nan)
        failed[~np.asarray(mask, bool)] = 0.0
        return failed, {"status": "operator-T-rejected", "failure_reason": str(error),
                        "iterations": 0, "method": "thin-plate", "maximum_laplace_residual_m": np.nan}
    return thickness, {**meta, "iterations": 0, "method": "thin-plate",
                       "maximum_laplace_residual_m": np.nan}


def tau_table(modern_end: pd.DataFrame) -> pd.DataFrame:
    solved = modern_end[modern_end.status == "solved"]
    return pd.DataFrame({
        "rgi_id": solved.rgi_id.astype(str), "bed_scenario": BED,
        "status": "unified-bed-profile-tau-checkpoint1",
        "tau_kpa": solved.profile_tau_kpa.astype(float),
        "kappa_kpa": solved.profile_tau_kpa.astype(float) / 0.8,
        "lia_unique_id": solved.lia_unique_id.astype(int),
    })


def volume_change(system_results: pd.DataFrame, modern_end: pd.DataFrame,
                  pilot: pd.DataFrame, pairing: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    lia = system_results[system_results.bed_scenario.astype(str) == BED].copy()
    lia["lia_complete"] = lia.status.astype(str).isin(["success", "success-with-documented-exclusion"])
    lia = lia.rename(columns={"volume_km3_finite_components": "V_lia_T_km3",
                              "finite_lia_cell_count": "lia_cells", "system_grid_dx_m": "lia_dx_m"})
    keep = ["lia_unique_id", "status", "lia_complete", "V_lia_T_km3", "lia_cells", "lia_dx_m",
            "missing_component_area_fraction"]
    t = pilot[["lia_unique_id", "region", "area_km2"]].merge(
        lia[[c for c in keep if c in lia.columns]], on="lia_unique_id", how="left")
    m = modern_end[["lia_unique_id", "status", "static_volume_thinplate_km3", "reference_volume_km3"]].rename(
        columns={"status": "modern_status", "static_volume_thinplate_km3": "V_mod_T_km3",
                 "reference_volume_km3": "V_mod_reference_km3"})
    t = t.merge(m, on="lia_unique_id", how="left")
    t["lia_complete"] = t.lia_complete.fillna(False).astype(bool)
    ok = t.lia_complete & (t.modern_status == "solved")
    t["dV_T_km3"] = np.where(ok, t.V_mod_T_km3 - t.V_lia_T_km3, np.nan)
    t["dV_T_percent"] = 100 * t.dV_T_km3 / t.V_lia_T_km3
    t["lia_area_km2"] = t.lia_cells * t.lia_dx_m ** 2 / 1e6
    t["lia_mean_H_T_m"] = 1000 * t.V_lia_T_km3 / t.lia_area_km2
    t["volume_loss"] = t.dV_T_km3 < 0
    if pairing is not None:
        p = pairing[["lia_unique_id", "lia_saved_array_volume_km3", "modern_reference_native_sum_km3"]]
        t = t.merge(p, on="lia_unique_id", how="left")
        t["dV_old_static_km3"] = t.modern_reference_native_sum_km3 - t.lia_saved_array_volume_km3
        t["dV_old_static_percent"] = 100 * t.dV_old_static_km3 / t.lia_saved_array_volume_km3
    complete = t[ok]
    n_all = len(t)
    summary = {
        "systems": n_all,
        "complete": int(ok.sum()),
        "technical_complete_share": float(ok.sum() / max(n_all, 1)),
        "volume_loss_share_of_complete": float(complete.volume_loss.mean()) if len(complete) else None,
        "sum_V_lia_T_km3": float(complete.V_lia_T_km3.sum()),
        "sum_V_mod_T_km3": float(complete.V_mod_T_km3.sum()),
        "sum_dV_T_km3": float(complete.dV_T_km3.sum()),
        "sum_dV_T_percent_of_lia": float(100 * complete.dV_T_km3.sum() / complete.V_lia_T_km3.sum()) if len(complete) else None,
        "median_dV_T_percent": float(complete.dV_T_percent.median()) if len(complete) else None,
        "systems_gaining_volume": [int(u) for u in complete.loc[~complete.volume_loss, "lia_unique_id"]],
    }
    if pairing is not None and len(complete):
        summary["sum_dV_old_static_percent_same_systems"] = float(
            100 * complete.dV_old_static_km3.sum() / complete.lia_saved_array_volume_km3.sum())
    summary["gate_technical"] = summary["technical_complete_share"] >= GATES["technical_complete_share_min"]
    summary["gate_volume_loss"] = (summary["volume_loss_share_of_complete"] or 0) >= GATES["volume_loss_share_min"]
    summary["checkpoint2_pass"] = bool(summary["gate_technical"] and summary["gate_volume_loss"])
    summary["gates"] = GATES
    return t, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--pilot-csv", required=True, type=Path)
    ap.add_argument("--modern-end-csv", required=True, type=Path)
    ap.add_argument("--pairing-csv", type=Path)
    ap.add_argument("--output-subdir", required=True)
    ap.add_argument("--external-flowline-root", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from ice_surface_reconstruction.paths import resolve
    config = copy.deepcopy(json.loads(args.config.read_text(encoding="utf-8")))
    pilot = pd.read_csv(args.pilot_csv, encoding="utf-8-sig")
    modern_end = pd.read_csv(args.modern_end_csv)
    taus = tau_table(modern_end)
    root = resolve(config["output_dir"])
    out_dir = root / args.output_subdir
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; outputs are immutable, choose a new --output-subdir")
    tau_subdir = f"{args.output_subdir}/unified_bed_tau"
    config["modern_tau_calibration"]["output_subdir"] = tau_subdir
    config.setdefault("lia_3d_system_grid_pilot_v1", {})["output_subdir"] = args.output_subdir
    config["bed_scenarios"] = [BED]
    selected = set(pilot.rgi_id.astype(str)) & set(taus.rgi_id)
    plan = {"output_dir": str(out_dir), "systems": len(selected),
            "missing_tau": sorted(set(pilot.rgi_id.astype(str)) - set(taus.rgi_id)),
            "legacy_lia_netcdf_dirs": sorted({str(Path(p).parent) for p in pilot.lia_netcdf}),
            "external_flowline_root": str(args.external_flowline_root) if args.external_flowline_root else None,
            "operator": "thin-plate (operator T)", "bed": BED, "gates": GATES}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    (root / tau_subdir).mkdir(parents=True, exist_ok=False)
    taus.to_csv(root / tau_subdir / "modern_tau_calibration_v0.csv", index=False)

    from ice_surface_reconstruction.drivers import lia_3d_system_grid_pilot_v1 as driver
    driver.harmonic_extend_thickness = thinplate_as_harmonic  # this process only
    started = time.perf_counter()
    driver.run(config, selected, args.external_flowline_root.resolve() if args.external_flowline_root else None)
    results = pd.read_csv(out_dir / "lia_system_grid_results_v1.csv")
    pairing = pd.read_csv(args.pairing_csv, encoding="utf-8-sig") if args.pairing_csv else None
    table, summary = volume_change(results, modern_end, pilot, pairing)
    summary.update({"elapsed_seconds": time.perf_counter() - started, "plan": plan,
                    "scope": "same-method static volume change on the frozen 22-system pilot; model-to-model comparisons only; no dynamic run"})
    table.to_csv(out_dir / "static_samemethod_volume_change.csv", index=False)
    (out_dir / "static_samemethod_volume_change_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "plan"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Checkpoint 3, part 3: same-method static LIA end for all legacy systems.

Identical method to checkpoint 2 (run_static_lia_end_20261021.py): production
LIA system-grid driver, process-local thin-plate substitution, farinotti2019,
new output and grid-cache subdirectories. The tau of each system is the main
member's unified-bed profile tau from run_static_modern_members_20261022.py
(system_tau_for_lia_end.csv), so both ends use the same tau.

Volume change per system: dV = V_modern(T) - V_LIA(T); negative = loss.
Gates, fixed before running (GATES), over ALL systems of the population:
  * at least 80 % of all systems have both ends complete;
  * among complete systems WITHOUT the QC flag, at least 80 % lose volume.
QC-flagged systems (system tau <= 45 kPa) are reported separately and are not
in the headline totals; they are not deleted. Regional totals are given for
all complete systems and for complete unflagged systems.

Usage (repo root, oggm_env2; same config/external root as checkpoint 2):
    python run_static_lia_end_191_20261022.py --config CONFIG.json \
        --npz-dir MEMBER_NPZ_DIR --modern-dir MODERN_OUT_DIR --pairing-csv PAIRING.csv \
        --output-subdir static_samemethod_ck3_lia_end_20261022_v1 \
        --external-flowline-root DIR [--ck2-csv CK2.csv] [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ck2", ROOT / "run_static_lia_end_20261021.py")
ck2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck2)

BED = "farinotti2019"
GATES = {"technical_complete_share_min": 0.8, "volume_loss_share_unflagged_min": 0.8}


def volume_change(population: pd.DataFrame, modern: pd.DataFrame, lia_results: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    lia = lia_results[lia_results.bed_scenario.astype(str) == BED].copy()
    lia["lia_complete"] = lia.status.astype(str).isin(["success", "success-with-documented-exclusion"])
    lia = lia.rename(columns={"volume_km3_finite_components": "V_lia_T_km3", "status": "lia_status",
                              "finite_lia_cell_count": "lia_cells", "system_grid_dx_m": "lia_dx_m"})
    for col in ["V_lia_T_km3", "lia_cells", "lia_dx_m"]:
        if col not in lia:
            lia[col] = np.nan
    t = population[["lia_unique_id", "region", "modern_member_count", "lia_saved_array_volume_km3",
                    "modern_reference_native_sum_km3"]].merge(
        modern.drop(columns=["region", "modern_member_count", "modern_reference_native_sum_km3"], errors="ignore"),
        on="lia_unique_id", how="left", validate="one_to_one")
    t = t.merge(lia[["lia_unique_id", "lia_status", "lia_complete", "V_lia_T_km3", "lia_cells", "lia_dx_m"]],
                on="lia_unique_id", how="left", validate="one_to_one")
    t["lia_complete"] = t.lia_complete.fillna(False).astype(bool)
    t["qc_flowline_solve_unreliable"] = t.qc_flowline_solve_unreliable.fillna(False).astype(bool)
    ok = (t.lia_complete & t.modern_status.eq("complete")
          & np.isfinite(t.V_lia_T_km3) & np.isfinite(t.V_mod_T_km3))
    t["both_ends_complete"] = ok
    t["dV_T_km3"] = np.where(ok, t.V_mod_T_km3 - t.V_lia_T_km3, np.nan)
    t["dV_T_percent"] = 100 * t.dV_T_km3 / t.V_lia_T_km3
    t["lia_area_km2"] = t.lia_cells * t.lia_dx_m ** 2 / 1e6
    t["dA_percent"] = 100 * (t.modern_area_solved_km2 - t.lia_area_km2) / t.lia_area_km2
    t["lia_mean_H_m"] = 1000 * t.V_lia_T_km3 / t.lia_area_km2
    t["modern_mean_H_m"] = 1000 * t.V_mod_T_km3 / t.modern_area_solved_km2
    for col in ["dA_percent", "lia_mean_H_m", "modern_mean_H_m"]:
        t.loc[~ok, col] = np.nan
    t["volume_loss"] = t.dV_T_km3 < 0
    t["dV_old_static_percent"] = 100 * (t.modern_reference_native_sum_km3 - t.lia_saved_array_volume_km3) / t.lia_saved_array_volume_km3
    complete = t[ok]
    clean = complete[~complete.qc_flowline_solve_unreliable]

    def totals(frame):
        if not len(frame):
            return {"systems": 0}
        return {"systems": int(len(frame)), "sum_V_lia_km3": float(frame.V_lia_T_km3.sum()),
                "sum_V_mod_km3": float(frame.V_mod_T_km3.sum()), "sum_dV_km3": float(frame.dV_T_km3.sum()),
                "sum_dV_percent": float(100 * frame.dV_T_km3.sum() / frame.V_lia_T_km3.sum()),
                "median_dV_percent": float(frame.dV_T_percent.median()),
                "loss_share": float(frame.volume_loss.mean())}

    n = len(t)
    summary = {
        "systems": n, "both_ends_complete": int(ok.sum()),
        "technical_complete_share": float(ok.sum() / max(n, 1)),
        "qc_flagged_complete": complete.loc[complete.qc_flowline_solve_unreliable, "lia_unique_id"].astype(int).tolist(),
        "all_complete": totals(complete), "complete_unflagged": totals(clean),
        "by_region_unflagged": {r: totals(g) for r, g in clean.groupby("region")},
        "by_member_type_unflagged": {k: totals(g) for k, g in clean.groupby(clean.modern_member_count.gt(1).map({True: "multi", False: "single"}))},
        "gaining_unflagged": clean.loc[~clean.volume_loss, "lia_unique_id"].astype(int).tolist(),
        "incomplete": t.loc[~ok, ["lia_unique_id", "modern_status", "lia_status"]].astype(str).to_dict("records"),
    }
    summary["gate_technical"] = summary["technical_complete_share"] >= GATES["technical_complete_share_min"]
    summary["gate_volume_loss"] = summary["complete_unflagged"].get("loss_share", 0) >= GATES["volume_loss_share_unflagged_min"]
    summary["checkpoint3_pass"] = bool(summary["gate_technical"] and summary["gate_volume_loss"])
    summary["gates"] = GATES
    return t, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--npz-dir", required=True, type=Path, help="member export dir (systems_population.csv)")
    ap.add_argument("--modern-dir", required=True, type=Path)
    ap.add_argument("--pairing-csv", required=True, type=Path)
    ap.add_argument("--output-subdir", required=True)
    ap.add_argument("--external-flowline-root", type=Path)
    ap.add_argument("--ck2-csv", type=Path, help="checkpoint-2 static_samemethod_volume_change.csv for a consistency check")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from ice_surface_reconstruction.paths import resolve
    config = copy.deepcopy(json.loads(args.config.read_text(encoding="utf-8")))
    population = pd.read_csv(args.npz_dir / "systems_population.csv")
    pairing = pd.read_csv(args.pairing_csv, encoding="utf-8-sig")
    if set(population.lia_unique_id) != set(pairing.lia_unique_id) or population.lia_unique_id.duplicated().any():
        raise ValueError("systems_population.csv must list exactly the pairing systems once each")
    modern = pd.read_csv(args.modern_dir / "modern_systems.csv")
    taus = pd.read_csv(args.modern_dir / "system_tau_for_lia_end.csv")
    if taus.rgi_id.duplicated().any() or not np.isfinite(taus.tau_kpa).all() or taus.empty:
        raise ValueError("system tau table must be non-empty, finite and unique per main rgi_id")
    root = resolve(config["output_dir"])
    out_dir = root / args.output_subdir
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; outputs are immutable, choose a new --output-subdir")
    tau_subdir = f"{args.output_subdir}/unified_bed_tau"
    config["modern_tau_calibration"]["output_subdir"] = tau_subdir
    config.setdefault("lia_3d_system_grid_pilot_v1", {})["output_subdir"] = args.output_subdir
    config.setdefault("lia_system_grid", {})["grid_output_subdir"] = f"{args.output_subdir}/system_grids"
    config["bed_scenarios"] = [BED]
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    selected = set(taus.rgi_id.astype(str)) & set(selection.main_rgi_id.astype(str))
    plan = {"output_dir": str(out_dir), "population_systems": int(len(population)),
            "systems_with_modern_tau": int(len(taus)), "systems_to_run": len(selected),
            "tau_rgi_not_in_driver_selection": sorted(set(taus.rgi_id.astype(str)) - selected),
            "external_flowline_root": str(args.external_flowline_root) if args.external_flowline_root else None,
            "grid_output_subdir": config["lia_system_grid"]["grid_output_subdir"],
            "operator": "thin-plate (operator T)", "bed": BED, "gates": GATES}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not selected:
        raise ValueError("empty selection; never launch the driver without an explicit selection")
    if args.dry_run:
        return
    (root / tau_subdir).mkdir(parents=True, exist_ok=False)
    taus.to_csv(root / tau_subdir / "modern_tau_calibration_v0.csv", index=False)
    from ice_surface_reconstruction.drivers import lia_3d_system_grid_pilot_v1 as driver
    driver.harmonic_extend_thickness = ck2.thinplate_as_harmonic  # this process only
    started = time.perf_counter()
    driver.run(config, selected, args.external_flowline_root.resolve() if args.external_flowline_root else None)
    lia_results = pd.read_csv(out_dir / "lia_system_grid_results_v1.csv")
    table, summary = volume_change(population, modern, lia_results)
    if args.ck2_csv:
        c2 = pd.read_csv(args.ck2_csv)[["lia_unique_id", "dV_T_km3"]].rename(columns={"dV_T_km3": "dV_ck2_km3"})
        cmp = table.merge(c2, on="lia_unique_id", how="inner")
        summary["ck2_overlap_systems"] = int(len(cmp))
        summary["ck2_max_abs_dV_difference_km3"] = float((cmp.dV_T_km3 - cmp.dV_ck2_km3).abs().max()) if len(cmp) else None
    summary.update({"elapsed_seconds": time.perf_counter() - started, "plan": plan,
                    "scope": "same-method static volume change, all legacy systems; model estimates; no dynamic run"})
    table.to_csv(out_dir / "static_samemethod_ck3_volume_change.csv", index=False)
    (out_dir / "static_samemethod_ck3_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("plan", "incomplete")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

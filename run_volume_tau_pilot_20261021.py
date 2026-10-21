"""Stratified pilot of the volume-tau route (fact 1499), one batch.

For every fullchain_inputs_<uid>.npz in NPZ_DIR:
  0. harness: reproduce the archived legacy tau (production _evaluate);
  1. full modern chain (unified bed, production network, H0 outlet, operator A)
     on the 40-260 kPa grid: flowline loss and A volume per tau;
  2. profile tau: legacy loss form, bounded refinement around the grid best;
  3. volume tau: Brent root of the A volume bias, verified by a fresh re-solve;
  4. gates, fixed before the pilot (see GATES), and identifiability flag.
A failing system is recorded with its reason and the batch continues.

Calibration only: volume closure is the target and cannot validate the model.
No LIA reconstruction, climate or dynamic run.

Usage (repo root):  python run_volume_tau_pilot_20261021.py NPZ_DIR OUT_DIR
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("replay", ROOT / "replay_modern_fullchain_20261021.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

OUTLET = "H0"
TAU_GRID_KPA = replay.TAU_GRID_KPA
GATES = {
    "closure_abs_bias_percent_max": 0.1,
    "flowline_rmse_ratio_max": 1.25,
    "surface_rmse_increase_m_max": 1.0,
    "tau_strictly_inside_kpa": [TAU_GRID_KPA[0], TAU_GRID_KPA[-1]],
}
PILOT_PASS_SHARE = 0.8  # decision rule proposed for the route, applied by the human


def run_system(npz_dir: Path, uid: int) -> tuple[dict, list[dict]]:
    d = replay.load(npz_dir, uid)
    row = {"lia_unique_id": uid, "region": str(d["region"]), "rgi_id": str(d["rgi_id"])}
    archived = float(d["legacy_tau_kpa"])
    legacy = replay.legacy_calibration(d)
    row["harness_legacy_tau_reproduced"] = abs(legacy["tau_kpa"] - archived) <= replay.LEGACY_TOLERANCE_KPA
    if not row["harness_legacy_tau_reproduced"]:
        return {**row, "status": "harness-failed", "archived_tau_kpa": archived,
                "replayed_tau_kpa": legacy["tau_kpa"]}, []
    chain = replay.ModernChain(d)
    cache: dict[float, dict] = {}

    def evaluate(tau: float, extend: bool = True) -> dict:
        key = round(float(tau), 6)
        hit = cache.get(key)
        if hit is not None and (not extend or "A_volume_bias_percent" in hit):
            return hit
        network, status = chain.solve(key, OUTLET)
        result = {"tau_kpa": key, **status}
        if network is not None:
            result.update(chain.flowline_fit(network))
            if extend:
                ext = chain.extend(network)
                ext.pop("_fields")
                result.update(ext)
        cache[key] = result
        return result

    curve = [evaluate(t) for t in TAU_GRID_KPA]
    curve_rows = [{"lia_unique_id": uid, **r} for r in curve]
    ok = [r for r in curve if r["status"] == "success" and np.isfinite(r.get("loss", np.nan))]
    if not ok:
        return {**row, "status": "network-failed-on-grid"}, curve_rows
    # profile tau
    best = min(ok, key=lambda r: r["loss"])["tau_kpa"]
    lo, hi = max(TAU_GRID_KPA[0], best - 20.0), min(TAU_GRID_KPA[-1], best + 20.0)

    def loss(t):
        r = evaluate(t, extend=False)
        return r["loss"] if np.isfinite(r.get("loss", np.nan)) else np.inf

    profile_tau = float(minimize_scalar(loss, bounds=(lo, hi), method="bounded",
                                        options={"xatol": 0.25}).x)
    profile = evaluate(profile_tau)
    losses = [(r["tau_kpa"], r["loss"]) for r in cache.values() if np.isfinite(r.get("loss", np.nan))]
    min_loss = min(v for _, v in losses)
    near = [t for t, v in losses if v <= min_loss * 1.05]
    weak = (max(near) - min(near)) >= 40.0
    # volume tau
    bias = np.array([r.get("A_volume_bias_percent", np.nan) for r in curve])
    taus = np.array([r["tau_kpa"] for r in curve])
    finite = np.isfinite(bias)
    crossings = np.flatnonzero(np.diff(np.sign(bias[finite])) != 0)
    row.update({"archived_tau_kpa": archived, "profile_tau_kpa": profile_tau,
                "profile_flowline_rmse_m": profile.get("flowline_rmse_m", np.nan),
                "profile_A_volume_bias_percent": profile.get("A_volume_bias_percent", np.nan),
                "profile_surface_rmse_m": profile.get("A_surface_rmse_vs_dem_m", np.nan),
                "profile_negative_cells_percent": profile.get("A_negative_raw_cells_percent", np.nan),
                "weak_identifiability": bool(weak),
                "near_optimal_tau_range_kpa": f"{min(near):.1f}-{max(near):.1f}",
                "reference_volume_km3": chain.ref_volume_km3})
    if len(crossings) == 0:
        return {**row, "status": "volume-root-outside-tau-bounds"}, curve_rows
    k = crossings[0]
    t_lo, t_hi = float(taus[finite][k]), float(taus[finite][k + 1])
    volume_tau = brentq(lambda t: evaluate(t)["A_volume_bias_percent"], t_lo, t_hi, xtol=0.01)
    fresh = replay.ModernChain(d)
    network, _ = fresh.solve(round(volume_tau, 6), OUTLET)
    ext = fresh.extend(network)
    ext.pop("_fields")
    verify = {**fresh.flowline_fit(network), **ext}
    row.update({"volume_tau_kpa": volume_tau,
                "closure_bias_percent": verify["A_volume_bias_percent"],
                "volume_flowline_rmse_m": verify["flowline_rmse_m"],
                "volume_flowline_bias_m": verify["flowline_bias_m"],
                "volume_surface_rmse_m": verify["A_surface_rmse_vs_dem_m"],
                "volume_thickness_rmse_m": verify["A_thickness_rmse_m"],
                "volume_negative_cells_percent": verify["A_negative_raw_cells_percent"],
                "tau_difference_volume_minus_profile_kpa": volume_tau - profile_tau,
                "status": "solved"})
    g = GATES
    row["gate_closure"] = abs(row["closure_bias_percent"]) <= g["closure_abs_bias_percent_max"]
    row["gate_flowline"] = row["volume_flowline_rmse_m"] <= g["flowline_rmse_ratio_max"] * row["profile_flowline_rmse_m"]
    row["gate_surface"] = row["volume_surface_rmse_m"] <= row["profile_surface_rmse_m"] + g["surface_rmse_increase_m_max"]
    row["gate_tau_inside"] = g["tau_strictly_inside_kpa"][0] < volume_tau < g["tau_strictly_inside_kpa"][1]
    row["pilot_pass"] = all(row[k] for k in ("gate_closure", "gate_flowline", "gate_surface", "gate_tau_inside"))
    return row, curve_rows


def main() -> None:
    npz_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    exported_uids = {int(p.stem.split("_")[-1]) for p in npz_dir.glob("fullchain_inputs_*.npz")}
    selection_path = npz_dir / "pilot_systems.csv"
    if selection_path.exists():
        selected = pd.read_csv(selection_path)
        if selected.lia_unique_id.duplicated().any():
            raise ValueError("duplicate system IDs in frozen pilot selection")
        uids = sorted(int(u) for u in selected.lia_unique_id)
        if exported_uids - set(uids):
            raise ValueError("NPZ directory contains systems outside the frozen pilot selection")
    else:
        uids = sorted(exported_uids)
    rows, curves = [], []
    for uid in uids:
        t0 = time.perf_counter()
        try:
            row, curve = run_system(npz_dir, uid)
        except Exception as error:  # recorded, never silently dropped
            row, curve = {"lia_unique_id": uid, "status": "error",
                          "error": f"{type(error).__name__}: {error}",
                          "traceback": traceback.format_exc()}, []
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        curves.extend(curve)
        pd.DataFrame(rows).to_csv(out_dir / "pilot_results.csv", index=False)
        pd.DataFrame(curves).to_csv(out_dir / "pilot_tau_curves.csv", index=False)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                          if k in ("lia_unique_id", "status", "profile_tau_kpa", "volume_tau_kpa",
                                   "pilot_pass", "weak_identifiability", "seconds")}), flush=True)
    table = pd.DataFrame(rows)
    solved = table[table.get("status", pd.Series(dtype=str)) == "solved"] if len(table) else table
    passed = int(solved["pilot_pass"].sum()) if len(solved) else 0
    summary = {
        "status": "completed", "elapsed_seconds": time.perf_counter() - started,
        "systems": len(table), "solved": int(len(solved)), "passed": passed,
        "selection_manifest_used": selection_path.exists(),
        "exported_systems": len(exported_uids),
        "missing_exports_counted_in_denominator": len(set(uids) - exported_uids),
        "pass_share_of_all_systems": passed / max(len(table), 1),
        "status_counts": table["status"].value_counts().to_dict() if len(table) else {},
        "weak_identifiability_systems": [int(u) for u in solved.loc[solved.weak_identifiability, "lia_unique_id"]] if len(solved) else [],
        "gates": GATES, "proposed_route_rule": f"proceed to the 191 static rerun only if at least {PILOT_PASS_SHARE:.0%} of ALL pilot systems pass; failures retained and explained",
        "scope": "modern calibration pilot; closure is the target, not validation; no LIA, climate or dynamic run",
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("systems", "solved", "passed", "pass_share_of_all_systems", "status_counts")}), flush=True)


if __name__ == "__main__":
    main()

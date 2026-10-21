"""Actual modern total-volume tau calibration on the three frozen controls.

Fact 1499 research route: tau is calibrated so that the full modern chain
(unified bed, production network, operator A, H0 outlet) closes the modern
reference volume; flowline-thickness tau is kept as a structural sensitivity.
This script solves the volume root (Brent, bracketed by the existing 20 kPa
curve), verifies closure by an independent re-solve at the root, and reports
whether flowline and surface errors worsen relative to flowline-fit tau.

Volume closure is a calibration target here, so it cannot validate the model.
Reuses ``ModernChain`` from replay_modern_fullchain_20261021.py unchanged.

Usage (repo root):
    python calibrate_volume_tau_20261021.py NPZ_DIR REPLAY_OUT_DIR OUT_DIR
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("replay", ROOT / "replay_modern_fullchain_20261021.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

OUTLET = "H0"
CLOSURE_TOLERANCE_PERCENT = 0.1
XTOL_KPA = 0.01


def run(npz_dir: Path, replay_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    curves = pd.read_csv(replay_dir / "tau_curves.csv")
    previous = pd.read_csv(replay_dir / "fullchain_summary.csv")
    rows = []
    for uid in replay.SYSTEMS:
        d = replay.load(npz_dir, uid)
        chain = replay.ModernChain(d)
        cache: dict[float, dict] = {}

        def evaluate(tau: float) -> dict:
            key = round(float(tau), 6)
            if key not in cache:
                network, status = chain.solve(key, OUTLET)
                if network is None:
                    raise RuntimeError(f"{uid} network failed at {key} kPa: {status}")
                ext = chain.extend(network)
                ext.pop("_fields")
                cache[key] = {**chain.flowline_fit(network), **ext}
            return cache[key]

        c = curves[(curves.lia_unique_id == uid) & (curves.outlet == OUTLET)].sort_values("tau_kpa")
        sign = np.sign(c.A_volume_bias_percent.to_numpy())
        k = int(np.flatnonzero(np.diff(sign) != 0)[0])
        lo, hi = float(c.tau_kpa.iloc[k]), float(c.tau_kpa.iloc[k + 1])
        interpolated = float(np.interp(0.0, c.A_volume_bias_percent.iloc[k:k + 2], c.tau_kpa.iloc[k:k + 2]))
        root = brentq(lambda t: evaluate(t)["A_volume_bias_percent"], lo, hi, xtol=XTOL_KPA)
        # independent verification: fresh chain, fresh solve at the root
        check = replay.ModernChain(d)
        network, _ = check.solve(round(root, 6), OUTLET)
        ext = check.extend(network)
        ext.pop("_fields")
        verify = {**check.flowline_fit(network), **ext}
        profile = previous[(previous.lia_unique_id == uid) & (previous.outlet == OUTLET)
                           & (previous.tau_source == "refit")].iloc[0]
        row = {
            "lia_unique_id": uid, "region": str(d["region"]),
            "bracket_kpa": f"{lo:g}-{hi:g}", "interpolated_tau_kpa": interpolated,
            "volume_tau_kpa": root, "volume_tau_minus_interpolated_kpa": root - interpolated,
            "closure_bias_percent": verify["A_volume_bias_percent"],
            "closure_passed": abs(verify["A_volume_bias_percent"]) <= CLOSURE_TOLERANCE_PERCENT,
            "profile_tau_kpa": float(profile.tau_kpa),
            "profile_tau_A_volume_bias_percent": float(profile.A_volume_bias_percent),
            "flowline_rmse_m_at_profile_tau": float(profile.flowline_rmse_m),
            "flowline_rmse_m_at_volume_tau": verify["flowline_rmse_m"],
            "flowline_bias_m_at_profile_tau": float(profile.flowline_bias_m),
            "flowline_bias_m_at_volume_tau": verify["flowline_bias_m"],
            "surface_rmse_m_at_profile_tau": float(profile.A_surface_rmse_vs_dem_m),
            "surface_rmse_m_at_volume_tau": verify["A_surface_rmse_vs_dem_m"],
            "surface_mean_error_m_at_volume_tau": verify["A_surface_mean_error_m"],
            "thickness_rmse_m_at_profile_tau": float(profile.A_thickness_rmse_m),
            "thickness_rmse_m_at_volume_tau": verify["A_thickness_rmse_m"],
            "negative_cells_percent_at_profile_tau": float(profile.A_negative_raw_cells_percent),
            "negative_cells_percent_at_volume_tau": verify["A_negative_raw_cells_percent"],
            "root_solves": len(cache),
        }
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                         ensure_ascii=False), flush=True)
    pd.DataFrame(rows).to_csv(out_dir / "volume_tau_calibration.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "status": "completed", "elapsed_seconds": time.perf_counter() - started,
        "outlet": OUTLET, "operator": "A", "closure_tolerance_percent": CLOSURE_TOLERANCE_PERCENT,
        "all_closed": all(r["closure_passed"] for r in rows),
        "scope": "modern calibration only; closure is the target, not validation; no LIA, climate or dynamic run",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

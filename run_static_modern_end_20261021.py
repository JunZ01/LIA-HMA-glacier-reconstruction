"""Checkpoint 1 of the same-method static route: the modern end.

Route (literature-aligned, Li 2023 PalaeoIce / Zhang et al. 2024): the LIA and
modern volumes are both produced by the same static pipeline -- perfect-
plasticity flowline network, then 2-D thickness interpolation with zero
thickness on the outline -- and differenced. The interpolator is the thin-plate
(minimum-curvature) operator T, the class used by ANUDEM/'Topo to Raster',
instead of the legacy harmonic operator.

This script runs the MODERN end on fullchain_inputs_<uid>.npz:
  0. harness: reproduce the archived legacy tau;
  1. profile tau on the unified bed (legacy loss, grid + bounded refinement);
  2. at that tau, one network solve, production constraints, then both the
     legacy harmonic operator and operator T;
  3. modern static volume / modern reference volume for both operators.
Checkpoint rule fixed before running (CHECKPOINT): a system passes when the
operator-T ratio lies in [0.7, 1.3], the accuracy range reported for VOLTA and
GlaRe against field volumes; the route continues to the LIA end only if at
least 80 % of ALL systems pass. The reference is an OGGM redistribution, so
this checks stability and plausibility, not truth. No LIA, climate or dynamic
run.

Usage (repo root):  python run_static_modern_end_20261021.py NPZ_DIR OUT_DIR
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
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("replay", ROOT / "replay_modern_fullchain_20261021.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)
from ice_surface_reconstruction.core import (  # noqa: E402  (path set by replay)
    harmonic_extend_thickness,
    thinplate_extend_thickness,
)

OUTLET = "H0"
CHECKPOINT = {"ratio_low": 0.7, "ratio_high": 1.3, "route_pass_share": 0.8}


def constraints(chain, network):
    """Identical filtering to ModernChain.extend (production collection, 100 m)."""
    rows, cols, thick, surf, _ = replay.collect_network_constraints(
        chain.inputs, chain.build, network, duplicate_surface_tolerance_m=chain.dup_tol,
        cross_branch_surface_tolerance_m=chain.cross_tol)
    inside = chain.mask[rows, cols]
    rows, cols, thick, surf = rows[inside], cols[inside], thick[inside], surf[inside]
    keep = ~chain.boundary[rows, cols] & (chain.boundary_distance[rows, cols] >= chain.exclusion_m)
    return rows[keep], cols[keep], thick[keep], surf[keep]


def run_system(npz_dir: Path, uid: int) -> dict:
    d = replay.load(npz_dir, uid)
    row = {"lia_unique_id": uid, "region": str(d["region"]), "rgi_id": str(d["rgi_id"])}
    archived = float(d["legacy_tau_kpa"])
    legacy = replay.legacy_calibration(d)
    if abs(legacy["tau_kpa"] - archived) > replay.LEGACY_TOLERANCE_KPA:
        return {**row, "status": "harness-failed"}
    chain = replay.ModernChain(d)
    cache = {}

    def loss(t):
        key = round(float(t), 6)
        if key not in cache:
            net, _ = chain.solve(key, OUTLET)
            cache[key] = chain.flowline_fit(net)["loss"] if net is not None else np.inf
        v = cache[key]
        return v if np.isfinite(v) else np.inf

    grid = {t: loss(t) for t in replay.TAU_GRID_KPA}
    finite = {t: v for t, v in grid.items() if np.isfinite(v)}
    if not finite:
        return {**row, "status": "network-failed-on-grid"}
    best = min(finite, key=finite.get)
    lo, hi = max(replay.TAU_GRID_KPA[0], best - 20.0), min(replay.TAU_GRID_KPA[-1], best + 20.0)
    tau = float(minimize_scalar(loss, bounds=(lo, hi), method="bounded", options={"xatol": 0.25}).x)
    network, _ = chain.solve(round(tau, 6), OUTLET)
    fit = chain.flowline_fit(network)
    rows, cols, thick, surf = constraints(chain, network)
    harm, hmeta = harmonic_extend_thickness(
        chain.mask, rows, cols, thick, centerline_surface_m=surf,
        duplicate_tolerance_m=chain.dup_tol, boundary_tolerance_m=1e-3,
        max_iterations=10000, tolerance_m=1e-3, method="conjugate-gradient", cg_relative_tolerance=1e-10)
    tp, tmeta = thinplate_extend_thickness(chain.mask, rows, cols, thick, centerline_surface_m=surf,
                                           duplicate_tolerance_m=chain.dup_tol)
    if hmeta["status"] != "converged" or tmeta["status"] != "converged":
        return {**row, "status": "extension-failed"}
    cell = chain.dx ** 2 / 1e9
    v_ref = chain.ref_volume_km3
    v_h = float(harm[chain.mask].sum() * cell)
    v_t = float(tp[chain.mask].sum() * cell)
    ratio_t = v_t / v_ref
    return {**row, "status": "solved", "profile_tau_kpa": tau,
            "flowline_rmse_m": fit["flowline_rmse_m"], "constraints": int(len(rows)),
            "reference_volume_km3": v_ref,
            "static_volume_harmonic_km3": v_h, "ratio_harmonic": v_h / v_ref,
            "static_volume_thinplate_km3": v_t, "ratio_thinplate": ratio_t,
            "thinplate_negative_cells_percent": tmeta["negative_raw_cells_percent"],
            "thickness_rmse_harmonic_m": float(np.sqrt(np.mean((harm[chain.mask] - chain.href[chain.mask]) ** 2))),
            "thickness_rmse_thinplate_m": float(np.sqrt(np.mean((tp[chain.mask] - chain.href[chain.mask]) ** 2))),
            "checkpoint_pass": CHECKPOINT["ratio_low"] <= ratio_t <= CHECKPOINT["ratio_high"]}


def main() -> None:
    npz_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    uids = sorted(int(p.stem.split("_")[-1]) for p in npz_dir.glob("fullchain_inputs_*.npz"))
    rows = []
    for uid in uids:
        t0 = time.perf_counter()
        try:
            row = run_system(npz_dir, uid)
        except Exception as error:  # recorded, never silently dropped
            row = {"lia_unique_id": uid, "status": "error", "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "static_modern_end.csv", index=False)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                          if k in ("lia_unique_id", "status", "profile_tau_kpa", "ratio_harmonic",
                                   "ratio_thinplate", "checkpoint_pass", "seconds")}), flush=True)
    t = pd.DataFrame(rows)
    solved = t[t.status == "solved"]
    passed = int(solved.checkpoint_pass.sum()) if len(solved) else 0
    summary = {
        "status": "completed", "elapsed_seconds": time.perf_counter() - started,
        "systems": int(len(t)), "solved": int(len(solved)), "passed": passed,
        "pass_share_of_all_systems": passed / max(len(t), 1),
        "route_continues_to_lia_end": passed / max(len(t), 1) >= CHECKPOINT["route_pass_share"],
        "median_ratio_harmonic": float(solved.ratio_harmonic.median()) if len(solved) else None,
        "median_ratio_thinplate": float(solved.ratio_thinplate.median()) if len(solved) else None,
        "sum_ratio_thinplate": float(solved.static_volume_thinplate_km3.sum() / solved.reference_volume_km3.sum()) if len(solved) else None,
        "status_counts": t.status.value_counts().to_dict(),
        "checkpoint": CHECKPOINT,
        "scope": "modern end of the same-method static route; reference is a model estimate; no LIA, climate or dynamic run",
    }
    (out_dir / "static_modern_end_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

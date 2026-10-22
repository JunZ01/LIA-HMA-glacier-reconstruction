"""Checkpoint 3, part 2: same-method static modern end for every member.

Input: the member NPZ directory from export_fullchain_inputs_members_20261022.py
(members_manifest.csv + fullchain_inputs_<uid>_<rgi_id>.npz).

Phase 1, per member: archived-tau harness where an archived tau exists, then
the unified-bed profile tau with the checkpoint-1 loss, grid and refinement.
Phase 2, per member: one network solve at the SYSTEM tau (the main member's
profile tau -- the same value the LIA end uses), production constraints,
thin-plate operator T, modern volume. The member's own-tau volume is kept as
a sensitivity.

Rules fixed before running (RULES):
  * system tau = main member profile tau; main member failure => system failed;
  * a system's modern end is complete when the modern area of failed members
    is <= 5 % of the system modern area (those members are listed and their
    reference volume is reported, never mixed into the static volume);
  * QC flag 'flowline-solve-unreliable' when system tau <= 45 kPa
    (within 5 kPa of the 40 kPa scan floor; facts: human decision 2026-10-22).
Each member result is written immediately to members/<uid>_<rgi>.json.
No LIA, climate or dynamic run.

Usage (repo root):
    python run_static_modern_members_20261022.py NPZ_DIR OUT_DIR [--workers N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("replay", ROOT / "replay_modern_fullchain_20261021.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)
from ice_surface_reconstruction.core import thinplate_extend_thickness  # noqa: E402

OUTLET = "H0"
RULES = {"failed_member_area_share_max": 0.05, "qc_tau_max_kpa": 45.0,
         "tau_scan_kpa": [replay.TAU_GRID_KPA[0], replay.TAU_GRID_KPA[-1]]}


def _load(npz_dir: Path, file: str, uid: int) -> dict:
    z = np.load(npz_dir / file, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["config"] = json.loads(str(d["legacy_config_json"]))
    d["uid"] = uid
    return d


def _thinplate_volume(chain, network) -> dict:
    rows, cols, thick, surf, _ = replay.collect_network_constraints(
        chain.inputs, chain.build, network, duplicate_surface_tolerance_m=chain.dup_tol,
        cross_branch_surface_tolerance_m=chain.cross_tol)
    inside = chain.mask[rows, cols]
    rows, cols, thick, surf = rows[inside], cols[inside], thick[inside], surf[inside]
    keep = ~chain.boundary[rows, cols] & (chain.boundary_distance[rows, cols] >= chain.exclusion_m)
    rows, cols, thick, surf = rows[keep], cols[keep], thick[keep], surf[keep]
    if not len(rows):
        raise ValueError("no constraint cells after 100 m exclusion")
    tp, meta = thinplate_extend_thickness(chain.mask, rows, cols, thick, centerline_surface_m=surf,
                                          duplicate_tolerance_m=chain.dup_tol)
    if meta["status"] != "converged":
        raise ValueError(f"thin-plate {meta['status']}")
    return {"constraints": int(len(rows)),
            "V_T_km3": float(tp[chain.mask].sum() * chain.dx ** 2 / 1e9),
            "negative_cells_percent": meta["negative_raw_cells_percent"]}


def phase1(task) -> dict:
    npz_dir, row = task
    uid, rgi = int(row["lia_unique_id"]), str(row["rgi_id"])
    out = {"lia_unique_id": uid, "rgi_id": rgi, "is_main": bool(row["is_main"])}
    t0 = time.perf_counter()
    try:
        d = _load(npz_dir, row["file"], uid)
        archived = float(d["legacy_tau_kpa"])
        if np.isfinite(archived):
            legacy = replay.legacy_calibration(d)
            out["harness"] = "reproduced" if abs(legacy["tau_kpa"] - archived) <= replay.LEGACY_TOLERANCE_KPA else "failed"
        else:
            out["harness"] = "not-available"
        if out["harness"] == "failed":
            return {**out, "status": "harness-failed", "seconds": time.perf_counter() - t0}
        chain = replay.ModernChain(d)
        out.update({"modern_area_km2": float(chain.mask.sum() * chain.dx ** 2 / 1e6),
                    "reference_volume_km3": chain.ref_volume_km3})
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
            return {**out, "status": "no-identifiable-tau", "seconds": time.perf_counter() - t0}
        best = min(finite, key=finite.get)
        lo, hi = max(replay.TAU_GRID_KPA[0], best - 20.0), min(replay.TAU_GRID_KPA[-1], best + 20.0)
        tau = float(minimize_scalar(loss, bounds=(lo, hi), method="bounded", options={"xatol": 0.25}).x)
        network, _ = chain.solve(round(tau, 6), OUTLET)
        own = _thinplate_volume(chain, network)
        out.update({"status": "solved", "own_tau_kpa": tau,
                    "own_flowline_rmse_m": chain.flowline_fit(network)["flowline_rmse_m"],
                    "own_V_T_km3": own["V_T_km3"], "own_constraints": own["constraints"]})
    except Exception as error:  # recorded, never silently dropped
        out.update({"status": "error", "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()})
    out["seconds"] = time.perf_counter() - t0
    return out


def phase2(task) -> dict:
    npz_dir, row, system_tau = task
    uid, rgi = int(row["lia_unique_id"]), str(row["rgi_id"])
    out = {"lia_unique_id": uid, "rgi_id": rgi}
    try:
        d = _load(npz_dir, row["file"], uid)
        chain = replay.ModernChain(d)
        network, status = chain.solve(round(system_tau, 6), OUTLET)
        if network is None:
            return {**out, "system_tau_status": status["status"]}
        v = _thinplate_volume(chain, network)
        out.update({"system_tau_status": "solved", "system_V_T_km3": v["V_T_km3"],
                    "system_constraints": v["constraints"],
                    "system_negative_cells_percent": v["negative_cells_percent"],
                    "system_flowline_rmse_m": chain.flowline_fit(network)["flowline_rmse_m"]})
    except Exception as error:
        out.update({"system_tau_status": "error", "system_error": f"{type(error).__name__}: {error}"})
    return out


def _map(fn, tasks, workers):
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(fn, tasks))
    return [fn(t) for t in tasks]


def aggregate(manifest: pd.DataFrame, members: pd.DataFrame, systems_population: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    uids = sorted(set(systems_population.lia_unique_id.astype(int)) if systems_population is not None
                  else set(manifest.lia_unique_id.astype(int)))
    for uid in uids:
        m = members[members.lia_unique_id == uid]
        listed = manifest[manifest.lia_unique_id == uid]
        main = m[m.is_main]
        row = {"lia_unique_id": uid, "members_listed": int(len(listed)),
               "members_exported": int((listed.status == "exported").sum())}
        if main.empty or main.iloc[0].get("status") != "solved":
            reason = "main-member-not-exported" if main.empty else f"main-member-{main.iloc[0].get('status')}"
            rows.append({**row, "modern_status": reason})
            continue
        mrow = main.iloc[0]
        tau = float(mrow.own_tau_kpa)
        ok = m.get("system_tau_status", pd.Series(dtype=str)).eq("solved")
        area_ok = float(m.loc[ok, "modern_area_km2"].sum())
        # failed members include those that never exported (area unknown -> counted as failure by count)
        area_failed = float(m.loc[~ok, "modern_area_km2"].fillna(0).sum())
        unexported = int(len(listed) - (listed.status == "exported").sum())
        unknown_area_failures = int((~ok & m.modern_area_km2.isna()).sum()) if "modern_area_km2" in m else int((~ok).sum())
        share_failed = area_failed / max(area_ok + area_failed, 1e-12)
        complete = (share_failed <= RULES["failed_member_area_share_max"]) and unexported == 0 \
            and unknown_area_failures == 0 and ok.any()
        own_ok = m.own_V_T_km3.notna() & ok
        rows.append({**row,
                     "main_rgi_id": mrow.rgi_id, "system_tau_kpa": tau,
                     "qc_flowline_solve_unreliable": bool(tau <= RULES["qc_tau_max_kpa"]),
                     "members_solved_at_system_tau": int(ok.sum()),
                     "members_unexported": unexported,
                     "members_failed_unknown_area": unknown_area_failures,
                     "failed_member_area_share": share_failed,
                     "modern_area_solved_km2": area_ok,
                     "V_mod_T_km3": float(m.loc[ok, "system_V_T_km3"].sum()),
                     "V_mod_T_own_tau_km3": float(m.loc[own_ok, "own_V_T_km3"].sum()) if own_ok.all() and ok.all() else np.nan,
                     "reference_volume_solved_members_km3": float(m.loc[ok, "reference_volume_km3"].sum()),
                     "reference_volume_failed_members_km3": float(m.loc[~ok, "reference_volume_km3"].fillna(0).sum()),
                     "modern_status": "complete" if complete else "incomplete-member-failures"})
    s = pd.DataFrame(rows)
    if len(s) and "V_mod_T_km3" in s:
        s["ratio_T_to_reference"] = s.V_mod_T_km3 / s.reference_volume_solved_members_km3
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "members").mkdir()
    started = time.perf_counter()
    manifest = pd.read_csv(args.npz_dir / "members_manifest.csv")
    pop_path = args.npz_dir / "systems_population.csv"
    population = pd.read_csv(pop_path) if pop_path.exists() else None
    exported = manifest[manifest.status == "exported"].to_dict("records")
    p1 = _map(phase1, [(args.npz_dir, r) for r in exported], args.workers)
    members = pd.DataFrame(p1).merge(
        manifest[["lia_unique_id", "rgi_id", "file", "region", "modern_member_count"]],
        on=["lia_unique_id", "rgi_id"], how="left", validate="one_to_one")
    system_tau = {int(r.lia_unique_id): float(r.own_tau_kpa) for r in members.itertuples()
                  if r.is_main and r.status == "solved"}
    tasks = [(args.npz_dir, r, system_tau[int(r["lia_unique_id"])]) for r in members.to_dict("records")
             if int(r["lia_unique_id"]) in system_tau and r["status"] in ("solved", "no-identifiable-tau")]
    p2 = pd.DataFrame(_map(phase2, tasks, args.workers))
    if len(p2):
        members = members.merge(p2, on=["lia_unique_id", "rgi_id"], how="left", validate="one_to_one")
    for r in members.to_dict("records"):
        with (args.out_dir / "members" / f"{r['lia_unique_id']}_{r['rgi_id']}.json").open("x", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False, default=str)
    members.drop(columns=["traceback"], errors="ignore").to_csv(args.out_dir / "modern_members.csv", index=False)
    systems = aggregate(manifest, members, population)
    if population is not None:
        systems = systems.merge(population[["lia_unique_id", "region", "modern_member_count",
                                            "modern_reference_native_sum_km3"]], on="lia_unique_id",
                                how="left", validate="one_to_one")
    systems.to_csv(args.out_dir / "modern_systems.csv", index=False)
    complete = systems[systems.modern_status == "complete"]
    tau_table = complete[["main_rgi_id", "system_tau_kpa", "lia_unique_id"]].rename(
        columns={"main_rgi_id": "rgi_id", "system_tau_kpa": "tau_kpa"})
    tau_table.insert(1, "bed_scenario", "farinotti2019")
    tau_table["status"] = "unified-bed-profile-tau-main-member"
    tau_table["kappa_kpa"] = tau_table.tau_kpa / 0.8
    tau_table.to_csv(args.out_dir / "system_tau_for_lia_end.csv", index=False)
    summary = {"status": "completed", "elapsed_seconds": time.perf_counter() - started,
               "systems": int(len(systems)), "modern_complete": int(len(complete)),
               "members_listed": int(len(manifest)), "members_phase1_solved": int((members.status == "solved").sum()),
               "qc_flagged_systems": complete.loc[complete.qc_flowline_solve_unreliable, "lia_unique_id"].astype(int).tolist(),
               "median_ratio_T_to_reference": float(complete.ratio_T_to_reference.median()) if len(complete) else None,
               "status_counts": systems.modern_status.value_counts().to_dict(),
               "rules": RULES,
               "scope": "same-method static modern end, all members; reference is a model estimate; no LIA, climate or dynamic run"}
    (args.out_dir / "modern_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

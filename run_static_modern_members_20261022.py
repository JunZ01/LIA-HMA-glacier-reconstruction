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
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _load(npz_dir: Path, file: str, uid: int, expected_sha=None) -> dict:
    path = npz_dir / file
    if expected_sha and hashlib.sha256(path.read_bytes()).hexdigest().lower() != str(expected_sha).lower():
        raise ValueError("NPZ hash differs from frozen member manifest")
    with np.load(path, allow_pickle=False) as z:
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
        d = _load(npz_dir, row["file"], uid, row.get("sha256"))
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
        d = _load(npz_dir, row["file"], uid, row.get("sha256"))
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


def _map(fn, tasks, workers, checkpoint_dir, fingerprint):
    checkpoint_dir.mkdir(exist_ok=True)
    completed, pending = [], []
    for task in tasks:
        row = task[1]
        path = checkpoint_dir / f"{row['lia_unique_id']}_{row['rgi_id']}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Checkpoint fingerprint mismatch")
            completed.append(saved["result"])
        else:
            pending.append(task)
    def commit(result):
        path = checkpoint_dir / f"{result['lia_unique_id']}_{result['rgi_id']}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump({"fingerprint": fingerprint, "result": result}, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
        completed.append(result)
        if len(completed) % 25 == 0 or result.get("status", result.get("system_tau_status")) != "solved":
            print(json.dumps({"phase": fn.__name__, "completed": len(completed), "total": len(tasks)}), flush=True)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            for future in as_completed([pool.submit(fn, task) for task in pending]):
                commit(future.result())
    else:
        for task in pending:
            commit(fn(task))
    return completed


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
        if mrow.get("system_tau_status") != "solved":
            rows.append({**row, "main_rgi_id": mrow.rgi_id, "system_tau_kpa": tau,
                         "qc_flowline_solve_unreliable": bool(tau <= RULES["qc_tau_max_kpa"]),
                         "modern_status": f"main-member-system-tau-{mrow.get('system_tau_status', 'not-solved')}"})
            continue
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
                     "failed_member_rgi_ids": ";".join(sorted(set(listed.rgi_id.astype(str)) - set(m.loc[ok, "rgi_id"].astype(str)))),
                     "modern_area_failed_km2": area_failed,
                     "modern_area_total_km2": area_ok + area_failed,
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
    ap.add_argument("--resume", action="store_true", help="Reuse only matching per-phase member checkpoints")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)
    if (args.out_dir / "modern_summary.json").exists():
        raise FileExistsError("Completed modern batch must not be recomputed")
    (args.out_dir / "members").mkdir(exist_ok=args.resume)
    started = time.perf_counter()
    manifest = pd.read_csv(args.npz_dir / "members_manifest.csv")
    pop_path = args.npz_dir / "systems_population.csv"
    population = pd.read_csv(pop_path) if pop_path.exists() else None
    if population is None or len(population) != 191 or population.lia_unique_id.duplicated().any():
        raise ValueError("Frozen 191-system population required")
    digest = hashlib.sha256()
    for path in [Path(__file__), ROOT / "replay_modern_fullchain_20261021.py",
                 args.npz_dir / "members_manifest.csv", pop_path]:
        digest.update(path.read_bytes())
    digest.update(json.dumps(RULES, sort_keys=True).encode())
    package = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction"
    for path in sorted((package / "core").glob("*.py")):
        digest.update(path.read_bytes())
    for name in ["modern_tau_calibration.py", "oggm_network_adapter.py", "junction_connector.py", "gridded_bed.py"]:
        digest.update((package / "drivers" / name).read_bytes())
    fingerprint = digest.hexdigest()
    exported = manifest[manifest.status == "exported"].to_dict("records")
    p1 = _map(phase1, [(args.npz_dir, r) for r in exported], args.workers,
              args.out_dir / "phase1", fingerprint)
    members = pd.DataFrame(p1).merge(
        manifest[["lia_unique_id", "rgi_id", "file", "region", "modern_member_count", "sha256"]],
        on=["lia_unique_id", "rgi_id"], how="left", validate="one_to_one")
    members = members.sort_values(["lia_unique_id", "rgi_id"]).reset_index(drop=True)
    # Known exported areas remain usable when fitting a member's sensitivity tau fails.
    for col in ["modern_area_km2", "reference_volume_km3"]:
        if col not in members:
            members[col] = np.nan
        if col in manifest:
            fallback = manifest.set_index(["lia_unique_id", "rgi_id"])[col]
            keys = pd.MultiIndex.from_frame(members[["lia_unique_id", "rgi_id"]])
            members[col] = members[col].fillna(pd.Series(fallback.reindex(keys).to_numpy(), index=members.index))
    for col in ["own_V_T_km3", "own_tau_kpa"]:
        if col not in members:
            members[col] = np.nan
    system_tau = {int(r.lia_unique_id): float(r.own_tau_kpa) for r in members.itertuples()
                  if r.is_main and r.status == "solved"}
    tasks = [(args.npz_dir, r, system_tau[int(r["lia_unique_id"])]) for r in members.to_dict("records")
             if int(r["lia_unique_id"]) in system_tau and r["status"] != "harness-failed"]
    p2 = pd.DataFrame(_map(phase2, tasks, args.workers, args.out_dir / "phase2", fingerprint))
    if len(p2):
        members = members.merge(p2, on=["lia_unique_id", "rgi_id"], how="left", validate="one_to_one")
    for col in ["system_tau_status", "system_V_T_km3"]:
        if col not in members:
            members[col] = np.nan
    for r in members.to_dict("records"):
        with (args.out_dir / "members" / f"{r['lia_unique_id']}_{r['rgi_id']}.json").open("w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False, default=str)
    members.drop(columns=["traceback"], errors="ignore").to_csv(args.out_dir / "modern_members.csv", index=False)
    systems = aggregate(manifest, members, population)
    for col in ["main_rgi_id", "system_tau_kpa", "qc_flowline_solve_unreliable", "ratio_T_to_reference"]:
        if col not in systems:
            systems[col] = np.nan
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

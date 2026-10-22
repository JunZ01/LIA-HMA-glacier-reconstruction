"""Run tau schemes A (main), B and C with the unified static engine.

Reports only what the human asked for: total dV%, regional dV%, and the
differences between schemes, computed over the systems complete in all three
schemes. Decision rule fixed by the human (2026-10-22): if |A-B| and |A-C| of
the total dV% are both <= 2 percentage points, scheme A is carried to the
829 systems. F19 sets the tau coefficients only; it is not used as a check.

Usage (repo root):
    python run_unified_abc_20261022.py NPZ_DIR OUT_DIR [--workers N]
"""

from __future__ import annotations

import argparse
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

import static_unified_engine_20261022 as eng
from unified_static_io_20261022 import atomic_json, sha256, fingerprint

GATE_PP = 2.0
_CACHE_ROOT = None
_SCIENCE_SIGNATURE = None

def _cached_task(task):
    name, arg, path = task
    value=globals()[name](arg)
    atomic_json(path,{'value':value})
    print(json.dumps({'completed':name,'file':Path(arg[0] if isinstance(arg,tuple) else arg).name}),flush=True)
    return value


def _meta(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {"file": path.name, "record_id": str(z["record_id"]), "region": str(z["region"]),
            "cohort": str(z["cohort"]), "f19_volume_km3": float(z["f19_volume_km3"]),
            "modern_member_failures":int(z['modern_member_failures']) if 'modern_member_failures' in z else 0,
            "modern_unrouted_component_cells":int(z['modern_unrouted_component_cells']) if 'modern_unrouted_component_cells' in z else 0}


def _modern_volume(task):
    path, tau = task
    try:
        d = eng.load_system(path)
        return eng.modern_end(d, tau)["V_km3"]
    except Exception:
        return np.nan


def _dk(path):
    try:
        return eng.dk_tau_kpa(eng.load_system(path))
    except Exception:
        return np.nan


def _full(task):
    path, scheme, tau = task
    out = {"file": path.name, "scheme": scheme}
    t0 = time.perf_counter()
    try:
        if not np.isfinite(tau):
            raise ValueError('tau calibration unavailable')
        out.update({"status": "ok", **eng.run_system(eng.load_system(path), tau)})
    except Exception as error:
        out.update({"status": "failed", "tau_kpa": tau, "error": f"{type(error).__name__}: {error}"[:300],
                    "traceback": traceback.format_exc()[-1500:]})
    out["seconds"] = time.perf_counter() - t0
    return out


def _map(fn, tasks, workers):
    if _CACHE_ROOT is not None:
        outputs=[None]*len(tasks);pending={}
        for i,arg in enumerate(tasks):
            p=Path(arg[0] if isinstance(arg,tuple) else arg)
            # Scheme labels do not change the physical full-chain computation.
            physical=(str(p),float(arg[2])) if fn.__name__=='_full' else arg
            key=fingerprint({'function':fn.__name__,'task':physical,'npz':sha256(p),'science':_SCIENCE_SIGNATURE})
            path=_CACHE_ROOT/fn.__name__/f'{key}.json'
            if path.exists():outputs[i]=json.loads(path.read_text(encoding='utf-8'))['value']
            else:pending.setdefault(key,{'arg':arg,'path':path,'indices':[]})['indices'].append(i)
        entries=list(pending.values());jobs=[(fn.__name__,e['arg'],e['path']) for e in entries]
        if workers>1:
            with ProcessPoolExecutor(workers) as pool:values=list(pool.map(_cached_task,jobs,chunksize=1))
        else:values=[_cached_task(j) for j in jobs]
        for e,value in zip(entries,values):
            for i in e['indices']:outputs[i]=value
        if fn.__name__=='_full':outputs=[{**v,'scheme':arg[1]} for v,arg in zip(outputs,tasks)]
        return outputs
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(fn, tasks, chunksize=1))
    return [fn(t) for t in tasks]


def calibrate(meta: pd.DataFrame, npz_dir: Path, workers: int) -> pd.DataFrame:
    paths = [npz_dir / f for f in meta.file]
    meta = meta.copy()
    meta["tau_dk_kpa"] = _map(_dk, paths, workers)
    meta["tau_C"] = meta.tau_dk_kpa.map(lambda t: eng.clamp_tau(t) if np.isfinite(t) else np.nan)
    meta["V_C"] = _map(_modern_volume, list(zip(paths, meta.tau_C)), workers)
    ok = np.isfinite(meta.V_C) & (meta.V_C > 0)
    # scheme A: one coefficient per region, two fixed-point iterations
    k = {r: g.f19_volume_km3.sum() / g.V_C.sum() for r, g in meta[ok].groupby("region")}
    for _ in range(2):
        meta["tau_A"] = [eng.clamp_tau(k.get(r, np.nan) * t) if np.isfinite(t) and r in k else np.nan
                         for r, t in zip(meta.region, meta.tau_dk_kpa)]
        meta["V_A"] = _map(_modern_volume, list(zip(paths, meta.tau_A)), workers)
        okA = ok & np.isfinite(meta.V_A)
        k = {r: k[r] * g.f19_volume_km3.sum() / g.V_A.sum() for r, g in meta[okA].groupby("region")}
    meta["k_region_A"] = meta.region.map(k)
    meta["tau_A"] = [eng.clamp_tau(k.get(r, np.nan) * t) if np.isfinite(t) and r in k else np.nan
                     for r, t in zip(meta.region, meta.tau_dk_kpa)]
    # scheme B: one coefficient per system, two fixed-point iterations
    meta["tau_B"] = [eng.clamp_tau(t * f / v) if np.isfinite(v) and v > 0 else np.nan
                     for t, f, v in zip(meta.tau_C, meta.f19_volume_km3, meta.V_C)]
    for _ in range(2):
        vb = _map(_modern_volume, list(zip(paths, meta.tau_B)), workers)
        meta["tau_B"] = [eng.clamp_tau(t * f / v) if np.isfinite(v) and v > 0 else np.nan
                         for t, f, v in zip(meta.tau_B, meta.f19_volume_km3, vb)]
    for s in "ABC":
        meta[f"tau_{s}_clamped"] = meta[f"tau_{s}"].isin(list(eng.TAU_LIMITS_KPA))
    return meta


def totals(frame: pd.DataFrame) -> dict:
    lia, mod = frame.V_lia_km3.sum(), frame.V_mod_km3.sum()
    return {"systems": int(len(frame)), "V_lia_km3": float(lia), "V_mod_km3": float(mod),
            "dV_km3": float(mod - lia), "dV_percent": float(100 * (mod - lia) / lia) if lia > 0 else None}


def main():
    global _CACHE_ROOT,_SCIENCE_SIGNATURE
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument('--resume',action='store_true')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)
    _CACHE_ROOT=args.out_dir/'task_cache'
    _SCIENCE_SIGNATURE=fingerprint({str(p):sha256(p) for p in [Path(__file__),Path(eng.__file__),Path(eng.ROOT/'replay_modern_fullchain_20261021.py'),*sorted((eng.ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/core').glob('*.py'))]})
    started = time.perf_counter()
    files = sorted(args.npz_dir.glob("system_*.npz"))
    meta = pd.DataFrame([_meta(p) for p in files])
    meta = calibrate(meta, args.npz_dir, args.workers)
    meta.to_csv(args.out_dir / "tau_schemes.csv", index=False)
    tasks = [(args.npz_dir / r.file, s, getattr(r, f"tau_{s}")) for r in meta.itertuples() for s in "ABC"]
    rows = pd.DataFrame(_map(_full, tasks, args.workers)).merge(
        meta[["file", "record_id", "region", "cohort", "f19_volume_km3",'modern_member_failures','modern_unrouted_component_cells']], on="file", how="left")
    rows.drop(columns=["traceback"], errors="ignore").to_csv(args.out_dir / "systems_by_scheme.csv", index=False)
    rows['input_complete']=(rows.modern_member_failures==0)&(rows.modern_unrouted_component_cells==0)
    rows.drop(columns=['traceback'],errors='ignore').to_csv(args.out_dir/'systems_by_scheme.csv',index=False)
    # Preserve the supplied common-numerical-success comparison rule. Coverage
    # flags are reported separately and must not become a new post-hoc gate.
    ok = rows[rows.status == "ok"]
    common = set.intersection(*[set(ok.loc[ok.scheme == s, "file"]) for s in "ABC"]) if len(ok) else set()
    c = ok[ok.file.isin(common)]
    summary = {"systems": int(len(meta)), "complete_in_all_schemes": int(len(common)),
               "failed_by_scheme": {s: int(((rows.scheme == s) & (rows.status != "ok")).sum()) for s in "ABC"},
               "k_region_A": {r: float(v) for r, v in meta.groupby("region").k_region_A.first().items()},
               "tau_median_kpa": {s: float(meta[f"tau_{s}"].median()) for s in "ABC"},
               "tau_clamped_systems": {s: int(meta[f"tau_{s}_clamped"].sum()) for s in "ABC"}}
    summary['input_incomplete_systems']=meta.loc[(meta.modern_member_failures>0)|(meta.modern_unrouted_component_cells>0),'record_id'].tolist()
    summary['input_coverage_policy']='common means numerical success in A/B/C, as supplied; member failures and unrouted islands are separate QA flags, not an added pass gate'
    summary['strict_input_complete_among_common']=int(meta.loc[meta.file.isin(common)].eval('modern_member_failures == 0 and modern_unrouted_component_cells == 0').sum())
    summary['common_system_ids']=meta.loc[meta.file.isin(common),'record_id'].tolist()
    summary['calibration_residuals']={}
    for scheme in 'AB':
        solved=rows[(rows.scheme==scheme)&(rows.status=='ok')]
        summary['calibration_residuals'][scheme]={r:{'systems':len(g),'modern_volume_km3':float(g.V_mod_km3.sum()),'F19_target_km3':float(g.f19_volume_km3.sum()),'relative_residual_percent':float(100*(g.V_mod_km3.sum()/g.f19_volume_km3.sum()-1))} for r,g in solved.groupby('region')}
    if len(rows):
        final_mod=rows.pivot(index='file',columns='scheme',values='V_mod_km3') if 'V_mod_km3' in rows else pd.DataFrame()
        for scheme in 'ABC':
            if scheme in final_mod:meta[f'V_{scheme}_final']=meta.file.map(final_mod[scheme]);meta[f'closure_{scheme}_percent']=100*(meta[f'V_{scheme}_final']/meta.f19_volume_km3-1)
        meta.to_csv(args.out_dir/'tau_schemes.csv',index=False)
    export_summary=args.npz_dir/'export_summary.json'
    if export_summary.exists():summary['export_population']=json.loads(export_summary.read_text(encoding='utf-8'))
    for s in "ABC":
        summary[f"total_{s}"] = totals(c[c.scheme == s])
        summary[f"by_region_{s}"] = {r: totals(g) for r, g in c[c.scheme == s].groupby("region")}
    dA = summary["total_A"]["dV_percent"]
    if dA is not None:
        summary["diff_A_minus_B_pp"] = dA - summary["total_B"]["dV_percent"]
        summary["diff_A_minus_C_pp"] = dA - summary["total_C"]["dV_percent"]
        summary["regional_diff_pp"] = {
            r: {"A-B": summary["by_region_A"][r]["dV_percent"] - summary["by_region_B"][r]["dV_percent"],
                "A-C": summary["by_region_A"][r]["dV_percent"] - summary["by_region_C"][r]["dV_percent"]}
            for r in summary["by_region_A"]}
        summary["gate_within_2pp"] = bool(abs(summary["diff_A_minus_B_pp"]) <= GATE_PP
                                          and abs(summary["diff_A_minus_C_pp"]) <= GATE_PP)
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["scope"] = "unified self-consistent static engine; F19 only sets tau coefficients; no dynamic run"
    (args.out_dir / "abc_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

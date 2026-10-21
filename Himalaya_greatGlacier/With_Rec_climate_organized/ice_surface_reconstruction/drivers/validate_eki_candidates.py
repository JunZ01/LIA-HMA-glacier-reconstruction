"""Run a small real-OGGM validation set for EKI candidate parameters."""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .lia_lhs_batch import _run_trial


GLEN_A_DEFAULT = 2.939387691339814e-24


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--processes", type=int, default=3)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates)
    candidates = candidates[candidates["eki_rank"].astype(int) <= args.top_n]
    base = pd.read_csv(args.base_results)
    payloads = []
    for _, candidate in candidates.iterrows():
        sid = int(candidate["system_id"])
        source_point = int(candidate["source_point"])
        source_rows = base[(base["system_id"].astype(int) == sid) & (base["point"].astype(int) == source_point)]
        # Candidate source points may come from a different saved design. The
        # system-level paths and precipitation initial value are invariant, so
        # fall back to any canonical local-refine row for that system.
        if len(source_rows) != 1:
            source_rows = base[base["system_id"].astype(int) == sid].head(1)
        if len(source_rows) != 1:
            raise RuntimeError(f"source-row-count:{sid}:{source_point}:{len(source_rows)}")
        source = source_rows.iloc[0]
        tag = f"system_{sid}/eki_{int(candidate['eki_rank']):03d}"
        output_dir = (args.output_root / tag).resolve()
        # Keep OGGM's working tree outside the results tree. Some source
        # glacier directories contain links/relative paths that fail when the
        # working directory is nested below the output directory.
        workdir = (Path(str(source["project_root"])) / "lrw" / f"eki_{sid}_{int(candidate['eki_rank']):03d}").resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if workdir.exists():
            shutil.rmtree(workdir)
        payloads.append({
            "system_id": sid,
            "rgi_id": str(source["rgi_id"]),
            "point": int(candidate["eki_rank"]),
            "delta_t": float(candidate["delta_t"]),
            "melt_f": float(candidate["melt_f"]),
            "prcp_fac_multiplier": float(candidate["prcp_fac_multiplier"]),
            "glen_a_multiplier": float(candidate["glen_a_multiplier"]),
            "glen_a": GLEN_A_DEFAULT * float(candidate["glen_a_multiplier"]),
            "temp_melt": float(candidate["temp_melt"]),
            "prcp_fac_initial": float(source["prcp_fac_initial"]),
            "python_exe": str(source["python_exe"]),
            "project_root": str(source["project_root"]),
            "references": str(source["references"]),
            "stage2_members": str(source["stage2_members"]),
            "climate_pixels": str(source["climate_pixels"]),
            "reconstruction_climate": str(source["reconstruction_climate"]),
            "source_gdir_root": str(source["source_gdir_root"]),
            "elevation_band_root": str(source["elevation_band_root"]),
            "output_dir": str(output_dir),
            "workdir": str(workdir),
            "timeout_seconds": 240,
        })
    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_trial, payload): payload for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"completed system={result['system_id']} status={result['status']}", flush=True)
    frame = pd.DataFrame(results).sort_values(["system_id", "point"])
    frame.to_csv(args.output_root / "eki_candidate_validation.csv", index=False)
    columns = ["system_id", "status"]
    for name in ("e_lia", "e_2000", "e_total", "l_spinup_m", "l_2000_m"):
        if name in frame.columns:
            columns.append(name)
    print(frame[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

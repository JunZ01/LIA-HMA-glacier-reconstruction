"""Search the parameter path between two conflicting real OGGM branches."""

from __future__ import annotations

import argparse
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_lhs_batch import _run_trial


GLEN_A_DEFAULT = 2.939387691339814e-24


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--canonical-base", type=Path, required=True)
    parser.add_argument("--branch-a", type=int, required=True)
    parser.add_argument("--branch-b", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    design = pd.read_csv(args.design)
    source_a = design[(design.system_id.astype(int) == 1545) & (design.point.astype(int) == args.branch_a)].iloc[0]
    source_b = design[(design.system_id.astype(int) == 1545) & (design.point.astype(int) == args.branch_b)].iloc[0]
    canonical = pd.read_csv(args.canonical_base)
    base = canonical[canonical.system_id.astype(int) == 1545].iloc[0]
    rng = np.random.default_rng(args.seed)
    alphas = np.linspace(0.0, 1.0, 7)
    payloads = []
    for index, alpha in enumerate(alphas):
        # Interpolate glen_a in log space; the other controls are linear.
        row = {
            "delta_t": (1 - alpha) * float(source_a.delta_t) + alpha * float(source_b.delta_t),
            "melt_f": (1 - alpha) * float(source_a.melt_f) + alpha * float(source_b.melt_f),
            "prcp_fac_multiplier": (1 - alpha) * float(source_a.prcp_fac_multiplier) + alpha * float(source_b.prcp_fac_multiplier),
            "glen_a_multiplier": math.exp((1 - alpha) * math.log(float(source_a.glen_a_multiplier)) + alpha * math.log(float(source_b.glen_a_multiplier))),
            "temp_melt": (1 - alpha) * float(source_a.temp_melt) + alpha * float(source_b.temp_melt),
        }
        # Add one perturbation at interior points to test local curvature.
        if 0 < index < len(alphas) - 1:
            row["melt_f"] = float(np.clip(row["melt_f"] + rng.normal(0, 0.25), 2.5, 7.5))
            row["temp_melt"] = float(np.clip(row["temp_melt"] + rng.normal(0, 0.12), -2.0, 0.0))
            row["prcp_fac_multiplier"] = float(np.clip(row["prcp_fac_multiplier"] + rng.normal(0, 0.025), 0.8, 1.2))
        tag = f"system_1545/branch_{index:02d}"
        output_dir = (args.output_root / tag).resolve()
        workdir = (Path(str(base.project_root)) / "lrw" / f"branch1545_{index:02d}").resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if workdir.exists():
            shutil.rmtree(workdir)
        payloads.append({
            "system_id": 1545,
            "rgi_id": str(base.rgi_id),
            "point": index,
            **row,
            "glen_a": GLEN_A_DEFAULT * row["glen_a_multiplier"],
            "prcp_fac_initial": float(base.prcp_fac_initial),
            "python_exe": str(base.python_exe),
            "project_root": str(base.project_root),
            "references": str(base.references),
            "stage2_members": str(base.stage2_members),
            "climate_pixels": str(base.climate_pixels),
            "reconstruction_climate": str(base.reconstruction_climate),
            "source_gdir_root": str(base.source_gdir_root),
            "elevation_band_root": str(base.elevation_band_root),
            "output_dir": str(output_dir),
            "workdir": str(workdir),
            "timeout_seconds": 240,
            "branch_alpha": alpha,
        })
    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_trial, payload): payload for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"completed branch_alpha={result['branch_alpha']:.3f} status={result['status']}", flush=True)
    pd.DataFrame(results).sort_values("point").to_csv(args.output_root / "results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify one signed-response trust step with real OGGM chains."""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .curve_objective_v2 import CurveObjectiveConfig, score_curve
from .lia_lhs_batch import _run_trial


DEFAULT_GLEN_A = 2.939387691339814e-24


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals-csv", type=Path, required=True)
    parser.add_argument("--ofat-results", type=Path, required=True)
    parser.add_argument("--targets-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--python-exe", type=Path, default=Path(r"D:\Anaconda\envs\oggm_env2\python.exe"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    proposals = pd.read_csv(args.proposals_csv)
    proposals = proposals[np.isclose(proposals["alpha"].astype(float), args.alpha)].copy()
    bases = pd.read_csv(args.ofat_results)
    bases = bases[bases["ofat_trial"].astype(str).eq("base")].set_index("system_id")
    payloads = []
    for _, proposal in proposals.iterrows():
        system_id = int(proposal.system_id)
        base = bases.loc[system_id]
        output_dir = (args.output_root / f"system_{system_id}" / f"alpha_{args.alpha:g}").resolve()
        workdir = (Path(str(base.project_root)) / "lrw" / f"signed_v2_{system_id}_{args.alpha:g}").resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if workdir.exists():
            shutil.rmtree(workdir)
        payloads.append({
            "system_id": system_id,
            "rgi_id": str(base.rgi_id),
            "point": 0,
            "delta_t": float(proposal.delta_t),
            "melt_f": float(proposal.melt_f),
            "prcp_fac_multiplier": float(proposal.prcp_fac_multiplier),
            "glen_a_multiplier": float(proposal.glen_a_multiplier),
            "glen_a": DEFAULT_GLEN_A * float(proposal.glen_a_multiplier),
            "temp_melt": float(proposal.temp_melt),
            "prcp_fac_initial": float(base.prcp_fac_initial),
            "python_exe": str(args.python_exe.resolve()),
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
            "alpha": args.alpha,
            "predicted_objective_linearized": float(proposal.predicted_objective_linearized),
            "base_objective_linearized": float(proposal.base_objective_linearized),
        })

    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_trial, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {**payload, "status": "crash", "error": f"{type(exc).__name__}:{exc}"}
            results.append(result)
            print(f"system={payload['system_id']} status={result['status']}", flush=True)

    targets = pd.read_csv(args.targets_csv).set_index("lia_unique_id")
    config = CurveObjectiveConfig()
    scored = []
    for result in results:
        record = dict(result)
        try:
            target = targets.loc[int(result["system_id"])]
            series = pd.read_csv(Path(str(result["output_dir"])) / "timeseries.csv")
            record.update(score_curve(series, float(target.lia_registered_length_m), float(target.rgi_registered_length_m), config))
            record["actual_improvement_fraction"] = 1.0 - float(record["objective_v2"]) / float(record["base_objective_linearized"])
        except Exception as exc:
            record.update({"objective_v2": np.inf, "accepted_curve_v2": False, "failure_reasons_v2": "unscorable", "objective_error_v2": str(exc)})
        scored.append(record)
    frame = pd.DataFrame(scored).sort_values("system_id")
    frame.to_csv(args.output_root / "verification_results.csv", index=False)
    print(frame[["system_id", "status", "base_objective_linearized", "predicted_objective_linearized", "objective_v2", "actual_improvement_fraction", "failure_reasons_v2"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

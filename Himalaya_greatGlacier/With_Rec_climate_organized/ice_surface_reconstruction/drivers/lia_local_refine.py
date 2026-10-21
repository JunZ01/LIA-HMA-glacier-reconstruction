"""Run a small local parameter refinement around selected D-layer trials."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_lhs_batch import _run_trial


def _local_trials(design: pd.DataFrame, seeds: dict[int, int], n_local: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for system_id, point in seeds.items():
        base = design[(design["system_id"].astype(int) == system_id) & (design["point"].astype(int) == point)]
        if len(base) != 1:
            raise ValueError(f"seed-point-missing:{system_id}:{point}")
        center = base.iloc[0].to_dict()
        values = [center]
        for _ in range(n_local - 1):
            row = center.copy()
            row["delta_t"] = np.clip(float(center["delta_t"]) + rng.normal(0.0, 0.35), -3.0, 1.5)
            row["melt_f"] = np.clip(float(center["melt_f"]) + rng.normal(0.0, 0.7), 2.5, 7.5)
            row["prcp_fac_multiplier"] = np.clip(
                float(center["prcp_fac_multiplier"]) + rng.normal(0.0, 0.07), 0.8, 1.2
            )
            row["glen_a_multiplier"] = np.clip(
                float(center["glen_a_multiplier"]) * math.exp(rng.normal(0.0, 0.22)), 0.5, 3.0
            )
            row["glen_a"] = 2.939387691339814e-24 * float(row["glen_a_multiplier"])
            row["temp_melt"] = np.clip(float(center["temp_melt"]) + rng.normal(0.0, 0.3), -2.0, 0.0)
            values.append(row)
        for local_point, row in enumerate(values):
            rows.append({
                "system_id": system_id,
                "rgi_id": str(row["rgi_id"]),
                "point": local_point,
                "seed_point": point,
                "delta_t": float(row["delta_t"]),
                "melt_f": float(row["melt_f"]),
                "prcp_fac_multiplier": float(row["prcp_fac_multiplier"]),
                "glen_a_multiplier": float(row["glen_a_multiplier"]),
                "glen_a": float(row["glen_a"]),
                "temp_melt": float(row["temp_melt"]),
                "prcp_fac_initial": float(row["prcp_fac_initial"]),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--reconstruction-climate", required=True, type=Path)
    parser.add_argument("--source-gdir-root", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--n-local", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--seeds", type=str, default="1044:48,1545:142,3584:3")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    project_root = str(Path(__file__).resolve().parents[4])
    design = pd.read_csv(args.design)
    seeds = {}
    for item in args.seeds.split(","):
        system_text, point_text = item.split(":", 1)
        seeds[int(system_text)] = int(point_text)
    trials = _local_trials(design, seeds, args.n_local, args.seed)
    pd.DataFrame(trials).to_csv(args.output_root / "parameter_design.csv", index=False)
    payloads = []
    for trial in trials:
        tag = f"system_{trial['system_id']}/point_{trial['point']:04d}"
        payloads.append({
            **trial,
            "python_exe": str(args.python_exe.resolve()),
            "project_root": project_root,
            "references": str(args.references.resolve()),
            "stage2_members": str(args.stage2_members.resolve()),
            "climate_pixels": str(args.climate_pixels.resolve()),
            "reconstruction_climate": str(args.reconstruction_climate.resolve()),
            "source_gdir_root": str(args.source_gdir_root.resolve()),
            "elevation_band_root": str(args.elevation_band_root.resolve()),
            "output_dir": str((args.output_root / tag).resolve()),
            "workdir": str((Path(project_root) / "lrw" / f"s{trial['system_id']}p{trial['point']:04d}").resolve()),
            "timeout_seconds": args.timeout_seconds,
        })
    payloads = [
        payload
        for payload in payloads
        if not (Path(payload["output_dir"]) / "summary.json").exists()
    ]
    for payload in payloads:
        stale_workdir = Path(payload["workdir"])
        if stale_workdir.exists():
            shutil.rmtree(stale_workdir)
    print(f"pending_trials={len(payloads)}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_trial, payload): payload for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"completed {index}/{len(futures)} system={result['system_id']} "
                f"point={result['point']} status={result['status']}",
                flush=True,
            )
    pd.DataFrame(results).sort_values(["system_id", "point"]).to_csv(
        args.output_root / "results.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

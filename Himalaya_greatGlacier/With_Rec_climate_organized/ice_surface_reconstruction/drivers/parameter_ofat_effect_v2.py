"""One-factor-at-a-time parameter effects around representative real chains."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .curve_objective_v2 import CurveObjectiveConfig, score_curve
from .lia_lhs_batch import _run_trial


DEFAULT_GLEN_A = 2.939387691339814e-24
PARAMS = ["delta_t", "melt_f", "prcp_fac_multiplier", "glen_a_multiplier", "temp_melt"]


def _load_bases(paths: list[Path], system_ids: list[int]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    table = pd.concat(frames, ignore_index=True, sort=False)
    table = table[table["system_id"].astype(int).isin(system_ids)].copy()
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna(subset=PARAMS + ["objective_v2", "output_dir"])
    return table.sort_values("objective_v2").drop_duplicates("system_id", keep="first")


def _design(base: pd.Series) -> list[dict]:
    center = {name: float(base[name]) for name in PARAMS}
    rows = [{"system_id": int(base.system_id), "trial": "base", **center}]
    perturbations = {
        "delta_t": (-0.25, 0.25),
        "melt_f": (-0.50, 0.50),
        "prcp_fac_multiplier": (-0.05, 0.05),
        "glen_a_multiplier": (None, None),
        "temp_melt": (-0.20, 0.20),
    }
    for parameter, offsets in perturbations.items():
        for side, offset in zip(("minus", "plus"), offsets):
            row = center.copy()
            if parameter == "glen_a_multiplier":
                row[parameter] = float(
                    np.clip(center[parameter] * math.exp(-0.15 if side == "minus" else 0.15), 0.5, 3.0)
                )
            else:
                row[parameter] = center[parameter] + float(offset)
            bounds = {
                "delta_t": (-3.0, 1.5),
                "melt_f": (2.5, 7.5),
                "prcp_fac_multiplier": (0.8, 1.2),
                "glen_a_multiplier": (0.5, 3.0),
                "temp_melt": (-2.0, 0.0),
            }[parameter]
            row[parameter] = float(np.clip(row[parameter], *bounds))
            rows.append({"system_id": int(base.system_id), "trial": f"{parameter}_{side}", **row})
    return rows


def _payload(row: pd.Series, output_root: Path, system_id: int, trial: str, python_exe: Path) -> dict:
    tag = f"system_{system_id}/{trial}"
    output_dir = (output_root / tag).resolve()
    workdir = (Path(str(row.project_root)) / "lrw" / f"ofat_v2_{system_id}_{trial}").resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if workdir.exists():
        shutil.rmtree(workdir)
    return {
        "system_id": system_id,
        "rgi_id": str(row.rgi_id),
        "point": 0,
        "delta_t": float(row.delta_t),
        "melt_f": float(row.melt_f),
        "prcp_fac_multiplier": float(row.prcp_fac_multiplier),
        "glen_a_multiplier": float(row.glen_a_multiplier),
        "glen_a": DEFAULT_GLEN_A * float(row.glen_a_multiplier),
        "temp_melt": float(row.temp_melt),
        "prcp_fac_initial": float(row.prcp_fac_initial),
        "python_exe": str(python_exe.resolve()),
        "project_root": str(row.project_root),
        "references": str(row.references),
        "stage2_members": str(row.stage2_members),
        "climate_pixels": str(row.climate_pixels),
        "reconstruction_climate": str(row.reconstruction_climate),
        "source_gdir_root": str(row.source_gdir_root),
        "elevation_band_root": str(row.elevation_band_root),
        "output_dir": str(output_dir),
        "workdir": str(workdir),
        "timeout_seconds": 240,
        "ofat_trial": trial,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", type=Path, action="append", required=True)
    parser.add_argument("--targets-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--system-ids", type=int, nargs="+", default=[1044, 1545, 3584])
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--python-exe", type=Path, default=Path(r"D:\Anaconda\envs\oggm_env2\python.exe"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    bases = _load_bases(args.base_results, args.system_ids)
    if len(bases) != len(args.system_ids):
        raise RuntimeError(f"missing representative base: {sorted(set(args.system_ids) - set(bases.system_id.astype(int)))}")
    designs = []
    payloads = []
    for _, base in bases.iterrows():
        for row in _design(base):
            designs.append(row)
            payloads.append(_payload(pd.Series({**base.to_dict(), **row}), args.output_root, int(row["system_id"]), row["trial"], args.python_exe))
    pd.DataFrame(designs).to_csv(args.output_root / "parameter_design.csv", index=False)
    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_trial, payload): payload for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            payload = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {**payload, "status": "crash", "error": f"{type(exc).__name__}:{exc}"}
            results.append(result)
            print(f"completed {index}/{len(futures)} system={payload['system_id']} trial={payload['ofat_trial']} status={result['status']}", flush=True)

    targets = pd.read_csv(args.targets_csv).set_index("lia_unique_id")
    scored = []
    config = CurveObjectiveConfig()
    for result in results:
        record = dict(result)
        try:
            target = targets.loc[int(result["system_id"])]
            series = pd.read_csv(Path(str(result["output_dir"])) / "timeseries.csv")
            record.update(score_curve(series, float(target.lia_registered_length_m), float(target.rgi_registered_length_m), config))
        except Exception as exc:
            record.update({"objective_v2": np.inf, "accepted_curve_v2": False, "failure_reasons_v2": "unscorable", "objective_error_v2": str(exc)})
        scored.append(record)
    frame = pd.DataFrame(scored).sort_values(["system_id", "objective_v2"])
    frame.to_csv(args.output_root / "results_curve_objective_v2.csv", index=False)
    summary = frame.groupby("system_id").agg(
        base_or_trials=("ofat_trial", "size"),
        best_objective=("objective_v2", "min"),
        accepted=("accepted_curve_v2", "sum"),
    ).reset_index()
    summary.to_csv(args.output_root / "summary.csv", index=False)
    (args.output_root / "configuration.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Checkpointed adaptive real-OGGM parameter loop for one LIA system."""

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


PARAMS = ["delta_t", "melt_f", "prcp_fac_multiplier", "glen_a_multiplier", "temp_melt"]
BOUNDS = np.array([[-3.0, 1.5], [2.5, 7.5], [0.8, 1.2], [0.5, 3.0], [-2.0, 0.0]])
GLEN_A_DEFAULT = 2.939387691339814e-24


def _normalize(values: np.ndarray) -> np.ndarray:
    return (values - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def _denormalize(values: np.ndarray) -> np.ndarray:
    return BOUNDS[:, 0] + values * (BOUNDS[:, 1] - BOUNDS[:, 0])


def _initial_pool(paths: list[Path], system_id: int) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[frame["system_id"].astype(int).eq(system_id)].copy()
        required = PARAMS + ["e_lia_current", "e_2000_current", "e_curve_current"]
        if not frame.empty and all(name in frame for name in required):
            frames.append(frame)
    if not frames:
        raise RuntimeError("no-compatible-initial-results")
    pool = pd.concat(frames, ignore_index=True, sort=False)
    pool = pool.replace([np.inf, -np.inf], np.nan).dropna(subset=PARAMS + ["e_curve_current"])
    return pool.drop_duplicates(subset=PARAMS, keep="last")


def _propose(pool: pd.DataFrame, n: int, iteration: int, rng: np.random.Generator) -> np.ndarray:
    valid = pool[
        pool["status"].astype(str).eq("ok")
        & np.isfinite(pd.to_numeric(pool["objective_v2"], errors="coerce"))
    ].sort_values("objective_v2")
    if len(valid) < 5:
        return rng.uniform(0.0, 1.0, size=(n, len(PARAMS)))
    elite = valid.head(min(12, max(5, len(valid) // 4)))
    x = _normalize(elite[PARAMS].to_numpy(float))
    weights = np.exp(-np.arange(len(x), dtype=float) / max(len(x) / 3.0, 1.0))
    weights /= weights.sum()
    mean = np.average(x, axis=0, weights=weights)
    centered = x - mean
    covariance = (centered * weights[:, None]).T @ centered
    covariance += np.eye(len(PARAMS)) * max(0.015, 0.07 * (0.78**iteration)) ** 2
    local_n = max(1, int(round(n * 0.75)))
    proposed = rng.multivariate_normal(mean, covariance, size=local_n)
    global_points = rng.uniform(0.0, 1.0, size=(n - local_n, len(PARAMS)))
    proposed = np.vstack([proposed, global_points])
    return np.clip(proposed, 0.0, 1.0)


def _score_results(
    frame: pd.DataFrame,
    lia: float,
    rgi: float,
    config: CurveObjectiveConfig,
) -> pd.DataFrame:
    frame = frame.copy()
    metrics = []
    for _, row in frame.iterrows():
        path = Path(str(row["output_dir"])) / "timeseries.csv"
        try:
            metrics.append(score_curve(pd.read_csv(path), lia, rgi, config))
        except Exception as exc:
            metrics.append({
                "objective_v2": np.inf,
                "accepted_curve_v2": False,
                "failure_reasons_v2": "unscorable",
                "objective_error_v2": f"{type(exc).__name__}:{exc}",
            })
    metric_frame = pd.DataFrame(metrics, index=frame.index)
    for column in metric_frame:
        frame[column] = metric_frame[column]
    frame["target_lia_length_m"] = lia
    frame["target_rgi_length_m"] = rgi
    frame["e_lia_current"] = frame.get("e_lia_v2", np.nan)
    frame["e_2000_current"] = frame.get("e_2000_v2", np.nan)
    frame["e_curve_current"] = frame[
        ["e_lia_current", "e_2000_current"]
    ].sum(axis=1)
    return frame


def _payloads(design: pd.DataFrame, base: pd.Series, output_root: Path, iteration: int) -> list[dict]:
    payloads = []
    for _, row in design.iterrows():
        candidate = int(row["candidate"])
        tag = f"iteration_{iteration:02d}/candidate_{candidate:03d}"
        output_dir = (output_root / tag).resolve()
        workdir = (Path(str(base.project_root)) / "lrw" / f"longloop_{int(base.system_id)}_{iteration:02d}_{candidate:03d}").resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if workdir.exists():
            shutil.rmtree(workdir)
        payloads.append({
            "system_id": int(base.system_id),
            "rgi_id": str(base.rgi_id),
            "point": candidate,
            **{name: float(row[name]) for name in PARAMS},
            "glen_a": GLEN_A_DEFAULT * float(row["glen_a_multiplier"]),
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
            "loop_iteration": iteration,
            "candidate": candidate,
        })
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", type=int, required=True)
    parser.add_argument("--initial-results", type=Path, action="append", required=True)
    parser.add_argument("--canonical-base", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=12)
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Deprecated alias for endpoint tolerance")
    parser.add_argument("--endpoint-tolerance", type=float, default=0.10)
    parser.add_argument("--spinup-volume-tolerance", type=float, default=0.02)
    parser.add_argument("--spinup-stability-years", type=int, default=200)
    parser.add_argument("--corridor-buffer-fraction", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pool = _initial_pool(args.initial_results, args.system_id)
    base_table = pd.read_csv(args.canonical_base)
    base = base_table[base_table["system_id"].astype(int).eq(args.system_id)].iloc[0]
    target_table = pd.read_csv(args.targets)
    target = target_table[target_table["lia_unique_id"].astype(int).eq(args.system_id)].iloc[0]
    lia = float(target.lia_registered_length_m)
    rgi = float(target.rgi_registered_length_m)
    endpoint_tolerance = args.endpoint_tolerance if args.tolerance is None else args.tolerance
    objective_config = CurveObjectiveConfig(
        endpoint_tolerance=endpoint_tolerance,
        spinup_volume_tolerance=args.spinup_volume_tolerance,
        spinup_stability_years=args.spinup_stability_years,
        corridor_buffer_fraction=args.corridor_buffer_fraction,
    )
    pool = _score_results(pool, lia, rgi, objective_config)
    rng = np.random.default_rng(args.seed)
    best_score = float(pool.loc[pool.status.astype(str).eq("ok"), "objective_v2"].min())
    stale_rounds = 0
    history = []

    for iteration in range(1, args.iterations + 1):
        iteration_dir = args.output_root / f"iteration_{iteration:02d}"
        scored_path = iteration_dir / "results_scored.csv"
        if scored_path.exists():
            scored = _score_results(pd.read_csv(scored_path), lia, rgi, objective_config)
        else:
            proposed_unit = _propose(pool, args.ensemble_size, iteration, rng)
            proposed = _denormalize(proposed_unit)
            design = pd.DataFrame(proposed, columns=PARAMS)
            design.insert(0, "candidate", np.arange(len(design), dtype=int))
            iteration_dir.mkdir(parents=True, exist_ok=True)
            design.to_csv(iteration_dir / "parameter_design.csv", index=False)
            results = []
            with ProcessPoolExecutor(max_workers=args.processes) as executor:
                futures = {executor.submit(_run_trial, payload): payload for payload in _payloads(design, base, args.output_root, iteration)}
                for future in as_completed(futures):
                    payload = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # keep the iteration resumable
                        result = {**payload, "status": "crash", "error": f"pool-worker:{type(exc).__name__}:{exc}"}
                    results.append(result)
                    print(f"iteration={iteration} candidate={result['candidate']} status={result['status']}", flush=True)
            scored = _score_results(pd.DataFrame(results), lia, rgi, objective_config)
            scored.sort_values("candidate").to_csv(scored_path, index=False)
        pool = pd.concat([pool, scored], ignore_index=True, sort=False).drop_duplicates(subset=PARAMS, keep="last")
        valid = pool[
            pool.status.astype(str).eq("ok")
            & np.isfinite(pd.to_numeric(pool["objective_v2"], errors="coerce"))
        ].sort_values("objective_v2")
        best = valid.iloc[0]
        current_score = float(best.objective_v2)
        improved = current_score < best_score - 0.002
        if improved:
            best_score = current_score
            stale_rounds = 0
        else:
            stale_rounds += 1
        accepted = bool(best.accepted_curve_v2)
        record = {
            "iteration": iteration,
            "best_objective_v2": current_score,
            "best_e_lia": float(best.e_lia_v2),
            "best_e_2000": float(best.e_2000_v2),
            "best_spinup_volume_change_last_200": float(best.spinup_volume_change_last_200),
            "best_outside_fraction": float(best.outside_fraction_v2),
            "best_max_corridor_exceed_m": float(best.max_corridor_exceed_m),
            "best_failure_reasons": str(best.failure_reasons_v2),
            "accepted_length_screen": accepted,
            "stale_rounds": stale_rounds,
            **{name: float(best[name]) for name in PARAMS},
            "best_output_dir": str(best.get("output_dir", "")),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(args.output_root / "loop_history.csv", index=False)
        (args.output_root / "loop_state.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        if accepted or stale_rounds >= args.patience:
            break
    pool.to_csv(args.output_root / "all_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

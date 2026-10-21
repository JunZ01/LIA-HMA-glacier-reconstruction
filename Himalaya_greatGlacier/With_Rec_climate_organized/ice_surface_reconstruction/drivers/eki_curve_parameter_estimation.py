"""ES-MDA/Ensemble-Kalman inverse update from saved LIA trial chains.

This is a first parameter-estimation pass. It uses the saved OGGM ensemble as
the forward ensemble and does not launch new simulations. The normalized
observation vector is [spin-up length / LIA length, forward peak length /
LIA length, 2000 length / RGI length].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PARAMS = ["delta_t", "melt_f", "prcp_fac_multiplier", "glen_a_multiplier", "temp_melt"]
BOUNDS = np.array([[-3.0, 1.5], [2.5, 7.5], [0.8, 1.2], [0.5, 3.0], [-2.0, 0.0]])
TARGET = np.ones(3, dtype=float)


def _peak_forward(row: pd.Series) -> float:
    path = Path(str(row["output_dir"])) / "timeseries.csv"
    frame = pd.read_csv(path)
    forward = frame[frame["phase"].astype(str).eq("forward")]
    if forward.empty:
        return np.nan
    return float(forward["length_m"].astype(float).max())


def _load_ensemble(results_csv: Path, system_id: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    table = pd.read_csv(results_csv)
    table = table[table["system_id"].astype(int).eq(system_id)].copy()
    table = table[table["status"].astype(str).eq("ok")].copy()
    table["peak_forward_m"] = table.apply(_peak_forward, axis=1)
    table = table.replace([np.inf, -np.inf], np.nan).dropna(subset=PARAMS + ["l_spinup_m", "l_2000_m", "peak_forward_m"])
    if len(table) < 10:
        raise RuntimeError(f"too-few-valid-trials:{system_id}:{len(table)}")
    first_dir = Path(str(table.iloc[0]["output_dir"]))
    summary = json.loads((first_dir / "summary.json").read_text(encoding="utf-8"))
    lia = float(table.iloc[0]["target_lia_length_m"]) if "target_lia_length_m" in table else float(summary["targets"]["lia_length_m"])
    rgi = float(table.iloc[0]["target_rgi_length_m"]) if "target_rgi_length_m" in table else float(summary["targets"]["rgi_length_m"])
    x = table[PARAMS].to_numpy(float)
    y = np.column_stack([
        table["l_spinup_m"].to_numpy(float) / lia,
        table["peak_forward_m"].to_numpy(float) / lia,
        table["l_2000_m"].to_numpy(float) / rgi,
    ])
    return table, x, y, np.array([lia, rgi], dtype=float)


def _es_mda(x: np.ndarray, y: np.ndarray, iterations: int, noise: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    current_x = x.copy()
    current_y = y.copy()
    # Equal tempering keeps the update stable while retaining the full ensemble.
    r = np.diag(noise**2 * iterations)
    for _ in range(iterations):
        xc = current_x - current_x.mean(axis=0)
        yc = current_y - current_y.mean(axis=0)
        cxy = xc.T @ yc / max(len(current_x) - 1, 1)
        cyy = yc.T @ yc / max(len(current_x) - 1, 1)
        gain = cxy @ np.linalg.pinv(cyy + r)
        perturbed = TARGET + rng.normal(0.0, noise, size=current_y.shape)
        current_x = current_x + (perturbed - current_y) @ gain.T
        current_x = np.clip(current_x, BOUNDS[:, 0], BOUNDS[:, 1])
        # Re-evaluate the linearized response with the local ensemble map. This
        # preserves the EKI update without pretending a new OGGM run occurred.
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(current_x)), x]), y, rcond=None)
        current_y = np.column_stack([np.ones(len(current_x)), current_x]) @ beta
    return current_x, current_y


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for system_id in args.system_id:
        table, x, y, targets = _load_ensemble(args.results_csv, system_id)
        updated_x, updated_y = _es_mda(x, y, args.iterations, np.array(args.noise), args.seed + system_id)
        score = np.sum(np.abs(updated_y - TARGET), axis=1)
        order = np.argsort(score)
        for rank, idx in enumerate(order):
            rows.append({
                "system_id": system_id,
                "eki_rank": rank + 1,
                "source_point": int(table.iloc[idx]["point"]),
                **{name: float(updated_x[idx, j]) for j, name in enumerate(PARAMS)},
                "norm_l_spinup": float(updated_y[idx, 0]),
                "norm_l_peak_forward": float(updated_y[idx, 1]),
                "norm_l_2000": float(updated_y[idx, 2]),
                "e_lia": float(abs(updated_y[idx, 0] - 1.0)),
                "e_peak": float(abs(updated_y[idx, 1] - 1.0)),
                "e_2000": float(abs(updated_y[idx, 2] - 1.0)),
                "e_total": float(score[idx]),
                "lia_length_m": float(targets[0]),
                "rgi_length_m": float(targets[1]),
                "is_candidate": bool(score[idx] <= args.accept_score and abs(updated_y[idx, 1] - 1.0) <= args.peak_tolerance),
            })
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "eki_parameter_ensemble.csv", index=False)
    result.groupby("system_id", as_index=False).head(args.top_n).to_csv(args.output_dir / "eki_top_candidates.csv", index=False)
    diagnostics = result.groupby("system_id").agg(
        ensemble_size=("system_id", "size"),
        min_e_total=("e_total", "min"),
        candidates=("is_candidate", "sum"),
    ).reset_index()
    diagnostics.to_csv(args.output_dir / "eki_diagnostics.csv", index=False)
    print(diagnostics.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-id", type=int, nargs="+", default=[1044, 1545, 3584])
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--noise", type=float, nargs=3, default=[0.04, 0.05, 0.04])
    parser.add_argument("--accept-score", type=float, default=0.15)
    parser.add_argument("--peak-tolerance", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260806)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

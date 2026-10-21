"""Fit bounded GP response surfaces and propose real OGGM candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler


PARAMS = ["delta_t", "melt_f", "prcp_fac_multiplier", "glen_a_multiplier", "temp_melt"]
BOUNDS = np.array([[-3.0, 1.5], [2.5, 7.5], [0.8, 1.2], [0.5, 3.0], [-2.0, 0.0]])


def _fit_gp(x: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(x.shape[1]), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=20260806)
    model.fit(x, y)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-id", type=int, nargs="+", default=[1044, 1545, 3584])
    parser.add_argument("--sobol-points", type=int, default=16384)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.results_csv)
    all_top = []
    sampler = qmc.Sobol(d=len(PARAMS), scramble=True, seed=args.seed)
    unit_candidates = sampler.random_base2(int(np.log2(args.sobol_points)))
    for sid in args.system_id:
        rows = table[(table["system_id"].astype(int) == sid) & table["status"].astype(str).eq("ok")].copy()
        rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=PARAMS + ["l_spinup_m", "l_2000_m", "peak_forward_m", "target_lia_length_m", "target_rgi_length_m"])
        if len(rows) < 15:
            raise RuntimeError(f"too-few-valid-trials:{sid}:{len(rows)}")
        x_raw = rows[PARAMS].to_numpy(float)
        x_min = np.maximum(BOUNDS[:, 0], x_raw.min(axis=0) - 0.10 * (BOUNDS[:, 1] - BOUNDS[:, 0]))
        x_max = np.minimum(BOUNDS[:, 1], x_raw.max(axis=0) + 0.10 * (BOUNDS[:, 1] - BOUNDS[:, 0]))
        x_scaler = MinMaxScaler().fit(np.vstack([x_min, x_max]))
        x = x_scaler.transform(x_raw)
        lia = float(rows.iloc[0]["target_lia_length_m"])
        rgi = float(rows.iloc[0]["target_rgi_length_m"])
        y = np.column_stack([
            rows["l_spinup_m"].to_numpy(float) / lia,
            rows["peak_forward_m"].to_numpy(float) / lia,
            rows["l_2000_m"].to_numpy(float) / rgi,
        ])
        gps = [_fit_gp(x, y[:, j]) for j in range(y.shape[1])]
        raw_candidates = x_min + unit_candidates * (x_max - x_min)
        candidates = x_scaler.transform(raw_candidates)
        means = np.column_stack([gp.predict(candidates) for gp in gps])
        stds = np.column_stack([gp.predict(candidates, return_std=True)[1] for gp in gps])
        objective = np.abs(means - 1.0).sum(axis=1)
        acquisition = objective - 0.35 * stds.sum(axis=1)
        order = np.argsort(acquisition)[: args.top_n]
        for rank, idx in enumerate(order, start=1):
            all_top.append({
                "system_id": sid,
                "rank": rank,
                **{name: float(raw_candidates[idx, j]) for j, name in enumerate(PARAMS)},
                "pred_norm_l_spinup": float(means[idx, 0]),
                "pred_norm_l_peak_forward": float(means[idx, 1]),
                "pred_norm_l_2000": float(means[idx, 2]),
                "pred_e_curve": float(objective[idx]),
                "acquisition": float(acquisition[idx]),
                "pred_std_sum": float(stds[idx].sum()),
                "lia_length_m": lia,
                "rgi_length_m": rgi,
                "training_size": len(rows),
                "training_bounds": str(list(zip(x_min, x_max))),
            })
        pd.DataFrame({"system_id": [sid], "training_size": [len(rows)], "min_training_e_curve": [float(np.abs(y - 1).sum(axis=1).min())], "candidate_min_pred_e_curve": [float(objective[order[0]])]}).to_csv(args.output_dir / f"system_{sid}_diagnostics.csv", index=False)
    output = pd.DataFrame(all_top)
    output.to_csv(args.output_dir / "surrogate_candidates.csv", index=False)
    print(output.groupby("system_id").head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

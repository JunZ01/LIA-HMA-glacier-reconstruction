"""Quantify parameter-response sensitivity from saved, rescored chains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor


PARAMS = ["delta_t", "melt_f", "prcp_fac_multiplier", "glen_a_multiplier", "temp_melt"]
METRICS = ["e_lia_current", "e_peak_current", "e_2000_current", "e_curve_current"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.results_csv)
    records = []
    for sid, group in table.groupby(table["system_id"].astype(int)):
        frame = group[group["status"].astype(str).eq("ok")].replace([np.inf, -np.inf], np.nan).dropna(subset=PARAMS + METRICS)
        if len(frame) < 8:
            continue
        x = frame[PARAMS].to_numpy(float)
        model = ExtraTreesRegressor(n_estimators=300, random_state=20260806, min_samples_leaf=2)
        model.fit(x, frame["e_curve_current"].to_numpy(float))
        importances = model.feature_importances_
        for j, param in enumerate(PARAMS):
            row = {"system_id": sid, "parameter": param, "n_valid": len(frame), "extra_trees_importance_e_curve": float(importances[j])}
            for metric in METRICS:
                rho, pvalue = spearmanr(frame[param].to_numpy(float), frame[metric].to_numpy(float))
                row[f"spearman_{metric}"] = float(rho) if np.isfinite(rho) else np.nan
                row[f"spearman_p_{metric}"] = float(pvalue) if np.isfinite(pvalue) else np.nan
            records.append(row)
    result = pd.DataFrame(records)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(result.sort_values(["system_id", "extra_trees_importance_e_curve"], ascending=[True, False]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

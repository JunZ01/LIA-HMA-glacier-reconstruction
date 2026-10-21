"""Estimate marginal parameter effects from saved curve-objective chains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor


PARAMS = ["delta_t", "melt_f", "prcp_fac_multiplier", "glen_a_multiplier", "temp_melt"]
METRICS = [
    "e_lia_v2",
    "e_2000_v2",
    "spinup_volume_change_last_200",
    "corridor_rmse_normalized_v2",
    "outside_fraction_v2",
    "objective_v2",
]


def _finite_frame(group: pd.DataFrame) -> pd.DataFrame:
    columns = PARAMS + METRICS
    return (
        group.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=columns)
        .copy()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.results_csv)
    rows: list[dict] = []
    for system_id, group in table.groupby(table["system_id"].astype(int)):
        frame = _finite_frame(group)
        if len(frame) < 8:
            continue
        for parameter in PARAMS:
            values = frame[parameter].astype(float)
            low_cut = values.quantile(0.25)
            high_cut = values.quantile(0.75)
            low = frame[values <= low_cut]
            high = frame[values >= high_cut]
            for metric in METRICS:
                rho, pvalue = spearmanr(values, frame[metric].astype(float))
                rows.append({
                    "system_id": int(system_id),
                    "parameter": parameter,
                    "metric": metric,
                    "n_total": len(frame),
                    "n_low_q25": len(low),
                    "n_high_q75": len(high),
                    "parameter_low_q25": float(low_cut),
                    "parameter_high_q75": float(values.quantile(0.75)),
                    "metric_low_median": float(low[metric].median()),
                    "metric_high_median": float(high[metric].median()),
                    "high_minus_low": float(high[metric].median() - low[metric].median()),
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "spearman_p": float(pvalue) if np.isfinite(pvalue) else np.nan,
                })

        x = frame[PARAMS].to_numpy(float)
        model = ExtraTreesRegressor(
            n_estimators=400,
            random_state=20260807,
            min_samples_leaf=3,
        )
        model.fit(x, frame["objective_v2"].to_numpy(float))
        importance_rows = [
            {
                "system_id": int(system_id),
                "parameter": parameter,
                "metric": "objective_v2_extra_trees_importance",
                "n_total": len(frame),
                "n_low_q25": np.nan,
                "n_high_q75": np.nan,
                "parameter_low_q25": np.nan,
                "parameter_high_q75": np.nan,
                "metric_low_median": np.nan,
                "metric_high_median": np.nan,
                "high_minus_low": np.nan,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "extra_trees_importance": float(model.feature_importances_[index]),
            }
            for index, parameter in enumerate(PARAMS)
        ]
        rows.extend(importance_rows)

    result = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(result[result["metric"].eq("objective_v2")].to_string(index=False))
    print(result[result["metric"].eq("objective_v2_extra_trees_importance")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

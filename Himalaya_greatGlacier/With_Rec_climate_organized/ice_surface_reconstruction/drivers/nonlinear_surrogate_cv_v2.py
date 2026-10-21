"""External validation of a nonlinear parameter-to-curve surrogate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score


PARAMS = ["delta_t", "melt_f", "prcp_fac_multiplier", "glen_a_multiplier", "temp_melt"]
TARGETS = [
    "objective_v2",
    "e_lia_v2",
    "e_2000_v2",
    "spinup_volume_change_last_200",
    "corridor_rmse_normalized_v2",
]


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=PARAMS + TARGETS).copy()
    return result.drop_duplicates(subset=PARAMS, keep="last")


def _source_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["test_source"] = path.parent.name + "/" + path.name
    if "ofat_trial" in frame:
        frame = frame[~frame["ofat_trial"].astype(str).eq("base")].copy()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r2-threshold", type=float, default=0.5)
    parser.add_argument("--spearman-threshold", type=float, default=0.7)
    parser.add_argument("--top-recall-threshold", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = _clean(pd.read_csv(args.train_csv))
    test = _clean(pd.concat([_source_frame(path) for path in args.test_csv], ignore_index=True, sort=False))

    metric_rows = []
    prediction_rows = []
    for system_id in sorted(set(train.system_id.astype(int)).intersection(test.system_id.astype(int))):
        train_system = _clean(train[train.system_id.astype(int).eq(system_id)])
        test_system = _clean(test[test.system_id.astype(int).eq(system_id)])
        if len(train_system) < 20 or len(test_system) < 5:
            continue
        x_train = train_system[PARAMS].to_numpy(float)
        x_test = test_system[PARAMS].to_numpy(float)
        target_predictions = {}
        for target in TARGETS:
            y_train = np.log1p(train_system[target].to_numpy(float))
            y_test = np.log1p(test_system[target].to_numpy(float))
            model = ExtraTreesRegressor(
                n_estimators=600,
                random_state=20260807,
                min_samples_leaf=2,
                max_features=1.0,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            predicted_log = model.predict(x_test)
            predicted = np.expm1(predicted_log)
            target_predictions[target] = predicted
            rho, pvalue = spearmanr(y_test, predicted_log)
            row = {
                "system_id": system_id,
                "target": target,
                "n_train": len(train_system),
                "n_external_test": len(test_system),
                "r2_log1p": float(r2_score(y_test, predicted_log)),
                "mae_log1p": float(mean_absolute_error(y_test, predicted_log)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(pvalue) if np.isfinite(pvalue) else np.nan,
            }
            if target == "objective_v2":
                count = max(2, int(np.ceil(0.20 * len(test_system))))
                actual_top = set(np.argsort(y_test)[:count])
                predicted_top = set(np.argsort(predicted_log)[:count])
                recall = len(actual_top.intersection(predicted_top)) / len(actual_top)
                row["top20_recall"] = recall
                row["eligible_to_propose"] = bool(
                    row["r2_log1p"] >= args.r2_threshold
                    and row["spearman_rho"] >= args.spearman_threshold
                    and recall >= args.top_recall_threshold
                )
            metric_rows.append(row)

        for index, (_, test_row) in enumerate(test_system.iterrows()):
            prediction_rows.append({
                "system_id": system_id,
                "test_source": test_row.get("test_source", ""),
                "output_dir": test_row.get("output_dir", ""),
                **{name: float(test_row[name]) for name in PARAMS},
                **{f"actual_{target}": float(test_row[target]) for target in TARGETS},
                **{f"predicted_{target}": float(target_predictions[target][index]) for target in TARGETS},
            })

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.output_dir / "external_validation_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "external_predictions.csv", index=False)
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

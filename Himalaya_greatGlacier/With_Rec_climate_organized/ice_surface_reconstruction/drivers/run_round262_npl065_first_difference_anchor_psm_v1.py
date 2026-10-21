from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CALIBRATION_DIFFERENCE_YEARS = np.arange(1961, 2014, dtype=int)
HOLDOUT_YEARS = np.arange(1952, 1960, dtype=int)
BACKCAST_DIFFERENCE_YEARS = np.arange(1953, 1961, dtype=int)
OOF_BLOCKS = (
    (1961, 1969),
    (1970, 1978),
    (1979, 1987),
    (1988, 1996),
    (1997, 2005),
    (2006, 2013),
)
METHODS = {
    "signal_free_constrained": ("signal_free_chronology", "signal_free_prediction"),
    "modnegex": ("modnegex_chronology", "modnegex_prediction"),
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise RuntimeError("nonfinite-regression-input")
    return np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]


def skill(observed: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> float:
    sse = float(np.sum((observed - predicted) ** 2))
    reference = float(np.sum((observed - baseline) ** 2))
    return float(1.0 - sse / reference)


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "round_258_npl065_precalibration_holdout_v2/precalibration_holdout_curves.csv"
    source_summary = loop / "round_258_npl065_precalibration_holdout_v2/summary.json"
    output = loop / "round_262_npl065_first_difference_anchor_psm_v1"
    output.mkdir(parents=True, exist_ok=False)

    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "round258_curves": source,
        "round258_summary": source_summary,
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    frame = pd.read_csv(source).sort_values("year").reset_index(drop=True)
    years = frame["year"].to_numpy(int)
    if not np.array_equal(years, np.arange(1952, 2014, dtype=int)):
        raise RuntimeError("unexpected-year-axis")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-present")

    target = frame["cru_box_djf_tmin"].to_numpy(float)
    target_by_year = dict(zip(years, target))
    calibration_level_mean = float(frame.loc[frame["year"] >= 1960, "cru_box_djf_tmin"].mean())
    metrics_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    holdout_output = pd.DataFrame({
        "year": HOLDOUT_YEARS,
        "cru_box_djf_tmin": [target_by_year[int(year)] for year in HOLDOUT_YEARS],
    })

    for method, (proxy_column, original_prediction_column) in METHODS.items():
        proxy = frame[proxy_column].to_numpy(float)
        proxy_by_year = dict(zip(years, proxy))
        difference_years = years[1:]
        delta_target = np.diff(target)
        delta_proxy = np.diff(proxy)
        calibration_mask = np.isin(difference_years, CALIBRATION_DIFFERENCE_YEARS)
        x_cal = delta_proxy[calibration_mask]
        y_cal = delta_target[calibration_mask]
        cal_years = difference_years[calibration_mask]

        oof_prediction = np.full(len(cal_years), np.nan)
        oof_baseline = np.full(len(cal_years), np.nan)
        slopes: list[float] = []
        fold_correlations: list[float] = []
        for fold_index, (start, end) in enumerate(OOF_BLOCKS, start=1):
            test = (cal_years >= start) & (cal_years <= end)
            train = ~test
            if not test.any() or train.sum() < 2:
                raise RuntimeError(f"invalid-fold:{start}-{end}")
            beta = fit_ols(x_cal[train], y_cal[train])
            prediction = beta[0] + beta[1] * x_cal[test]
            baseline = np.full(test.sum(), float(y_cal[train].mean()))
            oof_prediction[test] = prediction
            oof_baseline[test] = baseline
            fold_r = correlation(prediction, y_cal[test])
            slopes.append(float(beta[1]))
            fold_correlations.append(fold_r)
            fold_rows.append({
                "method": method,
                "fold": fold_index,
                "test_start": start,
                "test_end": end,
                "test_year_count": int(test.sum()),
                "fit_intercept": float(beta[0]),
                "fit_slope": float(beta[1]),
                "test_correlation": fold_r,
                "test_rmse_degC": float(np.sqrt(np.mean((prediction - y_cal[test]) ** 2))),
                "test_skill_vs_training_mean_delta": skill(y_cal[test], prediction, baseline),
            })
        if not np.isfinite(oof_prediction).all():
            raise RuntimeError(f"incomplete-oof:{method}")
        for year, observed, predicted, baseline in zip(cal_years, y_cal, oof_prediction, oof_baseline):
            oof_rows.append({
                "method": method,
                "difference_year": int(year),
                "observed_delta_t_degC": float(observed),
                "oof_predicted_delta_t_degC": float(predicted),
                "oof_baseline_delta_t_degC": float(baseline),
            })

        full_beta = fit_ols(x_cal, y_cal)
        predicted_delta = {
            int(year): float(full_beta[0] + full_beta[1] * (proxy_by_year[int(year)] - proxy_by_year[int(year) - 1]))
            for year in BACKCAST_DIFFERENCE_YEARS
        }
        anchored = {1960: float(target_by_year[1960])}
        for year in range(1960, 1952, -1):
            anchored[year - 1] = anchored[year] - predicted_delta[year]
        holdout_observed = np.asarray([target_by_year[int(year)] for year in HOLDOUT_YEARS], dtype=float)
        holdout_predicted = np.asarray([anchored[int(year)] for year in HOLDOUT_YEARS], dtype=float)
        original_predicted = frame.set_index("year").loc[HOLDOUT_YEARS, original_prediction_column].to_numpy(float)
        holdout_baseline = np.full(len(HOLDOUT_YEARS), calibration_level_mean)
        observed_difference = np.diff(holdout_observed)
        predicted_difference = np.diff(holdout_predicted)
        original_difference = np.diff(original_predicted)

        anchored_rmse = float(np.sqrt(np.mean((holdout_predicted - holdout_observed) ** 2)))
        anchored_bias = float(np.mean(holdout_predicted - holdout_observed))
        original_rmse = float(np.sqrt(np.mean((original_predicted - holdout_observed) ** 2)))
        original_bias = float(np.mean(original_predicted - holdout_observed))
        oof_r = correlation(oof_prediction, y_cal)
        oof_skill = skill(y_cal, oof_prediction, oof_baseline)
        holdout_diff_r = correlation(predicted_difference, observed_difference)
        gate_components = {
            "oof_delta_correlation_at_least_0p30": bool(oof_r >= 0.30),
            "oof_delta_skill_positive": bool(oof_skill > 0.0),
            "anchored_holdout_abs_bias_improves": bool(abs(anchored_bias) < abs(original_bias)),
            "anchored_holdout_rmse_improves": bool(anchored_rmse < original_rmse),
            "holdout_first_difference_correlation_at_least_0p50": bool(holdout_diff_r >= 0.50),
        }
        admitted = all(gate_components.values())
        metrics_rows.append({
            "method": method,
            "full_delta_fit_intercept": float(full_beta[0]),
            "full_delta_fit_slope": float(full_beta[1]),
            "oof_delta_correlation": oof_r,
            "oof_delta_skill_vs_training_mean": oof_skill,
            "oof_delta_rmse_degC": float(np.sqrt(np.mean((oof_prediction - y_cal) ** 2))),
            "positive_slope_fold_count": int(np.sum(np.asarray(slopes) > 0)),
            "positive_test_correlation_fold_count": int(np.sum(np.asarray(fold_correlations) > 0)),
            "anchored_holdout_correlation": correlation(holdout_predicted, holdout_observed),
            "anchored_holdout_rmse_degC": anchored_rmse,
            "anchored_holdout_bias_degC": anchored_bias,
            "anchored_holdout_skill_vs_calibration_climatology": skill(holdout_observed, holdout_predicted, holdout_baseline),
            "anchored_holdout_first_difference_correlation": holdout_diff_r,
            "anchored_holdout_first_difference_sign_agreement": float(np.mean(np.sign(predicted_difference) == np.sign(observed_difference))),
            "original_level_holdout_rmse_degC": original_rmse,
            "original_level_holdout_bias_degC": original_bias,
            "original_level_holdout_skill_vs_calibration_climatology": skill(holdout_observed, original_predicted, holdout_baseline),
            "original_level_holdout_first_difference_correlation": correlation(original_difference, observed_difference),
            "rmse_improvement_degC": original_rmse - anchored_rmse,
            "absolute_bias_improvement_degC": abs(original_bias) - abs(anchored_bias),
            **gate_components,
            "admitted_frequency_separated_operator": admitted,
        })
        holdout_output[f"{method}_anchored_prediction"] = holdout_predicted
        holdout_output[f"{method}_original_level_prediction"] = original_predicted

    metrics = pd.DataFrame(metrics_rows)
    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    metrics.to_csv(output / "first_difference_anchor_metrics.csv", index=False)
    folds.to_csv(output / "first_difference_anchor_fold_metrics.csv", index=False)
    oof.to_csv(output / "first_difference_oof_curves.csv", index=False)
    holdout_output.to_csv(output / "first_difference_anchor_holdout_curves.csv", index=False)

    colors = {"signal_free_constrained": "#D55E00", "modnegex": "#0072B2"}
    figure, axes = plt.subplots(2, 1, figsize=(11.8, 8.2), constrained_layout=True)
    axes[0].plot(HOLDOUT_YEARS, holdout_output["cru_box_djf_tmin"], color="#111111", linewidth=2.0, marker="o", label="CRU DJF Tmin")
    for method in METHODS:
        label = "Signal-free anchored" if method == "signal_free_constrained" else "ModNegEx anchored"
        axes[0].plot(HOLDOUT_YEARS, holdout_output[f"{method}_anchored_prediction"], color=colors[method], linewidth=1.6, marker="o", label=label)
        axes[0].plot(HOLDOUT_YEARS, holdout_output[f"{method}_original_level_prediction"], color=colors[method], linewidth=1.0, linestyle=":", alpha=0.7, label=label.replace("anchored", "original level"))
    axes[0].set_title("1952-1959 true holdout: 1960-anchored backward integration")
    axes[0].set_ylabel("degC")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)

    observed_delta = np.diff(holdout_output["cru_box_djf_tmin"].to_numpy(float))
    axes[1].plot(HOLDOUT_YEARS[1:], observed_delta, color="#111111", linewidth=2.0, marker="o", label="Observed delta Tmin")
    for method in METHODS:
        predicted_delta = np.diff(holdout_output[f"{method}_anchored_prediction"].to_numpy(float))
        row = metrics.loc[metrics["method"] == method].iloc[0]
        label = ("Signal-free" if method == "signal_free_constrained" else "ModNegEx") + f"; r={row['anchored_holdout_first_difference_correlation']:.3f}"
        axes[1].plot(HOLDOUT_YEARS[1:], predicted_delta, color=colors[method], linewidth=1.5, marker="o", label=label)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_title("Year-to-year temperature changes inside the holdout")
    axes[1].set_xlabel("Ring year")
    axes[1].set_ylabel("delta degC")
    axes[1].legend(frameon=False, ncol=3, fontsize=9)
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "npl065_first_difference_anchor_psm.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    admitted = metrics.loc[metrics["admitted_frequency_separated_operator"], "method"].tolist()
    if admitted:
        decision = "admit_first_difference_operator_to_independent_multi_record_generalization"
    else:
        decision = "close_npl065_frequency_separated_operator_retain_as_validation_evidence_only"
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "design": {
            "calibration_difference_years": [1961, 2013],
            "oof_blocks": [list(block) for block in OOF_BLOCKS],
            "holdout_years": [1952, 1959],
            "level_anchor": "observed_CRU_DJF_Tmin_at_1960",
            "operator": "OLS_delta_T_equals_intercept_plus_slope_times_delta_chronology",
        },
        "metrics": metrics_rows,
        "admitted_methods": admitted,
        "decision": decision,
        "interpretation_boundary": "offline_frequency_role_test_not_a_production_or_paleo_reconstruction",
        "locked_1921_1939_values_accessed": False,
        "paleo_proxy_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "rcsigfree_binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")

    lines = [
        "# Round 262 NPL065 first-difference anchored PSM",
        "",
        "The chronology predicts only annual temperature differences. Absolute level is anchored to observed 1960 CRU DJF Tmin and integrated backward into the untouched 1952-1959 holdout.",
        "",
    ]
    for row in metrics_rows:
        lines.extend([
            f"## {row['method']}",
            "",
            f"- Blocked OOF delta correlation / skill: `{row['oof_delta_correlation']:.3f}` / `{row['oof_delta_skill_vs_training_mean']:.3f}`.",
            f"- Anchored holdout RMSE / bias: `{row['anchored_holdout_rmse_degC']:.3f}` / `{row['anchored_holdout_bias_degC']:.3f} degC`.",
            f"- Original level holdout RMSE / bias: `{row['original_level_holdout_rmse_degC']:.3f}` / `{row['original_level_holdout_bias_degC']:.3f} degC`.",
            f"- Anchored holdout first-difference r: `{row['anchored_holdout_first_difference_correlation']:.3f}`.",
            f"- Admitted: `{row['admitted_frequency_separated_operator']}`.",
            "",
        ])
    lines.extend([f"Decision: `{decision}`.", ""])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines), encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

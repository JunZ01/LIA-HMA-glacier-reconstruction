"""Analyze held-out proxy response drift and its EnKF contribution relevance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_fold5_response_drift_v1"
    paths = {
        "metrics": output / "response_drift_metrics.mat",
        "attribution": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_fold5_attribution_v1/proxy9_fold5_contribution_attribution.csv",
        "operators": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    scalar_names = (
        "observed_count",
        "proxy_trend",
        "local_temperature_trend",
        "local_precipitation_trend",
        "proxy_temperature_correlation",
        "proxy_precipitation_correlation",
        "archived_prediction_trend",
        "fold5_prediction_trend",
        "archived_residual_trend",
        "fold5_residual_trend",
        "archived_prediction_correlation",
        "fold5_prediction_correlation",
    )
    with h5py.File(paths["metrics"], "r") as handle:
        data = {name: np.asarray(handle[name], dtype=float).squeeze() for name in scalar_names}
        years = np.asarray(handle["test_years"], dtype=float).squeeze().astype(int)
        proxy_values = np.asarray(handle["proxy_values"], dtype=float).T
        temperature_values = np.asarray(handle["temperature_values"], dtype=float).T
        fold5_prediction = np.asarray(handle["fold5_prediction"], dtype=float).T
    table = pd.DataFrame({"proxy9_record_1based": np.arange(1, 298), **data})
    table["fold5_prediction_trend_error"] = table["fold5_prediction_trend"] - table["proxy_trend"]
    table["archived_prediction_trend_error"] = table["archived_prediction_trend"] - table["proxy_trend"]
    table["fold5_response_sign_mismatch"] = np.sign(table["fold5_prediction_trend"]).ne(np.sign(table["proxy_trend"]))
    table = table.merge(pd.read_csv(paths["attribution"]), on="proxy9_record_1based", how="left", suffixes=("", "_attrib"))
    table.to_csv(output / "proxy9_fold5_response_drift_attribution.csv", index=False)

    valid = table["observed_count"] >= 3
    negative = table["trend_contribution_change_c_per_century"] < 0
    sign_summary = (
        table[valid]
        .groupby("fold5_response_sign_mismatch", as_index=False)
        .agg(
            record_count=("proxy9_record_1based", "size"),
            median_observed_proxy_trend=("proxy_trend", "median"),
            median_local_temperature_trend=("local_temperature_trend", "median"),
            median_fold5_prediction_trend=("fold5_prediction_trend", "median"),
            median_abs_fold5_residual_trend=("fold5_residual_trend", lambda values: float(np.nanmedian(np.abs(values)))),
            median_fold5_prediction_correlation=("fold5_prediction_correlation", "median"),
            hma_trend_contribution_change_sum=("trend_contribution_change_c_per_century", "sum"),
            median_abs_hma_trend_contribution_change=("trend_contribution_change_c_per_century", lambda values: float(np.nanmedian(np.abs(values)))),
        )
    )
    sign_summary.to_csv(output / "response_sign_mismatch_summary.csv", index=False)

    columns = [
        "proxy_trend",
        "local_temperature_trend",
        "local_precipitation_trend",
        "proxy_temperature_correlation",
        "proxy_precipitation_correlation",
        "fold5_prediction_trend",
        "fold5_residual_trend",
        "fold5_prediction_correlation",
        "fold5_prediction_trend_error",
        "trend_contribution_change_c_per_century",
        "absolute_trend_contribution_change",
    ]
    correlation = table.loc[valid, columns].corr(method="spearman")
    correlation.to_csv(output / "spearman_response_drift_associations.csv")

    top = table[valid].nlargest(12, "absolute_trend_contribution_change")
    figure, axes = plt.subplots(3, 4, figsize=(14.5, 10.2), sharex=True, constrained_layout=True)
    for axis, (_, row) in zip(axes.flat, top.iterrows()):
        index = int(row["proxy9_record_1based"]) - 1
        observed = proxy_values[index]
        predicted = fold5_prediction[index]
        temperature = temperature_values[index]
        valid_year = np.isfinite(observed) & np.isfinite(predicted)
        if np.any(valid_year):
            obs_std = np.nanstd(observed[valid_year])
            pred_std = np.nanstd(predicted[valid_year])
            temp_std = np.nanstd(temperature[valid_year])
            obs_scaled = (observed - np.nanmean(observed[valid_year])) / (obs_std if obs_std > 0 else 1)
            pred_scaled = (predicted - np.nanmean(predicted[valid_year])) / (pred_std if pred_std > 0 else 1)
            temp_scaled = (temperature - np.nanmean(temperature[valid_year])) / (temp_std if temp_std > 0 else 1)
            axis.plot(years, obs_scaled, color="#111111", lw=1.4, label="proxy")
            axis.plot(years, pred_scaled, color="#b2182b", lw=1.2, label="PSM")
            axis.plot(years, temp_scaled, color="#2166ac", lw=1.0, alpha=0.75, label="local T")
        axis.set_title(f"#{index + 1} {row['combined_name']}\nHMA delta {row['trend_contribution_change_c_per_century']:+.2f}", fontsize=8.5)
        axis.grid(alpha=0.18)
    axes.flat[0].legend(frameon=False, fontsize=7)
    figure.suptitle("Held-out proxy observations versus fold-5 PSM and local temperature")
    figure.savefig(output / "figure_top_response_drift_records.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    valid_table = table[valid]
    summary = {
        "experiment_id": "climate_reconstruction_proxy9_fold5_response_drift_v1",
        "status": "pass" if pure else "fail",
        "source_hashes_unchanged": pure,
        "records_with_at_least_3_holdout_observations": int(valid.sum()),
        "records_fold5_prediction_trend_sign_mismatch": int(valid_table["fold5_response_sign_mismatch"].sum()),
        "records_negative_hma_contribution_change": int((valid_table["trend_contribution_change_c_per_century"] < 0).sum()),
        "median_abs_fold5_residual_trend": float(valid_table["fold5_residual_trend"].abs().median()),
        "median_abs_archived_residual_trend": float(valid_table["archived_residual_trend"].abs().median()),
        "sign_mismatch_summary": sign_summary.to_dict(orient="records"),
        "spearman_with_hma_trend_contribution_change": {
            name: float(correlation.loc[name, "trend_contribution_change_c_per_century"])
            for name in columns
            if name != "trend_contribution_change_c_per_century"
        },
        "top_negative_records": table[valid & negative].nsmallest(15, "trend_contribution_change_c_per_century")[[
            "proxy9_record_1based",
            "combined_name",
            "proxy_trend",
            "local_temperature_trend",
            "fold5_prediction_trend",
            "fold5_residual_trend",
            "fold5_prediction_correlation",
            "trend_contribution_change_c_per_century",
        ]].to_dict(orient="records"),
        "interpretation_boundary": "Held-out response-drift audit only. It does not authorize proxy deletion, trend correction, or time-varying coefficients.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Fold-5 Response-Drift Audit v1\n\n"
        f"Status: **{summary['status']}**. See the held-out response table and top-record figure.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

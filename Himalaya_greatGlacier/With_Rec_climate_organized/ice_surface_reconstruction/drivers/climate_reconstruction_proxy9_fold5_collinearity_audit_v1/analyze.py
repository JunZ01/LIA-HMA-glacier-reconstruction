"""Merge exact predictor-conditioning metrics with fold-5 attribution."""

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
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_fold5_collinearity_audit_v1"
    paths = {
        "metrics": output / "collinearity_metrics.mat",
        "attribution": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_fold5_attribution_v1/proxy9_fold5_contribution_attribution.csv",
        "operators": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    with h5py.File(paths["metrics"], "r") as handle:
        arrays = {name: np.asarray(handle[name], dtype=float).T for name in ("effective_count", "predictor_correlation", "vif", "condition_number", "temperature_std", "precipitation_std")}
    attribution = pd.read_csv(paths["attribution"])
    table = pd.DataFrame({"proxy9_record_1based": np.arange(1, 291)})
    for name, values in arrays.items():
        table[f"fold5_{name}"] = values[:, 0]
        table[f"archived_{name}"] = values[:, 1]
        table[f"change_{name}"] = values[:, 0] - values[:, 1]
    table = table.merge(attribution, on="proxy9_record_1based", how="left")
    baseline_sign = np.sign(table["baseline_temperature_coefficient"])
    fold5_sign = np.sign(table["fold5_temperature_coefficient"])
    table["temperature_sign_changed"] = baseline_sign.ne(fold5_sign)
    table.to_csv(output / "proxy9_fold5_collinearity_attribution.csv", index=False)

    metrics = ["fold5_predictor_correlation", "fold5_vif", "fold5_condition_number", "change_predictor_correlation", "change_vif", "change_condition_number"]
    correlation = table[metrics + ["temperature_coefficient_change", "trend_contribution_change_c_per_century", "fold5_holdout_rmse", "fold5_holdout_correlation"]].corr(method="spearman")
    correlation.to_csv(output / "spearman_associations.csv")
    sign_summary = (
        table.groupby("temperature_sign_changed", as_index=False)
        .agg(
            record_count=("proxy9_record_1based", "size"),
            median_fold5_abs_predictor_correlation=("fold5_predictor_correlation", lambda values: float(np.nanmedian(np.abs(values)))),
            median_fold5_vif=("fold5_vif", "median"),
            median_fold5_condition_number=("fold5_condition_number", "median"),
            median_fold5_holdout_rmse=("fold5_holdout_rmse", "median"),
            median_fold5_holdout_correlation=("fold5_holdout_correlation", "median"),
            median_abs_trend_contribution_change=("trend_contribution_change_c_per_century", lambda values: float(np.nanmedian(np.abs(values)))),
        )
    )
    sign_summary.to_csv(output / "sign_change_conditioning_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), constrained_layout=True)
    colors = np.where(table["temperature_sign_changed"], "#b2182b", "#2166ac")
    axes[0].scatter(table["fold5_condition_number"], table["temperature_coefficient_change"], c=colors, s=24, alpha=0.68)
    axes[0].axhline(0.0, color="#777777", lw=0.7)
    axes[0].set_xlabel("Fold-5 standardized predictor condition number")
    axes[0].set_ylabel("Temperature coefficient change")
    axes[0].grid(alpha=0.18)
    axes[1].scatter(table["fold5_condition_number"], table["trend_contribution_change_c_per_century"], c=colors, s=24, alpha=0.68)
    axes[1].axhline(0.0, color="#777777", lw=0.7)
    axes[1].set_xlabel("Fold-5 standardized predictor condition number")
    axes[1].set_ylabel("1981-2000 HMA trend contribution change (degC/century)")
    axes[1].grid(alpha=0.18)
    figure.suptitle("Does temperature-precipitation collinearity explain fold-5 failure?")
    figure.savefig(output / "figure_collinearity_vs_fold5_failure.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": "climate_reconstruction_proxy9_fold5_collinearity_audit_v1",
        "status": "pass" if pure else "fail",
        "source_hashes_unchanged": pure,
        "fold5_predictor_abs_correlation_quantiles": {str(q): float(table["fold5_predictor_correlation"].abs().quantile(q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "fold5_vif_quantiles": {str(q): float(table["fold5_vif"].quantile(q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "fold5_condition_number_quantiles": {str(q): float(table["fold5_condition_number"].quantile(q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "sign_change_conditioning": sign_summary.to_dict(orient="records"),
        "spearman_with_temperature_coefficient_change": {name: float(correlation.loc[name, "temperature_coefficient_change"]) for name in metrics},
        "spearman_with_trend_contribution_change": {name: float(correlation.loc[name, "trend_contribution_change_c_per_century"]) for name in metrics},
        "interpretation_boundary": "Predictor-conditioning audit only. It does not authorize regularization, record deletion, or coefficient constraints.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Fold-5 Predictor Collinearity Audit v1\n\n"
        f"Status: **{summary['status']}**. See the merged attribution table and conditioning summary.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

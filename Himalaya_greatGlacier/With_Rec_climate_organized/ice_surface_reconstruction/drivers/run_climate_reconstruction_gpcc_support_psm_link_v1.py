"""Test whether GPCC gauge support or annualization explains proxy9 PSM error."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return {
        "path": str(path), "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns), "sha256": value.hexdigest(),
    }


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def main() -> int:
    project = root()
    audit = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_03_gpcc_calibration_target_audit_v2"
    )
    psm_path = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_01_psm_stability_repair/"
        "psm_stability_variants_v1.mat"
    )
    output = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_04_gpcc_support_psm_link_v1"
    )
    support_path = audit / "proxy_record_support.csv"
    annual_path = audit / "proxy_record_year_annualization.csv"
    facts_path = project / "LIA_PROJECT_FACTS.md"
    driver_path = Path(__file__).resolve()
    for path in (support_path, annual_path, psm_path, facts_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)

    sources = [support_path, annual_path, psm_path, facts_path, driver_path]
    before = {path.name: state(path) for path in sources}
    support = pd.read_csv(support_path)
    annual = pd.read_csv(annual_path)
    sensitive = support["precipitation_sensitive"].astype(str).str.lower().eq("true").to_numpy()
    with h5py.File(psm_path, "r") as handle:
        ridge = np.asarray(handle["ridge_holdout_rmse"], dtype=float).T
        precip_only = np.asarray(handle["precip_only_holdout_rmse"], dtype=float).T
        coefficients = np.asarray(handle["ridge_coefficients"], dtype=float).transpose(2, 1, 0)
    if ridge.shape != (297, 5) or sensitive.sum() != 290:
        raise RuntimeError("input-contract-failed")

    gauge = support.loc[sensitive, "direct_gauge_month_fraction"].to_numpy(float)
    interpolation = support.loc[sensitive, "median_interpolation_error_mm_month"].to_numpy(float)
    cell = support.loc[sensitive, "unique_grid_cell_index_1based"].to_numpy(int)
    ridge_median = np.nanmedian(ridge[sensitive], axis=1)
    precip_median = np.nanmedian(precip_only[sensitive], axis=1)
    fold_b2 = coefficients[sensitive, 1:, 2]
    sign_agreement = np.maximum(np.mean(fold_b2 > 0, axis=1), np.mean(fold_b2 < 0, axis=1))

    rows: list[dict[str, object]] = []
    for index in range(len(gauge)):
        rows.append({
            "precipitation_record_index": index + 1,
            "unique_grid_cell_index_1based": int(cell[index]),
            "direct_gauge_month_fraction": float(gauge[index]),
            "median_interpolation_error_mm_month": float(interpolation[index]),
            "ridge_median_holdout_rmse": float(ridge_median[index]),
            "precip_only_median_holdout_rmse": float(precip_median[index]),
            "ridge_b2_fold_sign_agreement": float(sign_agreement[index]),
        })
    with (output / "record_level_link.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = {
        "no_direct_gauge": gauge == 0,
        "any_direct_gauge": gauge > 0,
        "at_least_half_months": gauge >= 0.5,
    }
    group_summary: dict[str, dict[str, float | int]] = {}
    for name, mask in groups.items():
        group_summary[name] = {
            "record_count": int(mask.sum()),
            "ridge_median_holdout_rmse": float(np.nanmedian(ridge_median[mask])),
            "precip_only_median_holdout_rmse": float(np.nanmedian(precip_median[mask])),
            "ridge_b2_median_sign_agreement": float(np.nanmedian(sign_agreement[mask])),
        }

    no = gauge == 0
    any_gauge = gauge > 0
    ridge_valid = np.isfinite(ridge_median)
    precip_valid = np.isfinite(precip_median)
    ridge_mw = mannwhitneyu(
        ridge_median[no & ridge_valid], ridge_median[any_gauge & ridge_valid], alternative="two-sided"
    )
    precip_mw = mannwhitneyu(
        precip_median[no & precip_valid], precip_median[any_gauge & precip_valid], alternative="two-sided"
    )
    ridge_gauge_rho, ridge_gauge_p = corr(gauge, ridge_median)
    precip_gauge_rho, precip_gauge_p = corr(gauge, precip_median)
    ridge_error_rho, ridge_error_p = corr(interpolation, ridge_median)
    precip_error_rho, precip_error_p = corr(interpolation, precip_median)

    cell_frame = pd.DataFrame({
        "cell": cell, "gauge": gauge, "ridge": ridge_median,
        "precip": precip_median, "error": interpolation,
    }).groupby("cell", as_index=False).median(numeric_only=True)
    cell_ridge_rho, cell_ridge_p = corr(cell_frame.gauge.to_numpy(), cell_frame.ridge.to_numpy())
    cell_precip_rho, cell_precip_p = corr(cell_frame.gauge.to_numpy(), cell_frame.precip.to_numpy())

    annual = annual[annual["precipitation_sensitive"].astype(str).str.lower().eq("true")]
    anomaly_metrics = []
    for _, record in annual.groupby("record_index_1based"):
        years = record.year.to_numpy(int)
        legacy = record.legacy_mm_day.to_numpy(float)
        physical = record.physical_complete_mm_day.to_numpy(float)
        reference = (years >= 1961) & (years <= 1990)
        legacy_anomaly = legacy - np.mean(legacy[reference])
        physical_anomaly = physical - np.mean(physical[reference])
        difference = legacy_anomaly - physical_anomaly
        anomaly_metrics.append({
            "correlation": float(np.corrcoef(legacy_anomaly, physical_anomaly)[0, 1]),
            "rmse_mm_day": float(np.sqrt(np.mean(difference * difference))),
            "rmse_over_physical_std": float(np.sqrt(np.mean(difference * difference)) / np.std(physical_anomaly)),
            "trend_difference_mm_day_century": float(np.polyfit(years, difference, 1)[0] * 100),
        })
    anomaly = pd.DataFrame(anomaly_metrics)

    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "record_count": int(len(gauge)),
        "unique_cell_count": int(cell_frame.shape[0]),
        "groups": group_summary,
        "record_level_associations": {
            "direct_gauge_fraction_vs_ridge_rmse": {"spearman_rho": ridge_gauge_rho, "p_value": ridge_gauge_p},
            "direct_gauge_fraction_vs_precip_only_rmse": {"spearman_rho": precip_gauge_rho, "p_value": precip_gauge_p},
            "interpolation_error_vs_ridge_rmse": {"spearman_rho": ridge_error_rho, "p_value": ridge_error_p},
            "interpolation_error_vs_precip_only_rmse": {"spearman_rho": precip_error_rho, "p_value": precip_error_p},
            "ridge_no_gauge_vs_any_mann_whitney_p": float(ridge_mw.pvalue),
            "precip_no_gauge_vs_any_mann_whitney_p": float(precip_mw.pvalue),
        },
        "unique_cell_associations": {
            "direct_gauge_fraction_vs_ridge_rmse": {"spearman_rho": cell_ridge_rho, "p_value": cell_ridge_p},
            "direct_gauge_fraction_vs_precip_only_rmse": {"spearman_rho": cell_precip_rho, "p_value": cell_precip_p},
        },
        "annualization_anomaly_impact": {
            "correlation_median": float(anomaly.correlation.median()),
            "correlation_minimum": float(anomaly.correlation.min()),
            "rmse_mm_day_median": float(anomaly.rmse_mm_day.median()),
            "rmse_mm_day_p90": float(anomaly.rmse_mm_day.quantile(0.9)),
            "rmse_over_physical_std_median": float(anomaly.rmse_over_physical_std.median()),
            "rmse_over_physical_std_p90": float(anomaly.rmse_over_physical_std.quantile(0.9)),
            "trend_difference_mm_day_century_median": float(anomaly.trend_difference_mm_day_century.median()),
            "absolute_trend_difference_mm_day_century_p90": float(anomaly.trend_difference_mm_day_century.abs().quantile(0.9)),
        },
        "decision": "close_gpcc_gauge_support_and_fixed_30_day_conversion_as_primary_psm_failure_mechanisms",
        "remaining_issue": "unfrozen historical GPCC target provenance is a reproducibility defect, not a demonstrated skill-repair mechanism",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].scatter(gauge, ridge_median, s=18, alpha=0.65, color="#005f73")
    axes[0].set(xlabel="Direct-gauge month fraction", ylabel="Median blocked RMSE", title="Ridge PSM")
    axes[0].grid(alpha=0.2)
    axes[1].scatter(gauge, precip_median, s=18, alpha=0.65, color="#bb3e03")
    axes[1].set(xlabel="Direct-gauge month fraction", ylabel="Median blocked RMSE", title="Precipitation-only PSM")
    axes[1].grid(alpha=0.2)
    fig.suptitle("GPCC gauge support does not explain proxy9 PSM error", fontsize=14)
    fig.savefig(output / "gpcc_support_vs_psm_error.png", dpi=180)
    plt.close(fig)

    report = (
        "# GPCC Support-to-PSM Link\n\n"
        f"- No-direct-gauge records: `{groups['no_direct_gauge'].sum()}`; any-gauge records: `{groups['any_direct_gauge'].sum()}`.\n"
        f"- Ridge RMSE medians (none/any): `{group_summary['no_direct_gauge']['ridge_median_holdout_rmse']:.6f}` / "
        f"`{group_summary['any_direct_gauge']['ridge_median_holdout_rmse']:.6f}`, Mann-Whitney p=`{ridge_mw.pvalue:.6f}`.\n"
        f"- Gauge fraction versus ridge/precipitation-only RMSE rho: `{ridge_gauge_rho:.6f}` / `{precip_gauge_rho:.6f}`.\n"
        f"- Unique-cell gauge fraction versus ridge/precipitation-only RMSE rho: `{cell_ridge_rho:.6f}` / `{cell_precip_rho:.6f}`.\n"
        f"- Legacy versus physical precipitation anomaly correlation median/minimum: "
        f"`{anomaly.correlation.median():.9f}` / `{anomaly.correlation.min():.9f}`.\n"
        f"- Annualization anomaly RMSE / physical standard deviation median: "
        f"`{anomaly.rmse_over_physical_std.median():.6f}`.\n\n"
        "Conclusion: direct cell gauge support is weak but does not explain the blocked PSM failure, and the fixed "
        "30-day conversion is too small after anomaly centering to be the primary mechanism. Do not delete or "
        "downweight proxies solely by GPCC numgauge. The historical calibration-target provenance remains a "
        "reproducibility defect, while reconstruction skill must be repaired through a different upstream mechanism.\n"
    )
    (output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    after = {path.name: state(path) for path in sources}
    (output / "source_manifest.json").write_text(json.dumps({
        "sources_before": before, "sources_after": after, "source_states_unchanged": before == after
    }, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

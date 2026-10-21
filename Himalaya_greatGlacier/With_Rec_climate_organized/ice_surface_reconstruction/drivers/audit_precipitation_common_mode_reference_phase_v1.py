"""Compare the proxy common mode with disclosed HMA precipitation references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PERIOD = (1901, 1920)
LAGS = range(-5, 6)
FILL_VALUE_LIMIT = -9000.0


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


def weighted_mean(field: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[None, :, None]
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights, axis=(1, 2))
    denominator = np.sum(valid * weights, axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(field.shape[0], np.nan), where=denominator > 0)


def standardize(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values)
    std = np.nanstd(values, ddof=0)
    if not np.isfinite(std) or std == 0:
        raise RuntimeError("cannot-standardize-constant-series")
    return (values - mean) / std


def detrend(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return np.full(values.shape, np.nan)
    slope, intercept = np.polyfit(years[valid], values[valid], 1)
    return values - (slope * years + intercept)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    if np.nanstd(left[valid]) == 0 or np.nanstd(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def trend_per_century(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def lagged_correlation(signal: np.ndarray, reference: np.ndarray, lag: int) -> float:
    """Return corr(signal(t), reference(t + lag)); positive lag means reference is later."""
    if lag > 0:
        return correlation(signal[:-lag], reference[lag:])
    if lag < 0:
        return correlation(signal[-lag:], reference[:lag])
    return correlation(signal, reference)


def main() -> int:
    root = project_root()
    baseline_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1"
    common_mode_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2"
    positive_b2_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_proxy_mechanism_audit_v1"
    result_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_common_mode_reference_phase_audit_v1"
    input_netcdf = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    inputs = {
        "baseline_summary": baseline_dir / "summary.json",
        "locked_period_contract": baseline_dir / "LOCKED_FINAL_PERIOD.json",
        "common_mode_yearly": common_mode_dir / "innovation_pc1_yearly.csv",
        "positive_b2_yearly": positive_b2_dir / "positive_b2_pathway_yearly.csv",
        "netcdf": input_netcdf,
    }
    if missing := [str(path) for path in inputs.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if result_dir.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{result_dir}")

    baseline_summary = json.loads(inputs["baseline_summary"].read_text(encoding="utf-8"))
    locked = json.loads(inputs["locked_period_contract"].read_text(encoding="utf-8"))
    if baseline_summary["locked_final_period_evaluated"] or locked["status"] != "locked_not_evaluated":
        raise RuntimeError("locked-period-contract-failed")
    if baseline_summary["development_out_of_sample_period"] != list(PERIOD):
        raise RuntimeError("development-period-contract-failed")

    source_hashes_before = {name: sha256(path) for name, path in inputs.items() if name != "netcdf"}
    common_mode = pd.read_csv(inputs["common_mode_yearly"])
    positive_b2 = pd.read_csv(inputs["positive_b2_yearly"])
    if not np.array_equal(common_mode["year"].to_numpy(dtype=int), np.arange(PERIOD[0], PERIOD[1] + 1)):
        raise RuntimeError("common-mode-year-contract-failed")
    if not np.array_equal(positive_b2["year"].to_numpy(dtype=int), np.arange(PERIOD[0], PERIOD[1] + 1)):
        raise RuntimeError("positive-b2-year-contract-failed")

    variable_names = {
        "rec_result": "pr_ensemble_mean_rec_result",
        "rec_ref": "pr_ensemble_mean_rec_ref",
        "cru": "pr_obs_cru_pre_05deg",
        "gpcc": "pr_obs_gpcp_05deg",
    }
    with h5py.File(input_netcdf, "r") as handle:
        years_all = np.asarray(handle["time"], dtype=int)
        indices = np.flatnonzero((years_all >= PERIOD[0]) & (years_all <= PERIOD[1]))
        expected = np.arange(PERIOD[0], PERIOD[1] + 1)
        if not np.array_equal(years_all[indices], expected):
            raise RuntimeError("netcdf-time-index-contract-failed")
        latitudes = np.asarray(handle["lat"], dtype=float)
        missing_variables = [name for name in variable_names.values() if name not in handle]
        if missing_variables:
            raise KeyError(f"missing-precipitation-variables:{missing_variables}")
        hma_series = {}
        for label, variable in variable_names.items():
            values = np.asarray(handle[variable][indices, :, :], dtype=float)
            values[values <= FILL_VALUE_LIMIT] = np.nan
            hma_series[label] = weighted_mean(values, latitudes)

    years = expected
    table = pd.DataFrame({"year": years})
    table["proxy_network_mean_standardized"] = common_mode["standardized_network_mean"].to_numpy(dtype=float)
    table["proxy_innovation_pc1"] = common_mode["pc1_score"].to_numpy(dtype=float)
    table["positive_b2_direct_pathway_effect_mm_day"] = positive_b2["direct_positive_b2_effect_mm_day"].to_numpy(dtype=float)
    for label, values in hma_series.items():
        table[f"{label}_hma_anomaly_mm_day"] = values
    for label in ["rec_result", "rec_ref", "cru", "gpcc"]:
        table[f"{label}_hma_standardized"] = standardize(table[f"{label}_hma_anomaly_mm_day"].to_numpy())
    table["positive_b2_direct_pathway_effect_standardized"] = standardize(table["positive_b2_direct_pathway_effect_mm_day"].to_numpy())

    series_columns = [
        "proxy_network_mean_standardized",
        "proxy_innovation_pc1",
        "positive_b2_direct_pathway_effect_standardized",
        "rec_result_hma_standardized",
        "rec_ref_hma_standardized",
        "cru_hma_standardized",
        "gpcc_hma_standardized",
    ]
    correlation_rows: list[dict[str, object]] = []
    for left_index, left in enumerate(series_columns):
        for right in series_columns[left_index + 1 :]:
            correlation_rows.append(
                {
                    "left": left,
                    "right": right,
                    "zero_lag_correlation": correlation(table[left].to_numpy(), table[right].to_numpy()),
                    "linear_detrended_correlation": correlation(detrend(table[left].to_numpy(), years), detrend(table[right].to_numpy(), years)),
                }
            )
    correlations = pd.DataFrame(correlation_rows)

    lag_rows: list[dict[str, object]] = []
    signal = table["proxy_network_mean_standardized"].to_numpy()
    for reference_label in ["rec_result", "rec_ref", "cru", "gpcc"]:
        reference = table[f"{reference_label}_hma_standardized"].to_numpy()
        for lag in LAGS:
            lag_rows.append(
                {
                    "signal": "proxy_network_mean_standardized",
                    "reference": reference_label,
                    "lag_years_reference_relative_to_signal": int(lag),
                    "correlation": lagged_correlation(signal, reference, lag),
                    "sample_size": int(len(years) - abs(lag)),
                }
            )
    lag_table = pd.DataFrame(lag_rows)
    best_lags = lag_table.loc[lag_table.groupby("reference")["correlation"].apply(lambda values: values.abs().idxmax())].sort_values("reference")

    trend_rows = []
    for column in series_columns:
        trend_rows.append({"series": column, "trend_per_century": trend_per_century(years, table[column].to_numpy())})
    trends = pd.DataFrame(trend_rows)

    def pair_value(left: str, right: str, metric: str) -> float:
        match = correlations[((correlations["left"] == left) & (correlations["right"] == right)) | ((correlations["left"] == right) & (correlations["right"] == left))]
        if len(match) != 1:
            raise RuntimeError(f"pair-metric-not-found:{left}:{right}")
        return float(match.iloc[0][metric])

    comparison = {
        reference: {
            "zero_lag": pair_value("proxy_network_mean_standardized", f"{reference}_hma_standardized", "zero_lag_correlation"),
            "detrended": pair_value("proxy_network_mean_standardized", f"{reference}_hma_standardized", "linear_detrended_correlation"),
            "best_bounded_lag": best_lags.loc[best_lags["reference"] == reference].iloc[0].to_dict(),
        }
        for reference in ["rec_result", "rec_ref", "cru", "gpcc"]
    }
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_common_mode_reference_phase_audit_v1",
        "period": list(PERIOD),
        "locked_final_period_accessed": False,
        "reference_roles": {"rec_ref": "comparison_only", "cru": "instrumental observation", "gpcc": "GPCC-derived instrumental observation"},
        "hms_mean_definition": "cosine-latitude weighted mean of the 43x78 HMA-domain fields, matching the instrumental-baseline driver",
        "netcdf_access": "Only the 20 annual indices for 1901-1920 were read from precipitation variables.",
        "network_signal": "standardized mean of 147 HMA-network proxy innovations; PC1 reported only as secondary context because PC sign is arbitrary.",
        "network_mean_comparison": comparison,
        "trend_per_century": {row["series"]: row["trend_per_century"] for row in trend_rows},
        "interpretation_limit": "A 20-year disclosed-period phase diagnostic. Bounded lag maxima are descriptive, not a validation metric or a basis for changing the reconstruction.",
    }

    result_dir.mkdir(parents=True)
    table.to_csv(result_dir / "common_mode_reference_hma_series_1901_1920.csv", index=False)
    correlations.to_csv(result_dir / "common_mode_reference_correlations.csv", index=False)
    lag_table.to_csv(result_dir / "common_mode_reference_lag_correlations.csv", index=False)
    trends.to_csv(result_dir / "common_mode_reference_trends.csv", index=False)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 8})
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), constrained_layout=True)
    axes[0].plot(years, table["proxy_network_mean_standardized"], color="#1b1b1b", linewidth=2.2, label="proxy network mean")
    axes[0].plot(years, table["positive_b2_direct_pathway_effect_standardized"], color="#0072B2", linewidth=1.8, label="direct positive-b2 effect")
    axes[0].plot(years, table["rec_result_hma_standardized"], color="#D55E00", linewidth=1.6, label="rec_result")
    axes[0].plot(years, table["rec_ref_hma_standardized"], color="#009E73", linewidth=1.6, label="rec_ref")
    axes[0].plot(years, table["cru_hma_standardized"], color="#6A3D9A", linewidth=1.3, label="CRU")
    axes[0].plot(years, table["gpcc_hma_standardized"], color="#7A7A7A", linewidth=1.3, label="GPCC-derived")
    axes[0].axhline(0, color="#B5B5B5", linewidth=0.8)
    axes[0].set_ylabel("Standardized anomaly")
    axes[0].set_title("Proxy common mode and HMA precipitation, 1901-1920")
    axes[0].legend(frameon=False, ncol=3, loc="best")

    colors = {"rec_result": "#D55E00", "rec_ref": "#009E73", "cru": "#6A3D9A", "gpcc": "#7A7A7A"}
    for reference, color in colors.items():
        subset = lag_table[lag_table["reference"] == reference]
        axes[1].plot(subset["lag_years_reference_relative_to_signal"], subset["correlation"], marker="o", color=color, linewidth=1.5, label=reference)
    axes[1].axhline(0, color="#B5B5B5", linewidth=0.8)
    axes[1].axvline(0, color="#B5B5B5", linewidth=0.8)
    axes[1].set_xlabel("Reference lag relative to proxy signal (years)")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Descriptive bounded lag correlations: proxy network mean vs reference")
    axes[1].legend(frameon=False, ncol=4)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(result_dir / "common_mode_reference_phase_audit.png", dpi=180)
    plt.close(figure)

    report = [
        "# Common-mode Reference Phase Audit",
        "",
        "Period: 1901-1920 only. The 1921-1939 final test period was not accessed.",
        "",
        "## Zero-lag and Detrended Correlation",
        "",
    ]
    for reference in ["rec_result", "rec_ref", "cru", "gpcc"]:
        item = comparison[reference]
        best = item["best_bounded_lag"]
        report.append(f"- {reference}: zero-lag `{item['zero_lag']:.3f}`, detrended `{item['detrended']:.3f}`, best bounded lag `{int(best['lag_years_reference_relative_to_signal']):+d}` years with `{float(best['correlation']):.3f}`.")
    report.extend([
        "",
        "rec_ref remains comparison-only. The bounded-lag results are descriptive and do not authorize a reconstruction change.",
    ])
    (result_dir / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    source_hashes_after = {name: sha256(path) for name, path in inputs.items() if name != "netcdf"}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("source-inputs-changed-during-audit")
    manifest = {
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "netcdf_sha256_from_baseline_contract": baseline_summary["source_sha256_before"],
        "analysis_driver_sha256": sha256(Path(__file__)),
    }
    (result_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

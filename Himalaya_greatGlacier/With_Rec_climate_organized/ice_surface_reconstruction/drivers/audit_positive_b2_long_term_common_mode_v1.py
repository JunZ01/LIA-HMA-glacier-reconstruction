"""Audit long-term positive-b2 proxy common modes without opening 1921-1939."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LONG_YEARS = np.arange(1500, 1921)
CAL_YEARS = np.arange(1940, 2001)
ALL_YEARS = np.concatenate([LONG_YEARS, CAL_YEARS])
WINDOWS = [(1500, 1599), (1600, 1699), (1700, 1799), (1800, 1899), (1901, 1920), (1940, 2000)]


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_proxy_path() -> Path:
    archive = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive / paper / "work1" / "compared_result_all" / "python_data" / "proxy_data_6.mat"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def mean_pairwise_correlation(values: np.ndarray, minimum_fraction: float = 0.8) -> tuple[float, int]:
    keep = np.mean(np.isfinite(values), axis=0) >= minimum_fraction
    selected = values[:, keep]
    pair_values = []
    for left in range(selected.shape[1]):
        for right in range(left + 1, selected.shape[1]):
            value = correlation(selected[:, left], selected[:, right])
            if np.isfinite(value):
                pair_values.append(value)
    return (float(np.mean(pair_values)) if pair_values else float("nan"), int(keep.sum()))


def main() -> int:
    root = project_root()
    mechanism = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_proxy_mechanism_audit_v1/positive_b2_proxy_mechanism_table.csv"
    innovation = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2/innovation_pc1_yearly.csv"
    proxy6 = archive_proxy_path()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_long_term_common_mode_v1"
    if any(not path.exists() for path in (mechanism, innovation, proxy6)):
        raise FileNotFoundError("missing-input")
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    hashes_before = {"mechanism": digest(mechanism), "innovation": digest(innovation)}
    proxy_stat_before = source_stat(proxy6)
    table = pd.read_csv(mechanism)
    selected = table["proxy_index"].to_numpy(dtype=int)
    if len(selected) != 60 or not (table["coefficient_b2"] >= 0).all():
        raise RuntimeError("positive-b2-selection-contract-failed")

    values = np.full((len(ALL_YEARS), len(selected)), np.nan)
    long_slice = slice(1499, 1920)
    calibration_slice = slice(1939, 2000)
    with h5py.File(proxy6, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(selected):
            proxy = handle[refs[proxy_index - 1]]
            long_ages = np.asarray(proxy["age_yng"][0, long_slice], dtype=float)
            calibration_ages = np.asarray(proxy["age_yng"][0, calibration_slice], dtype=float)
            if not np.array_equal(long_ages, LONG_YEARS) or not np.array_equal(calibration_ages, CAL_YEARS):
                raise RuntimeError(f"annual-index-contract-failed:{proxy_index}")
            values[: len(LONG_YEARS), column] = np.asarray(proxy["data"][long_slice, 0], dtype=float)
            values[len(LONG_YEARS) :, column] = np.asarray(proxy["data"][calibration_slice, 0], dtype=float)

    calibration = values[len(LONG_YEARS) :]
    calibration_count = np.sum(np.isfinite(calibration), axis=0)
    calibration_mean = np.nanmean(calibration, axis=0)
    calibration_std = np.nanstd(calibration, axis=0)
    valid_calibration = (calibration_count >= 20) & np.isfinite(calibration_std) & (calibration_std > 0)
    if int(valid_calibration.sum()) < 3:
        raise RuntimeError("insufficient-calibration-records")
    standardized = (values - calibration_mean[None, :]) / calibration_std[None, :]
    standardized[:, ~valid_calibration] = np.nan
    valid_count = np.sum(np.isfinite(standardized), axis=1)
    network_mean = np.nanmean(standardized, axis=1)
    early_indices = np.flatnonzero((LONG_YEARS >= 1901) & (LONG_YEARS <= 1920))
    complete_early = np.all(np.isfinite(standardized[early_indices]), axis=0)
    fixed_panel_mean = np.nanmean(standardized[:, complete_early], axis=1) if int(complete_early.sum()) else np.full(len(ALL_YEARS), np.nan)

    annual = pd.DataFrame({"year": ALL_YEARS, "period": np.where(ALL_YEARS <= 1920, "prelocked_history", "calibration"), "positive_b2_network_mean_calibration_z": network_mean, "fixed_1901_1920_panel_mean_calibration_z": fixed_panel_mean, "valid_proxy_count": valid_count})
    window_rows = []
    for start, end in WINDOWS:
        mask = (ALL_YEARS >= start) & (ALL_YEARS <= end)
        coherence, coherent_count = mean_pairwise_correlation(standardized[mask])
        window_rows.append({"start_year": start, "end_year": end, "network_mean_trend_z_per_century": trend(ALL_YEARS[mask], network_mean[mask]), "fixed_panel_trend_z_per_century": trend(ALL_YEARS[mask], fixed_panel_mean[mask]), "median_valid_proxy_count": float(np.median(valid_count[mask])), "minimum_valid_proxy_count": int(np.min(valid_count[mask])), "network_mean_vs_valid_count_correlation": correlation(network_mean[mask], valid_count[mask].astype(float)), "mean_pairwise_proxy_correlation": coherence, "proxy_count_with_at_least_80pct_window_coverage": coherent_count})
    windows = pd.DataFrame(window_rows)
    innovation_early = pd.read_csv(innovation)
    if not np.array_equal(innovation_early["year"].to_numpy(dtype=int), np.arange(1901, 1921)):
        raise RuntimeError("innovation-period-contract-failed")
    raw_early = annual.set_index("year").loc[np.arange(1901, 1921), "positive_b2_network_mean_calibration_z"].to_numpy()
    fixed_early = annual.set_index("year").loc[np.arange(1901, 1921), "fixed_1901_1920_panel_mean_calibration_z"].to_numpy()
    summary = {"experiment_id": "climate_reconstruction_precipitation_positive_b2_long_term_common_mode_v1", "analyzed_periods": [[1500, 1920], [1940, 2000]], "locked_final_period_accessed": False, "locked_final_period": [1921, 1939], "positive_b2_proxy_count": 60, "calibration_standardizable_proxy_count": int(valid_calibration.sum()), "complete_1901_1920_fixed_panel_proxy_count": int(complete_early.sum()), "early20_diagnostics": {"positive_b2_network_vs_all_hma_innovation_network_correlation": correlation(raw_early, innovation_early["standardized_network_mean"].to_numpy()), "positive_b2_all_available_vs_fixed_panel_correlation": correlation(raw_early, fixed_early), "network_mean_trend_z_per_century": trend(np.arange(1901, 1921), raw_early), "fixed_panel_trend_z_per_century": trend(np.arange(1901, 1921), fixed_early)}, "window_diagnostics": window_rows, "interpretation_limit": "This audit detects persistence, coherence, and network-turnover sensitivity in stored proxy data. It does not determine whether the common mode is climatically true or identify a specific preprocessing operation without upstream raw-source lineage."}
    output.mkdir(parents=True)
    annual.to_csv(output / "positive_b2_long_term_common_mode_annual.csv", index=False)
    windows.to_csv(output / "positive_b2_long_term_common_mode_windows.csv", index=False)
    pd.DataFrame({"proxy_index": selected, "calibration_valid_count": calibration_count, "calibration_mean": calibration_mean, "calibration_std": calibration_std, "standardizable": valid_calibration, "complete_1901_1920_panel": complete_early}).to_csv(output / "positive_b2_proxy_coverage_and_scaling.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), constrained_layout=True)
    history = annual[annual["year"] <= 1920]
    axes[0].plot(history["year"], history["positive_b2_network_mean_calibration_z"].rolling(21, center=True, min_periods=11).mean(), color="#1b1b1b", linewidth=1.8, label="21-year network mean")
    axes[0].axhline(0, color="#B5B5B5", linewidth=0.8); axes[0].set_ylabel("Calibration-z mean"); axes[0].set_title("Positive-b2 proxy common mode, 1500-1920"); axes[0].legend(frameon=False)
    recent = history[history["year"] >= 1850]
    axes[1].plot(recent["year"], recent["positive_b2_network_mean_calibration_z"], color="#0072B2", linewidth=1.7, label="all available proxies")
    axes[1].plot(recent["year"], recent["fixed_1901_1920_panel_mean_calibration_z"], color="#D55E00", linewidth=1.5, linestyle="--", label="fixed 1901-1920 panel")
    axes[1].axhline(0, color="#B5B5B5", linewidth=0.8); axes[1].set_ylabel("Calibration-z mean"); axes[1].set_title("Network-turnover check, 1850-1920"); axes[1].legend(frameon=False)
    axes[2].plot(history["year"], history["valid_proxy_count"], color="#009E73", linewidth=1.5)
    axes[2].set_xlabel("Year"); axes[2].set_ylabel("Valid proxy count"); axes[2].set_title("Available positive-b2 records")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "positive_b2_long_term_common_mode.png", dpi=180); plt.close(figure)
    report = ["# Positive-b2 Long-term Common-mode Audit", "", "Analyzed: 1500-1920 and 1940-2000. The locked 1921-1939 period was not accessed.", "", f"- Calibration-standardizable proxies: `{int(valid_calibration.sum())}/60`.", f"- Complete fixed 1901-1920 panel: `{int(complete_early.sum())}/60`.", f"- Early-20 all-available versus fixed-panel correlation: `{summary['early20_diagnostics']['positive_b2_all_available_vs_fixed_panel_correlation']:.3f}`.", "", "This diagnoses persistence and network turnover; it does not establish whether the common mode is true precipitation."]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes_after = {"mechanism": digest(mechanism), "innovation": digest(innovation)}
    proxy_stat_after = source_stat(proxy6)
    if hashes_before != hashes_after or proxy_stat_before != proxy_stat_after:
        raise RuntimeError("source-inputs-changed-during-audit")
    (output / "source_manifest.json").write_text(json.dumps({"hashes_before": hashes_before, "hashes_after": hashes_after, "proxy6_stat_before": proxy_stat_before, "proxy6_stat_after": proxy_stat_after, "analysis_driver_sha256": digest(Path(__file__))}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

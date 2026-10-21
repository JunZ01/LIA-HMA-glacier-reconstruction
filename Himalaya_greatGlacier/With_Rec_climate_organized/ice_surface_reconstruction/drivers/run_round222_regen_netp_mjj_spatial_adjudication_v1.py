from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = Path(__file__).resolve().parent
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_222_regen_netp_mjj_spatial_adjudication_v1"
CONTRACT = OUT / "ROUND_CONTRACT.md"
ROUND220_DRIVER = DRIVERS / "run_round220_netp_mjj_cross_prior_spatial_oof_v1.py"
ROUND221 = LOOP / "round_221_regen_netp_mjj_data_recovery_v1"
ROUND218 = LOOP / "round_218_netp_mjj_prior_covariance_spatial_oof_v2"
ROUND219 = LOOP / "round_219_ccsm4_netp_mjj_prior_preflight_v1"
TARGET_LAT = np.asarray([35, 37, 39, 41] * 8, dtype=float)
TARGET_LON = np.repeat(np.arange(93, 109, 2), 4).astype(float)
CAL_YEARS = np.arange(1940, 2001, dtype=int)
COMPARE_YEARS = np.r_[np.arange(1955, 1965), np.arange(1975, 1996)]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= 100_000_000:
        result["sha256"] = sha256(path)
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def smooth_segments(round134, values: np.ndarray, years: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    for start, end in ((1950, 1969), (1970, 2000)):
        mask = (years >= start) & (years <= end)
        result = np.fmax(result, round134.smooth_inside_mask(values, mask))
    return result


def anomalies(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1961) & (years <= 1990)
    center = np.nanmean(values[baseline], axis=0)
    if not np.isfinite(center).all():
        raise RuntimeError("Non-finite 1961-1990 anomaly baseline")
    return values - center[None, :]


def annual_patterns(years: np.ndarray, reference: np.ndarray, candidate: np.ndarray, quality: np.ndarray, label: str) -> pd.DataFrame:
    rows = []
    for year in COMPARE_YEARS:
        index = int(np.flatnonzero(years == year)[0])
        value = correlation(reference[index, quality], candidate[index, quality])
        rows.append({"comparison": label, "year": int(year), "spatial_pattern_correlation": value})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    round220 = load_module("round220_for_round222", ROUND220_DRIVER)
    round134 = load_module("round134_for_round222", round220.ROUND134_DRIVER)
    audit = round134.load_module("round222_archive", DRIVERS / "audit_positive_b2_event_seasonal_psm_v1.py")
    r118 = round134.load_module(
        "round222_r118",
        DRIVERS / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "contract": CONTRACT,
        "driver": Path(__file__).resolve(),
        "round220_driver": ROUND220_DRIVER,
        "round221_summary": ROUND221 / "summary.json",
        "regen": ROUND221 / "regen_netp_mjj_1950_2000.npz",
        "round218_cells": ROUND218 / "cell_skill_and_prior_loadings.csv",
        "round219_prior": ROUND219 / "ccsm4_netp_mjj_prior_850_1850.npz",
        "proxy6": audit.archive_proxy_path(),
        "metadata": ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in paths.items()}

    recovery = json.loads(paths["round221_summary"].read_text(encoding="utf-8"))
    if recovery["decision"] != "admit_regen_as_source_correlated_independent_gridding_diagnostic":
        raise RuntimeError("Round 221 admission state mismatch")
    with np.load(paths["regen"]) as archive:
        regen_years = archive["years"].astype(int)
        regen_lat = archive["latitude"].astype(float)
        regen_lon = archive["longitude"].astype(float)
        quality = archive["high_quality_target"].astype(bool)
        regen = archive["mjj_mm_day"].astype(float)
    if not np.array_equal(regen_years, np.arange(1950, 2001)) or quality.sum() != 21:
        raise RuntimeError("REGEN year or quality-mask mismatch")
    if not np.array_equal(regen_lat, TARGET_LAT) or not np.array_equal(regen_lon, TARGET_LON):
        raise RuntimeError("REGEN target coordinates mismatch")

    cesm_frame = pd.read_csv(paths["round218_cells"])
    cesm_frame = cesm_frame[cesm_frame["product"] == "GPCC"].sort_values(["longitude", "latitude"])
    cesm = cesm_frame["prior_loading"].to_numpy(float)
    with np.load(paths["round219_prior"]) as archive:
        ccsm = archive["prior_loading"].astype(float)
    weights = np.cos(np.deg2rad(TARGET_LAT))
    loadings = {
        "cesm_lme010": cesm,
        "ccsm4_past1000": ccsm,
        "equal_prior": 0.5 * (cesm + ccsm),
    }
    for name, values in loadings.items():
        if abs(float(np.average(values, weights=weights)) - 1.0) > 1e-12:
            raise RuntimeError(f"Loading closure failed: {name}")

    metadata = pd.read_csv(paths["metadata"])
    _, groups = round134.groups_from_metadata(metadata)
    proxy = round134.read_proxy_values(paths["proxy6"], round134.TRAIN_IDS, CAL_YEARS)
    observed_cells = {}
    for product, path, variable, source_start in (
        ("GPCC", paths["gpcc"], "precip", 1891),
        ("CRU", paths["cru"], "pre", 1901),
    ):
        monthly = r118.read_netcdf3_points(path, variable, source_start, TARGET_LAT, TARGET_LON)
        observed_cells[product] = round134.aggregate_mjj(monthly)
    branches = {
        name: round220.run_branch(round134, proxy, groups, observed_cells, weights, values)
        for name, values in loadings.items()
    }

    regen_smooth = smooth_segments(round134, regen, regen_years)
    regen_anomaly = anomalies(regen_smooth, regen_years)
    observed_anomaly = {}
    for product, values in observed_cells.items():
        product_smooth = smooth_segments(round134, values[CAL_YEARS >= 1950], regen_years)
        observed_anomaly[product] = anomalies(product_smooth, regen_years)

    pattern_frames = []
    pattern_frames.append(annual_patterns(regen_years, regen_anomaly, observed_anomaly["GPCC"], quality, "regen_vs_gpcc_observed"))
    pattern_frames.append(annual_patterns(regen_years, regen_anomaly, observed_anomaly["CRU"], quality, "regen_vs_cru_observed"))
    for branch, result in branches.items():
        for product in round134.PRODUCTS:
            prediction_anomaly = result["prediction"][product] - result["baseline"][product]
            prediction_subset = prediction_anomaly[CAL_YEARS >= 1950]
            pattern_frames.append(
                annual_patterns(
                    regen_years,
                    regen_anomaly,
                    prediction_subset,
                    quality,
                    f"regen_vs_{branch}_{product.lower()}_calibrated_prediction",
                )
            )
    yearly = pd.concat(pattern_frames, ignore_index=True)
    yearly.to_csv(OUT / "yearly_spatial_pattern_correlations.csv", index=False)
    metrics = (
        yearly.groupby("comparison", as_index=False)["spatial_pattern_correlation"]
        .agg(["median", "mean", "min", "max", "count"])
        .reset_index()
        .rename(columns={"median": "median_spatial_correlation", "mean": "mean_spatial_correlation"})
    )
    metrics.to_csv(OUT / "comparison_metrics.csv", index=False)

    metric_map = dict(zip(metrics["comparison"], metrics["median_spatial_correlation"]))
    regen_gpcc = float(metric_map["regen_vs_gpcc_observed"])
    regen_cru = float(metric_map["regen_vs_cru_observed"])
    equal_gpcc = float(metric_map["regen_vs_equal_prior_gpcc_calibrated_prediction"])
    equal_cru = float(metric_map["regen_vs_equal_prior_cru_calibrated_prediction"])
    equal_primary = float(np.nanmean([equal_gpcc, equal_cru]))
    cru_minus_gpcc = regen_cru - regen_gpcc
    if equal_primary >= 0.30 and cru_minus_gpcc >= 0.10:
        decision = "gpcc_specific_interpolation_conflict_no_production_promotion"
    elif equal_primary < 0.30:
        decision = "rank1_spatialization_failure_supported_by_two_of_three_gridding_methods"
    else:
        decision = "regen_adjudication_unresolved_no_production_promotion"

    colors = {
        "regen_vs_gpcc_observed": "#B33A3A",
        "regen_vs_cru_observed": "#147D78",
        "regen_vs_equal_prior_gpcc_calibrated_prediction": "#D88928",
        "regen_vs_equal_prior_cru_calibrated_prediction": "#315B96",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.28})
    for label, color in colors.items():
        frame = yearly[yearly["comparison"] == label]
        axes[0].plot(frame["year"], frame["spatial_pattern_correlation"], marker="o", markersize=3, linewidth=1.3, color=color, label=label.replace("regen_vs_", ""))
    axes[0].axhline(0.30, color="#666666", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("Annual spatial-pattern correlation")
    axes[0].set_title("REGEN agreement across 21 quality cells")
    axes[0].legend(frameon=False, fontsize=7)
    selected = metrics[metrics["comparison"].isin(colors)].copy()
    axes[1].barh(selected["comparison"].str.replace("regen_vs_", "", regex=False), selected["median_spatial_correlation"], color=[colors[x] for x in selected["comparison"]])
    axes[1].axvline(0.30, color="#666666", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("Median spatial-pattern correlation")
    axes[1].set_title("Frozen median gate")
    for axis in axes:
        axis.grid(color="#DDDDDD", linewidth=0.6)
    figure.suptitle("NETP MJJ REGEN spatial adjudication")
    figure.savefig(OUT / "regen_netp_mjj_spatial_adjudication.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    after = {name: state(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("A source changed")
    summary = {
        "round": 222,
        "decision": decision,
        "quality_cell_count": int(quality.sum()),
        "comparison_years": COMPARE_YEARS.tolist(),
        "regen_vs_gpcc_observed_median_spatial_correlation": regen_gpcc,
        "regen_vs_cru_observed_median_spatial_correlation": regen_cru,
        "regen_cru_minus_gpcc_coherence": cru_minus_gpcc,
        "equal_prior_gpcc_calibrated_prediction_median": equal_gpcc,
        "equal_prior_cru_calibrated_prediction_median": equal_cru,
        "equal_prior_mean_median_spatial_correlation": equal_primary,
        "source_role": "independent_gridding_method_but_source_correlated_with_gpcc_and_ghcn",
        "round220_production_gate_overridden": False,
        "source_states_unchanged": True,
        "locked_1901_1939_values_read": False,
        "paleo_proxy_values_read": False,
        "external_validation_reserve_opened": False,
        "jionglacuo_accessed": False,
        "enkf_run": False,
        "production_changed": False,
        "oggm_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 222 Result\n\n"
        f"`{decision}`\n\n"
        f"REGEN median spatial agreement with observed GPCC/CRU is `{regen_gpcc:+.3f}` / `{regen_cru:+.3f}`. "
        f"Its agreement with the equal-prior GPCC/CRU-calibrated predictions is `{equal_gpcc:+.3f}` / `{equal_cru:+.3f}` "
        f"(mean `{equal_primary:+.3f}`). Round 220 remains unmodified.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

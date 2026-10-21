"""NETP May-July low-frequency proxy-composite reconstruction prototype."""

from __future__ import annotations

import calendar
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAL_YEARS = np.arange(1940, 2001, dtype=int)
PALEO_YEARS = np.arange(1000, 1921, dtype=int)
NETP_IDS = (9, 15, 22, 26, 46, 47, 51, 59, 64, 65, 66, 89, 106, 112, 114, 131, 143, 166, 167, 197)
EXCLUDED_ID = 15
TRAIN_IDS = tuple(value for value in NETP_IDS if value != EXCLUDED_ID)
FOLDS = (
    ("early_to_late", 1940, 1969, 1970, 2000),
    ("late_to_early", 1970, 2000, 1940, 1969),
)
PRODUCTS = ("GPCC", "CRU")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 100_000_000:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def centered_rolling(values: np.ndarray, window: int) -> np.ndarray:
    if values.ndim == 1:
        return pd.Series(values).rolling(window, center=True, min_periods=window).mean().to_numpy()
    return np.column_stack([centered_rolling(values[:, column], window) for column in range(values.shape[1])])


def smooth_inside_mask(values: np.ndarray, mask: np.ndarray, window: int = 11) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=float)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return output
    if not np.all(np.diff(indices) == 1):
        raise RuntimeError("noncontiguous-smoothing-mask")
    output[indices] = centered_rolling(values[indices], window)
    return output


def standardize_columns(values: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.nanmean(values[fit], axis=0)
    scale = np.nanstd(values[fit], axis=0, ddof=1)
    valid = np.isfinite(scale) & (scale > 1e-12)
    result = np.full(values.shape, np.nan, dtype=float)
    result[:, valid] = (values[:, valid] - center[valid]) / scale[valid]
    return result, center, scale


def groups_from_metadata(metadata: pd.DataFrame) -> tuple[list[tuple[float, float]], list[np.ndarray]]:
    selected = metadata.set_index("proxy_index").loc[list(TRAIN_IDS)].reset_index()
    selected["group_lat"] = np.round(selected.latitude.to_numpy() * 4) / 4
    selected["group_lon"] = np.round(selected.longitude.to_numpy() * 4) / 4
    keys = sorted(set(zip(selected.group_lat, selected.group_lon)))
    groups = []
    ids = np.asarray(TRAIN_IDS)
    for key in keys:
        members = selected.loc[(selected.group_lat == key[0]) & (selected.group_lon == key[1]), "proxy_index"].to_numpy(dtype=int)
        groups.append(np.flatnonzero(np.isin(ids, members)))
    return keys, groups


def composite(
    proxy_values: np.ndarray,
    target: np.ndarray,
    fit: np.ndarray,
    groups: list[np.ndarray],
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    signs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if center is None or scale is None:
        z, center, scale = standardize_columns(proxy_values, fit)
    else:
        valid = np.isfinite(scale) & (scale > 1e-12)
        z = np.full(proxy_values.shape, np.nan)
        z[:, valid] = (proxy_values[:, valid] - center[valid]) / scale[valid]
    if signs is None:
        signs = np.full(proxy_values.shape[1], np.nan)
        for column in range(proxy_values.shape[1]):
            value = correlation(z[fit, column], target[fit])
            if np.isfinite(value) and abs(value) > 1e-12:
                signs[column] = np.sign(value)
    z = z * signs[None, :]
    group_values = np.full((len(proxy_values), len(groups)), np.nan)
    for group_index, members in enumerate(groups):
        finite_count = np.isfinite(z[:, members]).sum(axis=1)
        summed = np.nansum(z[:, members], axis=1)
        group_values[:, group_index] = np.divide(summed, finite_count, out=np.full(len(z), np.nan), where=finite_count > 0)
    active = np.isfinite(group_values).sum(axis=1)
    result = np.divide(np.nansum(group_values, axis=1), active, out=np.full(len(z), np.nan), where=active > 0)
    return result, center, scale, signs


def linear_fit_predict(comp: np.ndarray, target: np.ndarray, fit: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_fit = fit & np.isfinite(comp) & np.isfinite(target)
    if int(valid_fit.sum()) < 10:
        raise RuntimeError("insufficient-composite-fit")
    coefficients = np.linalg.lstsq(np.column_stack((np.ones(valid_fit.sum()), comp[valid_fit])), target[valid_fit], rcond=None)[0]
    prediction = np.full(len(comp), np.nan)
    valid_test = test & np.isfinite(comp)
    prediction[valid_test] = coefficients[0] + coefficients[1] * comp[valid_test]
    return prediction, coefficients


def read_proxy_values(path: Path, proxy_ids: tuple[int, ...], years_requested: np.ndarray) -> np.ndarray:
    output = np.full((len(years_requested), len(proxy_ids)), np.nan)
    with h5py.File(path, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_id in enumerate(proxy_ids):
            proxy = handle[refs[proxy_id - 1]]
            years = np.asarray(proxy["age_yng"], dtype=float).reshape(-1).astype(int)
            lookup = {int(year): index for index, year in enumerate(years)}
            source_indices = [lookup.get(int(year)) for year in years_requested]
            present_rows = np.flatnonzero([index is not None for index in source_indices])
            if len(present_rows):
                indices = np.asarray([source_indices[row] for row in present_rows], dtype=int)
                values = np.asarray(proxy["data"][:, 0], dtype=float)[indices]
                output[present_rows, column] = values
    return output


def aggregate_mjj(monthly: np.ndarray) -> np.ndarray:
    result = np.full((len(CAL_YEARS), monthly.shape[1]), np.nan)
    reshaped = monthly.reshape(len(CAL_YEARS), 12, monthly.shape[1])
    for row, year in enumerate(CAL_YEARS):
        month_values = reshaped[row, 4:7]
        if np.all(np.isfinite(month_values), axis=0).any():
            days = np.asarray([calendar.monthrange(int(year), month)[1] for month in (5, 6, 7)], dtype=float)
            valid = np.all(np.isfinite(month_values), axis=0)
            result[row, valid] = np.sum(month_values[:, valid], axis=0) / days.sum()
    return result


def trend_sign(years: np.ndarray, values: np.ndarray) -> int:
    valid = np.isfinite(values)
    return int(np.sign(np.polyfit(years[valid], values[valid], 1)[0])) if int(valid.sum()) >= 3 else 0


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_134_netp_mjj_low_frequency_reconstruction_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round134_netp_mjj_low_frequency_reconstruction_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    r118 = load_module("round134_r118", driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py")
    audit = load_module("round134_archive", driver / "audit_positive_b2_event_seasonal_psm_v1.py")
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "proxy6": audit.archive_proxy_path(),
        "metadata": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
        "mjj_prior_summary": loop / "round_134_netp_mjj_exact_prior_v2/summary.json",
        "external": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_external_low_frequency_benchmark_20260818_v2/external_benchmark_allowed_values.csv",
        "round133_metrics": loop / "round_133_independent_precipitation_low_frequency_validation_v1/external_validation_metrics.csv",
        "round133_curves": loop / "round_133_independent_precipitation_low_frequency_validation_v1/branch_low_frequency_curves.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    prior_summary = json.loads(paths["mjj_prior_summary"].read_text(encoding="utf-8"))
    if not prior_summary["passes_all_gates"] or prior_summary["locked_1921_1939_accessed"]:
        raise RuntimeError("mjj-prior-contract-failed")

    metadata = pd.read_csv(paths["metadata"])
    selected_meta = metadata[metadata.proxy_index.isin(NETP_IDS)].copy()
    if len(selected_meta) != 20 or set(selected_meta.proxy_type) != {"tree"}:
        raise RuntimeError("netp-proxy-inventory-failed")
    excluded = selected_meta[selected_meta.proxy_index == EXCLUDED_ID].iloc[0]
    if abs(excluded.latitude - 37.93) > 0.01 or abs(excluded.longitude - 101.53) > 0.01:
        raise RuntimeError("qt-nearest-proxy-exclusion-failed")
    group_keys, groups = groups_from_metadata(metadata)
    if len(group_keys) != 15:
        raise RuntimeError(f"unexpected-spatial-group-count:{len(group_keys)}")

    proxy_modern = read_proxy_values(paths["proxy6"], TRAIN_IDS, CAL_YEARS)
    proxy_paleo = read_proxy_values(paths["proxy6"], TRAIN_IDS, PALEO_YEARS)
    if np.isfinite(proxy_modern).sum() == 0 or np.isfinite(proxy_paleo).sum() == 0:
        raise RuntimeError("proxy-read-failed")
    proxy_modern_smooth = centered_rolling(proxy_modern, 11)
    proxy_paleo_smooth = centered_rolling(proxy_paleo, 11)

    lat = np.asarray([35, 37, 39, 41] * 8, dtype=float)
    lon = np.repeat(np.arange(93, 109, 2), 4).astype(float)
    weights = np.cos(np.deg2rad(lat))
    mjj_targets = {}
    for product, path, variable, source_start in (
        ("GPCC", paths["gpcc"], "precip", 1891),
        ("CRU", paths["cru"], "pre", 1901),
    ):
        monthly = r118.read_netcdf3_points(path, variable, source_start, lat, lon)
        cells = aggregate_mjj(monthly)
        if cells.shape != (61, 32) or not np.isfinite(cells).all():
            raise RuntimeError(f"modern-mjj-target-failed:{product}")
        mjj_targets[product] = np.average(cells, axis=1, weights=weights)

    oof_rows = []
    oof_predictions = {product: np.full(len(CAL_YEARS), np.nan) for product in PRODUCTS}
    oof_climatology = {product: np.full(len(CAL_YEARS), np.nan) for product in PRODUCTS}
    for product in PRODUCTS:
        target = mjj_targets[product]
        for fold_name, fit_start, fit_end, test_start, test_end in FOLDS:
            fit_mask = (CAL_YEARS >= fit_start) & (CAL_YEARS <= fit_end)
            test_mask = (CAL_YEARS >= test_start) & (CAL_YEARS <= test_end)
            fold_proxy = smooth_inside_mask(proxy_modern, fit_mask) + smooth_inside_mask(proxy_modern, test_mask)
            fold_target = smooth_inside_mask(target, fit_mask) + smooth_inside_mask(target, test_mask)
            fit_valid = fit_mask & np.isfinite(fold_target)
            test_valid = test_mask & np.isfinite(fold_target)
            comp, center, scale, signs = composite(fold_proxy, fold_target, fit_valid, groups)
            prediction, coefficients = linear_fit_predict(comp, fold_target, fit_valid, test_valid)
            oof_predictions[product][test_valid] = prediction[test_valid]
            oof_climatology[product][test_valid] = np.nanmean(fold_target[fit_valid])
            fold_corr = correlation(fold_target[test_valid], prediction[test_valid])
            fold_rmse = float(np.sqrt(np.nanmean((fold_target[test_valid] - prediction[test_valid]) ** 2)))
            oof_rows.append({
                "product": product,
                "fold": fold_name,
                "fit_years": f"{fit_start}-{fit_end}",
                "test_years": f"{test_start}-{test_end}",
                "test_count": int(np.isfinite(prediction[test_valid]).sum()),
                "test_correlation": fold_corr,
                "test_rmse_mm_day": fold_rmse,
                "fit_active_proxy_count": int(np.isfinite(signs).sum()),
                "fit_slope": float(coefficients[1]),
            })
    oof = pd.DataFrame(oof_rows)
    oof_summary = []
    for product in PRODUCTS:
        target_smooth = smooth_inside_mask(mjj_targets[product], CAL_YEARS <= 1969) + smooth_inside_mask(mjj_targets[product], CAL_YEARS >= 1970)
        valid = np.isfinite(oof_predictions[product]) & np.isfinite(target_smooth)
        candidate_sse = float(np.sum((target_smooth[valid] - oof_predictions[product][valid]) ** 2))
        climatology_sse = float(np.sum((target_smooth[valid] - oof_climatology[product][valid]) ** 2))
        oof_summary.append({
            "product": product,
            "oof_count": int(valid.sum()),
            "oof_correlation": correlation(target_smooth[valid], oof_predictions[product][valid]),
            "oof_skill_vs_fold_climatology": 1 - candidate_sse / climatology_sse,
            "oof_rmse_mm_day": float(np.sqrt(candidate_sse / valid.sum())),
        })
    oof_summary_frame = pd.DataFrame(oof_summary).set_index("product")
    common_oof = np.isfinite(oof_predictions["GPCC"]) & np.isfinite(oof_predictions["CRU"])
    oof_product_correlation = correlation(oof_predictions["GPCC"][common_oof], oof_predictions["CRU"][common_oof])

    paleo_predictions = {}
    paleo_active_groups = {}
    calibration_rows = []
    for product in PRODUCTS:
        target_smooth = centered_rolling(mjj_targets[product], 11)
        modern_fit = np.isfinite(target_smooth)
        comp_modern, center, scale, signs = composite(proxy_modern_smooth, target_smooth, modern_fit, groups)
        full_prediction, coefficients = linear_fit_predict(comp_modern, target_smooth, modern_fit, np.ones(len(CAL_YEARS), dtype=bool))
        comp_paleo, _, _, _ = composite(proxy_paleo_smooth, np.full(len(PALEO_YEARS), np.nan), np.zeros(len(PALEO_YEARS), dtype=bool), groups, center, scale, signs)
        paleo_predictions[product] = coefficients[0] + coefficients[1] * comp_paleo
        z = np.full(proxy_paleo_smooth.shape, np.nan)
        valid_scale = np.isfinite(scale) & (scale > 1e-12) & np.isfinite(signs)
        z[:, valid_scale] = (proxy_paleo_smooth[:, valid_scale] - center[valid_scale]) / scale[valid_scale] * signs[valid_scale]
        group_values = np.column_stack([np.nanmean(z[:, members], axis=1) for members in groups])
        paleo_active_groups[product] = np.isfinite(group_values).sum(axis=1)
        calibration_rows.append({"product": product, "intercept": float(coefficients[0]), "slope": float(coefficients[1]), "active_proxy_count": int(valid_scale.sum()), "modern_fit_correlation": correlation(target_smooth, full_prediction)})
    paleo_predictions["marginalized"] = np.nanmean(np.column_stack((paleo_predictions["GPCC"], paleo_predictions["CRU"])), axis=1)
    product_31 = {name: centered_rolling(values, 31) for name, values in paleo_predictions.items()}
    paleo_product_correlation = correlation(product_31["GPCC"][(PALEO_YEARS >= 1500)], product_31["CRU"][(PALEO_YEARS >= 1500)])

    external = pd.read_csv(paths["external"])
    if ((external.year >= 1921) & (external.year <= 1939)).any():
        raise RuntimeError("locked-years-in-external")
    qt = external[(external.benchmark == "northwestern_qtp_mjj_precipitation") & external.year.between(1500, 1920)].sort_values("year")
    if not np.array_equal(qt.year.to_numpy(dtype=int), np.arange(1500, 1921)):
        raise RuntimeError("qt-axis-failed")
    qt31 = centered_rolling(qt.value.to_numpy(dtype=float), 31)
    baseline_metrics = pd.read_csv(paths["round133_metrics"])
    baseline_metrics = baseline_metrics[(baseline_metrics.benchmark == "qt_prec") & (baseline_metrics.branch == "baseline")].set_index("period")
    validation_rows = []
    for branch in ("GPCC", "CRU", "marginalized"):
        candidate = product_31[branch][PALEO_YEARS >= 1500]
        for start, end in ((1500, 1920), (1800, 1920)):
            use = (qt.year.to_numpy() >= start) & (qt.year.to_numpy() <= end) & np.isfinite(qt31) & np.isfinite(candidate)
            ext = qt31[use]
            model = candidate[use]
            ext_z = (ext - ext.mean()) / ext.std(ddof=1)
            model_z = (model - model.mean()) / model.std(ddof=1)
            period = f"{start}_{end}"
            baseline_corr = float(baseline_metrics.loc[period, "low_frequency_correlation"])
            validation_rows.append({
                "branch": branch,
                "period": period,
                "sample_count": int(use.sum()),
                "qt_correlation": correlation(ext_z, model_z),
                "round133_baseline_correlation": baseline_corr,
                "correlation_delta_vs_round133": correlation(ext_z, model_z) - baseline_corr,
                "qt_trend_sign": trend_sign(qt.year.to_numpy()[use], ext_z),
                "model_trend_sign": trend_sign(qt.year.to_numpy()[use], model_z),
                "trend_sign_agreement": trend_sign(qt.year.to_numpy()[use], ext_z) == trend_sign(qt.year.to_numpy()[use], model_z),
            })
    validation = pd.DataFrame(validation_rows)
    marginal = validation[validation.branch == "marginalized"].set_index("period")
    product_spread_ok = True
    for period in ("1500_1920", "1800_1920"):
        center_corr = float(marginal.loc[period, "qt_correlation"])
        individual = validation[(validation.period == period) & validation.branch.isin(PRODUCTS)].qt_correlation.to_numpy()
        product_spread_ok &= bool(np.max(np.abs(individual - center_corr)) <= 0.10)
    active_1500 = np.minimum(paleo_active_groups["GPCC"], paleo_active_groups["CRU"])[PALEO_YEARS >= 1500]
    gates = {
        "both_oof_correlations_ge_0_30": bool((oof_summary_frame.oof_correlation >= 0.30).all()),
        "both_oof_skills_positive": bool((oof_summary_frame.oof_skill_vs_fold_climatology > 0).all()),
        "all_four_fold_correlations_positive": bool((oof.test_correlation > 0).all()),
        "oof_product_correlation_ge_0_80": oof_product_correlation >= 0.80,
        "paleo_product_correlation_ge_0_80": paleo_product_correlation >= 0.80,
        "minimum_five_active_groups_1500_1920": int(active_1500.min()) >= 5,
        "qt_1500_improvement_ge_0_05": float(marginal.loc["1500_1920", "correlation_delta_vs_round133"]) >= 0.05,
        "qt_1800_not_lower_than_baseline": float(marginal.loc["1800_1920", "correlation_delta_vs_round133"]) >= 0,
        "qt_1800_trend_sign_agreement": bool(marginal.loc["1800_1920", "trend_sign_agreement"]),
        "individual_product_qt_correlations_within_0_10": product_spread_ok,
    }
    passed = bool(all(gates.values()))
    decision = "advance_to_spatial_netp_mjj_reconstruction" if passed else "close_equal_weight_netp_temporal_composite"

    oof.to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
    oof_summary_frame.reset_index().to_csv(output / "blocked_oof_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "full_calibration_coefficients.csv", index=False)
    validation.to_csv(output / "qt_prec_external_validation_metrics.csv", index=False)
    group_rows = []
    for key, members in zip(group_keys, groups):
        group_rows.append({"group_latitude": key[0], "group_longitude": key[1], "proxy_indices": ",".join(str(TRAIN_IDS[index]) for index in members), "proxy_count": len(members)})
    pd.DataFrame(group_rows).to_csv(output / "proxy_spatial_groups.csv", index=False)
    curve_frame = pd.DataFrame({"year": PALEO_YEARS})
    for branch, values in paleo_predictions.items():
        curve_frame[f"{branch}_mjj_mm_day"] = values
        curve_frame[f"{branch}_31yr"] = product_31[branch]
    curve_frame["active_group_count"] = np.minimum(paleo_active_groups["GPCC"], paleo_active_groups["CRU"])
    curve_frame.to_csv(output / "netp_mjj_reconstruction_curves.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.0), constrained_layout=True)
    for product, color in (("GPCC", "#0072B2"), ("CRU", "#D55E00")):
        target_smooth = smooth_inside_mask(mjj_targets[product], CAL_YEARS <= 1969) + smooth_inside_mask(mjj_targets[product], CAL_YEARS >= 1970)
        axes[0].plot(CAL_YEARS, target_smooth, color=color, linewidth=1.5, label=f"{product} observed MJJ")
        axes[0].plot(CAL_YEARS, oof_predictions[product], color=color, linewidth=1.0, linestyle="--", label=f"{product} blocked OOF")
    axes[0].set(title="NETP MJJ modern blocked OOF", ylabel="MJJ precipitation (mm/day)")
    qt_use = np.isfinite(qt31)
    qt_z = (qt31 - np.nanmean(qt31)) / np.nanstd(qt31, ddof=1)
    axes[1].plot(qt.year[qt_use], qt_z[qt_use], color="#CC79A7", linewidth=2.0, label="QT-Prec external")
    colors = {"GPCC": "#0072B2", "CRU": "#D55E00", "marginalized": "#111111"}
    for branch in ("GPCC", "CRU", "marginalized"):
        values = product_31[branch][PALEO_YEARS >= 1500]
        values_z = (values - np.nanmean(values)) / np.nanstd(values, ddof=1)
        corr_value = float(validation[(validation.branch == branch) & (validation.period == "1500_1920")].qt_correlation.iloc[0])
        axes[1].plot(PALEO_YEARS[PALEO_YEARS >= 1500], values_z, color=colors[branch], linewidth=1.2 if branch != "marginalized" else 1.8, label=f"{branch} (r={corr_value:.3f})")
    axes[1].set(title="NETP MJJ reconstruction versus independent QT-Prec", xlabel="year", ylabel="standardized 31-year anomaly")
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(output / "round134_netp_mjj_reconstruction_curves.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_134_netp_mjj_low_frequency_reconstruction_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "proxy_inventory": {"netp_count": 20, "excluded_proxy_index": EXCLUDED_ID, "training_count": len(TRAIN_IDS), "spatial_group_count": len(groups)},
        "years_read": {"proxy_paleo": [1000, 1920], "proxy_and_instrumental_calibration": [1940, 2000]},
        "locked_1921_1939_accessed": False,
        "external_validation_reserve_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "oof_product_correlation": oof_product_correlation,
        "paleo_product_correlation_1500_1920": paleo_product_correlation,
        "minimum_active_group_count_1500_1920": int(active_1500.min()),
        "gates": gates,
        "passes_all_gates": passed,
        "decision": decision,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 134 NETP MJJ Low-Frequency Reconstruction\n\n"
        f"- GPCC/CRU OOF correlations: `{oof_summary_frame.loc['GPCC','oof_correlation']:.3f}` / `{oof_summary_frame.loc['CRU','oof_correlation']:.3f}`.\n"
        f"- GPCC/CRU OOF skills: `{oof_summary_frame.loc['GPCC','oof_skill_vs_fold_climatology']:.3f}` / `{oof_summary_frame.loc['CRU','oof_skill_vs_fold_climatology']:.3f}`.\n"
        f"- Marginalized QT-Prec correlation changes: `{marginal.loc['1500_1920','correlation_delta_vs_round133']:+.3f}` (1500-1920), `{marginal.loc['1800_1920','correlation_delta_vs_round133']:+.3f}` (1800-1920).\n"
        f"- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Southern Tibetan Plateau seasonal precipitation reconstruction and held-out validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAL_YEARS = np.arange(1941, 2001, dtype=int)
SOURCE_MONTHLY_YEARS = np.arange(1940, 2001, dtype=int)
PALEO_YEARS = np.arange(1000, 1921, dtype=int)
PRODUCTS = ("GPCC", "CRU")
SEASONS = {
    "pSep_cJun": ((8, 9, 10, 11), (0, 1, 2, 3, 4, 5)),
    "pJul_cApr": ((6, 7, 8, 9, 10, 11), (0, 1, 2, 3)),
}
FOLDS = (
    ("early_to_late", 1941, 1970, 1971, 2000),
    ("late_to_early", 1971, 2000, 1941, 1970),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= 100_000_000:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def trend_per_century(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def generic_groups(metadata: pd.DataFrame, proxy_ids: tuple[int, ...]):
    selected = metadata.set_index("proxy_index").loc[list(proxy_ids)].reset_index()
    selected["group_latitude"] = np.round(selected.latitude.to_numpy() * 4.0) / 4.0
    selected["group_longitude"] = np.round(selected.longitude.to_numpy() * 4.0) / 4.0
    keys = sorted(set(zip(selected.group_latitude, selected.group_longitude)))
    ids = np.asarray(proxy_ids)
    groups: list[np.ndarray] = []
    for latitude, longitude in keys:
        members = selected.loc[
            (selected.group_latitude == latitude) & (selected.group_longitude == longitude),
            "proxy_index",
        ].to_numpy(dtype=int)
        groups.append(np.flatnonzero(np.isin(ids, members)))
    return selected, keys, groups


def aggregate_crossyear(monthly: np.ndarray, previous: tuple[int, ...], current: tuple[int, ...]) -> np.ndarray:
    expected = len(SOURCE_MONTHLY_YEARS) * 12
    if monthly.shape[0] != expected:
        raise RuntimeError(f"monthly-axis-failed:{monthly.shape[0]}:{expected}")
    cube = monthly.reshape(len(SOURCE_MONTHLY_YEARS), 12, monthly.shape[1])
    output = np.full((len(CAL_YEARS), monthly.shape[1]), np.nan)
    for row, year in enumerate(CAL_YEARS):
        source_row = int(year - SOURCE_MONTHLY_YEARS[0])
        values = np.concatenate((cube[source_row - 1, previous], cube[source_row, current]), axis=0)
        valid = np.all(np.isfinite(values), axis=0)
        output[row, valid] = np.sum(values[:, valid], axis=0)
    return output


def read_external(path: Path, years: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if frame.shape[1] != 2:
        raise RuntimeError(f"external-column-count:{path.name}:{frame.shape[1]}")
    year = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    value = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    if ((year >= 1921) & (year <= 1939)).any():
        raise RuntimeError(f"locked-row-present-in-safe-export:{path.name}")
    lookup = dict(zip(year.dropna().astype(int), value[year.notna()]))
    return np.asarray([lookup.get(int(item), np.nan) for item in years], dtype=float)


def standardized_pair(left: np.ndarray, right: np.ndarray):
    valid = np.isfinite(left) & np.isfinite(right)
    left_z = np.full(len(left), np.nan)
    right_z = np.full(len(right), np.nan)
    if int(valid.sum()) >= 3:
        left_z[valid] = (left[valid] - np.mean(left[valid])) / np.std(left[valid], ddof=1)
        right_z[valid] = (right[valid] - np.mean(right[valid])) / np.std(right[valid], ddof=1)
    return valid, left_z, right_z


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_141_south_tp_crossyear_precipitation_external_validation_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    contract = driver / "round141_south_tp_crossyear_precipitation_external_validation_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    base = load_module("round141_base", driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py")
    r118 = load_module(
        "round141_r118",
        driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    audit = load_module("round141_archive", driver / "audit_positive_b2_event_seasonal_psm_v1.py")
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    safe_external = loop / "round_139_east_hma_xlsx_preflight_v2"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "base_driver": driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py",
        "r118_driver": driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
        "proxy6": audit.archive_proxy_path(),
        "metadata": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
        "biru": safe_external / "sb__biru_reconstruction__allowed.csv",
        "jiali": safe_external / "sb__jiali_reconstruction__allowed.csv",
        "round139_summary": safe_external / "summary.json",
        "round140_summary": loop / "round_140_east_hma_external_identity_role_closure_v1/summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    round139 = json.loads(paths["round139_summary"].read_text(encoding="utf-8"))
    round140 = json.loads(paths["round140_summary"].read_text(encoding="utf-8"))
    if round139.get("locked_1921_1939_values_decoded", 1) != 0:
        raise RuntimeError("round139-boundary-failed")
    if round140.get("decision") != "advance_east_hma_reconstruction_with_biru_jiali_held_out":
        raise RuntimeError("round140-role-decision-failed")

    metadata = pd.read_csv(paths["metadata"])
    selected = metadata[
        metadata.latitude.between(27.0, 35.0)
        & metadata.longitude.between(90.0, 102.0)
        & (metadata.first_observed_year <= 1500)
        & (metadata.last_observed_year >= 2000)
    ].copy()
    proxy_ids = tuple(sorted(selected.proxy_index.astype(int)))
    if len(proxy_ids) != 38 or set(selected.proxy_type) != {"tree"}:
        raise RuntimeError(f"frozen-proxy-inventory-failed:{len(proxy_ids)}:{set(selected.proxy_type)}")
    selected, group_keys, groups = generic_groups(metadata, proxy_ids)

    proxy_modern = base.read_proxy_values(paths["proxy6"], proxy_ids, CAL_YEARS)
    proxy_paleo = base.read_proxy_values(paths["proxy6"], proxy_ids, PALEO_YEARS)
    if not np.isfinite(proxy_modern).any() or not np.isfinite(proxy_paleo).any():
        raise RuntimeError("proxy-read-failed")
    proxy_modern_smooth = base.centered_rolling(proxy_modern, 11)
    proxy_paleo_smooth = base.centered_rolling(proxy_paleo, 11)

    lat = np.tile(np.asarray([27, 29, 31, 33, 35], dtype=float), 6)
    lon = np.repeat(np.asarray([91, 93, 95, 97, 99, 101], dtype=float), 5)
    weights = np.cos(np.deg2rad(lat))
    targets: dict[tuple[str, str], np.ndarray] = {}
    for product, path, variable, source_start in (
        ("GPCC", paths["gpcc"], "precip", 1891),
        ("CRU", paths["cru"], "pre", 1901),
    ):
        monthly = r118.read_netcdf3_points(path, variable, source_start, lat, lon)
        if monthly.shape != (61 * 12, 30):
            raise RuntimeError(f"modern-monthly-shape:{product}:{monthly.shape}")
        for season, (previous, current) in SEASONS.items():
            cells = aggregate_crossyear(monthly, previous, current)
            if cells.shape != (60, 30) or not np.isfinite(cells).all():
                raise RuntimeError(f"seasonal-target-failed:{product}:{season}")
            targets[(season, product)] = np.average(cells, axis=1, weights=weights)

    fold_rows: list[dict[str, object]] = []
    oof_predictions: dict[tuple[str, str], np.ndarray] = {}
    oof_climatology: dict[tuple[str, str], np.ndarray] = {}
    oof_observed: dict[tuple[str, str], np.ndarray] = {}
    for season in SEASONS:
        for product in PRODUCTS:
            key = (season, product)
            target = targets[key]
            prediction_all = np.full(len(CAL_YEARS), np.nan)
            climatology_all = np.full(len(CAL_YEARS), np.nan)
            observed_all = np.full(len(CAL_YEARS), np.nan)
            for fold_name, fit_start, fit_end, test_start, test_end in FOLDS:
                fit_mask = (CAL_YEARS >= fit_start) & (CAL_YEARS <= fit_end)
                test_mask = (CAL_YEARS >= test_start) & (CAL_YEARS <= test_end)
                fold_proxy = np.fmax(
                    base.smooth_inside_mask(proxy_modern, fit_mask),
                    base.smooth_inside_mask(proxy_modern, test_mask),
                )
                fold_target = np.fmax(
                    base.smooth_inside_mask(target, fit_mask),
                    base.smooth_inside_mask(target, test_mask),
                )
                fit_valid = fit_mask & np.isfinite(fold_target)
                test_valid = test_mask & np.isfinite(fold_target)
                comp, center, scale, signs = base.composite(fold_proxy, fold_target, fit_valid, groups)
                prediction, coefficients = base.linear_fit_predict(comp, fold_target, fit_valid, test_valid)
                prediction_all[test_valid] = prediction[test_valid]
                climatology_all[test_valid] = np.nanmean(fold_target[fit_valid])
                observed_all[test_valid] = fold_target[test_valid]
                fold_rows.append({
                    "season": season,
                    "product": product,
                    "fold": fold_name,
                    "fit_years": f"{fit_start}-{fit_end}",
                    "test_years": f"{test_start}-{test_end}",
                    "test_count": int(np.isfinite(prediction[test_valid]).sum()),
                    "test_correlation": base.correlation(fold_target[test_valid], prediction[test_valid]),
                    "test_rmse_mm_season": float(np.sqrt(np.nanmean((fold_target[test_valid] - prediction[test_valid]) ** 2))),
                    "fit_active_proxy_count": int(np.isfinite(signs).sum()),
                    "fit_intercept": float(coefficients[0]),
                    "fit_slope": float(coefficients[1]),
                })
            oof_predictions[key] = prediction_all
            oof_climatology[key] = climatology_all
            oof_observed[key] = observed_all

    fold_metrics = pd.DataFrame(fold_rows)
    oof_rows: list[dict[str, object]] = []
    for season in SEASONS:
        for product in PRODUCTS:
            key = (season, product)
            valid = np.isfinite(oof_observed[key]) & np.isfinite(oof_predictions[key])
            candidate_sse = float(np.sum((oof_observed[key][valid] - oof_predictions[key][valid]) ** 2))
            climatology_sse = float(np.sum((oof_observed[key][valid] - oof_climatology[key][valid]) ** 2))
            oof_rows.append({
                "season": season,
                "product": product,
                "oof_count": int(valid.sum()),
                "oof_correlation": base.correlation(oof_observed[key][valid], oof_predictions[key][valid]),
                "oof_skill_vs_fold_climatology": 1.0 - candidate_sse / climatology_sse,
                "oof_rmse_mm_season": float(np.sqrt(candidate_sse / valid.sum())),
            })
        common = np.isfinite(oof_predictions[(season, "GPCC")]) & np.isfinite(oof_predictions[(season, "CRU")])
        product_corr = base.correlation(
            oof_predictions[(season, "GPCC")][common],
            oof_predictions[(season, "CRU")][common],
        )
        for row in oof_rows[-2:]:
            row["gpcc_cru_oof_prediction_correlation"] = product_corr
    oof_summary = pd.DataFrame(oof_rows)

    paleo: dict[tuple[str, str], np.ndarray] = {}
    paleo_31: dict[tuple[str, str], np.ndarray] = {}
    active_groups: dict[tuple[str, str], np.ndarray] = {}
    calibration_rows: list[dict[str, object]] = []
    for season in SEASONS:
        for product in PRODUCTS:
            key = (season, product)
            target_smooth = base.centered_rolling(targets[key], 11)
            fit = np.isfinite(target_smooth)
            comp_modern, center, scale, signs = base.composite(proxy_modern_smooth, target_smooth, fit, groups)
            fitted, coefficients = base.linear_fit_predict(comp_modern, target_smooth, fit, np.ones(len(CAL_YEARS), dtype=bool))
            comp_paleo, _, _, _ = base.composite(
                proxy_paleo_smooth,
                np.full(len(PALEO_YEARS), np.nan),
                np.zeros(len(PALEO_YEARS), dtype=bool),
                groups,
                center,
                scale,
                signs,
            )
            values = coefficients[0] + coefficients[1] * comp_paleo
            paleo[key] = values
            paleo_31[key] = base.centered_rolling(values, 31)
            z = np.full(proxy_paleo_smooth.shape, np.nan)
            usable = np.isfinite(scale) & (scale > 1e-12) & np.isfinite(signs)
            z[:, usable] = (proxy_paleo_smooth[:, usable] - center[usable]) / scale[usable] * signs[usable]
            group_values = np.column_stack([
                np.divide(
                    np.nansum(z[:, members], axis=1),
                    np.isfinite(z[:, members]).sum(axis=1),
                    out=np.full(len(PALEO_YEARS), np.nan),
                    where=np.isfinite(z[:, members]).sum(axis=1) > 0,
                )
                for members in groups
            ])
            active_groups[key] = np.isfinite(group_values).sum(axis=1)
            calibration_rows.append({
                "season": season,
                "product": product,
                "intercept_mm_season": float(coefficients[0]),
                "slope_mm_season_per_composite_z": float(coefficients[1]),
                "active_proxy_count": int(usable.sum()),
                "modern_full_fit_correlation": base.correlation(target_smooth, fitted),
            })
        product_pair = np.column_stack((paleo[(season, "GPCC")], paleo[(season, "CRU")]))
        paleo[(season, "marginalized")] = np.nanmean(product_pair, axis=1)
        paleo_31[(season, "marginalized")] = base.centered_rolling(paleo[(season, "marginalized")], 31)

    product_curve_correlations = {}
    for season in SEASONS:
        valid = (PALEO_YEARS >= 1500) & np.isfinite(paleo_31[(season, "GPCC")]) & np.isfinite(paleo_31[(season, "CRU")])
        product_curve_correlations[season] = base.correlation(
            paleo_31[(season, "GPCC")][valid],
            paleo_31[(season, "CRU")][valid],
        )

    external = {
        "Biru": read_external(paths["biru"], PALEO_YEARS),
        "Jiali": read_external(paths["jiali"], PALEO_YEARS),
    }
    external_31 = {name: base.centered_rolling(values, 31) for name, values in external.items()}
    matches = {"pSep_cJun": "Biru", "pJul_cApr": "Jiali"}
    validation_rows: list[dict[str, object]] = []
    for season, benchmark in matches.items():
        model = paleo_31[(season, "marginalized")]
        reference = external_31[benchmark]
        common = np.isfinite(model) & np.isfinite(reference) & (PALEO_YEARS <= 1900)
        if int(common.sum()) < 50:
            raise RuntimeError(f"insufficient-external-common-years:{benchmark}:{common.sum()}")
        common_start = int(PALEO_YEARS[common].min())
        for label, start, end in (
            ("full_effective", common_start, 1900),
            ("1800_1900", 1800, 1900),
        ):
            use = common & (PALEO_YEARS >= start) & (PALEO_YEARS <= end)
            valid, model_z, reference_z = standardized_pair(model[use], reference[use])
            years = PALEO_YEARS[use][valid]
            mz = model_z[valid]
            rz = reference_z[valid]
            model_trend = trend_per_century(years, mz)
            external_trend = trend_per_century(years, rz)
            validation_rows.append({
                "season": season,
                "benchmark": benchmark,
                "period": label,
                "start_year": int(years.min()),
                "end_year": int(years.max()),
                "sample_count": int(len(years)),
                "low_frequency_correlation": base.correlation(mz, rz),
                "model_standardized_trend_per_century": model_trend,
                "external_standardized_trend_per_century": external_trend,
                "trend_sign_agreement": bool(np.sign(model_trend) == np.sign(external_trend)),
                "raw_model_mean_mm_season": float(np.nanmean(model[use])),
                "raw_external_mean_mm": float(np.nanmean(reference[use])),
            })
    validation = pd.DataFrame(validation_rows)

    oof_index = oof_summary.set_index(["season", "product"])
    val_index = validation.set_index(["season", "period"])
    gates = {
        "all_eight_fold_correlations_positive": bool((fold_metrics.test_correlation > 0).all()),
        "all_four_oof_correlations_ge_0_50": bool((oof_summary.oof_correlation >= 0.50).all()),
        "all_four_oof_skills_positive": bool((oof_summary.oof_skill_vs_fold_climatology > 0).all()),
        "both_seasons_oof_product_correlation_ge_0_80": bool(
            all(oof_index.loc[(season, "GPCC"), "gpcc_cru_oof_prediction_correlation"] >= 0.80 for season in SEASONS)
        ),
        "both_seasons_paleo_product_correlation_ge_0_80": bool(
            all(product_curve_correlations[season] >= 0.80 for season in SEASONS)
        ),
        "biru_full_correlation_ge_0_25": bool(val_index.loc[("pSep_cJun", "full_effective"), "low_frequency_correlation"] >= 0.25),
        "jiali_full_correlation_ge_0_25": bool(val_index.loc[("pJul_cApr", "full_effective"), "low_frequency_correlation"] >= 0.25),
        "biru_1800_1900_correlation_ge_0_20": bool(val_index.loc[("pSep_cJun", "1800_1900"), "low_frequency_correlation"] >= 0.20),
        "jiali_1800_1900_correlation_ge_0_20": bool(val_index.loc[("pJul_cApr", "1800_1900"), "low_frequency_correlation"] >= 0.20),
        "biru_1800_1900_trend_sign_agreement": bool(val_index.loc[("pSep_cJun", "1800_1900"), "trend_sign_agreement"]),
        "jiali_1800_1900_trend_sign_agreement": bool(val_index.loc[("pJul_cApr", "1800_1900"), "trend_sign_agreement"]),
    }
    passed = bool(all(gates.values()))
    modern_keys = tuple(list(gates)[:5])
    modern_passed = bool(all(gates[key] for key in modern_keys))
    if passed:
        decision = "advance_south_tp_temporal_candidate_to_robustness"
    elif modern_passed:
        decision = "close_south_tp_transfer_external_validation_failed"
    else:
        decision = "close_south_tp_transfer_modern_skill_failed"

    fold_metrics.to_csv(output / "blocked_oof_fold_metrics.csv", index=False)
    oof_summary.to_csv(output / "blocked_oof_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "full_calibration_coefficients.csv", index=False)
    validation.to_csv(output / "external_validation_metrics.csv", index=False)
    inventory = selected.sort_values("proxy_index").copy()
    inventory.to_csv(output / "proxy_inventory.csv", index=False)
    group_rows = []
    for key, members in zip(group_keys, groups):
        group_rows.append({
            "group_latitude": key[0],
            "group_longitude": key[1],
            "proxy_indices": ",".join(str(proxy_ids[index]) for index in members),
            "proxy_count": len(members),
        })
    pd.DataFrame(group_rows).to_csv(output / "proxy_spatial_groups.csv", index=False)

    modern_curves = pd.DataFrame({"year": CAL_YEARS})
    for season in SEASONS:
        for product in PRODUCTS:
            key = (season, product)
            modern_curves[f"{season}_{product}_observed_11yr_blocked_mm_season"] = oof_observed[key]
            modern_curves[f"{season}_{product}_oof_mm_season"] = oof_predictions[key]
    modern_curves.to_csv(output / "modern_oof_curves.csv", index=False)
    paleo_curves = pd.DataFrame({"year": PALEO_YEARS})
    for season in SEASONS:
        for product in (*PRODUCTS, "marginalized"):
            paleo_curves[f"{season}_{product}_mm_season"] = paleo[(season, product)]
            paleo_curves[f"{season}_{product}_31yr"] = paleo_31[(season, product)]
        paleo_curves[f"{season}_minimum_active_group_count"] = np.minimum(
            active_groups[(season, "GPCC")], active_groups[(season, "CRU")]
        )
    paleo_curves["Biru_31yr_mm"] = external_31["Biru"]
    paleo_curves["Jiali_31yr_mm"] = external_31["Jiali"]
    paleo_curves.to_csv(output / "paleo_and_external_curves.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    colors = {"GPCC": "#0072B2", "CRU": "#D55E00"}
    for column, season in enumerate(SEASONS):
        for product in PRODUCTS:
            key = (season, product)
            axes[0, column].plot(CAL_YEARS, oof_observed[key], color=colors[product], linewidth=1.5, label=f"{product} observed")
            axes[0, column].plot(CAL_YEARS, oof_predictions[key], color=colors[product], linewidth=1.0, linestyle="--", label=f"{product} OOF")
        axes[0, column].set_title(f"{season}: blocked modern OOF")
        axes[0, column].set_ylabel("precipitation (mm/season)")
        benchmark = matches[season]
        model = paleo_31[(season, "marginalized")]
        reference = external_31[benchmark]
        plot_mask = (PALEO_YEARS >= 1600) & (PALEO_YEARS <= 1900) & np.isfinite(model) & np.isfinite(reference)
        _, model_z, external_z = standardized_pair(model[plot_mask], reference[plot_mask])
        full_corr = float(val_index.loc[(season, "full_effective"), "low_frequency_correlation"])
        corr_19 = float(val_index.loc[(season, "1800_1900"), "low_frequency_correlation"])
        axes[1, column].plot(PALEO_YEARS[plot_mask], external_z, color="#009E73", linewidth=1.8, label=f"{benchmark} held out")
        axes[1, column].plot(PALEO_YEARS[plot_mask], model_z, color="#111111", linewidth=1.6, label="GPCC/CRU mean")
        axes[1, column].axvspan(1800, 1900, color="#E6E6E6", zorder=0)
        axes[1, column].set_title(f"{benchmark}: full r={full_corr:.3f}, 1800-1900 r={corr_19:.3f}")
        axes[1, column].set_xlabel("year")
        axes[1, column].set_ylabel("standardized 31-year anomaly")
    for axis in axes.flat:
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output / "round141_south_tp_crossyear_precipitation_curves.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_141_south_tp_crossyear_precipitation_external_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "region": {"latitude": [27, 35], "longitude": [90, 102], "grid_cell_count": 30},
        "proxy_count": len(proxy_ids),
        "spatial_group_count": len(groups),
        "years_read": {"proxy_paleo": [1000, 1920], "modern_monthly": [1940, 2000], "calibration": [1941, 2000]},
        "locked_1921_1939_accessed": False,
        "external_validation_reserve_opened": False,
        "external_values_used_for_fitting": False,
        "enkf_run": False,
        "oggm_run": False,
        "oof": oof_summary.to_dict(orient="records"),
        "paleo_product_correlations": product_curve_correlations,
        "external_validation": validation.to_dict(orient="records"),
        "gates": gates,
        "modern_gates_pass": modern_passed,
        "passes_all_gates": passed,
        "decision": decision,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 141 Southern Tibetan Plateau Cross-Year Precipitation\n\n"
        + "\n".join(
            f"- {row.season} {row.product}: OOF r=`{row.oof_correlation:.3f}`, skill=`{row.oof_skill_vs_fold_climatology:.3f}`."
            for row in oof_summary.itertuples()
        )
        + "\n"
        + "\n".join(
            f"- {row.benchmark} {row.period}: r=`{row.low_frequency_correlation:.3f}`, trend sign agreement=`{row.trend_sign_agreement}`."
            for row in validation.itertuples()
        )
        + f"\n- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit the 1901-1920 HMA precipitation signal with UDel, CRU, and GPCC."""

from __future__ import annotations

import calendar
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EARLY_YEARS = np.arange(1901, 1921)
MODERN_YEARS = np.arange(1940, 2001)
OBSERVATIONS = {"CRU": "pr_obs_cru_pre_05deg", "GPCC": "pr_obs_gpcp_05deg"}
REGIONS = {"hma": (-np.inf, np.inf), "west_lt80": (-np.inf, 80.0), "central_80_90": (80.0, 90.0), "east_ge90": (90.0, np.inf)}


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    stat = path.stat(); return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float); fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze()); values[np.isclose(values, fill) | (values < -9000)] = np.nan; return values


def decode_dates(handle: h5py.File) -> list[datetime]:
    units = np.asarray(handle["time"].attrs["units"]).reshape(-1)[0]; units = units.decode() if isinstance(units, bytes) else str(units)
    if units != "days since 1900-01-01 00:00:00": raise RuntimeError(f"unexpected-udel-time-units:{units}")
    return [datetime(1900, 1, 1) + timedelta(days=float(value)) for value in np.asarray(handle["time"], dtype=float)]


def annualize_udel(path: Path, expected_years: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dates = decode_dates(handle); lat = np.asarray(handle["lat"], dtype=float); lon = np.asarray(handle["lon"], dtype=float)
        if [(date.year, date.month) for date in dates] != [(int(year), month) for year in expected_years for month in range(1, 13)]: raise RuntimeError(f"udel-month-axis-failed:{path.name}")
        lat_index = np.asarray([np.flatnonzero(np.isclose(lat, value, atol=1e-7))[0] for value in target_lat], dtype=int); lon_index = np.asarray([np.flatnonzero(np.isclose(lon, value, atol=1e-7))[0] for value in target_lon], dtype=int)
        monthly = np.asarray(handle["precip"][:, lat_index, :][:, :, lon_index], dtype=float)
    annual = np.full((len(expected_years), len(target_lat), len(target_lon)), np.nan)
    for index, year in enumerate(expected_years):
        values = monthly[index * 12:(index + 1) * 12]; complete = np.all(np.isfinite(values), axis=0); total_mm = np.nansum(values, axis=0) * 10.0; annual[index] = np.where(complete, total_mm / (366 if calendar.isleap(int(year)) else 365), np.nan)
    return annual


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask; valid = np.isfinite(values); return np.nansum(values * weights[None], axis=(1, 2)) / np.where(np.sum(valid * weights[None], axis=(1, 2)) == 0, np.nan, np.sum(valid * weights[None], axis=(1, 2)))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[valid], b[valid])[0, 1]) if valid.sum() >= 3 else np.nan


def smooth(values: np.ndarray, width: int = 15) -> np.ndarray:
    result = np.full(len(values), np.nan); half = width // 2
    for index in range(half, len(values) - half):
        if np.all(np.isfinite(values[index-half:index+half+1])): result[index] = np.mean(values[index-half:index+half+1])
    return result


def main() -> int:
    project = root(); loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"; output = loop / "round_45_udel_1901_1920_precipitation_product_audit_v1"
    paths = {"udel_early": output / "udel_precip_v501_1901_1920_hma_bbox.nc", "udel_modern": loop / "round_44_udel_allowed_subset_acquisition_v2/udel_precip_v501_1940_2000_hma_bbox.nc", "public_source": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"), "target_grid": loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc", "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy", "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    with netcdf_file(paths["target_grid"], "r", mmap=False) as target:
        lat = np.asarray(target.variables["lat"].data, dtype=float).copy(); lon = np.asarray(target.variables["lon"].data, dtype=float).copy()
    mask = np.load(paths["hma_mask"]).astype(bool); longitude_grid = np.broadcast_to(lon[None], mask.shape); region_masks = {name: mask & (longitude_grid >= lower) & (longitude_grid < upper) for name, (lower, upper) in REGIONS.items()}
    udel_early = annualize_udel(paths["udel_early"], EARLY_YEARS, lat, lon); udel_modern = annualize_udel(paths["udel_modern"], MODERN_YEARS, lat, lon); baseline_modern = (MODERN_YEARS >= 1961) & (MODERN_YEARS <= 1990); udel_baseline = np.nanmean(udel_modern[baseline_modern], axis=0); udel = {"early": udel_early - udel_baseline[None], "modern": udel_modern - udel_baseline[None]}
    public: dict[str, dict[str, np.ndarray]] = {name: {} for name in OBSERVATIONS}
    with h5py.File(paths["public_source"], "r") as handle:
        all_years = np.asarray(handle["time"], dtype=int).ravel(); early_idx = np.flatnonzero((all_years >= 1901) & (all_years <= 1920)); modern_idx = np.flatnonzero((all_years >= 1940) & (all_years <= 2000)); baseline_idx = np.flatnonzero((all_years >= 1961) & (all_years <= 1990))
        if not np.array_equal(all_years[early_idx], EARLY_YEARS) or not np.array_equal(all_years[modern_idx], MODERN_YEARS): raise RuntimeError("public-year-axis-failed")
        if any(np.any((all_years[index] >= 1921) & (all_years[index] <= 1939)) for index in (early_idx, modern_idx, baseline_idx)): raise RuntimeError("locked-period-index-requested")
        for name, variable in OBSERVATIONS.items():
            baseline = np.nanmean(clean(handle[variable], slice(int(baseline_idx[0]), int(baseline_idx[-1]) + 1)), axis=0)
            public[name]["early"] = clean(handle[variable], slice(int(early_idx[0]), int(early_idx[-1]) + 1)) - baseline[None]
            public[name]["modern"] = clean(handle[variable], slice(int(modern_idx[0]), int(modern_idx[-1]) + 1)) - baseline[None]

    modern_rows: list[dict[str, object]] = []; modern_series: dict[tuple[str, str], np.ndarray] = {}
    for observation in OBSERVATIONS:
        for region, region_mask in region_masks.items():
            common = region_mask & np.all(np.isfinite(udel["modern"]), axis=0) & np.all(np.isfinite(public[observation]["modern"]), axis=0)
            us = weighted_series(udel["modern"], common, lat); os = weighted_series(public[observation]["modern"], common, lat); modern_series[("UDel", region)] = us; modern_series[(observation, region)] = os
            modern_rows.append({"observation": observation, "region": region, "common_cell_count": int(common.sum()), "annual_correlation": correlation(us, os), "correlation_15year": correlation(smooth(us), smooth(os)), "udel_to_observation_std_ratio": float(np.std(us, ddof=1) / np.std(os, ddof=1)), "rmse_mm_day": float(np.sqrt(np.mean((us - os) ** 2)))})
    modern = pd.DataFrame(modern_rows); modern.to_csv(output / "modern_udel_cru_gpcc_agreement.csv", index=False)
    hma_modern = modern[modern.region == "hma"].set_index("observation"); gates = {"hma_annual_correlations_at_least_0_50": bool((hma_modern.annual_correlation >= 0.50).all()), "median_regional_annual_correlation_each_product_at_least_0_40": all(float(modern[modern.observation == product].annual_correlation.median()) >= 0.40 for product in OBSERVATIONS), "hma_std_ratios_between_0_5_and_2": bool(((hma_modern.udel_to_observation_std_ratio >= 0.5) & (hma_modern.udel_to_observation_std_ratio <= 2.0)).all()), "hma_15year_correlations_at_least_0_50": bool((hma_modern.correlation_15year >= 0.50).all())}; udel_compatible = bool(all(gates.values()))

    early_rows: list[dict[str, object]] = []; early_series_rows: list[dict[str, object]] = []
    products = {"UDel": udel["early"], "CRU": public["CRU"]["early"], "GPCC": public["GPCC"]["early"]}
    for region, region_mask in region_masks.items():
        common = region_mask.copy()
        for values in products.values(): common &= np.all(np.isfinite(values), axis=0)
        series_by_product = {name: weighted_series(values, common, lat) for name, values in products.items()}
        for index, year in enumerate(EARLY_YEARS): early_series_rows.append({"year": int(year), "region": region, **{name: float(values[index]) for name, values in series_by_product.items()}})
        for name, values in series_by_product.items():
            slope = float(np.polyfit(EARLY_YEARS, values, 1)[0] * 100); change = float(np.mean(values[-5:]) - np.mean(values[:5])); early_rows.append({"region": region, "product": name, "common_cell_count": int(common.sum()), "trend_mm_day_per_century": slope, "change_1916_1920_minus_1901_1905_mm_day": change, "mean_1901_1920_mm_day": float(np.mean(values))})
        for left, right in (("UDel", "CRU"), ("UDel", "GPCC"), ("CRU", "GPCC")): early_rows.append({"region": region, "product": f"correlation_{left}_{right}", "common_cell_count": int(common.sum()), "trend_mm_day_per_century": np.nan, "change_1916_1920_minus_1901_1905_mm_day": np.nan, "mean_1901_1920_mm_day": correlation(series_by_product[left], series_by_product[right])})
    early = pd.DataFrame(early_rows); early_series_frame = pd.DataFrame(early_series_rows); early.to_csv(output / "early_product_metrics.csv", index=False); early_series_frame.to_csv(output / "early_regional_series.csv", index=False)
    hma_early = early[(early.region == "hma") & early["product"].isin(products)].set_index("product"); all_trends_negative = bool((hma_early.trend_mm_day_per_century < 0).all()); all_changes_negative = bool((hma_early.change_1916_1920_minus_1901_1905_mm_day < 0).all()); robust_decline = bool(udel_compatible and all_trends_negative and all_changes_negative)
    summary = {"status": "complete", "locked_1921_1939_accessed": False, "udel_modern_compatibility_gates": gates, "udel_passes_modern_compatibility": udel_compatible, "hma_modern_metrics": hma_modern.reset_index().to_dict(orient="records"), "hma_1901_1920_metrics": hma_early.reset_index().to_dict(orient="records"), "all_three_hma_trends_negative": all_trends_negative, "all_three_hma_endpoint_changes_negative": all_changes_negative, "robust_three_product_1901_1920_decline": robust_decline, "decision": "treat_1901_1920_decline_as_three_product_supported" if robust_decline else ("diagnose_product_specific_early_disagreement" if udel_compatible else "do_not_use_udel_to_adjudicate_early_signal")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True)
    for product, color in [("UDel", "#222222"), ("CRU", "#d55e00"), ("GPCC", "#009e73")]: axes[0].plot(MODERN_YEARS, modern_series[(product, "hma")], label=product, color=color); axes[1].plot(EARLY_YEARS, early_series_frame[early_series_frame.region == "hma"][product], marker="o", label=product, color=color)
    axes[0].set(title="Modern HMA precipitation-product agreement", ylabel="Anomaly (mm/day)"); axes[1].set(title="1901-1920 HMA precipitation curves", xlabel="Year", ylabel="Anomaly (mm/day)"); axes[0].legend(frameon=False, ncol=3); axes[1].legend(frameon=False, ncol=3); figure.savefig(output / "udel_cru_gpcc_hma_curves.png", dpi=180); plt.close(figure)
    lines = ["# UDel 1901-1920 Precipitation Product Audit", "", f"- UDel modern compatibility passes: `{str(udel_compatible).lower()}`."]
    for product, row in hma_early.iterrows(): lines.append(f"- {product}: trend `{row.trend_mm_day_per_century:.3f}` mm/day/century; 1916-1920 minus 1901-1905 `{row.change_1916_1920_minus_1901_1905_mm_day:.3f}` mm/day.")
    lines.extend(["", f"Robust three-product decline: `{str(robust_decline).lower()}`.", f"Decision: `{summary['decision']}`."]); (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}; unchanged = before == after; (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged: raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

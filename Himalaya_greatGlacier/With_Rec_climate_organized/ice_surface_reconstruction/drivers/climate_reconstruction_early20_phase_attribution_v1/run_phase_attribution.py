from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import detrend


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=window).mean().to_numpy()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0.0 or np.std(right[valid]) == 0.0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _trend(years: np.ndarray, values: np.ndarray) -> float:
    x = years.astype(float) - float(np.mean(years))
    return float(100.0 * np.polyfit(x, values, 1)[0])


def _phase_metrics(
    years: np.ndarray,
    series: np.ndarray,
    reference: np.ndarray,
    period_name: str,
    period: tuple[int, int],
    windows: list[int],
) -> dict[str, float | bool | int | str]:
    selected = (years >= period[0]) & (years <= period[1])
    x = series[selected]
    y = reference[selected]
    selected_years = years[selected]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    result: dict[str, float | bool | int | str] = {
        "period": period_name,
        "start_year": int(period[0]),
        "end_year": int(period[1]),
        "year_count": int(selected.sum()),
        "raw_correlation": _correlation(x, y),
        "detrended_correlation": _correlation(detrend(x), detrend(y)),
        "first_difference_correlation": _correlation(np.diff(x), np.diff(y)),
        "centered_sign_agreement_fraction": float(np.mean(np.sign(x_centered) == np.sign(y_centered))),
        "series_trend_c_per_century": _trend(selected_years, x),
        "reference_trend_c_per_century": _trend(selected_years, y),
        "trend_sign_same": bool(np.sign(_trend(selected_years, x)) == np.sign(_trend(selected_years, y))),
        "series_std_c": float(np.std(x_centered)),
        "reference_std_c": float(np.std(y_centered)),
        "std_ratio": float(np.std(x_centered) / np.std(y_centered)),
    }
    for window in windows:
        result[f"rolling{window}_correlation"] = _correlation(_rolling(x, window), _rolling(y, window))
    return result


def _lag_scan(
    years: np.ndarray,
    series: np.ndarray,
    reference: np.ndarray,
    period: tuple[int, int],
    window: int,
    lag_min: int,
    lag_max: int,
) -> pd.DataFrame:
    series_smooth = dict(zip(years.tolist(), _rolling(series, window).tolist()))
    reference_smooth = dict(zip(years.tolist(), _rolling(reference, window).tolist()))
    rows = []
    for lag in range(lag_min, lag_max + 1):
        pairs = [
            (series_smooth[year], reference_smooth[year + lag])
            for year in range(period[0], period[1] + 1)
            if period[0] <= year + lag <= period[1]
            and np.isfinite(series_smooth[year])
            and np.isfinite(reference_smooth[year + lag])
        ]
        left, right = zip(*pairs) if pairs else ([], [])
        rows.append({"lag_years": lag, "pair_count": len(pairs), "correlation": _correlation(np.asarray(left), np.asarray(right))})
    return pd.DataFrame(rows)


def _regional(field: xr.DataArray, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(field.values, dtype=float)[:, mask]
    if not np.isfinite(values).all():
        raise ValueError(f"regional-nonfinite:{field.name}")
    normalized = weights[mask] / np.sum(weights[mask])
    return np.sum(values * normalized[None, :], axis=1)


def _spatial_correlation(
    left: xr.DataArray,
    right: xr.DataArray,
    years: np.ndarray,
    period: tuple[int, int],
    window: int,
    mask: np.ndarray,
) -> np.ndarray:
    selected = (years >= period[0]) & (years <= period[1])
    left_values = np.asarray(left.rolling(time=window, center=True, min_periods=window).mean().isel(time=np.where(selected)[0]).values, dtype=float)
    right_values = np.asarray(right.rolling(time=window, center=True, min_periods=window).mean().isel(time=np.where(selected)[0]).values, dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    count = valid.sum(axis=0)
    left_values[:, ~mask] = 0.0
    right_values[:, ~mask] = 0.0
    left_mean = np.nanmean(left_values, axis=0)
    right_mean = np.nanmean(right_values, axis=0)
    numerator = np.nansum((left_values - left_mean) * (right_values - right_mean), axis=0)
    denominator = np.sqrt(np.nansum((left_values - left_mean) ** 2, axis=0) * np.nansum((right_values - right_mean) ** 2, axis=0))
    result = np.full(mask.shape, np.nan, dtype=float)
    usable = mask & (count >= 3) & (denominator > 0)
    result[usable] = numerator[usable] / denominator[usable]
    return result


def _plot_curves(
    years: np.ndarray,
    series: dict[str, np.ndarray],
    period: tuple[int, int],
    title: str,
    output: Path,
) -> None:
    selected = (years >= period[0]) & (years <= period[1])
    base = (years >= 1901) & (years <= 1950)
    colors = {"rec_ref": "#2166ac", "corrected_ensemble": "#15803d", "public_rec_result": "#d95f02", "proxy6_mean": "#7b3294", "proxy9_mean": "#008837"}
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True, constrained_layout=True)
    for name, color in colors.items():
        values = series[name] - np.mean(series[name][base])
        axes[0].plot(years[selected], values[selected], color=color, lw=2.2 if name in {"rec_ref", "corrected_ensemble"} else 1.6, ls="--" if name == "public_rec_result" else "-", label=name)
        axes[1].plot(years[selected], _rolling(values, 15)[selected], color=color, lw=2.4 if name in {"rec_ref", "corrected_ensemble"} else 1.8, ls="--" if name == "public_rec_result" else "-", label=name)
    for axis in axes:
        axis.axhline(0.0, color="0.4", lw=0.8)
        axis.axvspan(1901, 1950, color="#d9d9d9", alpha=0.25)
        axis.grid(alpha=0.22)
        axis.legend(ncol=3, frameon=False, fontsize=9)
    axes[0].set_title(title)
    axes[0].set_ylabel("Annual anomaly (degC)")
    axes[1].set_ylabel("Centered 15-year anomaly (degC)")
    axes[1].set_xlabel("Year")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_members(years: np.ndarray, series: dict[str, np.ndarray], members: list[str], output: Path) -> None:
    selected = (years >= 1881) & (years <= 1970)
    base = (years >= 1901) & (years <= 1950)
    reference = _rolling(series["rec_ref"] - np.mean(series["rec_ref"][base]), 15)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, name in zip(axes.ravel(), members):
        values = _rolling(series[name] - np.mean(series[name][base]), 15)
        axis.plot(years[selected], reference[selected], color="#2166ac", lw=2.1, label="rec_ref")
        axis.plot(years[selected], values[selected], color="#b2182b", lw=2.0, label=name)
        axis.axhline(0.0, color="0.45", lw=0.7)
        axis.axvspan(1901, 1950, color="#d9d9d9", alpha=0.25)
        axis.grid(alpha=0.2)
        axis.set_title(name)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for axis in axes[:, 0]: axis.set_ylabel("15-year anomaly (degC)")
    for axis in axes[-1, :]: axis.set_xlabel("Year")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_lag(lags: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 6.5))
    palette = plt.get_cmap("tab10")
    for index, (name, frame) in enumerate(lags.groupby("series", sort=False)):
        primary = name in {"corrected_ensemble", "proxy6_mean", "proxy9_mean"}
        axis.plot(frame.lag_years, frame.correlation, color=palette(index % 10), lw=2.4 if primary else 1.15, alpha=1.0 if primary else 0.55, label=name)
    axis.axhline(0.0, color="0.35", lw=0.8)
    axis.axvline(0.0, color="0.35", lw=0.8, ls="--")
    axis.set_title("System 3584: 1901-1950 15-year lag correlation")
    axis.set_xlabel("Lag (years; positive means rec_result leads rec_ref)")
    axis.set_ylabel("Correlation")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_maps(lon: np.ndarray, lat: np.ndarray, corrected: np.ndarray, public: np.ndarray, target_lon: float, target_lat: float, period: str, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.3), constrained_layout=True)
    for axis, field, title in ((axes[0], corrected, "Corrected vs rec_ref"), (axes[1], public, "Public rec_result vs rec_ref"), (axes[2], corrected - public, "Corrected minus public")):
        mesh = axis.pcolormesh(lon, lat, field, shading="nearest", cmap="RdBu_r", vmin=-1, vmax=1)
        axis.scatter([target_lon], [target_lat], s=48, facecolors="none", edgecolors="black", linewidths=1.4)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis, shrink=0.82, label="15-year correlation")
    figure.suptitle(f"HMA phase structure: {period}")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    parser.add_argument("--output-dir", type=Path, default=root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_early20_phase_attribution_v1")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    corrected_path = (root / config["corrected_netcdf"]).resolve()
    reference_path = Path(config["reference_netcdf"]).resolve()
    hashes_before = {"corrected_netcdf": _sha256(corrected_path), "reference_netcdf": _sha256(reference_path)}
    periods = {name: tuple(value) for name, value in config["periods"].items()}
    members = config["members"]
    windows = [int(value) for value in config["lowpass_windows"]]
    primary_window = int(config["primary_lowpass_window"])
    lag_min, lag_max = (int(value) for value in config["lag_years"])

    with xr.open_dataset(corrected_path) as corrected, xr.open_dataset(reference_path) as reference:
        years = np.asarray(corrected.time.values, dtype=int)
        lat = np.asarray(corrected.lat.values, dtype=float); lon = np.asarray(corrected.lon.values, dtype=float)
        if not np.array_equal(years, np.asarray(reference.time.values, dtype=int)): raise ValueError("year-axis-conflict")
        if not np.array_equal(lat, np.asarray(reference.lat.values, dtype=float)) or not np.array_equal(lon, np.asarray(reference.lon.values, dtype=float)): raise ValueError("coordinate-axis-conflict")
        lat_index = int(np.where(np.isclose(lat, config["target"]["latitude"]))[0][0]); lon_index = int(np.where(np.isclose(lon, config["target"]["longitude"]))[0][0])
        mask = np.asarray(corrected["hma_mask"].values, dtype=bool)
        weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
        target = {name: np.asarray(corrected[var].isel(lat=lat_index, lon=lon_index).values, dtype=float) for name, var in members.items()}
        regional = {name: _regional(corrected[var], mask, weights) for name, var in members.items()}
        for collection, extract in ((target, lambda field: np.asarray(field.isel(lat=lat_index, lon=lon_index).values, dtype=float)), (regional, lambda field: _regional(field, mask, weights))):
            collection["corrected_ensemble"] = extract(corrected[config["corrected_ensemble_variable"]])
            collection["rec_ref"] = extract(reference[config["reference_variable"]])
            collection["public_rec_result"] = extract(reference[config["public_rec_result_variable"]])
        groups = {"proxy6_mean": [n for n in members if "proxy6" in n], "proxy9_mean": [n for n in members if "proxy9" in n], "AO_mean": [n for n in members if n.startswith("AO-")], "HGAO_mean": [n for n in members if n.startswith("HGAO-")], "OEnKF_mean": [n for n in members if n.startswith("OEnKF-")]}
        for collection in (target, regional):
            for group, names in groups.items(): collection[group] = np.mean(np.stack([collection[n] for n in names]), axis=0)
        spatial = {}
        for name, period in periods.items():
            spatial[(name, "corrected")] = _spatial_correlation(corrected[config["corrected_ensemble_variable"]], reference[config["reference_variable"]], years, period, primary_window, mask)
            spatial[(name, "public")] = _spatial_correlation(reference[config["public_rec_result_variable"]], reference[config["reference_variable"]], years, period, primary_window, mask)

    compared = [*members, "proxy6_mean", "proxy9_mean", "AO_mean", "HGAO_mean", "OEnKF_mean", "corrected_ensemble", "public_rec_result"]
    metric_rows = []; lag_rows = []; lag_summary_rows = []
    for scope, collection in (("target_3584", target), ("hma_regional", regional)):
        for name in compared:
            for period_name, period in periods.items(): metric_rows.append({"scope": scope, "series": name, **_phase_metrics(years, collection[name], collection["rec_ref"], period_name, period, windows)})
            lags = _lag_scan(years, collection[name], collection["rec_ref"], periods["early_20c"], primary_window, lag_min, lag_max)
            lags.insert(0, "series", name); lags.insert(0, "scope", scope); lag_rows.extend(lags.to_dict(orient="records"))
            valid = lags.dropna(subset=["correlation"]); best = valid.loc[valid.correlation.idxmax()]; zero = valid[valid.lag_years.eq(0)].iloc[0]
            lag_summary_rows.append({"scope": scope, "series": name, "zero_lag_correlation": float(zero.correlation), "best_lag_years": int(best.lag_years), "best_lag_correlation": float(best.correlation), "best_lag_pair_count": int(best.pair_count)})
    metrics = pd.DataFrame(metric_rows); lags = pd.DataFrame(lag_rows); lag_summary = pd.DataFrame(lag_summary_rows)
    metrics.to_csv(output / "phase_metrics.csv", index=False); lags.to_csv(output / "lag_correlation_scan.csv", index=False); lag_summary.to_csv(output / "lag_correlation_summary.csv", index=False)
    _plot_curves(years, target, tuple(config["plot_years"]), "System 3584: early twentieth-century phase comparison", output / "target_3584_early20_phase_curves.png")
    _plot_curves(years, regional, tuple(config["plot_years"]), "HMA regional mean: early twentieth-century phase comparison", output / "hma_regional_early20_phase_curves.png")
    _plot_members(years, target, list(members), output / "target_3584_six_member_phase_panels.png")
    _plot_lag(lags[(lags.scope == "target_3584") & lags.series.isin([*members, "proxy6_mean", "proxy9_mean", "corrected_ensemble"])], output / "target_3584_early20_lag_correlation.png")
    spatial_rows = []
    for period_name in periods:
        corrected_map = spatial[(period_name, "corrected")]; public_map = spatial[(period_name, "public")]
        target_corrected = float(corrected_map[lat_index, lon_index]); target_public = float(public_map[lat_index, lon_index]); finite = corrected_map[mask]
        spatial_rows.append({"period": period_name, "corrected_hma_median_correlation": float(np.nanmedian(corrected_map)), "public_hma_median_correlation": float(np.nanmedian(public_map)), "corrected_hma_negative_fraction": float(np.mean(finite < 0)), "target_corrected_correlation": target_corrected, "target_public_correlation": target_public, "target_corrected_hma_percentile": float(100 * (np.sum(finite < target_corrected) + .5 * np.sum(finite == target_corrected)) / len(finite))})
        _plot_maps(lon, lat, corrected_map, public_map, float(config["target"]["longitude"]), float(config["target"]["latitude"]), period_name, output / f"hma_phase_maps_{period_name}.png")
    pd.DataFrame(spatial_rows).to_csv(output / "spatial_phase_summary.csv", index=False)
    hashes_after = {"corrected_netcdf": _sha256(corrected_path), "reference_netcdf": _sha256(reference_path)}; pure = hashes_before == hashes_after
    early = metrics[(metrics.scope == "target_3584") & (metrics.period == "early_20c")].set_index("series"); full = metrics[(metrics.scope == "target_3584") & (metrics.period == "full_20c")].set_index("series"); target_lags = lag_summary[lag_summary.scope == "target_3584"].set_index("series")
    ensemble = early.loc["corrected_ensemble"]; ensemble_lag = target_lags.loc["corrected_ensemble"]
    if float(ensemble[f"rolling{primary_window}_correlation"]) < -0.3 and float(ensemble_lag.best_lag_correlation) < 0.3: classification = "robust_low_frequency_antiphase"
    elif float(ensemble[f"rolling{primary_window}_correlation"]) < 0 and float(ensemble_lag.best_lag_correlation) >= .5: classification = "bounded_timing_offset"
    elif float(ensemble.raw_correlation) < 0 and float(ensemble.detrended_correlation) >= 0: classification = "trend_sign_conflict_after_detrending_not_antiphase"
    else: classification = "mixed_phase_and_trend_conflict"
    summary = {"experiment_id": config["experiment_id"], "status": "complete" if pure else "source-purity-failed", "scientific_scope": "read-only; rec_ref is comparison-only, not truth", "classification": classification, "target_3584_corrected_ensemble": {"early_20c": ensemble.to_dict(), "full_20c": full.loc["corrected_ensemble"].to_dict(), "early20_lag": ensemble_lag.to_dict()}, "target_3584_group_metrics": {name: {"early_20c": early.loc[name].to_dict(), "full_20c": full.loc[name].to_dict(), "early20_lag": target_lags.loc[name].to_dict()} for name in ("proxy6_mean", "proxy9_mean", "AO_mean", "HGAO_mean", "OEnKF_mean")}, "spatial_phase_summary": spatial_rows, "source_purity": {"unchanged": pure, "hashes_before": hashes_before, "hashes_after": hashes_after}, "files": {"metrics": "phase_metrics.csv", "lag_scan": "lag_correlation_scan.csv", "target_curves": "target_3584_early20_phase_curves.png", "member_panels": "target_3584_six_member_phase_panels.png", "lag_plot": "target_3584_early20_lag_correlation.png", "early_map": "hma_phase_maps_early_20c.png"}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    text = f"""# Early twentieth-century phase attribution result

This read-only experiment compares the coordinate-corrected six-member product
with `rec_ref`; `rec_ref` is not treated as truth.

## System 3584, 1901-1950

- Raw correlation: {float(ensemble.raw_correlation):+.3f}.
- Detrended correlation: {float(ensemble.detrended_correlation):+.3f}.
- Centered 15-year correlation: {float(ensemble[f'rolling{primary_window}_correlation']):+.3f}.
- Corrected trend: {float(ensemble.series_trend_c_per_century):+.3f}; rec_ref trend:
  {float(ensemble.reference_trend_c_per_century):+.3f} degC/century.
- Best bounded lag: {int(ensemble_lag.best_lag_years):+d} years with correlation
  {float(ensemble_lag.best_lag_correlation):+.3f}; zero-lag correlation is
  {float(ensemble_lag.zero_lag_correlation):+.3f}.

Classification: `{classification}`.

proxy6 and proxy9 are reported separately in `phase_metrics.csv`; no proxy,
operator, climate, or glacier parameter was changed. Source-purity check:
{pure}.
"""
    (output / "FINAL_RESULT.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True)); return 0 if pure else 2


if __name__ == "__main__": raise SystemExit(main())

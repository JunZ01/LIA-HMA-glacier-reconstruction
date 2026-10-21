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
    return (
        pd.Series(values)
        .rolling(window, center=True, min_periods=window)
        .mean()
        .to_numpy()
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    if np.std(left[valid]) == 0.0 or np.std(right[valid]) == 0.0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _trend(years: np.ndarray, values: np.ndarray) -> float:
    x = years.astype(float)
    x -= np.mean(x)
    return float(100.0 * np.polyfit(x, values, 1)[0])


def _phase_metrics(
    years: np.ndarray,
    values: np.ndarray,
    reference: np.ndarray,
    period_name: str,
    period: tuple[int, int],
    windows: list[int],
) -> dict[str, float | int | str]:
    selected = (years >= period[0]) & (years <= period[1])
    period_years = years[selected]
    left = values[selected]
    right = reference[selected]
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    left_detrended = detrend(left, type="linear")
    right_detrended = detrend(right, type="linear")
    left_diff = np.diff(left)
    right_diff = np.diff(right)
    result: dict[str, float | int | str] = {
        "period": period_name,
        "start_year": int(period[0]),
        "end_year": int(period[1]),
        "year_count": int(selected.sum()),
        "raw_correlation": _correlation(left, right),
        "detrended_correlation": _correlation(left_detrended, right_detrended),
        "first_difference_correlation": _correlation(left_diff, right_diff),
        "centered_sign_agreement_fraction": float(
            np.mean(np.sign(left_centered) == np.sign(right_centered))
        ),
        "series_trend_c_per_century": _trend(period_years, left),
        "reference_trend_c_per_century": _trend(period_years, right),
        "trend_sign_same": bool(
            np.sign(_trend(period_years, left))
            == np.sign(_trend(period_years, right))
        ),
        "series_std_c": float(np.std(left_centered)),
        "reference_std_c": float(np.std(right_centered)),
        "std_ratio": float(np.std(left_centered) / np.std(right_centered)),
    }
    for window in windows:
        result[f"rolling{window}_correlation"] = _correlation(
            _rolling(left, window), _rolling(right, window)
        )
    return result


def _lag_scan(
    years: np.ndarray,
    values: np.ndarray,
    reference: np.ndarray,
    period: tuple[int, int],
    window: int,
    lag_min: int,
    lag_max: int,
) -> pd.DataFrame:
    left_smooth = _rolling(values, window)
    right_smooth = _rolling(reference, window)
    left_by_year = dict(zip(years.tolist(), left_smooth.tolist()))
    right_by_year = dict(zip(years.tolist(), right_smooth.tolist()))
    rows = []
    for lag in range(lag_min, lag_max + 1):
        left_values = []
        right_values = []
        for year in range(period[0], period[1] + 1):
            reference_year = year + lag
            if reference_year < period[0] or reference_year > period[1]:
                continue
            left_value = left_by_year.get(year, np.nan)
            right_value = right_by_year.get(reference_year, np.nan)
            if np.isfinite(left_value) and np.isfinite(right_value):
                left_values.append(left_value)
                right_values.append(right_value)
        rows.append(
            {
                "lag_years": lag,
                "pair_count": len(left_values),
                "correlation": _correlation(
                    np.asarray(left_values, dtype=float),
                    np.asarray(right_values, dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def _weighted_regional(
    field: xr.DataArray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(field.values, dtype=float)
    selected = values[:, mask]
    if not np.isfinite(selected).all():
        raise ValueError(f"regional-nonfinite:{field.name}")
    normalized = weights[mask] / float(np.sum(weights[mask]))
    return np.sum(selected * normalized[None, :], axis=1)


def _spatial_correlation(
    left: xr.DataArray,
    right: xr.DataArray,
    years: np.ndarray,
    period: tuple[int, int],
    window: int,
    mask: np.ndarray,
) -> np.ndarray:
    left_smooth = left.rolling(time=window, center=True, min_periods=window).mean()
    right_smooth = right.rolling(time=window, center=True, min_periods=window).mean()
    selected = (years >= period[0]) & (years <= period[1])
    left_values = np.asarray(left_smooth.isel(time=np.where(selected)[0]).values, dtype=float)
    right_values = np.asarray(right_smooth.isel(time=np.where(selected)[0]).values, dtype=float)
    valid_time = np.isfinite(left_values) & np.isfinite(right_values)
    count = np.sum(valid_time, axis=0)
    left_mean = np.nanmean(left_values, axis=0)
    right_mean = np.nanmean(right_values, axis=0)
    left_anomaly = left_values - left_mean
    right_anomaly = right_values - right_mean
    numerator = np.nansum(left_anomaly * right_anomaly, axis=0)
    denominator = np.sqrt(
        np.nansum(left_anomaly**2, axis=0) * np.nansum(right_anomaly**2, axis=0)
    )
    correlation = np.full(mask.shape, np.nan, dtype=float)
    usable = mask & (count >= 3) & (denominator > 0.0)
    correlation[usable] = numerator[usable] / denominator[usable]
    return correlation


def _percentile(values: np.ndarray, target: float) -> float:
    finite = values[np.isfinite(values)]
    return float(100.0 * (np.sum(finite < target) + 0.5 * np.sum(finite == target)) / len(finite))


def _plot_target_curves(
    years: np.ndarray,
    series: dict[str, np.ndarray],
    plot_period: tuple[int, int],
    output: Path,
) -> None:
    selected = (years >= plot_period[0]) & (years <= plot_period[1])
    plot_years = years[selected]
    baseline = (years >= 1801) & (years <= 1900)
    colors = {
        "rec_ref": "#2166ac",
        "corrected_ensemble": "#15803d",
        "public_rec_result": "#d95f02",
        "proxy6_mean": "#7b3294",
        "proxy9_mean": "#008837",
    }
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True, constrained_layout=True)
    for name in colors:
        centered = series[name] - float(np.mean(series[name][baseline]))
        axes[0].plot(
            plot_years,
            centered[selected],
            color=colors[name],
            lw=2.2 if name in {"rec_ref", "corrected_ensemble"} else 1.6,
            ls="--" if name == "public_rec_result" else "-",
            label=name,
        )
        axes[1].plot(
            plot_years,
            _rolling(centered, 15)[selected],
            color=colors[name],
            lw=2.4 if name in {"rec_ref", "corrected_ensemble"} else 1.8,
            ls="--" if name == "public_rec_result" else "-",
            label=name,
        )
    for axis in axes:
        axis.axhline(0.0, color="0.4", lw=0.8)
        axis.axvspan(1801, 1850, color="#d9d9d9", alpha=0.25)
        axis.grid(alpha=0.22)
        axis.legend(ncol=3, frameon=False, fontsize=9)
    axes[0].set_ylabel("Annual anomaly (degC)")
    axes[0].set_title("System 3584: early nineteenth-century reconstruction comparison")
    axes[1].set_ylabel("Centered 15-year mean (degC)")
    axes[1].set_xlabel("Year")
    axes[1].set_xlim(plot_period)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_member_panels(
    years: np.ndarray,
    series: dict[str, np.ndarray],
    member_names: list[str],
    output: Path,
) -> None:
    selected = (years >= 1781) & (years <= 1900)
    baseline = (years >= 1801) & (years <= 1900)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, name in zip(axes.ravel(), member_names):
        member = series[name] - float(np.mean(series[name][baseline]))
        reference = series["rec_ref"] - float(np.mean(series["rec_ref"][baseline]))
        axis.plot(years[selected], _rolling(reference, 15)[selected], color="#2166ac", lw=2.1, label="rec_ref")
        axis.plot(years[selected], _rolling(member, 15)[selected], color="#b2182b", lw=2.0, label=name)
        axis.axhline(0.0, color="0.45", lw=0.7)
        axis.axvspan(1801, 1850, color="#d9d9d9", alpha=0.25)
        axis.grid(alpha=0.2)
        axis.set_title(name)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for axis in axes[:, 0]:
        axis.set_ylabel("15-year anomaly (degC)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Year")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_lag_curves(lag_table: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 6.5))
    colors = plt.get_cmap("tab10")
    for index, (name, frame) in enumerate(lag_table.groupby("series", sort=False)):
        axis.plot(
            frame["lag_years"],
            frame["correlation"],
            lw=2.3 if name in {"corrected_ensemble", "proxy6_mean", "proxy9_mean"} else 1.2,
            alpha=1.0 if name in {"corrected_ensemble", "proxy6_mean", "proxy9_mean"} else 0.55,
            color=colors(index % 10),
            label=name,
        )
    axis.axhline(0.0, color="0.35", lw=0.8)
    axis.axvline(0.0, color="0.35", lw=0.8, ls="--")
    axis.set_xlabel("Lag (years; positive means rec_result leads rec_ref)")
    axis.set_ylabel("15-year low-pass correlation")
    axis.set_title("System 3584: 1801-1900 lag correlation")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_phase_maps(
    lon: np.ndarray,
    lat: np.ndarray,
    corrected: np.ndarray,
    public: np.ndarray,
    target_lon: float,
    target_lat: float,
    period_name: str,
    output: Path,
) -> None:
    difference = corrected - public
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.3), constrained_layout=True)
    for axis, field, title, limits in (
        (axes[0], corrected, "Corrected vs rec_ref", (-1.0, 1.0)),
        (axes[1], public, "Public rec_result vs rec_ref", (-1.0, 1.0)),
        (axes[2], difference, "Corrected minus public correlation", (-1.0, 1.0)),
    ):
        mesh = axis.pcolormesh(
            lon,
            lat,
            field,
            shading="nearest",
            cmap="RdBu_r",
            vmin=limits[0],
            vmax=limits[1],
        )
        axis.scatter([target_lon], [target_lat], s=48, facecolors="none", edgecolors="black", linewidths=1.4)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis, shrink=0.82, label="15-year correlation")
    figure.suptitle(f"HMA phase structure: {period_name}")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    config_default = Path(__file__).with_name("experiment_config.json")
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_early19_phase_attribution_v1"
    )
    parser.add_argument("--config", type=Path, default=config_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    corrected_path = (root / config["corrected_netcdf"]).resolve()
    reference_path = Path(config["reference_netcdf"]).resolve()
    sources = {"corrected_netcdf": corrected_path, "reference_netcdf": reference_path}
    hashes_before = {name: _sha256(path) for name, path in sources.items()}

    target_lat = float(config["target"]["latitude"])
    target_lon = float(config["target"]["longitude"])
    periods = {name: tuple(values) for name, values in config["periods"].items()}
    windows = [int(value) for value in config["lowpass_windows"]]
    primary_window = int(config["primary_lowpass_window"])
    lag_min, lag_max = (int(value) for value in config["lag_years"])
    member_variables = config["members"]

    with xr.open_dataset(corrected_path) as corrected, xr.open_dataset(reference_path) as reference:
        years = np.asarray(corrected.time.values, dtype=int)
        lat = np.asarray(corrected.lat.values, dtype=float)
        lon = np.asarray(corrected.lon.values, dtype=float)
        if not np.array_equal(years, np.asarray(reference.time.values, dtype=int)):
            raise ValueError("source-year-conflict")
        if not np.array_equal(lat, np.asarray(reference.lat.values, dtype=float)):
            raise ValueError("source-latitude-conflict")
        if not np.array_equal(lon, np.asarray(reference.lon.values, dtype=float)):
            raise ValueError("source-longitude-conflict")
        lat_index = int(np.where(np.isclose(lat, target_lat))[0][0])
        lon_index = int(np.where(np.isclose(lon, target_lon))[0][0])
        hma_mask = np.asarray(corrected["hma_mask"].values, dtype=bool)
        weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))

        target_series: dict[str, np.ndarray] = {
            name: np.asarray(
                corrected[variable].isel(lat=lat_index, lon=lon_index).values,
                dtype=float,
            )
            for name, variable in member_variables.items()
        }
        regional_series: dict[str, np.ndarray] = {
            name: _weighted_regional(corrected[variable], hma_mask, weights)
            for name, variable in member_variables.items()
        }
        for scope_series, selector in (
            (target_series, lambda field: np.asarray(field.isel(lat=lat_index, lon=lon_index).values, dtype=float)),
            (regional_series, lambda field: _weighted_regional(field, hma_mask, weights)),
        ):
            scope_series["corrected_ensemble"] = selector(
                corrected[config["corrected_ensemble_variable"]]
            )
            scope_series["rec_ref"] = selector(reference[config["reference_variable"]])
            scope_series["public_rec_result"] = selector(
                reference[config["public_rec_result_variable"]]
            )

        groups = {
            "proxy6_mean": [name for name in member_variables if "proxy6" in name],
            "proxy9_mean": [name for name in member_variables if "proxy9" in name],
            "AO_mean": [name for name in member_variables if name.startswith("AO-")],
            "HGAO_mean": [name for name in member_variables if name.startswith("HGAO-")],
            "OEnKF_mean": [name for name in member_variables if name.startswith("OEnKF-")],
        }
        for scope_series in (target_series, regional_series):
            for group, names in groups.items():
                scope_series[group] = np.mean(np.stack([scope_series[name] for name in names]), axis=0)

        spatial_maps = {}
        for period_name, period in periods.items():
            spatial_maps[(period_name, "corrected")] = _spatial_correlation(
                corrected[config["corrected_ensemble_variable"]],
                reference[config["reference_variable"]],
                years,
                period,
                primary_window,
                hma_mask,
            )
            spatial_maps[(period_name, "public")] = _spatial_correlation(
                reference[config["public_rec_result_variable"]],
                reference[config["reference_variable"]],
                years,
                period,
                primary_window,
                hma_mask,
            )

    metric_rows = []
    lag_rows = []
    lag_summary_rows = []
    compared_names = [
        *member_variables,
        "proxy6_mean",
        "proxy9_mean",
        "AO_mean",
        "HGAO_mean",
        "OEnKF_mean",
        "corrected_ensemble",
        "public_rec_result",
    ]
    for scope, scope_series in (("target_3584", target_series), ("hma_regional", regional_series)):
        for name in compared_names:
            for period_name, period in periods.items():
                metric_rows.append(
                    {
                        "scope": scope,
                        "series": name,
                        **_phase_metrics(
                            years,
                            scope_series[name],
                            scope_series["rec_ref"],
                            period_name,
                            period,
                            windows,
                        ),
                    }
                )
            lag_table = _lag_scan(
                years,
                scope_series[name],
                scope_series["rec_ref"],
                periods["full_19c"],
                primary_window,
                lag_min,
                lag_max,
            )
            lag_table.insert(0, "series", name)
            lag_table.insert(0, "scope", scope)
            lag_rows.extend(lag_table.to_dict(orient="records"))
            finite = lag_table.dropna(subset=["correlation"])
            best = finite.loc[finite["correlation"].idxmax()]
            zero = finite[finite["lag_years"].eq(0)].iloc[0]
            lag_summary_rows.append(
                {
                    "scope": scope,
                    "series": name,
                    "zero_lag_correlation": float(zero["correlation"]),
                    "best_lag_years": int(best["lag_years"]),
                    "best_lag_correlation": float(best["correlation"]),
                    "best_lag_pair_count": int(best["pair_count"]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    lag_table = pd.DataFrame(lag_rows)
    lag_summary = pd.DataFrame(lag_summary_rows)
    metrics.to_csv(output / "phase_metrics.csv", index=False)
    lag_table.to_csv(output / "lag_correlation_scan.csv", index=False)
    lag_summary.to_csv(output / "lag_correlation_summary.csv", index=False)

    _plot_target_curves(
        years,
        target_series,
        tuple(config["plot_years"]),
        output / "target_3584_early19_phase_curves.png",
    )
    _plot_target_curves(
        years,
        regional_series,
        tuple(config["plot_years"]),
        output / "hma_regional_early19_phase_curves.png",
    )
    _plot_member_panels(
        years,
        target_series,
        list(member_variables),
        output / "target_3584_six_member_phase_panels.png",
    )
    _plot_lag_curves(
        lag_table[
            lag_table["scope"].eq("target_3584")
            & lag_table["series"].isin([*member_variables, "proxy6_mean", "proxy9_mean", "corrected_ensemble"])
        ],
        output / "target_3584_lag_correlation.png",
    )

    spatial_rows = []
    for period_name in periods:
        corrected_map = spatial_maps[(period_name, "corrected")]
        public_map = spatial_maps[(period_name, "public")]
        target_corrected = float(corrected_map[lat_index, lon_index])
        target_public = float(public_map[lat_index, lon_index])
        spatial_rows.append(
            {
                "period": period_name,
                "corrected_hma_median_correlation": float(np.nanmedian(corrected_map)),
                "public_hma_median_correlation": float(np.nanmedian(public_map)),
                "corrected_hma_negative_fraction": float(
                    np.mean(corrected_map[hma_mask] < 0.0)
                ),
                "public_hma_negative_fraction": float(
                    np.mean(public_map[hma_mask] < 0.0)
                ),
                "target_corrected_correlation": target_corrected,
                "target_public_correlation": target_public,
                "target_corrected_hma_percentile": _percentile(
                    corrected_map[hma_mask], target_corrected
                ),
                "target_public_hma_percentile": _percentile(
                    public_map[hma_mask], target_public
                ),
            }
        )
        _plot_phase_maps(
            lon,
            lat,
            corrected_map,
            public_map,
            target_lon,
            target_lat,
            period_name,
            output / f"hma_phase_maps_{period_name}.png",
        )
    spatial_summary = pd.DataFrame(spatial_rows)
    spatial_summary.to_csv(output / "spatial_phase_summary.csv", index=False)

    hashes_after = {name: _sha256(path) for name, path in sources.items()}
    sources_unchanged = hashes_before == hashes_after
    early_metrics = metrics[
        metrics["scope"].eq("target_3584") & metrics["period"].eq("early_19c")
    ].set_index("series")
    full_metrics = metrics[
        metrics["scope"].eq("target_3584") & metrics["period"].eq("full_19c")
    ].set_index("series")
    target_lags = lag_summary[lag_summary["scope"].eq("target_3584")].set_index("series")
    ensemble_early = early_metrics.loc["corrected_ensemble"]
    ensemble_full = full_metrics.loc["corrected_ensemble"]
    best_lag = target_lags.loc["corrected_ensemble"]
    if (
        float(ensemble_early[f"rolling{primary_window}_correlation"]) < -0.3
        and float(best_lag["best_lag_correlation"]) < 0.3
    ):
        classification = "robust_low_frequency_antiphase"
    elif (
        float(ensemble_early[f"rolling{primary_window}_correlation"]) < 0.0
        and float(best_lag["best_lag_correlation"]) >= 0.5
    ):
        classification = "timing_displacement_with_recoverable_positive_lag"
    elif (
        float(ensemble_early["raw_correlation"]) < 0.0
        and float(ensemble_early["detrended_correlation"]) >= 0.0
    ):
        classification = "trend_sign_conflict_not_detrended_antiphase"
    else:
        classification = "mixed_phase_and_trend_conflict"

    summary = {
        "experiment_id": config["experiment_id"],
        "status": "complete" if sources_unchanged else "source-purity-failed",
        "scientific_scope": "read-only reconstruction comparison; rec_ref is reference-only, not truth",
        "classification": classification,
        "target_3584_corrected_ensemble": {
            "early_19c": ensemble_early.to_dict(),
            "full_19c": ensemble_full.to_dict(),
            "lag_scan_full_19c": best_lag.to_dict(),
        },
        "target_3584_group_metrics": {
            group: {
                "early_19c": early_metrics.loc[group].to_dict(),
                "full_19c": full_metrics.loc[group].to_dict(),
                "lag_scan_full_19c": target_lags.loc[group].to_dict(),
            }
            for group in ("proxy6_mean", "proxy9_mean", "AO_mean", "HGAO_mean", "OEnKF_mean")
        },
        "spatial_phase_summary": spatial_rows,
        "source_purity": {
            "unchanged": sources_unchanged,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
        },
        "files": {
            "metrics": "phase_metrics.csv",
            "lag_scan": "lag_correlation_scan.csv",
            "lag_summary": "lag_correlation_summary.csv",
            "spatial_summary": "spatial_phase_summary.csv",
            "target_curves": "target_3584_early19_phase_curves.png",
            "regional_curves": "hma_regional_early19_phase_curves.png",
            "member_panels": "target_3584_six_member_phase_panels.png",
            "lag_plot": "target_3584_lag_correlation.png",
            "early_map": "hma_phase_maps_early_19c.png",
            "full_map": "hma_phase_maps_full_19c.png",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    final_text = f"""# Early nineteenth-century phase attribution result

## Scope

This is a read-only comparison between the coordinate-corrected six-member
`rec_result` and `rec_ref`. `rec_ref` is a comparison reconstruction, not
truth. No instrumental selection, OGGM run, or parameter tuning was used.

## System 3584

- 1801-1850 raw correlation: {float(ensemble_early['raw_correlation']):+.3f}.
- 1801-1850 detrended correlation: {float(ensemble_early['detrended_correlation']):+.3f}.
- 1801-1850 centered 15-year correlation:
  {float(ensemble_early[f'rolling{primary_window}_correlation']):+.3f}.
- 1801-1850 trend: corrected ensemble
  {float(ensemble_early['series_trend_c_per_century']):+.3f} versus rec_ref
  {float(ensemble_early['reference_trend_c_per_century']):+.3f} degC/century.
- 1801-1900 best bounded lag: {int(best_lag['best_lag_years']):+d} years,
  correlation {float(best_lag['best_lag_correlation']):+.3f}, versus zero-lag
  {float(best_lag['zero_lag_correlation']):+.3f}.

Diagnostic classification: `{classification}`.

## Attribution

- proxy6 mean early 15-year correlation:
  {float(early_metrics.loc['proxy6_mean', f'rolling{primary_window}_correlation']):+.3f}.
- proxy9 mean early 15-year correlation:
  {float(early_metrics.loc['proxy9_mean', f'rolling{primary_window}_correlation']):+.3f}.
- The member, group, lag, and HMA spatial tables should be used together; a
  local phase mismatch is not automatically an HMA-wide reconstruction error.

No source file was modified. Source-purity check passed: {sources_unchanged}.
"""
    (output / "FINAL_RESULT.md").write_text(final_text, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if sources_unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())

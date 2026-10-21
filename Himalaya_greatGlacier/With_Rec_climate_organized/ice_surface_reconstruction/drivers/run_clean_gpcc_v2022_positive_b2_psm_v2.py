"""Build a clean GPCC v2022 branch and rerun the positive-b2 event PSM audit."""

from __future__ import annotations

import calendar
import importlib.util
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVENT_YEARS = np.arange(1905, 1921)
CALIBRATION_YEARS = np.arange(1941, 2001)
LOCKED_PERIOD = (1921, 1939)


def load_module(filename: str, name: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-import-failed:{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def monthly_slice(source_start_year: int, first_year: int, last_year: int) -> slice:
    """Return prior-December through final-December without touching locked years."""
    start = (first_year - 1 - source_start_year) * 12 + 11
    stop = (last_year - source_start_year + 1) * 12
    return slice(start, stop)


def monthly_labels(first_year: int, last_year: int) -> list[tuple[int, int]]:
    labels = [(first_year - 1, 12)]
    labels.extend((year, month) for year in range(first_year, last_year + 1) for month in range(1, 13))
    return labels


def read_netcdf3_points(
    path: Path,
    variable_names: list[str],
    source_start_year: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    periods = {
        "event": (monthly_slice(source_start_year, 1905, 1920), 193),
        "calibration": (monthly_slice(source_start_year, 1941, 2000), 721),
    }
    output = {
        period: {name: np.full((length, len(latitudes)), np.nan) for name in variable_names}
        for period, (_, length) in periods.items()
    }
    with netcdf_file(path, "r", mmap=True) as handle:
        source_lat = np.array(handle.variables["lat"][:], dtype=float, copy=True)
        source_lon = np.array(handle.variables["lon"][:], dtype=float, copy=True)
        lat_indices = [int(np.argmin(np.abs(source_lat - value))) for value in latitudes]
        lon_indices = [int(np.argmin(np.abs(source_lon - value))) for value in longitudes]
        for period, (selection, expected_length) in periods.items():
            for column, (lat_index, lon_index) in enumerate(zip(lat_indices, lon_indices)):
                for name in variable_names:
                    variable = handle.variables[name]
                    values = np.array(variable[selection, lat_index, lon_index], dtype=float, copy=True)
                    if len(values) != expected_length:
                        raise RuntimeError(f"monthly-slice-length-failed:{path.name}:{period}:{name}")
                    fill = float(variable._attributes.get("_FillValue", -99999.99))
                    values[(values == fill) | (values <= -9000) | (values >= 1e20)] = np.nan
                    output[period][name][:, column] = values
    return output


def read_berkeley_points(path: Path, latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, np.ndarray]:
    periods = {
        "event": (monthly_slice(1850, 1905, 1920), 193),
        "calibration": (monthly_slice(1850, 1941, 2000), 721),
    }
    output = {period: np.full((length, len(latitudes)), np.nan) for period, (_, length) in periods.items()}
    with h5py.File(path, "r") as handle:
        source_lat = np.asarray(handle["latitude"], dtype=float)
        source_lon = np.asarray(handle["longitude"], dtype=float)
        lat_indices = [int(np.argmin(np.abs(source_lat - value))) for value in latitudes]
        lon_indices = [int(np.argmin(np.abs(source_lon - value))) for value in longitudes]
        for period, (selection, expected_length) in periods.items():
            for column, (lat_index, lon_index) in enumerate(zip(lat_indices, lon_indices)):
                values = np.asarray(handle["temperature"][selection, lat_index, lon_index], dtype=float)
                if len(values) != expected_length:
                    raise RuntimeError(f"berkeley-monthly-slice-length-failed:{period}")
                values[(values <= -9000) | (values >= 1e20)] = np.nan
                output[period][:, column] = values
    return output


def aggregate_temperature(
    values: np.ndarray,
    labels: list[tuple[int, int]],
    years: np.ndarray,
    months: tuple[int, ...],
) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(labels)}
    result = np.full((len(years), values.shape[1]), np.nan)
    cross_year = months[0] > months[-1]
    for row, year in enumerate(years):
        requested = []
        for month in months:
            source_year = int(year - 1) if cross_year and month >= months[0] else int(year)
            requested.append(lookup[(source_year, month)])
        selected = values[requested]
        complete = np.all(np.isfinite(selected), axis=0)
        result[row, complete] = np.mean(selected[:, complete], axis=0)
    return result


def aggregate_precipitation(
    monthly_totals: np.ndarray,
    labels: list[tuple[int, int]],
    years: np.ndarray,
    months: tuple[int, ...],
) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(labels)}
    result = np.full((len(years), monthly_totals.shape[1]), np.nan)
    cross_year = months[0] > months[-1]
    for row, year in enumerate(years):
        requested = []
        days = []
        for month in months:
            source_year = int(year - 1) if cross_year and month >= months[0] else int(year)
            requested.append(lookup[(source_year, month)])
            days.append(calendar.monthrange(source_year, month)[1])
        selected = monthly_totals[requested]
        complete = np.all(np.isfinite(selected), axis=0)
        result[row, complete] = np.sum(selected[:, complete], axis=0) / float(np.sum(days))
    return result


def reference_anomalies(
    event: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    reference = (CALIBRATION_YEARS >= 1961) & (CALIBRATION_YEARS <= 1990)
    event_output = {}
    calibration_output = {}
    for season in event:
        mean = np.nanmean(calibration[season][reference], axis=0, keepdims=True)
        event_output[season] = event[season] - mean
        calibration_output[season] = calibration[season] - mean
    return event_output, calibration_output


def main() -> int:
    base = load_module("audit_positive_b2_event_seasonal_psm_v1.py", "positive_b2_seasonal_base")
    gpcc_base = load_module("audit_gpcc_monthly_to_annual_processing_v1.py", "gpcc_processing_base")
    base.EVENT_YEARS = EVENT_YEARS
    base.CALIBRATION_YEARS = CALIBRATION_YEARS
    root = base.project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_clean_gpcc_v2022_positive_b2_psm_v2"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    event_table_path = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_positive_b2_event_site_instrumental_consistency_v1/"
        "site_instrumental_event_consistency.csv"
    )
    proxy6 = base.archive_proxy_path()
    sources = {
        "event_table": event_table_path,
        "proxy6": proxy6,
        "raw_gpcc_v2022": base.GPCC_PRECIPITATION,
        "berkeley_temperature": base.BERKELEY_TEMPERATURE,
        "cru_temperature": base.CRU_TEMPERATURE,
        "stored_climate_comparison_only": gpcc_base.STORED_CLIMATE,
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: base.source_stat(path) for name, path in sources.items()}

    table = pd.read_csv(event_table_path).sort_values("proxy_index").reset_index(drop=True)
    if len(table) != 60 or not (table["coefficient_b2"] >= 0).all():
        raise RuntimeError("positive-b2-selection-contract-failed")
    latitudes = table["latitude"].to_numpy(dtype=float)
    longitudes = table["longitude"].to_numpy(dtype=float)

    proxy_event = np.full((len(EVENT_YEARS), len(table)), np.nan)
    proxy_calibration = np.full((len(CALIBRATION_YEARS), len(table)), np.nan)
    with h5py.File(proxy6, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(table["proxy_index"].to_numpy(dtype=int)):
            proxy = handle[refs[proxy_index - 1]]
            event_slice = slice(1904, 1920)
            calibration_slice = slice(1940, 2000)
            if not np.array_equal(np.asarray(proxy["age_yng"][0, event_slice], dtype=float), EVENT_YEARS):
                raise RuntimeError(f"proxy-event-year-contract-failed:{proxy_index}")
            if not np.array_equal(np.asarray(proxy["age_yng"][0, calibration_slice], dtype=float), CALIBRATION_YEARS):
                raise RuntimeError(f"proxy-calibration-year-contract-failed:{proxy_index}")
            proxy_event[:, column] = np.asarray(proxy["data"][event_slice, 0], dtype=float)
            proxy_calibration[:, column] = np.asarray(proxy["data"][calibration_slice, 0], dtype=float)
    proxy_scale = np.nanstd(proxy_calibration, axis=0)
    actual_event_z = base.period_change(EVENT_YEARS, proxy_event) / proxy_scale
    archived_event_z = table["event_drop_z"].to_numpy(dtype=float)
    event_scale_difference = actual_event_z - archived_event_z

    gpcc = read_netcdf3_points(
        base.GPCC_PRECIPITATION,
        ["precip", "numgauge", "infilled_numgauges", "interpolation_error", "interpolation_error_infilled"],
        1891,
        latitudes,
        longitudes,
    )
    cru = read_netcdf3_points(base.CRU_TEMPERATURE, ["tmp", "stn"], 1901, latitudes, longitudes)
    berkeley = read_berkeley_points(base.BERKELEY_TEMPERATURE, latitudes, longitudes)
    event_labels = monthly_labels(1905, 1920)
    calibration_labels = monthly_labels(1941, 2000)

    event_precipitation_raw = {}
    calibration_precipitation_raw = {}
    event_bea_temperature_raw = {}
    calibration_bea_temperature_raw = {}
    event_cru_temperature_raw = {}
    calibration_cru_temperature_raw = {}
    for season, months in base.SEASONS.items():
        event_precipitation_raw[season] = aggregate_precipitation(gpcc["event"]["precip"], event_labels, EVENT_YEARS, months)
        calibration_precipitation_raw[season] = aggregate_precipitation(gpcc["calibration"]["precip"], calibration_labels, CALIBRATION_YEARS, months)
        event_bea_temperature_raw[season] = aggregate_temperature(berkeley["event"], event_labels, EVENT_YEARS, months)
        calibration_bea_temperature_raw[season] = aggregate_temperature(berkeley["calibration"], calibration_labels, CALIBRATION_YEARS, months)
        event_cru_temperature_raw[season] = aggregate_temperature(cru["event"]["tmp"], event_labels, EVENT_YEARS, months)
        calibration_cru_temperature_raw[season] = aggregate_temperature(cru["calibration"]["tmp"], calibration_labels, CALIBRATION_YEARS, months)

    event_precipitation, calibration_precipitation = reference_anomalies(event_precipitation_raw, calibration_precipitation_raw)
    event_bea_temperature, calibration_bea_temperature = reference_anomalies(event_bea_temperature_raw, calibration_bea_temperature_raw)
    event_cru_temperature, calibration_cru_temperature = reference_anomalies(event_cru_temperature_raw, calibration_cru_temperature_raw)

    branches = {}
    for label, calibration_temperature, event_temperature in (
        ("BEA_clean_GPCC", calibration_bea_temperature, event_bea_temperature),
        ("CRU_clean_GPCC", calibration_cru_temperature, event_cru_temperature),
    ):
        branches[label] = base.branch_analysis(
            label,
            proxy_calibration,
            proxy_scale,
            calibration_temperature,
            calibration_precipitation,
            event_temperature,
            event_precipitation,
        )

    cv_table = pd.concat([result[0] for result in branches.values()], ignore_index=True)
    proxy_results = pd.concat([result[1] for result in branches.values()], ignore_index=True)
    proxy_results["proxy_index"] = table.iloc[proxy_results["proxy_column"].to_numpy(dtype=int)]["proxy_index"].to_numpy()

    predictions = {}
    for label, result in branches.items():
        predictions[label + "_annual_refit"] = base.period_change(EVENT_YEARS, result[2]) / proxy_scale
        predictions[label + "_selected_season"] = base.period_change(EVENT_YEARS, result[3]) / proxy_scale
    predictions["archived_production_PSM"] = table["production_psm_predicted_event_change_z"].to_numpy(dtype=float)
    predictions["archived_CRU_PSM"] = table["cru_psm_predicted_event_change_z"].to_numpy(dtype=float)
    rng = np.random.default_rng(base.RNG_SEED)
    metrics = pd.DataFrame([base.metric_row(label, actual_event_z, values, rng) for label, values in predictions.items()])

    clean_gpcc_event_change = base.period_change(EVENT_YEARS, event_precipitation["annual"])
    clean_vs_products = pd.DataFrame(
        [
            gpcc_base.metrics(
                "clean_GPCC_vs_stored_GPCC",
                clean_gpcc_event_change,
                table["precipitation_gpcc_event_change"].to_numpy(dtype=float),
            ),
            gpcc_base.metrics(
                "clean_GPCC_vs_CRU",
                clean_gpcc_event_change,
                table["precipitation_cru_event_change"].to_numpy(dtype=float),
            ),
        ]
    )

    event_month_mask = np.asarray([1905 <= year <= 1918 for year, _ in event_labels])
    quality = pd.DataFrame(
        {
            "proxy_index": table["proxy_index"],
            "latitude": latitudes,
            "longitude": longitudes,
            "direct_gauge_month_fraction_1905_1918": np.nanmean(gpcc["event"]["numgauge"][event_month_mask] > 0, axis=0),
            "mean_direct_gauge_count_1905_1918": np.nanmean(gpcc["event"]["numgauge"][event_month_mask], axis=0),
            "infilled_gauge_month_fraction_1905_1918": np.nanmean(gpcc["event"]["infilled_numgauges"][event_month_mask] > 0, axis=0),
            "mean_infilled_gauge_count_1905_1918": np.nanmean(gpcc["event"]["infilled_numgauges"][event_month_mask], axis=0),
            "mean_interpolation_error_mm_month_1905_1918": np.nanmean(gpcc["event"]["interpolation_error"][event_month_mask], axis=0),
            "mean_interpolation_error_infilled_mm_month_1905_1918": np.nanmean(gpcc["event"]["interpolation_error_infilled"][event_month_mask], axis=0),
            "cru_temperature_station_month_fraction_1905_1918": np.nanmean(cru["event"]["stn"][event_month_mask] > 0, axis=0),
            "clean_gpcc_event_change_mm_day": clean_gpcc_event_change,
            "stored_gpcc_event_change_mm_day": table["precipitation_gpcc_event_change"],
            "cru_event_change_mm_day": table["precipitation_cru_event_change"],
            "actual_proxy_event_change_z_1941_2000_scale": actual_event_z,
        }
    )
    selected_prediction = predictions["BEA_clean_GPCC_selected_season"]
    quality["absolute_selected_prediction_residual_z"] = np.abs(actual_event_z - selected_prediction)

    baseline_mask = (EVENT_YEARS >= 1905) & (EVENT_YEARS <= 1914)
    curves = pd.DataFrame(
        {
            "year": EVENT_YEARS,
            "actual_proxy_network_z_from_baseline": np.nanmean(
                (proxy_event - np.nanmean(proxy_event[baseline_mask], axis=0)[None, :]) / proxy_scale[None, :],
                axis=1,
            ),
        }
    )
    for label, result in branches.items():
        for suffix, values in (("annual_refit", result[2]), ("selected_season", result[3])):
            centered = (values - np.nanmean(values[baseline_mask], axis=0)[None, :]) / proxy_scale[None, :]
            curves[label + "_" + suffix + "_network_z_from_baseline"] = np.nanmean(centered, axis=1)

    output.mkdir(parents=True)
    cv_table.to_csv(output / "clean_gpcc_seasonal_psm_blocked_cv.csv", index=False)
    proxy_results.to_csv(output / "clean_gpcc_seasonal_psm_proxy_results.csv", index=False)
    metrics.to_csv(output / "clean_gpcc_event_prediction_metrics.csv", index=False)
    clean_vs_products.to_csv(output / "clean_gpcc_instrumental_product_comparison.csv", index=False)
    quality.to_csv(output / "clean_gpcc_quality_at_proxy_sites.csv", index=False)
    curves.to_csv(output / "clean_gpcc_event_network_curves.csv", index=False)

    metric_index = metrics.set_index("prediction")
    selected_counts = {
        branch: {str(key): int(value) for key, value in group["selected_season"].value_counts().to_dict().items()}
        for branch, group in proxy_results.groupby("branch")
    }
    high_support = quality["direct_gauge_month_fraction_1905_1918"] >= 0.5
    low_support = ~high_support
    summary = {
        "experiment_id": "climate_reconstruction_clean_gpcc_v2022_positive_b2_psm_v2",
        "locked_final_period": list(LOCKED_PERIOD),
        "locked_final_period_accessed": False,
        "read_periods": [[1904, 1920], [1940, 2000]],
        "calibration_years": [1941, 2000],
        "reference_period": [1961, 1990],
        "proxy_event_scale_comparison": {
            "new_scale_period": [1941, 2000],
            "archived_scale_period": [1940, 2000],
            "maximum_absolute_z_difference": float(np.nanmax(np.abs(event_scale_difference))),
            "median_absolute_z_difference": float(np.nanmedian(np.abs(event_scale_difference))),
        },
        "clean_precipitation_rule": "Seasonal precipitation total divided by the exact number of Gregorian calendar days; annual is Jan-Dec.",
        "event_prediction_metrics": metrics.to_dict(orient="records"),
        "instrumental_product_comparison": clean_vs_products.to_dict(orient="records"),
        "selected_season_counts": selected_counts,
        "quality_summary": {
            "median_direct_gauge_month_fraction": float(quality["direct_gauge_month_fraction_1905_1918"].median()),
            "site_count_direct_gauge_fraction_at_least_half": int(high_support.sum()),
            "site_count_direct_gauge_fraction_below_half": int(low_support.sum()),
            "mean_clean_gpcc_event_change_mm_day": float(np.mean(clean_gpcc_event_change)),
            "mean_stored_gpcc_event_change_mm_day": float(table["precipitation_gpcc_event_change"].mean()),
            "mean_cru_event_change_mm_day": float(table["precipitation_cru_event_change"].mean()),
            "mean_abs_residual_high_support_z": float(quality.loc[high_support, "absolute_selected_prediction_residual_z"].mean()) if high_support.any() else None,
            "mean_abs_residual_low_support_z": float(quality.loc[low_support, "absolute_selected_prediction_residual_z"].mean()) if low_support.any() else None,
        },
        "interpretation_limit": "This clean branch is a diagnostic reconstruction input candidate, not an authorized replacement for the production precipitation observation or PSM.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    axes[0, 0].plot(EVENT_YEARS, curves["actual_proxy_network_z_from_baseline"], color="#1B1B1B", linewidth=2.2, label="Stored proxies")
    axes[0, 0].plot(EVENT_YEARS, curves["BEA_clean_GPCC_annual_refit_network_z_from_baseline"], color="#0072B2", linewidth=1.5, label="BEA + clean GPCC annual")
    axes[0, 0].plot(EVENT_YEARS, curves["BEA_clean_GPCC_selected_season_network_z_from_baseline"], color="#009E73", linewidth=1.8, label="BEA + clean GPCC selected")
    axes[0, 0].plot(EVENT_YEARS, curves["CRU_clean_GPCC_selected_season_network_z_from_baseline"], color="#D55E00", linewidth=1.5, label="CRU + clean GPCC selected")
    axes[0, 0].axhline(0, color="#999999", linewidth=0.7); axes[0, 0].axvspan(1915, 1918, color="#D9D9D9", alpha=0.45)
    axes[0, 0].set_ylabel("Network mean change (calibration z)"); axes[0, 0].set_title("Clean GPCC event prediction"); axes[0, 0].legend(frameon=False, fontsize=8)

    limit = max(0.5, float(np.nanmax(np.abs(np.concatenate([actual_event_z, predictions["BEA_clean_GPCC_annual_refit"], predictions["BEA_clean_GPCC_selected_season"]])))))
    axes[0, 1].scatter(predictions["BEA_clean_GPCC_annual_refit"], actual_event_z, color="#0072B2", alpha=0.65, s=32, label="annual")
    axes[0, 1].scatter(predictions["BEA_clean_GPCC_selected_season"], actual_event_z, facecolors="none", edgecolors="#009E73", alpha=0.85, s=38, label="selected season")
    axes[0, 1].plot([-limit, limit], [-limit, limit], color="#777777", linewidth=0.8); axes[0, 1].axhline(0, color="#999999", linewidth=0.6); axes[0, 1].axvline(0, color="#999999", linewidth=0.6)
    axes[0, 1].set_xlim(-limit, limit); axes[0, 1].set_ylim(-limit, limit); axes[0, 1].set_xlabel("Predicted event change (z)"); axes[0, 1].set_ylabel("Proxy event change (z)"); axes[0, 1].set_title("Site-level prediction"); axes[0, 1].legend(frameon=False)

    order = list(base.SEASONS); x = np.arange(len(order)); width = 0.38
    for offset, branch, color in ((-width / 2, "BEA_clean_GPCC", "#0072B2"), (width / 2, "CRU_clean_GPCC", "#D55E00")):
        counts = proxy_results[proxy_results["branch"] == branch]["selected_season"].value_counts()
        axes[1, 0].bar(x + offset, [counts.get(season, 0) for season in order], width, color=color, label=branch.replace("_", " "))
    axes[1, 0].set_xticks(x, order); axes[1, 0].set_ylabel("Selected proxy count"); axes[1, 0].set_title("1941-2000 blocked-CV season selection"); axes[1, 0].legend(frameon=False, fontsize=8)

    scatter = axes[1, 1].scatter(quality["direct_gauge_month_fraction_1905_1918"], quality["absolute_selected_prediction_residual_z"], c=latitudes, cmap="viridis", s=42, alpha=0.8, edgecolor="#333333", linewidth=0.3)
    axes[1, 1].set_xlabel("Direct-gauge month fraction, 1905-1918"); axes[1, 1].set_ylabel("Absolute selected prediction residual (z)"); axes[1, 1].set_title("Early GPCC support versus residual"); figure.colorbar(scatter, ax=axes[1, 1], label="Latitude")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "clean_gpcc_v2022_positive_b2_psm.png", dpi=190)
    plt.close(figure)

    report = [
        "# Clean GPCC v2022 Positive-b2 PSM Result",
        "",
        "The clean branch uses raw GPCC v2022 monthly totals, exact Gregorian day counts, and 1961-1990 seasonal reference means.",
        "",
        f"- BEA + clean GPCC annual event r/sign: `{metric_index.loc['BEA_clean_GPCC_annual_refit', 'event_change_correlation']:.3f}` / `{metric_index.loc['BEA_clean_GPCC_annual_refit', 'sign_agreement_fraction']:.1%}`.",
        f"- BEA + clean GPCC selected-season event r/sign: `{metric_index.loc['BEA_clean_GPCC_selected_season', 'event_change_correlation']:.3f}` / `{metric_index.loc['BEA_clean_GPCC_selected_season', 'sign_agreement_fraction']:.1%}`.",
        f"- Selected-season network prediction: `{metric_index.loc['BEA_clean_GPCC_selected_season', 'network_mean_predicted_event_change_z']:.3f} z`; actual `{metric_index.loc['BEA_clean_GPCC_selected_season', 'network_mean_actual_event_change_z']:.3f} z`.",
        f"- Mean clean/stored/CRU instrumental event change: `{summary['quality_summary']['mean_clean_gpcc_event_change_mm_day']:.3f}` / `{summary['quality_summary']['mean_stored_gpcc_event_change_mm_day']:.3f}` / `{summary['quality_summary']['mean_cru_event_change_mm_day']:.3f}` mm/day.",
        f"- Median direct-gauge month fraction: `{summary['quality_summary']['median_direct_gauge_month_fraction']:.1%}`.",
        "",
        "This remains a diagnostic branch and does not modify the production reconstruction.",
        "No values from 1921-1939 were read or analyzed.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = {name: base.source_stat(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("source-inputs-changed-during-audit")
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "driver_sha256": base.digest(Path(__file__)),
                "facts_sha256": base.digest(root / "LIA_PROJECT_FACTS.md"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

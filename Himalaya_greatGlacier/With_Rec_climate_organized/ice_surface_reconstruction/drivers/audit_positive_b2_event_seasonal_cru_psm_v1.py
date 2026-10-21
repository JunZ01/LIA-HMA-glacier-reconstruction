"""Run the seasonal event PSM audit on the CRU branch that passes closure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_base() -> object:
    path = Path(__file__).with_name("audit_positive_b2_event_seasonal_psm_v1.py")
    spec = importlib.util.spec_from_file_location("positive_b2_event_seasonal_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("base-driver-import-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.time_slice = lambda start_year, first_year, last_year: slice(
        (first_year - 1 - start_year) * 12 + 11,
        (last_year - start_year + 1) * 12,
    )
    return module


def main() -> int:
    base = load_base()
    root = base.project_root()
    event_table_path = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_site_instrumental_consistency_v1/site_instrumental_event_consistency.csv"
    proxy6 = base.archive_proxy_path()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_seasonal_cru_psm_v1"
    sources = {
        "event_table": event_table_path,
        "proxy6": proxy6,
        "cru_temperature": base.CRU_TEMPERATURE,
        "cru_precipitation": base.CRU_PRECIPITATION,
        "pages_metadata": base.PAGES_METADATA,
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    event_hash_before = base.digest(event_table_path)
    stats_before = {name: base.source_stat(path) for name, path in sources.items() if name != "event_table"}

    table = pd.read_csv(event_table_path).sort_values("proxy_index").reset_index(drop=True)
    if len(table) != 60 or not (table["coefficient_b2"] >= 0).all():
        raise RuntimeError("positive-b2-selection-contract-failed")
    latitudes = table["latitude"].to_numpy(dtype=float)
    longitudes = table["longitude"].to_numpy(dtype=float)

    pages = loadmat(base.PAGES_METADATA, variable_names=["p_lat", "p_lon", "pages2k_dataName"], squeeze_me=True, struct_as_record=False)
    pages_lat = np.asarray(pages["p_lat"], dtype=float).reshape(-1)
    pages_lon = np.asarray(pages["p_lon"], dtype=float).reshape(-1)
    pages_names = np.asarray(pages["pages2k_dataName"]).reshape(-1)
    candidate_counts = []
    candidate_names = []
    for latitude, longitude in zip(latitudes, longitudes):
        candidates = np.flatnonzero(np.isclose(pages_lat, latitude, atol=1e-6) & np.isclose(pages_lon, longitude, atol=1e-6))
        candidate_counts.append(len(candidates))
        candidate_names.append("|".join(str(pages_names[index]).strip() for index in candidates))
    table["pages2k_coordinate_candidate_count"] = candidate_counts
    table["pages2k_coordinate_candidate_names"] = candidate_names

    proxy_event = np.full((len(base.EVENT_YEARS), len(table)), np.nan)
    proxy_calibration = np.full((len(base.CALIBRATION_YEARS), len(table)), np.nan)
    with h5py.File(proxy6, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(table["proxy_index"].to_numpy(dtype=int)):
            proxy = handle[refs[proxy_index - 1]]
            event_slice = slice(1904, 1920)
            calibration_slice = slice(1939, 2000)
            if not np.array_equal(np.asarray(proxy["age_yng"][0, event_slice], dtype=float), base.EVENT_YEARS):
                raise RuntimeError(f"proxy-event-year-contract-failed:{proxy_index}")
            if not np.array_equal(np.asarray(proxy["age_yng"][0, calibration_slice], dtype=float), base.CALIBRATION_YEARS):
                raise RuntimeError(f"proxy-calibration-year-contract-failed:{proxy_index}")
            proxy_event[:, column] = np.asarray(proxy["data"][event_slice, 0], dtype=float)
            proxy_calibration[:, column] = np.asarray(proxy["data"][calibration_slice, 0], dtype=float)
    proxy_scale = np.nanstd(proxy_calibration, axis=0)
    actual_event_z = base.period_change(base.EVENT_YEARS, proxy_event) / proxy_scale
    if not np.allclose(actual_event_z, table["event_drop_z"].to_numpy(), atol=1e-12, rtol=0):
        raise RuntimeError("proxy-event-closure-failed")

    cru_t, cru_lat, cru_lon = base.read_netcdf_points(base.CRU_TEMPERATURE, ["tmp", "stn"], 1901, latitudes, longitudes)
    cru_p, cru_p_lat, cru_p_lon = base.read_netcdf_points(base.CRU_PRECIPITATION, ["pre", "stn"], 1901, latitudes, longitudes)
    if not (np.array_equal(cru_lat, cru_p_lat) and np.array_equal(cru_lon, cru_p_lon)):
        raise RuntimeError("cru-grid-selection-disagreement")
    event_labels = base.monthly_labels(1905, 1920)
    calibration_labels = base.monthly_labels(1940, 2000)
    event_temperature = {}
    event_precipitation = {}
    calibration_temperature = {}
    calibration_precipitation = {}
    for season, months in base.SEASONS.items():
        event_temperature[season] = base.seasonal_aggregate(cru_t["event"]["tmp"], event_labels, base.EVENT_YEARS, months)
        event_precipitation[season] = base.seasonal_aggregate(cru_p["event"]["pre"], event_labels, base.EVENT_YEARS, months)
        calibration_temperature[season] = base.seasonal_aggregate(cru_t["calibration"]["tmp"], calibration_labels, base.CALIBRATION_YEARS, months)
        calibration_precipitation[season] = base.seasonal_aggregate(cru_p["calibration"]["pre"], calibration_labels, base.CALIBRATION_YEARS, months)

    monthly_temperature_change = base.period_change(base.EVENT_YEARS, event_temperature["annual"])
    monthly_precipitation_change = base.period_change(base.EVENT_YEARS, event_precipitation["annual"]) / (365.25 / 12.0)
    closure = pd.DataFrame(
        [
            {
                "product": "CRU_temperature",
                "monthly_vs_stored_event_change_correlation": base.correlation(monthly_temperature_change, table["temperature_cru_event_change"].to_numpy()),
                "rmse": float(np.sqrt(np.mean((monthly_temperature_change - table["temperature_cru_event_change"].to_numpy()) ** 2))),
            },
            {
                "product": "CRU_precipitation",
                "monthly_vs_stored_event_change_correlation": base.correlation(monthly_precipitation_change, table["precipitation_cru_event_change"].to_numpy()),
                "rmse": float(np.sqrt(np.mean((monthly_precipitation_change - table["precipitation_cru_event_change"].to_numpy()) ** 2))),
            },
        ]
    )
    if (closure["monthly_vs_stored_event_change_correlation"] < 0.95).any():
        raise RuntimeError("cru-monthly-to-annual-closure-failed")

    cv_table, proxy_results, annual_predictions, selected_predictions = base.branch_analysis(
        "CRU",
        proxy_calibration,
        proxy_scale,
        calibration_temperature,
        calibration_precipitation,
        event_temperature,
        event_precipitation,
    )
    proxy_results["proxy_index"] = table.iloc[proxy_results["proxy_column"].to_numpy(dtype=int)]["proxy_index"].to_numpy()
    proxy_table = table.merge(proxy_results.drop(columns="proxy_column"), on="proxy_index", how="left", validate="one_to_one")
    predictions = {
        "CRU_annual_refit": base.period_change(base.EVENT_YEARS, annual_predictions) / proxy_scale,
        "CRU_selected_season": base.period_change(base.EVENT_YEARS, selected_predictions) / proxy_scale,
        "archived_annual_CRU_combination": table["cru_psm_predicted_event_change_z"].to_numpy(dtype=float),
    }
    rng = np.random.default_rng(base.RNG_SEED)
    metrics = pd.DataFrame([base.metric_row(label, actual_event_z, values, rng) for label, values in predictions.items()])

    event_month_mask = np.asarray([1905 <= year <= 1918 for year, _ in event_labels])
    coverage = pd.DataFrame(
        {
            "proxy_index": table["proxy_index"],
            "latitude": latitudes,
            "longitude": longitudes,
            "cru_temperature_station_month_fraction": np.mean(cru_t["event"]["stn"][event_month_mask] > 0, axis=0),
            "cru_precipitation_station_month_fraction": np.mean(cru_p["event"]["stn"][event_month_mask] > 0, axis=0),
        }
    )
    baseline_mask = (base.EVENT_YEARS >= 1905) & (base.EVENT_YEARS <= 1914)
    network_curves = pd.DataFrame(
        {
            "year": base.EVENT_YEARS,
            "actual_proxy_network_z_from_baseline": np.nanmean((proxy_event - np.nanmean(proxy_event[baseline_mask], axis=0)[None, :]) / proxy_scale[None, :], axis=1),
        }
    )
    for label, values in (("CRU_annual_refit", annual_predictions), ("CRU_selected_season", selected_predictions)):
        centered = (values - np.nanmean(values[baseline_mask], axis=0)[None, :]) / proxy_scale[None, :]
        network_curves[label + "_network_z_from_baseline"] = np.nanmean(centered, axis=1)

    metric_index = metrics.set_index("prediction")
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_positive_b2_event_seasonal_cru_psm_v1",
        "analyzed_periods": [[1904, 1920], [1939, 2000]],
        "locked_final_period": [1921, 1939],
        "locked_final_period_accessed_by_this_driver": False,
        "candidate_seasons": list(base.SEASONS),
        "selection_rule": "Per-proxy minimum 1940-2000 blocked-CV normalized RMSE; event years excluded from fitting and selection.",
        "cru_monthly_annual_closure": closure.to_dict(orient="records"),
        "event_prediction_metrics": metrics.to_dict(orient="records"),
        "selected_season_counts": {str(key): int(value) for key, value in proxy_results["selected_season"].value_counts().to_dict().items()},
        "median_selected_minus_annual_cv_nrmse": float(proxy_results["selected_minus_annual_cv_nrmse"].median()),
        "median_cru_temperature_station_month_fraction_1905_1918": float(coverage["cru_temperature_station_month_fraction"].median()),
        "median_cru_precipitation_station_month_fraction_1905_1918": float(coverage["cru_precipitation_station_month_fraction"].median()),
        "pages2k_coordinate_identity": {
            "unique_count": int(np.sum(np.asarray(candidate_counts) == 1)),
            "ambiguous_count": int(np.sum(np.asarray(candidate_counts) > 1)),
            "unmatched_count": int(np.sum(np.asarray(candidate_counts) == 0)),
        },
        "excluded_production_seasonal_branch": "Monthly GPCC versus stored annual GPCC site-event closure correlation was 0.864, below the frozen 0.95 provenance gate. BEA+GPCC seasonal results were not produced.",
        "interpretation_limit": "This is a calibration-selected CRU seasonality sensitivity, not a replacement PSM. CRU station counts diagnose product support, not truth or formal uncertainty.",
    }

    output.mkdir(parents=True)
    cv_table.to_csv(output / "cru_seasonal_psm_blocked_cv.csv", index=False)
    proxy_table.to_csv(output / "cru_seasonal_psm_proxy_results.csv", index=False)
    metrics.to_csv(output / "cru_seasonal_event_prediction_metrics.csv", index=False)
    coverage.to_csv(output / "cru_early_station_coverage_at_proxy_sites.csv", index=False)
    closure.to_csv(output / "cru_monthly_annual_closure.csv", index=False)
    network_curves.to_csv(output / "cru_seasonal_event_network_curves.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "BOUNDARY_NOTICE.md").write_text(
        "# Boundary Notice\n\nA separate MATLAB preflight on legacy non-v7.3 PAGES/combined MAT files forced full-variable I/O despite indexed syntax. No 1921-1939 values were inspected, analyzed, exported, or used, and that numeric source-lineage branch was terminated. This CRU seasonal driver reads only 1904-1920 and 1939-2000 NetCDF/HDF slices. The BEA+GPCC seasonal branch was also stopped because monthly-to-stored annual GPCC closure failed its 0.95 gate.\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), constrained_layout=True)
    axes[0, 0].plot(base.EVENT_YEARS, network_curves["actual_proxy_network_z_from_baseline"], color="#1B1B1B", linewidth=2.2, label="Stored proxies")
    axes[0, 0].plot(base.EVENT_YEARS, network_curves["CRU_annual_refit_network_z_from_baseline"], color="#0072B2", linewidth=1.7, label="CRU annual refit")
    axes[0, 0].plot(base.EVENT_YEARS, network_curves["CRU_selected_season_network_z_from_baseline"], color="#D55E00", linewidth=1.8, label="CRU selected season")
    axes[0, 0].axhline(0, color="#999999", linewidth=0.8); axes[0, 0].axvspan(1915, 1918, color="#D9D9D9", alpha=0.45); axes[0, 0].set_ylabel("Network mean change (calibration z)"); axes[0, 0].set_title("CRU seasonal prediction versus proxy event"); axes[0, 0].legend(frameon=False)
    limit = max(0.5, float(np.nanmax(np.abs(np.concatenate([actual_event_z, predictions["CRU_annual_refit"], predictions["CRU_selected_season"]])))))
    axes[0, 1].scatter(predictions["CRU_annual_refit"], actual_event_z, color="#0072B2", alpha=0.68, s=32, label="annual refit")
    axes[0, 1].scatter(predictions["CRU_selected_season"], actual_event_z, facecolors="none", edgecolors="#D55E00", alpha=0.85, s=38, label="selected season")
    axes[0, 1].plot([-limit, limit], [-limit, limit], color="#777777", linewidth=0.8); axes[0, 1].axhline(0, color="#999999", linewidth=0.7); axes[0, 1].axvline(0, color="#999999", linewidth=0.7); axes[0, 1].set_xlim(-limit, limit); axes[0, 1].set_ylim(-limit, limit); axes[0, 1].set_xlabel("CRU predicted event change (z)"); axes[0, 1].set_ylabel("Stored proxy event change (z)"); axes[0, 1].set_title("Site-level event prediction"); axes[0, 1].legend(frameon=False)
    season_order = list(base.SEASONS); counts = proxy_results["selected_season"].value_counts(); axes[1, 0].bar(season_order, [counts.get(season, 0) for season in season_order], color=["#7A7A7A", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]); axes[1, 0].set_ylabel("Selected proxy count"); axes[1, 0].set_title("Seasons selected by 1940-2000 blocked CV")
    residual = np.abs(actual_event_z - predictions["CRU_selected_season"]); scatter = axes[1, 1].scatter(coverage["cru_precipitation_station_month_fraction"], residual, c=latitudes, cmap="viridis", s=42, alpha=0.8, edgecolor="#333333", linewidth=0.3); axes[1, 1].set_xlabel("CRU precipitation station-month fraction, 1905-1918"); axes[1, 1].set_ylabel("Absolute seasonal prediction residual (z)"); axes[1, 1].set_title("Early CRU support versus unexplained event"); figure.colorbar(scatter, ax=axes[1, 1], label="Latitude")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "positive_b2_event_seasonal_cru_psm.png", dpi=180); plt.close(figure)

    report = [
        "# Positive-b2 Event Seasonal CRU PSM Audit",
        "",
        "Season selection uses only 1940-2000 blocked cross-validation. Event years are excluded from fitting and selection.",
        "",
        f"- Annual-refit event correlation/sign agreement: `{metric_index.loc['CRU_annual_refit', 'event_change_correlation']:.3f}` / `{metric_index.loc['CRU_annual_refit', 'sign_agreement_fraction']:.1%}`.",
        f"- Selected-season event correlation/sign agreement: `{metric_index.loc['CRU_selected_season', 'event_change_correlation']:.3f}` / `{metric_index.loc['CRU_selected_season', 'sign_agreement_fraction']:.1%}`.",
        f"- Selected-season predicted network event: `{metric_index.loc['CRU_selected_season', 'network_mean_predicted_event_change_z']:.3f} z`; actual `{metric_index.loc['CRU_selected_season', 'network_mean_actual_event_change_z']:.3f} z`.",
        f"- Median CRU precipitation station-month fraction: `{summary['median_cru_precipitation_station_month_fraction_1905_1918']:.1%}`.",
        "",
        "The BEA+GPCC seasonal branch is excluded because its monthly source did not close against the stored annual GPCC product. See BOUNDARY_NOTICE.md.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    event_hash_after = base.digest(event_table_path)
    stats_after = {name: base.source_stat(path) for name, path in sources.items() if name != "event_table"}
    if event_hash_before != event_hash_after or stats_before != stats_after:
        raise RuntimeError("source-inputs-changed-during-audit")
    (output / "source_manifest.json").write_text(json.dumps({"event_table_sha256_before": event_hash_before, "event_table_sha256_after": event_hash_after, "source_stats_before": stats_before, "source_stats_after": stats_after, "analysis_driver_sha256": base.digest(Path(__file__))}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

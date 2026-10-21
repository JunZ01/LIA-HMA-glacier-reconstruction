from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

import benchmark_temperature_reconstructions_against_ghcnm_v1 as external
import evaluate_temperature_four_background_instrumental_metrics_v1 as metrics


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
MODEL_FIELDS = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_v1/"
    "temperature_equal_blend_annual_spatial_candidate_1500_1899.nc"
)
SCHEME_VALUES = LOOP / "temperature_four_background_scheme_evaluation_v2/modern_station_year_scheme_values.csv"
OUT = LOOP / "temperature_reconstruction_cru_target_benchmark_v1"

TARGET_VARIABLE = "tas_obs_cru_tmp_05deg"
PROJECT_VARIABLE = "tas_ensemble_mean_rec_result"
SCHEMES = {
    "current_CESM": "current_CESM_degC",
    "old_50_50": "old_50_50_degC",
    "equal_four": "equal_four_degC",
    "skill_independence": "skill_independence_oof_degC",
}
FIELD_PRODUCTS = {
    "project_current": "project_current",
    "local_EnKF_family": "local_enkf",
    "LMR": "lmr",
    "Neukom_family": "neukom",
    "PHYDA": "phyda",
    "ref_rec_family_equal": "family_equal",
}


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def clean(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result[(result < -9000) | ~np.isfinite(result)] = np.nan
    return result


def anomaly(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    center = np.nanmean(values[baseline], axis=0)
    return values - center[None, ...]


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def read_common_hma_fields() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray
]:
    with Dataset(MODEL_FIELDS) as dataset:
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
    with h5py.File(PUBLIC, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        selected_indices = np.flatnonzero((years >= 1951) & (years <= 2000))
        selected_years = years[selected_indices]
        if not np.array_equal(selected_years, np.arange(1951, 2001)):
            raise RuntimeError("CRU target year contract failed")
        baseline_indices = np.flatnonzero((years >= 1961) & (years <= 1990))
        baseline_selected = (selected_years >= 1961) & (selected_years <= 1990)
        public_lat = np.asarray(handle["lat"], dtype=float).ravel()
        public_lon = np.asarray(handle["lon"], dtype=float).ravel()
        if not (np.allclose(public_lat, lat) and np.allclose(public_lon, lon)):
            raise RuntimeError("CRU/project/model grid mismatch")

        required_variables = [TARGET_VARIABLE, PROJECT_VARIABLE, *external.MEMBERS.values()]
        common = hma_mask.copy()
        for variable in required_variables:
            baseline_values = clean(handle[variable][baseline_indices, :, :])
            common &= np.isfinite(baseline_values).sum(axis=0) >= 20
        if int(common.sum()) == 0:
            raise RuntimeError("No common HMA support across CRU target and reconstructions")

        target = anomaly(clean(handle[TARGET_VARIABLE][selected_indices, :, :]), baseline_selected)[:, common]
        project = anomaly(clean(handle[PROJECT_VARIABLE][selected_indices, :, :]), baseline_selected)[:, common]
        member_points = {
            name: anomaly(clean(handle[variable][selected_indices, :, :]), baseline_selected)[:, common]
            for name, variable in external.MEMBERS.items()
        }
    local_enkf = finite_mean([member_points["HGAO"], member_points["O"]])
    neukom = finite_mean([values for name, values in member_points.items() if name.startswith("Nk_")])
    products = {
        "project_current": project,
        "local_enkf": local_enkf,
        "lmr": member_points["LMR"],
        "neukom": neukom,
        "phyda": member_points["PHYDA"],
    }
    products["family_equal"] = finite_mean([
        products["local_enkf"], products["lmr"], products["neukom"], products["phyda"]
    ])
    cell_latitudes = np.broadcast_to(lat[:, None], common.shape)[common]
    cell_longitudes = np.broadcast_to(lon[None, :], common.shape)[common]
    return selected_years, target, common, products, cell_latitudes, cell_longitudes


def area_mean(points: np.ndarray, cell_latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(cell_latitudes))
    valid = np.isfinite(points)
    numerator = np.nansum(points * weights[None, :], axis=1)
    denominator = np.sum(valid * weights[None, :], axis=1)
    return numerator / np.where(denominator == 0, np.nan, denominator)


def field_metrics(
    years: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    cell_latitudes: np.ndarray,
    product: str,
) -> dict[str, object]:
    target_mean = area_mean(target, cell_latitudes)
    predicted_mean = area_mean(predicted, cell_latitudes)
    curve = metrics.paired_metrics(target_mean, predicted_mean)
    target_trend = metrics.slope_per_century(years.astype(float), target_mean)
    predicted_trend = metrics.slope_per_century(years.astype(float), predicted_mean)

    cell_rows = []
    for cell in range(target.shape[1]):
        result = metrics.paired_metrics(target[:, cell], predicted[:, cell])
        cell_rows.append(result)
    cells = pd.DataFrame(cell_rows)
    spatial_correlations = []
    for index in range(len(years)):
        spatial_correlations.append(
            metrics.paired_metrics(target[index], predicted[index])["correlation_r"]
        )
    valid = np.isfinite(target) & np.isfinite(predicted)
    weights = np.cos(np.deg2rad(cell_latitudes))[None, :]
    error = predicted - target
    weighted_count = float(np.sum(valid * weights))
    pooled_rmse = float(np.sqrt(np.nansum(np.square(error) * weights) / weighted_count))
    pooled_mae = float(np.nansum(np.abs(error) * weights) / weighted_count)
    pooled_bias = float(np.nansum(error * weights) / weighted_count)
    return {
        "product": product,
        "common_cell_count": int(target.shape[1]),
        "hma_mean_rmse_degC": curve["rmse_degC"],
        "hma_mean_mae_degC": curve["mae_degC"],
        "hma_mean_bias_degC": curve["bias_degC"],
        "hma_mean_correlation_r": curve["correlation_r"],
        "hma_mean_variance_ratio": curve["variance_ratio"],
        "hma_mean_re": curve["re_climatology_zero"],
        "hma_mean_ce": curve["ce_evaluation_mean"],
        "hma_mean_target_trend_degC_per_century": target_trend,
        "hma_mean_product_trend_degC_per_century": predicted_trend,
        "hma_mean_abs_trend_error_degC_per_century": abs(predicted_trend - target_trend),
        "area_time_pooled_rmse_degC": pooled_rmse,
        "area_time_pooled_mae_degC": pooled_mae,
        "area_time_pooled_bias_degC": pooled_bias,
        "median_cell_temporal_correlation_r": float(cells["correlation_r"].median()),
        "median_cell_rmse_degC": float(cells["rmse_degC"].median()),
        "median_cell_re": float(cells["re_climatology_zero"].median()),
        "median_cell_ce": float(cells["ce_evaluation_mean"].median()),
        "median_cell_variance_ratio": float(cells["variance_ratio"].median()),
        "median_annual_spatial_correlation_r": float(np.nanmedian(spatial_correlations)),
    }


def extract_cru_to_scheme_locations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stations = frame[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates()
    with Dataset(MODEL_FIELDS) as dataset:
        support = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
    with h5py.File(PUBLIC, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((years >= 1951) & (years <= 2000))
        selected_years = years[indices]
        baseline = (selected_years >= 1961) & (selected_years <= 1990)
        target_rows = []
        mapping_rows = []
        for station in stations.itertuples(index=False):
            distance = np.hypot(
                lat[:, None] - float(station.latitude), lon[None, :] - float(station.longitude)
            )
            distance[~support] = np.inf
            row, col = np.unravel_index(np.argmin(distance), distance.shape)
            values = clean(handle[TARGET_VARIABLE][indices, row, col])
            if int(np.isfinite(values[baseline]).sum()) < 20:
                raise RuntimeError(f"CRU baseline unavailable at frozen scheme location for {station.station_id}")
            values = values - np.nanmean(values[baseline])
            project = anomaly(clean(handle[PROJECT_VARIABLE][indices, row, col])[:, None], baseline).ravel()
            existing = frame[frame["station_id"].eq(station.station_id)].set_index("year")["current_CESM_degC"]
            common_years = existing.index.intersection(selected_years)
            closure = np.nanmax(np.abs(existing.loc[common_years].to_numpy(float) - project[common_years - 1951]))
            if closure > 1e-10:
                raise RuntimeError(f"Current-project station extraction closure failed: {station.station_id}, {closure}")
            mapping_rows.append({
                "station_id": station.station_id,
                "station_name": station.station_name,
                "source_latitude": float(lat[row]),
                "source_longitude": float(lon[col]),
                "source_distance_deg": float(distance[row, col]),
                "current_project_closure_max_abs_degC": float(closure),
            })
            for year, value in zip(selected_years, values):
                target_rows.append({"station_id": station.station_id, "year": int(year), "cru_target_degC": float(value)})
    return pd.DataFrame(target_rows), pd.DataFrame(mapping_rows)


def plot_field_curves(years: np.ndarray, target: np.ndarray, products: dict[str, np.ndarray], lat: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    axis.plot(years, area_mean(target, lat), color="#111111", linewidth=2.8, label="CRU target")
    colors = ["#1f4e79", "#c55a11", "#548235", "#8064a2", "#bf9000", "#666666"]
    for (label, key), color in zip(FIELD_PRODUCTS.items(), colors):
        axis.plot(years, area_mean(products[key], lat), linewidth=1.7, color=color, label=label)
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set(xlabel="Year", ylabel="Temperature anomaly (degC)", title="HMA mean temperature: CRU as evaluation target")
    axis.legend(ncol=2, frameon=False)
    figure.savefig(OUT / "cru_target_hma_mean_curves_1951_2000.png", dpi=220)
    plt.close(figure)


def plot_metric_grid(table: pd.DataFrame, path: Path, title: str) -> None:
    labels = table.iloc[:, 0].tolist()
    colors = ["#1f4e79", "#c55a11", "#548235", "#8064a2", "#bf9000", "#666666"][:len(labels)]
    panels = [
        ("hma_mean_rmse_degC", "HMA mean RMSE", "degC"),
        ("hma_mean_mae_degC", "HMA mean MAE", "degC"),
        ("hma_mean_correlation_r", "HMA mean correlation", "r"),
        ("hma_mean_bias_degC", "HMA mean bias", "degC"),
        ("hma_mean_variance_ratio", "HMA mean variance ratio", "ratio"),
        ("hma_mean_ce", "HMA mean CE", "score"),
        ("hma_mean_abs_trend_error_degC_per_century", "Trend error", "degC/century"),
        ("median_cell_temporal_correlation_r", "Median cell temporal r", "r"),
        ("median_annual_spatial_correlation_r", "Median annual spatial r", "r"),
    ]
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    x = np.arange(len(labels))
    for axis, (column, panel_title, ylabel) in zip(axes.ravel(), panels):
        axis.bar(x, table[column], color=colors)
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.set_title(panel_title)
        axis.set_ylabel(ylabel)
        if column in {"hma_mean_correlation_r", "hma_mean_ce", "median_cell_temporal_correlation_r", "median_annual_spatial_correlation_r"}:
            axis.axhline(0, color="#222222", linewidth=0.8)
        if column == "hma_mean_variance_ratio":
            axis.axhline(1, color="#222222", linewidth=0.8)
    figure.suptitle(title, fontsize=15)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    before = {name: state(path) for name, path in {
        "public": PUBLIC, "model_fields": MODEL_FIELDS, "scheme_values": SCHEME_VALUES
    }.items()}
    years, target, common, products, cell_latitudes, cell_longitudes = read_common_hma_fields()
    field_rows = [
        field_metrics(years, target, products[key], cell_latitudes, label)
        for label, key in FIELD_PRODUCTS.items()
    ]
    field_table = pd.DataFrame(field_rows)

    scheme_frame = pd.read_csv(SCHEME_VALUES)
    target_station, mapping = extract_cru_to_scheme_locations(scheme_frame)
    scheme_frame = scheme_frame.merge(target_station, on=["station_id", "year"], validate="one_to_one")
    scheme_rows = []
    scheme_station_tables = []
    for scheme, column in SCHEMES.items():
        station = metrics.station_metrics(scheme_frame, "cru_target_degC", scheme, column)
        decades = metrics.decadal_pairs(scheme_frame, "cru_target_degC", column)
        scheme_station_tables.append(station)
        scheme_rows.append(metrics.aggregate_metrics(
            scheme_frame, station, decades, "cru_target_degC", scheme, column
        ))
    scheme_table = pd.DataFrame(scheme_rows)

    field_table.to_csv(OUT / "hma_field_metrics_cru_target.csv", index=False)
    scheme_table.to_csv(OUT / "four_scheme_metrics_cru_target.csv", index=False)
    pd.concat(scheme_station_tables, ignore_index=True).to_csv(
        OUT / "four_scheme_station_metrics_cru_target.csv", index=False
    )
    mapping.to_csv(OUT / "scheme_station_cru_grid_mapping.csv", index=False)
    pd.DataFrame({
        "cell_latitude": cell_latitudes, "cell_longitude": cell_longitudes
    }).to_csv(OUT / "hma_common_cells.csv", index=False)
    curve_table = {"year": years, "CRU_target": area_mean(target, cell_latitudes)}
    for label, key in FIELD_PRODUCTS.items():
        curve_table[label] = area_mean(products[key], cell_latitudes)
    pd.DataFrame(curve_table).to_csv(OUT / "hma_mean_curves_cru_target.csv", index=False)
    plot_field_curves(years, target, products, cell_latitudes)
    plot_metric_grid(
        field_table,
        OUT / "cru_target_hma_full_metrics.png",
        "Reconstructions evaluated against CRU annual temperature",
    )

    scheme_plot = scheme_table.rename(columns={"branch": "product"}).copy()
    scheme_plot["hma_mean_rmse_degC"] = scheme_plot["annual_pooled_rmse_degC"]
    scheme_plot["hma_mean_mae_degC"] = scheme_plot["annual_pooled_mae_degC"]
    scheme_plot["hma_mean_correlation_r"] = scheme_plot["annual_pooled_correlation_r"]
    scheme_plot["hma_mean_bias_degC"] = scheme_plot["annual_pooled_bias_degC"]
    scheme_plot["hma_mean_variance_ratio"] = scheme_plot["annual_pooled_variance_ratio"]
    scheme_plot["hma_mean_ce"] = scheme_plot["annual_pooled_ce"]
    scheme_plot["hma_mean_abs_trend_error_degC_per_century"] = scheme_plot["median_abs_trend_error_degC_per_century"]
    scheme_plot["median_cell_temporal_correlation_r"] = scheme_plot["median_station_correlation_r"]
    scheme_plot["median_annual_spatial_correlation_r"] = np.nan
    plot_metric_grid(
        scheme_plot,
        OUT / "cru_target_four_scheme_full_metrics.png",
        "Four background schemes evaluated against CRU at nine matched locations",
    )

    after = {name: state(Path(item["path"])) for name, item in before.items()}
    if before != after:
        raise RuntimeError("A read-only source changed during the CRU-target experiment")
    result = {
        "experiment": OUT.name,
        "status": "complete",
        "target": "CRU annual TMP on the public 0.5-degree project grid",
        "target_variable": TARGET_VARIABLE,
        "period": [1951, 2000],
        "baseline": [1961, 1990],
        "hma_common_cell_count": int(common.sum()),
        "scheme_station_count": int(scheme_frame["station_id"].nunique()),
        "field_metrics": field_table.to_dict(orient="records"),
        "scheme_metrics": scheme_table.to_dict(orient="records"),
        "source_states_before": before,
        "source_states_after": after,
        "source_unchanged": True,
        "locked_period_access": "none; values read only for 1951-2000",
        "interpretation_boundary": (
            "CRU is treated as the fixed evaluation target for this sensitivity experiment, not asserted to be "
            "error-free truth. Metrics partly measure agreement with CRU station selection, homogenization, and gridding."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(
        "# Experiment contract\n\n"
        "- Treat CRU annual TMP as the fixed evaluation target over 1951-2000.\n"
        "- Use product-native 1961-1990 anomalies and strict common HMA support.\n"
        "- Evaluate the four background schemes at their frozen nine locations and public reconstructions over the HMA field.\n"
        "- Do not read 1921-1939, modify production files, run EnKF, or run OGGM.\n"
        "- Interpret the result as CRU-conditioned agreement, not proof that CRU is error-free.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

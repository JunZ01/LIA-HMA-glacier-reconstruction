from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

import benchmark_temperature_reconstructions_against_ghcnm_v1 as external
import benchmark_temperature_reconstructions_with_cru_target_v1 as cru_base
import evaluate_temperature_four_background_instrumental_metrics_v1 as metrics


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
PUBLIC = cru_base.PUBLIC
MODEL_FIELDS = cru_base.MODEL_FIELDS
SCHEME_VALUES = cru_base.SCHEME_VALUES
OUT = LOOP / "temperature_reconstruction_multi_instrumental_target_benchmark_v1"

TARGETS = {
    "CRU": "tas_obs_cru_tmp_05deg",
    "Berkeley": "tas_obs_bea_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
}
SCHEMES = cru_base.SCHEMES
FIELD_PRODUCTS = cru_base.FIELD_PRODUCTS
PRODUCT_LABELS = list(FIELD_PRODUCTS)
SCHEME_LABELS = list(SCHEMES)


def anomaly_points(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return values - np.nanmean(values[baseline], axis=0)[None, :]


def read_universal_fields() -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    with Dataset(MODEL_FIELDS) as dataset:
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
    with h5py.File(PUBLIC, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        selected_indices = np.flatnonzero((years >= 1951) & (years <= 2000))
        selected_years = years[selected_indices]
        baseline_full = np.flatnonzero((years >= 1961) & (years <= 1990))
        baseline_selected = (selected_years >= 1961) & (selected_years <= 1990)
        if not np.array_equal(selected_years, np.arange(1951, 2001)):
            raise RuntimeError("Multi-target year contract failed")
        if not (
            np.allclose(np.asarray(handle["lat"], dtype=float).ravel(), lat)
            and np.allclose(np.asarray(handle["lon"], dtype=float).ravel(), lon)
        ):
            raise RuntimeError("Multi-target grid contract failed")

        required = [*TARGETS.values(), cru_base.PROJECT_VARIABLE, *external.MEMBERS.values()]
        common = hma_mask.copy()
        for variable in required:
            baseline_values = cru_base.clean(handle[variable][baseline_full, :, :])
            common &= np.isfinite(baseline_values).sum(axis=0) >= 20
        if int(common.sum()) < 100:
            raise RuntimeError(f"Universal common support too small: {int(common.sum())}")

        targets = {}
        for label, variable in TARGETS.items():
            values = cru_base.clean(handle[variable][selected_indices, :, :])[:, common]
            targets[label] = anomaly_points(values, baseline_selected)
        project = anomaly_points(
            cru_base.clean(handle[cru_base.PROJECT_VARIABLE][selected_indices, :, :])[:, common],
            baseline_selected,
        )
        member_points = {
            name: anomaly_points(
                cru_base.clean(handle[variable][selected_indices, :, :])[:, common], baseline_selected
            )
            for name, variable in external.MEMBERS.items()
        }
    local_enkf = cru_base.finite_mean([member_points["HGAO"], member_points["O"]])
    neukom = cru_base.finite_mean([
        values for name, values in member_points.items() if name.startswith("Nk_")
    ])
    products = {
        "project_current": project,
        "local_enkf": local_enkf,
        "lmr": member_points["LMR"],
        "neukom": neukom,
        "phyda": member_points["PHYDA"],
    }
    products["family_equal"] = cru_base.finite_mean([
        products["local_enkf"], products["lmr"], products["neukom"], products["phyda"]
    ])
    cell_latitudes = np.broadcast_to(lat[:, None], common.shape)[common]
    cell_longitudes = np.broadcast_to(lon[None, :], common.shape)[common]
    return selected_years, targets, products, common, cell_latitudes, cell_longitudes


def extract_targets_at_scheme_locations(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
        mappings = []
        eligible_ids = []
        for station in stations.itertuples(index=False):
            distance = np.hypot(
                lat[:, None] - float(station.latitude), lon[None, :] - float(station.longitude)
            )
            distance[~support] = np.inf
            row, col = np.unravel_index(np.argmin(distance), distance.shape)
            target_values = {}
            eligible = True
            for label, variable in TARGETS.items():
                values = cru_base.clean(handle[variable][indices, row, col])
                if int(np.isfinite(values[baseline]).sum()) < 20:
                    eligible = False
                    break
                target_values[label] = values - np.nanmean(values[baseline])
            mappings.append({
                "station_id": station.station_id,
                "station_name": station.station_name,
                "source_latitude": float(lat[row]),
                "source_longitude": float(lon[col]),
                "source_distance_deg": float(distance[row, col]),
                "universal_target_eligible": eligible,
            })
            if not eligible:
                continue
            eligible_ids.append(str(station.station_id))
            for index, year in enumerate(selected_years):
                target_rows.append({
                    "station_id": station.station_id,
                    "year": int(year),
                    **{f"target_{label}_degC": float(values[index]) for label, values in target_values.items()},
                })
    return pd.DataFrame(target_rows), pd.DataFrame(mappings), eligible_ids


def rank_summary(field_table: pd.DataFrame, scheme_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    field = field_table.copy()
    for metric_name, ascending in {
        "hma_mean_rmse_degC": True,
        "hma_mean_mae_degC": True,
        "hma_mean_correlation_r": False,
        "hma_mean_abs_trend_error_degC_per_century": True,
        "median_cell_temporal_correlation_r": False,
        "median_annual_spatial_correlation_r": False,
    }.items():
        field[f"rank_{metric_name}"] = field.groupby("target")[metric_name].rank(
            method="average", ascending=ascending
        )
    field["rank_abs_bias"] = field.assign(abs_bias=field["hma_mean_bias_degC"].abs()).groupby("target")[
        "abs_bias"
    ].rank(method="average")
    field["rank_variance_distance"] = field.assign(
        variance_distance=(field["hma_mean_variance_ratio"] - 1).abs()
    ).groupby("target")["variance_distance"].rank(method="average")
    field_rank_columns = [column for column in field if column.startswith("rank_")]
    field_summary = field.groupby("product")[field_rank_columns].mean()
    field_summary["mean_rank"] = field_summary.mean(axis=1)
    field_summary["target_count"] = field.groupby("product")["target"].nunique()
    field_summary = field_summary.reset_index().sort_values("mean_rank")

    scheme = scheme_table.copy()
    for metric_name, ascending in {
        "annual_pooled_rmse_degC": True,
        "annual_pooled_mae_degC": True,
        "annual_pooled_correlation_r": False,
        "median_abs_trend_error_degC_per_century": True,
        "decadal_rmse_degC": True,
        "decadal_correlation_r": False,
    }.items():
        scheme[f"rank_{metric_name}"] = scheme.groupby("target")[metric_name].rank(
            method="average", ascending=ascending
        )
    scheme["rank_abs_bias"] = scheme.assign(abs_bias=scheme["annual_pooled_bias_degC"].abs()).groupby(
        "target"
    )["abs_bias"].rank(method="average")
    scheme["rank_variance_distance"] = scheme.assign(
        variance_distance=(scheme["annual_pooled_variance_ratio"] - 1).abs()
    ).groupby("target")["variance_distance"].rank(method="average")
    scheme_rank_columns = [column for column in scheme if column.startswith("rank_")]
    scheme_summary = scheme.groupby("branch")[scheme_rank_columns].mean()
    scheme_summary["mean_rank"] = scheme_summary.mean(axis=1)
    scheme_summary["target_count"] = scheme.groupby("branch")["target"].nunique()
    scheme_summary = scheme_summary.reset_index().sort_values("mean_rank")
    return field_summary, scheme_summary


def annotated_heatmap(axis, frame: pd.DataFrame, value: str, title: str, cmap: str, center=None) -> None:
    pivot = frame.pivot(index="target", columns="product", values=value).reindex(
        index=list(TARGETS), columns=PRODUCT_LABELS
    )
    array = pivot.to_numpy(float)
    if center is None:
        image = axis.imshow(array, cmap=cmap, aspect="auto")
    else:
        limit = float(np.nanmax(np.abs(array - center)))
        image = axis.imshow(array, cmap=cmap, aspect="auto", vmin=center - limit, vmax=center + limit)
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_title(title)
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            axis.text(col, row, f"{array[row, col]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=axis, shrink=0.75)


def plot_field_heatmaps(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    panels = [
        ("hma_mean_rmse_degC", "HMA mean RMSE", "YlOrRd", None),
        ("hma_mean_correlation_r", "HMA mean correlation", "YlGn", None),
        ("hma_mean_bias_degC", "HMA mean bias", "RdBu_r", 0.0),
        ("hma_mean_variance_ratio", "HMA mean variance ratio", "RdBu_r", 1.0),
        ("hma_mean_ce", "HMA mean CE", "RdYlGn", 0.0),
        ("hma_mean_abs_trend_error_degC_per_century", "Trend error", "YlOrRd", None),
    ]
    for axis, panel in zip(axes.ravel(), panels):
        annotated_heatmap(axis, frame, *panel)
    figure.suptitle("Reconstructions under five instrumental gridded targets", fontsize=15)
    figure.savefig(OUT / "multi_target_hma_field_metrics.png", dpi=220)
    plt.close(figure)


def plot_scheme_heatmaps(frame: pd.DataFrame) -> None:
    renamed = frame.rename(columns={"branch": "product"})
    global PRODUCT_LABELS
    original = PRODUCT_LABELS
    PRODUCT_LABELS = SCHEME_LABELS
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    panels = [
        ("annual_pooled_rmse_degC", "Annual RMSE", "YlOrRd", None),
        ("annual_pooled_correlation_r", "Annual correlation", "YlGn", None),
        ("annual_pooled_bias_degC", "Annual bias", "RdBu_r", 0.0),
        ("annual_pooled_variance_ratio", "Annual variance ratio", "RdBu_r", 1.0),
        ("annual_pooled_ce", "Annual CE", "RdYlGn", 0.0),
        ("decadal_correlation_r", "Decadal correlation", "YlGn", None),
    ]
    for axis, panel in zip(axes.ravel(), panels):
        annotated_heatmap(axis, renamed, *panel)
    figure.suptitle("Four background schemes under five instrumental gridded targets", fontsize=15)
    figure.savefig(OUT / "multi_target_four_scheme_metrics.png", dpi=220)
    plt.close(figure)
    PRODUCT_LABELS = original


def plot_curves(
    years: np.ndarray,
    targets: dict[str, np.ndarray],
    products: dict[str, np.ndarray],
    latitudes: np.ndarray,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    project = cru_base.area_mean(products["project_current"], latitudes)
    ref_rec = cru_base.area_mean(products["family_equal"], latitudes)
    neukom = cru_base.area_mean(products["neukom"], latitudes)
    for axis, (target, values) in zip(axes.ravel(), targets.items()):
        axis.plot(years, cru_base.area_mean(values, latitudes), color="#111111", lw=2.5, label=target)
        axis.plot(years, project, color="#1f4e79", lw=1.8, label="Project")
        axis.plot(years, ref_rec, color="#666666", lw=1.8, label="ref_rec")
        axis.plot(years, neukom, color="#548235", lw=1.5, label="Neukom")
        axis.axhline(0, color="#999999", lw=0.7)
        axis.set(title=f"{target} as target", ylabel="Anomaly (degC)")
        axis.legend(frameon=False, ncol=2)
    axes.ravel()[-1].axis("off")
    figure.suptitle("HMA mean curves on universal common support", fontsize=15)
    figure.savefig(OUT / "multi_target_hma_mean_curves.png", dpi=220)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    before = {name: cru_base.state(path) for name, path in {
        "public": PUBLIC, "model_fields": MODEL_FIELDS, "scheme_values": SCHEME_VALUES
    }.items()}
    years, targets, products, common, cell_latitudes, cell_longitudes = read_universal_fields()
    field_rows = []
    for target_label, target in targets.items():
        for product_label, product_key in FIELD_PRODUCTS.items():
            row = cru_base.field_metrics(
                years, target, products[product_key], cell_latitudes, product_label
            )
            row["target"] = target_label
            field_rows.append(row)
    field_table = pd.DataFrame(field_rows)

    scheme_frame = pd.read_csv(SCHEME_VALUES)
    target_station, mapping, eligible_ids = extract_targets_at_scheme_locations(scheme_frame)
    if len(eligible_ids) < 3:
        raise RuntimeError(f"Too few universal target stations: {len(eligible_ids)}")
    scheme_frame = scheme_frame[scheme_frame["station_id"].astype(str).isin(eligible_ids)].merge(
        target_station, on=["station_id", "year"], validate="one_to_one"
    )
    scheme_rows = []
    for target_label in TARGETS:
        target_column = f"target_{target_label}_degC"
        for branch, column in SCHEMES.items():
            station = metrics.station_metrics(scheme_frame, target_column, branch, column)
            decades = metrics.decadal_pairs(scheme_frame, target_column, column)
            row = metrics.aggregate_metrics(
                scheme_frame, station, decades, target_column, branch, column
            )
            row["target"] = target_label
            scheme_rows.append(row)
    scheme_table = pd.DataFrame(scheme_rows)
    field_rank, scheme_rank = rank_summary(field_table, scheme_table)

    field_table.to_csv(OUT / "multi_target_hma_field_metrics.csv", index=False)
    scheme_table.to_csv(OUT / "multi_target_four_scheme_metrics.csv", index=False)
    field_rank.to_csv(OUT / "field_product_cross_target_ranks.csv", index=False)
    scheme_rank.to_csv(OUT / "scheme_cross_target_ranks.csv", index=False)
    mapping.to_csv(OUT / "scheme_station_universal_target_mapping.csv", index=False)
    pd.DataFrame({"cell_latitude": cell_latitudes, "cell_longitude": cell_longitudes}).to_csv(
        OUT / "universal_common_hma_cells.csv", index=False
    )
    curve_rows = []
    for index, year in enumerate(years):
        row = {"year": int(year)}
        for target, values in targets.items():
            row[f"target_{target}"] = cru_base.area_mean(values, cell_latitudes)[index]
        for product, key in FIELD_PRODUCTS.items():
            row[f"product_{product}"] = cru_base.area_mean(products[key], cell_latitudes)[index]
        curve_rows.append(row)
    pd.DataFrame(curve_rows).to_csv(OUT / "multi_target_hma_mean_curves.csv", index=False)
    plot_field_heatmaps(field_table)
    plot_scheme_heatmaps(scheme_table)
    plot_curves(years, targets, products, cell_latitudes)

    after = {name: cru_base.state(Path(item["path"])) for name, item in before.items()}
    if before != after:
        raise RuntimeError("A read-only source changed during multi-target analysis")
    project_rows = field_table[field_table["product"].eq("project_current")]
    current_scheme_rows = scheme_table[scheme_table["branch"].eq("current_CESM")]
    result = {
        "experiment": OUT.name,
        "status": "complete",
        "targets": list(TARGETS),
        "period": [1951, 2000],
        "baseline": [1961, 1990],
        "universal_common_hma_cell_count": int(common.sum()),
        "universal_scheme_station_count": int(len(eligible_ids)),
        "field_cross_target_ranks": field_rank.to_dict(orient="records"),
        "scheme_cross_target_ranks": scheme_rank.to_dict(orient="records"),
        "project_positive_bias_target_count": int((project_rows["hma_mean_bias_degC"] > 0).sum()),
        "project_excess_variance_target_count": int((project_rows["hma_mean_variance_ratio"] > 1).sum()),
        "current_scheme_best_rmse_target_count": int(sum(
            row.annual_pooled_rmse_degC
            == scheme_table[scheme_table["target"].eq(row.target)]["annual_pooled_rmse_degC"].min()
            for row in current_scheme_rows.itertuples(index=False)
        )),
        "source_states_before": before,
        "source_states_after": after,
        "source_unchanged": True,
        "locked_period_access": "none; values read only for 1951-2000",
        "interpretation_boundary": (
            "Targets share station sources and are not independent votes. Cross-target consistency diagnoses "
            "sensitivity to observational gridding and homogenization."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "EXPERIMENT_CONTRACT.md").write_text(
        "# Experiment contract\n\n"
        "- Repeat the frozen CRU-target method for CRU, Berkeley Earth, GISTEMP, HadCRUT5, and NOAA.\n"
        "- Use 1951-2000, product-native 1961-1990 anomalies, and universal common HMA support.\n"
        "- Compare four schemes at the same universally eligible frozen locations.\n"
        "- Treat products as correlated alternative targets, not independent evidence votes.\n"
        "- Do not read 1921-1939, modify production files, run EnKF, or run OGGM.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

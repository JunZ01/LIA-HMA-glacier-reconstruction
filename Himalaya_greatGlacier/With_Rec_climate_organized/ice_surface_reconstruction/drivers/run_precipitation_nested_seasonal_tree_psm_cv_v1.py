"""Nested blocked validation of seasonal tree-proxy PSMs over 1941-2000."""

from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_ID = "climate_reconstruction_precipitation_nested_seasonal_tree_psm_cv_v1"
YEARS = np.arange(1941, 2001, dtype=int)
REFERENCE = (1961, 1990)
SEASONS = {
    "annual": tuple(range(1, 13)),
    "DJF": (12, 1, 2),
    "JJA": (6, 7, 8),
    "MAMJJA": (3, 4, 5, 6, 7, 8),
    "JJASON": (6, 7, 8, 9, 10, 11),
}
MIN_FIT_COUNT = 30
LOCKED = (1921, 1939)
CRU_ROOT = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\input-recon\cruts.2004151855.v4.04")
CRU_TEMPERATURE = CRU_ROOT / "tmp/cru_ts4.04.1901.2019.tmp.dat.nc"
CRU_PRECIPITATION = CRU_ROOT / "pre/cru_ts4.04.1901.2019.pre.dat.nc"
GPCC_PRECIPITATION = Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc")
BERKELEY_TEMPERATURE = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\input-recon\Land_and_Ocean_LatLong1_1850_2016.nc")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_proxy_path() -> Path:
    archive = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive / paper / "work1" / "compared_result_all/python_data/proxy_data_6.mat"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_state(path: Path, include_hash: bool) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        result["sha256"] = sha256(path)
    return result


def monthly_slice(source_start_year: int) -> slice:
    # December 1940 through December 2000. This never touches 1921-1939.
    start = (1940 - source_start_year) * 12 + 11
    stop = (2000 - source_start_year + 1) * 12
    return slice(start, stop)


def monthly_labels() -> list[tuple[int, int]]:
    return [(1940, 12), *((year, month) for year in YEARS for month in range(1, 13))]


def nearest_indices(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmin(np.abs(source - value))) for value in targets], dtype=int)


def read_netcdf_points(
    path: Path,
    variable: str,
    source_start_year: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = 721
    output = np.full((expected, len(latitudes)), np.nan)
    with netcdf_file(path, "r", mmap=True) as handle:
        source_lat = np.array(handle.variables["lat"][:], dtype=float, copy=True)
        source_lon = np.array(handle.variables["lon"][:], dtype=float, copy=True)
        lat_index = nearest_indices(source_lat, latitudes)
        lon_index = nearest_indices(source_lon, longitudes)
        source_variable = handle.variables[variable]
        fill = float(source_variable._attributes.get("_FillValue", -99999.99))
        for column, (iy, ix) in enumerate(zip(lat_index, lon_index)):
            values = np.array(source_variable[monthly_slice(source_start_year), iy, ix], dtype=float, copy=True)
            if len(values) != expected:
                raise RuntimeError(f"monthly-length-conflict:{path.name}:{len(values)}")
            values[(values == fill) | (values <= -9000) | (values >= 1e20)] = np.nan
            output[:, column] = values
    return output, source_lat[lat_index], source_lon[lon_index]


def read_berkeley_points(latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = 721
    output = np.full((expected, len(latitudes)), np.nan)
    with h5py.File(BERKELEY_TEMPERATURE, "r") as handle:
        source_lat = np.asarray(handle["latitude"], dtype=float)
        source_lon = np.asarray(handle["longitude"], dtype=float)
        lat_index = nearest_indices(source_lat, latitudes)
        lon_index = nearest_indices(source_lon, longitudes)
        for column, (iy, ix) in enumerate(zip(lat_index, lon_index)):
            values = np.asarray(handle["temperature"][monthly_slice(1850), iy, ix], dtype=float)
            if len(values) != expected:
                raise RuntimeError(f"berkeley-monthly-length-conflict:{len(values)}")
            values[(values <= -9000) | (values >= 1e20)] = np.nan
            output[:, column] = values
    return output, source_lat[lat_index], source_lon[lon_index]


def aggregate_temperature(values: np.ndarray, months: tuple[int, ...]) -> np.ndarray:
    labels = monthly_labels()
    lookup = {label: index for index, label in enumerate(labels)}
    result = np.full((len(YEARS), values.shape[1]), np.nan)
    cross_year = months[0] > months[-1]
    for row, year in enumerate(YEARS):
        requested = [lookup[(int(year - 1) if cross_year and month >= months[0] else int(year), month)] for month in months]
        selected = values[requested]
        complete = np.all(np.isfinite(selected), axis=0)
        result[row, complete] = np.mean(selected[:, complete], axis=0)
    return result


def aggregate_precipitation(values: np.ndarray, months: tuple[int, ...]) -> np.ndarray:
    labels = monthly_labels()
    lookup = {label: index for index, label in enumerate(labels)}
    result = np.full((len(YEARS), values.shape[1]), np.nan)
    cross_year = months[0] > months[-1]
    for row, year in enumerate(YEARS):
        requested = []
        days = []
        for month in months:
            source_year = int(year - 1) if cross_year and month >= months[0] else int(year)
            requested.append(lookup[(source_year, month)])
            days.append(calendar.monthrange(source_year, month)[1])
        selected = values[requested]
        complete = np.all(np.isfinite(selected), axis=0)
        result[row, complete] = np.sum(selected[:, complete], axis=0) / float(np.sum(days))
    return result


def center_reference(values: np.ndarray) -> np.ndarray:
    reference = (YEARS >= REFERENCE[0]) & (YEARS <= REFERENCE[1])
    return values - np.nanmean(values[reference], axis=0, keepdims=True)


def fit_ols(response: np.ndarray, temperature: np.ndarray, precipitation: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(response) & np.isfinite(temperature) & np.isfinite(precipitation)
    if int(valid.sum()) < MIN_FIT_COUNT:
        return None
    design = np.column_stack([np.ones(valid.sum()), temperature[valid], precipitation[valid]])
    return np.linalg.lstsq(design, response[valid], rcond=None)[0]


def predict(coefficients: np.ndarray | None, temperature: np.ndarray, precipitation: np.ndarray) -> np.ndarray:
    result = np.full(len(temperature), np.nan)
    valid = np.isfinite(temperature) & np.isfinite(precipitation)
    if coefficients is not None:
        result[valid] = coefficients[0] + coefficients[1] * temperature[valid] + coefficients[2] * precipitation[valid]
    return result


def nrmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if int(valid.sum()) < 30 or np.std(observed[valid]) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(observed[valid] - predicted[valid]))) / np.std(observed[valid]))


def correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if int(valid.sum()) < 3 or np.std(observed[valid]) == 0 or np.std(predicted[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(observed[valid], predicted[valid])[0, 1])


def inner_score(
    response: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    outer_train: np.ndarray,
) -> float:
    predictions = np.full(len(YEARS), np.nan)
    blocks = (YEARS - YEARS[0]) // 10
    for block in np.unique(blocks[outer_train]):
        test = outer_train & (blocks == block)
        train = outer_train & (blocks != block)
        coefficients = fit_ols(response[train], temperature[train], precipitation[train])
        predictions[test] = predict(coefficients, temperature[test], precipitation[test])
    return nrmse(response[outer_train], predictions[outer_train])


def nested_proxy(
    proxy_index: int,
    response: np.ndarray,
    climates: dict[str, dict[str, dict[str, np.ndarray]]],
    column: int,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    blocks = (YEARS - YEARS[0]) // 10
    predictions = {
        name: np.full(len(YEARS), np.nan)
        for name in (
            "bea_selected", "bea_annual", "cru_selected", "cru_annual",
            "bea_to_cru_selected", "bea_to_cru_annual", "cru_to_bea_selected", "cru_to_bea_annual",
        )
    }
    fold_rows: list[dict[str, object]] = []
    season_order = list(SEASONS)
    for outer_block in np.unique(blocks):
        test = blocks == outer_block
        train = ~test
        season_scores = []
        for season in season_order:
            bea_score = inner_score(response, climates["BEA_GPCC"]["temperature"][season][:, column], climates["BEA_GPCC"]["precipitation"][season][:, column], train)
            cru_score = inner_score(response, climates["CRU"]["temperature"][season][:, column], climates["CRU"]["precipitation"][season][:, column], train)
            combined = float(np.nanmean([bea_score, cru_score])) if np.isfinite(bea_score) and np.isfinite(cru_score) else float("nan")
            season_scores.append((season, bea_score, cru_score, combined))
        valid_scores = [item for item in season_scores if np.isfinite(item[3])]
        if not valid_scores:
            continue
        selected, selected_bea_inner, selected_cru_inner, selected_combined = min(valid_scores, key=lambda item: (item[3], season_order.index(item[0])))
        bea_selected_independent = min(valid_scores, key=lambda item: (item[1], season_order.index(item[0])))[0]
        cru_selected_independent = min(valid_scores, key=lambda item: (item[2], season_order.index(item[0])))[0]
        for branch, prefix, alternate in (("BEA_GPCC", "bea", "CRU"), ("CRU", "cru", "BEA_GPCC")):
            for season, suffix in ((selected, "selected"), ("annual", "annual")):
                coefficients = fit_ols(
                    response[train],
                    climates[branch]["temperature"][season][train, column],
                    climates[branch]["precipitation"][season][train, column],
                )
                predictions[f"{prefix}_{suffix}"][test] = predict(
                    coefficients,
                    climates[branch]["temperature"][season][test, column],
                    climates[branch]["precipitation"][season][test, column],
                )
                alternate_prefix = "bea_to_cru" if branch == "BEA_GPCC" else "cru_to_bea"
                predictions[f"{alternate_prefix}_{suffix}"][test] = predict(
                    coefficients,
                    climates[alternate]["temperature"][season][test, column],
                    climates[alternate]["precipitation"][season][test, column],
                )
        fold_rows.append({
            "proxy_index": proxy_index,
            "outer_fold": int(outer_block + 1),
            "outer_test_start": int(YEARS[test][0]),
            "outer_test_end": int(YEARS[test][-1]),
            "selected_season_joint": selected,
            "selected_season_bea_independent": bea_selected_independent,
            "selected_season_cru_independent": cru_selected_independent,
            "independent_product_season_agreement": bea_selected_independent == cru_selected_independent,
            "selected_inner_nrmse_bea": selected_bea_inner,
            "selected_inner_nrmse_cru": selected_cru_inner,
            "selected_inner_nrmse_mean": selected_combined,
        })
    selections = [row["selected_season_joint"] for row in fold_rows]
    mode = pd.Series(selections).value_counts().index[0] if selections else "unresolved"
    stability = float(np.mean(np.asarray(selections) == mode)) if selections else float("nan")
    metrics = {name: nrmse(response, values) for name, values in predictions.items()}
    result: dict[str, object] = {
        "proxy_index": proxy_index,
        "valid_response_year_count": int(np.isfinite(response).sum()),
        "outer_prediction_count": int(np.isfinite(predictions["bea_selected"]).sum()),
        "selected_season_mode": mode,
        "selected_season_mode_fraction": stability,
        "independent_product_season_agreement_fraction": float(np.mean([row["independent_product_season_agreement"] for row in fold_rows])) if fold_rows else float("nan"),
    }
    for name, value in metrics.items():
        result[f"nested_nrmse_{name}"] = value
    result["selected_minus_annual_nrmse_bea"] = metrics["bea_selected"] - metrics["bea_annual"]
    result["selected_minus_annual_nrmse_cru"] = metrics["cru_selected"] - metrics["cru_annual"]
    result["selected_minus_annual_nrmse_cross_mean"] = float(np.nanmean([
        metrics["bea_to_cru_selected"] - metrics["bea_to_cru_annual"],
        metrics["cru_to_bea_selected"] - metrics["cru_to_bea_annual"],
    ]))
    result["selected_improves_both_self_products"] = bool(
        result["selected_minus_annual_nrmse_bea"] < 0 and result["selected_minus_annual_nrmse_cru"] < 0
    )
    result["selected_improves_self_and_cross"] = bool(
        result["selected_improves_both_self_products"] and result["selected_minus_annual_nrmse_cross_mean"] < 0
    )
    result["nested_correlation_bea_selected"] = correlation(response, predictions["bea_selected"])
    result["nested_correlation_bea_annual"] = correlation(response, predictions["bea_annual"])
    result["nested_correlation_cru_selected"] = correlation(response, predictions["cru_selected"])
    result["nested_correlation_cru_annual"] = correlation(response, predictions["cru_annual"])
    return result, fold_rows, predictions


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "proxy6": archive_proxy_path(),
        "proxy6_metadata": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "berkeley_temperature": BERKELEY_TEMPERATURE,
        "gpcc_precipitation": GPCC_PRECIPITATION,
        "cru_temperature": CRU_TEMPERATURE,
        "cru_precipitation": CRU_PRECIPITATION,
        "production_psm_provenance": result_root / "climate_reconstruction_precipitation_production_psm_season_provenance_v2/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if np.any((YEARS >= LOCKED[0]) & (YEARS <= LOCKED[1])):
        raise RuntimeError("locked-period-requested")
    before = {name: source_state(path, include_hash=name not in {"berkeley_temperature", "gpcc_precipitation", "cru_temperature", "cru_precipitation", "proxy6"}) for name, path in sources.items()}

    metadata = pd.read_csv(sources["proxy6_metadata"])
    tree = metadata[(metadata["proxy_type"] == "tree") & (metadata["coefficient_count"] == 3)].copy().sort_values("proxy_index")
    if len(tree) != 201:
        raise RuntimeError(f"tree-proxy-contract-conflict:{len(tree)}")
    latitudes = tree["latitude"].to_numpy(dtype=float)
    longitudes = tree["longitude"].to_numpy(dtype=float)
    response = np.full((len(YEARS), len(tree)), np.nan)
    with h5py.File(sources["proxy6"], "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(tree["proxy_index"].to_numpy(dtype=int)):
            proxy = handle[refs[proxy_index - 1]]
            ages = np.asarray(proxy["age_yng"]).reshape(-1, order="F")
            values = np.asarray(proxy["data"]).reshape(-1, order="F")
            lookup = {int(age): float(value) for age, value in zip(ages, values) if np.isfinite(age) and np.isfinite(value)}
            response[:, column] = [lookup.get(int(year), np.nan) for year in YEARS]

    bea_monthly, bea_lat, bea_lon = read_berkeley_points(latitudes, longitudes)
    gpcc_monthly, gpcc_lat, gpcc_lon = read_netcdf_points(GPCC_PRECIPITATION, "precip", 1891, latitudes, longitudes)
    cru_t_monthly, cru_t_lat, cru_t_lon = read_netcdf_points(CRU_TEMPERATURE, "tmp", 1901, latitudes, longitudes)
    cru_p_monthly, cru_p_lat, cru_p_lon = read_netcdf_points(CRU_PRECIPITATION, "pre", 1901, latitudes, longitudes)
    grid_rows = []
    for column, proxy_index in enumerate(tree["proxy_index"].to_numpy(dtype=int)):
        grid_rows.append({
            "proxy_index": proxy_index,
            "proxy_latitude": latitudes[column],
            "proxy_longitude": longitudes[column],
            "berkeley_grid_latitude": bea_lat[column],
            "berkeley_grid_longitude": bea_lon[column],
            "gpcc_grid_latitude": gpcc_lat[column],
            "gpcc_grid_longitude": gpcc_lon[column],
            "cru_temperature_grid_latitude": cru_t_lat[column],
            "cru_temperature_grid_longitude": cru_t_lon[column],
            "cru_precipitation_grid_latitude": cru_p_lat[column],
            "cru_precipitation_grid_longitude": cru_p_lon[column],
        })
    climates = {
        "BEA_GPCC": {"temperature": {}, "precipitation": {}},
        "CRU": {"temperature": {}, "precipitation": {}},
    }
    for season, months in SEASONS.items():
        climates["BEA_GPCC"]["temperature"][season] = center_reference(aggregate_temperature(bea_monthly, months))
        climates["BEA_GPCC"]["precipitation"][season] = center_reference(aggregate_precipitation(gpcc_monthly, months))
        climates["CRU"]["temperature"][season] = center_reference(aggregate_temperature(cru_t_monthly, months))
        climates["CRU"]["precipitation"][season] = center_reference(aggregate_precipitation(cru_p_monthly, months))

    proxy_rows = []
    fold_rows = []
    prediction_store: dict[str, np.ndarray] = {
        name: np.full_like(response, np.nan)
        for name in (
            "bea_selected", "bea_annual", "cru_selected", "cru_annual",
            "bea_to_cru_selected", "bea_to_cru_annual", "cru_to_bea_selected", "cru_to_bea_annual",
        )
    }
    for column, proxy_index in enumerate(tree["proxy_index"].to_numpy(dtype=int)):
        result, folds, predictions = nested_proxy(proxy_index, response[:, column], climates, column)
        proxy_rows.append(result)
        fold_rows.extend(folds)
        for name, values in predictions.items():
            prediction_store[name][:, column] = values
    proxy_results = tree.merge(pd.DataFrame(proxy_rows), on="proxy_index", validate="one_to_one")
    folds = pd.DataFrame(fold_rows)
    valid = proxy_results[proxy_results["outer_prediction_count"] >= 30].copy()
    if len(valid) < 150:
        raise RuntimeError(f"insufficient-nested-validation-proxies:{len(valid)}")

    network_rows = []
    for name, values in prediction_store.items():
        proxy_metric = [nrmse(response[:, column], values[:, column]) for column in range(response.shape[1])]
        network_rows.append({
            "prediction": name,
            "valid_proxy_count": int(np.isfinite(proxy_metric).sum()),
            "median_proxy_nrmse": float(np.nanmedian(proxy_metric)),
            "mean_proxy_nrmse": float(np.nanmean(proxy_metric)),
            "network_curve_correlation": correlation(np.nanmean(response, axis=1), np.nanmean(values, axis=1)),
        })
    network = pd.DataFrame(network_rows)

    gates = {
        "valid_proxy_count_at_least_150": len(valid) >= 150,
        "median_bea_selected_improvement_at_least_0_02": float(valid["selected_minus_annual_nrmse_bea"].median()) <= -0.02,
        "median_cru_selected_improvement_at_least_0_02": float(valid["selected_minus_annual_nrmse_cru"].median()) <= -0.02,
        "fraction_improves_both_self_products_at_least_half": float(valid["selected_improves_both_self_products"].mean()) >= 0.50,
        "median_cross_product_change_nonpositive": float(valid["selected_minus_annual_nrmse_cross_mean"].median()) <= 0,
        "median_mode_fraction_at_least_half": float(valid["selected_season_mode_fraction"].median()) >= 0.50,
    }
    passes = bool(all(gates.values()))

    annual_curves = pd.DataFrame({"year": YEARS})
    proxy_mean = np.nanmean(response, axis=0)
    proxy_scale = np.nanstd(response, axis=0)
    proxy_scale[proxy_scale == 0] = np.nan
    annual_curves["proxy_network_z"] = np.nanmean((response - proxy_mean[None, :]) / proxy_scale[None, :], axis=1)
    for name in ("bea_selected", "bea_annual", "cru_selected", "cru_annual"):
        annual_curves[name + "_network_z"] = np.nanmean((prediction_store[name] - proxy_mean[None, :]) / proxy_scale[None, :], axis=1)

    output.mkdir(parents=True)
    proxy_results.to_csv(output / "nested_seasonal_proxy_scores.csv", index=False)
    folds.to_csv(output / "nested_seasonal_outer_fold_choices.csv", index=False)
    network.to_csv(output / "nested_seasonal_network_scores.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(output / "proxy_climate_grid_crosswalk.csv", index=False)
    annual_curves.to_csv(output / "nested_oof_network_curves_1941_2000.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), constrained_layout=True)
    axes[0, 0].plot(YEARS, annual_curves["proxy_network_z"], color="#111111", linewidth=2.3, label="Tree proxies")
    axes[0, 0].plot(YEARS, annual_curves["bea_annual_network_z"], color="#0072B2", linewidth=1.2, alpha=0.8, label="BEA+GPCC annual")
    axes[0, 0].plot(YEARS, annual_curves["bea_selected_network_z"], color="#009E73", linewidth=1.8, label="BEA+GPCC nested season")
    axes[0, 0].plot(YEARS, annual_curves["cru_selected_network_z"], color="#D55E00", linewidth=1.4, label="CRU nested season")
    axes[0, 0].set_title("Nested out-of-fold network curves")
    axes[0, 0].set_ylabel("Network mean proxy z")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    axes[0, 1].scatter(valid["nested_nrmse_bea_annual"], valid["nested_nrmse_bea_selected"], color="#009E73", s=25, alpha=0.7)
    limit = float(np.nanmax(valid[["nested_nrmse_bea_annual", "nested_nrmse_bea_selected"]].to_numpy())) * 1.02
    axes[0, 1].plot([0, limit], [0, limit], color="#777777", linewidth=0.8)
    axes[0, 1].set_xlim(0, limit); axes[0, 1].set_ylim(0, limit)
    axes[0, 1].set_xlabel("Annual nested NRMSE")
    axes[0, 1].set_ylabel("Season-selected nested NRMSE")
    axes[0, 1].set_title("BEA+GPCC per-proxy skill")

    season_counts = valid["selected_season_mode"].value_counts()
    axes[1, 0].bar(list(SEASONS), [season_counts.get(name, 0) for name in SEASONS], color=["#666666", "#56B4E9", "#E69F00", "#009E73", "#CC79A7"])
    axes[1, 0].set_ylabel("Proxy count")
    axes[1, 0].set_title("Modal season across six outer folds")

    delta_bea = valid["selected_minus_annual_nrmse_bea"].to_numpy(dtype=float)
    delta_cru = valid["selected_minus_annual_nrmse_cru"].to_numpy(dtype=float)
    axes[1, 1].scatter(delta_bea, delta_cru, color="#0072B2", s=27, alpha=0.7)
    axes[1, 1].axhline(0, color="#777777", linewidth=0.8); axes[1, 1].axvline(0, color="#777777", linewidth=0.8)
    axes[1, 1].set_xlabel("Selected - annual NRMSE, BEA+GPCC")
    axes[1, 1].set_ylabel("Selected - annual NRMSE, CRU")
    axes[1, 1].set_title("Cross-product direction of improvement")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "nested_seasonal_tree_psm_validation.png", dpi=220)
    plt.close(figure)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "analysis_period": [1941, 2000],
        "locked_1921_1939_accessed": False,
        "candidate_seasons": {name: list(months) for name, months in SEASONS.items()},
        "validation_design": "Six outer 10-year blocks; season selected by inner blocked CV on the other 50 years; annual and selected predictions scored only on outer-held-out years.",
        "tree_proxy_count": int(len(tree)),
        "valid_nested_proxy_count": int(len(valid)),
        "median_selected_minus_annual_nrmse_bea": float(valid["selected_minus_annual_nrmse_bea"].median()),
        "median_selected_minus_annual_nrmse_cru": float(valid["selected_minus_annual_nrmse_cru"].median()),
        "median_selected_minus_annual_nrmse_cross_mean": float(valid["selected_minus_annual_nrmse_cross_mean"].median()),
        "fraction_selected_improves_both_self_products": float(valid["selected_improves_both_self_products"].mean()),
        "fraction_selected_improves_self_and_cross": float(valid["selected_improves_self_and_cross"].mean()),
        "median_season_mode_fraction": float(valid["selected_season_mode_fraction"].median()),
        "median_independent_product_season_agreement_fraction": float(valid["independent_product_season_agreement_fraction"].median()),
        "modal_season_counts": {str(key): int(value) for key, value in valid["selected_season_mode"].value_counts().to_dict().items()},
        "network_scores": network.to_dict(orient="records"),
        "gates": gates,
        "passes_seasonal_psm_gate": passes,
        "decision": "advance_to_isolated_seasonal_operator_reconstruction" if passes else "close_proxy_specific_season_selection_as_production_repair",
        "interpretation_boundary": "This validates proxy prediction over 1941-2000 only. It does not use MADA or 1901-1920, alter the production PSM, or authorize an EnKF production replacement.",
    }
    after = {name: source_state(path, include_hash=name not in {"berkeley_temperature", "gpcc_precipitation", "cru_temperature", "cru_precipitation", "proxy6"}) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Nested Seasonal Tree-PSM Contract\n\n"
        "Use only 1941-2000. Candidate seasons are frozen as annual, DJF, JJA, MAMJJA, and JJASON. Select seasons inside each outer fold and score only outer-held-out predictions. Do not access 1921-1939, use MADA for selection, modify production files, or run OGGM.\n",
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Nested Seasonal Tree-PSM Validation\n\n"
        f"- Valid tree proxies: `{len(valid)}/{len(tree)}`.\n"
        f"- Median selected-minus-annual NRMSE: BEA+GPCC `{summary['median_selected_minus_annual_nrmse_bea']:+.4f}`, CRU `{summary['median_selected_minus_annual_nrmse_cru']:+.4f}`, cross-product mean `{summary['median_selected_minus_annual_nrmse_cross_mean']:+.4f}`.\n"
        f"- Fraction improving both self-product paths: `{summary['fraction_selected_improves_both_self_products']:.1%}`; self and cross paths: `{summary['fraction_selected_improves_self_and_cross']:.1%}`.\n"
        f"- Modal seasons: `{summary['modal_season_counts']}`.\n"
        f"- Gate pass: `{str(passes).lower()}`; decision: `{summary['decision']}`.\n\n"
        "No 1901-1920 or 1921-1939 values were read. MADA was not used for fitting or selection.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-nested-seasonal-validation")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

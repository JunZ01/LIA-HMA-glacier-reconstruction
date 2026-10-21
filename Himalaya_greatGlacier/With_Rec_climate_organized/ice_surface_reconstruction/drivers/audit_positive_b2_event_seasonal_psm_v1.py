"""Test whether calibration-selected seasonal PSMs explain the 1915-1918 event."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, netcdf_file


EVENT_YEARS = np.arange(1905, 1921)
CALIBRATION_YEARS = np.arange(1940, 2001)
BASELINE = (1905, 1914)
EVENT = (1915, 1918)
SEASONS = {
    "annual": tuple(range(1, 13)),
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
    "AMJJAS": (4, 5, 6, 7, 8, 9),
}
RNG_SEED = 20260814
PERMUTATIONS = 10000
CRU_ROOT = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\input-recon\cruts.2004151855.v4.04")
CRU_TEMPERATURE = CRU_ROOT / "tmp/cru_ts4.04.1901.2019.tmp.dat.nc"
CRU_PRECIPITATION = CRU_ROOT / "pre/cru_ts4.04.1901.2019.pre.dat.nc"
GPCC_PRECIPITATION = Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc")
BERKELEY_TEMPERATURE = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\input-recon\Land_and_Ocean_LatLong1_1850_2016.nc")
PAGES_METADATA = Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\pages2k_dataset_p.mat")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_proxy_path() -> Path:
    archive = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive / paper / "work1" / "compared_result_all" / "python_data" / "proxy_data_6.mat"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def permutation_correlation(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    observed = correlation(left, right)
    if not np.isfinite(observed):
        return observed, float("nan")
    null = np.empty(PERMUTATIONS)
    for index in range(PERMUTATIONS):
        null[index] = correlation(left, rng.permutation(right))
    return observed, float((1 + np.sum(np.abs(null) >= abs(observed))) / (PERMUTATIONS + 1))


def time_slice(start_year: int, first_year: int, last_year: int) -> slice:
    start = (first_year - start_year) * 12 + 11
    stop = (last_year - start_year + 1) * 12
    return slice(start, stop)


def monthly_labels(first_year: int, last_year: int) -> list[tuple[int, int]]:
    labels = [(first_year - 1, 12)]
    labels.extend((year, month) for year in range(first_year, last_year + 1) for month in range(1, 13))
    return labels


def read_netcdf_points(
    path: Path,
    variable_names: list[str],
    source_start_year: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    periods = {
        "event": (time_slice(source_start_year, 1905, 1920), 193),
        "calibration": (time_slice(source_start_year, 1940, 2000), 733),
    }
    output = {period: {name: np.full((length, len(latitudes)), np.nan) for name in variable_names} for period, (_, length) in periods.items()}
    with netcdf_file(path, "r", mmap=True) as handle:
        source_lat = np.array(handle.variables["lat"][:], dtype=float, copy=True)
        source_lon = np.array(handle.variables["lon"][:], dtype=float, copy=True)
        lat_indices = np.asarray([int(np.argmin(np.abs(source_lat - value))) for value in latitudes])
        lon_indices = np.asarray([int(np.argmin(np.abs(source_lon - value))) for value in longitudes])
        for period, (selection, expected_length) in periods.items():
            for column, (lat_index, lon_index) in enumerate(zip(lat_indices, lon_indices)):
                for name in variable_names:
                    values = np.array(handle.variables[name][selection, lat_index, lon_index], dtype=float, copy=True)
                    if len(values) != expected_length:
                        raise RuntimeError(f"monthly-slice-length-failed:{path.name}:{period}:{name}")
                    values[(values <= -9000) | (values >= 1e20)] = np.nan
                    output[period][name][:, column] = values
    return output, source_lat[lat_indices], source_lon[lon_indices]


def read_berkeley_points(latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, np.ndarray]:
    periods = {
        "event": (time_slice(1850, 1905, 1920), 193),
        "calibration": (time_slice(1850, 1940, 2000), 733),
    }
    output = {period: np.full((length, len(latitudes)), np.nan) for period, (_, length) in periods.items()}
    with h5py.File(BERKELEY_TEMPERATURE, "r") as handle:
        source_lat = np.asarray(handle["latitude"], dtype=float)
        source_lon = np.asarray(handle["longitude"], dtype=float)
        time = np.asarray(handle["time"], dtype=float)
        if not (abs(time[0] - 1850.0416666667) < 1e-6 and abs(time[-1] - 2016.5416666667) < 1e-6):
            raise RuntimeError("berkeley-time-axis-contract-failed")
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


def seasonal_aggregate(values: np.ndarray, labels: list[tuple[int, int]], years: np.ndarray, months: tuple[int, ...]) -> np.ndarray:
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


def fit_ols(response: np.ndarray, temperature: np.ndarray, precipitation: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(response) & np.isfinite(temperature) & np.isfinite(precipitation)
    if int(valid.sum()) < 30:
        return None
    design = np.column_stack([np.ones(valid.sum()), temperature[valid], precipitation[valid]])
    return np.linalg.lstsq(design, response[valid], rcond=None)[0]


def predict(coefficients: np.ndarray | None, temperature: np.ndarray, precipitation: np.ndarray) -> np.ndarray:
    if coefficients is None:
        return np.full(len(temperature), np.nan)
    values = coefficients[0] + coefficients[1] * temperature + coefficients[2] * precipitation
    values[~(np.isfinite(temperature) & np.isfinite(precipitation))] = np.nan
    return values


def blocked_cv(response: np.ndarray, temperature: np.ndarray, precipitation: np.ndarray, years: np.ndarray) -> tuple[float, float, int]:
    predictions = np.full(len(years), np.nan)
    blocks = (years - years[0]) // 10
    for block in np.unique(blocks):
        test = blocks == block
        train = ~test
        coefficients = fit_ols(response[train], temperature[train], precipitation[train])
        predictions[test] = predict(coefficients, temperature[test], precipitation[test])
    valid = np.isfinite(response) & np.isfinite(predictions)
    if int(valid.sum()) < 30 or np.std(response[valid]) == 0:
        return float("nan"), float("nan"), int(valid.sum())
    nrmse = float(np.sqrt(np.mean((response[valid] - predictions[valid]) ** 2)) / np.std(response[valid]))
    return nrmse, correlation(response[valid], predictions[valid]), int(valid.sum())


def period_change(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    baseline = (years >= BASELINE[0]) & (years <= BASELINE[1])
    event = (years >= EVENT[0]) & (years <= EVENT[1])
    return np.nanmean(values[event], axis=0) - np.nanmean(values[baseline], axis=0)


def branch_analysis(
    branch: str,
    proxy_calibration: np.ndarray,
    proxy_scale: np.ndarray,
    calibration_temperature: dict[str, np.ndarray],
    calibration_precipitation: dict[str, np.ndarray],
    event_temperature: dict[str, np.ndarray],
    event_precipitation: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    cv_rows = []
    proxy_rows = []
    annual_predictions = np.full((len(EVENT_YEARS), proxy_calibration.shape[1]), np.nan)
    selected_predictions = np.full_like(annual_predictions, np.nan)
    for column in range(proxy_calibration.shape[1]):
        season_scores = []
        for season in SEASONS:
            nrmse, cv_r, count = blocked_cv(proxy_calibration[:, column], calibration_temperature[season][:, column], calibration_precipitation[season][:, column], CALIBRATION_YEARS)
            row = {"branch": branch, "proxy_column": column, "season": season, "cv_nrmse": nrmse, "cv_correlation": cv_r, "cv_prediction_count": count}
            cv_rows.append(row); season_scores.append(row)
        valid_scores = [row for row in season_scores if np.isfinite(row["cv_nrmse"])]
        if not valid_scores:
            proxy_rows.append({"branch": branch, "proxy_column": column, "selected_season": "unresolved"})
            continue
        selected = min(valid_scores, key=lambda row: (row["cv_nrmse"], list(SEASONS).index(row["season"])))
        annual_score = next(row for row in valid_scores if row["season"] == "annual")
        annual_coefficients = fit_ols(proxy_calibration[:, column], calibration_temperature["annual"][:, column], calibration_precipitation["annual"][:, column])
        selected_coefficients = fit_ols(proxy_calibration[:, column], calibration_temperature[selected["season"]][:, column], calibration_precipitation[selected["season"]][:, column])
        annual_predictions[:, column] = predict(annual_coefficients, event_temperature["annual"][:, column], event_precipitation["annual"][:, column])
        selected_predictions[:, column] = predict(selected_coefficients, event_temperature[selected["season"]][:, column], event_precipitation[selected["season"]][:, column])
        proxy_rows.append({
            "branch": branch,
            "proxy_column": column,
            "selected_season": selected["season"],
            "selected_cv_nrmse": selected["cv_nrmse"],
            "annual_cv_nrmse": annual_score["cv_nrmse"],
            "selected_minus_annual_cv_nrmse": selected["cv_nrmse"] - annual_score["cv_nrmse"],
            "annual_refit_event_change_z": period_change(EVENT_YEARS, annual_predictions[:, [column]])[0] / proxy_scale[column],
            "selected_season_event_change_z": period_change(EVENT_YEARS, selected_predictions[:, [column]])[0] / proxy_scale[column],
        })
    return pd.DataFrame(cv_rows), pd.DataFrame(proxy_rows), annual_predictions, selected_predictions


def metric_row(label: str, actual: np.ndarray, predicted: np.ndarray, rng: np.random.Generator) -> dict[str, float | str]:
    value, pvalue = permutation_correlation(actual, predicted, rng)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    return {
        "prediction": label,
        "proxy_count": int(valid.sum()),
        "event_change_correlation": value,
        "permutation_pvalue": pvalue,
        "sign_agreement_fraction": float(np.mean(np.sign(actual[valid]) == np.sign(predicted[valid]))),
        "network_mean_predicted_event_change_z": float(np.mean(predicted[valid])),
        "network_mean_actual_event_change_z": float(np.mean(actual[valid])),
    }


def main() -> int:
    root = project_root()
    event_table_path = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_site_instrumental_consistency_v1/site_instrumental_event_consistency.csv"
    proxy6 = archive_proxy_path()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_seasonal_psm_v1"
    sources = {"event_table": event_table_path, "proxy6": proxy6, "cru_temperature": CRU_TEMPERATURE, "cru_precipitation": CRU_PRECIPITATION, "gpcc_precipitation": GPCC_PRECIPITATION, "berkeley_temperature": BERKELEY_TEMPERATURE, "pages_metadata": PAGES_METADATA}
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    event_hash_before = digest(event_table_path)
    stats_before = {name: source_stat(path) for name, path in sources.items() if name != "event_table"}
    table = pd.read_csv(event_table_path).sort_values("proxy_index").reset_index(drop=True)
    if len(table) != 60 or not (table["coefficient_b2"] >= 0).all():
        raise RuntimeError("positive-b2-selection-contract-failed")
    latitudes = table["latitude"].to_numpy(dtype=float); longitudes = table["longitude"].to_numpy(dtype=float)

    pages = loadmat(PAGES_METADATA, variable_names=["p_lat", "p_lon", "pages2k_dataName"], squeeze_me=True, struct_as_record=False)
    pages_lat = np.asarray(pages["p_lat"], dtype=float).reshape(-1); pages_lon = np.asarray(pages["p_lon"], dtype=float).reshape(-1); pages_names = np.asarray(pages["pages2k_dataName"]).reshape(-1)
    candidate_counts = []; candidate_names = []
    for latitude, longitude in zip(latitudes, longitudes):
        candidates = np.flatnonzero(np.isclose(pages_lat, latitude, atol=1e-6) & np.isclose(pages_lon, longitude, atol=1e-6))
        candidate_counts.append(len(candidates)); candidate_names.append("|".join(str(pages_names[index]).strip() for index in candidates))
    table["pages2k_coordinate_candidate_count"] = candidate_counts; table["pages2k_coordinate_candidate_names"] = candidate_names

    proxy_event = np.full((len(EVENT_YEARS), len(table)), np.nan); proxy_calibration = np.full((len(CALIBRATION_YEARS), len(table)), np.nan)
    with h5py.File(proxy6, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(table["proxy_index"].to_numpy(dtype=int)):
            proxy = handle[refs[proxy_index - 1]]
            event_slice = slice(1904, 1920); calibration_slice = slice(1939, 2000)
            if not np.array_equal(np.asarray(proxy["age_yng"][0, event_slice], dtype=float), EVENT_YEARS) or not np.array_equal(np.asarray(proxy["age_yng"][0, calibration_slice], dtype=float), CALIBRATION_YEARS):
                raise RuntimeError(f"proxy-year-contract-failed:{proxy_index}")
            proxy_event[:, column] = np.asarray(proxy["data"][event_slice, 0], dtype=float); proxy_calibration[:, column] = np.asarray(proxy["data"][calibration_slice, 0], dtype=float)
    proxy_scale = np.nanstd(proxy_calibration, axis=0)
    actual_event_z = period_change(EVENT_YEARS, proxy_event) / proxy_scale
    if not np.allclose(actual_event_z, table["event_drop_z"].to_numpy(), atol=1e-12, rtol=0):
        raise RuntimeError("proxy-event-closure-failed")

    cru_t, cru_lat, cru_lon = read_netcdf_points(CRU_TEMPERATURE, ["tmp", "stn"], 1901, latitudes, longitudes)
    cru_p, cru_p_lat, cru_p_lon = read_netcdf_points(CRU_PRECIPITATION, ["pre", "stn"], 1901, latitudes, longitudes)
    gpcc, gpcc_lat, gpcc_lon = read_netcdf_points(GPCC_PRECIPITATION, ["precip", "numgauge", "infilled_numgauges", "interpolation_error"], 1891, latitudes, longitudes)
    berkeley = read_berkeley_points(latitudes, longitudes)
    if not (np.array_equal(cru_lat, cru_p_lat) and np.array_equal(cru_lon, cru_p_lon)):
        raise RuntimeError("cru-grid-selection-disagreement")

    event_labels = monthly_labels(1905, 1920); calibration_labels = monthly_labels(1940, 2000)
    seasonal = {}
    for season, months in SEASONS.items():
        seasonal[("event", "bea_t", season)] = seasonal_aggregate(berkeley["event"], event_labels, EVENT_YEARS, months)
        seasonal[("calibration", "bea_t", season)] = seasonal_aggregate(berkeley["calibration"], calibration_labels, CALIBRATION_YEARS, months)
        for period, years, labels in (("event", EVENT_YEARS, event_labels), ("calibration", CALIBRATION_YEARS, calibration_labels)):
            seasonal[(period, "cru_t", season)] = seasonal_aggregate(cru_t[period]["tmp"], labels, years, months)
            seasonal[(period, "cru_p", season)] = seasonal_aggregate(cru_p[period]["pre"], labels, years, months)
            seasonal[(period, "gpcc_p", season)] = seasonal_aggregate(gpcc[period]["precip"], labels, years, months)

    closure_rows = []
    annual_monthly_changes = {
        "temperature_bea": period_change(EVENT_YEARS, seasonal[("event", "bea_t", "annual")]),
        "temperature_cru": period_change(EVENT_YEARS, seasonal[("event", "cru_t", "annual")]),
        "precipitation_gpcc": period_change(EVENT_YEARS, seasonal[("event", "gpcc_p", "annual")]) / (365.25 / 12.0),
        "precipitation_cru": period_change(EVENT_YEARS, seasonal[("event", "cru_p", "annual")]) / (365.25 / 12.0),
    }
    for name, monthly_change in annual_monthly_changes.items():
        stored = table[f"{name}_event_change"].to_numpy(dtype=float)
        closure_rows.append({"product": name, "monthly_vs_stored_event_change_correlation": correlation(monthly_change, stored), "mean_monthly_derived_change": float(np.nanmean(monthly_change)), "mean_stored_annual_change": float(np.nanmean(stored)), "rmse": float(np.sqrt(np.nanmean((monthly_change - stored) ** 2)))})
    closure = pd.DataFrame(closure_rows)
    if (closure["monthly_vs_stored_event_change_correlation"] < 0.95).any():
        raise RuntimeError("monthly-to-stored-annual-product-closure-failed")

    production_inputs = (
        {season: seasonal[("calibration", "bea_t", season)] for season in SEASONS},
        {season: seasonal[("calibration", "gpcc_p", season)] for season in SEASONS},
        {season: seasonal[("event", "bea_t", season)] for season in SEASONS},
        {season: seasonal[("event", "gpcc_p", season)] for season in SEASONS},
    )
    cru_inputs = (
        {season: seasonal[("calibration", "cru_t", season)] for season in SEASONS},
        {season: seasonal[("calibration", "cru_p", season)] for season in SEASONS},
        {season: seasonal[("event", "cru_t", season)] for season in SEASONS},
        {season: seasonal[("event", "cru_p", season)] for season in SEASONS},
    )
    prod_cv, prod_proxy, prod_annual, prod_selected = branch_analysis("BEA_GPCC", proxy_calibration, proxy_scale, *production_inputs)
    cru_cv, cru_proxy, cru_annual, cru_selected = branch_analysis("CRU", proxy_calibration, proxy_scale, *cru_inputs)
    cv_table = pd.concat([prod_cv, cru_cv], ignore_index=True)
    proxy_results = pd.concat([prod_proxy, cru_proxy], ignore_index=True)
    proxy_results["proxy_index"] = table.iloc[proxy_results["proxy_column"].to_numpy(dtype=int)]["proxy_index"].to_numpy()
    joined = table.merge(proxy_results.drop(columns="proxy_column"), on="proxy_index", how="left", validate="one_to_many")

    rng = np.random.default_rng(RNG_SEED)
    metric_rows = []
    predictions = {
        "BEA_GPCC_annual_refit": period_change(EVENT_YEARS, prod_annual) / proxy_scale,
        "BEA_GPCC_selected_season": period_change(EVENT_YEARS, prod_selected) / proxy_scale,
        "CRU_annual_refit": period_change(EVENT_YEARS, cru_annual) / proxy_scale,
        "CRU_selected_season": period_change(EVENT_YEARS, cru_selected) / proxy_scale,
        "archived_annual_production_psm": table["production_psm_predicted_event_change_z"].to_numpy(dtype=float),
    }
    for label, values in predictions.items():
        metric_rows.append(metric_row(label, actual_event_z, values, rng))
    metrics = pd.DataFrame(metric_rows)

    event_month_mask = np.asarray([1905 <= year <= 1918 for year, _ in event_labels])
    coverage = pd.DataFrame({
        "proxy_index": table["proxy_index"],
        "latitude": latitudes,
        "longitude": longitudes,
        "cru_temperature_station_month_fraction": np.mean(cru_t["event"]["stn"][event_month_mask] > 0, axis=0),
        "cru_precipitation_station_month_fraction": np.mean(cru_p["event"]["stn"][event_month_mask] > 0, axis=0),
        "gpcc_direct_gauge_month_fraction": np.mean(gpcc["event"]["numgauge"][event_month_mask] > 0, axis=0),
        "gpcc_mean_infilled_gauge_count": np.nanmean(gpcc["event"]["infilled_numgauges"][event_month_mask], axis=0),
        "gpcc_mean_interpolation_error_mm_month": np.nanmean(gpcc["event"]["interpolation_error"][event_month_mask], axis=0),
    })

    network_curves = pd.DataFrame({"year": EVENT_YEARS, "actual_proxy_network_z_from_baseline": np.nanmean((proxy_event - np.nanmean(proxy_event[(EVENT_YEARS >= 1905) & (EVENT_YEARS <= 1914)], axis=0)[None, :]) / proxy_scale[None, :], axis=1)})
    for label, values in (("BEA_GPCC_annual_refit", prod_annual), ("BEA_GPCC_selected_season", prod_selected), ("CRU_selected_season", cru_selected)):
        centered = (values - np.nanmean(values[(EVENT_YEARS >= 1905) & (EVENT_YEARS <= 1914)], axis=0)[None, :]) / proxy_scale[None, :]
        network_curves[label + "_network_z_from_baseline"] = np.nanmean(centered, axis=1)

    summary = {
        "experiment_id": "climate_reconstruction_precipitation_positive_b2_event_seasonal_psm_v1",
        "seasonal_driver_analyzed_periods": [[1904, 1920], [1939, 2000]],
        "locked_final_period": [1921, 1939],
        "seasonal_driver_locked_final_period_accessed": False,
        "source_lineage_numeric_branch_status": "terminated_before_analysis",
        "source_lineage_preflight_boundary_incident": "MATLAB reported that the two legacy non-v7.3 MAT variables were loaded in full despite indexed syntax. No locked-period values were inspected, analyzed, exported, or used. The numeric lineage branch was stopped.",
        "candidate_seasons": list(SEASONS),
        "season_selection_rule": "Per-proxy minimum blocked-CV normalized RMSE over 1940-2000; event years are excluded from fitting and selection.",
        "pages2k_identity_from_coordinates": {"unique_id_count": int(np.sum(np.asarray(candidate_counts) == 1)), "ambiguous_id_count": int(np.sum(np.asarray(candidate_counts) > 1)), "unmatched_id_count": int(np.sum(np.asarray(candidate_counts) == 0))},
        "annual_product_closure": closure_rows,
        "event_prediction_metrics": metrics.to_dict(orient="records"),
        "selected_season_counts": {branch: {str(k): int(v) for k, v in group["selected_season"].value_counts().to_dict().items()} for branch, group in proxy_results.groupby("branch")},
        "median_selected_minus_annual_cv_nrmse": {branch: float(group["selected_minus_annual_cv_nrmse"].median()) for branch, group in proxy_results.groupby("branch")},
        "early_product_coverage": {
            "median_cru_temperature_station_month_fraction": float(coverage["cru_temperature_station_month_fraction"].median()),
            "median_cru_precipitation_station_month_fraction": float(coverage["cru_precipitation_station_month_fraction"].median()),
            "median_gpcc_direct_gauge_month_fraction": float(coverage["gpcc_direct_gauge_month_fraction"].median()),
            "median_gpcc_infilled_gauge_count": float(coverage["gpcc_mean_infilled_gauge_count"].median()),
            "median_gpcc_interpolation_error_mm_month": float(coverage["gpcc_mean_interpolation_error_mm_month"].median()),
        },
        "interpretation_limit": "Season selection is a controlled sensitivity, not an authorized replacement PSM. Early CRU/GPCC station and gauge fields describe local product support, not observational truth or formal uncertainty.",
    }

    output.mkdir(parents=True)
    cv_table.to_csv(output / "seasonal_psm_blocked_cv.csv", index=False)
    joined.to_csv(output / "seasonal_psm_proxy_results.csv", index=False)
    metrics.to_csv(output / "seasonal_event_prediction_metrics.csv", index=False)
    coverage.to_csv(output / "early_instrumental_coverage_at_proxy_sites.csv", index=False)
    closure.to_csv(output / "monthly_annual_product_closure.csv", index=False)
    network_curves.to_csv(output / "seasonal_psm_event_network_curves.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "BOUNDARY_NOTICE.md").write_text("# Boundary Notice\n\nA MATLAB R2024b preflight attempted indexed reads from two legacy MAT files. MATLAB warned that their non-v7.3 format forced full-variable loading. No values from 1921-1939 were inspected, analyzed, exported, or used, and the numeric source-lineage branch was terminated. This seasonal experiment reads only NetCDF/HDF slices for 1904-1920 and 1939-2000.\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13.4, 8.8), constrained_layout=True)
    axes[0, 0].plot(EVENT_YEARS, network_curves["actual_proxy_network_z_from_baseline"], color="#1B1B1B", linewidth=2.2, label="Stored proxies")
    axes[0, 0].plot(EVENT_YEARS, network_curves["BEA_GPCC_annual_refit_network_z_from_baseline"], color="#0072B2", linewidth=1.6, label="BEA+GPCC annual refit")
    axes[0, 0].plot(EVENT_YEARS, network_curves["BEA_GPCC_selected_season_network_z_from_baseline"], color="#009E73", linewidth=1.8, label="BEA+GPCC selected season")
    axes[0, 0].plot(EVENT_YEARS, network_curves["CRU_selected_season_network_z_from_baseline"], color="#D55E00", linewidth=1.6, label="CRU selected season")
    axes[0, 0].axhline(0, color="#999999", linewidth=0.8); axes[0, 0].axvspan(1915, 1918, color="#D9D9D9", alpha=0.45); axes[0, 0].set_ylabel("Network mean change (calibration z)"); axes[0, 0].set_title("Calibration-selected seasons versus event curve"); axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].scatter(predictions["BEA_GPCC_annual_refit"], actual_event_z, color="#0072B2", alpha=0.7, s=32, label="annual refit")
    axes[0, 1].scatter(predictions["BEA_GPCC_selected_season"], actual_event_z, facecolors="none", edgecolors="#009E73", alpha=0.85, s=38, label="selected season")
    limit = max(0.5, float(np.nanmax(np.abs(np.concatenate([actual_event_z, predictions["BEA_GPCC_annual_refit"], predictions["BEA_GPCC_selected_season"]])))))
    axes[0, 1].plot([-limit, limit], [-limit, limit], color="#777777", linewidth=0.8); axes[0, 1].axhline(0, color="#999999", linewidth=0.7); axes[0, 1].axvline(0, color="#999999", linewidth=0.7); axes[0, 1].set_xlim(-limit, limit); axes[0, 1].set_ylim(-limit, limit); axes[0, 1].set_xlabel("BEA+GPCC predicted event change (z)"); axes[0, 1].set_ylabel("Stored proxy event change (z)"); axes[0, 1].set_title("Does seasonality improve event prediction?"); axes[0, 1].legend(frameon=False)
    season_order = list(SEASONS)
    x = np.arange(len(season_order)); width = 0.38
    prod_counts = proxy_results[proxy_results.branch == "BEA_GPCC"].selected_season.value_counts(); cru_counts = proxy_results[proxy_results.branch == "CRU"].selected_season.value_counts()
    axes[1, 0].bar(x - width / 2, [prod_counts.get(season, 0) for season in season_order], width, color="#0072B2", label="BEA+GPCC")
    axes[1, 0].bar(x + width / 2, [cru_counts.get(season, 0) for season in season_order], width, color="#D55E00", label="CRU")
    axes[1, 0].set_xticks(x, season_order); axes[1, 0].set_ylabel("Selected proxy count"); axes[1, 0].set_title("Seasons selected by 1940-2000 blocked CV"); axes[1, 0].legend(frameon=False)
    axes[1, 1].scatter(coverage["gpcc_direct_gauge_month_fraction"], np.abs(actual_event_z - predictions["BEA_GPCC_selected_season"]), c=latitudes, cmap="viridis", s=42, alpha=0.8, edgecolor="#333333", linewidth=0.3)
    axes[1, 1].set_xlabel("GPCC direct-gauge month fraction, 1905-1918"); axes[1, 1].set_ylabel("Absolute seasonal prediction residual (z)"); axes[1, 1].set_title("Early product support versus unexplained proxy event")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "positive_b2_event_seasonal_psm.png", dpi=180); plt.close(figure)

    metrics_index = metrics.set_index("prediction")
    report = [
        "# Positive-b2 Event Seasonal PSM Audit",
        "",
        "Season selection uses only 1940-2000 blocked cross-validation. The 1915-1918 event is not used for fitting or selection.",
        "",
        f"- BEA+GPCC annual refit event correlation: `{metrics_index.loc['BEA_GPCC_annual_refit', 'event_change_correlation']:.3f}`.",
        f"- BEA+GPCC selected-season event correlation: `{metrics_index.loc['BEA_GPCC_selected_season', 'event_change_correlation']:.3f}`.",
        f"- BEA+GPCC selected-season predicted network event: `{metrics_index.loc['BEA_GPCC_selected_season', 'network_mean_predicted_event_change_z']:.3f} z`; actual `{metrics_index.loc['BEA_GPCC_selected_season', 'network_mean_actual_event_change_z']:.3f} z`.",
        f"- Median GPCC direct-gauge month fraction at proxy cells: `{summary['early_product_coverage']['median_gpcc_direct_gauge_month_fraction']:.1%}`.",
        f"- Coordinate-only public ID recovery: `{summary['pages2k_identity_from_coordinates']['unique_id_count']}/60` unique, `{summary['pages2k_identity_from_coordinates']['ambiguous_id_count']}/60` ambiguous.",
        "",
        "See BOUNDARY_NOTICE.md for the terminated legacy-MAT source-lineage preflight. No legacy MAT values are used in this experiment.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    event_hash_after = digest(event_table_path); stats_after = {name: source_stat(path) for name, path in sources.items() if name != "event_table"}
    if event_hash_before != event_hash_after or stats_before != stats_after:
        raise RuntimeError("source-inputs-changed-during-audit")
    (output / "source_manifest.json").write_text(json.dumps({"event_table_sha256_before": event_hash_before, "event_table_sha256_after": event_hash_after, "source_stats_before": stats_before, "source_stats_after": stats_after, "analysis_driver_sha256": digest(Path(__file__))}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import linregress

import diagnose_temperature_forced_background_members_at_stations_v1 as model_base
import run_climate_reconstruction_loop_v2_round21_temperature_reference_field_v1 as round21
import run_round277_direct_temperature_station_panel_1850_1950_v1 as ghcn_v1
import run_round277_direct_temperature_station_panel_1850_1950_v2 as ghcn_v2


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "temperature_four_background_scheme_evaluation_v1"
DRIVER = Path(__file__).resolve()
FACTS = ROOT / "LIA_PROJECT_FACTS.md"

PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
MODEL_FIELDS = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_v1/"
    "temperature_equal_blend_annual_spatial_candidate_1500_1899.nc"
)
ROUND277 = LOOP / "round_277_direct_temperature_station_panel_1850_1950_v2"
STATION_METADATA = ROUND277 / "station_metadata.csv"
GHCN_QCF = ROUND277 / "prepared/ghcnm_tavg_qcf.data"
GHCN_QCU = ROUND277 / "prepared/ghcnm_tavg_qcu.data"

LOCAL_INPUTS = {
    "Huang winter Tmin": {
        "path": LOOP / "round_206_setp_winter_temperature_locked_safe_comparison_v1/temperature_50year_windows_1500_1899.csv",
        "external": "external_winter_tmin_anomaly_degC",
        "project": "project_setp_annual_anomaly_degC",
        "bounds": (94.5, 97.5, 29.5, 31.5),
        "season": "previous-November to current-February Tmin",
        "independence": "identity unresolved; CRU calibrated",
        "smooth_correction": 1,
    },
    "Baiyu Jun-Jul": {
        "path": LOOP / "round_212_wscp_temperature_locked_safe_comparison_v1/temperature_50year_windows_1500_1899.csv",
        "external": "wscp_jj_reconstruction_anomaly_degC",
        "project": "project_local_box_annual_anomaly_degC",
        "bounds": (98.81, 100.81, 30.15, 32.15),
        "season": "June-July mean temperature",
        "independence": "source-independent tree network",
        "smooth_correction": 1,
    },
    "Haizi annual": {
        "path": LOOP / "round_216_haizi_annual_temperature_locked_safe_comparison_v1/temperature_50year_windows_1500_1899.csv",
        "external": "haizi_annual_reconstruction_anomaly_degC",
        "project": "project_local_box_annual_anomaly_degC",
        "bounds": (99.65, 101.65, 28.58, 30.58),
        "season": "previous-October to current-September mean temperature",
        "independence": "sequence-independent annual record",
        "smooth_correction": 1,
    },
    "Yang Jan-Jun": {
        "path": LOOP / "round_230_yang2010_project_reference_locked_safe_comparison_v2/temperature_50year_windows_1500_1899.csv",
        "external": "yang_jan_jun_11y_anomaly_degC",
        "project": "project_annual_11y_anomaly_degC",
        "bounds": (93.2666667, 95.2666667, 28.3666667, 30.3666667),
        "season": "January-June mean temperature; digitized 11-year low-pass",
        "independence": "independent cores; shared investigator; digitized",
        "smooth_correction": 11,
    },
}

YEARS_EARLY = np.arange(1500, 1900)
YEARS_MODERN = np.arange(1951, 2001)
BASELINE_YEARS = np.arange(1961, 1991)
ALL_PUBLIC_YEARS = np.arange(1000, 2001)
BLOCKS = [(1951, 1960), (1961, 1970), (1971, 1980), (1981, 1990), (1991, 2000)]
MODEL_ORDER = ["CESM", "ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"]
BRANCH_ORDER = ["current_CESM", "old_50_50", "equal_four", "skill_independence"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)
    }
    if stat.st_size <= 256 * 1024 * 1024:
        result["sha256"] = sha256(path)
    else:
        result["sha256"] = "not_rehashed_over_256MiB"
    return result


def clean_hdf_slice(dataset: h5py.Dataset, start_year: int, end_year: int) -> np.ndarray:
    start = start_year - 1000
    stop = end_year - 999
    values = np.asarray(dataset[start:stop], dtype=float)
    values[(values < -9000) | ~np.isfinite(values)] = np.nan
    return values


def public_anomalies(dataset: h5py.Dataset) -> tuple[np.ndarray, np.ndarray]:
    early = clean_hdf_slice(dataset, 1500, 1899)
    modern = clean_hdf_slice(dataset, 1951, 2000)
    baseline = clean_hdf_slice(dataset, 1961, 1990)
    with np.errstate(invalid="ignore"):
        center = np.nanmean(baseline, axis=0)
    return early - center[None, :, :], modern - center[None, :, :]


def parse_station_monthly(path: Path, series: str, station_ids: set[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    original_intervals = ghcn_v1.ALLOWED_INTERVALS
    ghcn_v1.ALLOWED_INTERVALS = ((1951, 2000),)
    audit: dict[str, int] = {}
    try:
        with path.open("rb") as stream:
            rows = ghcn_v2.parse_ghcnm_lines_v2(stream, station_ids, series, audit)
    finally:
        ghcn_v1.ALLOWED_INTERVALS = original_intervals
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["analysis_temperature_c"] = pd.to_numeric(frame["analysis_temperature_c"], errors="coerce")
    return frame, {str(key): int(value) for key, value in audit.items()}


def station_annual_anomalies(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = monthly[monthly["year"].between(1961, 1990)].copy()
    coverage = (
        baseline.groupby(["station_id", "series", "month"])["analysis_temperature_c"]
        .agg(["count", "mean"])
        .reset_index()
    )
    eligible = (
        coverage.assign(ok=lambda frame: frame["count"] >= 20)
        .groupby(["station_id", "series"])["ok"]
        .agg(["sum", "count"])
        .reset_index()
    )
    eligible["baseline_eligible"] = (eligible["sum"] == 12) & (eligible["count"] == 12)
    climatology = coverage.loc[coverage["count"] >= 20, ["station_id", "series", "month", "mean"]]
    values = monthly.merge(climatology, on=["station_id", "series", "month"], how="left")
    values["monthly_anomaly_degC"] = values["analysis_temperature_c"] - values["mean"]
    values["days"] = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(values.year, values.month)]
    rows = []
    for (station_id, series, year), group in values.groupby(["station_id", "series", "year"]):
        valid = group["monthly_anomaly_degC"].notna()
        if int(valid.sum()) != 12:
            anomaly = np.nan
        else:
            anomaly = float(np.average(group.loc[valid, "monthly_anomaly_degC"], weights=group.loc[valid, "days"]))
        rows.append({"station_id": station_id, "series": series, "year": int(year), "station_anomaly_degC": anomaly})
    annual = pd.DataFrame(rows)
    wide = annual.pivot(index=["station_id", "year"], columns="series", values="station_anomaly_degC").reset_index()
    wide = wide.rename(columns={"qcf": "qcf_station_degC", "qcu": "qcu_station_degC"})
    return wide, eligible


def extract_to_stations(
    fields: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
    stations: pd.DataFrame,
    value_name: str,
) -> pd.DataFrame:
    frame = model_base.grid_points_to_stations(fields, lat, lon, mask, stations)
    return frame.rename(columns={"value": value_name, "source_distance_deg": f"{value_name}_source_distance_deg"})


def block_means(values: pd.DataFrame, observed: str, model_columns: list[str]) -> pd.DataFrame:
    rows = []
    for start, end in BLOCKS:
        subset = values[values["year"].between(start, end)]
        for station_id, group in subset.groupby("station_id"):
            valid = group[[observed, *model_columns]].notna().all(axis=1)
            if int(valid.sum()) < 6:
                continue
            row: dict[str, object] = {
                "station_id": station_id, "start_year": start, "end_year": end,
                "observed_degC": float(group.loc[valid, observed].mean()),
            }
            for column in model_columns:
                row[column] = float(group.loc[valid, column].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def skill_independence_weights(blocks: pd.DataFrame, model_columns: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    observed = blocks["observed_degC"].to_numpy(float)
    matrix = blocks[model_columns].to_numpy(float)
    distances = np.sqrt(np.mean((matrix - observed[:, None]) ** 2, axis=0))
    separation = np.sqrt(np.mean((matrix[:, :, None] - matrix[:, None, :]) ** 2, axis=0))
    positive_d = distances[distances > 0]
    positive_s = separation[np.triu_indices(len(model_columns), 1)]
    positive_s = positive_s[positive_s > 0]
    sigma_d = float(np.median(positive_d)) if len(positive_d) else 1.0
    sigma_s = float(np.median(positive_s)) if len(positive_s) else 1.0
    skill = np.exp(-0.5 * np.square(distances / sigma_d))
    dependence = np.exp(-0.5 * np.square(separation / sigma_s)).sum(axis=1)
    raw = skill / dependence
    weights = raw / raw.sum()
    details = {
        "performance_rmse_degC": {name: float(value) for name, value in zip(MODEL_ORDER, distances)},
        "sigma_performance_degC": sigma_d,
        "sigma_independence_degC": sigma_s,
        "weights": {name: float(value) for name, value in zip(MODEL_ORDER, weights)},
    }
    return weights, details


def add_oof_skill_branch(values: pd.DataFrame, model_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    blocks = block_means(values, "qcf_station_degC", model_columns)
    full_weights, full_details = skill_independence_weights(blocks, model_columns)
    values = values.copy()
    values["skill_independence_fixed_degC"] = values[model_columns].to_numpy(float) @ full_weights
    values["skill_independence_oof_degC"] = np.nan
    fold_rows = []
    for start, end in BLOCKS:
        training = blocks[~blocks["start_year"].eq(start)]
        weights, details = skill_independence_weights(training, model_columns)
        selected = values["year"].between(start, end)
        values.loc[selected, "skill_independence_oof_degC"] = values.loc[selected, model_columns].to_numpy(float) @ weights
        fold_rows.append({"held_out_start": start, "held_out_end": end, **details["weights"]})
    return values, pd.DataFrame(fold_rows), full_details


def slope(values: pd.DataFrame, column: str) -> float:
    valid = values[column].notna()
    if int(valid.sum()) < 30:
        return np.nan
    return float(linregress(values.loc[valid, "year"], values.loc[valid, column]).slope * 100.0)


def modern_metrics(values: pd.DataFrame, qcu: bool = False) -> pd.DataFrame:
    observed = "qcu_station_degC" if qcu else "qcf_station_degC"
    branch_columns = {
        "current_CESM": "candidate_CESM_degC",
        "old_50_50": "old_50_50_degC",
        "equal_four": "equal_four_degC",
        "skill_independence": "skill_independence_oof_degC",
    }
    rows = []
    for branch, column in branch_columns.items():
        station_rows = []
        for station_id, group in values.groupby("station_id"):
            valid = group[[observed, column]].notna().all(axis=1)
            if int(valid.sum()) < 10:
                continue
            obs = group.loc[valid, observed].to_numpy(float)
            pred = group.loc[valid, column].to_numpy(float)
            station_rows.append({
                "station_id": station_id,
                "annual_mae": float(np.mean(np.abs(pred - obs))),
                "bias": float(np.mean(pred - obs)),
                "trend_error": abs(slope(group.loc[valid], column) - slope(group.loc[valid], observed)),
            })
        station = pd.DataFrame(station_rows)
        blocks = block_means(values, observed, [column])
        rows.append({
            "observation": "qcu" if qcu else "qcf",
            "branch": branch,
            "station_count": int(len(station)),
            "station_decade_count": int(len(blocks)),
            "decadal_rmse_degC": float(np.sqrt(np.mean(np.square(blocks[column] - blocks["observed_degC"])))),
            "median_station_annual_mae_degC": float(station["annual_mae"].median()),
            "equal_station_mean_bias_degC": float(station["bias"].mean()),
            "median_abs_trend_error_degC_per_century": float(station["trend_error"].median()),
        })
    return pd.DataFrame(rows)


def area_mean(fields: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(fields)
    numerator = np.nansum(fields * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def window_means(series: np.ndarray, smooth: int = 1) -> np.ndarray:
    values = pd.Series(series)
    if smooth > 1:
        values = values.rolling(smooth, center=True, min_periods=(smooth // 2 + 1)).mean()
    return np.asarray([values.iloc[start - 1500 : start - 1500 + 50].mean() for start in range(1500, 1900, 50)])


def box_mask(
    lat: np.ndarray, lon: np.ndarray, support: np.ndarray, bounds: tuple[float, float, float, float]
) -> tuple[np.ndarray, str, float]:
    lon_min, lon_max, lat_min, lat_max = bounds
    mask = support & (lat[:, None] >= lat_min) & (lat[:, None] <= lat_max) & (lon[None, :] >= lon_min) & (lon[None, :] <= lon_max)
    if int(mask.sum()):
        return mask, "box_common_support", 0.0
    center_lat = 0.5 * (lat_min + lat_max)
    center_lon = 0.5 * (lon_min + lon_max)
    distance = np.square(lat[:, None] - center_lat) + np.square(lon[None, :] - center_lon)
    distance[~support] = np.inf
    row, col = np.unravel_index(np.argmin(distance), distance.shape)
    mask = np.zeros_like(support)
    mask[row, col] = True
    return mask, "nearest_common_support_fallback", float(np.sqrt(distance[row, col]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    contract = OUT / "EXPERIMENT_CONTRACT.md"
    contract.write_text(
        "# Four background scheme evaluation\n\n"
        "Compare current CESM, historical 50:50, equal-four, and fixed skill-independence "
        "background schemes. Derive skill weights only from 1951-2000 qcf station decadal "
        "anomalies using five held-out decade blocks. Keep the archived project increment fixed. "
        "Evaluate 1500-1899 against each reconstruction family and four local records. No EnKF, "
        "OGGM, production edit, or 1921-1939 numeric access is authorized.\n",
        encoding="ascii",
    )
    sources = {
        "facts": FACTS, "public": PUBLIC, "model_fields": MODEL_FIELDS,
        "station_metadata": STATION_METADATA, "ghcn_qcf": GHCN_QCF, "ghcn_qcu": GHCN_QCU,
        "catalog": model_base.CATALOG, "mode_sim": model_base.MODE_SIM,
        "prior": model_base.PRIOR, "indices": model_base.INDICES,
        "proxy6": model_base.PROXY6, "proxy9": model_base.PROXY9,
        "grid": model_base.GRID, "driver": DRIVER, "contract": contract,
        **{f"local_{name}": item["path"] for name, item in LOCAL_INPUTS.items()},
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in sources.items()}

    stations = pd.read_csv(STATION_METADATA)
    station_ids = set(stations["station_id"].astype(str))
    qcf, qcf_audit = parse_station_monthly(GHCN_QCF, "qcf", station_ids)
    qcu, qcu_audit = parse_station_monthly(GHCN_QCU, "qcu", station_ids)
    annual_station, station_eligibility = station_annual_anomalies(pd.concat([qcf, qcu], ignore_index=True))

    with Dataset(MODEL_FIELDS) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        if not np.array_equal(years, YEARS_EARLY):
            raise RuntimeError("Early model year contract failed")
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        support = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        early_backgrounds = {
            "CESM": np.asarray(dataset.variables["tas_background_six_member"][:], dtype=float),
            "ModE-Sim": np.asarray(dataset.variables["tas_mode_sim"][:], dtype=float),
            "MPI-ESM1-2-LR": np.asarray(dataset.variables["tas_mpi_esm1_2_lr"][:], dtype=float),
            "MIROC-ES2L": np.asarray(dataset.variables["tas_miroc_es2l"][:], dtype=float),
        }

    with h5py.File(PUBLIC, "r") as handle:
        public_years = np.asarray(handle["time"], dtype=int).ravel()
        public_lat = np.asarray(handle["lat"], dtype=float).ravel()
        public_lon = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(public_years, ALL_PUBLIC_YEARS):
            raise RuntimeError("Public year contract failed")
        if not (np.allclose(public_lat, lat) and np.allclose(public_lon, lon)):
            raise RuntimeError("Public/model grid mismatch")
        project_early, project_modern = public_anomalies(handle["tas_ensemble_mean_rec_result"])
        reference_members = {
            name: public_anomalies(handle[variable])[0]
            for name, variable in round21.MEMBERS.items()
        }
    reference_families = {
        "local_enkf": np.nanmean(np.stack([reference_members["HGAO"], reference_members["O"]]), axis=0),
        "lmr": reference_members["LMR"],
        "neukom": np.nanmean(np.stack([field for name, field in reference_members.items() if name.startswith("Nk_")]), axis=0),
        "phyda": reference_members["PHYDA"],
    }

    model_base.ANALYSIS_YEARS = YEARS_MODERN
    model_base.BASELINE_YEARS = BASELINE_YEARS
    model_base.RECOVERY_YEARS = YEARS_MODERN
    modern_backgrounds = {"CESM": model_base.archived_background_fields(lat, lon)[:, support]}
    modern_backgrounds["ModE-Sim"] = model_base.annual_mode_sim_fields(lat, lon, support)
    catalog = json.loads(model_base.CATALOG.read_text(encoding="utf-8"))
    request_audit = []
    for model in ["MPI-ESM1-2-LR", "MIROC-ES2L"]:
        modern_backgrounds[model], audit = model_base.annual_remote_fields(
            model, [item for item in catalog if item["model"] == model], lat, lon, support
        )
        request_audit.extend(audit)

    project_points = project_modern[:, support]
    station_values = annual_station.merge(stations, on="station_id", how="left", validate="many_to_one")
    station_grid = stations[["station_id", "station_name", "latitude", "longitude"]]
    extracted = extract_to_stations(project_points, lat, lon, support, station_grid, "project_degC")
    for model in MODEL_ORDER:
        frame = extract_to_stations(modern_backgrounds[model], lat, lon, support, station_grid, f"background_{model}_degC")
        extracted = extracted.merge(frame, on=["station_id", "year"], validate="one_to_one")
    station_values = station_values.merge(extracted, on=["station_id", "year"], validate="one_to_one")
    distance_columns = [column for column in station_values if column.endswith("source_distance_deg")]
    station_values["maximum_source_distance_deg"] = station_values[distance_columns].max(axis=1)
    supported_ids = set(
        station_values.groupby("station_id")["maximum_source_distance_deg"].max().loc[lambda value: value <= 2.0].index
    )
    eligible_qcf = set(
        station_eligibility.loc[
            station_eligibility["series"].eq("qcf") & station_eligibility["baseline_eligible"], "station_id"
        ]
    )
    station_values = station_values[station_values["station_id"].isin(supported_ids & eligible_qcf)].copy()
    station_values["archived_increment_degC"] = station_values["project_degC"] - station_values["background_CESM_degC"]
    model_columns = []
    for model in MODEL_ORDER:
        column = f"candidate_{model}_degC"
        station_values[column] = station_values[f"background_{model}_degC"] + station_values["archived_increment_degC"]
        model_columns.append(column)
    station_values["current_CESM_degC"] = station_values[model_columns[0]]
    station_values["old_50_50_degC"] = (
        0.5 * station_values[model_columns[0]]
        + station_values[model_columns[1:]].mean(axis=1) * 0.5
    )
    station_values["equal_four_degC"] = station_values[model_columns].mean(axis=1)
    station_values, fold_weights, weight_details = add_oof_skill_branch(station_values, model_columns)
    modern = pd.concat([modern_metrics(station_values, False), modern_metrics(station_values, True)], ignore_index=True)

    weights = np.asarray([weight_details["weights"][name] for name in MODEL_ORDER])
    scheme_weights = {
        "current_CESM": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "old_50_50": np.asarray([0.5, 1 / 6, 1 / 6, 1 / 6]),
        "equal_four": np.full(4, 0.25),
        "skill_independence": weights,
    }
    background_stack = np.stack([early_backgrounds[name] for name in MODEL_ORDER])
    early_branches = {}
    for branch, branch_weights in scheme_weights.items():
        blended = np.tensordot(branch_weights, background_stack, axes=(0, 0))
        early_branches[branch] = project_early + (blended - early_backgrounds["CESM"])

    family_series = {name: area_mean(field, hma_mask, lat) for name, field in reference_families.items()}
    branch_series = {name: area_mean(field, hma_mask, lat) for name, field in early_branches.items()}
    family_rows = []
    family_windows = {name: window_means(series) for name, series in family_series.items()}
    for branch, series in branch_series.items():
        branch_windows = window_means(series)
        for family, target in family_windows.items():
            family_rows.append({
                "branch": branch, "reference_family": family,
                "window_rmse_degC": float(np.sqrt(np.mean(np.square(branch_windows - target)))),
                "window_bias_degC": float(np.mean(branch_windows - target)),
            })
    family_metrics = pd.DataFrame(family_rows)
    current_family = family_metrics[family_metrics["branch"].eq("current_CESM")].set_index("reference_family")["window_rmse_degC"]
    family_metrics["rmse_improvement_vs_current_fraction"] = [
        1.0 - row.window_rmse_degC / current_family.loc[row.reference_family]
        for row in family_metrics.itertuples(index=False)
    ]

    local_rows = []
    local_window_rows = []
    for record, config in LOCAL_INPUTS.items():
        source = pd.read_csv(config["path"])
        mask, support_method, support_distance = box_mask(lat, lon, support, config["bounds"])
        external = source[config["external"]].to_numpy(float)
        project = source[config["project"]].to_numpy(float)
        for branch in BRANCH_ORDER:
            correction = area_mean(early_branches[branch] - project_early, mask, lat)
            correction_windows = window_means(correction, int(config["smooth_correction"]))
            candidate = project + correction_windows
            local_rows.append({
                "record": record, "season": config["season"], "independence": config["independence"],
                "support_method": support_method, "support_cell_count": int(mask.sum()),
                "support_distance_deg": support_distance, "branch": branch,
                "window_rmse_degC": float(np.sqrt(np.mean(np.square(candidate - external)))),
                "window_bias_degC": float(np.mean(candidate - external)),
                "window_correlation": float(np.corrcoef(candidate, external)[0, 1]),
            })
            for index, row in source.iterrows():
                local_window_rows.append({
                    "record": record, "branch": branch,
                    "start_year": int(row.start_year), "end_year": int(row.end_year),
                    "external_degC": float(external[index]), "candidate_degC": float(candidate[index]),
                    "correction_degC": float(correction_windows[index]),
                })
    local_metrics = pd.DataFrame(local_rows)
    current_local = local_metrics[local_metrics["branch"].eq("current_CESM")].set_index("record")["window_rmse_degC"]
    local_metrics["rmse_improvement_vs_current_fraction"] = [
        1.0 - row.window_rmse_degC / current_local.loc[row.record]
        for row in local_metrics.itertuples(index=False)
    ]

    modern.to_csv(OUT / "modern_1951_2000_scheme_metrics.csv", index=False)
    station_values.to_csv(OUT / "modern_station_year_scheme_values.csv", index=False)
    station_eligibility.to_csv(OUT / "station_baseline_eligibility.csv", index=False)
    fold_weights.to_csv(OUT / "skill_independence_fold_weights.csv", index=False)
    pd.DataFrame(request_audit).to_csv(OUT / "remote_request_audit.csv", index=False)
    family_metrics.to_csv(OUT / "preindustrial_family_metrics.csv", index=False)
    local_metrics.to_csv(OUT / "preindustrial_local_record_metrics.csv", index=False)
    pd.DataFrame(local_window_rows).to_csv(OUT / "preindustrial_local_record_windows.csv", index=False)
    curves = pd.DataFrame({"year": YEARS_EARLY, **{f"branch_{name}": series for name, series in branch_series.items()}, **{f"reference_{name}": series for name, series in family_series.items()}})
    curves.to_csv(OUT / "hma_scheme_and_family_curves_1500_1899.csv", index=False)

    qcf_metrics = modern[modern["observation"].eq("qcf")].set_index("branch")
    family_aggregate = family_metrics.groupby("branch")["window_rmse_degC"].agg(["median", "max"])
    local_aggregate = local_metrics.groupby("branch")["window_rmse_degC"].agg(["median", "max"])
    evaluation = []
    for branch in BRANCH_ORDER:
        evaluation.append({
            "branch": branch,
            "modern_qcf_decadal_rmse_degC": float(qcf_metrics.loc[branch, "decadal_rmse_degC"]),
            "modern_qcf_median_annual_mae_degC": float(qcf_metrics.loc[branch, "median_station_annual_mae_degC"]),
            "family_median_50year_rmse_degC": float(family_aggregate.loc[branch, "median"]),
            "family_worst_50year_rmse_degC": float(family_aggregate.loc[branch, "max"]),
            "local_record_median_50year_rmse_degC": float(local_aggregate.loc[branch, "median"]),
            "local_record_worst_50year_rmse_degC": float(local_aggregate.loc[branch, "max"]),
        })
    evaluation = pd.DataFrame(evaluation)
    for metric in [column for column in evaluation if column != "branch"]:
        evaluation[f"rank_{metric}"] = evaluation[metric].rank(method="min").astype(int)
    evaluation.to_csv(OUT / "four_scheme_evaluation_summary.csv", index=False)

    family_stack_series = np.stack(list(family_series.values()))
    family_mean = np.nanmean(family_stack_series, axis=0)
    family_min = np.nanmin(family_stack_series, axis=0)
    family_max = np.nanmax(family_stack_series, axis=0)
    colors = {"current_CESM": "#c0392b", "old_50_50": "#2166ac", "equal_four": "#00876c", "skill_independence": "#7b3294"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axis = axes[0, 0]
    axis.fill_between(YEARS_EARLY, pd.Series(family_min).rolling(31, center=True, min_periods=16).mean(), pd.Series(family_max).rolling(31, center=True, min_periods=16).mean(), color="#bbbbbb", alpha=0.35, label="Four-family range")
    axis.plot(YEARS_EARLY, pd.Series(family_mean).rolling(31, center=True, min_periods=16).mean(), color="#111111", lw=2.3, label="Four-family mean")
    for branch in BRANCH_ORDER:
        axis.plot(YEARS_EARLY, pd.Series(branch_series[branch]).rolling(31, center=True, min_periods=16).mean(), color=colors[branch], lw=1.8, label=branch)
    axis.set(title="HMA candidates and reconstruction families, 1500-1899", ylabel="Temperature anomaly (degC)", xlim=(1500, 1899))
    axis.grid(axis="y", color="#dddddd", lw=0.6)
    axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes[0, 1]
    station_mean = station_values.groupby("year")[["qcf_station_degC", "current_CESM_degC", "old_50_50_degC", "equal_four_degC", "skill_independence_oof_degC"]].mean()
    axis.plot(station_mean.index, station_mean["qcf_station_degC"], color="#111111", lw=2.5, label="GHCN-M qcf")
    for branch, column in [("current_CESM", "current_CESM_degC"), ("old_50_50", "old_50_50_degC"), ("equal_four", "equal_four_degC"), ("skill_independence", "skill_independence_oof_degC")]:
        axis.plot(station_mean.index, station_mean[column].rolling(5, center=True, min_periods=3).mean(), color=colors[branch], lw=1.8, label=branch)
    axis.set(title="Matched-station mean, 1951-2000", ylabel="Temperature anomaly (degC)", xlim=(1951, 2000))
    axis.grid(axis="y", color="#dddddd", lw=0.6)
    axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes[1, 0]
    x = np.arange(len(evaluation))
    width = 0.26
    axis.bar(x - width, evaluation["modern_qcf_decadal_rmse_degC"], width, color="#d95f02", label="Modern qcf decadal")
    axis.bar(x, evaluation["family_median_50year_rmse_degC"], width, color="#1b9e77", label="Four-family median")
    axis.bar(x + width, evaluation["local_record_median_50year_rmse_degC"], width, color="#7570b3", label="Local-record median")
    axis.set_xticks(x, evaluation["branch"], rotation=25, ha="right")
    axis.set(ylabel="RMSE (degC)", title="Three independent evaluation dimensions")
    axis.grid(axis="y", color="#dddddd", lw=0.6)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    axis.bar(MODEL_ORDER, weights, color=["#c0392b", "#7b3294", "#d95f02", "#7570b3"])
    axis.axhline(0.25, color="#111111", ls="--", lw=1, label="Equal weight")
    axis.set(ylabel="Fixed weight", title="1951-2000 skill-independence weights", ylim=(0, max(0.45, float(weights.max()) * 1.18)))
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", color="#dddddd", lw=0.6)
    axis.legend(frameon=False)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    figure = OUT / "temperature_four_background_scheme_evaluation.png"
    fig.savefig(figure, dpi=190, facecolor="white")
    plt.close(fig)

    after = {name: state(path) for name, path in sources.items()}
    unchanged = before == after
    best = {
        "modern_qcf_decadal": str(evaluation.loc[evaluation["modern_qcf_decadal_rmse_degC"].idxmin(), "branch"]),
        "four_family_median": str(evaluation.loc[evaluation["family_median_50year_rmse_degC"].idxmin(), "branch"]),
        "local_record_median": str(evaluation.loc[evaluation["local_record_median_50year_rmse_degC"].idxmin(), "branch"]),
    }
    summary = {
        "experiment": OUT.name, "status": "complete",
        "schemes": {branch: {model: float(value) for model, value in zip(MODEL_ORDER, scheme_weights[branch])} for branch in BRANCH_ORDER},
        "skill_independence_details": weight_details,
        "modern_inventory_station_count": int(len(stations)),
        "qcf_baseline_eligible_station_count": int(len(eligible_qcf)),
        "common_support_eligible_station_count": int(len(supported_ids & eligible_qcf)),
        "best_by_dimension": best,
        "qcf_metrics": modern[modern["observation"].eq("qcf")].to_dict(orient="records"),
        "evaluation": evaluation.to_dict(orient="records"),
        "ghcn_parser_audit": {"qcf": qcf_audit, "qcu": qcu_audit},
        "locked_1921_1939_numeric_values_accessed": 0,
        "production_modified": False, "enkf_run": False, "oggm_run": False,
        "all_declared_source_states_unchanged": unchanged,
        "interpretation_boundary": "Modern forced-model weather phase is not scored as prediction skill; weights use station-decadal anomalies. Early references and local records are comparison evidence, not absolute truth.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    report = [
        "# Four Temperature Background Schemes", "",
        f"- Modern qcf station inventory/eligible/common-support: `{len(stations)}/{len(eligible_qcf)}/{len(supported_ids & eligible_qcf)}`.",
        "- Skill-independence weights: " + ", ".join(f"{name} `{weight_details['weights'][name]:.3f}`" for name in MODEL_ORDER) + ".",
        f"- Best modern qcf decadal branch: `{best['modern_qcf_decadal']}`.",
        f"- Best four-family 50-year branch: `{best['four_family_median']}`.",
        f"- Best four-local-record branch: `{best['local_record_median']}`.", "",
        "This is a fixed-increment diagnostic comparison. It does not authorize a production correction or imply that reconstruction families are observations.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A declared source changed")


if __name__ == "__main__":
    main()

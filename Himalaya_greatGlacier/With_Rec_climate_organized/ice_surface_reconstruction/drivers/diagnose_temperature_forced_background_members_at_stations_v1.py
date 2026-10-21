"""Compare archived and three forced temperature backgrounds at qcf stations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import preflight_temperature_pmip_equal_blend_spatial_field_v1 as spatial
import validate_temperature_pmip_low_frequency_station_bridge_v1 as station_base


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUTPUT = LOOP / "temperature_forced_background_member_station_diagnostic_v1"
CATALOG = RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2/remote_source_catalog.json"
MODE_SIM = spatial.MODE_SIM
PRIOR = spatial.regional_base.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy"
INDICES = spatial.regional_base.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
PROXY6 = spatial.regional_base.ARCHIVE / "python_data/proxy_data_6.mat"
PROXY9 = spatial.regional_base.ARCHIVE / "python_data/proxy_data_9.mat"
GRID = RESULTS / (
    "climate_reconstruction_temperature_low_frequency_correction_bridge_1900_1940_v1/"
    "temperature_low_frequency_correction_only_1500_1940.nc"
)
V1_STATION = LOOP / "temperature_pmip_low_frequency_station_bridge_validation_v1/station_year_values_allowed.csv"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()

ANALYSIS_YEARS = np.r_[np.arange(1880, 1921), np.arange(1940, 1951)]
BASELINE_YEARS = np.arange(1961, 1991)
RECOVERY_YEARS = np.r_[ANALYSIS_YEARS, BASELINE_YEARS]
LOCKED = (1921, 1939)
PERIODS = {"1880-1900": (1880, 1900), "1901-1920": (1901, 1920), "1940-1950": (1940, 1950)}
MPI_1910_1929 = {
    "model": "MPI-ESM1-2-LR",
    "experiment_role": "historical",
    "title": "tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_191001-192912.nc",
    "start_year_ce": 1910,
    "end_year_ce": 1929,
    "offset": 0,
    "url": (
        "https://esgf-node.ornl.gov/thredds/dodsC/css03_data/CMIP6/CMIP/MPI-M/"
        "MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/tas/gn/v20190710/"
        "tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_191001-192912.nc"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allowed(years: np.ndarray) -> np.ndarray:
    result = np.isin(years, RECOVERY_YEARS)
    if np.any(result & (years >= LOCKED[0]) & (years <= LOCKED[1])):
        raise RuntimeError("Locked year selected")
    return result


def annual_remote_fields(
    model: str,
    files: list[dict[str, Any]],
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    audit: list[dict[str, object]] = []
    required = set(int(year) for year in RECOVERY_YEARS)
    for item in files:
        if not required.intersection(range(int(item["start_year_ce"]), int(item["end_year_ce"]) + 1)):
            continue
        with Dataset(item["url"]) as dataset:
            time = dataset.variables["time"]
            dates = num2date(
                time[:], units=time.units, calendar=getattr(time, "calendar", "standard"),
                only_use_cftime_datetimes=True,
            )
            years = np.asarray([int(date.year) + int(item.get("offset", 0)) for date in dates])
            indices = np.flatnonzero(allowed(years))
            if len(indices) == 0:
                continue
            segments = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
            for segment in segments:
                values = spatial.remote_segment_values(dataset, segment, lat, lon, mask)
                for row, source_index in enumerate(segment):
                    year = int(years[source_index])
                    sums[year] = sums.get(year, np.zeros(values.shape[1], dtype=np.float64)) + values[row]
                    counts[year] = counts.get(year, 0) + 1
            audit.append({
                "model": model,
                "url": item["url"],
                "requested_month_count": int(len(indices)),
                "first_requested_year": int(years[indices].min()),
                "last_requested_year": int(years[indices].max()),
                "locked_months_requested": 0,
            })
    incomplete = [year for year in RECOVERY_YEARS if counts.get(int(year), 0) != 12]
    if incomplete:
        raise RuntimeError(f"Incomplete {model} coverage: {incomplete[:10]}")
    annual = {int(year): sums[int(year)] / 12.0 for year in RECOVERY_YEARS}
    baseline = np.mean(np.stack([annual[int(year)] for year in BASELINE_YEARS]), axis=0)
    return np.stack([annual[int(year)] - baseline for year in ANALYSIS_YEARS]), audit


def annual_mode_sim_fields(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> np.ndarray:
    points = spatial.target_points(lat, lon, mask)
    annual: dict[int, np.ndarray] = {}
    with Dataset(MODE_SIM) as dataset:
        source_years = np.asarray(dataset.variables["time"][:], dtype=int)
        source_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        source_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        variable = dataset.variables["temp2_anomaly_1961_1990"]
        for year in RECOVERY_YEARS:
            if LOCKED[0] <= int(year) <= LOCKED[1]:
                raise RuntimeError("Locked ModE-Sim year selected")
            index = int(np.flatnonzero(source_years == int(year))[0])
            annual[int(year)] = RegularGridInterpolator(
                (source_lat, source_lon), np.asarray(variable[index], dtype=float),
                bounds_error=False, fill_value=np.nan,
            )(points)
    baseline = np.mean(np.stack([annual[int(year)] for year in BASELINE_YEARS]), axis=0)
    return np.stack([annual[int(year)] - baseline for year in ANALYSIS_YEARS])


def archived_background_fields(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    prior = np.load(PRIOR)
    with h5py.File(INDICES, "r") as handle:
        experiment_indices = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    adaptive: dict[int, np.ndarray] = {}
    for network, proxy_path in [(6, PROXY6), (9, PROXY9)]:
        hxb, observations, locked_reads = spatial.round25.read_proxy_allowed(proxy_path, RECOVERY_YEARS)
        if locked_reads != 0:
            raise RuntimeError("Locked proxy values read")
        adaptive[network], _ = spatial.round25.ao_background_mean(
            prior, experiment_indices, hxb, observations
        )
    fixed_one = np.mean(
        np.stack([np.mean(prior[:, pool], axis=1) for pool in experiment_indices]), axis=0
    )
    fixed = np.repeat(fixed_one[None, :], len(RECOVERY_YEARS), axis=0)
    native_lat, native_lon = spatial.round25.native_coordinates()
    members = []
    for method, network in spatial.round25.MEMBERS.values():
        flat = fixed if method == "O" else adaptive[network]
        anomaly = spatial.round25.anomaly_field(flat, RECOVERY_YEARS)
        members.append(spatial.round25.interpolate_fields(anomaly, native_lat, native_lon, lat, lon))
    ensemble = np.nanmean(np.stack(members), axis=0)
    return ensemble[: len(ANALYSIS_YEARS)]


def grid_points_to_stations(
    fields: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    point_fields = np.full((len(fields), len(lat), len(lon)), np.nan)
    point_fields[:, mask] = fields
    for year_index, year in enumerate(ANALYSIS_YEARS):
        field = point_fields[year_index]
        for station in stations.itertuples(index=False):
            value, distance = station_base.nearest_finite_correction(
                field, lat, lon, float(station.latitude), float(station.longitude)
            )
            rows.append({
                "station_id": station.station_id,
                "year": int(year),
                "value": value,
                "source_distance_deg": distance,
            })
    return pd.DataFrame(rows)


def branch_metrics(values: pd.DataFrame, branches: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, (start, end) in PERIODS.items():
        period_values = values[values["year"].between(start, end)]
        for branch in branches:
            station_rows = []
            for station_id, group in period_values.groupby("station_id"):
                observed = group["qcf_station_degC"].to_numpy(float)
                predicted = group[branch].to_numpy(float)
                valid = np.isfinite(observed) & np.isfinite(predicted)
                if int(valid.sum()) < 5:
                    continue
                station_rows.append({
                    "station_id": station_id,
                    "bias": float(np.mean(predicted[valid] - observed[valid])),
                    "mae": float(np.mean(np.abs(predicted[valid] - observed[valid]))),
                })
            if not station_rows:
                continue
            frame = pd.DataFrame(station_rows)
            rows.append({
                "period": period,
                "branch": branch,
                "station_count": len(frame),
                "equal_station_mean_bias_degC": float(frame["bias"].mean()),
                "median_station_mae_degC": float(frame["mae"].median()),
                "station_bias_min_degC": float(frame["bias"].min()),
                "station_bias_max_degC": float(frame["bias"].max()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "EXPERIMENT_CONTRACT.md"
    contract.write_text(
        "# Forced-background member station diagnostic\n\n"
        "Compare the archived six-member background with ModE-Sim, MPI-ESM1-2-LR, "
        "MIROC-ES2L, their equal mean, and untuned 50:50 annual backgrounds at eligible "
        "GHCN-M qcf stations. Retain the archived posterior increment as a fixed residual "
        "term. Use only 1880-1920, 1940-1950, and the 1961-1990 anomaly baseline. Do not "
        "request or decode 1921-1939. This identifies source amplitude and cancellation; it "
        "does not select a fitted model weight or authorize an EnKF/production/OGGM run.\n",
        encoding="utf-8",
    )
    sources = {
        "catalog": CATALOG, "mode_sim": MODE_SIM, "prior": PRIOR, "indices": INDICES,
        "proxy6": PROXY6, "proxy9": PROXY9, "grid": GRID, "station": V1_STATION,
        "facts": FACTS, "driver": DRIVER, "contract": contract,
    }
    before = {name: sha256(path) for name, path in sources.items()}
    with Dataset(GRID) as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        mask = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)

    station_values = pd.read_csv(V1_STATION)
    qcf_ids = set(
        station_values.loc[np.isfinite(station_values["qcf_station_degC"]), "station_id"].astype(str)
    )
    station_values = station_values[
        station_values["station_id"].isin(qcf_ids) & station_values["year"].isin(ANALYSIS_YEARS)
    ].copy()
    stations = station_values[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates()

    background_grid = archived_background_fields(lat, lon)[:, mask]
    model_grids: dict[str, np.ndarray] = {"mode_sim": annual_mode_sim_fields(lat, lon, mask)}
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog.append(MPI_1910_1929)
    request_audit: list[dict[str, object]] = []
    for model, branch in [("MPI-ESM1-2-LR", "mpi"), ("MIROC-ES2L", "miroc")]:
        model_grids[branch], audit = annual_remote_fields(
            model, [item for item in catalog if item["model"] == model], lat, lon, mask
        )
        request_audit.extend(audit)
    model_grids["forced_mean"] = np.mean(np.stack(list(model_grids.values())), axis=0)

    extracted = grid_points_to_stations(background_grid, lat, lon, mask, stations).rename(
        columns={"value": "archived_background_degC", "source_distance_deg": "background_source_distance_deg"}
    )
    for name, fields in model_grids.items():
        frame = grid_points_to_stations(fields, lat, lon, mask, stations).rename(
            columns={"value": f"{name}_background_degC", "source_distance_deg": f"{name}_source_distance_deg"}
        )
        extracted = extracted.merge(frame, on=["station_id", "year"], how="left", validate="one_to_one")
    values = station_values.merge(extracted, on=["station_id", "year"], how="left", validate="one_to_one")
    values["archived_increment_degC"] = (
        values["project_six_member_degC"] - values["archived_background_degC"]
    )
    branches = ["project_six_member_degC"]
    for name in ["mode_sim", "mpi", "miroc", "forced_mean"]:
        column = f"fixed_increment_{name}_degC"
        values[column] = values[f"{name}_background_degC"] + values["archived_increment_degC"]
        branches.append(column)
    values["fixed_increment_equal_half_forced_mean_degC"] = (
        0.5 * values["archived_background_degC"]
        + 0.5 * values["forced_mean_background_degC"]
        + values["archived_increment_degC"]
    )
    branches.append("fixed_increment_equal_half_forced_mean_degC")

    metrics = branch_metrics(values, branches)
    metrics.to_csv(OUTPUT / "branch_period_metrics.csv", index=False)
    values.to_csv(OUTPUT / "station_year_background_members_allowed.csv", index=False)
    pd.DataFrame(request_audit).to_csv(OUTPUT / "remote_request_audit.csv", index=False)

    early = metrics[metrics["period"].isin(["1880-1900", "1901-1920"])]
    best_rows = early.loc[early.groupby("period")["equal_station_mean_bias_degC"].apply(lambda x: x.abs().idxmin())]
    best = {row.period: row.branch for row in best_rows.itertuples(index=False)}
    original = metrics[metrics["branch"].eq("project_six_member_degC")].set_index("period")
    forced = metrics[metrics["branch"].eq("fixed_increment_forced_mean_degC")].set_index("period")
    full_forced_improvement = {
        period: float(
            (abs(original.loc[period, "equal_station_mean_bias_degC"]) - abs(forced.loc[period, "equal_station_mean_bias_degC"]))
            / abs(original.loc[period, "equal_station_mean_bias_degC"])
        )
        for period in ["1880-1900", "1901-1920"]
    }
    individual_biases = {
        period: {
            branch: float(metrics[(metrics["period"] == period) & (metrics["branch"] == branch)]["equal_station_mean_bias_degC"].iloc[0])
            for branch in [
                "fixed_increment_mode_sim_degC", "fixed_increment_mpi_degC", "fixed_increment_miroc_degC"
            ]
        }
        for period in ["1880-1900", "1901-1920"]
    }
    all_individuals_remain_warm = all(
        value > 0 for period_values in individual_biases.values() for value in period_values.values()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = {
        "project_six_member_degC": "Original",
        "fixed_increment_mode_sim_degC": "ModE-Sim",
        "fixed_increment_mpi_degC": "MPI",
        "fixed_increment_miroc_degC": "MIROC",
        "fixed_increment_forced_mean_degC": "3-model mean",
        "fixed_increment_equal_half_forced_mean_degC": "50:50 annual",
    }
    for axis, period in zip(axes, ["1880-1900", "1901-1920"], strict=True):
        rows = metrics[metrics["period"].eq(period)].set_index("branch").loc[list(labels)]
        colors = ["#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#0072B2", "#E69F00"]
        axis.bar(np.arange(len(rows)), rows["equal_station_mean_bias_degC"], color=colors)
        axis.axhline(0, color="#555555", lw=0.8)
        axis.set_xticks(np.arange(len(rows)), [labels[name] for name in rows.index], rotation=35, ha="right")
        axis.set_title(period)
        axis.set_ylabel("Equal-station mean bias (degC)")
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "forced_background_member_station_bias.png", dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    decision = (
        "forced_background_family_reduces_but_does_not_close_western_station_warm_residual"
        if all_individuals_remain_warm
        else "at_least_one_forced_background_member_brackets_western_station_level"
    )
    summary = {
        "status": "complete",
        "experiment": "temperature_forced_background_member_station_diagnostic_v1",
        "decision": decision,
        "best_branch_by_period": best,
        "full_forced_mean_absolute_bias_improvement_fraction": full_forced_improvement,
        "individual_model_fixed_increment_biases_degC": individual_biases,
        "all_individual_model_counterfactuals_remain_warm": all_individuals_remain_warm,
        "locked_1921_1939_numeric_values_accessed": 0,
        "remote_locked_months_requested": 0,
        "all_declared_sources_unchanged": unchanged,
        "production_authorized": False,
        "enkf_run": False,
        "oggm_run": False,
        "interpretation_boundary": (
            "Forced-model annual weather is not expected to match observed annual weather. "
            "Period-mean level differences diagnose background amplitude only; member weights "
            "are not fitted or promoted."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Forced Background Members at Direct Stations",
        "",
        f"Decision: `{decision}`.",
        "",
    ]
    for period in ["1880-1900", "1901-1920"]:
        lines.append(f"- {period}: best untuned branch `{best[period]}`.")
        lines.append(
            f"- {period}: full three-model background changes absolute warm bias by "
            f"`{full_forced_improvement[period]:+.1%}`."
        )
        for branch, bias in individual_biases[period].items():
            lines.append(f"- {period} `{branch}` bias: `{bias:+.3f} degC`.")
    lines.extend([
        "",
        f"- All individual forced-background fixed-increment branches remain warm: `{str(all_individuals_remain_warm).lower()}`.",
        "- No model weight was fitted. No 1921-1939 value, EnKF, production file, or OGGM run was used.",
    ])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

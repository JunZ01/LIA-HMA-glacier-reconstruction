"""Build and validate a full-amplitude forced low-frequency background candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import linregress

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diagnose_temperature_forced_background_members_at_stations_v1 as forced


PROJECT = forced.PROJECT
RESULTS = forced.RESULTS
LOOP = forced.LOOP
OUTPUT = LOOP / "temperature_background_candidate_v2_allowed_years"
PRE = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_annual_spatial_candidate_v1/"
    "temperature_equal_blend_annual_spatial_candidate_1500_1899.nc"
)
OLD_LOW = RESULTS / (
    "climate_reconstruction_temperature_pmip_constrained_low_frequency_correction_v1/"
    "temperature_equal_blend_constrained_low_frequency_candidate_1500_1899.nc"
)
EARLY_STATION = LOOP / (
    "temperature_forced_background_member_station_diagnostic_v1/"
    "station_year_background_members_allowed.csv"
)
MODERN_STATION = LOOP / (
    "temperature_forced_background_member_station_1940_1950_validation_v1/"
    "station_year_background_members_1940_1950.csv"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
POST_YEARS = np.r_[np.arange(1900, 1921), np.arange(1940, 1951)]
BASELINE_YEARS = np.arange(1961, 1991)
OUTPUT_YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 1951)]
PERIODS = {"1880-1900": (1880, 1900), "1901-1920": (1901, 1920), "1940-1950": (1940, 1950)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rolling_low31(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rolling(31, center=True, min_periods=16).mean().to_numpy(float)


def slope(year: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 8:
        return np.nan
    return float(linregress(year[valid], values[valid]).slope * 100.0)


def station_metrics(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for period, (start, end) in PERIODS.items():
        segment = values[values["year"].between(start, end)]
        for station_id, group in segment.groupby("station_id"):
            observed = group["qcf_station_degC"].to_numpy(float)
            original = group["project_six_member_degC"].to_numpy(float)
            candidate = group["candidate_posterior_fixed_increment_degC"].to_numpy(float)
            year = group["year"].to_numpy(float)
            valid = np.isfinite(observed) & np.isfinite(original) & np.isfinite(candidate)
            if int(valid.sum()) < 5:
                continue
            observed_trend = slope(year[valid], observed[valid])
            original_trend = slope(year[valid], original[valid])
            candidate_trend = slope(year[valid], candidate[valid])
            rows.append({
                "period": period, "station_id": station_id,
                "station_name": group["station_name"].iloc[0], "n_years": int(valid.sum()),
                "original_mean_bias_degC": float(np.mean(original[valid] - observed[valid])),
                "candidate_mean_bias_degC": float(np.mean(candidate[valid] - observed[valid])),
                "original_mae_degC": float(np.mean(np.abs(original[valid] - observed[valid]))),
                "candidate_mae_degC": float(np.mean(np.abs(candidate[valid] - observed[valid]))),
                "original_abs_trend_error": abs(original_trend - observed_trend),
                "candidate_abs_trend_error": abs(candidate_trend - observed_trend),
                "mean_correction_degC": float(np.mean(group.loc[valid, "candidate_background_correction_degC"])),
            })
    station = pd.DataFrame(rows)
    aggregates = []
    for period, group in station.groupby("period"):
        original_bias = float(group["original_mean_bias_degC"].mean())
        candidate_bias = float(group["candidate_mean_bias_degC"].mean())
        original_mae = float(group["original_mae_degC"].median())
        candidate_mae = float(group["candidate_mae_degC"].median())
        original_trend = float(group["original_abs_trend_error"].median())
        candidate_trend = float(group["candidate_abs_trend_error"].median())
        aggregates.append({
            "period": period, "station_count": len(group),
            "original_equal_station_mean_bias_degC": original_bias,
            "candidate_equal_station_mean_bias_degC": candidate_bias,
            "absolute_bias_improvement_fraction": (abs(original_bias) - abs(candidate_bias)) / abs(original_bias),
            "original_median_station_mae_degC": original_mae,
            "candidate_median_station_mae_degC": candidate_mae,
            "median_station_mae_improvement_fraction": (original_mae - candidate_mae) / original_mae,
            "stations_improving_mae_fraction": float(np.mean(group["candidate_mae_degC"] < group["original_mae_degC"])),
            "original_median_abs_trend_error": original_trend,
            "candidate_median_abs_trend_error": candidate_trend,
            "median_trend_error_change_fraction": (
                (candidate_trend - original_trend) / original_trend if original_trend else np.nan
            ),
            "mean_correction_degC": float(group["mean_correction_degC"].mean()),
        })
    return station, pd.DataFrame(aggregates)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "EXPERIMENT_CONTRACT.md"
    contract.write_text(
        "# Temperature Background Candidate v2\n\n"
        "Build a diagnostic background on allowed years only. For 1500-1920, add the centered "
        "31-year mean of the full equal-three-model forced-background difference to the archived "
        "background; minimum 16 years is used only at segment endpoints. For 1940-1950, add the "
        "cellwise 1940-1950 mean difference as a constant so the short validation segment retains "
        "its original trend. The 1921-1939 interval is absent and no smoothing crosses it. Validate "
        "a fixed-increment posterior counterfactual against qcf. Require early bias and median MAE "
        "improvement, >=50% stations improving, early median trend-error regression <=10%, modern "
        "MAE regression <=5%, modern absolute-bias regression <=0.05 degC, and 1500-1899 50-year "
        "correction means within 0.05 degC of twice the previously accepted 50:50 constraints. "
        "This is not a production reconstruction, EnKF replay, or OGGM forcing.\n",
        encoding="utf-8",
    )
    sources = {
        "pre": PRE, "old_low": OLD_LOW, "early_station": EARLY_STATION,
        "modern_station": MODERN_STATION, "catalog": forced.CATALOG,
        "mode_sim": forced.MODE_SIM, "prior": forced.PRIOR, "indices": forced.INDICES,
        "proxy6": forced.PROXY6, "proxy9": forced.PROXY9, "grid": forced.GRID,
        "facts": FACTS, "driver": DRIVER, "contract": contract,
    }
    before = {name: sha256(path) for name, path in sources.items()}

    with Dataset(PRE) as dataset:
        pre_years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        common = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        pre_background = np.asarray(dataset.variables["tas_background_six_member"][:, common], dtype=float)
        pre_models = {
            "mode_sim": np.asarray(dataset.variables["tas_mode_sim"][:, common], dtype=float),
            "mpi": np.asarray(dataset.variables["tas_mpi_esm1_2_lr"][:, common], dtype=float),
            "miroc": np.asarray(dataset.variables["tas_miroc_es2l"][:, common], dtype=float),
        }
    if not np.array_equal(pre_years, np.arange(1500, 1900)):
        raise RuntimeError("Unexpected pre-candidate years")

    forced.ANALYSIS_YEARS = POST_YEARS
    forced.BASELINE_YEARS = BASELINE_YEARS
    forced.RECOVERY_YEARS = np.r_[POST_YEARS, BASELINE_YEARS]
    post_background = forced.archived_background_fields(lat, lon)[:, common]
    post_models = {"mode_sim": forced.annual_mode_sim_fields(lat, lon, common)}
    catalog = json.loads(forced.CATALOG.read_text(encoding="utf-8"))
    catalog.append(forced.MPI_1910_1929)
    request_audit = []
    for model, branch in [("MPI-ESM1-2-LR", "mpi"), ("MIROC-ES2L", "miroc")]:
        post_models[branch], audit = forced.annual_remote_fields(
            model, [item for item in catalog if item["model"] == model], lat, lon, common
        )
        request_audit.extend(audit)

    background = np.concatenate([pre_background, post_background], axis=0)
    model_fields = {
        name: np.concatenate([pre_models[name], post_models[name]], axis=0)
        for name in ["mode_sim", "mpi", "miroc"]
    }
    forced_mean = np.mean(np.stack(list(model_fields.values())), axis=0)
    raw_difference = forced_mean - background
    first_segment = OUTPUT_YEARS <= 1920
    modern_segment = OUTPUT_YEARS >= 1940
    correction = np.empty_like(raw_difference)
    correction[first_segment] = rolling_low31(raw_difference[first_segment])
    correction[modern_segment] = np.mean(raw_difference[modern_segment], axis=0, keepdims=True)
    candidate = background + correction
    if not all(np.all(np.isfinite(array)) for array in [background, forced_mean, correction, candidate]):
        raise RuntimeError("Non-finite value on common support")

    with Dataset(OLD_LOW) as dataset:
        old_years = np.asarray(dataset.variables["year"][:], dtype=int)
        old_correction = np.asarray(dataset.variables["tas_low_frequency_correction"][:, common], dtype=float)
    closure_errors = []
    for start in range(1500, 1900, 50):
        new_mean = np.mean(correction[(OUTPUT_YEARS >= start) & (OUTPUT_YEARS <= start + 49)], axis=0)
        old_mean = np.mean(old_correction[(old_years >= start) & (old_years <= start + 49)], axis=0) * 2.0
        closure_errors.append(float(np.max(np.abs(new_mean - old_mean))))
    window_closure = max(closure_errors)

    station_values = pd.concat([pd.read_csv(EARLY_STATION), pd.read_csv(MODERN_STATION)], ignore_index=True)
    station_values = station_values.sort_values(["station_id", "year"]).drop_duplicates(["station_id", "year"], keep="last")
    station_values = station_values[station_values["year"].isin(OUTPUT_YEARS)].copy()
    if station_values["year"].between(1921, 1939).any():
        raise RuntimeError("Locked station year present")
    stations = station_values[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates()
    correction_frame = forced.grid_points_to_stations(correction, lat, lon, common, stations).rename(
        columns={"value": "candidate_background_correction_degC", "source_distance_deg": "candidate_source_distance_deg"}
    )
    station_values = station_values.merge(correction_frame, on=["station_id", "year"], validate="one_to_one")
    station_values["candidate_posterior_fixed_increment_degC"] = (
        station_values["project_six_member_degC"] + station_values["candidate_background_correction_degC"]
    )
    station, aggregate = station_metrics(station_values)
    indexed = aggregate.set_index("period")
    gates = {
        "fifty_year_full_amplitude_constraint_closure_at_most_0p05_degC": window_closure <= 0.05,
        "early_absolute_bias_improves": bool(np.all(indexed.loc[["1880-1900", "1901-1920"], "absolute_bias_improvement_fraction"] > 0)),
        "early_median_station_mae_improves": bool(np.all(indexed.loc[["1880-1900", "1901-1920"], "median_station_mae_improvement_fraction"] > 0)),
        "early_majority_of_stations_improve": bool(np.all(indexed.loc[["1880-1900", "1901-1920"], "stations_improving_mae_fraction"] >= 0.5)),
        "early_median_trend_error_regression_at_most_10pct": bool(np.all(indexed.loc[["1880-1900", "1901-1920"], "median_trend_error_change_fraction"] <= 0.10)),
        "modern_median_station_mae_regression_at_most_5pct": bool(indexed.loc["1940-1950", "median_station_mae_improvement_fraction"] >= -0.05),
        "modern_absolute_bias_regression_at_most_0p05_degC": bool(
            abs(indexed.loc["1940-1950", "candidate_equal_station_mean_bias_degC"])
            - abs(indexed.loc["1940-1950", "original_equal_station_mean_bias_degC"]) <= 0.05
        ),
    }
    passes = bool(all(gates.values()))
    decision = (
        "retain_background_candidate_v2_for_frozen_enkf_replay_proposal"
        if passes else "reject_background_candidate_v2_at_allowed_year_station_or_constraint_gate"
    )

    output_nc = OUTPUT / "temperature_background_candidate_v2_allowed_years.nc"
    with Dataset(output_nc, "w", format="NETCDF4") as dataset:
        dataset.createDimension("year", len(OUTPUT_YEARS))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createVariable("year", "i4", ("year",))[:] = OUTPUT_YEARS
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("hma_mask", "i1", ("lat", "lon"))[:] = hma_mask.astype(np.int8)
        dataset.createVariable("common_support_mask", "i1", ("lat", "lon"))[:] = common.astype(np.int8)
        for name, values in {
            "tas_background_archived": background,
            "tas_background_forced_three_model_mean": forced_mean,
            "tas_background_low_frequency_correction": correction,
            "tas_background_candidate_v2": candidate,
        }.items():
            grid = np.full((len(OUTPUT_YEARS), len(lat), len(lon)), np.nan, dtype=np.float32)
            grid[:, common] = values.astype(np.float32)
            variable = dataset.createVariable(name, "f4", ("year", "lat", "lon"), zlib=True, complevel=4, chunksizes=(1, len(lat), len(lon)), fill_value=np.float32(np.nan))
            variable[:] = grid
            variable.units = "degC anomaly relative to 1961-1990"
        dataset.title = "Temperature Background Candidate v2, allowed years only"
        dataset.status = "diagnostic; not production reconstruction or OGGM forcing"
        dataset.locked_years = "1921-1939 absent and unread"
        dataset.candidate_definition = "full three-model forced-minus-archived low-frequency background correction"

    station_values.to_csv(OUTPUT / "station_year_candidate_values.csv", index=False)
    station.to_csv(OUTPUT / "station_period_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT / "aggregate_metrics.csv", index=False)
    pd.DataFrame(request_audit).to_csv(OUTPUT / "remote_request_audit.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    x = np.arange(len(aggregate))
    axes[0].bar(x - 0.18, aggregate["original_equal_station_mean_bias_degC"], 0.36, color="#D55E00", label="Original")
    axes[0].bar(x + 0.18, aggregate["candidate_equal_station_mean_bias_degC"], 0.36, color="#0072B2", label="Background Candidate v2")
    axes[0].set_xticks(x, aggregate["period"], rotation=25, ha="right")
    axes[0].set_ylabel("Equal-station mean bias (degC)")
    axes[0].legend(frameon=False)
    axes[1].bar(x, aggregate["median_trend_error_change_fraction"] * 100, color="#009E73")
    axes[1].axhline(10, color="#CC3311", lw=1.0, ls="--")
    axes[1].axhline(0, color="#555555", lw=0.8)
    axes[1].set_xticks(x, aggregate["period"], rotation=25, ha="right")
    axes[1].set_ylabel("Median trend-error change (%)")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "background_candidate_v2_station_validation.png", dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "status": "complete", "experiment": "temperature_background_candidate_v2_allowed_years",
        "decision": decision, "passes_all_frozen_gates": passes,
        "year_segments": ["1500-1920", "1940-1950"],
        "locked_1921_1939_numeric_values_accessed": 0,
        "fifty_year_full_amplitude_constraint_closure_max_degC": window_closure,
        "aggregate": aggregate.set_index("period").to_dict(orient="index"),
        "gates": gates, "candidate_sha256": sha256(output_nc),
        "all_declared_sources_unchanged": unchanged, "production_authorized": False,
        "enkf_run": False, "oggm_run": False,
        "interpretation_boundary": "Passing supports a proposal for one frozen EnKF replay; it is not a posterior reconstruction.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    lines = ["# Temperature Background Candidate v2", "", f"Decision: `{decision}`.", ""]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"- {row.period}: bias `{row.original_equal_station_mean_bias_degC:+.3f}` to "
            f"`{row.candidate_equal_station_mean_bias_degC:+.3f} degC`; median MAE change "
            f"`{row.median_station_mae_improvement_fraction:+.1%}`; median trend-error change "
            f"`{row.median_trend_error_change_fraction:+.1%}`."
        )
    lines.extend([
        f"- Full-amplitude 50-year constraint closure: `{window_closure:.4f} degC`.",
        f"- All frozen gates pass: `{str(passes).lower()}`.",
        "- 1921-1939 is absent; no EnKF, production, or OGGM run occurred.",
    ])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

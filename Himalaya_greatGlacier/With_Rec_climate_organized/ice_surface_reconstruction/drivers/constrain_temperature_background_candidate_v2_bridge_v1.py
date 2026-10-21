"""Constrain Candidate v2 to exact pre-1900 window means with a smooth 1900-1920 bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_temperature_background_candidate_v2_allowed_years as candidate_base
import diagnose_temperature_forced_background_members_at_stations_v1 as station_base


PROJECT = candidate_base.PROJECT
LOOP = candidate_base.LOOP
RESULTS = candidate_base.RESULTS
RAW = LOOP / (
    "temperature_background_candidate_v2_allowed_years_v4/"
    "temperature_background_candidate_v2_allowed_years.nc"
)
OLD = candidate_base.OLD_LOW
EARLY_STATION = candidate_base.EARLY_STATION
MODERN_STATION = candidate_base.MODERN_STATION
OUTPUT = LOOP / "temperature_background_candidate_v2_constrained_bridge_v1"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "EXPERIMENT_CONTRACT.md"
    contract.write_text(
        "# Constrained Background Candidate v2 bridge\n\n"
        "Replace the raw v4 correction in 1500-1899 by exactly twice the previously accepted "
        "50:50 constrained correction. For 1900-1920, retain the raw full-forced low-frequency "
        "correction plus a deterministic cubic smoothstep offset: match the 1899 constrained "
        "correction exactly at 1900 and decay the offset to zero by 1920. Keep 1940-1950 unchanged. "
        "Require exact 1500-1899 window closure <=2e-6 degC, 1899-1900 correction jump <=1e-8 "
        "degC, and all prior station gates unchanged. No threshold is relaxed and no station value "
        "fits the bridge. The locked interval is absent. No EnKF, production, or OGGM run.\n",
        encoding="utf-8",
    )
    sources = {"raw": RAW, "old": OLD, "early_station": EARLY_STATION, "modern_station": MODERN_STATION, "facts": FACTS, "driver": DRIVER, "contract": contract}
    before = {name: sha256(path) for name, path in sources.items()}
    with Dataset(RAW) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        common = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        background = np.asarray(dataset.variables["tas_background_archived"][:], dtype=float)[:, common]
        forced_mean = np.asarray(dataset.variables["tas_background_forced_three_model_mean"][:], dtype=float)[:, common]
        raw_correction = np.asarray(dataset.variables["tas_background_low_frequency_correction"][:], dtype=float)[:, common]
    with Dataset(OLD) as dataset:
        old_years = np.asarray(dataset.variables["year"][:], dtype=int)
        old_correction = np.asarray(dataset.variables["tas_low_frequency_correction"][:], dtype=float)[:, common]
    if years[0] != 1500 or 1921 in years or 1939 in years:
        raise RuntimeError("Unexpected candidate years")

    correction = raw_correction.copy()
    pre = years <= 1899
    correction[pre] = 2.0 * old_correction
    bridge = (years >= 1900) & (years <= 1920)
    bridge_years = years[bridge]
    t = (bridge_years - 1900.0) / 20.0
    taper = 1.0 - (3.0 * t**2 - 2.0 * t**3)
    constrained_1899 = correction[np.flatnonzero(years == 1899)[0]]
    raw_1900 = raw_correction[np.flatnonzero(years == 1900)[0]]
    offset = constrained_1899 - raw_1900
    correction[bridge] = raw_correction[bridge] + taper[:, None] * offset[None, :]
    candidate = background + correction

    window_errors = []
    for start in range(1500, 1900, 50):
        new_mean = np.mean(correction[(years >= start) & (years <= start + 49)], axis=0)
        target = 2.0 * np.mean(old_correction[(old_years >= start) & (old_years <= start + 49)], axis=0)
        window_errors.append(float(np.max(np.abs(new_mean - target))))
    window_closure = max(window_errors)
    splice_jump = float(np.max(np.abs(
        correction[np.flatnonzero(years == 1900)[0]] - correction[np.flatnonzero(years == 1899)[0]]
    )))

    station_values = pd.concat([pd.read_csv(EARLY_STATION), pd.read_csv(MODERN_STATION)], ignore_index=True)
    station_values = station_values.sort_values(["station_id", "year"]).drop_duplicates(["station_id", "year"], keep="last")
    station_values = station_values[station_values["year"].isin(years)].copy()
    stations = station_values[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates()
    previous_years = station_base.ANALYSIS_YEARS
    station_base.ANALYSIS_YEARS = years
    try:
        frame = station_base.grid_points_to_stations(correction, lat, lon, common, stations).rename(
            columns={"value": "candidate_background_correction_degC", "source_distance_deg": "candidate_source_distance_deg"}
        )
    finally:
        station_base.ANALYSIS_YEARS = previous_years
    station_values = station_values.merge(frame, on=["station_id", "year"], validate="one_to_one")
    station_values["candidate_posterior_fixed_increment_degC"] = station_values["project_six_member_degC"] + station_values["candidate_background_correction_degC"]
    station, aggregate = candidate_base.station_metrics(station_values)
    indexed = aggregate.set_index("period")
    gates = {
        "fifty_year_full_amplitude_constraint_closure_at_most_2e_6_degC": window_closure <= 2e-6,
        "correction_1899_1900_jump_at_most_1e_8_degC": splice_jump <= 1e-8,
        "early_absolute_bias_improves": bool(np.all(indexed.loc[["1880-1900", "1901-1920"], "absolute_bias_improvement_fraction"] > 0)),
        "early_median_station_mae_improves": bool(np.all(indexed.loc[["1880-1900", "1901-1920"], "median_station_mae_improvement_fraction"] > 0)),
        "early_majority_of_stations_improve": bool(np.all(indexed.loc[["1880-1900", "1901-1920"], "stations_improving_mae_fraction"] >= 0.5)),
        "early_median_trend_error_regression_at_most_10pct": bool(np.all(indexed.loc[["1880-1900", "1901-1920"], "median_trend_error_change_fraction"] <= 0.10)),
        "modern_median_station_mae_regression_at_most_5pct": bool(indexed.loc["1940-1950", "median_station_mae_improvement_fraction"] >= -0.05),
        "modern_absolute_bias_regression_at_most_0p05_degC": bool(abs(indexed.loc["1940-1950", "candidate_equal_station_mean_bias_degC"]) - abs(indexed.loc["1940-1950", "original_equal_station_mean_bias_degC"]) <= 0.05),
    }
    passes = bool(all(gates.values()))
    decision = "retain_constrained_background_candidate_v2_for_frozen_enkf_replay_proposal" if passes else "reject_constrained_background_candidate_v2"

    output_nc = OUTPUT / "temperature_background_candidate_v2_constrained_allowed_years.nc"
    with Dataset(output_nc, "w", format="NETCDF4") as dataset:
        dataset.createDimension("year", len(years))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createVariable("year", "i4", ("year",))[:] = years
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("hma_mask", "i1", ("lat", "lon"))[:] = hma.astype(np.int8)
        dataset.createVariable("common_support_mask", "i1", ("lat", "lon"))[:] = common.astype(np.int8)
        for name, values in {
            "tas_background_archived": background,
            "tas_background_forced_three_model_mean": forced_mean,
            "tas_background_low_frequency_correction": correction,
            "tas_background_candidate_v2": candidate,
        }.items():
            grid = np.full((len(years), len(lat), len(lon)), np.nan, dtype=np.float32)
            grid[:, common] = values.astype(np.float32)
            variable = dataset.createVariable(name, "f4", ("year", "lat", "lon"), zlib=True, complevel=4, chunksizes=(1, len(lat), len(lon)), fill_value=np.float32(np.nan))
            variable[:] = grid
            variable.units = "degC anomaly relative to 1961-1990"
        dataset.title = "Constrained Temperature Background Candidate v2"
        dataset.status = "diagnostic; not production reconstruction or OGGM forcing"
        dataset.locked_years = "1921-1939 absent and unread"
        dataset.bridge = "cubic smoothstep offset from constrained 1899 state to raw full-forced 1920 state"

    station_values.to_csv(OUTPUT / "station_year_candidate_values.csv", index=False)
    station.to_csv(OUTPUT / "station_period_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT / "aggregate_metrics.csv", index=False)
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(len(aggregate))
    axis.bar(x - 0.18, aggregate["original_equal_station_mean_bias_degC"], 0.36, color="#D55E00", label="Original")
    axis.bar(x + 0.18, aggregate["candidate_equal_station_mean_bias_degC"], 0.36, color="#0072B2", label="Constrained Candidate v2")
    axis.set_xticks(x, aggregate["period"])
    axis.set_ylabel("Equal-station mean bias (degC)")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#DDDDDD", lw=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "constrained_background_candidate_v2_bias.png", dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "status": "complete", "experiment": "temperature_background_candidate_v2_constrained_bridge_v1",
        "decision": decision, "passes_all_frozen_gates": passes,
        "locked_1921_1939_numeric_values_accessed": 0,
        "fifty_year_constraint_closure_max_degC": window_closure,
        "correction_1899_1900_jump_max_degC": splice_jump,
        "aggregate": aggregate.set_index("period").to_dict(orient="index"),
        "gates": gates, "candidate_sha256": sha256(output_nc),
        "all_declared_sources_unchanged": unchanged, "production_authorized": False,
        "enkf_run": False, "oggm_run": False,
        "interpretation_boundary": "Passing authorizes proposing one frozen EnKF replay; the posterior response remains unknown.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    lines = ["# Constrained Temperature Background Candidate v2", "", f"Decision: `{decision}`.", ""]
    for row in aggregate.itertuples(index=False):
        lines.append(f"- {row.period}: bias `{row.original_equal_station_mean_bias_degC:+.3f}` to `{row.candidate_equal_station_mean_bias_degC:+.3f} degC`; median MAE change `{row.median_station_mae_improvement_fraction:+.1%}`; trend-error change `{row.median_trend_error_change_fraction:+.1%}`.")
    lines.extend([f"- 50-year constraint closure: `{window_closure:.3e} degC`.", f"- 1899-1900 correction jump: `{splice_jump:.3e} degC`.", f"- All frozen gates pass: `{str(passes).lower()}`.", "- No 1921-1939 value, EnKF, production file, or OGGM run was used."])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

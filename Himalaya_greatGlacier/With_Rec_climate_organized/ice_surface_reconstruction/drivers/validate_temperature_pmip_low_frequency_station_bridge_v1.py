"""Validate the frozen PMIP low-frequency correction against direct stations."""

from __future__ import annotations

import csv
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


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
STATION = LOOP / "temperature_station_annual_anomaly_benchmark_v4"
STATION_VALUES = STATION / "station_annual_anomalies_and_products.csv"
STATION_COVERAGE = STATION / "station_baseline_coverage.csv"
CORRECTION = RESULTS / (
    "climate_reconstruction_temperature_low_frequency_correction_bridge_1900_1940_v1/"
    "temperature_low_frequency_correction_only_1500_1940.nc"
)
OUTPUT = LOOP / "temperature_pmip_low_frequency_station_bridge_validation_v1"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()

PERIODS = {"1880-1900": (1880, 1900), "1901-1920": (1901, 1920)}
LOCKED = (1921, 1939)
PUBLISHED_QCF_BIAS = {"1880-1900": 1.4175198151273858, "1901-1920": 0.8468282727596853}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: str) -> float:
    if value is None or value.strip() == "":
        return np.nan
    return float(value)


def read_allowed_station_rows() -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    locked_rows_skipped = 0
    with STATION_VALUES.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            year = int(raw["year"])
            if LOCKED[0] <= year <= LOCKED[1]:
                locked_rows_skipped += 1
                continue
            if not (1880 <= year <= 1920 or year == 1940):
                continue
            rows.append({
                "station_id": raw["station_id"],
                "station_name": raw["station_name"],
                "region_group": raw["region_group"],
                "latitude": float(raw["latitude"]),
                "longitude": float(raw["longitude"]),
                "year": year,
                "qcf_station_degC": finite(raw["qcf_station_degC"]),
                "qcu_station_degC": finite(raw["qcu_station_degC"]),
                "project_six_member_degC": finite(raw["project_six_member_degC"]),
            })
    return pd.DataFrame(rows), locked_rows_skipped


def eligible_stations() -> dict[str, set[str]]:
    coverage = pd.read_csv(STATION_COVERAGE)
    result: dict[str, set[str]] = {}
    for series in ["qcf", "qcu"]:
        subset = coverage[(coverage["series"] == series) & coverage["baseline_eligible"].astype(bool)]
        result[series] = set(subset["station_id"].astype(str))
    return result


def nearest_finite_correction(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> tuple[float, float]:
    valid = np.isfinite(field)
    if not np.any(valid):
        return np.nan, np.nan
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    distance = np.sqrt((yy - target_lat) ** 2 + ((xx - target_lon) * np.cos(np.deg2rad(target_lat))) ** 2)
    distance[~valid] = np.inf
    index = np.unravel_index(int(np.argmin(distance)), distance.shape)
    return float(field[index]), float(distance[index])


def extract_corrections(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    requested_years = sorted(set(rows["year"].astype(int)))
    if any(LOCKED[0] <= year <= LOCKED[1] for year in requested_years):
        raise RuntimeError("Locked year entered correction request")
    metadata_rows = rows[["station_id", "latitude", "longitude"]].drop_duplicates()
    extracted: list[dict[str, object]] = []
    distances: dict[str, float] = {}
    with Dataset(CORRECTION) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        variable = dataset.variables["tas_low_frequency_correction_only"]
        year_indices = {year: int(np.flatnonzero(years == year)[0]) for year in requested_years}
        for year in requested_years:
            field = np.asarray(variable[year_indices[year]], dtype=float)
            for station in metadata_rows.itertuples(index=False):
                value, distance = nearest_finite_correction(
                    field, lat, lon, float(station.latitude), float(station.longitude)
                )
                extracted.append({
                    "station_id": station.station_id,
                    "year": year,
                    "low_frequency_correction_degC": value,
                    "correction_source_distance_deg": distance,
                })
                distances[str(station.station_id)] = max(
                    distances.get(str(station.station_id), 0.0), distance
                )
    return pd.DataFrame(extracted), distances


def slope_per_century(year: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(year) & np.isfinite(values)
    if int(valid.sum()) < 8:
        return np.nan
    return float(linregress(year[valid], values[valid]).slope * 100.0)


def period_metrics(values: pd.DataFrame, eligibility: dict[str, set[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series in ["qcf", "qcu"]:
        observed_column = f"{series}_station_degC"
        for period, (start, end) in PERIODS.items():
            period_values = values[
                values["year"].between(start, end)
                & values["station_id"].isin(eligibility[series])
            ]
            for station_id, group in period_values.groupby("station_id"):
                observed = group[observed_column].to_numpy(float)
                original = group["project_six_member_degC"].to_numpy(float)
                candidate = group["fixed_increment_candidate_degC"].to_numpy(float)
                year = group["year"].to_numpy(float)
                valid = np.isfinite(observed) & np.isfinite(original) & np.isfinite(candidate)
                if int(valid.sum()) < 5:
                    continue
                observed = observed[valid]
                original = original[valid]
                candidate = candidate[valid]
                year = year[valid]
                observed_slope = slope_per_century(year, observed)
                original_slope = slope_per_century(year, original)
                candidate_slope = slope_per_century(year, candidate)
                rows.append({
                    "series": series,
                    "period": period,
                    "station_id": station_id,
                    "station_name": group["station_name"].iloc[0],
                    "region_group": group["region_group"].iloc[0],
                    "n_years": int(valid.sum()),
                    "original_mean_bias_degC": float(np.mean(original - observed)),
                    "candidate_mean_bias_degC": float(np.mean(candidate - observed)),
                    "original_mae_degC": float(np.mean(np.abs(original - observed))),
                    "candidate_mae_degC": float(np.mean(np.abs(candidate - observed))),
                    "observed_trend_degC_per_century": observed_slope,
                    "original_trend_degC_per_century": original_slope,
                    "candidate_trend_degC_per_century": candidate_slope,
                    "original_abs_trend_error": abs(original_slope - observed_slope),
                    "candidate_abs_trend_error": abs(candidate_slope - observed_slope),
                    "mean_correction_degC": float(np.mean(candidate - original)),
                })
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for (series, period), group in metrics.groupby(["series", "period"]):
        original_bias = float(group["original_mean_bias_degC"].mean())
        candidate_bias = float(group["candidate_mean_bias_degC"].mean())
        original_mae = float(group["original_mae_degC"].median())
        candidate_mae = float(group["candidate_mae_degC"].median())
        original_trend = float(group["original_abs_trend_error"].median())
        candidate_trend = float(group["candidate_abs_trend_error"].median())
        row = {
            "series": series,
            "period": period,
            "station_count": int(len(group)),
            "original_equal_station_mean_bias_degC": original_bias,
            "candidate_equal_station_mean_bias_degC": candidate_bias,
            "absolute_bias_improvement_fraction": (
                (abs(original_bias) - abs(candidate_bias)) / abs(original_bias)
                if original_bias != 0 else np.nan
            ),
            "original_median_station_mae_degC": original_mae,
            "candidate_median_station_mae_degC": candidate_mae,
            "median_station_mae_improvement_fraction": (
                (original_mae - candidate_mae) / original_mae if original_mae != 0 else np.nan
            ),
            "original_median_abs_trend_error": original_trend,
            "candidate_median_abs_trend_error": candidate_trend,
            "median_trend_error_change_fraction": (
                (candidate_trend - original_trend) / original_trend if original_trend != 0 else np.nan
            ),
            "stations_with_mae_improvement_fraction": float(
                np.mean(group["candidate_mae_degC"] < group["original_mae_degC"])
            ),
            "mean_correction_degC": float(group["mean_correction_degC"].mean()),
        }
        rows.append(row)
        summary[f"{series}_{period}"] = row
    return pd.DataFrame(rows), summary


def plot_curves(values: pd.DataFrame, eligibility: dict[str, set[str]], output: Path) -> None:
    qcf = values[values["station_id"].isin(eligibility["qcf"])].copy()
    yearly = qcf.groupby("year")[[
        "qcf_station_degC", "project_six_member_degC", "fixed_increment_candidate_degC",
        "low_frequency_correction_degC",
    ]].mean()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
    for start, end in PERIODS.values():
        segment = yearly.loc[start:end]
        axes[0].plot(segment.index, segment["qcf_station_degC"], color="#111111", lw=1.8)
        axes[0].plot(segment.index, segment["project_six_member_degC"], color="#D55E00", lw=1.5)
        axes[0].plot(segment.index, segment["fixed_increment_candidate_degC"], color="#0072B2", lw=1.5)
        axes[1].plot(segment.index, segment["low_frequency_correction_degC"], color="#0072B2", lw=1.8)
    axes[0].plot([], [], color="#111111", label="GHCN-M qcf")
    axes[0].plot([], [], color="#D55E00", label="Original reconstruction")
    axes[0].plot([], [], color="#0072B2", label="Low-frequency background candidate")
    axes[0].axhline(0, color="#888888", lw=0.7)
    axes[0].set_ylabel("Temperature anomaly (degC)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].axhline(0, color="#888888", lw=0.7)
    axes[1].set_ylabel("Applied correction (degC)")
    axes[1].set_xlabel("Year")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "EXPERIMENT_CONTRACT.md"
    contract.write_text(
        "# PMIP Low-Frequency Station Bridge Validation\n\n"
        "Validate the already frozen 50:50 PMIP-derived low-frequency correction against "
        "GHCN-M qcf stations over 1880-1900 and 1901-1920. Apply the correction to the "
        "archived six-member reconstruction while freezing its assimilation increment. "
        "qcu is sensitivity only. No amplitude, station subset, season, or spatial weight is "
        "fit. Numeric values for 1921-1939 are skipped before decoding and do not enter any "
        "metric. This is a background-mechanism counterfactual, not an EnKF replay, production "
        "replacement, or OGGM forcing.\n",
        encoding="utf-8",
    )
    sources = {
        "station_values": STATION_VALUES,
        "station_coverage": STATION_COVERAGE,
        "correction": CORRECTION,
        "facts": FACTS,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: sha256(path) for name, path in sources.items()}

    values, locked_rows_skipped = read_allowed_station_rows()
    eligibility = eligible_stations()
    corrections, distances = extract_corrections(values)
    values = values.merge(corrections, on=["station_id", "year"], how="left", validate="many_to_one")
    values["fixed_increment_candidate_degC"] = (
        values["project_six_member_degC"] + values["low_frequency_correction_degC"]
    )
    values["locked_period_value_accessed"] = False

    metrics = period_metrics(values, eligibility)
    aggregate, aggregate_summary = aggregate_metrics(metrics)
    qcf_rows = aggregate[aggregate["series"] == "qcf"].set_index("period")
    closure = max(
        abs(float(qcf_rows.loc[period, "original_equal_station_mean_bias_degC"]) - expected)
        for period, expected in PUBLISHED_QCF_BIAS.items()
    )
    gates = {
        "published_qcf_bias_closure_at_most_0p03_degC": bool(closure <= 0.03),
        "all_station_correction_sources_within_2_degrees": bool(max(distances.values()) <= 2.0),
        "qcf_absolute_bias_improves_in_both_periods": bool(
            np.all(qcf_rows["absolute_bias_improvement_fraction"] > 0)
        ),
        "qcf_median_station_mae_improves_in_both_periods": bool(
            np.all(qcf_rows["median_station_mae_improvement_fraction"] > 0)
        ),
        "qcf_majority_of_stations_improve_in_both_periods": bool(
            np.all(qcf_rows["stations_with_mae_improvement_fraction"] >= 0.5)
        ),
        "qcf_median_trend_error_not_worse_by_more_than_10pct": bool(
            np.all(qcf_rows["median_trend_error_change_fraction"] <= 0.10)
        ),
    }
    passes = bool(all(gates.values()))
    decision = (
        "retain_low_frequency_background_bridge_for_frozen_enkf_replay_preflight"
        if passes
        else "reject_low_frequency_background_bridge_at_direct_station_gate"
    )

    values.to_csv(OUTPUT / "station_year_values_allowed.csv", index=False)
    metrics.to_csv(OUTPUT / "station_period_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT / "aggregate_metrics.csv", index=False)
    plot_curves(values, eligibility, OUTPUT / "qcf_station_background_candidate_curves.png")

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "status": "complete",
        "experiment": "temperature_pmip_low_frequency_station_bridge_validation_v1",
        "decision": decision,
        "passes_all_frozen_gates": passes,
        "locked_1921_1939_numeric_values_accessed": 0,
        "locked_rows_skipped_before_numeric_decode": locked_rows_skipped,
        "candidate_definition": "archived posterior plus frozen PMIP low-frequency background correction; assimilation increment fixed",
        "qcf_primary_qcu_sensitivity": True,
        "published_bias_closure_max_degC": closure,
        "maximum_station_to_correction_source_distance_deg": max(distances.values()),
        "aggregate": aggregate_summary,
        "gates": gates,
        "all_declared_sources_unchanged": unchanged,
        "production_authorized": False,
        "enkf_run": False,
        "oggm_run": False,
        "interpretation_boundary": (
            "Passing supports a frozen EnKF replay preflight only. It does not show how the "
            "assimilation increment would respond to the corrected background."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# PMIP Low-Frequency Background vs Direct Stations",
        "",
        f"Decision: `{decision}`.",
        "",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"- {row.series} {row.period}: bias {row.original_equal_station_mean_bias_degC:+.3f} "
            f"to {row.candidate_equal_station_mean_bias_degC:+.3f} degC; median MAE change "
            f"{row.median_station_mae_improvement_fraction:+.1%}; median trend-error change "
            f"{row.median_trend_error_change_fraction:+.1%}; stations improving "
            f"{row.stations_with_mae_improvement_fraction:.1%}."
        )
    lines.extend([
        "",
        f"- Published qcf bias closure: `{closure:.4f} degC`.",
        f"- Maximum correction-source distance: `{max(distances.values()):.3f} degrees`.",
        f"- All frozen gates pass: `{str(passes).lower()}`.",
        "- No numeric value from 1921-1939 entered the calculation; no EnKF, production, or OGGM run occurred.",
        "",
        "This is a fixed-increment counterfactual. A passing result would justify proposing, not silently running, a frozen EnKF replay with the corrected background.",
    ])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

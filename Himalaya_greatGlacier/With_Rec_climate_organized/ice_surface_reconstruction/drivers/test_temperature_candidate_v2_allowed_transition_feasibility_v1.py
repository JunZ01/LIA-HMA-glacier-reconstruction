"""Test allowed-year tapers that remove Candidate v2 sealed-boundary jumps."""

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


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
FULL = LOOP / (
    "temperature_candidate_v2_full_posterior_materialization_v1/"
    "temperature_candidate_v2_full_posterior_allowed_years.nc"
)
FULL_SUMMARY = LOOP / "temperature_candidate_v2_full_posterior_materialization_v1/summary.json"
BOUNDARY = LOOP / "temperature_candidate_v2_sealed_boundary_transition_audit_v1/summary.json"
RELEASE = LOOP / (
    "temperature_candidate_v2_full_posterior_independent_verification_v2/"
    "RELEASE_ADDENDUM.json"
)
STATION = LOOP / (
    "temperature_background_candidate_v2_allowed_years_v4/"
    "station_year_candidate_values.csv"
)
STATION_AGGREGATE = LOOP / (
    "temperature_background_candidate_v2_allowed_years_v4/aggregate_metrics.csv"
)
FROZEN_CURVE = LOOP / (
    "temperature_candidate_v2_formal_freeze_v1/hma_temperature_candidate_v2_allowed.csv"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"
OUTPUT = LOOP / "temperature_candidate_v2_allowed_transition_feasibility_v1"

CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"
LOCKED = (1921, 1939)
PERIODS = {"1880-1900": (1880, 1900), "1901-1920": (1901, 1920), "1940-1950": (1940, 1950)}
TAPER_STARTS = (1880, 1890, 1900, 1910)
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def factor(years: np.ndarray, start: int | None) -> np.ndarray:
    result = np.ones(len(years), dtype=np.float64)
    if start is None:
        result[(years >= 1951)] = 0.0
        return result
    taper = (years > start) & (years <= 1920)
    result[taper] = (1920.0 - years[taper]) / (1920.0 - start)
    result[(years >= 1920)] = 0.0
    return result


def slope(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 8:
        return np.nan
    return float(linregress(years[valid], values[valid]).slope * 100.0)


def station_metrics(values: pd.DataFrame, branch: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_rows = []
    aggregate_rows = []
    for period, (start, end) in PERIODS.items():
        segment = values[values["year"].between(start, end)]
        current = []
        for station_id, group in segment.groupby("station_id"):
            observed = group["qcf_station_degC"].to_numpy(float)
            original = group["project_six_member_degC"].to_numpy(float)
            candidate = group[branch].to_numpy(float)
            years = group["year"].to_numpy(float)
            valid = np.isfinite(observed) & np.isfinite(original) & np.isfinite(candidate)
            if int(valid.sum()) < 5:
                continue
            original_trend = slope(years[valid], original[valid])
            candidate_trend = slope(years[valid], candidate[valid])
            observed_trend = slope(years[valid], observed[valid])
            row = {
                "branch": branch,
                "period": period,
                "station_id": station_id,
                "n_years": int(valid.sum()),
                "original_bias_degC": float(np.mean(original[valid] - observed[valid])),
                "candidate_bias_degC": float(np.mean(candidate[valid] - observed[valid])),
                "original_mae_degC": float(np.mean(np.abs(original[valid] - observed[valid]))),
                "candidate_mae_degC": float(np.mean(np.abs(candidate[valid] - observed[valid]))),
                "original_abs_trend_error": abs(original_trend - observed_trend),
                "candidate_abs_trend_error": abs(candidate_trend - observed_trend),
            }
            current.append(row)
            station_rows.append(row)
        frame = pd.DataFrame(current)
        original_bias = float(frame["original_bias_degC"].mean())
        candidate_bias = float(frame["candidate_bias_degC"].mean())
        original_mae = float(frame["original_mae_degC"].median())
        candidate_mae = float(frame["candidate_mae_degC"].median())
        original_trend = float(frame["original_abs_trend_error"].median())
        candidate_trend = float(frame["candidate_abs_trend_error"].median())
        aggregate_rows.append(
            {
                "branch": branch,
                "period": period,
                "station_count": len(frame),
                "original_equal_station_mean_bias_degC": original_bias,
                "candidate_equal_station_mean_bias_degC": candidate_bias,
                "absolute_bias_improvement_fraction": (
                    abs(original_bias) - abs(candidate_bias)
                ) / abs(original_bias),
                "original_median_station_mae_degC": original_mae,
                "candidate_median_station_mae_degC": candidate_mae,
                "median_station_mae_improvement_fraction": (
                    original_mae - candidate_mae
                ) / original_mae,
                "stations_improving_mae_fraction": float(
                    np.mean(frame["candidate_mae_degC"] < frame["original_mae_degC"])
                ),
                "original_median_abs_trend_error": original_trend,
                "candidate_median_abs_trend_error": candidate_trend,
                "median_trend_error_change_fraction": (
                    (candidate_trend - original_trend) / original_trend
                    if original_trend > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(station_rows), pd.DataFrame(aggregate_rows)


def weighted_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    return np.nansum(values * weights[None], axis=(1, 2)) / np.sum(weights)


def weighted_rms(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    return np.sqrt(np.nansum(values * values * weights[None], axis=(1, 2)) / np.sum(weights))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "TRANSITION_FEASIBILITY_CONTRACT.md"
    contract.write_text(
        "# Allowed-year transition feasibility\n\n"
        "Compare frozen Candidate v2 with four structurally fixed transition diagnostics. "
        "For starts 1880, 1890, 1900, and 1910, retain the frozen correction through the start, "
        "linearly taper it to exactly zero at 1920, and retain zero correction from 1940 onward. "
        "No weight is fitted and no branch may be promoted here. Require zero correction at all "
        "three exposed boundaries, unchanged 1500-1849 correction, frozen qcf early station gates, "
        "no 1940-1950 regression versus the original project, and report 50-year HMA mean/RMS "
        "departure from frozen Candidate v2. Do not read or generate 1921-1939. This is a new-"
        "candidate feasibility test only; Candidate v2 and production remain unchanged.\n",
        encoding="utf-8",
    )
    sources = {
        "full_candidate": FULL,
        "full_summary": FULL_SUMMARY,
        "boundary_audit": BOUNDARY,
        "independent_release": RELEASE,
        "station_values": STATION,
        "station_aggregate": STATION_AGGREGATE,
        "frozen_curve": FROZEN_CURVE,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "contract": contract,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in sources.items()}
    full_summary = json.loads(FULL_SUMMARY.read_text(encoding="utf-8"))
    boundary = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    if any(item["candidate_id"] != CANDIDATE_ID for item in [full_summary, boundary, release]):
        raise RuntimeError("candidate-id-chain-failed")
    if "requires_new_explicit_transition_contract" not in boundary["classification"]:
        raise RuntimeError("boundary-audit-does-not-require-transition")

    with Dataset(FULL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        common = np.asarray(dataset.variables["candidate_common_support_mask"][:], dtype=bool)
        correction = np.asarray(dataset.variables["tas_candidate_v2_correction"][:], dtype=np.float64)
    if np.any((years >= LOCKED[0]) & (years <= LOCKED[1])):
        raise RuntimeError("locked-year-in-field")
    stations = pd.read_csv(STATION)
    if stations["year"].between(*LOCKED).any():
        raise RuntimeError("locked-year-in-station-input")
    frozen_curve = pd.read_csv(FROZEN_CURVE)
    if frozen_curve["year"].between(*LOCKED).any():
        raise RuntimeError("locked-year-in-frozen-curve")

    branch_specs = {"frozen_candidate_v2": None}
    branch_specs.update({f"taper_{start}_to_1920_postgap_zero": start for start in TAPER_STARTS})
    station_tables = []
    aggregate_tables = []
    spatial_rows = []
    factor_rows = []
    early_mask = years <= 1920
    correction_early = correction[early_mask]
    years_early = years[early_mask]
    for branch, start in branch_specs.items():
        factors = factor(years, start)
        factor_rows.extend(
            {"branch": branch, "year": int(year), "correction_factor": float(value)}
            for year, value in zip(years, factors)
        )
        station_factors = factor(stations["year"].to_numpy(int), start)
        candidate_column = f"candidate_{branch}_degC"
        stations[candidate_column] = (
            stations["project_six_member_degC"]
            + station_factors * stations["candidate_background_correction_degC"]
        )
        station_table, aggregate_table = station_metrics(stations, candidate_column)
        station_table["candidate_family"] = branch
        aggregate_table["candidate_family"] = branch
        station_tables.append(station_table)
        aggregate_tables.append(aggregate_table)

        transitioned = correction_early * factors[early_mask, None, None]
        annual_change = np.diff(transitioned, axis=0)
        hma_change = np.abs(np.diff(weighted_mean(transitioned, common, lat)))
        hma_rms_change = weighted_rms(annual_change, common, lat)
        window_mean_departures = []
        window_rms_departures = []
        for window_start, window_end in WINDOWS:
            selection = (years_early >= window_start) & (years_early <= window_end)
            difference = transitioned[selection] - correction_early[selection]
            mean_field = np.mean(difference, axis=0)
            window_mean_departures.append(abs(float(weighted_mean(mean_field[None], common, lat)[0])))
            window_rms_departures.append(float(weighted_rms(mean_field[None], common, lat)[0]))
        spatial_rows.append(
            {
                "branch": branch,
                "taper_start_year": start,
                "correction_1920_hma_degC": float(
                    weighted_mean(transitioned[years_early == 1920], common, lat)[0]
                ),
                "correction_1940_hma_degC": float(
                    weighted_mean((correction[years == 1940] * factors[years == 1940, None, None]), common, lat)[0]
                ),
                "correction_1950_hma_degC": float(
                    weighted_mean((correction[years == 1950] * factors[years == 1950, None, None]), common, lat)[0]
                ),
                "correction_1951_hma_degC": float(
                    weighted_mean((correction[years == 1951] * factors[years == 1951, None, None]), common, lat)[0]
                ),
                "hma_internal_annual_change_max_abs_degC": float(np.max(hma_change)),
                "hma_spatial_rms_annual_change_max_degC": float(np.max(hma_rms_change)),
                "max_1500_1899_50year_hma_mean_departure_from_v2_degC": float(max(window_mean_departures)),
                "max_1500_1899_50year_hma_spatial_rms_departure_from_v2_degC": float(max(window_rms_departures)),
                "correction_1500_1849_exactly_unchanged": bool(
                    np.array_equal(
                        transitioned[years_early <= 1849], correction_early[years_early <= 1849]
                    )
                ),
            }
        )

    station_all = pd.concat(station_tables, ignore_index=True)
    aggregate = pd.concat(aggregate_tables, ignore_index=True)
    spatial = pd.DataFrame(spatial_rows)
    factors = pd.DataFrame(factor_rows)
    station_all.to_csv(OUTPUT / "station_period_transition_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT / "aggregate_transition_metrics.csv", index=False)
    spatial.to_csv(OUTPUT / "spatial_transition_diagnostics.csv", index=False)
    factors.to_csv(OUTPUT / "transition_factors_allowed_years.csv", index=False)

    original_aggregate = pd.read_csv(STATION_AGGREGATE).set_index("period")
    v2_aggregate = aggregate[aggregate["candidate_family"].eq("frozen_candidate_v2")].set_index("period")
    metric_closure = max(
        float(
            np.max(
                np.abs(
                    v2_aggregate[column].to_numpy(float)
                    - original_aggregate.loc[v2_aggregate.index, column].to_numpy(float)
                )
            )
        )
        for column in [
            "candidate_equal_station_mean_bias_degC",
            "candidate_median_station_mae_degC",
            "median_station_mae_improvement_fraction",
            "stations_improving_mae_fraction",
            "candidate_median_abs_trend_error",
            "median_trend_error_change_fraction",
        ]
    )
    if metric_closure > 1e-12:
        raise RuntimeError(f"frozen-station-metric-closure-failed:{metric_closure}")

    branch_rows = []
    for branch, rows in aggregate.groupby("candidate_family", sort=False):
        indexed = rows.set_index("period")
        early = indexed.loc[["1880-1900", "1901-1920"]]
        modern = indexed.loc["1940-1950"]
        spatial_row = spatial.set_index("branch").loc[branch]
        gates = {
            "early_absolute_bias_improves": bool((early["absolute_bias_improvement_fraction"] > 0).all()),
            "early_median_station_mae_improves": bool((early["median_station_mae_improvement_fraction"] > 0).all()),
            "early_majority_stations_improve": bool((early["stations_improving_mae_fraction"] >= 0.5).all()),
            "early_median_trend_error_regression_at_most_10pct": bool((early["median_trend_error_change_fraction"] <= 0.10).all()),
            "modern_median_station_mae_no_regression": bool(modern["median_station_mae_improvement_fraction"] >= 0),
            "modern_absolute_bias_no_regression": bool(modern["absolute_bias_improvement_fraction"] >= 0),
            "correction_zero_at_1920": abs(float(spatial_row["correction_1920_hma_degC"])) <= 1e-12,
            "correction_zero_at_1940": abs(float(spatial_row["correction_1940_hma_degC"])) <= 1e-12,
            "correction_zero_at_1950": abs(float(spatial_row["correction_1950_hma_degC"])) <= 1e-12,
            "correction_zero_at_1951": abs(float(spatial_row["correction_1951_hma_degC"])) <= 1e-12,
            "correction_1500_1849_exactly_unchanged": bool(spatial_row["correction_1500_1849_exactly_unchanged"]),
        }
        branch_rows.append(
            {
                "branch": branch,
                "passes_all_transition_and_station_gates": bool(all(gates.values())),
                "gates": gates,
                "bias_improvement_1880_1900": float(early.loc["1880-1900", "absolute_bias_improvement_fraction"]),
                "bias_improvement_1901_1920": float(early.loc["1901-1920", "absolute_bias_improvement_fraction"]),
                "mae_improvement_1880_1900": float(early.loc["1880-1900", "median_station_mae_improvement_fraction"]),
                "mae_improvement_1901_1920": float(early.loc["1901-1920", "median_station_mae_improvement_fraction"]),
                "max_50year_hma_mean_departure_from_v2_degC": float(spatial_row["max_1500_1899_50year_hma_mean_departure_from_v2_degC"]),
                "max_50year_hma_spatial_rms_departure_from_v2_degC": float(spatial_row["max_1500_1899_50year_hma_spatial_rms_departure_from_v2_degC"]),
                "hma_internal_annual_change_max_abs_degC": float(spatial_row["hma_internal_annual_change_max_abs_degC"]),
                "hma_spatial_rms_annual_change_max_degC": float(spatial_row["hma_spatial_rms_annual_change_max_degC"]),
            }
        )
    feasible = [row for row in branch_rows if row["passes_all_transition_and_station_gates"]]
    diagnostic_feasible = [row for row in feasible if row["branch"] != "frozen_candidate_v2"]
    classification = (
        "at_least_one_allowed_year_taper_removes_all_boundary_jumps_and_retains_station_gates"
        if diagnostic_feasible
        else "no_tested_allowed_year_taper_jointly_retains_station_gates_and_removes_boundary_jumps"
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]
    correction_curve = frozen_curve.set_index("year")["candidate_background_correction_degC"]
    for color, (branch, start) in zip(colors, branch_specs.items()):
        current_years = frozen_curve["year"].to_numpy(int)
        current = correction_curve.to_numpy(float) * factor(current_years, start)
        selected = (current_years >= 1850) & (current_years <= 1955)
        for left, right in [(1850, 1920), (1940, 1955)]:
            segment = selected & (current_years >= left) & (current_years <= right)
            axes[0].plot(current_years[segment], current[segment], color=color, lw=1.8, label=branch if left == 1850 else None)
    axes[0].axvspan(1921, 1939, color="#EEEEEE", alpha=0.9)
    axes[0].set_ylabel("HMA correction (degC)")
    axes[0].set_title("Allowed-year transition diagnostics")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    plot_rows = [row for row in branch_rows if row["branch"] != "frozen_candidate_v2"]
    x = np.arange(len(plot_rows))
    axes[1].bar(x - 0.18, [row["bias_improvement_1880_1900"] * 100 for row in plot_rows], 0.36, color="#0072B2", label="1880-1900")
    axes[1].bar(x + 0.18, [row["bias_improvement_1901_1920"] * 100 for row in plot_rows], 0.36, color="#009E73", label="1901-1920")
    axes[1].axhline(0, color="#555555", lw=0.8)
    axes[1].set_xticks(x, [row["branch"].replace("taper_", "").replace("_postgap_zero", "") for row in plot_rows], rotation=25, ha="right")
    axes[1].set_ylabel("Absolute-bias improvement (%)")
    axes[1].set_title("Direct qcf station bias retention")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure = OUTPUT / "candidate_v2_allowed_transition_feasibility.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "schema": "hma-temperature-candidate-v2-allowed-transition-feasibility-v1",
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "classification": classification,
        "frozen_v2_station_metric_reproduction_max_abs": metric_closure,
        "branches": branch_rows,
        "diagnostic_feasible_branch_count": len(diagnostic_feasible),
        "diagnostic_feasible_branches": [row["branch"] for row in diagnostic_feasible],
        "interpretation": (
            "Feasibility does not promote a branch. A transition branch changes the frozen "
            "Candidate v2 temporal operator and would require a separately preregistered "
            "Candidate v3 validation contract."
        ),
        "locked_1921_1939_numeric_values_accessed": 0,
        "locked_1921_1939_values_generated": 0,
        "all_declared_sources_unchanged": unchanged,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    lines = ["# Candidate v2 allowed-year transition feasibility", "", f"Classification: `{classification}`.", ""]
    for row in branch_rows:
        lines.append(
            f"- `{row['branch']}`: joint pass `{str(row['passes_all_transition_and_station_gates']).lower()}`; "
            f"bias improvement `{row['bias_improvement_1880_1900']:+.1%}` / "
            f"`{row['bias_improvement_1901_1920']:+.1%}`; maximum 50-year HMA mean departure "
            f"`{row['max_50year_hma_mean_departure_from_v2_degC']:.4f} degC`."
        )
    lines.extend([
        "",
        "No branch is promoted. Candidate v2 and production remain unchanged; no sealed-period value was read or generated.",
    ])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

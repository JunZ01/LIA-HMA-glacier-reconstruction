from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER = Path(__file__).resolve()
DRIVER_DIR = DRIVER.parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import temperature_candidate_b2_stage_c_core_v1 as core


ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
DEFAULT_INPUT = LOOP / (
    "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v2/"
    "candidate_b2_stage_c_real_proxy_output_v1.npz"
)
DEFAULT_OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_evaluation_v1"
EARLY_STATIONS = LOOP / (
    "temperature_station_annual_anomaly_benchmark_v4/"
    "station_annual_anomalies_and_products.csv"
)
MODERN_STATIONS = LOOP / (
    "temperature_reconstruction_external_instrumental_benchmark_v2/"
    "station_year_reconstruction_values.csv"
)
MONTHLY_PANEL = LOOP / (
    "round_277_direct_temperature_station_panel_1850_1950_v2/"
    "monthly_temperature_panel.csv"
)
MONTHLY_CLIM = LOOP / (
    "temperature_station_annual_anomaly_benchmark_v4/"
    "station_monthly_climatology_1961_1990.csv"
)
EVIDENCE = LOOP / "temperature_1850_1950_low_frequency_evidence_matrix_v13/evidence_matrix.csv"
LONG_CURVES = LOOP / (
    "temperature_background_candidate_v2_longterm_hma_curve_v1/"
    "hma_temperature_before_after_ref_instrument_allowed.csv"
)
BRANCHES = {
    "balanced_background": "no_proxy_balanced_background",
    "candidate_proxy6": "proxy6_posterior_temperature_anomaly",
    "candidate_proxy9": "proxy9_posterior_temperature_anomaly",
    "candidate_joint": "candidate_b2_joint_temperature_anomaly",
}
SEASONS = {
    "calendar_DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def paired_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    obs = np.asarray(observed, dtype=float)[valid]
    pred = np.asarray(predicted, dtype=float)[valid]
    if len(obs) == 0:
        return {name: np.nan for name in ("n", "rmse_degC", "mae_degC", "bias_degC", "correlation_r", "sign_agreement")}
    error = pred - obs
    if len(obs) >= 3 and np.std(obs, ddof=1) > 0 and np.std(pred, ddof=1) > 0:
        correlation = float(pearsonr(obs, pred).statistic)
    else:
        correlation = np.nan
    sign = float(np.mean(np.sign(obs) == np.sign(pred)))
    return {
        "n": int(len(obs)),
        "rmse_degC": float(np.sqrt(np.mean(error**2))),
        "mae_degC": float(np.mean(np.abs(error))),
        "bias_degC": float(np.mean(error)),
        "correlation_r": correlation,
        "sign_agreement": sign,
    }


def trend(values: pd.DataFrame, column: str) -> float:
    valid = values[["year", column]].dropna()
    if len(valid) < 10:
        return np.nan
    return float(linregress(valid["year"], valid[column]).slope * 100.0)


def region_group(longitude: float) -> str:
    if longitude < 80.0:
        return "West"
    if longitude < 90.0:
        return "Central"
    return "East"


def own_baseline(field: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1961) & (years <= 1990)
    if int(baseline.sum()) != 30:
        raise RuntimeError("candidate-baseline-year-count")
    return field - field[baseline].mean(axis=0, keepdims=True)


def load_station_panel() -> pd.DataFrame:
    early = pd.read_csv(EARLY_STATIONS)
    early = early.rename(
        columns={
            "qcf_station_degC": "qcf",
            "qcu_station_degC": "qcu",
            "project_six_member_degC": "archived_project",
        }
    )
    modern = pd.read_csv(MODERN_STATIONS)
    modern = modern.rename(
        columns={
            "qcf_station_degC": "qcf",
            "qcu_station_degC": "qcu",
            "project_degC": "archived_project",
        }
    )
    columns = [
        "station_id", "station_name", "year", "latitude", "longitude",
        "qcf", "qcu", "archived_project",
    ]
    panel = pd.concat([early[columns], modern[columns]], ignore_index=True)
    panel = panel.drop_duplicates(["station_id", "year"], keep="last")
    panel = panel[~panel["year"].between(1921, 1939)].copy()
    panel["region_group"] = panel["longitude"].map(region_group)
    return panel


def add_candidate_values(
    panel: pd.DataFrame,
    years: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    fields: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    year_index = {int(year): index for index, year in enumerate(years)}
    mappings = []
    station_to_cell = {}
    for station in panel[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates().itertuples(index=False):
        distance = np.hypot(latitude - float(station.latitude), longitude - float(station.longitude))
        cell = int(np.argmin(distance))
        station_to_cell[station.station_id] = cell
        mappings.append(
            {
                "station_id": station.station_id,
                "station_name": station.station_name,
                "station_latitude": float(station.latitude),
                "station_longitude": float(station.longitude),
                "state_cell_index": cell,
                "state_latitude": float(latitude[cell]),
                "state_longitude": float(longitude[cell]),
                "distance_deg": float(distance[cell]),
            }
        )
    output = panel.copy()
    for branch, field in fields.items():
        values = []
        for row in output.itertuples(index=False):
            index = year_index.get(int(row.year))
            values.append(np.nan if index is None else float(field[index, station_to_cell[row.station_id]]))
        output[branch] = values
    return output, pd.DataFrame(mappings)


def decadal_pairs(frame: pd.DataFrame, observation: str, prediction: str) -> pd.DataFrame:
    rows = []
    for start in range(1940, 2000, 10):
        subset = frame[frame["year"].between(start, start + 9)]
        for station_id, group in subset.groupby("station_id"):
            valid = group[[observation, prediction]].dropna()
            if len(valid) < 6:
                continue
            rows.append(
                {
                    "station_id": station_id,
                    "start_year": start,
                    "end_year": start + 9,
                    "observed": float(valid[observation].mean()),
                    "predicted": float(valid[prediction].mean()),
                    "valid_year_count": int(len(valid)),
                }
            )
    return pd.DataFrame(rows)


def metric_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    branches = [*BRANCHES, "archived_project"]
    aggregate_rows = []
    region_rows = []
    early_rows = []
    modern = panel[panel["year"].between(1940, 2000)]
    early = panel[panel["year"].between(1850, 1920)]
    for observation in ("qcf", "qcu"):
        for branch in branches:
            valid = modern[["year", "station_id", "region_group", observation, branch]].dropna()
            annual = paired_metrics(valid[observation].to_numpy(), valid[branch].to_numpy())
            decades = decadal_pairs(modern, observation, branch)
            decadal = paired_metrics(decades.get("observed", pd.Series(dtype=float)).to_numpy(), decades.get("predicted", pd.Series(dtype=float)).to_numpy())
            matched = valid.groupby("year")[[observation, branch]].mean().dropna()
            matched_metric = paired_metrics(matched[observation].to_numpy(), matched[branch].to_numpy())
            aggregate_rows.append(
                {
                    "period": "1940-2000",
                    "observation": observation,
                    "branch": branch,
                    **{f"annual_{key}": value for key, value in annual.items()},
                    **{f"decadal_{key}": value for key, value in decadal.items()},
                    "matched_station_mean_correlation_r": matched_metric["correlation_r"],
                    "observed_trend_degC_per_century": trend(valid, observation),
                    "predicted_trend_degC_per_century": trend(valid, branch),
                    "absolute_trend_error_degC_per_century": abs(trend(valid, branch) - trend(valid, observation)),
                }
            )
            early_valid = early[["year", "station_id", "region_group", observation, branch]].dropna()
            early_metric = paired_metrics(early_valid[observation].to_numpy(), early_valid[branch].to_numpy())
            early_rows.append(
                {
                    "period": "1850-1920",
                    "observation": observation,
                    "branch": branch,
                    **early_metric,
                    "observed_trend_degC_per_century": trend(early_valid, observation),
                    "predicted_trend_degC_per_century": trend(early_valid, branch),
                }
            )
            for region, group in valid.groupby("region_group"):
                metric = paired_metrics(group[observation].to_numpy(), group[branch].to_numpy())
                region_rows.append(
                    {
                        "period": "1940-2000",
                        "region_group": region,
                        "observation": observation,
                        "branch": branch,
                        **metric,
                    }
                )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(region_rows), pd.DataFrame(early_rows)


def seasonal_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    monthly = pd.read_csv(MONTHLY_PANEL)
    climatology = pd.read_csv(MONTHLY_CLIM)
    monthly = monthly[~monthly["year"].between(1921, 1939)].copy()
    monthly = monthly.merge(climatology, on=["station_id", "series", "month"], how="left")
    monthly["monthly_anomaly"] = monthly["analysis_temperature_c"] - monthly["baseline_temperature_c"]
    branch_values = panel.set_index(["station_id", "year"])[[*BRANCHES, "archived_project"]]
    rows = []
    for season, months in SEASONS.items():
        subset = monthly[(monthly["series"] == "qcf") & monthly["month"].isin(months)]
        season_values = subset.groupby(["station_id", "year"])["monthly_anomaly"].agg(["mean", "count"]).reset_index()
        season_values = season_values[season_values["count"] == 3]
        for branch in [*BRANCHES, "archived_project"]:
            pairs = []
            for row in season_values.itertuples(index=False):
                key = (row.station_id, int(row.year))
                if key in branch_values.index:
                    value = branch_values.loc[key, branch]
                    if np.isfinite(value):
                        pairs.append((float(row.mean), float(value)))
            observed = np.asarray([pair[0] for pair in pairs], dtype=float)
            predicted = np.asarray([pair[1] for pair in pairs], dtype=float)
            rows.append({"diagnostic_role": "annual_target_vs_station_season_mapping_sensitivity", "season": season, "branch": branch, **paired_metrics(observed, predicted)})
    return pd.DataFrame(rows)


def plot_curves(curves: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(13.2, 6.4), constrained_layout=True)
    colors = {
        "candidate_joint": "#C33C2E", "candidate_proxy6": "#2878B5",
        "candidate_proxy9": "#2E8B57", "balanced_background": "#777777",
    }
    for start, end in ((1500, 1920), (1940, 2000)):
        segment = curves[curves["year"].between(start, end)]
        for branch, color in colors.items():
            axis.plot(segment["year"], segment[branch], color=color, linewidth=2.0 if branch == "candidate_joint" else 1.0, alpha=1.0 if branch == "candidate_joint" else 0.75, label=branch if start == 1500 else None)
        axis.plot(segment["year"], segment["original_project_degC"], color="#111111", linewidth=1.2, label="archived_project" if start == 1500 else None)
        axis.plot(segment["year"], segment["external_ref_mean_degC"], color="#7A4EAB", linewidth=1.0, label="external_reference_mean" if start == 1500 else None)
        axis.plot(segment["year"], segment["instrument_mean_degC"], color="#E08B27", linewidth=1.4, label="instrumental_mean" if start == 1500 else None)
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    axis.set(xlabel="Year", ylabel="Temperature anomaly (degC)", title="Candidate B2 Stage C: reconstruction, background, references, and instruments")
    axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def main(input_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(input_path, allow_pickle=False) as result:
        years = np.asarray(result["years"], dtype=np.int32)
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("sealed-year-present")
        latitude = np.asarray(result["retained_state_latitude"], dtype=float)
        longitude = np.asarray(result["retained_state_longitude"], dtype=float)
        weights = np.asarray(result["hma_weights"], dtype=float)
        fields = {
            branch: own_baseline(np.asarray(result[array], dtype=float), years)
            for branch, array in BRANCHES.items()
        }
    panel = load_station_panel()
    panel, mapping = add_candidate_values(panel, years, latitude, longitude, fields)
    aggregate, regional, early = metric_tables(panel)
    seasonal = seasonal_diagnostics(panel)
    evidence = pd.read_csv(EVIDENCE)
    if evidence["sealed_period_values_used"].astype(bool).any():
        raise RuntimeError("evidence-matrix-sealed-value-flag")

    curve_rows = {"year": years}
    valid_weight = weights > 0
    for branch, field in fields.items():
        curve_rows[branch] = np.average(field[:, valid_weight], axis=1, weights=weights[valid_weight])
    curves = pd.DataFrame(curve_rows).merge(pd.read_csv(LONG_CURVES), on="year", how="left", validate="one_to_one")

    primary = aggregate[aggregate["observation"].eq("qcf")].set_index("branch")
    joint = primary.loc["candidate_joint"]
    background = primary.loc["balanced_background"]
    archived = primary.loc["archived_project"]
    comparator_annual_rmse = min(background["annual_rmse_degC"], archived["annual_rmse_degC"])
    comparator_annual_r = max(background["annual_correlation_r"], archived["annual_correlation_r"])
    comparator_decadal_rmse = min(background["decadal_rmse_degC"], archived["decadal_rmse_degC"])
    comparator_decadal_r = max(background["decadal_correlation_r"], archived["decadal_correlation_r"])
    gates = {
        "joint_decadal_rmse_beats_both_comparators": bool(joint["decadal_rmse_degC"] < comparator_decadal_rmse),
        "joint_decadal_correlation_beats_both_comparators": bool(joint["decadal_correlation_r"] > comparator_decadal_r),
        "joint_annual_rmse_within_5pct_of_best_comparator": bool(joint["annual_rmse_degC"] <= 1.05 * comparator_annual_rmse),
        "joint_annual_correlation_within_0p05_of_best_comparator": bool(joint["annual_correlation_r"] >= comparator_annual_r - 0.05),
    }
    instrumental_gate_pass = all(gates.values())

    panel.to_csv(output / "station_year_candidate_b2_values.csv", index=False)
    mapping.to_csv(output / "station_to_candidate_state_mapping.csv", index=False)
    aggregate.to_csv(output / "instrumental_aggregate_metrics.csv", index=False)
    regional.to_csv(output / "instrumental_west_central_east_metrics.csv", index=False)
    early.to_csv(output / "early_1850_1920_station_metrics.csv", index=False)
    seasonal.to_csv(output / "season_mapping_sensitivity_metrics.csv", index=False)
    evidence.to_csv(output / "independent_evidence_matrix_snapshot.csv", index=False)
    curves.to_csv(output / "candidate_b2_long_term_hma_curves_allowed.csv", index=False)
    plot_curves(curves, output / "candidate_b2_long_term_hma_curves_allowed.png")

    summary = {
        "experiment": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "instrumental_gate_pass": instrumental_gate_pass,
        "frozen_instrumental_gates": gates,
        "qcf_primary_metrics": joint.to_dict(),
        "qcu_role": "homogenization sensitivity only, not an independent vote",
        "seasonal_role": "annual target versus station-season mapping sensitivity, not seasonal truth validation",
        "independent_evidence_row_count": int(len(evidence)),
        "sealed_1921_1939_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
        "decision_boundary": "instrumental pass is necessary but not sufficient; independent evidence and regional diagnostics require final adjudication before production",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "stage_c_output": input_path,
        "early_stations": EARLY_STATIONS,
        "modern_stations": MODERN_STATIONS,
        "monthly_panel": MONTHLY_PANEL,
        "monthly_climatology": MONTHLY_CLIM,
        "evidence_matrix": EVIDENCE,
        "allowed_long_curves": LONG_CURVES,
        "driver": DRIVER,
    }
    manifest = {name: {"path": str(path), "sha256": core.sha256(path)} for name, path in sources.items()}
    manifest["sealed_period_values_used"] = False
    manifest["prohibited_source_accessed"] = False
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Candidate B2 Stage C evaluation\n\n"
        f"Frozen instrumental gate pass: `{instrumental_gate_pass}`.\n\n"
        + "\n".join(f"- `{name}`: `{value}`" for name, value in gates.items())
        + "\n\nThis result does not modify or automatically replace the production reconstruction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.input, args.output)

"""Validate forced-background member counterfactuals over 1940-1950."""

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

import diagnose_temperature_forced_background_members_at_stations_v1 as base


PROJECT = base.PROJECT
RESULTS = base.RESULTS
LOOP = base.LOOP
OUTPUT = LOOP / "temperature_forced_background_member_station_1940_1950_validation_v1"
STATION_VALUES = LOOP / (
    "temperature_station_annual_anomaly_benchmark_v4/"
    "station_annual_anomalies_and_products.csv"
)
STATION_COVERAGE = LOOP / (
    "temperature_station_annual_anomaly_benchmark_v4/station_baseline_coverage.csv"
)
DRIVER = Path(__file__).resolve()
ANALYSIS_YEARS = np.arange(1940, 1951)
BASELINE_YEARS = np.arange(1961, 1991)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: str | None) -> float:
    if value is None or value.strip() == "":
        return np.nan
    return float(value)


def read_station_values() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with STATION_VALUES.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            year = int(raw["year"])
            if 1921 <= year <= 1939:
                continue
            if year not in ANALYSIS_YEARS:
                continue
            rows.append({
                "station_id": raw["station_id"],
                "station_name": raw["station_name"],
                "latitude": float(raw["latitude"]),
                "longitude": float(raw["longitude"]),
                "year": year,
                "qcf_station_degC": finite(raw["qcf_station_degC"]),
                "project_six_member_degC": finite(raw["project_six_member_degC"]),
            })
    coverage = pd.read_csv(STATION_COVERAGE)
    eligible = set(
        coverage.loc[
            coverage["series"].eq("qcf") & coverage["baseline_eligible"].astype(bool),
            "station_id",
        ].astype(str)
    )
    return pd.DataFrame(rows).loc[lambda frame: frame["station_id"].isin(eligible)].copy()


def slope(year: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 8:
        return np.nan
    return float(linregress(year[valid], values[valid]).slope * 100.0)


def metrics(values: pd.DataFrame, branches: list[str]) -> pd.DataFrame:
    rows = []
    for branch in branches:
        station_rows = []
        for station_id, group in values.groupby("station_id"):
            observed = group["qcf_station_degC"].to_numpy(float)
            predicted = group[branch].to_numpy(float)
            year = group["year"].to_numpy(float)
            valid = np.isfinite(observed) & np.isfinite(predicted)
            if int(valid.sum()) < 5:
                continue
            observed_trend = slope(year[valid], observed[valid])
            predicted_trend = slope(year[valid], predicted[valid])
            station_rows.append({
                "station_id": station_id,
                "bias": float(np.mean(predicted[valid] - observed[valid])),
                "mae": float(np.mean(np.abs(predicted[valid] - observed[valid]))),
                "abs_trend_error": abs(predicted_trend - observed_trend),
            })
        frame = pd.DataFrame(station_rows)
        rows.append({
            "branch": branch,
            "station_count": len(frame),
            "equal_station_mean_bias_degC": float(frame["bias"].mean()),
            "median_station_mae_degC": float(frame["mae"].median()),
            "median_abs_trend_error_degC_per_century": float(frame["abs_trend_error"].median()),
            "stations_with_positive_bias_fraction": float(np.mean(frame["bias"] > 0)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "EXPERIMENT_CONTRACT.md"
    contract.write_text(
        "# 1940-1950 forced-background validation\n\n"
        "Apply the unchanged archived, ModE-Sim, MPI, MIROC, three-model-mean, and equal-half "
        "background definitions to qcf stations over 1940-1950. Preserve the archived posterior "
        "increment. Use 1961-1990 anomalies. Skip 1921-1939 before numeric decoding. No model "
        "weight is fitted and no EnKF, production, or OGGM run is authorized.\n",
        encoding="utf-8",
    )
    sources = {
        "station_values": STATION_VALUES, "station_coverage": STATION_COVERAGE,
        "catalog": base.CATALOG, "mode_sim": base.MODE_SIM, "prior": base.PRIOR,
        "indices": base.INDICES, "proxy6": base.PROXY6, "proxy9": base.PROXY9,
        "grid": base.GRID, "facts": base.FACTS, "driver": DRIVER, "contract": contract,
    }
    before = {name: sha256(path) for name, path in sources.items()}

    base.ANALYSIS_YEARS = ANALYSIS_YEARS
    base.BASELINE_YEARS = BASELINE_YEARS
    base.RECOVERY_YEARS = np.r_[ANALYSIS_YEARS, BASELINE_YEARS]
    with Dataset(base.GRID) as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        mask = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
    station_values = read_station_values()
    stations = station_values[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates()

    background = base.archived_background_fields(lat, lon)[:, mask]
    model_grids = {"mode_sim": base.annual_mode_sim_fields(lat, lon, mask)}
    catalog = json.loads(base.CATALOG.read_text(encoding="utf-8"))
    request_audit = []
    for model, branch in [("MPI-ESM1-2-LR", "mpi"), ("MIROC-ES2L", "miroc")]:
        model_grids[branch], audit = base.annual_remote_fields(
            model, [item for item in catalog if item["model"] == model], lat, lon, mask
        )
        request_audit.extend(audit)
    model_grids["forced_mean"] = np.mean(np.stack(list(model_grids.values())), axis=0)

    extracted = base.grid_points_to_stations(background, lat, lon, mask, stations).rename(
        columns={"value": "archived_background_degC", "source_distance_deg": "background_source_distance_deg"}
    )
    for name, fields in model_grids.items():
        frame = base.grid_points_to_stations(fields, lat, lon, mask, stations).rename(
            columns={"value": f"{name}_background_degC", "source_distance_deg": f"{name}_source_distance_deg"}
        )
        extracted = extracted.merge(frame, on=["station_id", "year"], validate="one_to_one")
    values = station_values.merge(extracted, on=["station_id", "year"], validate="one_to_one")
    values["archived_increment_degC"] = values["project_six_member_degC"] - values["archived_background_degC"]
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
    result = metrics(values, branches)
    result.to_csv(OUTPUT / "branch_metrics_1940_1950.csv", index=False)
    values.to_csv(OUTPUT / "station_year_background_members_1940_1950.csv", index=False)
    pd.DataFrame(request_audit).to_csv(OUTPUT / "remote_request_audit.csv", index=False)

    indexed = result.set_index("branch")
    original_mae = float(indexed.loc["project_six_member_degC", "median_station_mae_degC"])
    branch_changes = {
        branch: float((original_mae - row.median_station_mae_degC) / original_mae)
        for branch, row in indexed.iterrows() if branch != "project_six_member_degC"
    }
    no_full_model_material_regression = all(value >= -0.10 for value in branch_changes.values())
    best = str(result.loc[result["median_station_mae_degC"].idxmin(), "branch"])

    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.bar(np.arange(len(result)), result["median_station_mae_degC"], color=["#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#0072B2", "#E69F00"])
    axis.set_xticks(np.arange(len(result)), [name.replace("fixed_increment_", "").replace("_degC", "") for name in result["branch"]], rotation=35, ha="right")
    axis.set_ylabel("Median station MAE (degC)")
    axis.grid(axis="y", color="#DDDDDD", lw=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "forced_background_member_mae_1940_1950.png", dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "status": "complete",
        "experiment": "temperature_forced_background_member_station_1940_1950_validation_v1",
        "best_branch_by_median_station_mae": best,
        "median_station_mae_improvement_fraction_vs_original": branch_changes,
        "no_forced_branch_regresses_median_mae_by_more_than_10pct": no_full_model_material_regression,
        "locked_1921_1939_numeric_values_accessed": 0,
        "remote_locked_months_requested": 0,
        "all_declared_sources_unchanged": unchanged,
        "production_authorized": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    lines = ["# 1940-1950 Forced-Background Validation", "", f"- Best branch: `{best}`."]
    for branch, change in branch_changes.items():
        lines.append(f"- `{branch}` median station MAE change: `{change:+.1%}`.")
    lines.extend([
        f"- No branch regresses median MAE by more than 10%: `{str(no_full_model_material_regression).lower()}`.",
        "- No 1921-1939 value, EnKF, production file, or OGGM run was used.",
    ])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

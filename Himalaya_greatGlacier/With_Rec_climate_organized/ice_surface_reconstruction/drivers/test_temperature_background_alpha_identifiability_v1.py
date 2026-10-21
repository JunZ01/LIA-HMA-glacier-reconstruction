from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "temperature_background_alpha_identifiability_v1"

PUBLIC = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
ALPHA_ONE = LOOP / (
    "temperature_background_candidate_v2_allowed_years_v4/"
    "temperature_background_candidate_v2_allowed_years.nc"
)
EARLY_STATIONS = LOOP / (
    "temperature_background_candidate_v2_allowed_years_v4/"
    "station_year_candidate_values.csv"
)
MODERN_STATIONS = LOOP / (
    "temperature_four_background_scheme_evaluation_v2/"
    "modern_station_year_scheme_values.csv"
)
LAKE_POINTS = LOOP / "temperature_original_lake_author_reconstructions_v2/original_author_safe_points.csv"
AWENG_POINTS = LOOP / "temperature_noaa_hma_additional_physical_records_v1/additional_physical_safe_points.csv"
NGAMRING_POINTS = LOOP / "temperature_noaa_ngamring_original_v1/safe_ngamring_maat_points.csv"
HAIZI_WINDOWS = LOOP / (
    "round_216_haizi_annual_temperature_locked_safe_comparison_v1/"
    "temperature_50year_windows_1500_1899.csv"
)

TRAIN_END = 1849
LOCKED = (1921, 1939)
ALPHA_GRID = np.linspace(0.0, 1.0, 1001)
FULL_ALPHA_CONSTRAINT_CLOSURE = 0.07142772029594613


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values[(values < -9000) | ~np.isfinite(values)] = np.nan
    return values


def area_mean(fields: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(fields) & mask[None, :, :]
    numerator = np.nansum(np.where(valid, fields * weights[None, :, :], 0.0), axis=(1, 2))
    denominator = np.sum(np.where(valid, weights[None, :, :], 0.0), axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(len(fields), np.nan), where=denominator > 0)


def nearest_index(lat: np.ndarray, lon: np.ndarray, support: np.ndarray, site_lat: float, site_lon: float) -> tuple[int, int, float]:
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    distance = np.sqrt((yy - site_lat) ** 2 + ((xx - site_lon) * np.cos(np.deg2rad(site_lat))) ** 2)
    distance[~support] = np.inf
    flat = int(np.argmin(distance))
    iy, ix = np.unravel_index(flat, distance.shape)
    return int(iy), int(ix), float(distance[iy, ix])


def sample_years(series: np.ndarray, years: np.ndarray, requested: np.ndarray) -> np.ndarray:
    lookup = {int(year): value for year, value in zip(years, series)}
    return np.asarray([lookup.get(int(round(year)), np.nan) for year in requested], dtype=float)


def centered(values: np.ndarray) -> np.ndarray:
    return values - np.nanmean(values)


def alpha_fit(y: np.ndarray, project: np.ndarray, correction: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(project) & np.isfinite(correction)
    if valid.sum() < 2:
        return np.nan
    target = centered(y[valid]) - centered(project[valid])
    delta = centered(correction[valid])
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return np.nan
    return float(np.dot(delta, target) / denominator)


def record_metrics(record: dict, alpha: float) -> dict:
    y = np.asarray(record["external"], dtype=float)
    project = np.asarray(record["project"], dtype=float)
    correction = np.asarray(record["correction"], dtype=float)
    valid = np.isfinite(y) & np.isfinite(project) & np.isfinite(correction)
    y = centered(y[valid])
    candidate = centered(project[valid] + alpha * correction[valid])
    residual = candidate - y
    variance = float(np.mean(np.square(y)))
    correlation = float(np.corrcoef(y, candidate)[0, 1]) if len(y) >= 3 else np.nan
    return {
        "rmse_degC": float(np.sqrt(np.mean(np.square(residual)))),
        "normalized_mse": float(np.mean(np.square(residual)) / variance) if variance > 0 else np.nan,
        "correlation": correlation,
    }


def station_period_metrics(frame: pd.DataFrame, alpha: float, start: int, end: int) -> dict:
    part = frame[frame["year"].between(start, end) & frame["qcf_station_degC"].notna()].copy()
    part["candidate"] = part["project_six_member_degC"] + alpha * part["low_frequency_correction_degC"]
    rows = []
    for station_id, group in part.groupby("station_id"):
        valid = group[["qcf_station_degC", "candidate", "project_six_member_degC"]].notna().all(axis=1)
        group = group.loc[valid]
        if len(group) < 2:
            continue
        observed = group["qcf_station_degC"].to_numpy(float)
        candidate = group["candidate"].to_numpy(float)
        original = group["project_six_member_degC"].to_numpy(float)
        years = group["year"].to_numpy(float)
        rows.append({
            "station_id": station_id,
            "bias": float(np.mean(candidate - observed)),
            "mae": float(np.mean(np.abs(candidate - observed))),
            "original_mae": float(np.mean(np.abs(original - observed))),
            "trend_error": float(abs(np.polyfit(years, candidate, 1)[0] - np.polyfit(years, observed, 1)[0]) * 100.0),
            "original_trend_error": float(abs(np.polyfit(years, original, 1)[0] - np.polyfit(years, observed, 1)[0]) * 100.0),
        })
    metrics = pd.DataFrame(rows)
    return {
        "period": f"{start}-{end}",
        "alpha": float(alpha),
        "station_count": int(len(metrics)),
        "equal_station_mean_bias_degC": float(metrics["bias"].mean()),
        "median_station_mae_degC": float(metrics["mae"].median()),
        "median_abs_trend_error_degC_per_century": float(metrics["trend_error"].median()),
        "stations_improving_mae_fraction": float((metrics["mae"] < metrics["original_mae"]).mean()),
    }


def modern_metrics(frame: pd.DataFrame, alpha: float) -> dict:
    part = frame[frame["qcf_station_degC"].notna()].copy()
    others = part[["candidate_ModE-Sim_degC", "candidate_MPI-ESM1-2-LR_degC", "candidate_MIROC-ES2L_degC"]].mean(axis=1)
    part["candidate"] = part["current_CESM_degC"] + alpha * (others - part["current_CESM_degC"])
    valid = part[["qcf_station_degC", "candidate"]].notna().all(axis=1)
    part = part.loc[valid]
    residual = part["candidate"] - part["qcf_station_degC"]
    station_mae = part.assign(abs_error=residual.abs()).groupby("station_id")["abs_error"].mean()
    block_rows = []
    for start in (1951, 1961, 1971, 1981, 1991):
        end = min(start + 9, 2000)
        for station_id, group in part[part["year"].between(start, end)].groupby("station_id"):
            if len(group) >= 6:
                block_rows.append((group["qcf_station_degC"].mean(), group["candidate"].mean()))
    blocks = np.asarray(block_rows, dtype=float)
    return {
        "alpha": float(alpha),
        "annual_pair_count": int(len(part)),
        "annual_pooled_rmse_degC": float(np.sqrt(np.mean(np.square(residual)))),
        "annual_pooled_correlation": float(np.corrcoef(part["qcf_station_degC"], part["candidate"])[0, 1]),
        "median_station_mae_degC": float(station_mae.median()),
        "decadal_pair_count": int(len(blocks)),
        "decadal_rmse_degC": float(np.sqrt(np.mean(np.square(blocks[:, 1] - blocks[:, 0])))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [PUBLIC, ALPHA_ONE, EARLY_STATIONS, MODERN_STATIONS, LAKE_POINTS, AWENG_POINTS, NGAMRING_POINTS, HAIZI_WINDOWS]
    before = {str(path): sha256(path) for path in sources if path.stat().st_size <= 256 * 1024 * 1024}

    with Dataset(ALPHA_ONE) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        assert not np.any((years >= LOCKED[0]) & (years <= LOCKED[1]))
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        support = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        correction = clean(dataset.variables["tas_background_low_frequency_correction"][:])

    with h5py.File(PUBLIC, "r") as dataset:
        public_years = np.asarray(dataset["time"], dtype=int).ravel()
        assert np.array_equal(public_years, np.arange(1000, 2001))
        baseline = clean(dataset["tas_ensemble_mean_rec_result"][1961 - 1000:1991 - 1000])
        early = clean(dataset["tas_ensemble_mean_rec_result"][1500 - 1000:1900 - 1000])
        project = early - np.nanmean(baseline, axis=0)[None, :, :]
        project_years = np.arange(1500, 1900)

    records = []
    point_specs = [
        ("Aweng", "West", 32.75, 81.67, pd.read_csv(AWENG_POINTS).query("site_id == 'aweng_co'").rename(columns={"mid_year_ce": "year", "proxy_value": "external"})),
        ("Ngamring", "Central-south", 29.31, 87.18, pd.read_csv(NGAMRING_POINTS).rename(columns={"year_ce": "year", "maat_c": "external"})),
    ]
    lake = pd.read_csv(LAKE_POINTS)
    point_specs.extend([
        ("Tengchongqinghai", "Southeast margin", 25.132, 98.57, lake.query("record == 'Tengchongqinghai'").rename(columns={"year_ce": "year", "author_admitted_maat_c": "external"})),
        ("Lugu", "Southeast", 27.7133, 100.79, lake.query("record == 'Lugu'").rename(columns={"year_ce": "year", "author_admitted_maat_c": "external"})),
    ])
    for name, region, site_lat, site_lon, points in point_specs:
        points = points[pd.to_numeric(points["year"], errors="coerce") <= TRAIN_END].copy()
        points["year"] = pd.to_numeric(points["year"], errors="coerce")
        points["external"] = pd.to_numeric(points["external"], errors="coerce")
        points = points.dropna(subset=["year", "external"]).sort_values("year")
        iy, ix, distance = nearest_index(lat, lon, support, site_lat, site_lon)
        records.append({
            "record": name,
            "region": region,
            "year": points["year"].to_numpy(float),
            "external": points["external"].to_numpy(float),
            "project": sample_years(project[:, iy, ix], project_years, points["year"].to_numpy(float)),
            "correction": sample_years(correction[:, iy, ix], years, points["year"].to_numpy(float)),
            "support_distance_deg": distance,
            "role": "direction_only" if len(points) < 4 else "alpha_training",
        })

    haizi = pd.read_csv(HAIZI_WINDOWS)
    haizi = haizi[haizi["end_year"] <= TRAIN_END].copy()
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    box = support & (yy >= 28.58) & (yy <= 30.58) & (xx >= 99.65) & (xx <= 101.65)
    correction_box = area_mean(correction, box, lat)
    correction_windows = []
    for row in haizi.itertuples(index=False):
        selected = (years >= int(row.start_year)) & (years <= int(row.end_year))
        correction_windows.append(float(np.nanmean(correction_box[selected])))
    records.append({
        "record": "Haizi",
        "region": "East",
        "year": haizi["center_year"].to_numpy(float),
        "external": haizi["haizi_annual_reconstruction_anomaly_degC"].to_numpy(float),
        "project": haizi["project_local_box_annual_anomaly_degC"].to_numpy(float),
        "correction": np.asarray(correction_windows),
        "support_distance_deg": 0.0,
        "role": "alpha_training",
    })

    training = [record for record in records if record["role"] == "alpha_training"]
    if len(training) != 4:
        raise RuntimeError(f"expected-four-training-records-got-{len(training)}")

    record_rows = []
    grid_rows = []
    for record in records:
        fitted = alpha_fit(record["external"], record["project"], record["correction"])
        loo = []
        if len(record["year"]) >= 4:
            for index in range(len(record["year"])):
                keep = np.arange(len(record["year"])) != index
                loo.append(alpha_fit(record["external"][keep], record["project"][keep], record["correction"][keep]))
        base_metrics = record_metrics(record, 0.0)
        for alpha in ALPHA_GRID:
            metrics = record_metrics(record, float(alpha))
            grid_rows.append({"record": record["record"], "alpha": float(alpha), **metrics})
        record_rows.append({
            "record": record["record"], "region": record["region"], "role": record["role"],
            "point_count": int(len(record["year"])), "first_year": float(np.min(record["year"])),
            "last_year": float(np.max(record["year"])), "support_distance_deg": float(record["support_distance_deg"]),
            "correction_range_degC": float(np.ptp(record["correction"])),
            "alpha_unconstrained": fitted, "alpha_clipped_0_1": float(np.clip(fitted, 0, 1)) if np.isfinite(fitted) else np.nan,
            "point_leave_one_out_alpha_min": float(np.nanmin(loo)) if loo else np.nan,
            "point_leave_one_out_alpha_max": float(np.nanmax(loo)) if loo else np.nan,
            "alpha0_rmse_degC": base_metrics["rmse_degC"],
        })
    record_table = pd.DataFrame(record_rows)
    grid_table = pd.DataFrame(grid_rows)

    def pooled_objective(alpha: float, subset: list[dict]) -> float:
        return float(np.mean([record_metrics(record, alpha)["normalized_mse"] for record in subset]))

    pooled_values = np.asarray([pooled_objective(float(alpha), training) for alpha in ALPHA_GRID])
    pooled_alpha = float(ALPHA_GRID[int(np.nanargmin(pooled_values))])
    pooled_rows = pd.DataFrame({"alpha": ALPHA_GRID, "equal_record_normalized_mse": pooled_values})
    leave_one_out = []
    for held in training:
        subset = [record for record in training if record["record"] != held["record"]]
        values = np.asarray([pooled_objective(float(alpha), subset) for alpha in ALPHA_GRID])
        selected = float(ALPHA_GRID[int(np.nanargmin(values))])
        held_base = record_metrics(held, 0.0)
        held_candidate = record_metrics(held, selected)
        leave_one_out.append({
            "held_out_record": held["record"], "held_out_region": held["region"], "selected_alpha": selected,
            "held_out_rmse_alpha0_degC": held_base["rmse_degC"],
            "held_out_rmse_selected_degC": held_candidate["rmse_degC"],
            "held_out_improvement_fraction": 1.0 - held_candidate["rmse_degC"] / held_base["rmse_degC"],
        })
    loo_table = pd.DataFrame(leave_one_out)

    named_alphas = {"CESM": 0.0, "old_50_50": 0.5, "equal_four": 0.75, "full_non_CESM_mean": 1.0, "pre1850_selected": pooled_alpha}
    early_frame = pd.read_csv(EARLY_STATIONS)
    if early_frame["locked_period_value_accessed"].fillna(False).astype(bool).any():
        raise RuntimeError("locked-period-access-flag")
    holdout_rows = []
    for branch, alpha in named_alphas.items():
        for start, end in ((1880, 1900), (1901, 1920), (1940, 1950)):
            holdout_rows.append({"branch": branch, **station_period_metrics(early_frame, alpha, start, end)})
    holdout = pd.DataFrame(holdout_rows)

    modern_frame = pd.read_csv(MODERN_STATIONS)
    modern_rows = [{"branch": branch, **modern_metrics(modern_frame, alpha)} for branch, alpha in named_alphas.items()]
    modern = pd.DataFrame(modern_rows)

    for branch, alpha in named_alphas.items():
        for record in records:
            metrics = record_metrics(record, alpha)
            record_rows.append({
                "record": record["record"], "region": record["region"], "role": record["role"],
                "branch": branch, "alpha": alpha, **metrics,
            })
    branch_record_metrics = pd.DataFrame([row for row in record_rows if "branch" in row])

    alpha0_objective = pooled_objective(0.0, training)
    selected_objective = pooled_objective(pooled_alpha, training)
    loo_range = float(loo_table["selected_alpha"].max() - loo_table["selected_alpha"].min())
    records_improved = 0
    for record in training:
        if record_metrics(record, pooled_alpha)["rmse_degC"] < record_metrics(record, 0.0)["rmse_degC"]:
            records_improved += 1

    holdout_selected = holdout[holdout["branch"] == "pre1850_selected"].set_index("period")
    holdout_original = holdout[holdout["branch"] == "CESM"].set_index("period")
    modern_selected = modern[modern["branch"] == "pre1850_selected"].iloc[0]
    modern_original = modern[modern["branch"] == "CESM"].iloc[0]
    station_gate = True
    for period in ("1880-1900", "1901-1920"):
        candidate = holdout_selected.loc[period]
        original = holdout_original.loc[period]
        station_gate &= abs(candidate.equal_station_mean_bias_degC) < abs(original.equal_station_mean_bias_degC)
        station_gate &= candidate.median_station_mae_degC < original.median_station_mae_degC
        station_gate &= candidate.median_abs_trend_error_degC_per_century <= 1.10 * original.median_abs_trend_error_degC_per_century

    gates = {
        "four_independent_pre1850_training_records": len(training) == 4,
        "equal_record_objective_improves_at_least_10pct": selected_objective <= 0.90 * alpha0_objective,
        "at_least_three_of_four_training_records_improve": records_improved >= 3,
        "leave_one_record_out_alpha_range_at_most_0p25": loo_range <= 0.25,
        "western_1880_1920_bias_mae_and_trend_holdout_pass": bool(station_gate),
        "1940_1950_median_mae_not_worse_5pct": bool(holdout_selected.loc["1940-1950"].median_station_mae_degC <= 1.05 * holdout_original.loc["1940-1950"].median_station_mae_degC),
        "modern_decadal_rmse_not_worse_10pct": bool(modern_selected.decadal_rmse_degC <= 1.10 * modern_original.decadal_rmse_degC),
        "modern_annual_correlation_drop_at_most_0p05": bool(modern_selected.annual_pooled_correlation >= modern_original.annual_pooled_correlation - 0.05),
        "historical_50year_closure_at_most_0p05_degC": bool(pooled_alpha * FULL_ALPHA_CONSTRAINT_CLOSURE <= 0.05),
    }
    all_pass = all(gates.values())

    record_table.to_csv(OUT / "record_alpha_estimates.csv", index=False)
    pooled_rows.to_csv(OUT / "pooled_alpha_objective.csv", index=False)
    loo_table.to_csv(OUT / "leave_one_record_out_alpha.csv", index=False)
    branch_record_metrics.to_csv(OUT / "named_alpha_record_metrics.csv", index=False)
    holdout.to_csv(OUT / "station_holdout_metrics.csv", index=False)
    modern.to_csv(OUT / "modern_transfer_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)
    colors = {"Aweng": "#1b9e77", "Tengchongqinghai": "#d95f02", "Lugu": "#7570b3", "Haizi": "#1f78b4", "Ngamring": "#666666"}
    for record in records:
        subset = grid_table[grid_table["record"] == record["record"]]
        axes[0, 0].plot(subset["alpha"], subset["normalized_mse"], label=record["record"], color=colors[record["record"]], lw=2 if record["role"] == "alpha_training" else 1, ls="-" if record["role"] == "alpha_training" else "--")
    axes[0, 0].axvline(pooled_alpha, color="#111111", lw=1.5, ls=":", label=f"selected {pooled_alpha:.3f}")
    axes[0, 0].set(xlabel="alpha", ylabel="Centered normalized MSE", title="A. Pre-1850 annual evidence")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0, 0].grid(alpha=0.2)

    y = np.arange(len(record_table))
    axes[0, 1].axvspan(0, 1, color="#efefef")
    axes[0, 1].axvline(pooled_alpha, color="#111111", lw=1.5, ls=":")
    axes[0, 1].scatter(record_table["alpha_unconstrained"], y, color=[colors[x] for x in record_table["record"]], s=70)
    for index, row in record_table.iterrows():
        if np.isfinite(row.point_leave_one_out_alpha_min):
            axes[0, 1].plot([row.point_leave_one_out_alpha_min, row.point_leave_one_out_alpha_max], [index, index], color="#555555", lw=1)
    axes[0, 1].set_yticks(y, record_table["record"])
    axes[0, 1].set(xlabel="Unconstrained record-specific alpha", title="B. Cross-record identifiability")
    axes[0, 1].grid(axis="x", alpha=0.2)

    branch_order = list(named_alphas)
    x = np.arange(len(branch_order))
    for period, offset, color in (("1880-1900", -0.22, "#d95f02"), ("1901-1920", 0.0, "#7570b3"), ("1940-1950", 0.22, "#1b9e77")):
        values = holdout.set_index(["branch", "period"]).loc[[(branch, period) for branch in branch_order], "equal_station_mean_bias_degC"].to_numpy()
        axes[1, 0].bar(x + offset, values, width=0.21, label=period, color=color)
    axes[1, 0].axhline(0, color="#555555", lw=0.8)
    axes[1, 0].set_xticks(x, [name.replace("_", "\n") for name in branch_order], fontsize=8)
    axes[1, 0].set(ylabel="Project minus qcf (degC)", title="C. Station holdout bias")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.2)

    labels = list(gates)
    values = [1 if gates[label] else 0 for label in labels]
    axes[1, 1].barh(np.arange(len(labels)), values, color=["#2a9d8f" if value else "#c44536" for value in values])
    axes[1, 1].set_yticks(np.arange(len(labels)), [label.replace("_", " ") for label in labels], fontsize=7.5)
    axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].set_xticks([0, 1], ["fail", "pass"])
    axes[1, 1].set_title("D. Frozen decision gates")
    axes[1, 1].invert_yaxis()

    fig.suptitle("Physically defined background-alpha identifiability test", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "background_alpha_identifiability.png", dpi=220)
    plt.close(fig)

    decision = "candidate_v2_alpha_identified_and_all_gates_pass" if all_pass else "global_alpha_not_promotable"
    summary = {
        "experiment": OUT.name,
        "status": "complete",
        "alpha_definition": "CESM + alpha * (equal_mean_ModE_MPI_MIROC - CESM), applied as the archived 31-year low-frequency correction",
        "training_window": "1500-1849 only",
        "training_records": [record["record"] for record in training],
        "direction_only_record": "Ngamring_two_points",
        "selected_alpha_from_pre1850_equal_record_objective": pooled_alpha,
        "alpha0_objective": alpha0_objective,
        "selected_objective": selected_objective,
        "objective_improvement_fraction": 1.0 - selected_objective / alpha0_objective,
        "leave_one_record_out_alpha_range": loo_range,
        "training_records_improved": records_improved,
        "historical_constraint_closure_degC": pooled_alpha * FULL_ALPHA_CONSTRAINT_CLOSURE,
        "gates": gates,
        "all_gates_pass": all_pass,
        "decision": decision,
        "candidate_v2_authorized": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {str(path): {"sha256": sha256(path)} for path in sources if path.stat().st_size <= 256 * 1024 * 1024}
    manifest["large_source_not_rehashed"] = str(PUBLIC)
    manifest["rules"] = {
        "selection_uses_1880_1920": False,
        "locked_period_accessed": False,
        "secondary_recalibrations_counted": False,
        "prohibited_source_accessed": False,
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Background alpha identifiability test", "",
        f"Decision: `{decision}`.", "",
        f"Pre-1850 annual evidence selected alpha `{pooled_alpha:.3f}` on the fixed CESM-to-three-model-mean axis.",
        f"The equal-record centered objective improved by `{100 * (1.0 - selected_objective / alpha0_objective):.1f}%` relative to alpha=0.",
        f"Leave-one-record-out selected-alpha range was `{loo_range:.3f}`.",
        f"Historical 50-year closure at the selected alpha was `{pooled_alpha * FULL_ALPHA_CONSTRAINT_CLOSURE:.4f} degC`.", "",
        "## Gates", "",
    ]
    report.extend([f"- `{name}`: `{value}`" for name, value in gates.items()])
    report.extend([
        "", "## Boundaries", "",
        "- Alpha was selected without 1880-1920 station values.",
        "- Ngamring has two safe pre-1850 points and is direction-only.",
        "- Fits use within-record centered shape, so proxy calibration offsets cannot select alpha.",
        "- The 1921-1939 interval was absent and unread.",
        "- This is a diagnostic fixed-increment field test, not an EnKF rerun or production candidate.",
    ])
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = {str(path): sha256(path) for path in sources if path.stat().st_size <= 256 * 1024 * 1024}
    if before != after:
        raise RuntimeError("source-file-changed")


if __name__ == "__main__":
    main()
